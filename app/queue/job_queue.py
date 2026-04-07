"""
Redis-backed job queue for asynchronous task processing.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import redis.asyncio as aioredis

from app.api.schemas import JobStatus
from app.config import get_settings

logger = logging.getLogger(__name__)


class JobQueue:
    """
    Provides enqueue / dequeue / status / result operations backed by Redis.

    Key layout
    ----------
    - ``llm:tasks``          – LIST used as the task queue (LPUSH / BRPOP)
    - ``job:{id}:status``    – STRING holding the current :class:`JobStatus`
    - ``job:{id}:payload``   – STRING holding the serialised job payload
    - ``job:{id}:result``    – STRING holding the serialised result or error
    """

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client
        self._settings = get_settings()
        self._queue_name = self._settings.QUEUE_NAME
        self._result_ttl = self._settings.RESULT_TTL

    # ── Enqueue ──────────────────────────────────────────────────────────

    async def enqueue_job(
        self,
        task: str,
        text: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new job, persist its payload, and push it onto the queue.

        Returns the generated job ID (UUID).
        """
        job_id = str(uuid.uuid4())
        payload = {
            "id": job_id,
            "task": task,
            "text": text,
            "parameters": parameters or {},
        }

        pipe = self._client.pipeline(transaction=True)
        pipe.set(f"job:{job_id}:status", JobStatus.PENDING.value)
        pipe.set(f"job:{job_id}:payload", json.dumps(payload))
        pipe.lpush(self._queue_name, json.dumps(payload))
        await pipe.execute()

        logger.info("Enqueued job %s (task=%s)", job_id, task)
        return job_id

    # ── Dequeue ──────────────────────────────────────────────────────────

    async def dequeue_job(self, timeout: int = 0) -> dict[str, Any] | None:
        """
        Blocking pop from the task queue.

        Parameters
        ----------
        timeout:
            Seconds to block; 0 = block indefinitely.

        Returns the deserialised job dict or ``None`` on timeout.
        """
        result = await self._client.brpop(self._queue_name, timeout=timeout)
        if result is None:
            return None

        _queue_name, raw = result
        job: dict[str, Any] = json.loads(raw)
        await self._client.set(f"job:{job['id']}:status", JobStatus.PROCESSING.value)
        logger.info("Dequeued job %s", job["id"])
        return job

    # ── Status ───────────────────────────────────────────────────────────

    async def get_job_status(self, job_id: str) -> str | None:
        """Return the current status string for *job_id*, or ``None``."""
        return await self._client.get(f"job:{job_id}:status")

    async def set_job_status(self, job_id: str, status: JobStatus) -> None:
        """Update the status key of a job."""
        await self._client.set(f"job:{job_id}:status", status.value)

    # ── Results ──────────────────────────────────────────────────────────

    async def store_result(
        self,
        job_id: str,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """
        Persist the result (or error) of a completed job and update status.
        Keys are set with a TTL for automatic cleanup.
        """
        status = JobStatus.COMPLETED if error is None else JobStatus.FAILED
        data = json.dumps({"result": result, "error": error})

        pipe = self._client.pipeline(transaction=True)
        pipe.set(f"job:{job_id}:status", status.value, ex=self._result_ttl)
        pipe.set(f"job:{job_id}:result", data, ex=self._result_ttl)
        await pipe.execute()

        logger.info("Stored result for job %s (status=%s)", job_id, status.value)

    async def get_result(self, job_id: str) -> dict[str, Any] | None:
        """
        Return the stored result dict for *job_id*, or ``None`` if not ready.
        """
        raw = await self._client.get(f"job:{job_id}:result")
        if raw is None:
            return None
        return json.loads(raw)
