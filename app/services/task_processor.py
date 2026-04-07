"""
Task processor — orchestrates enqueue → poll → return for API routes.
"""

from __future__ import annotations

import asyncio
import logging

from app.api.schemas import JobStatus, TaskResponse, TaskType
from app.config import get_settings
from app.queue.job_queue import JobQueue

logger = logging.getLogger(__name__)


class TaskProcessor:
    """
    Used by FastAPI route handlers to submit a task and wait for the result.

    Flow:
        1. Enqueue the job via :class:`JobQueue`
        2. Poll Redis for the result with a configurable timeout
        3. Return a :class:`TaskResponse`
    """

    def __init__(self, job_queue: JobQueue) -> None:
        self._queue = job_queue
        self._settings = get_settings()

    async def process(
        self,
        task_type: TaskType,
        text: str,
        max_tokens: int,
        temperature: float,
    ) -> TaskResponse:
        """
        Submit a task and block until the result is available or timeout.
        """
        job_id = await self._queue.enqueue_job(
            task=task_type.value,
            text=text,
            parameters={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        logger.info("Job %s enqueued (task=%s)", job_id, task_type.value)

        # Poll for result
        timeout = self._settings.JOB_TIMEOUT
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < timeout:
            status_str = await self._queue.get_job_status(job_id)

            if status_str in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                result_data = await self._queue.get_result(job_id)

                if result_data is None:
                    return TaskResponse(
                        id=job_id,
                        task=task_type,
                        status=JobStatus.FAILED,
                        error="Result data was lost",
                    )

                if status_str == JobStatus.COMPLETED.value:
                    return TaskResponse(
                        id=job_id,
                        task=task_type,
                        status=JobStatus.COMPLETED,
                        result=result_data.get("result"),
                    )
                else:
                    return TaskResponse(
                        id=job_id,
                        task=task_type,
                        status=JobStatus.FAILED,
                        error=result_data.get("error", "Unknown error"),
                    )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout
        logger.warning("Job %s timed out after %ss", job_id, timeout)
        return TaskResponse(
            id=job_id,
            task=task_type,
            status=JobStatus.FAILED,
            error=f"Job timed out after {timeout}s",
        )
