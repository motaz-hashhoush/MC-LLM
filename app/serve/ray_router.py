"""
Ray Serve router and background queue consumer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray
from ray import serve

from app.config import get_settings
from app.db.db_logger import DBLogger
from app.queue.job_queue import JobQueue
from app.serve.ray_worker import LLMWorker

logger = logging.getLogger(__name__)


class RayRouter:
    """
    Sends individual tasks to the :class:`LLMWorker` deployment.
    """

    def __init__(self) -> None:
        self._handle: serve.DeploymentHandle = LLMWorker.get_handle()

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Submit *task* to the LLMWorker and return the result dict.
        """
        result: dict[str, Any] = await self._handle.remote(task)
        return result


class QueueConsumer:
    """
    Background loop that continuously dequeues jobs from Redis,
    dispatches them to :class:`RayRouter`, and stores results back.
    """

    def __init__(
        self,
        job_queue: JobQueue,
        router: RayRouter,
        db_logger: DBLogger | None = None,
    ) -> None:
        self._queue = job_queue
        self._router = router
        self._db_logger = db_logger
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Begin consuming the queue in the background."""
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())
        logger.info("QueueConsumer started")

    async def stop(self) -> None:
        """Signal the consumer loop to stop and await termination."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("QueueConsumer stopped")

    async def _consume_loop(self) -> None:
        """Main loop: dequeue → process → store, forever."""
        while self._running:
            try:
                job = await self._queue.dequeue_job(timeout=2)
                if job is None:
                    continue

                job_id = job["id"]

                # Log to DB
                if self._db_logger:
                    await self._db_logger.log_request(
                        job_id=job_id,
                        task_type=job["task"],
                        input_text=job["text"],
                    )

                # Process
                result = await self._router.process_task(job)

                # Store result
                if "error" in result:
                    await self._queue.store_result(
                        job_id, error=result["error"]
                    )
                    if self._db_logger:
                        await self._db_logger.log_error(
                            job_id=job_id,
                            error_message=result["error"],
                            latency_ms=result.get("latency_ms"),
                        )
                else:
                    await self._queue.store_result(
                        job_id, result=result["result"]
                    )
                    if self._db_logger:
                        await self._db_logger.log_completion(
                            job_id=job_id,
                            output_text=result["result"],
                            latency_ms=result.get("latency_ms"),
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in QueueConsumer loop")
                await asyncio.sleep(1)
