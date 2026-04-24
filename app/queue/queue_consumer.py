"""
Background queue consumer — dequeues jobs and processes them via the inference engine.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from app.db.db_logger import DBLogger
from app.queue.job_queue import JobQueue

logger = logging.getLogger(__name__)


class Processor(Protocol):
    """Protocol for the inference processor."""
    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        ...


class QueueConsumer:
    """
    Background loop that continuously dequeues jobs from Redis,
    dispatched them to a processor, and stores results back.
    """

    def __init__(
        self,
        job_queue: JobQueue,
        processor: Processor,
        db_logger: DBLogger | None = None,
    ) -> None:
        self._queue = job_queue
        self._processor = processor
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
                result = await self._processor.process_task(job)

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
