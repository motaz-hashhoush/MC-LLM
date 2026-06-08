"""
Task processor — calls the inference engine directly and logs to PostgreSQL.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from app.api.schemas import JobStatus, TaskResponse, TaskType
from app.db.db_logger import DBLogger
from app.services.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class TaskProcessor:
    """
    Thin orchestration layer between the API routes and the inference engine.

    Each call to :meth:`process` directly awaits the engine — multiple
    concurrent HTTP requests therefore reach vLLM concurrently, allowing
    its continuous-batching scheduler to group them into a single forward
    pass. DB logging is fire-and-forget so it never adds to response latency.
    """

    def __init__(self, inference_engine: InferenceEngine, db_logger: DBLogger) -> None:
        self._engine = inference_engine
        self._db_logger = db_logger

    async def process(
        self,
        task_type: TaskType,
        text: str,
        think: bool,
        max_tokens: int,
        temperature: float,
    ) -> TaskResponse:
        job_id = str(uuid.uuid4())

        asyncio.create_task(
            self._db_logger.log_request(
                job_id=job_id,
                task_type=task_type.value,
                input_text=text,
            )
        )

        result = await self._engine.process_task({
            "id": job_id,
            "task": task_type.value,
            "text": text,
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "think": think,
            },
        })

        if "error" in result:
            asyncio.create_task(
                self._db_logger.log_error(
                    job_id=job_id,
                    error_message=result["error"],
                    latency_ms=result.get("latency_ms"),
                )
            )
            return TaskResponse(
                id=uuid.UUID(job_id),
                task=task_type,
                status=JobStatus.FAILED,
                error=result["error"],
            )

        asyncio.create_task(
            self._db_logger.log_completion(
                job_id=job_id,
                output_text=result["result"],
                latency_ms=result.get("latency_ms"),
            )
        )
        return TaskResponse(
            id=uuid.UUID(job_id),
            task=task_type,
            status=JobStatus.COMPLETED,
            result=result["result"],
        )
