"""
FastAPI API routes for the LLM inference gateway.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.schemas import HealthResponse, TaskRequest, TaskResponse, TaskType
from app.services.task_processor import TaskProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency ───────────────────────────────────────────────────────────────

def _get_processor(request: Request) -> TaskProcessor:
    """Retrieve the TaskProcessor from application state."""
    processor: TaskProcessor | None = request.app.state.task_processor
    if processor is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return processor


# ── Task Endpoints ───────────────────────────────────────────────────────────

async def _handle_task(
    task_type: TaskType,
    body: TaskRequest,
    processor: TaskProcessor,
) -> TaskResponse:
    """Common handler shared by all task endpoints."""
    start = time.perf_counter()
    logger.info(
        "Received %s request (text_len=%d, max_tokens=%d, temp=%.2f)",
        task_type.value,
        len(body.text),
        body.max_tokens,
        body.temperature,
    )

    response = await processor.process(
        task_type=task_type,
        text=body.text,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )

    latency = (time.perf_counter() - start) * 1000
    logger.info(
        "Finished %s request id=%s status=%s latency=%.1fms",
        task_type.value,
        response.id,
        response.status,
        latency,
    )
    return response


@router.post("/summarize", response_model=TaskResponse)
async def summarize(
    body: TaskRequest,
    processor: Annotated[TaskProcessor, Depends(_get_processor)],
) -> TaskResponse:
    """Summarise the provided text."""
    return await _handle_task(TaskType.SUMMARIZE, body, processor)


@router.post("/rewrite", response_model=TaskResponse)
async def rewrite(
    body: TaskRequest,
    processor: Annotated[TaskProcessor, Depends(_get_processor)],
) -> TaskResponse:
    """Rewrite the provided text professionally."""
    return await _handle_task(TaskType.REWRITE, body, processor)


@router.post("/generate", response_model=TaskResponse)
async def generate(
    body: TaskRequest,
    processor: Annotated[TaskProcessor, Depends(_get_processor)],
) -> TaskResponse:
    """Generate new content based on the provided text."""
    return await _handle_task(TaskType.GENERATE, body, processor)


# ── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health status."""
    return HealthResponse()
