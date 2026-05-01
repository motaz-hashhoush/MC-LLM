"""
FastAPI API routes for the LLM inference gateway.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from app.api.schemas import HealthResponse, TaskRequest, TaskResponse, TaskType, TranscribeResponse
from app.services.task_processor import TaskProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

# Accepted MIME types for audio uploads
_ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "application/octet-stream",  # some clients send this for any binary
}


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
    # Decide think flag based on task type and user input
    if task_type == TaskType.GENERATE:
        think_val = body.think if body.think is not None else False
    else:
        # Defaults to False for summarize/rewrite, even if user sends True
        think_val = False

    logger.info(
        "Received %s request (prompt_len=%d, max_tokens=%d, temp=%.2f, think=%s)",
        task_type.value,
        len(body.prompt),
        body.max_tokens,
        body.temperature,
        think_val,
    )

    response = await processor.process(
        task_type=task_type,
        text=body.prompt,
        think=think_val,
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


# ── Speech-to-Text ────────────────────────────────────────────────────────────

@router.post("/stt", response_model=TranscribeResponse)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe (WAV, MP3, M4A, OGG, FLAC, WEBM)."),
    language: str = Query(
        default=None,
        description="Language hint for Whisper (e.g. 'ar', 'en'). Defaults to the server's 'ar'.",
    ),
) -> TranscribeResponse:
    """Transcribe an uploaded audio file and return the Arabic (or specified language) text."""
    import uuid
    from app.api.schemas import JobStatus
    from app.config import get_settings

    settings = get_settings()
    effective_language = language or settings.STT_LANGUAGE

    whisper_client = getattr(request.app.state, "whisper_client", None)
    if whisper_client is None:
        raise HTTPException(status_code=503, detail="STT service not ready")

    # ── Validate content type ────────────────────────────────────────────────
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Accepted: WAV, MP3, M4A, OGG, FLAC, WEBM.",
        )

    # ── Read & size-check ────────────────────────────────────────────────────
    audio_bytes = await file.read()
    max_bytes = settings.STT_MAX_AUDIO_MB * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds the {settings.STT_MAX_AUDIO_MB} MB limit.",
        )

    job_id = uuid.uuid4()
    logger.info(
        "Received /stt request (file=%s, size=%d bytes, lang=%s)",
        file.filename,
        len(audio_bytes),
        effective_language,
    )

    try:
        transcript = await whisper_client.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            language=effective_language,
        )
        logger.info("Transcription job %s completed (%d chars)", job_id, len(transcript))
        return TranscribeResponse(
            id=job_id,
            status=JobStatus.COMPLETED,
            transcript=transcript,
            language=effective_language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcription job %s failed", job_id)
        return TranscribeResponse(
            id=job_id,
            status=JobStatus.FAILED,
            language=effective_language,
            error=str(exc),
        )
