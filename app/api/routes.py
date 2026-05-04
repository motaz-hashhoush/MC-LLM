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


# ── Text-to-Speech ────────────────────────────────────────────────────────────

import asyncio
import io

from fastapi.responses import StreamingResponse

from app.api.schemas import TTSRequest
from app.tts.silma_client import SilmaTTSClient, TTSError
from app.tts.tts_processor import TTSProcessor, TTSJobSchema

tts_router = APIRouter(prefix="/v1", tags=["Text-to-Speech"])


def _get_silma_client(request: Request) -> SilmaTTSClient:
    """Retrieve SilmaTTSClient from application state."""
    client: SilmaTTSClient | None = getattr(request.app.state, "silma_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    return client


def _get_tts_processor(request: Request) -> TTSProcessor:
    """Retrieve TTSProcessor from application state."""
    processor: TTSProcessor | None = getattr(request.app.state, "tts_processor", None)
    if processor is None:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    return processor


@tts_router.post("/tts")
async def tts_speech(
    body: TTSRequest,
    request: Request,
) -> StreamingResponse:
    """
    Synthesise text to speech and return an audio stream.

    Supports Arabic (MSA) and English. Optionally accepts a base64-encoded
    WAV reference audio clip for zero-shot voice cloning.

    Returns
    -------
    StreamingResponse
        ``audio/wav`` (default) or ``audio/mpeg`` if ``format="mp3"``.

    Errors
    ------
    - **400** — Pydantic validation errors (text too long, invalid language/format)
    - **503** — TTS model not loaded
    - **500** — Internal synthesis failure
    """
    client = _get_silma_client(request)
    processor = _get_tts_processor(request)

    # Guard: model must be loaded
    if not client.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded.",
        )

    logger.info(
        "Received /v1/tts request (lang=%s, speed=%.2f, fmt=%s, chars=%d)",
        body.language,
        body.speed,
        body.format,
        len(body.text),
    )

    job = TTSJobSchema(
        text=body.text,
        language=body.language,
        speed=body.speed,
        format=body.format,
        clone_audio=body.clone_audio,
    )

    try:
        result = await processor.process_tts_job(job)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TTSError as exc:
        logger.exception("TTS synthesis error")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected TTS error")
        raise HTTPException(status_code=500, detail="Internal TTS error") from exc

    return StreamingResponse(
        io.BytesIO(result.audio_bytes),
        media_type=result.mime_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{body.format}",
            "X-TTS-Duration-Ms": f"{result.duration_ms:.1f}",
        },
    )


