"""
TTSProcessor — business logic layer for TTS job execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.tts.supertonic_client import SupertonicTTSClient, TTSError

if TYPE_CHECKING:
    from app.db.db_logger import DBLogger

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TTSResult:
    """Holds the outcome of a completed TTS synthesis job."""

    audio_bytes: bytes
    mime_type: str
    duration_ms: float


# ── Job schema ────────────────────────────────────────────────────────────────

@dataclass
class TTSJobSchema:
    """
    Validated input for a TTS synthesis request.

    Mirrors TTSRequest (Pydantic) but is a plain dataclass so TTSProcessor
    has no FastAPI dependency.
    """

    text: str
    language: str = "ar"
    speed: float = 1.05
    format: str = "wav"         # "wav" | "mp3"
    voice_name: str = "M1"      # M1–M5, F1–F5
    total_steps: int = 8        # denoising steps: 5 (fast) – 12 (quality)


# ── Processor ─────────────────────────────────────────────────────────────────

class TTSProcessor:
    """
    Orchestrates TTS synthesis for a single HTTP request.

    Flow
    ----
    1. Validate text length
    2. Acquire gpu_semaphore, offload synthesis to a thread pool, release
    3. (Optional) convert WAV → MP3 in thread pool
    4. Log result to the tts_requests DB table
    5. Return TTSResult
    """

    MAX_TEXT_CHARS: int = 2000

    def __init__(
        self,
        tts_client: SupertonicTTSClient,
        db_logger: DBLogger,
        gpu_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._client = tts_client
        self._db_logger = db_logger
        self._gpu_semaphore = gpu_semaphore or asyncio.Semaphore(1)

    # ── Public API ────────────────────────────────────────────────────────────

    async def process_tts_job(self, job: TTSJobSchema) -> TTSResult:
        """
        Execute a TTS synthesis job asynchronously.

        Raises
        ------
        ValueError
            If text length exceeds MAX_TEXT_CHARS.
        TTSError
            If synthesis or format conversion fails.
        """
        if len(job.text) > self.MAX_TEXT_CHARS:
            raise ValueError(
                f"text must be {self.MAX_TEXT_CHARS} characters or fewer "
                f"(got {len(job.text)})."
            )

        start_ms = time.monotonic()

        try:
            # Serialise CPU-bound synthesis via shared semaphore
            async with self._gpu_semaphore:
                wav_bytes: bytes = await asyncio.to_thread(
                    self._client.synthesize,
                    job.text,
                    job.language,
                    job.speed,
                    job.voice_name,
                    job.total_steps,
                )

            if job.format == "mp3":
                audio_bytes: bytes = await asyncio.to_thread(
                    self._client.convert_to_mp3, wav_bytes
                )
                mime_type = "audio/mpeg"
            else:
                audio_bytes = wav_bytes
                mime_type = "audio/wav"

            duration_ms = (time.monotonic() - start_ms) * 1000

            await self._log_tts_request(
                text=job.text,
                language=job.language,
                speed=job.speed,
                fmt=job.format,
                voice_name=job.voice_name,
                duration_ms=duration_ms,
                audio_size_b=len(audio_bytes),
                status="success",
            )

            logger.info(
                "TTS job complete (lang=%s, voice=%s, fmt=%s, size=%d bytes, %.1f ms)",
                job.language,
                job.voice_name,
                job.format,
                len(audio_bytes),
                duration_ms,
            )

            return TTSResult(
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                duration_ms=duration_ms,
            )

        except (ValueError, TTSError):
            duration_ms = (time.monotonic() - start_ms) * 1000
            await self._log_tts_request(
                text=job.text,
                language=job.language,
                speed=job.speed,
                fmt=job.format,
                voice_name=job.voice_name,
                duration_ms=duration_ms,
                audio_size_b=None,
                status="error",
                error_message=traceback.format_exc(),
            )
            raise

    # ── Private ───────────────────────────────────────────────────────────────

    async def _log_tts_request(
        self,
        text: str,
        language: str,
        speed: float,
        fmt: str,
        voice_name: str,
        duration_ms: float,
        audio_size_b: int | None,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Persist a TTS request record to the tts_requests table."""
        try:
            from app.db.models import TTSRequestLog
            from app.db.database import DatabaseManager

            db: DatabaseManager = self._db_logger._db

            async with db.get_session() as session:
                entry = TTSRequestLog(
                    text=text,
                    language=language,
                    speed=speed,
                    format=fmt,
                    voice_name=voice_name,
                    duration_ms=duration_ms,
                    audio_size_b=audio_size_b,
                    status=status,
                    error_message=error_message,
                )
                session.add(entry)
            logger.debug("Logged TTS request to DB (status=%s)", status)
        except Exception:
            logger.exception("Failed to log TTS request to DB")
