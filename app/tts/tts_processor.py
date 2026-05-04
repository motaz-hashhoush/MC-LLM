"""
TTSProcessor — business logic layer for TTS job execution.

Handles validation, optional voice-clone temp file management, synthesis
offloading, format conversion, and DB logging. Modeled after
app/services/task_processor.py.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.tts.silma_client import SilmaTTSClient, TTSError

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

    This mirrors the TTSRequest Pydantic model but is a plain dataclass so
    TTSProcessor has no FastAPI dependency.
    """

    text: str
    language: str = "ar"
    speed: float = 1.0
    format: str = "wav"                 # "wav" | "mp3"
    clone_audio: str | None = None      # base64-encoded WAV string


# ── Processor ─────────────────────────────────────────────────────────────────

class TTSProcessor:
    """
    Orchestrates TTS synthesis for a single HTTP request.

    Flow
    ----
    1. Validate text length
    2. (Optional) decode reference audio to a temp file for voice cloning
    3. Offload CPU-bound synthesis to a thread pool via asyncio.to_thread()
    4. (Optional) convert WAV → MP3 in thread pool
    5. Log result to the tts_requests DB table
    6. Return TTSResult
    """

    #: Hard limit enforced before touching the model (config-driven at call site)
    MAX_TEXT_CHARS: int = 2000

    def __init__(self, silma_client: SilmaTTSClient, db_logger: DBLogger) -> None:
        self._client = silma_client
        self._db_logger = db_logger

    # ── Public API ────────────────────────────────────────────────────────────

    async def process_tts_job(self, job: TTSJobSchema) -> TTSResult:
        """
        Execute a TTS synthesis job asynchronously.

        Parameters
        ----------
        job:
            Validated TTS request parameters.

        Returns
        -------
        TTSResult
            Audio bytes, MIME type, and synthesis duration.

        Raises
        ------
        ValueError
            If text length exceeds the 2 000-character limit.
        TTSError
            If the underlying synthesis or format conversion fails.
        """
        # 1. Validate text length
        if len(job.text) > self.MAX_TEXT_CHARS:
            raise ValueError(
                f"text must be {self.MAX_TEXT_CHARS} characters or fewer "
                f"(got {len(job.text)})."
            )

        start_ms = time.monotonic()
        ref_path: str | None = None
        tmp_file = None

        try:
            # 2. Decode clone audio (voice cloning)
            if job.clone_audio:
                try:
                    ref_bytes = base64.b64decode(job.clone_audio)
                except Exception as exc:
                    raise ValueError(
                        "clone_audio is not valid base64-encoded data."
                    ) from exc

                # Write to named temp file; delete=False so we control deletion
                tmp_file = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                )
                tmp_file.write(ref_bytes)
                tmp_file.flush()
                tmp_file.close()
                ref_path = tmp_file.name
                logger.debug("Voice clone reference audio written to %s", ref_path)

            # 3. Run CPU-bound synthesis in thread pool
            wav_bytes: bytes = await asyncio.to_thread(
                self._client.synthesize,
                job.text,
                job.language,
                job.speed,
                ref_path,  # clone_audio temp file path
            )

            # 4. Format conversion
            if job.format == "mp3":
                audio_bytes: bytes = await asyncio.to_thread(
                    self._client.convert_to_mp3, wav_bytes
                )
                mime_type = "audio/mpeg"
            else:
                audio_bytes = wav_bytes
                mime_type = "audio/wav"

            duration_ms = (time.monotonic() - start_ms) * 1000

            # 5. Log success to DB (fire-and-forget; failures are caught inside)
            await self._log_tts_request(
                text=job.text,
                language=job.language,
                speed=job.speed,
                fmt=job.format,
                has_ref_audio=ref_path is not None,
                duration_ms=duration_ms,
                audio_size_b=len(audio_bytes),
                status="success",
            )

            logger.info(
                "TTS job complete (lang=%s, fmt=%s, size=%d bytes, %.1f ms)",
                job.language,
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
            # Re-raise validation / synthesis errors after DB logging
            duration_ms = (time.monotonic() - start_ms) * 1000
            import traceback
            await self._log_tts_request(
                text=job.text,
                language=job.language,
                speed=job.speed,
                fmt=job.format,
                has_ref_audio=ref_path is not None,
                duration_ms=duration_ms,
                audio_size_b=None,
                status="error",
                error_message=traceback.format_exc(),
            )
            raise

        finally:
            # Always clean up temp file
            if tmp_file is not None:
                import os
                try:
                    os.unlink(tmp_file.name)
                    logger.debug("Deleted temp voice-clone file %s", tmp_file.name)
                except OSError:
                    logger.warning(
                        "Could not delete temp voice-clone file %s", tmp_file.name
                    )

    # ── Private ───────────────────────────────────────────────────────────────

    async def _log_tts_request(
        self,
        text: str,
        language: str,
        speed: float,
        fmt: str,
        has_ref_audio: bool,
        duration_ms: float,
        audio_size_b: int | None,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Persist a TTS request record to the tts_requests table."""
        try:
            from app.db.models import TTSRequestLog
            from app.db.database import DatabaseManager

            # Access the DatabaseManager via the db_logger's private handle
            db: DatabaseManager = self._db_logger._db

            async with db.get_session() as session:
                entry = TTSRequestLog(
                    text=text,
                    language=language,
                    speed=speed,
                    format=fmt,
                    has_ref_audio=has_ref_audio,
                    duration_ms=duration_ms,
                    audio_size_b=audio_size_b,
                    status=status,
                    error_message=error_message,
                )
                session.add(entry)
            logger.debug("Logged TTS request to DB (status=%s)", status)
        except Exception:
            logger.exception("Failed to log TTS request to DB")
