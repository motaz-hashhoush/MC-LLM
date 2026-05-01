"""
WhisperClient — singleton wrapping openai/whisper-large-v3-turbo via faster-whisper.

faster-whisper uses CTranslate2 for efficient inference (2–4× faster than
the standard transformers pipeline, lower VRAM usage).

Model format required: CTranslate2 (Systran/faster-whisper-large-v3-turbo).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from typing import Any

from faster_whisper import WhisperModel

from app.config import get_settings

logger = logging.getLogger(__name__)


class WhisperClient:
    """
    Async-friendly wrapper around the faster-whisper WhisperModel.

    Usage
    -----
    client = WhisperClient()                               # call once at startup
    text   = await client.transcribe(audio_bytes, filename="audio.mp3")
    await  client.close()                                  # call at shutdown
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._max_audio_bytes: int = settings.STT_MAX_AUDIO_MB * 1024 * 1024

        # Prefer local bind-mounted path; fall back to HuggingFace hub download
        model_source: str = settings.STT_MODEL_PATH.strip() or settings.STT_MODEL_NAME
        local = bool(settings.STT_MODEL_PATH.strip())

        logger.info(
            "Loading faster-whisper model from %s '%s' (device=%s, compute_type=%s) …",
            "local path" if local else "HuggingFace hub",
            model_source,
            settings.STT_DEVICE,
            settings.STT_COMPUTE_TYPE,
        )

        self._model = WhisperModel(
            model_source,
            device=settings.STT_DEVICE,
            compute_type=settings.STT_COMPUTE_TYPE,
        )

        logger.info(
            "WhisperClient ready (source=%s, device=%s, compute_type=%s)",
            model_source,
            settings.STT_DEVICE,
            settings.STT_COMPUTE_TYPE,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str = "ar",
    ) -> str:
        """
        Transcribe *audio_bytes* and return the text string.

        Parameters
        ----------
        audio_bytes:
            Raw audio content received from the HTTP upload.
        filename:
            Original filename — used to preserve the file extension so
            ffmpeg can detect the codec automatically.
        language:
            ISO 639-1 language code (e.g. ``"ar"``, ``"en"``).
            Defaults to ``"ar"`` (Arabic).
        """
        if len(audio_bytes) > self._max_audio_bytes:
            raise ValueError(
                f"Audio file exceeds the {get_settings().STT_MAX_AUDIO_MB} MB limit."
            )

        # Preserve original extension so ffmpeg picks the right decoder
        ext = os.path.splitext(filename)[-1] or ".wav"
        tmp_path = os.path.join(
            tempfile.gettempdir(), f"whisper_{uuid.uuid4().hex}{ext}"
        )

        try:
            # Write bytes to temp file (sync — negligible I/O overhead)
            with open(tmp_path, "wb") as fh:
                fh.write(audio_bytes)

            # Run synchronous faster-whisper inference in a thread pool
            loop = asyncio.get_event_loop()
            transcript, detected_lang, duration = await loop.run_in_executor(
                None,
                self._run_transcription,
                tmp_path,
                language,
            )

            logger.info(
                "Transcription complete (lang=%s, detected=%s, duration=%.1fs, chars=%d)",
                language,
                detected_lang,
                duration,
                len(transcript),
            )
            return transcript

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def close(self) -> None:
        """Release the model from memory."""
        del self._model
        logger.info("WhisperClient closed")

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_transcription(
        self, tmp_path: str, language: str
    ) -> tuple[str, str, float]:
        """
        Synchronous faster-whisper transcription (runs in thread pool).

        Returns
        -------
        tuple[str, str, float]
            (transcript, detected_language_code, audio_duration_seconds)
        """
        segments, info = self._model.transcribe(
            tmp_path,
            language=language,     # ISO 639-1 code: "ar", "en", etc.
            beam_size=5,
            task="transcribe",
            vad_filter=True,       # skip silent segment — reduces hallucinations
            vad_parameters={"min_silence_duration_ms": 500},
        )

        # Materialise the generator and join all segment texts
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        return transcript, info.language, info.duration
