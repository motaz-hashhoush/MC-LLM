"""
SupertonicTTSClient — wrapping supertone-inc/supertonic-3 (ONNX, CPU-first).

Model: Supertone/supertonic-3 (~260 MB ONNX, auto-downloaded on first run)
Sample rate: 44 100 Hz
Voices: M1–M5 (male), F1–F5 (female)
Languages: 31 languages + "na" (auto-detect)

The model is loaded once at startup via asyncio.to_thread(client.load) and
shared across all requests. synthesis() is thread-safe via self._lock.
"""

from __future__ import annotations

import gc
import io
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Raised when Supertonic TTS synthesis fails."""


class SupertonicTTSClient:
    """
    Async-friendly wrapper around the Supertonic 3 TTS model.

    Usage
    -----
    client = SupertonicTTSClient()
    client.load()                        # call once at startup
    wav = client.synthesize("مرحباً", language="ar", voice_name="M1")
    client.unload()                      # call at shutdown
    """

    SAMPLE_RATE: int = 44_100

    VALID_VOICES: frozenset[str] = frozenset(
        ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
    )

    # All ISO 639-1 codes supported by Supertonic 3, plus "na" for auto-detect.
    VALID_LANGUAGES: frozenset[str] = frozenset([
        "en", "de", "fr", "it", "es", "pt", "nl", "pl", "ru", "uk",
        "cs", "sk", "hr", "sl", "ro", "bg", "el", "sv", "da", "et",
        "fi", "hu", "lt", "lv", "tr", "ja", "ko", "zh", "vi", "id",
        "hi", "ar", "na",
    ])

    def __init__(self) -> None:
        self._tts: Any = None
        self._lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the Supertonic 3 model.

        On first call the model (~260 MB) is downloaded from HuggingFace and
        cached in HF_HOME (defaults to /root/.cache/huggingface, which is
        bind-mounted via the app_cache Docker volume).

        Synchronous / CPU-bound — call via asyncio.to_thread() at startup.

        Raises
        ------
        RuntimeError
            If the supertonic package is not installed or the model fails to load.
        """
        logger.info("Loading Supertonic 3 TTS model (auto_download=True)…")
        try:
            from supertonic import TTS  # type: ignore[import]

            self._tts = TTS(auto_download=True)
            logger.info(
                "Supertonic 3 TTS model loaded (sample_rate=%d Hz)", self.SAMPLE_RATE
            )

            # Warmup: one cheap synthesis to pre-compile the ONNX execution plan.
            # total_steps=4 keeps it fast; any real request will use >= 8.
            logger.info("Running TTS warmup…")
            try:
                style = self._tts.get_voice_style(voice_name="M1")
                self._tts.synthesize(
                    text="Hello.",
                    lang="en",
                    voice_style=style,
                    total_steps=4,
                    speed=1.0,
                )
                logger.info("TTS warmup complete.")
            except Exception as warmup_exc:
                logger.warning("TTS warmup failed (non-fatal): %s", warmup_exc)

        except ImportError as exc:
            raise RuntimeError(
                "supertonic package is not installed. "
                "Ensure 'supertonic' is in requirements.txt."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load Supertonic TTS model: {exc}") from exc

    def unload(self) -> None:
        """Release the model from memory and run GC."""
        self._tts = None
        gc.collect()
        logger.info("Supertonic TTS model unloaded")

    def is_loaded(self) -> bool:
        """Return True if the model is currently in memory."""
        return self._tts is not None

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        language: str = "ar",
        speed: float = 1.05,
        voice_name: str = "M1",
        total_steps: int = 8,
    ) -> bytes:
        """
        Synthesise *text* and return raw WAV bytes at 44 100 Hz.

        Thread-safe — acquires self._lock before calling the model so that
        concurrent HTTP requests do not collide on the single ONNX session.

        Parameters
        ----------
        text:
            Input text. Supertonic auto-chunks long inputs and inserts 0.3 s
            of silence between chunks.
        language:
            ISO 639-1 code from VALID_LANGUAGES, or ``"na"`` for auto-detect.
        speed:
            Playback speed in [0.7, 2.0]. Default 1.05.
        voice_name:
            One of M1–M5 (male) or F1–F5 (female).
        total_steps:
            Denoising steps: 5 (fast / lower quality) to 12 (slower / higher
            quality). Default 8 balances speed and quality for CPU inference.

        Returns
        -------
        bytes
            Raw WAV audio (16-bit PCM, 44 100 Hz, mono).

        Raises
        ------
        TTSError
            On synthesis failure or invalid arguments.
        """
        if not self.is_loaded():
            raise TTSError("Supertonic model is not loaded. Call load() first.")
        if voice_name not in self.VALID_VOICES:
            raise TTSError(
                f"Invalid voice '{voice_name}'. "
                f"Choose from: {', '.join(sorted(self.VALID_VOICES))}"
            )
        if language not in self.VALID_LANGUAGES:
            raise TTSError(
                f"Unsupported language '{language}'. "
                f"Supported: {', '.join(sorted(self.VALID_LANGUAGES))}"
            )

        start_ms = time.monotonic()
        try:
            import soundfile as sf  # type: ignore[import]

            with self._lock:
                style = self._tts.get_voice_style(voice_name=voice_name)
                wav, _ = self._tts.synthesize(
                    text=text,
                    lang=language,
                    voice_style=style,
                    total_steps=total_steps,
                    speed=speed,
                )

            # wav is float32 numpy array, shape (1, num_samples)
            buffer = io.BytesIO()
            sf.write(buffer, wav.squeeze(), self.SAMPLE_RATE, format="WAV")
            wav_bytes = buffer.getvalue()

        except TTSError:
            raise
        except Exception as exc:
            logger.exception("Supertonic synthesis failed")
            raise TTSError(f"Synthesis failed: {exc}") from exc

        elapsed_ms = (time.monotonic() - start_ms) * 1000
        logger.info(
            "Supertonic synthesis complete "
            "(lang=%s, voice=%s, steps=%d, size=%d bytes, %.1f ms)",
            language,
            voice_name,
            total_steps,
            len(wav_bytes),
            elapsed_ms,
        )
        return wav_bytes

    # ── Format conversion ─────────────────────────────────────────────────────

    def convert_to_mp3(self, wav_bytes: bytes) -> bytes:
        """
        Convert raw WAV bytes to MP3 bytes via pydub / ffmpeg.

        Raises
        ------
        TTSError
            If pydub is not installed or ffmpeg is unavailable.
        """
        try:
            from pydub import AudioSegment  # type: ignore[import]

            audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format="mp3")
            return mp3_buffer.getvalue()
        except ImportError as exc:
            raise TTSError(
                "pydub is not installed. Add 'pydub>=0.25.1' to requirements.txt."
            ) from exc
        except Exception as exc:
            raise TTSError(f"MP3 conversion failed: {exc}") from exc
