"""
SilmaTTSClient — singleton wrapping silma-ai/silma-tts (F5-TTS diffusion backend).

Model loading is deferred to an explicit load() call (triggered at startup via
FastAPI lifespan) so that the import itself is lightweight.

Testing
-------
1. Arabic synthesis:
   curl -X POST http://localhost:8080/v1/audio/speech \\
     -H "Content-Type: application/json" \\
     -d '{"text":"مرحباً بكم","language":"ar"}' \\
     --output test_ar.wav

2. English synthesis:
   curl -X POST http://localhost:8080/v1/audio/speech \\
     -H "Content-Type: application/json" \\
     -d '{"text":"Hello from NNU Media Center","language":"en","format":"mp3"}' \\
     --output test_en.mp3
"""

from __future__ import annotations

import gc
import io
import logging
import threading
import time
from typing import Any
import os
import tempfile
import warnings

# Suppress annoying transformers/hub warnings triggered by silma-tts
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

logger = logging.getLogger(__name__)


# ── Custom exception ──────────────────────────────────────────────────────────

class TTSError(Exception):
    """Raised when SILMA TTS synthesis fails."""


# ── Client ────────────────────────────────────────────────────────────────────

class SilmaTTSClient:
    """
    Async-friendly wrapper around the SILMA TTS model.

    Usage
    -----
    client = SilmaTTSClient(model_path="/models/silma-ai/silma-tts")
    client.load()                                        # call once at startup
    wav_bytes = client.synthesize(text="مرحباً", language="ar")
    client.unload()                                      # call at shutdown
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """
        Store configuration. Model is NOT loaded here.

        Parameters
        ----------
        model_path:
            Absolute path to the locally downloaded SILMA TTS model directory.
        device:
            Inference device. Use ``"cuda"`` for high performance or ``"cpu"``.
        """
        self.model_path: str = model_path
        self.device: str = device
        self._tts: Any = None
        self._lock: threading.Lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the SILMA TTS model from model_path.

        This is a synchronous, CPU-intensive operation that should be called
        from a thread pool via ``asyncio.to_thread(client.load)`` during
        FastAPI startup.

        Raises
        ------
        RuntimeError
            If model files are not found at model_path.
        """
        

        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"SILMA TTS model files not found at '{self.model_path}'. "
                "Run scripts/download_model.py to download the model first."
            )

        logger.info(
            "Loading SILMA TTS model from '%s' on device='%s' …",
            self.model_path,
            self.device,
        )

        try:
            from silma_tts.api import SilmaTTS  # type: ignore[import]

            # The API requires passing full paths to model files rather than just the directory
            ckpt_file = os.path.join(self.model_path, "model.pt")
            vocab_file = os.path.join(self.model_path, "vocab.txt")
            
            if not os.path.exists(ckpt_file) or not os.path.exists(vocab_file):
                 raise RuntimeError(f"Missing required model files (model.pt or vocab.txt) in {self.model_path}")

            self._tts = SilmaTTS(
                model="SilmaTTS_V1_Small",
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                device=self.device
            )
            logger.info(
                "SILMA TTS model loaded from %s on %s", self.model_path, self.device
            )

            # ── Warmup: pre-populate the ref-audio/text cache ──────────────
            # The library auto-transcribes the ref audio (via whisper-large-v3-turbo)
            # on the FIRST call if ref_text is empty. This takes 1-2 min.
            # providing the exact hardcoded transcript skips transcription entirely.
            logger.info("Running TTS warmup to pre-cache reference audio…")
            _warmup_out = os.path.join(tempfile.gettempdir(), "_silma_warmup.wav")
            _DEFAULT_REF = ("/usr/local/lib/python3.12/site-packages"
                            "/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav")
            # Exact transcript of ar.ref.24k.wav to skip internal transcription
            _DEFAULT_REF_TEXT = (
                "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية "
                "ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
            )
            # Use 16 steps for GPU (High Quality), 8 steps for CPU (Speed)
            _nfe = 16 if "cuda" in self.device else 8
            try:
                self._tts.infer(
                    ref_file=_DEFAULT_REF,
                    ref_text=_DEFAULT_REF_TEXT,
                    gen_text="مرحباً.",  # minimal Arabic text
                    speed=1.0,
                    nfe_step=_nfe,
                    file_wave=_warmup_out,
                )
                logger.info("TTS warmup complete — models pre-cached.")
            except Exception as warmup_exc:
                logger.warning("TTS warmup failed (non-fatal): %s", warmup_exc)
            finally:
                if os.path.exists(_warmup_out):
                    os.unlink(_warmup_out)
        except ImportError as exc:
            raise RuntimeError(
                "silma-tts package is not correctly installed. "
                "Ensure 'silma-tts>=1.0.0' is in requirements.txt."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SILMA TTS model: {exc}"
            ) from exc

    def unload(self) -> None:
        """
        Release the model from memory and run garbage collection.

        Safe to call even if the model was never loaded.
        """
        self._tts = None
        gc.collect()
        logger.info("SILMA TTS model unloaded")

    def is_loaded(self) -> bool:
        """Return ``True`` if the model is currently loaded in memory."""
        return self._tts is not None

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        language: str = "ar",
        speed: float = 1.0,
        clone_audio_path: str | None = None,
    ) -> bytes:
        """
        Synthesize *text* and return raw WAV bytes.

        Thread-safe — acquires ``self._lock`` before calling the model so that
        concurrent HTTP requests do not collide on the single model instance.

        Parameters
        ----------
        text:
            Input text to synthesise (Arabic or English).
        language:
            ISO 639-1 code: ``"ar"`` (Arabic MSA) or ``"en"`` (English).
        speed:
            Playback speed multiplier in the range ``[0.5, 2.0]``.
        clone_audio_path:
            Absolute path to a WAV file used for voice cloning.
            Pass ``None`` to use the model's built-in default voice.

        Returns
        -------
        bytes
            Raw WAV audio bytes.

        Raises
        ------
        TTSError
            If synthesis fails for any reason.
        """
        if not self.is_loaded():
            raise TTSError("SILMA TTS model is not loaded. Call load() first.")

        start_ms = time.monotonic()

        try:
            with self._lock:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    out_path = tmp.name

                # The package requires a reference audio (zero-shot architecture).
                # If the user didn't provide one, use the Arabic sample bundled with the package.
                default_ref = "/usr/local/lib/python3.12/site-packages/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav"
                
                if not clone_audio_path:
                    ref_file = default_ref
                    # Use hardcoded transcript to hit the cache perfectly
                    ref_text = (
                        "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية "
                        "ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
                    )
                else:
                    ref_file = clone_audio_path
                    # Empty string triggers auto-transcription for custom audio (cached after 1st time)
                    ref_text = ""

                # Use 16 steps for GPU (High Quality), 8 steps for CPU (Speed)
                _nfe = 16 if "cuda" in self.device else 8

                self._tts.infer(
                    ref_file=ref_file,
                    ref_text=ref_text,
                    gen_text=text,
                    speed=speed,
                    nfe_step=_nfe,
                    file_wave=out_path
                )

                with open(out_path, "rb") as f:
                    wav_bytes = f.read()

                os.unlink(out_path)

        except Exception as exc:
            logger.exception(
                "SILMA TTS synthesis failed (lang=%s, speed=%.2f, ref=%s)",
                language,
                speed,
                clone_audio_path,
            )
            raise TTSError(f"Synthesis failed: {exc}") from exc

        elapsed_ms = (time.monotonic() - start_ms) * 1000
        logger.info(
            "SILMA TTS synthesis complete (lang=%s, speed=%.2f, size=%d bytes, %.1f ms)",
            language,
            speed,
            len(wav_bytes),
            elapsed_ms,
        )
        return wav_bytes

    # ── Format conversion ─────────────────────────────────────────────────────

    def convert_to_mp3(self, wav_bytes: bytes) -> bytes:
        """
        Convert raw WAV bytes to MP3 bytes using pydub / ffmpeg.

        Parameters
        ----------
        wav_bytes:
            WAV audio data as returned by :meth:`synthesize`.

        Returns
        -------
        bytes
            MP3 audio bytes.

        Raises
        ------
        TTSError
            If the conversion fails (e.g. ffmpeg not installed).
        """
        try:
            from pydub import AudioSegment  # type: ignore[import]

            audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format="mp3")
            mp3_bytes = mp3_buffer.getvalue()
            logger.debug("WAV→MP3 conversion complete (%d bytes)", len(mp3_bytes))
            return mp3_bytes
        except ImportError as exc:
            raise TTSError(
                "pydub is not installed. Add 'pydub>=0.25.1' to requirements.txt."
            ) from exc
        except Exception as exc:
            logger.exception("WAV→MP3 conversion failed")
            raise TTSError(f"MP3 conversion failed: {exc}") from exc
