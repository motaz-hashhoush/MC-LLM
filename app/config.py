"""
Application configuration using Pydantic BaseSettings.

All settings are loaded from environment variables with sensible defaults.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the LLM inference infrastructure."""

    # --- Application ---
    APP_NAME: str = "MC-LLM Inference Gateway"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- LLM Model ---
    MODEL_NAME: str = "Qwen/Qwen3-14B"

    # --- STT Model (faster-whisper / CTranslate2) ---
    STT_MODEL_NAME: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    STT_MODEL_PATH: str = ""  # If set, load from this local path instead of HuggingFace hub
    STT_DEVICE: str = "cpu"          # "cuda" | "cpu"
    STT_COMPUTE_TYPE: str = "int8"   # "float16" (GPU) | "int8" (CPU)
    STT_MAX_AUDIO_MB: int = 25
    STT_LANGUAGE: str = "ar"          # ISO 639-1 code: "ar"=Arabic, "en"=English, etc.

    # --- vLLM ---
    VLLM_ENDPOINT: str = "http://vllm:8000"

    # --- Redis ---
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    # --- PostgreSQL ---
    DATABASE_URL: str = "postgresql+asyncpg://llm_user:llm_pass@host.docker.internal:5432/llm_logs"

    # --- Inference Defaults ---
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7

    # --- Queue ---
    QUEUE_NAME: str = "llm:tasks"
    RESULT_TTL: int = 3600  # seconds
    JOB_TIMEOUT: int = 120  # seconds

    # --- TTS (Supertonic 3) ---
    # Model auto-downloads to HF_HOME (~/.cache/huggingface) on first run.
    TTS_MODEL_NAME: str = "Supertone/supertonic-3"
    TTS_MAX_CHARS: int = 10000
    TTS_DEFAULT_LANGUAGE: str = "ar"
    TTS_DEFAULT_VOICE: str = "M1"       # M1–M5 (male) | F1–F5 (female)
    TTS_TOTAL_STEPS: int = 8            # denoising steps: 5 (fast) – 12 (quality)
    TTS_DEFAULT_SPEED: float = 1.05
    TTS_DEFAULT_FORMAT: str = "wav"
    # ONNX Runtime threading. 0 = auto (ORT picks physical cores only).
    # On CPU-only inference, set INTRA close to the container's logical core count
    # for maximum single-request throughput. INTER stays low (graph is mostly sequential).
    TTS_INTRA_OP_THREADS: int = 0
    TTS_INTER_OP_THREADS: int = 0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
