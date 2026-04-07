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

    # --- Model ---
    MODEL_NAME: str = "Qwen/Qwen3-14B"

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

    # --- Ray Serve ---
    RAY_HEAD_ADDRESS: str = "ray://ray-head:10001"
    RAY_MIN_REPLICAS: int = 1
    RAY_MAX_REPLICAS: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
