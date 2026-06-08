"""
Pydantic request / response schemas for the API gateway.
"""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field

from app.config import get_settings

_settings = get_settings()


# ── Enums ────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """Supported NLP task types."""

    SUMMARIZE = "summarize"
    REWRITE = "rewrite"
    GENERATE = "generate"
    TRANSCRIBE = "transcribe"


class JobStatus(str, Enum):
    """Lifecycle status of a queued job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Requests ─────────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    """Incoming request body shared by all task endpoints."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="The input text to process.",
    )
    think: bool = Field(
        default=False,
        description="Whether to include thinking tags in the generation process. Only applies to /generate.",
    )
    max_tokens: int = Field(
        default=_settings.MAX_TOKENS,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=_settings.TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )


# ── Responses ────────────────────────────────────────────────────────────────

class TaskResponse(BaseModel):
    """Standard response returned to the client."""

    id: UUID
    task: TaskType
    status: JobStatus
    result: str | None = None
    error: str | None = None


class TranscribeResponse(BaseModel):
    """Response returned by the /transcribe endpoint."""

    id: UUID
    status: JobStatus
    transcript: str | None = None
    language: str = "ar"
    error: str | None = None


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    version: str = _settings.APP_VERSION
    model: str = _settings.MODEL_NAME
    stt_model: str = _settings.STT_MODEL_NAME


# ── TTS Schemas ───────────────────────────────────────────────────────────────

_TTS_LANGUAGES = frozenset([
    "en", "de", "fr", "it", "es", "pt", "nl", "pl", "ru", "uk",
    "cs", "sk", "hr", "sl", "ro", "bg", "el", "sv", "da", "et",
    "fi", "hu", "lt", "lv", "tr", "ja", "ko", "zh", "vi", "id",
    "hi", "ar", "na",
])

_TTS_VOICES = frozenset(["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"])


class TTSRequest(BaseModel):
    """Request body for the POST /v1/tts endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to synthesise. Max 10 000 characters.",
    )
    language: str = Field(
        default="ar",
        description=(
            "ISO 639-1 language code (e.g. 'ar', 'en', 'fr') "
            "or 'na' for automatic language detection. "
            "Supertonic 3 supports 31 languages."
        ),
    )
    voice_name: str = Field(
        default="M1",
        description="Voice style: M1–M5 (male) or F1–F5 (female).",
    )
    speed: float = Field(
        default=1.05,
        ge=0.7,
        le=2.0,
        description="Playback speed multiplier (0.7 – 2.0).",
    )
    total_steps: int = Field(
        default=8,
        ge=5,
        le=12,
        description="Denoising steps: 5 (fastest) to 12 (highest quality).",
    )
    format: str = Field(
        default="wav",
        description="Output audio format: 'wav' or 'mp3'.",
    )

    from pydantic import field_validator

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v

    @field_validator("language")
    @classmethod
    def valid_language(cls, v: str) -> str:
        if v not in _TTS_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported codes: {', '.join(sorted(_TTS_LANGUAGES))}"
            )
        return v

    @field_validator("voice_name")
    @classmethod
    def valid_voice(cls, v: str) -> str:
        if v not in _TTS_VOICES:
            raise ValueError(
                f"Invalid voice '{v}'. Choose from: {', '.join(sorted(_TTS_VOICES))}"
            )
        return v

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        if v not in ("wav", "mp3"):
            raise ValueError("format must be 'wav' or 'mp3'")
        return v
