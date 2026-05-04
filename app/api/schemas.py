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
    think: bool | None = Field(
        default=None,
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

class TTSRequest(BaseModel):
    """Request body for the POST /v1/tts endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text to synthesise (Arabic or English). Max 2 000 characters.",
    )
    language: str = Field(
        default="ar",
        description="Language code: 'ar' (Arabic MSA) or 'en' (English).",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Playback speed multiplier (0.5 – 2.0).",
    )
    format: str = Field(
        default="wav",
        description="Output audio format: 'wav' or 'mp3'.",
    )
    clone_audio: str | None = Field(
        default=None,
        description=(
            "Base64-encoded WAV audio for voice cloning. "
            "When provided, the model will attempt to match the voice in the clip."
        ),
    )

    @classmethod
    def __get_validators__(cls):  # pydantic v1 compat hook (unused in v2)
        yield cls.model_validate

    # ── Field validators ──────────────────────────────────────────────────────

    from pydantic import field_validator

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > 2000:
            raise ValueError("text must be 2000 characters or fewer")
        return v

    @field_validator("language")
    @classmethod
    def valid_language(cls, v: str) -> str:
        if v not in ("ar", "en"):
            raise ValueError("language must be 'ar' or 'en'")
        return v

    @field_validator("speed")
    @classmethod
    def valid_speed(cls, v: float) -> float:
        if not (0.5 <= v <= 2.0):
            raise ValueError("speed must be between 0.5 and 2.0")
        return v

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        if v not in ("wav", "mp3"):
            raise ValueError("format must be 'wav' or 'mp3'")
        return v
