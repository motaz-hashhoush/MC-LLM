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


class JobStatus(str, Enum):
    """Lifecycle status of a queued job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Requests ─────────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    """Incoming request body shared by all task endpoints."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="The input text to process.",
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


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    version: str = _settings.APP_VERSION
    model: str = _settings.MODEL_NAME
