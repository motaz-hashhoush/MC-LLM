"""
SQLAlchemy ORM models for the PostgreSQL logging database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all models."""

    pass


class RequestLog(Base):
    """Persisted log entry for every inference request."""

    __tablename__ = "request_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True
    )
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<RequestLog id={self.id} task={self.task_type} "
            f"status={self.status}>"
        )


class TTSRequestLog(Base):
    """Persisted log entry for every TTS synthesis request."""

    __tablename__ = "tts_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="ar")
    speed: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    format: Mapped[str] = mapped_column(String(10), nullable=False, default="wav")
    has_ref_audio: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_size_b: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="success", index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<TTSRequestLog id={self.id} lang={self.language} "
            f"status={self.status}>"
        )

