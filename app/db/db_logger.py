"""
Database logger — persists structured request logs to PostgreSQL.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.db.database import DatabaseManager
from app.db.models import RequestLog

logger = logging.getLogger(__name__)


class DBLogger:
    """
    High-level API for recording request lifecycle events in the database.

    Each method is safe to call independently; failures are logged but
    never propagated to callers so that logging issues cannot break the
    inference pipeline.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def log_request(
        self,
        job_id: str,
        task_type: str,
        input_text: str,
    ) -> None:
        """Record a new incoming request."""
        try:
            async with self._db.get_session() as session:
                entry = RequestLog(
                    id=uuid.UUID(job_id),
                    task_type=task_type,
                    input_text=input_text,
                    status="pending",
                )
                session.add(entry)
            logger.debug("Logged request %s to DB", job_id)
        except Exception:
            logger.exception("Failed to log request %s", job_id)

    async def log_completion(
        self,
        job_id: str,
        output_text: str,
        tokens_used: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Update a request log with its successful result."""
        try:
            async with self._db.get_session() as session:
                entry = await session.get(RequestLog, uuid.UUID(job_id))
                if entry:
                    entry.status = "completed"
                    entry.output_text = output_text
                    entry.tokens_used = tokens_used
                    entry.latency_ms = latency_ms
                    entry.completed_at = datetime.now(timezone.utc)
            logger.debug("Logged completion for %s", job_id)
        except Exception:
            logger.exception("Failed to log completion for %s", job_id)

    async def log_error(
        self,
        job_id: str,
        error_message: str,
        latency_ms: float | None = None,
    ) -> None:
        """Update a request log with failure details."""
        try:
            async with self._db.get_session() as session:
                entry = await session.get(RequestLog, uuid.UUID(job_id))
                if entry:
                    entry.status = "failed"
                    entry.error_message = error_message
                    entry.latency_ms = latency_ms
                    entry.completed_at = datetime.now(timezone.utc)
            logger.debug("Logged error for %s", job_id)
        except Exception:
            logger.exception("Failed to log error for %s", job_id)
