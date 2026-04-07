"""
Async database engine and session management.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages the async SQLAlchemy engine and session factory.

    Typical lifecycle::

        db = DatabaseManager()
        await db.init_db()      # create tables + verify connection
        ...
        await db.dispose()      # shutdown
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Create all tables (if they do not already exist)."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialised")

    async def dispose(self) -> None:
        """Dispose of the engine connection pool."""
        await self._engine.dispose()
        logger.info("Database engine disposed")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a request-scoped async session."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def health_check(self) -> bool:
        """Return True if the database is reachable."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(
                    __import__("sqlalchemy").text("SELECT 1")
                )
            return True
        except Exception:
            return False
