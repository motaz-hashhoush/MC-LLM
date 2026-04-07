"""
Async Redis client with connection pooling.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Manages an async Redis connection pool."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._pool: aioredis.ConnectionPool | None = None
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Create the connection pool and verify connectivity."""
        self._pool = aioredis.ConnectionPool(
            host=self._settings.REDIS_HOST,
            port=self._settings.REDIS_PORT,
            db=self._settings.REDIS_DB,
            password=self._settings.REDIS_PASSWORD,
            decode_responses=True,
            max_connections=20,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        await self._client.ping()
        logger.info(
            "Connected to Redis at %s:%s",
            self._settings.REDIS_HOST,
            self._settings.REDIS_PORT,
        )

    async def disconnect(self) -> None:
        """Gracefully close the connection pool."""
        if self._client:
            await self._client.aclose()
            logger.info("Redis connection closed")
        if self._pool:
            await self._pool.disconnect()

    def get_client(self) -> aioredis.Redis:
        """Return the active Redis client instance."""
        if self._client is None:
            raise RuntimeError("Redis client not initialised – call connect() first")
        return self._client

    async def health_check(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            if self._client:
                await self._client.ping()
                return True
        except Exception:
            pass
        return False
