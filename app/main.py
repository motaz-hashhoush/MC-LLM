"""
MC-LLM Inference Gateway — FastAPI application entry point.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.api.routes import router, tts_router
from app.config import get_settings
from app.db.database import DatabaseManager
from app.queue.job_queue import JobQueue
from app.queue.redis_client import RedisClient
from app.services.task_processor import TaskProcessor
from app.services.inference_engine import InferenceEngine
from app.queue.queue_consumer import QueueConsumer
from app.db.db_logger import DBLogger
from app.stt.whisper_client import WhisperClient
from app.tts.silma_client import SilmaTTSClient
from app.tts.tts_processor import TTSProcessor

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def _configure_logging() -> None:
    settings = get_settings()
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
    # Quieten noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup / shutdown of Redis and PostgreSQL connections."""
    _configure_logging()
    settings = get_settings()

    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # Redis
    redis_client = RedisClient()
    await redis_client.connect()

    # Database
    db = DatabaseManager()
    await db.init_db()

    # Job queue & processor
    job_queue = JobQueue(redis_client.get_client())
    task_processor = TaskProcessor(job_queue)

    # Initialize Inference Engine
    inference_engine = InferenceEngine()

    # Queue consumer for background processing
    db_logger = DBLogger(db)
    queue_consumer = QueueConsumer(
        job_queue=job_queue,
        processor=inference_engine,
        db_logger=db_logger,
    )
    await queue_consumer.start()

    # Attach to app state for dependency injection
    app.state.redis_client = redis_client
    app.state.db = db
    app.state.job_queue = job_queue
    app.state.task_processor = task_processor
    app.state.queue_consumer = queue_consumer
    app.state.inference_engine = inference_engine

    # Whisper STT + SILMA TTS — construct/load both concurrently to cut startup time.
    # WhisperClient loads the model in __init__, so we wrap it in a thread as well.
    silma_client = SilmaTTSClient(
        model_path=settings.TTS_MODEL_PATH,
        device=settings.TTS_DEVICE,
    )

    logger.info("Loading Whisper STT and SILMA TTS models in parallel…")
    whisper_client, _ = await asyncio.gather(
        asyncio.to_thread(WhisperClient),
        asyncio.to_thread(silma_client.load),
    )
    logger.info("Whisper STT and SILMA TTS models ready")

    app.state.whisper_client = whisper_client
    app.state.silma_client = silma_client
    app.state.tts_processor = TTSProcessor(silma_client, db_logger)

    logger.info("Application startup complete")
    yield

    # Shutdown
    logger.info("Shutting down…")
    await queue_consumer.stop()
    await inference_engine.close()
    await whisper_client.close()
    silma_client.unload()
    logger.info("SILMA TTS model unloaded on shutdown")
    await redis_client.disconnect()
    await db.dispose()
    logger.info("Shutdown complete")


# ── App ──────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Factory function for the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-ready LLM inference gateway powered by vLLM and Redis.",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.include_router(tts_router)
    return app


app = create_app()
