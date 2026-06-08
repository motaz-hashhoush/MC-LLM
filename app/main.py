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
from app.db.db_logger import DBLogger
from app.services.inference_engine import InferenceEngine
from app.services.task_processor import TaskProcessor
from app.stt.whisper_client import WhisperClient
from app.tts.supertonic_client import SupertonicTTSClient
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
    """Manage startup / shutdown of database and model connections."""
    _configure_logging()
    settings = get_settings()

    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # Database
    db = DatabaseManager()
    await db.init_db()
    db_logger = DBLogger(db)

    # Inference engine — direct vLLM calls, no queue indirection
    inference_engine = InferenceEngine()
    task_processor = TaskProcessor(inference_engine, db_logger)

    # Single semaphore serialises all in-process GPU work (Whisper STT + SILMA TTS)
    # so concurrent requests cannot exhaust VRAM on the shared fastapi-container GPU.
    gpu_semaphore = asyncio.Semaphore(1)

    app.state.db = db
    app.state.task_processor = task_processor
    app.state.inference_engine = inference_engine
    app.state.gpu_semaphore = gpu_semaphore

    # Whisper STT + Supertonic TTS — load both concurrently to cut startup time
    tts_client = SupertonicTTSClient(
        intra_op_num_threads=settings.TTS_INTRA_OP_THREADS or None,
        inter_op_num_threads=settings.TTS_INTER_OP_THREADS or None,
    )

    logger.info("Loading Whisper STT and Supertonic TTS models in parallel…")
    whisper_client, _ = await asyncio.gather(
        asyncio.to_thread(WhisperClient),
        asyncio.to_thread(tts_client.load),
    )
    logger.info("Whisper STT and Supertonic TTS models ready")

    app.state.whisper_client = whisper_client
    app.state.tts_client = tts_client
    app.state.tts_processor = TTSProcessor(tts_client, db_logger, gpu_semaphore)

    logger.info("Application startup complete")
    yield

    # Shutdown
    logger.info("Shutting down…")
    await inference_engine.close()
    await whisper_client.close()
    tts_client.unload()
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
