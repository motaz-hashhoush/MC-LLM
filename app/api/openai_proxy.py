"""
OpenAI-compatible proxy route.

Exposes ``POST /v1/chat/completions`` so standard OpenAI SDK clients can talk
to the gateway instead of hitting vLLM directly. Every call is logged to
PostgreSQL (``request_logs``) using the same :class:`DBLogger` as the rest of
the gateway, then proxied to vLLM and returned verbatim — both streaming and
non-streaming responses preserve the OpenAI wire format.

Point your OpenAI client at this gateway's ``/v1`` base URL (port 8080) rather
than vLLM's port 8000, and keep 8000 closed so requests can't bypass logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.db.db_logger import DBLogger
from app.llm.vllm_client import VLLMClient

logger = logging.getLogger(__name__)

openai_router = APIRouter(prefix="/v1", tags=["OpenAI-compatible"])

# task_type recorded in request_logs for proxied chat completions
_TASK_TYPE = "chat"


def _get_vllm_client(request: Request) -> VLLMClient:
    """Retrieve the shared vLLM client from application state."""
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")
    return engine.client


def _get_db_logger(request: Request) -> DBLogger:
    """Retrieve the shared DBLogger from application state."""
    db_logger = getattr(request.app.state, "db_logger", None)
    if db_logger is None:
        raise HTTPException(status_code=503, detail="Logging not ready")
    return db_logger


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _inject_no_think(messages: list[dict]) -> list[dict]:
    """Append /no_think to the last user message if not already present."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            content = messages[i].get("content") or ""
            if "/no_think" not in content:
                messages = [m.copy() for m in messages]
                messages[i] = {**messages[i], "content": f"{content}\n/no_think"}
            return messages
    return messages


def _summarise_messages(messages: list[dict]) -> str:
    """Serialise the OpenAI ``messages`` array for the ``input_text`` column."""
    try:
        return json.dumps(messages, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(messages)


@openai_router.post("/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat-completions endpoint that proxies to vLLM and logs
    the request/response to PostgreSQL.
    """
    client = _get_vllm_client(request)
    db_logger = _get_db_logger(request)

    try:
        payload = await request.json()
    except Exception as exc:  # malformed body
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    if not isinstance(payload, dict) or "messages" not in payload:
        raise HTTPException(
            status_code=400, detail="Request body must contain a 'messages' array"
        )

    payload["messages"] = _inject_no_think(payload.get("messages") or [])

    job_id = str(uuid.uuid4())
    input_text = _summarise_messages(payload.get("messages", []))
    stream = bool(payload.get("stream", False))
    start = time.perf_counter()

    # Fire-and-forget request log (never adds to response latency)
    asyncio.create_task(
        db_logger.log_request(
            job_id=job_id,
            task_type=_TASK_TYPE,
            input_text=input_text,
        )
    )

    if stream:
        return await _proxy_streaming(client, db_logger, job_id, payload, start)
    return await _proxy_non_streaming(client, db_logger, job_id, payload, start)


async def _proxy_non_streaming(
    client: VLLMClient,
    db_logger: DBLogger,
    job_id: str,
    payload: dict,
    start: float,
) -> JSONResponse:
    """Forward a non-streaming request, log the outcome, return vLLM's JSON."""
    try:
        response = await client.chat_completion(payload)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception("Chat-completion proxy %s failed to reach vLLM", job_id)
        asyncio.create_task(
            db_logger.log_error(job_id=job_id, error_message=str(exc), latency_ms=latency_ms)
        )
        raise HTTPException(status_code=502, detail=f"Upstream vLLM error: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    data = response.json()

    if response.status_code != 200:
        error_msg = (
            data.get("error", {}).get("message")
            if isinstance(data, dict)
            else str(data)
        ) or f"vLLM returned status {response.status_code}"
        asyncio.create_task(
            db_logger.log_error(job_id=job_id, error_message=error_msg, latency_ms=latency_ms)
        )
        return JSONResponse(status_code=response.status_code, content=data)

    # Extract assistant text + token usage for the completion log
    try:
        raw_content = data["choices"][0]["message"]["content"] or ""
        cleaned = _strip_think(raw_content)
        data["choices"][0]["message"]["content"] = cleaned
    except (KeyError, IndexError, TypeError):
        cleaned = ""
    tokens_used = (data.get("usage") or {}).get("total_tokens")

    asyncio.create_task(
        db_logger.log_completion(
            job_id=job_id,
            output_text=cleaned,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
    )
    return JSONResponse(status_code=200, content=data)


async def _proxy_streaming(
    client: VLLMClient,
    db_logger: DBLogger,
    job_id: str,
    payload: dict,
    start: float,
) -> StreamingResponse:
    """
    Forward a streaming request, relaying SSE chunks unchanged while
    accumulating the assistant text so the full response can be logged once
    the stream finishes.
    """
    # Ask vLLM to include usage in the final chunk so we can log token counts.
    payload.setdefault("stream_options", {})
    if isinstance(payload["stream_options"], dict):
        payload["stream_options"].setdefault("include_usage", True)

    async def event_stream():
        chunks: list[str] = []
        tokens_used: int | None = None
        try:
            async for raw in client.stream_chat_completion(payload):
                yield raw
                chunks.append(raw.decode("utf-8", errors="ignore"))
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception("Streaming chat-completion proxy %s failed", job_id)
            asyncio.create_task(
                db_logger.log_error(job_id=job_id, error_message=str(exc), latency_ms=latency_ms)
            )
            return

        latency_ms = (time.perf_counter() - start) * 1000
        output_text, tokens_used = _parse_sse_stream("".join(chunks))
        asyncio.create_task(
            db_logger.log_completion(
                job_id=job_id,
                output_text=_strip_think(output_text),
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _parse_sse_stream(buffer: str) -> tuple[str, int | None]:
    """
    Reconstruct the assistant text and token usage from a buffered SSE stream
    of OpenAI ``chat.completion.chunk`` events.
    """
    parts: list[str] = []
    tokens_used: int | None = None
    for line in buffer.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices", []) or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                parts.append(delta["content"])
        if obj.get("usage"):
            tokens_used = obj["usage"].get("total_tokens", tokens_used)
    return "".join(parts), tokens_used
