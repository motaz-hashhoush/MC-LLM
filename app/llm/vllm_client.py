"""
Async client for the vLLM OpenAI-compatible inference API.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class VLLMClient:
    """
    Wraps the vLLM ``/v1/chat/completions`` endpoint using :mod:`httpx`.

    The client is designed to be long-lived and reused across requests.
    Call :meth:`close` (or use as an async context-manager) to release
    the underlying connection pool.
    """

    def __init__(self, endpoint: str | None = None) -> None:
        settings = get_settings()
        self._endpoint = (endpoint or settings.VLLM_ENDPOINT).rstrip("/")
        self._model = settings.MODEL_NAME
        self._http = httpx.AsyncClient(
            base_url=self._endpoint,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )

    # ── Async Context Manager ────────────────────────────────────────────

    async def __aenter__(self) -> "VLLMClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP connection pool."""
        await self._http.aclose()
        logger.info("VLLMClient connection closed")

    # ── Inference ────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Send a chat-completion request to vLLM and return the generated text.

        Parameters
        ----------
        messages:
            OpenAI-compatible message list.
        max_tokens:
            Max tokens to generate (defaults to ``Settings.MAX_TOKENS``).
        temperature:
            Sampling temperature (defaults to ``Settings.TEMPERATURE``).

        Returns
        -------
        str
            The generated completion text.

        Raises
        ------
        httpx.HTTPStatusError
            If the vLLM server returns a non-2xx status.
        """
        settings = get_settings()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or settings.MAX_TOKENS,
            "temperature": temperature if temperature is not None else settings.TEMPERATURE,
        }

        logger.debug("vLLM request → %s/v1/chat/completions", self._endpoint)

        response = await self._http.post("/v1/chat/completions", json=payload)
        
        if response.status_code != 200:
            try:
                error_data = response.json()
                # vLLM returns error in {"error": {"message": "...", "type": "...", ...}}
                if isinstance(error_data, dict) and "error" in error_data:
                    error_msg = error_data["error"].get("message", response.text)
                    raise Exception(f"{error_msg}")
            except (ValueError, KeyError, AttributeError):
                pass
            response.raise_for_status()

        data = response.json()
        content: str = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {})
        logger.info(
            "vLLM response: prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )

        return content

    # ── Health ───────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Return True if the vLLM server is reachable."""
        try:
            resp = await self._http.get("/health")
            return resp.status_code == 200
        except Exception:
            return False
