"""
Inference engine — handles prompt construction and vLLM client interaction.
Replaces the Ray Serve LLMWorker.
"""

from __future__ import annotations

import logging
import time
# No prompt template? Oh it was in LLMWorker
import re
from typing import Any

from app.llm.prompts import PromptTemplate
from app.llm.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Directly processes inference tasks using VLLMClient.
    """

    def __init__(self) -> None:
        self._client = VLLMClient()
        logger.info("InferenceEngine initialised")

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single task dict and return the result dict.

        Parameters
        ----------
        task:
            Must contain ``id``, ``task`` (type), ``text``, and ``parameters``.

        Returns
        -------
        dict
            ``{"result": str}`` on success, ``{"error": str}`` on failure.
        """
        task_type: str = task["task"]
        text: str = task["text"]
        params: dict[str, Any] = task.get("parameters", {})
        job_id: str = task["id"]

        think_enabled: bool = params.get("think", True)
        if not think_enabled:
            text = f"{text}\n/no_think"

        start = time.perf_counter()
        try:
            if task_type == "generate":
                # User provides the full prompt, no template needed
                messages = [{"role": "user", "content": text}]
            else:
                messages = PromptTemplate.build_prompt(task_type, text)

            raw_result = await self._client.generate(
                messages=messages,
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
            )

            # Remove <think>...</think> tags only if think_enabled is False
            if not think_enabled:
                result = re.sub(r"<think>.*?</think>", "", raw_result, flags=re.DOTALL).strip()
            else:
                result = raw_result.strip()

            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Job %s completed in %.1f ms (task=%s, think=%s)",
                job_id,
                latency_ms,
                task_type,
                think_enabled,
            )
            return {"result": result, "latency_ms": latency_ms}

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception("Job %s failed after %.1f ms", job_id, latency_ms)
            return {"error": str(exc), "latency_ms": latency_ms}

    async def close(self) -> None:
        """Close the underlying vLLM client."""
        await self._client.close()
