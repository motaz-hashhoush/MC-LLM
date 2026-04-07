"""
Ray Serve LLM worker deployment.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import ray
from ray import serve

from app.config import get_settings
from app.llm.prompts import PromptTemplate
from app.llm.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


@serve.deployment(
    name="LLMWorker",
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={
        "min_replicas": get_settings().RAY_MIN_REPLICAS,
        "max_replicas": get_settings().RAY_MAX_REPLICAS,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10,
)
class LLMWorker:
    """
    Ray Serve deployment that processes inference tasks.

    Each replica holds its own :class:`VLLMClient` connection.
    """

    def __init__(self) -> None:
        self._client = VLLMClient()
        logger.info("LLMWorker replica initialised")

    async def __call__(self, task: dict[str, Any]) -> dict[str, Any]:
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

        start = time.perf_counter()
        try:
            messages = PromptTemplate.build_prompt(task_type, text)

            result = await self._client.generate(
                messages=messages,
                max_tokens=params.get("max_tokens"),
                temperature=params.get("temperature"),
            )

            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Job %s completed in %.1f ms (task=%s)",
                job_id,
                latency_ms,
                task_type,
            )
            return {"result": result, "latency_ms": latency_ms}

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception("Job %s failed after %.1f ms", job_id, latency_ms)
            return {"error": str(exc), "latency_ms": latency_ms}
