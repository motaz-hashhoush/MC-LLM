"""
Task-specific prompt templates for LLM inference.

Each task type maps to a system message and a user message template.
Templates can be extended by adding new entries to :data:`TEMPLATES`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Template:
    """A single prompt template with system and user parts."""

    system: str
    user: str


# ── Template Registry ────────────────────────────────────────────────────────

TEMPLATES: dict[str, _Template] = {
    "summarize": _Template(
        system=(
            "You are an expert summariser. Produce a clear, concise summary "
            "that captures the key points of the provided text."
        ),
        user="Summarize the following text in a concise paragraph:\n\n{text}",
    ),
    "rewrite": _Template(
        system=(
            "You are a professional editor. Rewrite the text to be clearer "
            "and more polished without changing its meaning."
        ),
        user=(
            "Rewrite the following text professionally without changing "
            "its meaning:\n\n{text}"
        ),
    ),
    "generate": _Template(
        system=(
            "You are a skilled content writer. Produce high-quality, detailed "
            "content based on the given topic or outline."
        ),
        user="Write a detailed article about:\n\n{text}",
    ),
}


# ── Public API ───────────────────────────────────────────────────────────────

class PromptTemplate:
    """Builds chat-completion message lists from registered templates."""

    @staticmethod
    def get_available_tasks() -> list[str]:
        """Return all registered task names."""
        return list(TEMPLATES.keys())

    @staticmethod
    def build_prompt(task: str, text: str) -> list[dict[str, str]]:
        """
        Build an OpenAI-compatible ``messages`` list for the given *task*.

        Parameters
        ----------
        task:
            One of the registered task types (e.g. ``"summarize"``).
        text:
            The user-supplied input text.

        Returns
        -------
        list[dict[str, str]]
            ``[{"role": "system", ...}, {"role": "user", ...}]``

        Raises
        ------
        ValueError
            If *task* is not a recognised template name.
        """
        template = TEMPLATES.get(task)
        if template is None:
            raise ValueError(
                f"Unknown task '{task}'. "
                f"Available: {PromptTemplate.get_available_tasks()}"
            )

        return [
            {"role": "system", "content": template.system},
            {"role": "user", "content": template.user.format(text=text)},
        ]
