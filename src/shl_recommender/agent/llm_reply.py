"""Optional Gemini-backed reply polishing (catalog-grounded)."""

from __future__ import annotations

from typing import List, Optional

import logfire

from ..config import get_settings
from ..data_models import ChatRecommendationItem

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


class GroundedReplyGenerator:
    """Generate natural-language replies constrained to provided facts."""

    def __init__(self, *, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=(
                "You are an SHL assessment advisor. Use ONLY the assessment facts provided. "
                "Do not invent assessments or attributes. Keep replies concise (under 120 words)."
            ),
        )

    def polish(self, draft_reply: str, recommendations: List[ChatRecommendationItem]) -> str:
        facts = _format_facts(recommendations)
        prompt = (
            f"Draft reply:\n{draft_reply}\n\n"
            f"Catalog facts you may reference:\n{facts}\n\n"
            "Rewrite the draft as a helpful assistant message. Do not add new assessments."
        )
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 256,
                },
            )
            text = getattr(response, "text", "") or ""
            cleaned = str(text).strip()
            return cleaned or draft_reply
        except Exception as exc:  # pragma: no cover - network
            logfire.warn("Reply polish failed; using template", error=str(exc))
            return draft_reply


def maybe_create_reply_generator() -> Optional[GroundedReplyGenerator]:
    settings = get_settings()
    if not settings.gemini_api_key or not settings.chat_use_gemini_replies:
        return None
    if genai is None:
        return None
    try:
        return GroundedReplyGenerator(api_key=settings.gemini_api_key)
    except Exception as exc:
        logfire.warn("Grounded reply generator unavailable", error=str(exc))
        return None


def _format_facts(items: List[ChatRecommendationItem]) -> str:
    lines: List[str] = []
    for item in items[:8]:
        types = item.test_type or "N/A"
        lines.append(f"- {item.name}: types={types}; url={item.url}")
    return "\n".join(lines) if lines else "No assessments selected."
