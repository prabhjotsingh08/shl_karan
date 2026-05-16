"""Off-topic and prompt-injection detection."""

from __future__ import annotations

import re
from dataclasses import dataclass


_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(your\s+)?(system|safety)\s+",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+)?(dan|jailbreak)",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"pretend\s+you\s+are\s+not",
)

_OFF_TOPIC_KEYWORDS = (
    "weather",
    "stock price",
    "bitcoin",
    "recipe",
    "football",
    "movie",
    "write me a poem",
    "write code for",
    "solve this math",
    "who won the election",
    "tell me a joke",
    "translate this paragraph",
    "homework help",
    "python script that",
    "javascript function",
)


_SHL_ANCHOR_KEYWORDS = (
    "shl",
    "assessment",
    "assessments",
    "test",
    "tests",
    "hire",
    "hiring",
    "recruit",
    "job",
    "role",
    "candidate",
    "screening",
    "personality",
    "aptitude",
    "cognitive",
    "sjt",
    "simulation",
    "competenc",
    "developer",
    "engineer",
    "analyst",
    "manager",
    "graduate",
    "compare",
    "recommend",
    "refine",
    "fewer",
    "more",
    "remote",
    "adaptive",
)


@dataclass(frozen=True)
class RefusalDecision:
    should_refuse: bool
    reason: str | None = None


def check_message(text: str, *, has_prior_assistant_turn: bool = False) -> RefusalDecision:
    """Return whether to refuse the latest user message."""
    normalized = text.strip().lower()
    if not normalized:
        return RefusalDecision(should_refuse=True, reason="empty")

    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return RefusalDecision(should_refuse=True, reason="injection")

    if _looks_off_topic(normalized) and not has_prior_assistant_turn:
        if not _has_shl_anchor(normalized):
            return RefusalDecision(should_refuse=True, reason="off_topic")

    return RefusalDecision(should_refuse=False)


def _looks_off_topic(text: str) -> bool:
    return any(keyword in text for keyword in _OFF_TOPIC_KEYWORDS)


def _has_shl_anchor(text: str) -> bool:
    return any(keyword in text for keyword in _SHL_ANCHOR_KEYWORDS)
