"""Prompt and reply templates for the conversational agent."""

from __future__ import annotations

from typing import List

from ..data_models import ChatRecommendationItem


REFUSAL_REPLY = (
    "I can only help with SHL individual assessments from the official catalog — "
    "for example finding, refining, or comparing tests for a role. "
    "Please ask about hiring needs or SHL assessments."
)

TURN_LIMIT_REPLY = (
    "We have reached the maximum of 8 conversation turns. "
    "Please start a new conversation if you need further help with SHL assessments."
)

END_CONVERSATION_REPLY = (
    "Glad I could help. If you need more SHL assessment guidance later, feel free to start a new conversation."
)

NO_CATALOG_MATCH_REPLY = (
    "I could not find matching assessments in the SHL catalog for that request. "
    "Could you share more about the role, skills, or assessment types you need?"
)


def clarification_reply(question: str) -> str:
    return question


def recommendation_intro(count: int, role_hint: str | None = None) -> str:
    prefix = f"Based on your needs{f' for {role_hint}' if role_hint else ''}, "
    if count == 1:
        return prefix + "here is 1 SHL assessment from the catalog that fits well:"
    return prefix + f"here are {count} SHL assessments from the catalog that fit well:"


def format_recommendation_summary(items: List[ChatRecommendationItem], *, max_items: int = 5) -> str:
    lines: List[str] = []
    for idx, item in enumerate(items[:max_items], start=1):
        types = item.test_type or "N/A"
        lines.append(f"{idx}. **{item.name}** (type {types})")
    if len(items) > max_items:
        lines.append(f"... and {len(items) - max_items} more in the recommendations list.")
    return "\n".join(lines)


def refinement_acknowledgement(cue: str | None = None) -> str:
    if cue:
        return f"I have refined the recommendations with your preference: {cue}."
    return "I have updated the recommendations based on your latest message."


def comparison_prompt_for_names() -> str:
    return (
        "Which SHL assessments would you like me to compare? "
        "Please name two or more from the catalog, or refer to ones I recently recommended."
    )
