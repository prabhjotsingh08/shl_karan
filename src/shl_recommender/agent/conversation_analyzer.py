"""Analyze conversation history for intent and context sufficiency."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence

from ..data_models import ChatMessage


class ConversationIntent(str, Enum):
    OFF_TOPIC = "off_topic"
    END = "end"
    COMPARE = "compare"
    REFINE = "refine"
    CLARIFY = "clarify"
    RECOMMEND = "recommend"


_ROLE_KEYWORDS = (
    "developer",
    "engineer",
    "analyst",
    "manager",
    "consultant",
    "graduate",
    "intern",
    "sales",
    "customer service",
    "nurse",
    "accountant",
    "designer",
    "administrator",
    "technician",
    "specialist",
    "lead",
    "director",
)

_SKILL_KEYWORDS = (
    "python",
    "java",
    "sql",
    "javascript",
    "collaborat",
    "communicat",
    "leadership",
    "problem solving",
    "numerical",
    "verbal",
    "cognitive",
    "personality",
    "technical",
    "stakeholder",
    "team",
)

_ASSESSMENT_GOAL_KEYWORDS = (
    "assessment",
    "assessments",
    "test",
    "tests",
    "screen",
    "screening",
    "evaluate",
    "measure",
    "aptitude",
    "personality",
    "sjt",
    "simulation",
    "competenc",
    "ability",
    "knowledge",
)

_COMPARE_CUES = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    " vs.",
    "difference between",
    "differences between",
    "which is better",
    "better between",
)

_REFINE_CUES = (
    "refine",
    "narrow",
    "broader",
    "instead",
    "focus on",
    "more ",
    "less ",
    "fewer",
    "only ",
    "exclude",
    "remove",
    "add ",
    "without ",
    "shorter",
    "longer",
    "remote",
    "adaptive",
    "personality",
    "aptitude",
    "technical",
)

_END_CUES = (
    "thanks",
    "thank you",
    "that's all",
    "that is all",
    "goodbye",
    "bye",
    "done for now",
    "no more questions",
    "i'm done",
    "im done",
)


@dataclass
class ContextSignals:
    has_role: bool = False
    has_skills: bool = False
    has_assessment_goal: bool = False
    has_substantive_length: bool = False
    merged_user_text: str = ""

    @property
    def sufficiency_score(self) -> int:
        return sum(
            [
                self.has_role,
                self.has_skills,
                self.has_assessment_goal,
                self.has_substantive_length,
            ]
        )

    def is_sufficient(self) -> bool:
        if self.sufficiency_score >= 2:
            return True
        if self.has_assessment_goal and self.has_substantive_length:
            return True
        if len(self.merged_user_text) >= 120:
            return True
        return False


@dataclass
class AnalysisResult:
    intent: ConversationIntent
    signals: ContextSignals = field(default_factory=ContextSignals)
    latest_user_message: str = ""
    user_turn_count: int = 0
    role_hint: str | None = None
    refinement_cue: str | None = None


def count_user_turns(messages: Sequence[ChatMessage]) -> int:
    return sum(1 for message in messages if message.role == "user")


def merge_user_messages(messages: Sequence[ChatMessage]) -> str:
    parts = [message.content.strip() for message in messages if message.role == "user" and message.content.strip()]
    return "\n".join(parts)


def latest_user_message(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content.strip()
    return ""


def analyze(messages: Sequence[ChatMessage]) -> AnalysisResult:
    latest = latest_user_message(messages)
    merged = merge_user_messages(messages)
    user_turns = count_user_turns(messages)
    signals = _extract_signals(merged)
    role_hint = _extract_role_hint(merged)

    normalized_latest = latest.lower()
    if _matches_any(normalized_latest, _END_CUES) and user_turns > 0:
        return AnalysisResult(
            intent=ConversationIntent.END,
            signals=signals,
            latest_user_message=latest,
            user_turn_count=user_turns,
            role_hint=role_hint,
        )

    if _matches_any(normalized_latest, _COMPARE_CUES):
        return AnalysisResult(
            intent=ConversationIntent.COMPARE,
            signals=signals,
            latest_user_message=latest,
            user_turn_count=user_turns,
            role_hint=role_hint,
        )

    has_prior_assistant = any(message.role == "assistant" for message in messages)
    if has_prior_assistant and _matches_any(normalized_latest, _REFINE_CUES):
        return AnalysisResult(
            intent=ConversationIntent.REFINE,
            signals=signals,
            latest_user_message=latest,
            user_turn_count=user_turns,
            role_hint=role_hint,
            refinement_cue=latest[:200],
        )

    if not signals.is_sufficient():
        return AnalysisResult(
            intent=ConversationIntent.CLARIFY,
            signals=signals,
            latest_user_message=latest,
            user_turn_count=user_turns,
            role_hint=role_hint,
        )

    return AnalysisResult(
        intent=ConversationIntent.RECOMMEND,
        signals=signals,
        latest_user_message=latest,
        user_turn_count=user_turns,
        role_hint=role_hint,
    )


def _extract_signals(text: str) -> ContextSignals:
    lowered = text.lower()
    return ContextSignals(
        has_role=_contains_any(lowered, _ROLE_KEYWORDS),
        has_skills=_contains_any(lowered, _SKILL_KEYWORDS),
        has_assessment_goal=_contains_any(lowered, _ASSESSMENT_GOAL_KEYWORDS),
        has_substantive_length=len(text.strip()) >= 40,
        merged_user_text=text,
    )


def _extract_role_hint(text: str) -> str | None:
    lowered = text.lower()
    for keyword in _ROLE_KEYWORDS:
        if keyword in lowered:
            match = re.search(rf"(\w+\s+)?{re.escape(keyword)}", lowered)
            if match:
                return match.group(0).strip()
    hiring_match = re.search(
        r"(?:hiring|hire|role|position)(?:\s+for)?\s+(?:a\s+)?([\w\s-]{3,40})",
        lowered,
    )
    if hiring_match:
        return hiring_match.group(1).strip()
    return None


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _matches_any(text: str, cues: Sequence[str]) -> bool:
    return any(cue in text for cue in cues)
