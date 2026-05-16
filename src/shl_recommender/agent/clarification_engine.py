"""Generate clarification questions when hiring context is insufficient."""

from __future__ import annotations

from .conversation_analyzer import ContextSignals


def build_clarification_question(signals: ContextSignals) -> str:
    """Return a single targeted clarification question."""
    if not signals.has_role and not signals.has_substantive_length:
        return (
            "To recommend SHL assessments from the catalog, what role or job title are you hiring for "
            "(for example software engineer, customer service advisor, or graduate analyst)?"
        )

    if not signals.has_skills and not signals.has_assessment_goal:
        return (
            "What skills or qualities should the assessments measure for this role "
            "(for example technical knowledge, cognitive ability, personality, or situational judgement)?"
        )

    if not signals.has_assessment_goal:
        return (
            "Which types of SHL assessments are you looking for — for example aptitude, personality, "
            "technical knowledge, or job simulations?"
        )

    return (
        "Could you share a bit more detail about the role level, key competencies, "
        "or whether you need remote-friendly or shorter assessments?"
    )
