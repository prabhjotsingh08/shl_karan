"""Compare SHL catalog assessments using metadata only."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Sequence

from ..data_models import AssessmentMetadata, ChatRecommendationItem


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _extract_compare_phrases(query_text: str) -> List[str]:
    """Pull assessment name fragments from compare-style user messages."""
    text = query_text.strip()
    patterns = (
        r"(?:compare|difference between|differences between)\s+(.+?)\s+and\s+(.+?)(?:\s+tests?|\s+assessments?)?\s*[?.!]*\s*$",
        r"(?:compare|difference between|differences between)\s+(.+?)(?:\s+and\s+|\s+vs\.?\s+)(.+?)(?:\s+tests?|\s+assessments?)?\s*[?.!]*\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            left, right = match.group(1).strip(" ."), match.group(2).strip(" .")
            if left and right:
                return [left, right]
    return []


def _score_name_against_phrase(record: AssessmentMetadata, phrase: str) -> float:
    phrase_norm = _normalize_name(phrase)
    name_norm = _normalize_name(record.name)
    if not phrase_norm:
        return 0.0
    if len(phrase_norm) <= 4:
        if phrase_norm in name_norm.split() or re.search(
            rf"\b{re.escape(phrase_norm)}\b",
            name_norm,
        ):
            return 0.95
        description = (record.description or "").lower()
        if re.search(rf"\({re.escape(phrase_norm)}\)", description):
            return 0.93
        if re.search(rf"\b{re.escape(phrase_norm)}\b", description):
            return 0.9
    if phrase_norm in name_norm or name_norm in phrase_norm:
        return 0.98
    phrase_tokens = [token for token in re.split(r"[^\w]+", phrase_norm) if len(token) > 2]
    name_tokens = [token for token in re.split(r"[^\w]+", name_norm) if len(token) > 2]
    if not phrase_tokens:
        return 0.0
    hits = sum(1 for token in phrase_tokens if token in name_norm)
    if hits == len(phrase_tokens):
        return 0.92
    if hits >= max(2, len(phrase_tokens) - 1):
        return 0.85
    return SequenceMatcher(None, name_norm, phrase_norm).ratio()


def find_catalog_matches(
    query_text: str,
    catalog: Dict[str, AssessmentMetadata],
    *,
    limit: int = 4,
    min_ratio: float = 0.55,
) -> List[AssessmentMetadata]:
    """Fuzzy-match assessment names mentioned in user text."""
    phrases = _extract_compare_phrases(query_text)
    if phrases:
        phrase_matches: List[AssessmentMetadata] = []
        seen: set[str] = set()
        for phrase in phrases:
            best_score = 0.0
            best_record: AssessmentMetadata | None = None
            for record in catalog.values():
                score = _score_name_against_phrase(record, phrase)
                if score > best_score:
                    best_score = score
                    best_record = record
            if best_record is not None and best_score >= 0.75:
                key = _normalize_name(best_record.name)
                if key not in seen:
                    seen.add(key)
                    phrase_matches.append(best_record)
        if len(phrase_matches) >= 2:
            return phrase_matches[:limit]

    lowered = query_text.lower()
    matches: List[tuple[float, AssessmentMetadata]] = []

    for record in catalog.values():
        name_lower = record.name.lower()
        ratio = SequenceMatcher(None, name_lower, lowered).ratio()
        if name_lower in lowered:
            ratio = max(ratio, 0.95)
        else:
            tokens = [token for token in re.split(r"[^\w]+", name_lower) if len(token) > 3]
            token_hits = sum(1 for token in tokens if token in lowered)
            if token_hits >= 2:
                ratio = max(ratio, 0.7 + 0.05 * token_hits)

        if ratio >= min_ratio:
            matches.append((ratio, record))

    matches.sort(key=lambda item: item[0], reverse=True)
    seen_names: set[str] = set()
    ordered: List[AssessmentMetadata] = []
    for _, record in matches:
        key = _normalize_name(record.name)
        if key in seen_names:
            continue
        seen_names.add(key)
        ordered.append(record)
        if len(ordered) >= limit:
            break
    return ordered


def match_from_recommendations(
    query_text: str,
    recommendations: Sequence[ChatRecommendationItem],
) -> List[ChatRecommendationItem]:
    lowered = query_text.lower()
    matched: List[ChatRecommendationItem] = []
    for item in recommendations:
        if item.name.lower() in lowered:
            matched.append(item)
    return matched


def build_comparison_reply(records: Sequence[AssessmentMetadata]) -> str:
    if len(records) < 2:
        return ""

    lines = ["Here is a catalog-grounded comparison of the requested SHL assessments:\n"]
    for record in records:
        types = ", ".join(record.human_readable_types()) or "Not specified"
        duration = record.assessment_length or "Not listed"
        remote = record.remote_label()
        adaptive = record.adaptive_label()
        description = (record.description or "No description in catalog.").strip()
        if len(description) > 220:
            description = description[:217] + "..."

        lines.append(f"**{record.name}**")
        lines.append(f"- Types: {types}")
        lines.append(f"- Duration: {duration}")
        lines.append(f"- Remote testing: {remote}")
        lines.append(f"- Adaptive: {adaptive}")
        lines.append(f"- Summary: {description}")
        lines.append("")

    lines.append("All details above are taken only from the SHL catalog metadata.")
    return "\n".join(lines)

