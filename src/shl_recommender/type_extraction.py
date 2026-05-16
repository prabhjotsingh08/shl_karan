"""Utilities for extracting assessment types from user queries."""

from __future__ import annotations

import re
from typing import Any, List, Set

import google.generativeai as genai
import json
import logfire


_TYPE_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}
_CODE_PATTERN = re.compile(r"\b([A-HKPS])\b")


_CATEGORY_GUIDANCE = (
    "A = Ability & Aptitude (cognitive, numerical, verbal, and logical reasoning)\n"
    "B = Biodata & Situational Judgement (scenario-based SJTs and biodata inventories)\n"
    "C = Competencies (behavioral or competency-based evaluations)\n"
    "D = Development & 360 (multi-rater feedback, coaching, development diagnostics)\n"
    "E = Assessment Exercises (in-tray, group, role-play, or assessment center exercises)\n"
    "K = Knowledge & Skills (technical, compliance, or certification knowledge tests)\n"
    "P = Personality & Behaviour (personality traits, motivations, values, behavioral styles)\n"
    "S = Simulations (job simulations or realistic job previews)"
)

_SYSTEM_INSTRUCTION = (
    "Select the most relevant assessment type codes for the user's request. "
    "Use only uppercase letters from {codes}. "
    "Provide at most three distinct codes, separated by spaces or commas, with no extra commentary. "
    "Category meanings:\n{guidance}"
).format(codes=", ".join(sorted(_TYPE_CODES)), guidance=_CATEGORY_GUIDANCE)


class GeminiExtractionError(Exception):
    """Raised when Gemini cannot return usable type codes."""

    def __init__(self, message: str, *, finish_reasons: List[str] | None = None) -> None:
        super().__init__(message)
        self.finish_reasons = finish_reasons or []


class GeminiTypeExtractor:
    """LLM-backed assessment type extraction using Gemini."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        response_tokens: int = 128,
    ) -> None:
        self.model_name = model_name
        self._response_tokens = response_tokens

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_SYSTEM_INSTRUCTION,
        )

    def extract(self, query: str) -> List[str]:
        """Return up to three assessment type codes inferred from the query."""

        prompt = self._build_prompt(query)
        response = _generate_with_logging(
            self.model,
            prompt,
            self._response_tokens,
            response_mime_type=None,
        )
        result_text = _coalesce_response_text(response)
        _log_response_contents(query=query, response=response, text=result_text or None)
        if not result_text:
            finish_reasons = _collect_finish_reasons(response)
            fallback_query = self._compact_prompt(query)
            retry_response = _generate_with_logging(
                self.model,
                fallback_query,
                self._response_tokens,
                response_mime_type=None,
            )
            retry_text = _coalesce_response_text(retry_response)
            _log_response_contents(query=f"{query} [retry]", response=retry_response, text=retry_text or None)
            if retry_text:
                result_text = retry_text
            else:
                raise GeminiExtractionError(
                    "Gemini returned no text",
                    finish_reasons=finish_reasons or _collect_finish_reasons(retry_response),
                )

        candidates = _extract_codes_from_text(result_text)
        if not candidates:
            raise GeminiExtractionError(
                "Gemini returned no recognizable codes",
                finish_reasons=_collect_finish_reasons(response),
            )
        return candidates

    @staticmethod
    def _compact_prompt(query: str) -> str:
        return (
            "Provide JSON with key \"types\" listing up to three codes from "
            "[\"A\",\"B\",\"C\",\"D\",\"E\",\"K\",\"P\",\"S\"] relevant to this request. "
            "Base your selection on the following category meanings:\n"
            f"{_CATEGORY_GUIDANCE}\n"
            "Reply with JSON only.\n"
            f"Request: {query}"
        )

    @staticmethod
    def _build_prompt(query: str) -> str:
        return (
            "Identify which assessment type codes apply to the following request. "
            "Available categories:\n"
            f"{_CATEGORY_GUIDANCE}\n"
            "Return the codes only, separated by spaces or commas.\n"
            f"Request: {query}"
        )


def _extract_codes_from_text(value: str) -> List[str]:
    candidates = [match.group(1).upper() for match in _CODE_PATTERN.finditer(value)]
    seen: Set[str] = set()
    ordered = []
    for code in candidates:
        if code in _TYPE_CODES and code not in seen:
            ordered.append(code)
            seen.add(code)
    return ordered[:3]


def _generate_with_logging(
    model: genai.GenerativeModel,
    query: str,
    response_tokens: int,
    *,
    response_mime_type: str | None,
):
    try:
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.1,
            "max_output_tokens": response_tokens,
        }
        if response_mime_type:
            generation_config["response_mime_type"] = response_mime_type

        return model.generate_content(
            query,
            generation_config=generation_config,
        )
    except Exception as exc:  # pragma: no cover - network call
        logfire.warn(
            "Gemini type extraction request failed",
            error=str(exc),
            exc_info=True,
        )
        raise


def _coalesce_response_text(response) -> str:
    texts: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) if content else []
        for part in parts or []:
            text = getattr(part, "text", None)
            if text:
                texts.append(str(text))
                continue
            function_call = getattr(part, "function_call", None)
            if function_call:
                args = getattr(function_call, "args", None)
                if args:
                    try:
                        texts.append(json.dumps(args))
                    except TypeError:
                        texts.append(str(args))
                    continue
            inline_data = getattr(part, "inline_data", None)
            if inline_data:
                data = getattr(inline_data, "data", None)
                if data is not None:
                    texts.append(str(data))
                    continue
    if texts:
        return "\n".join(texts).strip()

    try:
        fallback_text = getattr(response, "text", "")
    except Exception:
        fallback_text = ""
    return str(fallback_text or "").strip()


def _collect_finish_reasons(response) -> List[str]:
    reasons: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            reasons.append(str(reason))
    return reasons


def _log_response_contents(*, query: str, response: Any, text: str | None) -> None:
    """Emit structured details about the Gemini response for observability."""
    candidates_payload: List[dict[str, Any]] = []
    for candidate in getattr(response, "candidates", []) or []:
        payload: dict[str, Any] = {
            "finish_reason": getattr(candidate, "finish_reason", None),
            "safety_ratings": getattr(candidate, "safety_ratings", None),
        }
        content = getattr(candidate, "content", None)
        parts_repr: List[dict[str, Any]] = []
        if content:
            for part in getattr(content, "parts", []) or []:
                part_repr: dict[str, Any] = {}
                if hasattr(part, "text") and getattr(part, "text"):
                    part_repr["text"] = str(part.text)
                function_call = getattr(part, "function_call", None)
                if function_call:
                    part_repr["function_call"] = {
                        "name": getattr(function_call, "name", None),
                        "args": getattr(function_call, "args", None),
                    }
                inline_data = getattr(part, "inline_data", None)
                if inline_data:
                    part_repr["inline_data"] = {
                        "mime_type": getattr(inline_data, "mime_type", None),
                        "data_preview": str(getattr(inline_data, "data", None))[:200],
                    }
                if part_repr:
                    parts_repr.append(part_repr)
        if parts_repr:
            payload["parts"] = parts_repr
        candidates_payload.append(payload)

    logfire.info(
        "Gemini raw response",
        query=query,
        response_text=text,
        finish_reasons=_collect_finish_reasons(response) or None,
        candidates=candidates_payload or None,
    )


