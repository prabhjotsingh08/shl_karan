"""Main conversational agent orchestrating dialogue flows."""

from __future__ import annotations

from typing import List, Sequence

import logfire

from ..config import get_settings
from ..data_models import ChatMessage, ChatRecommendationItem, ChatResponse
from ..recommender import RecommendationEngine
from .clarification_engine import build_clarification_question
from .comparison_engine import (
    build_comparison_reply,
    find_catalog_matches,
    match_from_recommendations,
)
from .conversation_analyzer import ConversationIntent, analyze
from .llm_reply import maybe_create_reply_generator
from .prompt_templates import (
    END_CONVERSATION_REPLY,
    NO_CATALOG_MATCH_REPLY,
    REFUSAL_REPLY,
    TURN_LIMIT_REPLY,
    clarification_reply,
    comparison_prompt_for_names,
    format_recommendation_summary,
    recommendation_intro,
    refinement_acknowledgement,
)
from .recommendation_orchestrator import RecommendationOrchestrator
from .refusal_guard import check_message


class ChatAgent:
    """Stateless conversational agent over the SHL catalog."""

    def __init__(self, engine: RecommendationEngine | None = None) -> None:
        self.settings = get_settings()
        self.engine = engine or RecommendationEngine()
        self.orchestrator = RecommendationOrchestrator(self.engine)
        self._reply_generator = maybe_create_reply_generator()

    def handle(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        if not messages:
            return ChatResponse(
                reply="Please send a message describing the role or assessments you need.",
                recommendations=[],
                end_of_conversation=False,
            )

        if len(messages) > self.settings.max_conversation_turns:
            return ChatResponse(
                reply=TURN_LIMIT_REPLY,
                recommendations=[],
                end_of_conversation=True,
            )

        analysis = analyze(messages)
        latest = analysis.latest_user_message
        if not latest:
            return ChatResponse(
                reply="Please provide your hiring or assessment question.",
                recommendations=[],
                end_of_conversation=False,
            )

        has_prior_assistant = any(message.role == "assistant" for message in messages)
        refusal = check_message(latest, has_prior_assistant_turn=has_prior_assistant)
        if refusal.should_refuse:
            logfire.info("Refused user message", reason=refusal.reason)
            return ChatResponse(reply=REFUSAL_REPLY, recommendations=[], end_of_conversation=False)

        if analysis.intent == ConversationIntent.END:
            return ChatResponse(
                reply=END_CONVERSATION_REPLY,
                recommendations=[],
                end_of_conversation=True,
            )

        if analysis.intent == ConversationIntent.COMPARE:
            return self._handle_compare(messages, latest)

        if analysis.intent == ConversationIntent.CLARIFY:
            question = build_clarification_question(analysis.signals)
            return ChatResponse(
                reply=clarification_reply(question),
                recommendations=[],
                end_of_conversation=False,
            )

        user_texts = [message.content for message in messages if message.role == "user"]
        recommendations = self.orchestrator.recommend_chat_items(
            user_texts,
            min_results=self.settings.chat_min_recommendations,
            max_results=self.settings.chat_max_recommendations,
        )

        if not recommendations:
            return ChatResponse(
                reply=NO_CATALOG_MATCH_REPLY,
                recommendations=[],
                end_of_conversation=False,
            )

        if analysis.intent == ConversationIntent.REFINE:
            intro = refinement_acknowledgement(analysis.refinement_cue)
        else:
            intro = recommendation_intro(len(recommendations), analysis.role_hint)

        summary = format_recommendation_summary(recommendations)
        draft = f"{intro}\n\n{summary}"
        reply = self._maybe_polish(draft, recommendations)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=False,
        )

    def _handle_compare(self, messages: Sequence[ChatMessage], latest: str) -> ChatResponse:
        catalog = self.engine.catalog_index
        matched = find_catalog_matches(latest, catalog)

        if len(matched) < 2:
            prior_items = self._prior_recommendations_from_history(messages)
            matched_items = match_from_recommendations(latest, prior_items)
            if len(matched_items) >= 2:
                matched = [
                    catalog[record.entity_id]
                    for record in catalog.values()
                    if str(record.url) in {str(item.url) for item in matched_items}
                ][:4]

        if len(matched) < 2:
            return ChatResponse(
                reply=comparison_prompt_for_names(),
                recommendations=[],
                end_of_conversation=False,
            )

        comparison_records = matched[:4]
        reply = build_comparison_reply(comparison_records)
        recs = [record.to_chat_recommendation_item() for record in comparison_records]

        return ChatResponse(
            reply=reply or comparison_prompt_for_names(),
            recommendations=recs,
            end_of_conversation=False,
        )

    def _prior_recommendations_from_history(
        self,
        messages: Sequence[ChatMessage],
    ) -> List[ChatRecommendationItem]:
        """Re-run retrieval on prior user context to recover comparable items."""
        user_texts = [message.content for message in messages if message.role == "user"]
        if len(user_texts) < 2:
            return []
        return self.orchestrator.recommend_chat_items(
            user_texts[:-1],
            min_results=2,
            max_results=8,
        )

    def _maybe_polish(self, draft: str, recommendations: List[ChatRecommendationItem]) -> str:
        if self._reply_generator is None:
            return draft
        try:
            return self._reply_generator.polish(draft, recommendations)
        except Exception as exc:
            logfire.warn("Reply polish skipped", error=str(exc))
            return draft
