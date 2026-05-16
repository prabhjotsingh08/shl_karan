"""Unit tests for conversational agent modules."""

from __future__ import annotations

import unittest

from src.shl_recommender.agent.clarification_engine import build_clarification_question
from src.shl_recommender.agent.comparison_engine import find_catalog_matches
from src.shl_recommender.agent.conversation_analyzer import (
    ConversationIntent,
    ContextSignals,
    analyze,
)
from src.shl_recommender.agent.conversation_analyzer import merge_user_messages
from src.shl_recommender.agent.refusal_guard import check_message
from src.shl_recommender.agent.recommendation_orchestrator import build_search_query
from src.shl_recommender.agent.chat_agent import ChatAgent
from src.shl_recommender.data_models import AssessmentMetadata, ChatMessage


class TestConversationAnalyzer(unittest.TestCase):
    def test_vague_query_triggers_clarify(self) -> None:
        messages = [ChatMessage(role="user", content="Hi")]
        result = analyze(messages)
        self.assertEqual(result.intent, ConversationIntent.CLARIFY)

    def test_rich_query_triggers_recommend(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content=(
                    "Hiring a Java developer with strong collaboration skills; "
                    "need cognitive and personality screening assessments."
                ),
            )
        ]
        result = analyze(messages)
        self.assertEqual(result.intent, ConversationIntent.RECOMMEND)

    def test_compare_intent(self) -> None:
        messages = [ChatMessage(role="user", content="Compare numerical reasoning and verbal reasoning tests")]
        result = analyze(messages)
        self.assertEqual(result.intent, ConversationIntent.COMPARE)


class TestRefusalGuard(unittest.TestCase):
    def test_off_topic_refused(self) -> None:
        decision = check_message("What is the weather today?")
        self.assertTrue(decision.should_refuse)

    def test_shl_query_allowed(self) -> None:
        decision = check_message("Recommend personality assessments for managers")
        self.assertFalse(decision.should_refuse)


class TestClarificationEngine(unittest.TestCase):
    def test_role_question_when_missing_role(self) -> None:
        question = build_clarification_question(ContextSignals())
        self.assertIn("role", question.lower())


class TestRecommendationOrchestrator(unittest.TestCase):
    def test_merge_user_messages(self) -> None:
        query = build_search_query(["Need Java developer", "Also personality tests"])
        self.assertIn("Latest request", query)
        self.assertIn("personality", query.lower())


class TestComparisonEngine(unittest.TestCase):
    def test_find_catalog_matches(self) -> None:
        catalog = {
            "1": AssessmentMetadata(
                entity_id="1",
                name="Numerical Reasoning",
                url="https://www.shl.com/products/product-catalog/view/numerical-reasoning/",
                assessment_types={"A"},
            ),
            "2": AssessmentMetadata(
                entity_id="2",
                name="Verbal Reasoning",
                url="https://www.shl.com/products/product-catalog/view/verbal-reasoning/",
                assessment_types={"A"},
            ),
        }
        matches = find_catalog_matches(
            "Compare Numerical Reasoning and Verbal Reasoning",
            catalog,
        )
        self.assertGreaterEqual(len(matches), 2)


class TestTurnLimit(unittest.TestCase):
    def test_turn_cap_counts_user_and_assistant(self) -> None:
        agent = ChatAgent.__new__(ChatAgent)
        from src.shl_recommender.config import get_settings

        agent.settings = get_settings()
        messages = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ] * 5  # 10 messages
        response = ChatAgent.handle(agent, messages)
        self.assertTrue(response.end_of_conversation)


class TestMergeUserMessages(unittest.TestCase):
    def test_merge(self) -> None:
        messages = [
            ChatMessage(role="user", content="Hiring analyst"),
            ChatMessage(role="assistant", content="What skills?"),
            ChatMessage(role="user", content="Cognitive tests"),
        ]
        merged = merge_user_messages(messages)
        self.assertIn("analyst", merged)
        self.assertIn("Cognitive", merged)


if __name__ == "__main__":
    unittest.main()
