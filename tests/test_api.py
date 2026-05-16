"""API integration tests aligned with SHL_AI_Intern_Assignment.pdf."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

CATALOG_CSV = Path(__file__).resolve().parents[1] / "data" / "shl_individual_assessments.csv"


@pytest.fixture(scope="module")
def catalog_urls() -> set[str]:
    df = pd.read_csv(CATALOG_CSV)
    return set(df["url"].astype(str))


def _chat(client: TestClient, messages: list[dict]) -> dict:
    response = client.post("/chat", json={"messages": messages})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_returns_200_ok(client: TestClient) -> None:
    """Assignment: GET /health returns {\"status\": \"ok\"} with HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_response_schema(client: TestClient) -> None:
    """Assignment: non-negotiable reply / recommendations / end_of_conversation schema."""
    body = _chat(
        client,
        [{"role": "user", "content": "I need an assessment"}],
    )
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(body["reply"], str)
    assert isinstance(body["recommendations"], list)
    assert isinstance(body["end_of_conversation"], bool)


def test_vague_first_turn_clarifies_without_recommendations(client: TestClient) -> None:
    """Behavior probe: vague query should not shortlist on turn 1."""
    body = _chat(client, [{"role": "user", "content": "I need an assessment"}])
    assert body["recommendations"] == []
    assert body["end_of_conversation"] is False
    assert body["reply"].strip()


def test_rich_query_returns_catalog_shortlist(client: TestClient, catalog_urls: set[str]) -> None:
    """Assignment: 1–10 recommendations with name, url, and test_type when ready."""
    body = _chat(
        client,
        [
            {
                "role": "user",
                "content": (
                    "Hiring a mid-level Java developer who collaborates with stakeholders; "
                    "need cognitive aptitude and personality screening assessments."
                ),
            }
        ],
    )
    recs = body["recommendations"]
    assert 1 <= len(recs) <= 10
    for item in recs:
        assert "name" in item and item["name"]
        assert "url" in item and str(item["url"]) in catalog_urls
        assert "test_type" in item
        assert isinstance(item["test_type"], str)


def test_off_topic_refusal_empty_recommendations(client: TestClient) -> None:
    """Behavior probe: refuse general off-topic requests; recommendations stay empty."""
    body = _chat(client, [{"role": "user", "content": "What is the weather today?"}])
    assert body["recommendations"] == []
    assert "shl" in body["reply"].lower() or "assessment" in body["reply"].lower()


def test_compare_intent_grounded_reply(client: TestClient) -> None:
    """Assignment: comparison answers are grounded in catalog assessments (e.g. OPQ vs GSA)."""
    body = _chat(
        client,
        [
            {
                "role": "user",
                "content": "What is the difference between OPQ and GSA?",
            }
        ],
    )
    assert body["reply"].strip()
    names = {item["name"].lower() for item in body["recommendations"]}
    reply_lower = body["reply"].lower()
    assert any("opq" in name for name in names) or "opq" in reply_lower
    assert any("global skills" in name or "gsa" in name for name in names) or "gsa" in reply_lower


def test_refine_updates_shortlist_after_prior_turn(client: TestClient) -> None:
    """Behavior probe: mid-conversation refinement updates the shortlist."""
    turn1 = _chat(
        client,
        [
            {
                "role": "user",
                "content": (
                    "Hiring a software engineer; need technical knowledge and aptitude tests."
                ),
            }
        ],
    )
    assert turn1["recommendations"]

    turn2 = _chat(
        client,
        [
            {
                "role": "user",
                "content": (
                    "Hiring a software engineer; need technical knowledge and aptitude tests."
                ),
            },
            {"role": "assistant", "content": turn1["reply"]},
            {"role": "user", "content": "Actually, add personality tests as well."},
        ],
    )
    assert turn2["recommendations"]
    types = {item["test_type"] for item in turn2["recommendations"]}
    assert any("P" in code for code in types)


def test_recommendation_urls_are_catalog_only(client: TestClient, catalog_urls: set[str]) -> None:
    """Hard eval: every returned URL must exist in the scraped catalog."""
    body = _chat(
        client,
        [
            {
                "role": "user",
                "content": (
                    "Screening graduates for customer service roles with verbal reasoning "
                    "and situational judgement assessments."
                ),
            }
        ],
    )
    for item in body["recommendations"]:
        assert str(item["url"]).startswith("https://www.shl.com/")
        assert str(item["url"]) in catalog_urls


def test_turn_cap_sets_end_of_conversation(client: TestClient) -> None:
    """Hard eval: honor max 8 messages (user + assistant combined)."""
    messages: list[dict] = []
    for _ in range(4):
        messages.append({"role": "user", "content": "Need assessments for analysts"})
        messages.append({"role": "assistant", "content": "Tell me more about the role."})
    messages.append({"role": "user", "content": "Mid-level business analyst role"})

    body = _chat(client, messages)
    assert body["end_of_conversation"] is True
    assert body["recommendations"] == []


def test_multi_turn_clarify_then_recommend(client: TestClient) -> None:
    """Realistic flow: clarify first, then shortlist after enough context."""
    turn1 = _chat(client, [{"role": "user", "content": "I need an assessment"}])
    assert turn1["recommendations"] == []

    turn2 = _chat(
        client,
        [
            {"role": "user", "content": "I need an assessment"},
            {"role": "assistant", "content": turn1["reply"]},
            {
                "role": "user",
                "content": "Hiring a Java developer with stakeholder skills; personality and technical tests.",
            },
        ],
    )
    assert 1 <= len(turn2["recommendations"]) <= 10
    assert turn2["end_of_conversation"] is False

