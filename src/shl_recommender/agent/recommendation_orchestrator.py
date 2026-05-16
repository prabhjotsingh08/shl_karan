"""Orchestrate retrieval and ranking for conversational recommendations."""

from __future__ import annotations

from typing import Dict, List, Sequence

import logfire

from ..data_models import AssessmentMetadata, ChatRecommendationItem, RecommendationItem
from ..recommender import RecommendationEngine


def build_search_query(user_messages: Sequence[str], *, latest: str | None = None) -> str:
    """Merge conversation user text into a single retrieval query."""
    parts = [part.strip() for part in user_messages if part and part.strip()]
    if not parts:
        return latest or ""
    if len(parts) == 1:
        return parts[0]
    history = " ".join(parts[:-1])
    current = parts[-1]
    return f"{history}\nLatest request: {current}"


def validate_catalog_recommendations(
    items: Sequence[RecommendationItem],
    catalog_index: Dict[str, AssessmentMetadata],
) -> List[RecommendationItem]:
    """Ensure every recommendation exists in the SHL catalog."""
    allowed_urls = {str(record.url) for record in catalog_index.values()}
    validated: List[RecommendationItem] = []

    for item in items:
        url = str(item.url)
        if url not in allowed_urls:
            logfire.warn("Dropping recommendation not in catalog", name=item.name, url=url)
            continue
        validated.append(item)

    return validated


def enrich_from_catalog(
    items: Sequence[RecommendationItem],
    catalog_index: Dict[str, AssessmentMetadata],
) -> List[RecommendationItem]:
    """Replace items with full catalog metadata when available."""
    by_url = {str(record.url): record for record in catalog_index.values()}
    enriched: List[RecommendationItem] = []
    for item in items:
        record = by_url.get(str(item.url))
        enriched.append(record.to_recommendation_item() if record else item)
    return enriched


class RecommendationOrchestrator:
    """Thin wrapper around RecommendationEngine for chat flows."""

    def __init__(self, engine: RecommendationEngine) -> None:
        self.engine = engine

    def recommend(
        self,
        query: str,
        *,
        min_results: int = 1,
        max_results: int = 10,
    ) -> List[RecommendationItem]:
        result = self.engine.recommend(
            query,
            min_results=min_results,
            max_results=max_results,
        )
        items = enrich_from_catalog(result.recommendations, self.engine.catalog_index)
        return validate_catalog_recommendations(items, self.engine.catalog_index)

    def recommend_for_messages(
        self,
        user_messages: Sequence[str],
        *,
        min_results: int = 1,
        max_results: int = 10,
    ) -> List[RecommendationItem]:
        query = build_search_query(user_messages)
        return self.recommend(query, min_results=min_results, max_results=max_results)

    def recommend_chat_items(
        self,
        user_messages: Sequence[str],
        *,
        min_results: int = 1,
        max_results: int = 10,
    ) -> List[ChatRecommendationItem]:
        items = self.recommend_for_messages(
            user_messages,
            min_results=min_results,
            max_results=max_results,
        )
        return to_chat_recommendation_items(items, self.engine.catalog_index)


def to_chat_recommendation_items(
    items: Sequence[RecommendationItem],
    catalog_index: Dict[str, AssessmentMetadata],
) -> List[ChatRecommendationItem]:
    """Map internal items to assignment /chat recommendation objects."""
    by_url = {str(record.url): record for record in catalog_index.values()}
    chat_items: List[ChatRecommendationItem] = []
    for item in items:
        record = by_url.get(str(item.url))
        if record:
            chat_items.append(record.to_chat_recommendation_item())
        else:
            chat_items.append(
                ChatRecommendationItem(
                    name=item.name,
                    url=item.url,
                    test_type="",
                )
            )
    return chat_items
