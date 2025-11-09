"""Recommendation engine orchestrating retrieval and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import chromadb
import logfire

from .config import get_settings
from .data_models import (
    ASSESSMENT_TYPE_LABELS,
    AssessmentMetadata,
    RecommendationItem,
)
from .embedding import EmbeddingService, load_catalog
from .logging_setup import configure_logging


@dataclass
class Candidate:
    id: str
    name: str
    url: str
    assessment_types: List[str]
    document: str
    embedding_similarity: float = 0.0

    def short_description(self) -> str | None:
        parts = self.document.split("\n", 2)
        if len(parts) >= 2:
            value = parts[1].strip()
            return value or None
        return None


class RecommendationEngine:
    """Coordinate vector retrieval and ranking."""

    def __init__(self) -> None:
        configure_logging("shl-recommender")
        self.settings = get_settings()
        self.embedder = EmbeddingService()
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self.collection = self.client.get_collection(self.settings.collection_name)
        self.catalog_index: Dict[str, AssessmentMetadata] = {}
        try:
            self.catalog_index = {record.entity_id: record for record in load_catalog()}
        except Exception:
            logfire.warning("Failed to load full catalog metadata; responses will be limited", exc_info=True)

    def _retrieve_candidates(self, query: str) -> List[Candidate]:
        query_embedding = self.embedder.embed([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.settings.candidate_pool_size,
            include=["metadatas", "documents", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        candidates: List[Candidate] = []
        for idx, metadata, document, distance in zip(ids, metadatas, documents, distances):
            if not metadata:
                continue
            assessment_types = self._parse_assessment_types(metadata.get("assessment_types"))
            # Convert distance to similarity (ChromaDB uses cosine distance, similarity = 1 - distance)
            similarity = 1.0 - distance if distance is not None else 0.0
            candidate = Candidate(
                id=idx,
                name=metadata.get("name", ""),
                url=metadata.get("url", ""),
                assessment_types=assessment_types,
                document=document or "",
                embedding_similarity=similarity,
            )
            candidates.append(candidate)

        logfire.info("Retrieved candidates", count=len(candidates))
        return candidates

    def _rank_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Rank candidates by embedding similarity (higher is better)."""
        return sorted(candidates, key=lambda c: c.embedding_similarity, reverse=True)

    @staticmethod
    def _parse_assessment_types(raw: Any) -> List[str]:
        if isinstance(raw, list):
            return [str(item).strip().upper() for item in raw if str(item).strip()]
        if isinstance(raw, str):
            return [part.strip().upper() for part in raw.split(",") if part.strip()]
        return []

    @staticmethod
    def _apply_balance(require_mix: bool, items: List[Candidate], min_per_type: int) -> List[Candidate]:
        if not require_mix:
            return items

        k_items = [item for item in items if any(t.upper() == "K" for t in item.assessment_types)]
        p_items = [item for item in items if any(t.upper() == "P" for t in item.assessment_types)]

        if len(k_items) < min_per_type or len(p_items) < min_per_type:
            return items

        selected: List[Candidate] = []

        def append_if_absent(target: Candidate) -> None:
            if target not in selected:
                selected.append(target)

        for index in range(min_per_type):
            append_if_absent(k_items[index])
            append_if_absent(p_items[index])

        for candidate in items:
            append_if_absent(candidate)
            if len(selected) >= len(items):
                break

        return selected[: len(items)]

    def recommend(self, query: str) -> List[RecommendationItem]:
        candidates = self._retrieve_candidates(query)
        if not candidates:
            return []

        # Rank candidates by embedding similarity
        ranked_candidates = self._rank_candidates(candidates)
        
        # Apply balanced mixing (always check for both K and P types)
        # Since we no longer have query_focus detection, we'll apply balance heuristically
        # by checking if we have enough candidates of both types
        ranked_candidates = self._apply_balance(
            require_mix=False,  # Disable automatic mix requirement without LLM
            items=ranked_candidates,
            min_per_type=self.settings.min_category_mix
        )
        ranked_candidates = ranked_candidates[: self.settings.recommendation_limit]

        recommendations = [self._build_recommendation_item(candidate) for candidate in ranked_candidates]

        # Log recommended assessments with their similarity scores
        logfire.info(
            "Recommended assessments",
            query=query,
            count=len(recommendations),
            assessments=[
                {
                    "name": candidate.name,
                    "id": candidate.id,
                    "url": candidate.url,
                    "embedding_similarity": round(candidate.embedding_similarity, 4),
                }
                for candidate in ranked_candidates
            ],
        )

        if len(recommendations) >= 5:
            return recommendations[: self.settings.recommendation_limit]

        return recommendations

    def _build_recommendation_item(self, candidate: Candidate) -> RecommendationItem:
        metadata = self.catalog_index.get(candidate.id)
        if metadata:
            return metadata.to_recommendation_item()

        return RecommendationItem(
            name=candidate.name,
            url=candidate.url,
            description=candidate.short_description(),
            test_type=[ASSESSMENT_TYPE_LABELS.get(code.upper(), code.upper()) for code in candidate.assessment_types],
        )


