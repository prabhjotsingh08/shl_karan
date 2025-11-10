"""Recommendation engine orchestrating retrieval and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set
import os

import chromadb
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .config import get_settings
from .data_models import (
    ASSESSMENT_TYPE_LABELS,
    AssessmentMetadata,
    RecommendationItem,
)
from .embedding import EmbeddingService, load_catalog
from .logging_setup import configure_logging


class ExtractedTypes(BaseModel):
    """Assessment types extracted from user query."""
    types: List[str] = Field(
        default_factory=list,
        description="List of assessment type codes (A, B, C, D, E, K, P, S) relevant to the query"
    )


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
        
        # Initialize Pydantic AI agent for type extraction
        self.type_extraction_agent = None
        if self.settings.gemini_api_key:
            # Set API key in environment for Pydantic AI
            os.environ['GEMINI_API_KEY'] = self.settings.gemini_api_key
            
            self.type_extraction_agent = Agent(
                'gemini-2.5-flash',
                output_type=ExtractedTypes,
                retries=2,
                system_prompt="""You are an expert at analyzing user queries and identifying the most relevant SHL assessment types.

The valid assessment types are:
- A: Ability & Aptitude
- B: Biodata & Situational Judgement
- C: Competencies
- D: Development & 360
- E: Assessment Exercises
- K: Knowledge & Skills
- P: Personality & Behaviour
- S: Simulations

Return one or two single-letter codes (A, B, C, D, E, K, P, S).
Only provide a third code when the query explicitly references three distinct assessment areas and the relevance is unmistakable.
If you are unsure, provide the single most relevant assessment type."""
            )

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

    def _extract_types_from_query(self, query: str) -> Set[str]:
        """Extract assessment types (A, B, C, D, E, K, P, S) from query using Pydantic AI agent."""
        valid_types = {"A", "B", "C", "D", "E", "K", "P", "S"}
        
        if not self.type_extraction_agent:
            logfire.debug("Type extraction agent not configured; skipping type extraction")
            return set()
        
        try:
            result = self.type_extraction_agent.run_sync(query)
            ordered_unique: List[str] = []
            for type_code in result.output.types:
                normalized = type_code.strip().upper()
                if normalized in valid_types and normalized not in ordered_unique:
                    ordered_unique.append(normalized)

            max_types = 3
            limited_types = ordered_unique[:max_types]

            if len(limited_types) > 2:
                query_lower = query.lower()
                multi_type_cues = (
                    " and ",
                    " & ",
                    " / ",
                    ", ",
                    " plus ",
                    " combination",
                    " combinations",
                    " mixed ",
                    " mix ",
                    " mixture",
                    " range of",
                    " multiple ",
                    " variety",
                    " diverse ",
                )
                allow_three = any(token in query_lower for token in multi_type_cues)
                if not allow_three:
                    limited_types = limited_types[:2]

            extracted_types = set(limited_types)
            logfire.info("Extracted types from query", query=query, types=limited_types)
            return extracted_types
        except Exception as e:
            logfire.warning(f"Error extracting types with agent: {e}", query=query, exc_info=True)
            return set()

    def _rank_candidates(self, candidates: List[Candidate], extracted_types: Set[str]) -> List[Candidate]:
        """Rank candidates by type match and embedding similarity."""
        if not extracted_types:
            # If no types extracted, rank by embedding similarity only
            return sorted(candidates, key=lambda c: c.embedding_similarity, reverse=True)
        
        # Rank candidates: prioritize those matching extracted types, then by similarity
        def rank_key(candidate: Candidate) -> tuple:
            # Check if candidate has any matching types
            candidate_types = {t.upper() for t in candidate.assessment_types}
            has_match = bool(candidate_types & extracted_types)
            # Return (has_match, similarity) - True sorts before False, so matches come first
            return (not has_match, -candidate.embedding_similarity)
        
        return sorted(candidates, key=rank_key)

    def _determine_result_count(self, candidates: List[Candidate], extracted_types: Set[str]) -> int:
        """Determine optimal number of results (5-10) based on match quality."""
        if not candidates:
            return 5
        
        min_results = 5
        max_results = 10
        
        # Count candidates matching extracted types
        type_match_count = 0
        if extracted_types:
            for candidate in candidates:
                candidate_types = {t.upper() for t in candidate.assessment_types}
                if candidate_types & extracted_types:
                    type_match_count += 1
        
        # Simple logic: return more results if we have good type matches
        if type_match_count >= 10:
            return max_results
        elif type_match_count >= 7:
            return 8
        elif type_match_count >= 5:
            return 7
        else:
            return min_results

    @staticmethod
    def _parse_assessment_types(raw: Any) -> List[str]:
        if isinstance(raw, list):
            return [str(item).strip().upper() for item in raw if str(item).strip()]
        if isinstance(raw, str):
            return [part.strip().upper() for part in raw.split(",") if part.strip()]
        return []

    def recommend(self, query: str) -> List[RecommendationItem]:
        # Step 1: Extract types from query using Pydantic AI agent
        extracted_types = self._extract_types_from_query(query)
        
        # Step 2: Retrieve candidates using vector search
        candidates = self._retrieve_candidates(query)
        if not candidates:
            return []

        # Step 3: Rank candidates by type match and embedding similarity
        ranked_candidates = self._rank_candidates(candidates, extracted_types)
        
        # Step 4: Determine optimal number of results (5-10) based on match quality
        optimal_count = self._determine_result_count(ranked_candidates, extracted_types)
        min_results = 5
        max_results = min(self.settings.recommendation_limit, 10)
        desired_count = max(min_results, min(optimal_count, max_results))
        desired_count = min(desired_count, len(ranked_candidates))
        ranked_candidates = ranked_candidates[:desired_count]

        # Step 5: Build recommendation items
        recommendations = [self._build_recommendation_item(candidate) for candidate in ranked_candidates]

        # Log recommended assessments
        logfire.info(
            "Recommended assessments",
            query=query,
            extracted_types=list(extracted_types) if extracted_types else None,
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


