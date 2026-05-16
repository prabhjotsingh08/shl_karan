"""Recommendation engine orchestrating retrieval and ranking."""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
from typing import Any, Dict, List, Set
import os

import logfire
from .config import get_settings
from .data_models import ASSESSMENT_TYPE_LABELS, AssessmentMetadata, RecommendationItem
from .embedding import (
    EmbeddingService,
    build_vector_store,
    get_chroma_client,
    get_or_create_assessment_collection,
    load_catalog,
)
from .logging_setup import configure_logging
from .type_extraction import GeminiExtractionError, GeminiTypeExtractor


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


@dataclass
class RecommendationResult:
    recommendations: List[RecommendationItem]
    extracted_types: List[str]


class RecommendationEngine:
    """Coordinate vector retrieval and ranking."""

    def __init__(self) -> None:
        configure_logging("shl-recommender")
        self.settings = get_settings()
        self.embedder = EmbeddingService()
        self.client = get_chroma_client()
        self._collection_refresh_lock = threading.Lock()
        self.collection = get_or_create_assessment_collection(
            self.client,
            self.settings.collection_name,
        )
        self.catalog_index: Dict[str, AssessmentMetadata] = {}
        try:
            self.catalog_index = {record.entity_id: record for record in load_catalog()}
        except Exception:
            logfire.warn("Failed to load full catalog metadata; responses will be limited", exc_info=True)
        
        self._type_keyword_map = self._build_type_keyword_map()
        self.type_extractor: GeminiTypeExtractor | None = None
        self._gemini_disabled = False
        if self.settings.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.settings.gemini_api_key
            try:
                self.type_extractor = GeminiTypeExtractor(api_key=self.settings.gemini_api_key)
            except Exception as exc:
                logfire.warn(
                    "Gemini type extractor unavailable; falling back to heuristic extraction",
                    error=str(exc),
                    exc_info=True,
                )
                self.type_extractor = None

    def _rebuild_collection(self, reason: str | None = None) -> None:
        with self._collection_refresh_lock:
            logfire.warn(
                "Rebuilding Chroma collection due to incompatibility",
                collection=self.settings.collection_name,
                reason=reason,
            )
            try:
                build_vector_store(reset=True)
            except Exception as exc:
                logfire.error(
                    "Failed to rebuild Chroma collection",
                    collection=self.settings.collection_name,
                    reason=reason,
                    error=str(exc),
                    exc_info=True,
                )
                raise

            self.collection = get_or_create_assessment_collection(
                self.client,
                self.settings.collection_name,
            )

    @staticmethod
    def _is_dimensionality_error(error: AttributeError) -> bool:
        message = str(error)
        return "dimensionality" in message.lower() and "attribute" in message.lower()

    def _query_collection(self, query_embedding: List[float]) -> Dict[str, Any]:
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": self.settings.candidate_pool_size,
            "include": ["metadatas", "documents", "distances"],
        }

        try:
            return self.collection.query(**query_params)
        except AttributeError as exc:
            if not self._is_dimensionality_error(exc):
                raise

            logfire.error(
                "Chroma collection query failed due to incompatible persisted index; attempting rebuild",
                collection=self.settings.collection_name,
                error=str(exc),
            )
            self._rebuild_collection(reason="missing dimensionality attribute")

            try:
                return self.collection.query(**query_params)
            except AttributeError as second_exc:
                if self._is_dimensionality_error(second_exc):
                    raise RuntimeError(
                        "Chroma vector store remains incompatible after rebuild. "
                        "Please rerun the build script manually."
                    ) from second_exc
                raise

    _TYPE_FOCUS_PHRASES: Dict[str, str] = {
        "A": "cognitive aptitude and ability assessment",
        "B": "situational judgement and biodata assessment",
        "C": "competency-based behavioural assessment",
        "D": "development and 360 feedback assessment",
        "E": "assessment centre exercise",
        "K": "knowledge and skills technical assessment",
        "P": "personality and behaviour assessment",
        "S": "job simulation assessment",
    }

    def _retrieve_candidates(self, query: str) -> List[Candidate]:
        query_embedding = self.embedder.embed([query])[0].tolist()
        results = self._query_collection(query_embedding)

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

    def _expand_candidates_for_types(
        self,
        query: str,
        candidates: List[Candidate],
        extracted_types: List[str],
    ) -> List[Candidate]:
        """Backfill retrieval when vector search omits requested assessment types."""
        if not extracted_types:
            return candidates

        present_types: Set[str] = set()
        for candidate in candidates:
            present_types.update(value.upper() for value in candidate.assessment_types)

        merged = list(candidates)
        seen_ids = {candidate.id for candidate in merged}

        for type_code in extracted_types:
            upper = type_code.upper()
            if upper in present_types:
                continue
            focus = self._TYPE_FOCUS_PHRASES.get(upper, ASSESSMENT_TYPE_LABELS.get(upper, upper))
            supplemental_query = f"{query}\n{focus}"
            for candidate in self._retrieve_candidates(supplemental_query):
                if upper not in {value.upper() for value in candidate.assessment_types}:
                    continue
                if candidate.id in seen_ids:
                    continue
                merged.append(candidate)
                seen_ids.add(candidate.id)
                present_types.add(upper)
                break

        return merged

    def _extract_types_from_query(self, query: str) -> List[str]:
        """Extract assessment types (A, B, C, D, E, K, P, S) from query using Gemini if available."""
        valid_types = {"A", "B", "C", "D", "E", "K", "P", "S"}

        if self.type_extractor and not self._gemini_disabled:
            try:
                extracted = self.type_extractor.extract(query)
                ordered_unique: List[str] = []
                for type_code in extracted:
                    normalized = type_code.strip().upper()
                    if normalized in valid_types and normalized not in ordered_unique:
                        ordered_unique.append(normalized)

                limited_types = self._limit_type_count(query, ordered_unique)
                logfire.info("Extracted types from query", query=query, types=limited_types)
                return limited_types
            except GeminiExtractionError as exc:
                logfire.info(
                    "Disabling Gemini type extraction after unusable response; falling back to heuristics",
                    error=str(exc),
                    finish_reasons=exc.finish_reasons or None,
                )
                self._gemini_disabled = True
            except Exception as exc:
                logfire.warn(
                    "Gemini type extraction failed; falling back to heuristic extraction",
                    error=str(exc),
                    query=query,
                    exc_info=True,
                )

        return self._heuristic_extract_types(query)

    @staticmethod
    def _keyword_matches(query_lower: str, keyword: str) -> bool:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        return re.search(pattern, query_lower) is not None

    @staticmethod
    def _limit_type_count(query: str, codes: List[str]) -> List[str]:
        if len(codes) <= 2:
            return codes

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
        refinement_cues = (
            "add ",
            "also ",
            "actually",
            "as well",
            "include ",
            "plus ",
        )
        allow_three = any(token in query_lower for token in multi_type_cues) or any(
            cue in query_lower for cue in refinement_cues
        )
        return codes[:3] if allow_three else codes[:2]

    def _heuristic_extract_types(self, query: str) -> List[str]:
        query_lower = query.lower()
        scores: Dict[str, int] = {}
        for type_code, keywords in self._type_keyword_map.items():
            score = 0
            for keyword in keywords:
                if self._keyword_matches(query_lower, keyword):
                    score += max(1, len(keyword.split()))
            if score:
                scores[type_code] = score

        if not scores:
            return []

        ordered_codes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_codes = [code for code, _ in ordered_codes[:3]]
        limited = self._limit_type_count(query, top_codes)
        logfire.debug(
            "Heuristic type extraction",
            query=query,
            matches=limited,
            keyword_scores=scores,
        )
        return limited

    def _build_type_keyword_map(self) -> Dict[str, Set[str]]:
        synonyms: Dict[str, Set[str]] = {
            "A": {
                "ability",
                "abilities",
                "aptitude",
                "aptitudes",
                "cognitive",
                "numerical",
                "verbal reasoning",
                "logical reasoning",
                "reasoning",
                "quantitative",
            },
            "B": {
                "situational judgement",
                "situational judgment",
                "situational judgement test",
                "situational judgment test",
                "sjt",
                "scenario based",
                "biodata",
            },
            "C": {
                "competency",
                "competencies",
                "behavioural competency",
                "competency assessment",
            },
            "D": {
                "development",
                "360",
                "multi rater",
                "feedback program",
                "coaching",
            },
            "E": {
                "assessment centre",
                "assessment center",
                "exercise",
                "in tray",
                "group exercise",
                "role play",
            },
            "K": {
                "knowledge",
                "skills",
                "technical test",
                "compliance",
                "certification",
            },
            "P": {
                "personality",
                "behaviour",
                "behavior",
                "traits",
                "motivation",
                "values",
            },
            "S": {
                "simulation",
                "job simulation",
                "virtual job tryout",
                "realistic preview",
            },
        }

        for code, label in ASSESSMENT_TYPE_LABELS.items():
            normalized_label = label.lower()
            synonyms.setdefault(code, set()).add(normalized_label)
            synonyms[code].add(normalized_label.replace("&", "and"))

        return {code: {keyword.strip() for keyword in keywords if keyword.strip()} for code, keywords in synonyms.items()}

    def _rank_candidates(self, candidates: List[Candidate], extracted_types: List[str]) -> List[Candidate]:
        """Rank candidates by type match and embedding similarity."""
        type_set = {code.upper() for code in extracted_types}
        if not type_set:
            # If no types extracted, rank by embedding similarity only
            return sorted(candidates, key=lambda c: c.embedding_similarity, reverse=True)
        
        # Rank candidates: prioritize those matching extracted types, then by similarity
        def rank_key(candidate: Candidate) -> tuple:
            # Check if candidate has any matching types
            candidate_types = {t.upper() for t in candidate.assessment_types}
            has_match = bool(candidate_types & type_set)
            # Return (has_match, similarity) - True sorts before False, so matches come first
            return (not has_match, -candidate.embedding_similarity)
        
        return sorted(candidates, key=rank_key)

    def _balance_by_extracted_types(
        self,
        ranked_candidates: List[Candidate],
        extracted_types: List[str],
        desired_count: int,
    ) -> List[Candidate]:
        """Ensure each requested assessment type appears when candidates exist."""
        if not extracted_types or desired_count <= 0:
            return ranked_candidates[:desired_count]

        selected: List[Candidate] = []
        seen_ids: set[str] = set()

        for type_code in extracted_types:
            upper = type_code.upper()
            for candidate in ranked_candidates:
                if candidate.id in seen_ids:
                    continue
                candidate_types = {value.upper() for value in candidate.assessment_types}
                if upper in candidate_types:
                    selected.append(candidate)
                    seen_ids.add(candidate.id)
                    break

        for candidate in ranked_candidates:
            if len(selected) >= desired_count:
                break
            if candidate.id not in seen_ids:
                selected.append(candidate)
                seen_ids.add(candidate.id)

        return selected[:desired_count]

    def _determine_result_count(self, candidates: List[Candidate], extracted_types: List[str]) -> int:
        """Determine optimal number of results (5-10) based on match quality."""
        if not candidates:
            return 5
        
        min_results = 5
        max_results = 10
        
        # Count candidates matching extracted types
        type_match_count = 0
        type_set = {code.upper() for code in extracted_types}
        if type_set:
            for candidate in candidates:
                candidate_types = {t.upper() for t in candidate.assessment_types}
                if candidate_types & type_set:
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

    def recommend(
        self,
        query: str,
        *,
        min_results: int | None = None,
        max_results: int | None = None,
    ) -> RecommendationResult:
        # Step 1: Extract types from query (Gemini + heuristics)
        extracted_types = self._extract_types_from_query(query)
        
        # Step 2: Retrieve candidates using vector search
        candidates = self._retrieve_candidates(query)
        candidates = self._expand_candidates_for_types(query, candidates, extracted_types)
        if not candidates:
            return RecommendationResult(recommendations=[], extracted_types=extracted_types)

        # Step 3: Rank candidates (type matches first, then similarity backfill)
        ranked_candidates = self._rank_candidates(candidates, extracted_types)
        
        # Step 4: Determine optimal number of results based on match quality
        optimal_count = self._determine_result_count(ranked_candidates, extracted_types)
        lower_bound = min_results if min_results is not None else 5
        upper_bound = max_results if max_results is not None else min(self.settings.recommendation_limit, 10)
        upper_bound = max(1, min(upper_bound, 10))
        lower_bound = max(1, min(lower_bound, upper_bound))
        desired_count = max(lower_bound, min(optimal_count, upper_bound))
        desired_count = min(desired_count, len(ranked_candidates))
        ranked_candidates = self._balance_by_extracted_types(
            ranked_candidates,
            extracted_types,
            desired_count,
        )

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

        return RecommendationResult(recommendations=recommendations, extracted_types=extracted_types)

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


