"""Unit tests for recommendation helpers."""

from __future__ import annotations

import unittest

from src.shl_recommender.recommender import RecommendationEngine


class TestRecommendationHelpers(unittest.TestCase):
    def test_parse_assessment_types_from_string(self) -> None:
        result = RecommendationEngine._parse_assessment_types("a, K, p")
        self.assertEqual(result, ["A", "K", "P"])

    def test_parse_assessment_types_from_list(self) -> None:
        result = RecommendationEngine._parse_assessment_types(["c", "e"])
        self.assertEqual(result, ["C", "E"])

    def test_limit_type_count_defaults_to_two(self) -> None:
        codes = ["A", "B", "C", "D"]
        limited = RecommendationEngine._limit_type_count("software engineer role", codes)
        self.assertEqual(limited, ["A", "B"])

    def test_limit_type_count_allows_three_with_cue(self) -> None:
        codes = ["A", "B", "C", "D"]
        limited = RecommendationEngine._limit_type_count(
            "aptitude and personality assessments",
            codes,
        )
        self.assertEqual(limited, ["A", "B", "C"])

    def test_keyword_matches_word_boundary(self) -> None:
        self.assertFalse(
            RecommendationEngine._keyword_matches(
                "candidates with disabilities",
                "abilities",
            )
        )
        self.assertTrue(
            RecommendationEngine._keyword_matches(
                "cognitive abilities assessment",
                "abilities",
            )
        )

    def test_heuristic_extract_types(self) -> None:
        engine = RecommendationEngine.__new__(RecommendationEngine)
        engine._type_keyword_map = RecommendationEngine._build_type_keyword_map(engine)
        types = RecommendationEngine._heuristic_extract_types(
            engine,
            "We need personality and aptitude tests for graduates",
        )
        self.assertIn("P", types)
        self.assertIn("A", types)


if __name__ == "__main__":
    unittest.main()
