"""Utility script to validate Gemini type extraction availability."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shl_recommender.config import get_settings
from src.shl_recommender.type_extraction import GeminiTypeExtractor


def main() -> int:
    settings = get_settings()
    has_key = bool(settings.gemini_api_key)
    print(f"Gemini key configured: {has_key}")

    if not has_key or not settings.gemini_api_key:
        print("No Gemini API key found in configuration.")
        return 1

    os.environ["GEMINI_API_KEY"] = settings.gemini_api_key

    try:
        extractor = GeminiTypeExtractor(api_key=settings.gemini_api_key)
    except Exception as exc:
        print(f"Failed to initialize Gemini type extractor: {exc!r}")
        return 1

    try:
        result = extractor.extract("We need an aptitude and personality assessment for graduates.")
        print("Gemini extractor output:", result)
        return 0
    except Exception as exc:  # pragma: no cover - diagnostic utility
        print(f"Gemini extractor call failed: {exc!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


