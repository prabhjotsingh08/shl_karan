"""Generate assessment recommendations for a batch of queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shl_recommender.recommender import RecommendationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHL recommendations from an input dataset.")
    parser.add_argument("--input", required=True, help="Path to a CSV or Excel file containing queries.")
    parser.add_argument("--column", default="query", help="Column name that holds the job description text.")
    parser.add_argument(
        "--output",
        default=str(Path("data") / "predictions.csv"),
        help="Destination CSV path for the predictions output.",
    )
    return parser.parse_args()


def load_queries(path: str, column: str) -> Iterable[str]:
    extension = Path(path).suffix.lower()
    if extension == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in input file.")

    return df[column].fillna("").astype(str)


def main() -> int:
    args = parse_args()
    engine = RecommendationEngine()

    queries = list(load_queries(args.input, args.column))
    results = []

    for query in queries:
        recommendations = engine.recommend(query)
        results.append(
            {
                args.column: query,
                "assessments": json.dumps([item.model_dump() for item in recommendations]),
            }
        )

    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

