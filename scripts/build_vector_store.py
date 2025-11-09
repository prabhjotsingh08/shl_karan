"""CLI entry point to rebuild the Chroma vector store."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shl_recommender.embedding import build_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Chroma vector store for SHL assessments.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before inserting records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_vector_store(reset=args.reset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

