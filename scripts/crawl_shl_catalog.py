"""CLI utility to crawl the SHL catalog and export to CSV."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shl_recommender.crawler import crawl_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl the SHL catalog and produce a CSV dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the generated CSV (defaults to settings.data_csv_path).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional output path for the generated JSON (defaults to settings.data_json_path).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path, json_path = crawl_and_save(args.output, args.json_output)
    print(f"Catalog CSV written to {csv_path}")
    print(f"Catalog JSON written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

