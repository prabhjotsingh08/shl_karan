"""CLI utility to crawl the SHL catalog and export to CSV."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shl_recommender.crawler import CrawlerNetworkError, crawl_and_save


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
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Parse cached data_pages/page_*.html instead of fetching www.shl.com.",
    )
    parser.add_argument(
        "--from-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Rebuild CSV from an existing catalog JSON export (no network).",
    )
    parser.add_argument(
        "--fetch-details-offline",
        action="store_true",
        help="With --offline, still HTTP-fetch each assessment detail page (needs network).",
    )
    parser.add_argument(
        "--clear-page-cache",
        action="store_true",
        help="Delete data_pages/*.html before a live crawl (default: keep cache).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        csv_path, json_path = crawl_and_save(
            args.output,
            args.json_output,
            offline=args.offline,
            from_json=args.from_json,
            fetch_details_offline=args.fetch_details_offline,
            clear_page_cache=args.clear_page_cache,
        )
    except (CrawlerNetworkError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Catalog CSV written to {csv_path}")
    print(f"Catalog JSON written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
