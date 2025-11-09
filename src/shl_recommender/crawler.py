"""Web crawler to extract SHL individual assessment metadata."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import logfire
import requests
from bs4 import BeautifulSoup

from .config import get_settings
from .data_models import AssessmentMetadata
from .logging_setup import configure_logging


BASE_CATALOG_URL = "https://www.shl.com/products/product-catalog/"
INDIVIDUAL_TABLE_HEADING = "Individual Test Solutions"
INDIVIDUAL_SOLUTIONS_TYPE = 1
REQUEST_TIMEOUT = 30
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SHLBot/1.0)"}
EXPECTED_TOTAL_PAGES = 32
REQUEST_DELAY_SECONDS = 0.1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogRow:
    entity_id: str
    name: str
    detail_url: str
    remote_testing: Optional[bool]
    adaptive: Optional[bool]
    assessment_types: List[str]


def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def _parse_boolean(cell: Optional[BeautifulSoup]) -> Optional[bool]:
    if not cell:
        return None
    if cell.find("span", class_="catalogue__circle -yes"):
        return True
    if cell.find("span", class_="catalogue__circle -no"):
        return False
    return None


def _parse_row(tr: BeautifulSoup, base_url: str) -> CatalogRow:
    entity_id = tr.get("data-entity-id") or tr.get("data-course-id") or ""
    cells = tr.find_all("td")
    if len(cells) < 4:
        raise ValueError("Unexpected table structure for catalog row")

    link = cells[0].find("a")
    if not link or not link.get("href"):
        raise ValueError("Catalog row link missing")

    name = link.get_text(strip=True)
    detail_url = urljoin(base_url, link["href"])

    remote = _parse_boolean(cells[1])
    adaptive = _parse_boolean(cells[2])
    type_spans = cells[3].find_all("span", class_="product-catalogue__key")
    types = [span.get_text(strip=True) for span in type_spans if span.get_text(strip=True)]

    return CatalogRow(
        entity_id=entity_id or name,
        name=name,
        detail_url=detail_url,
        remote_testing=remote,
        adaptive=adaptive,
        assessment_types=types,
    )


def _extract_detail_info(session: requests.Session, url: str) -> dict:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    def _text_after_heading(heading: str) -> Optional[str]:
        header_tag = soup.find("h4", string=lambda value: isinstance(value, str) and value.strip().lower() == heading.lower())
        if header_tag:
            value_tag = header_tag.find_next("p")
            if value_tag:
                return value_tag.get_text(" ", strip=True)
        return None

    description = _text_after_heading("Description")

    job_levels_raw = _text_after_heading("Job levels") or ""
    job_levels = [level.strip() for level in job_levels_raw.split(",") if level.strip()]

    languages_raw = _text_after_heading("Languages") or ""
    languages = [lang.strip() for lang in languages_raw.split(",") if lang.strip()]

    assessment_length = _text_after_heading("Assessment length")

    return {
        "description": description,
        "job_levels": job_levels,
        "languages": languages,
        "assessment_length": assessment_length,
    }


def _extract_total_pages(soup: BeautifulSoup) -> Optional[int]:
    pagination_container = soup.find("ul", class_="pagination__list") or soup.find("ul", class_="pagination")
    if not pagination_container:
        return None

    page_numbers: List[int] = []
    for link in pagination_container.find_all("a"):
        text = (link.get_text() or "").strip()
        if text.isdigit():
            try:
                page_numbers.append(int(text))
            except ValueError:
                continue

    return max(page_numbers) if page_numbers else None


def _parse_catalog_page(session: requests.Session, url: str) -> Tuple[List[CatalogRow], Optional[int], str]:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    page_html = response.text
    soup = BeautifulSoup(page_html, "html.parser")

    total_pages = _extract_total_pages(soup)

    table_candidates = soup.select("div.custom__table-wrapper table")

    individual_tables = []
    for table in table_candidates:
        heading_cell = table.find("th", class_="custom__table-heading__title")
        heading_text = heading_cell.get_text(" ", strip=True) if heading_cell else ""
        if heading_text.lower() == INDIVIDUAL_TABLE_HEADING.lower():
            individual_tables.append(table)

    if not individual_tables:
        return [], total_pages, page_html

    rows = []
    for table in individual_tables:
        for tr in table.find_all("tr"):
            if tr.find("th"):
                continue
            try:
                row = _parse_row(tr, BASE_CATALOG_URL)
            except ValueError:
                continue
            rows.append(row)

    return rows, total_pages, page_html


def _page_url(page_index: int, page_size: int = 12) -> str:
    if page_index <= 0:
        return f"{BASE_CATALOG_URL}?type={INDIVIDUAL_SOLUTIONS_TYPE}"
    start = page_index * page_size
    return f"{BASE_CATALOG_URL}?type={INDIVIDUAL_SOLUTIONS_TYPE}&start={start}"


def _prepare_pages_dir(pages_dir: Path) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)
    for existing in pages_dir.glob("*.html"):
        existing.unlink()


def _persist_page_html(page_html: str, pages_dir: Path, page_number: int) -> Path:
    destination = pages_dir / f"page_{page_number:02d}.html"
    destination.write_text(page_html, encoding="utf-8")
    return destination


def crawl_catalog() -> List[AssessmentMetadata]:
    """Crawl the SHL catalog and return metadata for individual assessments."""

    configure_logging("shl-crawler")
    settings = get_settings()
    session = _create_session()
    pages_dir = Path(settings.data_pages_dir)
    _prepare_pages_dir(pages_dir)

    collected: List[AssessmentMetadata] = []
    seen_ids: set[str] = set()

    page_index = 0
    total_pages: Optional[int] = None

    while page_index < EXPECTED_TOTAL_PAGES:
        page_number = page_index + 1
        url = _page_url(page_index)
        logfire.info("Parsing page", page_index=page_index, url=url)
        logger.info("Scraping page %d of %d from %s", page_number, EXPECTED_TOTAL_PAGES, url)

        rows, detected_total_pages, page_html = _parse_catalog_page(session, url)
        saved_path = _persist_page_html(page_html, pages_dir, page_number)
        logger.info("Saved page %d HTML to %s", page_number, saved_path)

        if detected_total_pages and detected_total_pages != total_pages:
            total_pages = detected_total_pages
            logfire.info("Detected catalog page count", total_pages=total_pages)
            if total_pages < EXPECTED_TOTAL_PAGES:
                logger.warning(
                    "Detected only %d pages on the site, expected %d",
                    total_pages,
                    EXPECTED_TOTAL_PAGES,
                )

        logfire.info("Parsed page", page_index=page_index, url=url, row_count=len(rows))

        if not rows:
            logfire.warning("No rows found for catalog page", page_index=page_index, url=url)
            logger.warning("No rows found on page %d; stopping crawl", page_number)
            break

        for row in rows:
            if row.entity_id in seen_ids:
                continue

            detail_info = _extract_detail_info(session, row.detail_url)

            metadata = AssessmentMetadata(
                entity_id=str(row.entity_id),
                name=row.name,
                url=row.detail_url,
                assessment_types=set(filter(None, row.assessment_types)),
                description=detail_info.get("description"),
                job_levels=detail_info.get("job_levels", []),
                languages=detail_info.get("languages", []),
                assessment_length=detail_info.get("assessment_length"),
                remote_testing=row.remote_testing,
                adaptive=row.adaptive,
            )

            collected.append(metadata)
            seen_ids.add(row.entity_id)
            logfire.debug(
                "Collected assessment",
                entity_id=row.entity_id,
                name=row.name,
                assessment_types=list(row.assessment_types),
            )

            time.sleep(REQUEST_DELAY_SECONDS)

        page_index += 1
        logger.info("Completed page %d with %d assessments", page_number, len(rows))
        time.sleep(REQUEST_DELAY_SECONDS)

    logfire.info("Crawl complete", total=len(collected), pages_visited=page_index)
    if page_index < EXPECTED_TOTAL_PAGES:
        logger.warning(
            "Crawled %d pages, fewer than the expected %d",
            page_index,
            EXPECTED_TOTAL_PAGES,
        )
    return collected


def write_catalog_to_csv(records: Iterable[AssessmentMetadata], output_path: Optional[str] = None) -> str:
    """Persist catalog records to CSV and return the resolved path."""

    settings = get_settings()
    destination = output_path or str(settings.data_csv_path)
    fieldnames = [
        "entity_id",
        "name",
        "url",
        "assessment_types",
        "description",
        "job_levels",
        "languages",
        "assessment_length",
        "remote_testing",
        "adaptive",
    ]

    with open(destination, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "entity_id": record.entity_id,
                    "name": record.name,
                    "url": str(record.url),
                    "assessment_types": ",".join(sorted(record.assessment_types)),
                    "description": record.description or "",
                    "job_levels": ",".join(record.job_levels),
                    "languages": ",".join(record.languages),
                    "assessment_length": record.assessment_length or "",
                    "remote_testing": record.remote_testing,
                    "adaptive": record.adaptive,
                }
            )

    logfire.info("Catalog written", path=destination)
    return destination


def write_catalog_to_json(records: Iterable[AssessmentMetadata], output_path: Optional[str] = None) -> str:
    """Persist catalog records to JSON and return the resolved path."""

    settings = get_settings()
    destination = output_path or str(settings.data_json_path)
    payload = [record.model_dump(mode="json") for record in records]

    with open(destination, "w", encoding="utf-8") as jsonfile:
        json.dump(payload, jsonfile, ensure_ascii=False, indent=2)

    logfire.info("Catalog JSON written", path=destination)
    return destination


def crawl_and_save(
    output_path: Optional[str] = None,
    json_output_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Utility helper combining crawl and CSV persistence."""

    records = crawl_catalog()
    csv_path = write_catalog_to_csv(records, output_path)
    json_path = write_catalog_to_json(records, json_output_path)
    return csv_path, json_path

