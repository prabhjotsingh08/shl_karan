"""Embedding utilities and vector store management."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import chromadb
import logfire
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import get_settings
from .data_models import AssessmentMetadata
from .logging_setup import configure_logging


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_to_root(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_catalog(path: str | None = None) -> List[AssessmentMetadata]:
    settings = get_settings()
    csv_path = Path(path) if path else Path(settings.data_csv_path)
    resolved_csv_path = _resolve_to_root(csv_path)
    df = pd.read_csv(resolved_csv_path)

    records: List[AssessmentMetadata] = []
    for _, row in df.iterrows():
        types = set(
            part.strip().upper()
            for part in str(row.get("assessment_types", "")).split(",")
            if part.strip()
        )
        job_levels = [
            part.strip()
            for part in str(row.get("job_levels", "")).split(",")
            if part.strip()
        ]
        languages = [
            part.strip()
            for part in str(row.get("languages", "")).split(",")
            if part.strip()
        ]

        metadata = AssessmentMetadata(
            entity_id=str(row.get("entity_id")),
            name=str(row.get("name")),
            url=str(row.get("url")),
            assessment_types=types,
            description=str(row.get("description")) if pd.notna(row.get("description")) else None,
            job_levels=job_levels,
            languages=languages,
            assessment_length=str(row.get("assessment_length")) if pd.notna(row.get("assessment_length")) else None,
            remote_testing=bool(row.get("remote_testing")) if not pd.isna(row.get("remote_testing")) else None,
            adaptive=bool(row.get("adaptive")) if not pd.isna(row.get("adaptive")) else None,
        )
        records.append(metadata)

    return records


class EmbeddingService:
    """Thin wrapper around SentenceTransformers for reuse."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        target_model = model_name or settings.embeddings_model_name
        logfire.info("Loading embedding model", model=target_model)
        self.model = SentenceTransformer(target_model)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)


def build_vector_store(reset: bool = False) -> None:
    """Build or refresh the persistent ChromaDB collection."""

    configure_logging("shl-embedder")
    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))

    if reset:
        try:
            client.delete_collection(settings.collection_name)
            logfire.info("Existing collection deleted", collection=settings.collection_name)
        except ValueError:
            pass

    collection = client.get_or_create_collection(name=settings.collection_name)

    records = load_catalog()
    embedder = EmbeddingService()
    documents = [record.combined_text() for record in records]
    embeddings = embedder.embed(documents)

    ids = [record.entity_id for record in records]
    def serialize_metadata(record: AssessmentMetadata) -> dict[str, str | bool | None]:
        def join(values: Iterable[str]) -> str | None:
            items = [value for value in values if value]
            return ", ".join(sorted(items)) if items else None

        return {
            "entity_id": record.entity_id,
            "name": record.name,
            "url": str(record.url),
            "assessment_types": join(record.assessment_types),
        }

    metadatas = [serialize_metadata(record) for record in records]

    logfire.info(
        "Upserting embeddings",
        count=len(records),
        collection=settings.collection_name,
        chroma_path=str(settings.chroma_path),
    )

    collection.upsert(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=documents)

    logfire.info("Vector store build complete", total=len(records))

