"""Embedding utilities and vector store management."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import chromadb
import logfire
import numpy as np
import pandas as pd
import huggingface_hub
from urllib.parse import urlparse

if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download

    def _cached_download(
        *,
        url: str,
        cache_dir: str | None = None,
        force_filename: str | None = None,
        library_name: str | None = None,
        library_version: str | None = None,
        user_agent: dict | str | None = None,
        use_auth_token: str | bool | None = None,
        legacy_cache_layout: bool | None = None,
        etag_timeout: float | None = None,
        local_files_only: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        headers: dict | None = None,
        **kwargs,
    ) -> str:
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        try:
            resolve_idx = parts.index("resolve")
        except ValueError as exc:
            raise ValueError(f"Unexpected Hugging Face URL: {url}") from exc

        repo_id = "/".join(parts[:resolve_idx])
        revision = parts[resolve_idx + 1] if len(parts) > resolve_idx + 1 else None
        filename = "/".join(parts[resolve_idx + 2 :])

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=None,
            force_filename=force_filename,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            token=use_auth_token,
            etag_timeout=etag_timeout or 10,
            local_files_only=local_files_only,
            resume_download=resume_download,
            proxies=proxies,
            headers=headers,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )

    huggingface_hub.cached_download = _cached_download  # type: ignore[attr-defined]

from sentence_transformers import SentenceTransformer

from .config import get_settings
from .data_models import AssessmentMetadata
from .logging_setup import configure_logging
from .paths import resolve_project_path


COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _is_missing_collection_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "does not exist" in message or "not found" in message


def resolved_chroma_path() -> Path:
    settings = get_settings()
    return resolve_project_path(Path(settings.chroma_path))


def resolved_catalog_csv_path() -> Path:
    settings = get_settings()
    return resolve_project_path(Path(settings.data_csv_path))


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(resolved_chroma_path()))


def get_or_create_assessment_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
):
    try:
        return client.get_collection(collection_name)
    except ValueError as exc:
        if not _is_missing_collection_error(exc):
            raise
        logfire.info(
            "Chroma collection missing; creating new collection",
            collection=collection_name,
        )
        return client.get_or_create_collection(
            name=collection_name,
            metadata=COLLECTION_METADATA,
        )


def delete_assessment_collection_if_exists(
    client: chromadb.PersistentClient,
    collection_name: str,
) -> None:
    try:
        client.delete_collection(collection_name)
        logfire.info("Existing collection deleted", collection=collection_name)
    except ValueError as exc:
        if not _is_missing_collection_error(exc):
            raise


def load_catalog(path: str | None = None) -> List[AssessmentMetadata]:
    settings = get_settings()
    csv_path = Path(path) if path else Path(settings.data_csv_path)
    resolved_csv_path = resolve_project_path(csv_path)
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
    client = get_chroma_client()

    if reset:
        delete_assessment_collection_if_exists(client, settings.collection_name)

    collection = get_or_create_assessment_collection(client, settings.collection_name)

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
        chroma_path=str(resolved_chroma_path()),
    )

    collection.upsert(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=documents)

    logfire.info("Vector store build complete", total=len(records))

