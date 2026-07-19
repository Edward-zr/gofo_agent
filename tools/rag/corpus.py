"""Shared corpus utilities for Chroma and BM25 indexing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import config

BM25_CORPUS_FILENAME = "bm25_corpus.json"
CORPUS_VERSION = 1

METADATA_FIELDS = (
    "filename",
    "source",
    "page",
    "page_label",
    "section_title",
    "sop_version",
    "hub",
    "department",
    "language",
)


def bm25_corpus_path() -> Path:
    return config.CHROMA_PERSIST_DIR / BM25_CORPUS_FILENAME


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """ChromaDB only accepts scalar metadata values."""
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def enrich_metadata(metadata: dict[str, Any], *, source_path: Path, docs_dir: Path) -> dict[str, Any]:
    """Add operational metadata derived from document paths and PDF properties."""
    enriched = dict(metadata)
    relative = source_path.relative_to(docs_dir)
    parts = relative.parts

    enriched.setdefault("source", str(relative))
    enriched.setdefault("filename", source_path.name)
    enriched.setdefault("language", "en")

    if len(parts) >= 3:
        enriched.setdefault("hub", parts[0])
        enriched.setdefault("department", parts[1])
    elif len(parts) == 2:
        enriched.setdefault("department", parts[0])

    title = metadata.get("title") or metadata.get("/Title")
    if title and "section_title" not in enriched:
        enriched["section_title"] = str(title)

    for key in ("sop_version", "version"):
        if key in metadata and "sop_version" not in enriched:
            enriched["sop_version"] = str(metadata[key])
            break

    return enriched


def build_chunk_id(source: str, index: int) -> str:
    return f"{source}::chunk-{index:05d}"


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing."""
    return re.findall(r"[a-z0-9]+", text.lower())


def save_bm25_corpus(
    *,
    records: list[dict[str, Any]],
    collection_name: str | None = None,
    path: Path | None = None,
) -> Path:
    """Persist the BM25 corpus alongside the Chroma collection."""
    target = path or bm25_corpus_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CORPUS_VERSION,
        "collection": collection_name or config.COLLECTION_NAME,
        "records": records,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def load_bm25_corpus(path: Path | None = None) -> list[dict[str, Any]]:
    """Load BM25 corpus records from disk."""
    target = path or bm25_corpus_path()
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    return [record for record in records if record.get("id") and record.get("text")]
