"""Plan which data sources should answer an attachment-aware question."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class DataSource(StrEnum):
    SQLITE = "SQLITE"
    RAG = "RAG"
    ATTACHMENT = "ATTACHMENT"
    ATTACHMENT_AND_SQL = "ATTACHMENT_AND_SQL"
    ATTACHMENT_AND_RAG = "ATTACHMENT_AND_RAG"
    MEMORY = "MEMORY"


def plan_data_sources(
    question: str,
    *,
    has_attachments: bool,
    has_active_attachment_memory: bool = False,
    file_types: list[str] | None = None,
) -> list[DataSource]:
    """Select explicit data sources for a question."""
    normalized = question.lower()
    file_types = file_types or []

    if has_attachments or has_active_attachment_memory:
        if any(phrase in normalized for phrase in ("compare", "match", "discrep", "different totals", "missing records")):
            if any(word in normalized for word in ("database", "sqlite", "operational database", "today's database", "db")):
                return [DataSource.ATTACHMENT_AND_SQL]
            if len(file_types) >= 2 or "these" in normalized or "two reports" in normalized:
                return [DataSource.ATTACHMENT]
        if any(phrase in normalized for phrase in ("sop", "procedure", "follow our", "match our sop", "policy")):
            return [DataSource.ATTACHMENT_AND_RAG]
        if "image" in file_types or "screenshot" in normalized or "dashboard.png" in normalized:
            if any(word in normalized for word in ("database", "compare", "match")):
                return [DataSource.ATTACHMENT_AND_SQL]
            return [DataSource.ATTACHMENT]
        if any(phrase in normalized for phrase in ("summarize", "analyze", "biggest problems", "biggest risks", "report")):
            return [DataSource.ATTACHMENT]
        if any(phrase in normalized for phrase in ("what should operations do", "recommend", "root cause", "why")):
            return [DataSource.ATTACHMENT, DataSource.MEMORY]
        return [DataSource.ATTACHMENT]

    if any(word in normalized for word in ("sop", "policy", "procedure", "cbt")):
        return [DataSource.RAG]
    if any(
        word in normalized
        for word in ("pickup", "hub", "driver", "customer", "operations", "rank", "performance", "today")
    ):
        return [DataSource.SQLITE]
    if any(phrase in normalized for phrase in ("what should operations do", "recommend")):
        return [DataSource.MEMORY]
    return [DataSource.SQLITE]
