"""Build attachment-aware context for the reasoning engine."""

from __future__ import annotations

import re
from typing import Any

from tools.files.models import ProcessedFileContext


def build_file_context(
    *,
    question: str,
    contexts: list[ProcessedFileContext],
    active_attachment_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Convert processed attachments into compact reasoning context."""
    if not contexts:
        return {}

    selected = _select_relevant_contexts(question, contexts)
    return {
        "attachment_ids": [context.attachment_id for context in selected],
        "filenames": [context.filename for context in selected],
        "file_types": [context.file_type for context in selected],
        "summaries": [context.summary for context in selected if context.summary],
        "schemas": [context.file_schema for context in selected if context.file_schema],
        "statistics": [context.statistics for context in selected if context.statistics],
        "sample_rows": _relevant_rows(question, selected),
        "pages": _relevant_pages(question, selected),
        "paragraphs": _relevant_paragraphs(question, selected),
        "image_analysis": [context.image_analysis for context in selected if context.image_analysis],
        "source_references": _source_references(selected),
        "active_attachment_ids": active_attachment_ids or [context.attachment_id for context in selected],
        "contexts": [context.model_dump() for context in selected],
    }


def resolve_file_reference(question: str, contexts: list[ProcessedFileContext]) -> list[ProcessedFileContext]:
    """Resolve conversational file references to specific attachments.

    Contexts are assumed in registration order (oldest → newest). Generic
    "the file" / inspect asks prefer the newest upload so a later file is not
    shadowed by the first one still in session memory.
    """
    if not contexts:
        return []
    normalized = question.lower()
    if any(phrase in normalized for phrase in ("first file", "first report", "first csv")):
        return [contexts[0]]
    if any(phrase in normalized for phrase in ("second file", "second report")) and len(contexts) > 1:
        return [contexts[1]]

    # Multi-file analysis must keep the full set (before single-file narrowing).
    if _asks_multi_file_context(normalized):
        return contexts

    if any(
        phrase in normalized
        for phrase in (
            "new file",
            "newest file",
            "latest file",
            "latest upload",
            "just uploaded",
            "newly uploaded",
            "i uploaded",
            "i have upload",
            "i have uploaded",
        )
    ):
        return contexts[-1:]
    if any(phrase in normalized for phrase in ("the csv", "the spreadsheet", "excel file")):
        matches = [context for context in contexts if context.file_type in {"csv", "excel"}]
        return matches or contexts[-1:]
    if any(phrase in normalized for phrase in ("the pdf", "the report", "uploaded report")):
        matches = [context for context in contexts if context.file_type == "pdf"]
        return matches or contexts[-1:]
    if any(phrase in normalized for phrase in ("the image", "the screenshot", "this image", "this screenshot")):
        matches = [context for context in contexts if context.file_type == "image"]
        return matches or contexts[-1:]
    if any(
        phrase in normalized
        for phrase in (
            "this file",
            "that file",
            "the file",
            "inspect the file",
            "inspect file",
            "uploaded",
            "upload you",
        )
    ):
        return contexts[-1:]
    # Single-file follow-ups with several actives: prefer newest upload.
    if len(contexts) > 1:
        return contexts[-1:]
    return contexts


def _asks_multi_file_context(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "compare",
            " vs ",
            "versus",
            "both files",
            "both reports",
            "all files",
            "these reports",
            "these files",
            "between the",
            "changed the most",
            "changed most",
            "which hub changed",
            "difference between",
            "diff between",
        )
    )


def _select_relevant_contexts(question: str, contexts: list[ProcessedFileContext]) -> list[ProcessedFileContext]:
    resolved = resolve_file_reference(question, contexts)
    if "sheet" in question.lower():
        sheet_name = _extract_sheet_name(question)
        if sheet_name:
            for context in resolved:
                if context.file_type == "excel":
                    matching = [sheet for sheet in context.sheets if sheet.get("name", "").lower() == sheet_name.lower()]
                    if matching:
                        context.sample_rows = matching[0].get("sample_rows", [])
                        context.statistics = {"row_count": matching[0].get("row_count", 0), "sheet": sheet_name}
    return resolved


def _relevant_rows(question: str, contexts: list[ProcessedFileContext]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        if context.file_type in {"csv", "excel"}:
            rows.extend(context.sample_rows[:3])
    return rows


def _relevant_pages(question: str, contexts: list[ProcessedFileContext]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for context in contexts:
        if context.file_type == "pdf":
            pages.extend(context.pages[:3])
    return pages


def _relevant_paragraphs(question: str, contexts: list[ProcessedFileContext]) -> list[str]:
    paragraphs: list[str] = []
    for context in contexts:
        if context.paragraphs:
            paragraphs.extend(context.paragraphs[:5])
    return paragraphs


def _source_references(contexts: list[ProcessedFileContext]) -> list[str]:
    references: list[str] = []
    for context in contexts:
        references.extend(context.source_references)
    return list(dict.fromkeys(references))


def _extract_sheet_name(question: str) -> str | None:
    match = re.search(r"sheet\s+([A-Za-z0-9 _-]+)", question, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None
