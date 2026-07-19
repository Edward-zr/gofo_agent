"""CSV attachment processing with deterministic analytics."""

from __future__ import annotations

import csv
import io
from typing import Any

from tools.files.models import AttachmentMetadata, ProcessedFileContext, ProcessingStatus


def process_csv(metadata: AttachmentMetadata, content: bytes) -> ProcessedFileContext:
    """Extract schema, statistics, and sample rows from a CSV file."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return _failed(metadata, "The CSV file could not be decoded as UTF-8 text.")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return _failed(metadata, "The CSV file contains no data rows.")

    columns = list(rows[0].keys())
    null_counts = {column: sum(1 for row in rows if not str(row.get(column, "")).strip()) for column in columns}
    numeric_summary = _numeric_summary(rows, columns)
    inferred_types = _infer_types(rows, columns)

    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="csv",
        processing_status=ProcessingStatus.READY,
        summary=f"CSV file with {len(rows)} rows and {len(columns)} columns.",
        file_schema={"columns": columns, "inferred_types": inferred_types},
        statistics={
            "row_count": len(rows),
            "null_counts": null_counts,
            "duplicate_count": _duplicate_count(rows),
            "numeric_summary": numeric_summary,
        },
        sample_rows=rows[:5],
        full_data={"rows": rows, "columns": columns},
        source_references=[metadata.original_filename],
    )


def _numeric_summary(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in columns:
        values = []
        for row in rows:
            raw = row.get(column)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                values.append(float(str(raw).replace(",", "")))
            except ValueError:
                continue
        if values:
            summary[column] = {
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 4),
                "count": len(values),
            }
    return summary


def _infer_types(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, str]:
    inferred: dict[str, str] = {}
    for column in columns:
        sample_values = [str(row.get(column, "")).strip() for row in rows[:20] if str(row.get(column, "")).strip()]
        if not sample_values:
            inferred[column] = "empty"
            continue
        if all(_is_number(value) for value in sample_values):
            inferred[column] = "numeric"
        elif all(_looks_like_date(value) for value in sample_values):
            inferred[column] = "date"
        else:
            inferred[column] = "categorical"
    return inferred


def _duplicate_count(rows: list[dict[str, Any]]) -> int:
    seen = set()
    duplicates = 0
    for row in rows:
        key = tuple(sorted((key, str(value)) for key, value in row.items()))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _is_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def _looks_like_date(value: str) -> bool:
    return any(separator in value for separator in ("-", "/")) and any(char.isdigit() for char in value)


def _failed(metadata: AttachmentMetadata, message: str) -> ProcessedFileContext:
    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="csv",
        processing_status=ProcessingStatus.FAILED,
        error_message=message,
        source_references=[metadata.original_filename],
    )
