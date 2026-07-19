"""Detect logical file types for processor dispatch."""

from __future__ import annotations

from pathlib import Path

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
_SPREADSHEET_EXTENSIONS = {"csv", "xlsx", "xls"}
_DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt", "md"}


def detect_file_type(filename: str, extension: str | None = None) -> str:
    """Return a normalized processor file type."""
    ext = (extension or Path(filename).suffix.lower().lstrip(".")).lower()
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext == "csv":
        return "csv"
    if ext in {"xlsx", "xls"}:
        return "excel"
    if ext == "pdf":
        return "pdf"
    if ext == "docx":
        return "docx"
    if ext in {"txt", "md"}:
        return "text"
    return ext
