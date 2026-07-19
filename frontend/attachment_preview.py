"""Attachment preview helpers for the GOFO Streamlit dashboard."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

IMAGE_TYPES = frozenset({"png", "jpg", "jpeg", "webp"})
TABLE_TYPES = frozenset({"csv", "xlsx", "xls"})
TEXT_TYPES = frozenset({"txt", "md", "json", "log"})
DOCUMENT_TYPES = frozenset({"pdf", "docx"})

FILE_ICONS = {
    "pdf": "📄",
    "csv": "📄",
    "xlsx": "📄",
    "xls": "📄",
    "txt": "📄",
    "md": "📄",
    "json": "📄",
    "log": "📄",
    "docx": "📄",
    "png": "🖼",
    "jpg": "🖼",
    "jpeg": "🖼",
    "webp": "🖼",
}

DEFAULT_TABLE_ROWS = 20
DEFAULT_TEXT_CHARS = 5000


def get_file_type(filename: str, content_type: str | None = None) -> str:
    """Return normalized file type from filename or content type."""
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension:
        return extension
    if content_type:
        lowered = content_type.lower()
        if "pdf" in lowered:
            return "pdf"
        if "spreadsheet" in lowered or "excel" in lowered:
            return "xlsx"
        if "csv" in lowered:
            return "csv"
        if "image/png" in lowered:
            return "png"
        if "image/jpeg" in lowered:
            return "jpg"
        if "image/webp" in lowered:
            return "webp"
        if "word" in lowered:
            return "docx"
        if "json" in lowered:
            return "json"
        if "text" in lowered:
            return "txt"
    return "file"


def format_file_size(size: int) -> str:
    """Return human-readable file size."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{round(size / 1024, 1)} KB"
    return f"{round(size / (1024 * 1024), 1)} MB"


def attachment_identity(filename: str, size: int, content: bytes) -> str:
    """Return a stable identity for deduplicating pending attachments."""
    digest = hashlib.sha256(content).hexdigest()[:16]
    return f"{filename}:{size}:{digest}"


def file_icon(file_type: str) -> str:
    """Return display icon for a file type."""
    return FILE_ICONS.get(file_type.lower(), "📎")


def is_image_type(file_type: str) -> bool:
    return file_type.lower() in IMAGE_TYPES


def is_table_type(file_type: str) -> bool:
    return file_type.lower() in TABLE_TYPES


def is_text_type(file_type: str) -> bool:
    return file_type.lower() in TEXT_TYPES


def is_previewable(file_type: str) -> bool:
    normalized = file_type.lower()
    return (
        normalized in IMAGE_TYPES
        or normalized in TABLE_TYPES
        or normalized in TEXT_TYPES
        or normalized in DOCUMENT_TYPES
    )


def build_preview_payload(
    *,
    filename: str,
    file_type: str,
    content: bytes,
    sheet_name: str | None = None,
    max_rows: int = DEFAULT_TABLE_ROWS,
    max_chars: int = DEFAULT_TEXT_CHARS,
) -> dict[str, Any]:
    """Build structured preview data for an attachment."""
    normalized = file_type.lower()
    if normalized in IMAGE_TYPES:
        return build_image_preview(content, filename, normalized)
    if normalized == "pdf":
        return build_pdf_preview(content, filename)
    if normalized in TABLE_TYPES:
        return build_table_preview(content, normalized, sheet_name=sheet_name, max_rows=max_rows)
    if normalized == "docx":
        return build_docx_preview(content, filename, max_chars=max_chars)
    if normalized in TEXT_TYPES:
        return build_text_preview(content, filename, normalized, max_chars=max_chars)
    return {
        "preview_type": "unsupported",
        "filename": filename,
        "file_type": normalized,
        "message": "Preview unavailable for this file type.",
    }


def build_image_preview(content: bytes, filename: str, file_type: str) -> dict[str, Any]:
    """Validate image bytes for thumbnail rendering."""
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(content))
        width, height = image.size
        return {
            "preview_type": "image",
            "filename": filename,
            "file_type": file_type,
            "width": width,
            "height": height,
            "content": content,
            "error": None,
        }
    except Exception as exc:
        logger.exception("Image preview failed for %s", filename)
        return {
            "preview_type": "image",
            "filename": filename,
            "file_type": file_type,
            "content": None,
            "error": "Unable to preview this file.",
            "detail": str(exc),
        }


def build_pdf_preview(content: bytes, filename: str) -> dict[str, Any]:
    """Extract PDF metadata and first-page text for preview."""
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for index, page in enumerate(reader.pages[:3], start=1):
            pages.append({"page": index, "text": (page.extract_text() or "").strip()})
        first_page_text = pages[0]["text"] if pages else ""
        return {
            "preview_type": "pdf",
            "filename": filename,
            "file_type": "pdf",
            "page_count": len(reader.pages),
            "pages": pages,
            "first_page_text": first_page_text[:DEFAULT_TEXT_CHARS],
            "content": content,
            "error": None,
        }
    except Exception as exc:
        logger.exception("PDF preview failed for %s", filename)
        return {
            "preview_type": "pdf",
            "filename": filename,
            "file_type": "pdf",
            "page_count": 0,
            "pages": [],
            "first_page_text": "",
            "content": content,
            "error": "Unable to preview this file.",
            "detail": str(exc),
        }


def build_table_preview(
    content: bytes,
    file_type: str,
    *,
    sheet_name: str | None = None,
    max_rows: int = DEFAULT_TABLE_ROWS,
) -> dict[str, Any]:
    """Build tabular preview for CSV or Excel files."""
    normalized = file_type.lower()
    try:
        if normalized == "csv":
            return _csv_table_preview(content, max_rows=max_rows)
        return _excel_table_preview(content, sheet_name=sheet_name, max_rows=max_rows)
    except Exception as exc:
        logger.exception("Table preview failed for %s", file_type)
        return {
            "preview_type": "table",
            "filename": "",
            "file_type": normalized,
            "columns": [],
            "rows": [],
            "sheet_names": [],
            "active_sheet": None,
            "truncated": False,
            "error": "Unable to preview this file.",
            "detail": str(exc),
        }


def build_text_preview(
    content: bytes,
    filename: str,
    file_type: str,
    *,
    max_chars: int = DEFAULT_TEXT_CHARS,
) -> dict[str, Any]:
    """Build text or JSON preview."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        logger.exception("Text preview decode failed for %s", filename)
        return {
            "preview_type": "text",
            "filename": filename,
            "file_type": file_type,
            "text": "",
            "truncated": False,
            "error": "Unable to preview this file.",
            "detail": str(exc),
        }

    if not text.strip():
        return {
            "preview_type": "text",
            "filename": filename,
            "file_type": file_type,
            "text": "",
            "truncated": False,
            "error": "The file is empty.",
            "detail": None,
        }

    if file_type.lower() == "json":
        try:
            parsed = json.loads(text)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            truncated = len(formatted) > max_chars
            return {
                "preview_type": "text",
                "filename": filename,
                "file_type": "json",
                "text": formatted[:max_chars],
                "truncated": truncated,
                "error": None,
                "detail": None,
            }
        except json.JSONDecodeError:
            pass

    truncated = len(text) > max_chars
    return {
        "preview_type": "text",
        "filename": filename,
        "file_type": file_type,
        "text": text[:max_chars],
        "truncated": truncated,
        "error": None,
        "detail": None,
    }


def build_docx_preview(content: bytes, filename: str, *, max_chars: int = DEFAULT_TEXT_CHARS) -> dict[str, Any]:
    """Extract limited DOCX text for preview."""
    try:
        document = Document(io.BytesIO(content))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        text = "\n".join(paragraphs)
        if not text.strip():
            return {
                "preview_type": "text",
                "filename": filename,
                "file_type": "docx",
                "text": "",
                "truncated": False,
                "error": "The document contains no readable text.",
                "detail": None,
            }
        truncated = len(text) > max_chars
        return {
            "preview_type": "text",
            "filename": filename,
            "file_type": "docx",
            "text": text[:max_chars],
            "truncated": truncated,
            "error": None,
            "detail": None,
        }
    except Exception as exc:
        logger.exception("DOCX preview failed for %s", filename)
        return {
            "preview_type": "text",
            "filename": filename,
            "file_type": "docx",
            "text": "",
            "truncated": False,
            "error": "Unable to preview this file.",
            "detail": str(exc),
        }


def merge_pending_attachments(
    pending: list[dict[str, Any]],
    uploaded_items: list[dict[str, Any]],
    contents_by_name: dict[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Add uploaded attachments without duplicates."""
    preview_cache: dict[str, bytes] = {}
    existing_identities = {item.get("identity") for item in pending if item.get("identity")}
    existing_ids = {item.get("attachment_id") for item in pending if item.get("attachment_id")}
    merged = list(pending)

    for metadata in uploaded_items:
        attachment_id = metadata.get("attachment_id")
        if not attachment_id or attachment_id in existing_ids:
            continue
        filename = metadata.get("original_filename") or metadata.get("filename") or "upload.bin"
        content = contents_by_name.get(filename, b"")
        identity = attachment_identity(filename, len(content), content)
        if identity in existing_identities:
            continue
        enriched = dict(metadata)
        enriched["identity"] = identity
        merged.append(enriched)
        preview_cache[attachment_id] = content
        existing_identities.add(identity)
        existing_ids.add(attachment_id)

    return merged, preview_cache


def remove_pending_attachment(
    pending: list[dict[str, Any]],
    attachment_id: str,
) -> list[dict[str, Any]]:
    """Remove one pending attachment by ID."""
    return [item for item in pending if item.get("attachment_id") != attachment_id]


def _csv_table_preview(content: bytes, *, max_rows: int) -> dict[str, Any]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    columns = list(rows[0].keys()) if rows else []
    return {
        "preview_type": "table",
        "filename": "",
        "file_type": "csv",
        "columns": columns,
        "rows": rows[:max_rows],
        "sheet_names": [],
        "active_sheet": None,
        "total_rows": len(rows),
        "truncated": len(rows) > max_rows,
        "error": None,
        "detail": None,
    }


def _excel_table_preview(
    content: bytes,
    *,
    sheet_name: str | None,
    max_rows: int,
) -> dict[str, Any]:
    # Use the shared robust reader. openpyxl read-only mode often truncates
    # logistics Excel exports that have incorrect worksheet dimensions.
    from tools.files.excel_reader import read_excel_sheet

    columns, all_rows, sheet_names, active_sheet = read_excel_sheet(
        content,
        sheet_name=sheet_name,
        max_rows=None,
    )
    if not sheet_names:
        return {
            "preview_type": "table",
            "filename": "",
            "file_type": "xlsx",
            "columns": [],
            "rows": [],
            "sheet_names": [],
            "active_sheet": None,
            "total_rows": 0,
            "truncated": False,
            "error": "The workbook contains no readable sheets.",
            "detail": None,
        }

    total_rows = len(all_rows)
    if total_rows == 0:
        return {
            "preview_type": "table",
            "filename": "",
            "file_type": "xlsx",
            "columns": columns,
            "rows": [],
            "sheet_names": sheet_names,
            "active_sheet": active_sheet,
            "total_rows": 0,
            "truncated": False,
            "error": "The selected sheet is empty.",
            "detail": None,
        }

    preview_rows = all_rows[:max_rows]
    return {
        "preview_type": "table",
        "filename": "",
        "file_type": "xlsx",
        "columns": columns or (list(preview_rows[0].keys()) if preview_rows else []),
        "rows": preview_rows,
        "sheet_names": sheet_names,
        "active_sheet": active_sheet,
        "total_rows": total_rows,
        "truncated": total_rows > max_rows,
        "error": None,
        "detail": None,
    }
