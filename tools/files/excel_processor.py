"""Excel attachment processing."""

from __future__ import annotations

from typing import Any

from tools.files.excel_reader import read_excel_sheets
from tools.files.models import AttachmentMetadata, ProcessedFileContext, ProcessingStatus


def process_excel(metadata: AttachmentMetadata, content: bytes) -> ProcessedFileContext:
    """Extract sheet metadata, schema, and statistics from an Excel workbook."""
    try:
        sheet_map = read_excel_sheets(content)
    except Exception:
        return _failed(metadata, "The spreadsheet could not be read.")

    if not sheet_map:
        return _failed(metadata, "The workbook contains no readable sheets.")

    sheets: list[dict[str, Any]] = []
    for sheet_name, data_rows in sheet_map.items():
        columns = list(data_rows[0].keys()) if data_rows else []
        sheets.append(
            {
                "name": sheet_name,
                "row_count": len(data_rows),
                "column_count": len(columns),
                "columns": columns,
                "sample_rows": data_rows[:5],
                "rows": data_rows,
            }
        )

    active_sheet = sheets[0]["name"]
    primary = sheets[0]
    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="excel",
        processing_status=ProcessingStatus.READY,
        summary=f"Excel workbook with {len(sheets)} sheet(s) and {primary.get('row_count', 0)} row(s).",
        file_schema={"columns": primary.get("columns", []), "active_sheet": primary.get("name")},
        statistics={"row_count": primary.get("row_count", 0), "sheet_count": len(sheets)},
        sample_rows=primary.get("sample_rows", []),
        sheets=sheets,
        full_data={"sheets": sheets, "active_sheet": active_sheet},
        source_references=[f"{metadata.original_filename}, sheet {primary.get('name')}"],
    )


def _failed(metadata: AttachmentMetadata, message: str) -> ProcessedFileContext:
    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="excel",
        processing_status=ProcessingStatus.FAILED,
        error_message=message,
        source_references=[metadata.original_filename],
    )
