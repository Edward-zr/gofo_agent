"""Parse-once DataFrame store for ADA-style attachment analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tools.files.models import ProcessedFileContext


@dataclass
class StoredAttachment:
    """Parsed attachment kept in conversation state for repeated analysis."""

    attachment_id: str
    filename: str
    file_type: str
    dataframe: pd.DataFrame | None = None
    sheet_names: list[str] = field(default_factory=list)
    active_sheet: str | None = None
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    context: ProcessedFileContext | None = None
    text_content: str | None = None
    image_analysis: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tabular(self) -> bool:
        return self.dataframe is not None and not self.dataframe.empty

    def snapshot(self) -> dict[str, Any]:
        return {
            "attachment_id": self.attachment_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "sheet_names": list(self.sheet_names),
            "active_sheet": self.active_sheet,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "row_count": int(len(self.dataframe)) if self.dataframe is not None else 0,
            "metadata": dict(self.metadata),
        }


def context_to_stored_attachment(context: ProcessedFileContext) -> StoredAttachment:
    """Convert a processed file context into a reusable stored attachment."""
    if context.file_type in {"csv", "excel"}:
        dataframe, sheet_names, active_sheet = _tabular_dataframe(context)
        columns = list(dataframe.columns) if dataframe is not None else []
        dtypes = (
            {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}
            if dataframe is not None
            else {}
        )
        return StoredAttachment(
            attachment_id=context.attachment_id,
            filename=context.filename,
            file_type=context.file_type,
            dataframe=dataframe,
            sheet_names=sheet_names,
            active_sheet=active_sheet,
            columns=columns,
            dtypes=dtypes,
            metadata={
                "schema": context.file_schema,
                "statistics": context.statistics,
                "sample_rows": context.sample_rows,
            },
            context=context,
        )

    text_content = None
    if context.paragraphs:
        text_content = "\n".join(context.paragraphs)
    elif context.pages:
        text_content = "\n".join(str(page.get("text") or "") for page in context.pages)
    elif context.full_data.get("text"):
        text_content = str(context.full_data["text"])

    return StoredAttachment(
        attachment_id=context.attachment_id,
        filename=context.filename,
        file_type=context.file_type,
        metadata={
            "schema": context.file_schema,
            "statistics": context.statistics,
            "summary": context.summary,
        },
        context=context,
        text_content=text_content,
        image_analysis=dict(context.image_analysis or {}),
    )


def _tabular_dataframe(
    context: ProcessedFileContext,
) -> tuple[pd.DataFrame | None, list[str], str | None]:
    if context.file_type == "csv":
        rows = (context.full_data or {}).get("rows") or context.sample_rows or []
        columns = (context.full_data or {}).get("columns") or list(context.file_schema.get("columns") or [])
        if not rows:
            return pd.DataFrame(columns=columns), [], None
        frame = pd.DataFrame(rows)
        if columns:
            ordered = [column for column in columns if column in frame.columns]
            remaining = [column for column in frame.columns if column not in ordered]
            frame = frame[ordered + remaining]
        return _coerce_numeric(frame), [], None

    sheets = (context.full_data or {}).get("sheets") or context.sheets or []
    active_sheet = (context.full_data or {}).get("active_sheet") or context.file_schema.get("active_sheet")
    sheet_names = [str(sheet.get("name") or f"Sheet{index}") for index, sheet in enumerate(sheets, start=1)]
    selected = None
    if sheets:
        if active_sheet:
            selected = next((sheet for sheet in sheets if sheet.get("name") == active_sheet), sheets[0])
        else:
            selected = sheets[0]
            active_sheet = selected.get("name")
    rows = (selected or {}).get("rows") or context.sample_rows or []
    if not rows:
        return pd.DataFrame(), sheet_names, active_sheet
    return _coerce_numeric(pd.DataFrame(rows)), sheet_names, active_sheet


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            converted = pd.to_numeric(result[column], errors="coerce")
            if converted.notna().sum() >= max(1, int(len(result) * 0.5)):
                result[column] = converted
    return result
