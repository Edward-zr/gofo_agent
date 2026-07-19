"""Structured models for uploaded attachments and processed file context."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ProcessingStatus(StrEnum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class AttachmentMetadata(BaseModel):
    """Metadata for a stored uploaded file."""

    attachment_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    original_filename: str
    content_type: str | None = None
    file_type: str
    file_size: int
    storage_path: str
    uploaded_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )
    conversation_id: str = "default"
    processing_status: ProcessingStatus = ProcessingStatus.UPLOADED
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class ProcessedFileContext(BaseModel):
    """Structured context extracted from a processed attachment."""

    attachment_id: str
    filename: str
    file_type: str
    processing_status: ProcessingStatus = ProcessingStatus.READY
    summary: str | None = None
    file_schema: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    pages: list[dict[str, Any]] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    sheets: list[dict[str, Any]] = Field(default_factory=list)
    image_analysis: dict[str, Any] = Field(default_factory=dict)
    full_data: dict[str, Any] = Field(default_factory=dict)
    source_references: list[str] = Field(default_factory=list)
    error_message: str | None = None
