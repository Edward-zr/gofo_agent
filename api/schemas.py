"""Public API schemas for the GOFO Operations Intelligence Dashboard."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, description="User question.")
    session_id: str = Field(
        default="default",
        min_length=1,
        description="Client conversation session identifier.",
    )
    attachments: list[str] = Field(
        default_factory=list,
        description="Attachment IDs to include with this question.",
    )


class AttachmentInfo(BaseModel):
    """Uploaded attachment metadata returned to clients."""

    attachment_id: str
    filename: str
    original_filename: str
    content_type: str | None = None
    file_type: str
    file_size: int
    processing_status: str
    uploaded_at: str
    conversation_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class AttachmentUploadResponse(BaseModel):
    """Response for one or more uploaded attachments."""

    attachments: list[AttachmentInfo] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Dashboard-friendly response from the GOFO agent."""

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    sql: str = ""
    data: list[dict[str, Any]] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    kpi: dict[str, Any] = Field(default_factory=dict)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Service health payload."""

    status: str
    version: str
    database: str
    memory: str
    rag: str
