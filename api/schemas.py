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


class AskResponse(BaseModel):
    """Dashboard-friendly response from the GOFO agent."""

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    sql: str = ""
    data: list[dict[str, Any]] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    kpi: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Service health payload."""

    status: str
    version: str
    database: str
    memory: str
    rag: str
