"""Models for natural-language capability planning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanningDecision(BaseModel):
    """Decision object returned by the natural-language planner."""

    capability: Literal["sql", "rag", "multi", "unknown"]
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_sql: bool
    requires_rag: bool
    reasoning: str
    entities: dict[str, Any] = Field(default_factory=dict)
