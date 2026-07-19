"""Models for semantic request analysis and response orchestration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Domain = Literal[
    "general_conversation",
    "sop_knowledge",
    "data_analytics",
    "attachment",
    "memory",
]

DataSubIntent = Literal[
    "ranking",
    "detail",
    "analysis",
    "lookup",
    "comparison",
    "trend",
    "summary",
    "recommendation",
    "root_cause",
    "explanation",
    "overview",
    "anomaly",
    "greeting",
    "help",
    "clarification",
    "unknown",
]

ResponseMode = Literal[
    "direct",
    "analytical",
    "executive",
    "investigative",
    "conversational",
]

Capability = Literal["sql", "rag", "multi", "conversation", "unknown"]


class SemanticAnalysis(BaseModel):
    """Structured semantic understanding of a user request."""

    domain: Domain = "data_analytics"
    sub_intent: DataSubIntent = "analysis"
    capability: Capability = "sql"
    response_mode: ResponseMode = "analytical"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    resolved_question: str = ""
    reasoning: str = ""
    entities: dict[str, Any] = Field(default_factory=dict)
    metric: str | None = None
    dimension: str | None = None
    use_previous_result: bool = False
    use_previous_analysis: bool = False
    requires_sql: bool = False
    requires_rag: bool = False
    direct_reply: str | None = None
    planner_intent: str = "operational_analysis"

    def to_planning_decision(self) -> "PlanningDecision":
        """Convert semantic analysis into a planner-compatible decision."""
        from tools.planner.models import PlanningDecision

        capability = self.capability
        if capability == "conversation":
            capability = "unknown"
        return PlanningDecision(
            capability=capability,  # type: ignore[arg-type]
            intent=self.planner_intent,
            confidence=self.confidence,
            requires_sql=self.requires_sql,
            requires_rag=self.requires_rag,
            reasoning=self.reasoning,
            entities=self.entities,
        )

    def intent_hint(self) -> str | None:
        """Map semantic analysis to conversation intent classifier hints."""
        mapping = {
            "ranking": "SQL_QUERY",
            "lookup": "SQL_QUERY",
            "comparison": "COMPARISON",
            "trend": "TREND",
            "summary": "SUMMARY",
            "recommendation": "RECOMMENDATION",
            "root_cause": "ROOT_CAUSE",
            "explanation": "EXPLANATION",
            "detail": "DRILLDOWN",
            "greeting": "GREETING",
            "help": "EXPLANATION",
            "clarification": "CLARIFICATION",
        }
        return mapping.get(self.sub_intent)

    def is_actionable(self) -> bool:
        """Return whether the request should execute a GOFO capability."""
        return self.capability in {"sql", "rag", "multi"}

    def should_answer_directly(self) -> bool:
        """Return whether a conversational direct reply is appropriate."""
        return bool(self.direct_reply) and self.domain == "general_conversation"
