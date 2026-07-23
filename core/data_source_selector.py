"""Data Source Selection layer for the GOFO Planner.

Runs BEFORE Tool Orchestration. Chooses the minimum set of data sources needed
to answer a question well. Never executes tools and never answers the user.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.intent_classifier import IntentClassification, IntentType
from core.logger import get_logger

logger = get_logger("data_source_selector")


class DataSource(StrEnum):
    """Canonical data sources the Planner may select."""

    SQL = "SQL"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    RAG = "RAG"
    MEMORY = "MEMORY"
    TRANSFORM = "TRANSFORM"
    STATISTICS = "STATISTICS"
    PYTHON = "PYTHON"  # legacy alias; prefer TRANSFORM + STATISTICS
    VISUALIZATION = "VISUALIZATION"
    RECOMMENDATION = "RECOMMENDATION"
    ATTACHMENT = "ATTACHMENT"
    LLM = "LLM"


class SelectedSource(BaseModel):
    """One chosen source with rationale and expected output."""

    source: str
    reason: str
    expected_output: str
    priority: int = Field(ge=1, description="Lower runs earlier when independent.")


class DataSourceSelection(BaseModel):
    """Structured selection result consumed by the Planner."""

    question: str
    selected_sources: list[str] = Field(default_factory=list)
    source_details: list[SelectedSource] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    combine_reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = ""
    requires_clarification: bool = False
    clarification_question: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    def reasons_map(self) -> dict[str, str]:
        return {item.source: item.reason for item in self.source_details}


# ---------------------------------------------------------------------------
# Signal lexicons
# ---------------------------------------------------------------------------

_SQL_METRIC_PHRASES = (
    "pickup rate",
    "success rate",
    "completion rate",
    "failure rate",
    "kpi",
    "metric",
    "metrics",
    "count",
    "counts",
    "how many",
    "number of",
    "total",
    "average",
    "avg",
    "ranking",
    "rank",
    "lowest",
    "highest",
    "worst",
    "best",
    "top ",
    "bottom ",
    "performance",
    "pickups",
    "failed pickup",
    "delayed",
    "today",
    "yesterday",
    "last week",
    "this week",
    "this month",
    "historical",
    "trend",
    "vs ",
    "versus",
    "compare",
    "comparison",
    "week over week",
    "day over day",
)

_KG_RELATION_PHRASES = (
    "belong to",
    "belongs to",
    "belong",
    "which region",
    "what region",
    "who manages",
    "managed by",
    "manager of",
    "reports to",
    "ownership",
    "owns the sop",
    "sop owner",
    "who owns",
    "relationship",
    "related to",
    "assigned to",
    "works at",
    "works for",
    "based at",
    "which hub does",
    "what hub does",
    "hub for driver",
    "driver → hub",
    "driver -> hub",
    "manager → hub",
    "manager -> hub",
)

_KG_ENTITY_TERMS = (
    "driver",
    "hub",
    "region",
    "manager",
    "sop ownership",
    "org chart",
    "organization",
)

_RAG_PHRASES = (
    "sop",
    "policy",
    "policies",
    "procedure",
    "procedures",
    "guideline",
    "guidelines",
    "document",
    "documents",
    "explain the",
    "how do we",
    "how should",
    "what is the process",
    "playbook",
    "standard operating",
    "unstructured",
    "possible reasons",
    "explain possible",
    "why might",
    "root cause guidance",
)

_MEMORY_PHRASES = (
    "follow up",
    "follow-up",
    "as i said",
    "as above",
    "previous answer",
    "previous result",
    "that number",
    "that hub",
    "same filter",
    "same date",
    "what about",
    "and for ",
    "show only",
    "actually i mean",
    "i meant",
    "instead",
    "explain that",
    "explain those",
    "those results",
)

_PYTHON_PHRASES = (
    "calculate",
    "calculation",
    "statistics",
    "statistical",
    "correlation",
    "forecast",
    "forecasting",
    "predict",
    "prediction",
    "regression",
    "stddev",
    "standard deviation",
    "percentile",
    "z-score",
    "moving average",
)

_CHART_PHRASES = (
    "chart",
    "charts",
    "plot",
    "graph",
    "visualize",
    "visualise",
    "visualization",
    "dashboard",
    "trend chart",
    "heatmap",
    "histogram",
    "scatter",
    "pie chart",
    "bar chart",
    "line chart",
)

_EXPLAIN_REASON_PHRASES = (
    "explain possible reasons",
    "possible reasons",
    "explain why",
    "why might",
    "what could cause",
    "what caused",
    "root cause",
    "reasons behind",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _needs_sql(text: str, classification: IntentClassification | None) -> bool:
    if classification and classification.intent in {
        IntentType.SQL_Query,
        IntentType.SQL_Analysis,
        IntentType.Dashboard,
    }:
        return True
    if classification and classification.requires_sql:
        return True
    return _has_any(text, _SQL_METRIC_PHRASES)


def _needs_knowledge_graph(text: str) -> bool:
    if _has_any(text, _KG_RELATION_PHRASES):
        return True
    # "Who manages the hub ..." / "region does Driver X belong"
    if re.search(r"\bwho manages\b", text):
        return True
    if re.search(r"\b(driver|manager)\b.+\b(hub|region)\b", text):
        return True
    if re.search(r"\b(hub|region)\b.+\b(driver|manager)\b", text) and _has_any(
        text, ("belong", "manage", "assigned", "works")
    ):
        return True
    if "sop ownership" in text or ("who owns" in text and "sop" in text):
        return True
    return False


def _needs_rag(text: str, classification: IntentClassification | None) -> bool:
    if classification and classification.intent in {
        IntentType.SOP_QA,
        IntentType.SOP_Summary,
        IntentType.SOP_Compare,
    }:
        return True
    if classification and classification.requires_rag:
        return True
    if _has_any(text, _RAG_PHRASES):
        return True
    if _has_any(text, _EXPLAIN_REASON_PHRASES) and _needs_sql(text, classification):
        # Operational compare + explanation → SQL metrics + RAG policy/context
        return True
    return False


def _needs_memory(text: str, classification: IntentClassification | None, has_history: bool) -> bool:
    if classification and classification.intent in {IntentType.Follow_Up, IntentType.Explain_Result}:
        return True
    if classification and classification.requires_memory and has_history:
        return True
    if has_history and _has_any(text, _MEMORY_PHRASES):
        return True
    return False


_RECOMMEND_PHRASES = (
    "recommend",
    "recommendation",
    "recommendations",
    "suggest",
    "suggestion",
    "what should we do",
    "next steps",
    "insight",
    "insights",
    "how can we improve",
    "actionable",
)


def _needs_python(text: str) -> bool:
    if _has_any(text, _PYTHON_PHRASES):
        return True
    # Charts often need transform + statistics before visualization.
    if _has_any(text, _CHART_PHRASES):
        return True
    return False


def _needs_visualization(text: str, classification: IntentClassification | None) -> bool:
    if classification and classification.intent == IntentType.Dashboard:
        return True
    return _has_any(text, _CHART_PHRASES)


def _needs_recommendation(text: str, classification: IntentClassification | None) -> bool:
    if _has_any(text, _RECOMMEND_PHRASES):
        return True
    if classification and classification.intent == IntentType.Dashboard:
        return True
    return False


def _needs_attachment(classification: IntentClassification | None, attachment_context: dict[str, Any] | None) -> bool:
    if classification and classification.intent == IntentType.Upload_File:
        return True
    if attachment_context and attachment_context.get("use_attachments"):
        return True
    return False


def _pure_relationship_query(text: str, classification: IntentClassification | None) -> bool:
    """True when the question is about org/entity links, not metrics."""
    if not _needs_knowledge_graph(text):
        return False
    metric_hit = _has_any(
        text,
        (
            "rate",
            "count",
            "kpi",
            "performance",
            "pickups",
            "lowest",
            "highest",
            "worst",
            "best",
            "today",
            "yesterday",
            "trend",
            "compare",
        ),
    )
    # "Who manages the hub with the lowest pickup rate?" needs SQL + KG.
    if metric_hit and ("lowest" in text or "highest" in text or "worst" in text or "best" in text or "rate" in text):
        return False
    if classification and classification.intent in {
        IntentType.SOP_QA,
        IntentType.SOP_Summary,
        IntentType.SOP_Compare,
        IntentType.Dashboard,
        IntentType.Upload_File,
    }:
        return False
    return True


def _add(
    selected: dict[str, SelectedSource],
    source: DataSource,
    reason: str,
    expected_output: str,
    priority: int,
) -> None:
    selected[source.value] = SelectedSource(
        source=source.value,
        reason=reason,
        expected_output=expected_output,
        priority=priority,
    )


def _finalize(
    question: str,
    selected: dict[str, SelectedSource],
    *,
    combine_reason: str,
    confidence: float,
    reasoning: str,
    requires_clarification: bool = False,
    clarification_question: str | None = None,
) -> DataSourceSelection:
    # Always keep LLM for final synthesis when any evidence source is present,
    # except pure clarification.
    if selected and DataSource.LLM.value not in selected and not requires_clarification:
        _add(
            selected,
            DataSource.LLM,
            "Synthesize a concise operations answer from selected evidence sources.",
            "Final natural-language answer",
            priority=90,
        )

    details = sorted(selected.values(), key=lambda item: (item.priority, item.source))
    # Stable execution order by priority; LLM last among selected.
    order = [item.source for item in details]
    outputs = [f"{item.source}: {item.expected_output}" for item in details]
    sources = [item.source for item in details]

    result = DataSourceSelection(
        question=question,
        selected_sources=sources,
        source_details=details,
        execution_order=order,
        expected_outputs=outputs,
        combine_reason=combine_reason,
        confidence=confidence,
        reasoning=reasoning,
        requires_clarification=requires_clarification,
        clarification_question=clarification_question,
    )
    logger.info(
        "Data sources selected=%s order=%s reason=%s",
        result.selected_sources,
        result.execution_order,
        result.combine_reason,
    )
    return result


class DataSourceSelector:
    """Select the minimum data sources required before planning tool steps."""

    def select(
        self,
        question: str,
        classification: IntentClassification | None = None,
        conversation_memory: Any = None,
        *,
        attachment_context: dict[str, Any] | None = None,
    ) -> DataSourceSelection:
        text = _normalize(question)
        selected: dict[str, SelectedSource] = {}
        has_history = False
        if conversation_memory is not None:
            if hasattr(conversation_memory, "get_recent_history"):
                has_history = bool(conversation_memory.get_recent_history())
            elif isinstance(conversation_memory, list):
                has_history = bool(conversation_memory)

        if classification and (
            classification.requires_clarification or classification.intent == IntentType.Unknown
        ):
            return _finalize(
                question,
                {},
                combine_reason="Clarification required before selecting tools.",
                confidence=classification.confidence,
                reasoning="Classifier requested clarification.",
                requires_clarification=True,
                clarification_question=(
                    "I need a bit more detail to choose the right data sources.\n\n"
                    "Do you want:\n"
                    "• today's data\n"
                    "• this week's data\n"
                    "• all historical data?"
                ),
            )

        if classification and classification.intent == IntentType.Greeting:
            _add(selected, DataSource.LLM, "Greeting / capability intro only.", "Greeting reply", 1)
            return _finalize(
                question,
                selected,
                combine_reason="Greeting does not need operational data sources.",
                confidence=0.99,
                reasoning="Greeting → LLM only.",
            )

        if classification and classification.intent in {
            IntentType.ChitChat,
            IntentType.General_Knowledge,
            IntentType.Coding,
        }:
            _add(selected, DataSource.LLM, "General / non-ops request.", "General reply", 1)
            return _finalize(
                question,
                selected,
                combine_reason="Non-operational request answered by LLM only.",
                confidence=0.95,
                reasoning=f"{classification.intent.value} → LLM only.",
            )

        # --- Primary selection rules (minimum necessary set) ---
        want_attachment = _needs_attachment(classification, attachment_context)
        want_memory = _needs_memory(text, classification, has_history)
        want_kg = _needs_knowledge_graph(text)
        want_sql = _needs_sql(text, classification)
        want_rag = _needs_rag(text, classification)
        want_python = _needs_python(text)
        want_viz = _needs_visualization(text, classification)
        want_recommend = _needs_recommendation(text, classification)

        # Pure relationship queries: KG only (no SQL/RAG).
        if want_kg and _pure_relationship_query(text, classification):
            want_sql = False
            want_rag = False
            want_python = False
            want_viz = False
            want_recommend = False

        # Pure SOP explain without metrics: RAG only.
        if want_rag and not want_sql and not want_kg and classification and classification.intent in {
            IntentType.SOP_QA,
            IntentType.SOP_Summary,
            IntentType.SOP_Compare,
        }:
            want_python = False
            want_viz = False
            want_recommend = False

        # Pure metric / KPI questions: SQL only (no Python unless calc/chart).
        if want_sql and not want_kg and not want_rag and not want_python and not want_viz:
            pass

        if want_kg and want_sql:
            if not want_python and not _has_any(text, _PYTHON_PHRASES + _CHART_PHRASES):
                want_python = False
                want_viz = False

        # Chart/trend requests: SQL + Transform + Statistics + Visualization.
        if want_viz or (want_python and _has_any(text, _CHART_PHRASES)):
            want_sql = True
            want_python = True
            want_viz = True
            if not _has_any(text, _RAG_PHRASES) and not _has_any(text, _EXPLAIN_REASON_PHRASES):
                want_rag = False

        if want_attachment:
            _add(
                selected,
                DataSource.ATTACHMENT,
                "User uploaded file(s) are the primary evidence.",
                "Attachment / ADA analysis result",
                priority=5,
            )

        if want_memory:
            _add(
                selected,
                DataSource.MEMORY,
                "Follow-up / conversational context must inherit prior filters and results.",
                "Prior conversation state and last result context",
                priority=10,
            )

        if want_sql:
            _add(
                selected,
                DataSource.SQL,
                "Metrics, counts, KPIs, aggregations, rankings, or historical operational data.",
                "SQL result rows / KPI values",
                priority=20,
            )

        if want_kg:
            reason = (
                "Entity relationships (Driver → Hub → Region, Manager → Hub, SOP ownership)."
            )
            if want_sql:
                reason = (
                    "Resolve organizational relationships for entities discovered by SQL "
                    "(e.g. manager of the lowest-rate hub)."
                )
            _add(
                selected,
                DataSource.KNOWLEDGE_GRAPH,
                reason,
                "Relationship facts (hub/region/manager/ownership)",
                priority=30 if want_sql else 20,
            )

        if want_rag:
            _add(
                selected,
                DataSource.RAG,
                "Unstructured SOP / policy / document knowledge or explanation of possible reasons.",
                "Retrieved SOP / policy chunks",
                priority=40,
            )

        # Specialized Python tools (never monolithic analytics.py)
        if want_python:
            _add(
                selected,
                DataSource.TRANSFORM,
                "Prepare / reshape DataFrame before statistics or charts.",
                "Transformed DataFrame",
                priority=50,
            )
            _add(
                selected,
                DataSource.STATISTICS,
                "Compute summary stats, ratios, growth, rankings (LLM does not calculate).",
                "Statistics dictionary",
                priority=55,
            )

        if want_viz:
            _add(
                selected,
                DataSource.VISUALIZATION,
                "User requested a chart, plot, trend visualization, or dashboard.",
                "Chart image(s) / chart metadata",
                priority=60,
            )

        if want_recommend and (want_python or want_viz or _has_any(text, _RECOMMEND_PHRASES)):
            _add(
                selected,
                DataSource.RECOMMENDATION,
                "Business recommendations from computed analytics (not raw SQL).",
                "Recommendation list / operational insights",
                priority=70,
            )

        # Fallback: if nothing matched, use classification flags or LLM.
        if not selected:
            if classification and classification.requires_sql:
                _add(
                    selected,
                    DataSource.SQL,
                    "Classifier flagged SQL requirement.",
                    "SQL result rows",
                    20,
                )
            elif classification and classification.requires_rag:
                _add(
                    selected,
                    DataSource.RAG,
                    "Classifier flagged RAG requirement.",
                    "Retrieved document chunks",
                    40,
                )
            else:
                _add(selected, DataSource.LLM, "No stronger data source matched.", "General reply", 1)

        sources = set(selected)
        if len(sources - {DataSource.LLM.value}) <= 1:
            combine_reason = "Single primary data source is sufficient."
        else:
            combine_reason = (
                "Multiple sources combined only because the question needs both "
                "quantitative evidence and relational/unstructured context."
            )

        confidence = 0.92
        if classification is not None:
            confidence = max(0.7, min(0.98, (classification.confidence + 0.9) / 2))

        reasoning = (
            f"signals sql={want_sql} kg={want_kg} rag={want_rag} "
            f"memory={want_memory} transform/stats={want_python} viz={want_viz} "
            f"recommend={want_recommend} attachment={want_attachment}"
        )
        return _finalize(
            question,
            selected,
            combine_reason=combine_reason,
            confidence=confidence,
            reasoning=reasoning,
        )


def select_data_sources(
    question: str,
    classification: IntentClassification | None = None,
    conversation_memory: Any = None,
    *,
    attachment_context: dict[str, Any] | None = None,
) -> DataSourceSelection:
    """Module-level helper."""
    return DataSourceSelector().select(
        question,
        classification,
        conversation_memory,
        attachment_context=attachment_context,
    )
