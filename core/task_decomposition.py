"""Task decomposition — extract structured constraints before planning.

Stage 2 of the planner-driven execution loop. Does not call tools or LLMs.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.intent_classifier import IntentClassification, IntentType
from core.intent_router import RouteIntent
from core.logger import get_logger

logger = get_logger("task_decomposition")

Domain = Literal["SOP", "SQL", "ADA", "CHAT", "HYBRID", "WAIT_UPLOAD", "UNKNOWN"]


class TaskSpec(BaseModel):
    """Structured user constraints for planning and verification."""

    question: str = ""
    resolved_question: str = ""
    primary_intent: str = ""
    secondary_intent: str | None = None
    domain: Domain = "UNKNOWN"
    is_continuation: bool = False
    attachment_relevant: bool = False
    reuse_previous_context: bool = False

    # Analytical constraints
    ranking_direction: Literal["top", "bottom"] | None = None
    limit: int | None = None
    metric: str | None = None
    dimension: str | None = None
    aggregation: str | None = None
    chart_type: str | None = None
    date_range: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    grouping: str | None = None
    sorting: str | None = None
    output_type: str | None = None
    knowledge_source: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)

    # Provenance
    route_intent: str | None = None
    classifier_intent: str | None = None


_DOMAIN_BY_ROUTE: dict[str, Domain] = {
    RouteIntent.SOP_QA: "SOP",
    RouteIntent.SQL_ANALYTICS: "SQL",
    RouteIntent.FOLLOW_UP: "SQL",
    RouteIntent.ATTACHMENT_ANALYSIS: "ADA",
    RouteIntent.ATTACHMENT_VISUALIZATION: "ADA",
    RouteIntent.GENERAL_CHAT: "CHAT",
    RouteIntent.OPENAI_FALLBACK: "CHAT",
    RouteIntent.WAIT_FOR_UPLOAD: "WAIT_UPLOAD",
}

_KNOWLEDGE_BY_DOMAIN: dict[str, str] = {
    "SOP": "RAG",
    "SQL": "SQLITE",
    "ADA": "ATTACHMENT",
    "CHAT": "LLM",
    "HYBRID": "MULTI",
    "WAIT_UPLOAD": "NONE",
}


def decompose(
    *,
    question: str,
    resolved_question: str | None = None,
    classification: IntentClassification | dict[str, Any] | None = None,
    route_decision: dict[str, Any] | None = None,
    conversation_snapshot: dict[str, Any] | None = None,
    repair: dict[str, Any] | None = None,
    semantic: dict[str, Any] | None = None,
) -> TaskSpec:
    """Extract TaskSpec from existing classifier/router/semantic/conversation signals."""
    route = dict(route_decision or {})
    snap = dict(conversation_snapshot or {})
    repair = dict(repair or {})
    semantic = dict(semantic or {})
    resolved = (resolved_question or question or "").strip()
    normalized = resolved.lower()

    classifier_intent = _classification_intent(classification)
    route_intent = str(route.get("intent") or "")
    primary = route_intent or classifier_intent or IntentType.Unknown.value
    domain = _domain_for(route_intent, classifier_intent, route)
    is_continuation = bool(route.get("followup") or repair.get("repair_detected"))
    attachment_relevant = bool(route.get("use_attachments"))
    reuse_previous = bool(
        is_continuation
        or (classification is not None and _requires_memory(classification))
        or semantic.get("requires_memory")
    )

    ranking_direction = _ranking_direction(normalized)
    limit = _extract_limit(normalized)
    metric = (
        semantic.get("metric")
        or snap.get("metric")
        or snap.get("last_metric")
        or snap.get("current_metric")
    )
    dimension = (
        semantic.get("dimension")
        or snap.get("dimension")
        or snap.get("last_dimension")
    )
    date_range = (
        semantic.get("date_range")
        or snap.get("date_range")
        or snap.get("last_date_range")
        or _infer_date_range(normalized)
    )
    filters = dict(semantic.get("filters") or snap.get("last_filters") or {})
    entities = dict(semantic.get("entities") or snap.get("last_entities") or {})
    chart_type = route.get("chart_type")
    sorting = "DESC" if ranking_direction == "top" else ("ASC" if ranking_direction == "bottom" else None)
    aggregation = _infer_aggregation(normalized)
    grouping = dimension
    output_type = _output_type(domain, chart_type, ranking_direction)
    knowledge_source = _KNOWLEDGE_BY_DOMAIN.get(domain)
    if domain == "HYBRID":
        knowledge_source = ",".join(route.get("data_sources") or []) or "MULTI"

    secondary = None
    if attachment_relevant and domain == "SOP":
        secondary = "ADA"
        domain = "HYBRID"
    elif chart_type and domain == "SQL":
        secondary = "VISUALIZATION"

    spec = TaskSpec(
        question=question,
        resolved_question=resolved,
        primary_intent=primary,
        secondary_intent=secondary,
        domain=domain,  # type: ignore[arg-type]
        is_continuation=is_continuation,
        attachment_relevant=attachment_relevant,
        reuse_previous_context=reuse_previous,
        ranking_direction=ranking_direction,
        limit=limit,
        metric=str(metric) if metric else None,
        dimension=str(dimension) if dimension else None,
        aggregation=aggregation,
        chart_type=str(chart_type) if chart_type else None,
        date_range=str(date_range) if date_range else None,
        filters=filters,
        grouping=str(grouping) if grouping else None,
        sorting=sorting,
        output_type=output_type,
        knowledge_source=knowledge_source,
        entities=entities,
        route_intent=route_intent or None,
        classifier_intent=classifier_intent or None,
    )
    logger.info(
        "[TaskSpec] domain=%s primary=%s chart=%s limit=%s attachment=%s",
        spec.domain,
        spec.primary_intent,
        spec.chart_type,
        spec.limit,
        spec.attachment_relevant,
    )
    return spec


def task_spec_to_plan_inputs(spec: TaskSpec) -> dict[str, Any]:
    """Flatten TaskSpec into planner / executor step inputs."""
    return {
        "question": spec.resolved_question or spec.question,
        "metric": spec.metric,
        "dimension": spec.dimension,
        "date_range": spec.date_range,
        "filters": dict(spec.filters or {}),
        "limit": spec.limit,
        "ranking_direction": spec.ranking_direction,
        "chart_type": spec.chart_type,
        "aggregation": spec.aggregation,
        "grouping": spec.grouping,
        "sorting": spec.sorting,
        "output_type": spec.output_type,
        "knowledge_source": spec.knowledge_source,
        "entities": dict(spec.entities or {}),
        "attachment_relevant": spec.attachment_relevant,
        "domain": spec.domain,
    }


def _classification_intent(classification: IntentClassification | dict[str, Any] | None) -> str:
    if classification is None:
        return ""
    if isinstance(classification, IntentClassification):
        return classification.intent.value
    intent = classification.get("intent")
    return str(intent.value if hasattr(intent, "value") else intent or "")


def _requires_memory(classification: IntentClassification | dict[str, Any]) -> bool:
    if isinstance(classification, IntentClassification):
        return bool(classification.requires_memory)
    return bool(classification.get("requires_memory"))


def _domain_for(route_intent: str, classifier_intent: str, route: dict[str, Any]) -> Domain:
    sources = {str(s) for s in (route.get("data_sources") or [])}
    if sources & {"ATTACHMENT_AND_SQL", "ATTACHMENT_AND_RAG"}:
        return "HYBRID"
    if route_intent in _DOMAIN_BY_ROUTE:
        return _DOMAIN_BY_ROUTE[route_intent]
    mapping = {
        IntentType.SOP_QA.value: "SOP",
        IntentType.SOP_Summary.value: "SOP",
        IntentType.SOP_Compare.value: "SOP",
        IntentType.SQL_Query.value: "SQL",
        IntentType.SQL_Analysis.value: "SQL",
        IntentType.Dashboard.value: "SQL",
        IntentType.Upload_File.value: "ADA",
        IntentType.Greeting.value: "CHAT",
        IntentType.ChitChat.value: "CHAT",
        IntentType.General_Knowledge.value: "CHAT",
    }
    return mapping.get(classifier_intent, "UNKNOWN")  # type: ignore[return-value]


def _ranking_direction(normalized: str) -> Literal["top", "bottom"] | None:
    if any(w in normalized for w in ("bottom", "worst", "lowest", "least")):
        return "bottom"
    if any(w in normalized for w in ("top", "best", "highest", "most", "rank")):
        return "top"
    return None


def _extract_limit(normalized: str) -> int | None:
    match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", normalized)
    if match:
        return int(match.group(1))
    match = re.search(r"\blimit\s+(\d+)\b", normalized)
    if match:
        return int(match.group(1))
    return None


def _infer_date_range(normalized: str) -> str | None:
    for phrase in (
        "today",
        "yesterday",
        "this week",
        "last week",
        "this month",
        "last month",
    ):
        if phrase in normalized:
            return phrase
    return None


def _infer_aggregation(normalized: str) -> str | None:
    if any(w in normalized for w in ("sum", "total", "volume")):
        return "sum"
    if any(w in normalized for w in ("average", "avg", "mean")):
        return "avg"
    if any(w in normalized for w in ("count", "how many", "number of")):
        return "count"
    if "rate" in normalized or "percentage" in normalized or "percent" in normalized:
        return "rate"
    return None


def _output_type(domain: str, chart_type: str | None, ranking: str | None) -> str | None:
    if chart_type:
        return "chart"
    if ranking:
        return "ranking"
    if domain == "SOP":
        return "knowledge_answer"
    if domain == "ADA":
        return "file_analysis"
    if domain == "CHAT":
        return "conversation"
    if domain == "SQL":
        return "analytics"
    return None
