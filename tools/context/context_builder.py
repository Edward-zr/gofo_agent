"""Build explicit execution context from semantic conversation memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.conversation import ConversationResolution
from tools.planner.intent_classifier import Intent


@dataclass
class ContextBuild:
    """Execution-ready context for the current turn."""

    original_question: str
    resolved_question: str
    intent: Intent
    metric: str | None = None
    dimension: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    date_range: str | None = None
    inherited_context: dict[str, Any] = field(default_factory=dict)
    use_previous_result: bool = False
    use_previous_analysis: bool = False
    needs_sql: bool = True
    needs_response_from_memory: bool = False
    cache_key: tuple[Any, ...] | None = None


def build_context(
    resolution: ConversationResolution,
    intent: Intent,
    conversation_state: dict[str, Any],
) -> ContextBuild:
    """Rebuild missing semantic context while avoiding accidental filters."""
    inherited = dict(resolution.inherited_context or {})
    explicit_filters = dict(resolution.filters or {})
    metric = resolution.metric or inherited.get("metric") or conversation_state.get("metric")
    dimension = resolution.dimension or inherited.get("dimension") or conversation_state.get("dimension")
    date_range = resolution.date_range

    if intent in {Intent.ROOT_CAUSE, Intent.RECOMMENDATION, Intent.EXPLANATION, Intent.DRILLDOWN, Intent.COMPARISON}:
        inherited = _merge_missing_context(inherited, conversation_state)
        metric = metric or inherited.get("metric")
        dimension = dimension or inherited.get("dimension")
        if date_range is None and intent != Intent.COMPARISON:
            date_range = inherited.get("date_range")

    filters = explicit_filters
    if intent in {Intent.ROOT_CAUSE, Intent.RECOMMENDATION, Intent.EXPLANATION, Intent.DRILLDOWN}:
        filters = _safe_inherited_filters(explicit_filters, inherited)

    has_prior_analysis = bool(
        conversation_state.get("summary")
        or conversation_state.get("business_findings")
        or conversation_state.get("ranking")
    )
    needs_response_from_memory = (
        intent in {Intent.RECOMMENDATION, Intent.EXPLANATION}
        or (intent == Intent.SUMMARY and resolution.use_previous_analysis)
    ) and has_prior_analysis
    use_previous_result = resolution.use_previous_result or intent == Intent.DRILLDOWN
    use_previous_analysis = resolution.use_previous_analysis or needs_response_from_memory

    return ContextBuild(
        original_question=resolution.original_question,
        resolved_question=resolution.resolved_question,
        intent=intent,
        metric=metric,
        dimension=dimension,
        entities=dict(resolution.entities or {}),
        filters=filters,
        date_range=date_range,
        inherited_context=inherited,
        use_previous_result=use_previous_result,
        use_previous_analysis=use_previous_analysis,
        needs_sql=not needs_response_from_memory,
        needs_response_from_memory=needs_response_from_memory,
        cache_key=resolution.cache_key,
    )


def _merge_missing_context(
    inherited: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(inherited)
    for key in (
        "intent",
        "metric",
        "dimension",
        "entities",
        "filters",
        "ranking",
        "best_entity",
        "worst_entity",
        "date_range",
        "summary",
        "business_findings",
        "recommendations",
        "root_cause",
    ):
        if not merged.get(key):
            merged[key] = state.get(key)
    return merged


def _safe_inherited_filters(
    explicit_filters: dict[str, Any],
    inherited: dict[str, Any],
) -> dict[str, Any]:
    """Inherit filters only for contextual intents or explicitly resolved entities."""
    if explicit_filters:
        return explicit_filters
    filters = dict(inherited.get("filters") or {})
    # Ranking questions should be global by default. Contextual root-cause and
    # recommendation questions may inherit target entity filters because the
    # resolver has turned "worst hub" or "that driver" into a concrete target.
    return filters
