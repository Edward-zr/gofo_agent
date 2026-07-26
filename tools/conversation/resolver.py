"""Semantic conversation resolver for operations follow-up reasoning.

This module is the session-level source of truth for conversational meaning.
It keeps structured state instead of relying on raw text snippets, so follow-up
questions can refer to prior rankings, entities, metrics, dates, findings, and
recommendations without adding a new one-off rule for every phrasing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import re

from core.models import QueryResponse

MAX_TURNS = 10

_REPAIR_PHRASES = (
    "actually",
    "correction",
    "i mean",
    "i meant",
    "instead",
    "no",
    "not that",
    "wrong",
)

_DIMENSION_WORDS = {
    "hub": ("hub", "hubs", "warehouse", "warehouses", "station", "stations"),
    "driver": ("driver", "drivers"),
    "customer": ("customer", "customers"),
    "pickup": ("pickup", "pickups"),
    "package": ("package", "packages"),
    "status": ("status", "completed", "delayed", "failed"),
    "date": ("date", "today", "yesterday", "week", "month"),
}

_DATE_PHRASES = (
    "today",
    "yesterday",
    "this week",
    "last week",
    "this month",
    "last month",
)


@dataclass
class ConversationResolution:
    """Resolved conversational meaning for the current user question."""

    original_question: str
    resolved_question: str
    intent_hint: str | None = None
    metric: str | None = None
    dimension: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    date_range: str | None = None
    inherited_context: dict[str, Any] = field(default_factory=dict)
    repair_detected: bool = False
    repair_type: str | None = None
    changed_dimension: str | None = None
    use_previous_result: bool = False
    use_previous_analysis: bool = False
    cache_key: tuple[Any, ...] | None = None


class ConversationResolver:
    """Maintain semantic memory and resolve follow-up questions."""

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self.max_turns = max_turns
        self.turns: list[dict[str, Any]] = []
        self.state: dict[str, Any] = _empty_state()
        self._sql_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the semantic state."""
        state = deepcopy(self.state)
        state["turn_count"] = len(self.turns)
        state["recent_turns"] = deepcopy(self.turns[-self.max_turns :])
        return state

    def resolve(self, question: str) -> ConversationResolution:
        """Resolve a new user question using structured semantic memory."""
        original = question.strip()
        if not original:
            raise ValueError("Question must not be empty.")

        normalized = _normalize(original)
        context = self.snapshot()
        resolution = ConversationResolution(
            original_question=original,
            resolved_question=original,
            metric=_infer_metric(normalized) or self.state.get("metric"),
            dimension=_infer_dimension(normalized),
            entities=_extract_entities(original),
            filters=_explicit_filters(original),
            date_range=_infer_date_range(normalized),
            inherited_context={},
        )

        if _is_repair(normalized):
            return self._resolve_repair(resolution, normalized, context)

        if normalized in {
            "what happened today",
            "what happened today?",
            "today's report",
            "todays report",
            "summarize today",
        }:
            resolution.resolved_question = "Generate today's executive operational summary"
            resolution.intent_hint = "SUMMARY"
            resolution.metric = "operations_health"
            resolution.dimension = "hub"
            resolution.date_range = "today"
            resolution.cache_key = _cache_key(resolution)
            return resolution

        special = self._resolve_special_followup(resolution, normalized, context)
        if special:
            return special

        if _is_contextual_followup(normalized):
            return self._resolve_contextual_followup(resolution, normalized, context)

        # Fresh direct questions should not inherit old entity filters unless
        # the user explicitly mentions them. This prevents stale hub filters
        # from constraining global rankings.
        resolution.metric = resolution.metric or _infer_metric(normalized)
        resolution.dimension = resolution.dimension or _infer_dimension(normalized)
        resolution.date_range = resolution.date_range or _infer_date_range(normalized)
        resolution.filters = _explicit_filters(original)
        resolution.cache_key = _cache_key(resolution)
        return resolution

    def cached_response(self, resolution: ConversationResolution) -> QueryResponse | None:
        """Return a cached response for an identical semantic SQL request."""
        if not resolution.cache_key or resolution.repair_detected:
            return None
        cached = self._sql_cache.get(resolution.cache_key)
        if not cached:
            return None
        return QueryResponse.model_validate(deepcopy(cached))

    def update(
        self,
        *,
        original_question: str,
        resolved_question: str,
        response: QueryResponse,
        intent: str | None = None,
        inherited_context: dict[str, Any] | None = None,
        cache_key: tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        """Update semantic memory from a completed response."""
        semantic = _semantic_from_response(
            original_question=original_question,
            resolved_question=resolved_question,
            response=response,
            intent=intent,
            inherited_context=inherited_context or {},
        )
        self.turns.append(semantic)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

        self.state.update(
            {
                "original_question": original_question,
                "resolved_question": resolved_question,
                "intent": semantic.get("intent"),
                "metric": semantic.get("metric"),
                "dimension": semantic.get("dimension"),
                "entities": semantic.get("entities", {}),
                "filters": semantic.get("filters", {}),
                "generated_sql": response.generated_sql,
                "sql_rows": deepcopy(response.sql_rows),
                "business_findings": semantic.get("business_findings", []),
                "kpi_summary": deepcopy(response.kpi_summary),
                "ranking": semantic.get("ranking", []),
                "best_entity": semantic.get("best_entity"),
                "worst_entity": semantic.get("worst_entity"),
                "root_cause": deepcopy(response.root_cause),
                "recommendations": semantic.get("recommendations", []),
                "summary": response.answer,
                "date_range": semantic.get("date_range"),
                "last_response": response.model_dump(),
            }
        )
        if cache_key and response.generated_sql:
            self._sql_cache[cache_key] = response.model_dump()
        return self.snapshot()

    def _resolve_repair(
        self,
        resolution: ConversationResolution,
        normalized: str,
        context: dict[str, Any],
    ) -> ConversationResolution:
        # "No, for <column> …" is a full replacement ask, not a patch of the prior
        # question. Keep the user's new wording so file-column analysis is preserved.
        remainder = _strip_repair_prefix(resolution.original_question)
        if _is_substantive_replacement(remainder, normalized):
            resolution.resolved_question = remainder
            resolution.repair_detected = True
            resolution.repair_type = "correction"
            resolution.changed_dimension = None
            resolution.inherited_context = _compact_context(context)
            remainder_norm = _normalize(remainder)
            resolution.metric = _infer_metric(remainder_norm)
            resolution.dimension = _infer_dimension(remainder_norm)
            resolution.date_range = _infer_date_range(remainder_norm) or resolution.date_range
            resolution.filters = _explicit_filters(remainder)
            resolution.cache_key = _cache_key(resolution)
            return resolution

        base = str(self.state.get("resolved_question") or self.state.get("original_question") or resolution.original_question)
        repaired = base
        changed: str | None = None
        repair_type = "correction"

        new_dimension = _infer_dimension(normalized)
        if new_dimension and new_dimension != self.state.get("dimension"):
            repaired = _replace_dimension(repaired, new_dimension)
            changed = new_dimension
            repair_type = "dimension_replacement"

        new_date = _infer_date_range(normalized)
        if new_date:
            repaired = _replace_date(repaired, new_date)
            changed = changed or "date_range"
            repair_type = "date_replacement"
            resolution.date_range = new_date

        if "all hubs" in normalized or "all warehouses" in normalized:
            repaired = _remove_filter_for_all_hubs(repaired)
            changed = "hub"
            repair_type = "scope_replacement"
            resolution.filters = {}

        if "remove atlanta" in normalized:
            repaired = _remove_named_filter(repaired, "Atlanta")
            changed = "hub"
            repair_type = "filter_removal"
            resolution.filters = {}

        if any(word in normalized for word in ("lowest", "worst", "bottom")):
            dimension = new_dimension or self.state.get("dimension") or "driver"
            repaired = _ranking_question(dimension, "lowest", self.state.get("metric"))
            changed = "ranking_direction"
            repair_type = "ranking_direction"

        if any(word in normalized for word in ("highest", "best", "top")):
            dimension = new_dimension or self.state.get("dimension") or "driver"
            repaired = _ranking_question(dimension, "highest", self.state.get("metric"))
            changed = "ranking_direction"
            repair_type = "ranking_direction"

        hub_name = _extract_named_hub(resolution.original_question)
        if hub_name:
            repaired = f"Show performance for {hub_name}"
            changed = "hub"
            repair_type = "hub_replacement"
            resolution.filters = {"hub": hub_name}
            resolution.dimension = "hub"

        resolution.resolved_question = _clean_question(repaired)
        resolution.repair_detected = True
        resolution.repair_type = repair_type
        resolution.changed_dimension = changed
        resolution.inherited_context = _compact_context(context)
        resolution.metric = _infer_metric(_normalize(resolution.resolved_question)) or self.state.get("metric")
        resolution.dimension = _infer_dimension(_normalize(resolution.resolved_question)) or self.state.get("dimension")
        resolution.cache_key = _cache_key(resolution)
        return resolution

    def _resolve_special_followup(
        self,
        resolution: ConversationResolution,
        normalized: str,
        context: dict[str, Any],
    ) -> ConversationResolution | None:
        if not self.turns:
            return None

        if normalized in {"why", "why?"} or normalized.startswith("why "):
            target = _target_from_extreme_reference(normalized, self.state)
            topic = target or self.state.get("summary") or self.state.get("resolved_question")
            resolution.resolved_question = _why_question(topic)
            resolution.intent_hint = "ROOT_CAUSE"
            resolution.dimension = self.state.get("dimension")
            resolution.metric = self.state.get("metric")
            resolution.filters = _entity_filter(target, self.state)
            resolution.inherited_context = _compact_context(context)
            resolution.cache_key = _cache_key(resolution)
            return resolution

        if _asks_for_recommendation(normalized):
            target = _target_from_state(self.state)
            resolution.resolved_question = _recommendation_question(target)
            resolution.intent_hint = "RECOMMENDATION"
            resolution.use_previous_analysis = True
            resolution.dimension = self.state.get("dimension")
            resolution.metric = self.state.get("metric")
            resolution.inherited_context = _compact_context(context)
            return resolution

        if _asks_for_explanation(normalized):
            resolution.resolved_question = f"Explain more about {self.state.get('resolved_question') or 'the previous analysis'}"
            resolution.intent_hint = "EXPLANATION"
            resolution.use_previous_analysis = True
            resolution.inherited_context = _compact_context(context)
            return resolution

        if _asks_for_drilldown(normalized):
            resolution.resolved_question = f"Show details for {self.state.get('resolved_question') or 'the previous result'}"
            resolution.intent_hint = "DRILLDOWN"
            resolution.use_previous_result = True
            resolution.inherited_context = _compact_context(context)
            return resolution

        if "which one" in normalized:
            dimension = self.state.get("dimension") or "hub"
            metric_label = "delayed pickups"
            if "delay" in normalized:
                metric_label = "delayed pickups"
            elif "fail" in normalized:
                metric_label = "failed pickups"
            elif "completion" in normalized or "performance" in normalized:
                metric_label = "completion rate"
            resolution.resolved_question = f"Which {dimension} has the most {metric_label}?"
            resolution.intent_hint = "SQL_QUERY"
            resolution.dimension = dimension
            resolution.metric = _infer_metric(normalized) or self.state.get("metric")
            resolution.inherited_context = _compact_context(context)
            resolution.cache_key = _cache_key(resolution)
            return resolution

        if "summarize everything" in normalized or "generate executive report" in normalized:
            resolution.resolved_question = "Generate executive summary of the current conversation analysis"
            resolution.intent_hint = "SUMMARY"
            resolution.use_previous_analysis = True
            resolution.inherited_context = _compact_context(context)
            return resolution

        if "which driver" in normalized and any(word in normalized for word in ("responsible", "accountable", "causing")):
            target = self.state.get("worst_entity") or self.state.get("best_entity") or "the previous issue"
            resolution.resolved_question = f"Which driver is responsible for {target}?"
            resolution.intent_hint = "SQL_QUERY"
            resolution.dimension = "driver"
            resolution.filters = _entity_filter(str(target), self.state)
            resolution.inherited_context = _compact_context(context)
            resolution.cache_key = _cache_key(resolution)
            return resolution

        return None

    def _resolve_contextual_followup(
        self,
        resolution: ConversationResolution,
        normalized: str,
        context: dict[str, Any],
    ) -> ConversationResolution:
        resolution.inherited_context = _compact_context(context)

        if normalized.startswith("compare") or " compare " in f" {normalized} ":
            period = _infer_date_range(normalized)
            topic = self.state.get("metric") or self.state.get("resolved_question") or "pickup performance"
            resolution.resolved_question = _comparison_question(topic, period, self.state)
            resolution.intent_hint = "COMPARISON"
            resolution.metric = self.state.get("metric")
            resolution.dimension = self.state.get("dimension")
            resolution.date_range = period
            resolution.cache_key = _cache_key(resolution)
            return resolution

        target = _target_from_extreme_reference(normalized, self.state)
        if target:
            resolution.resolved_question = _replace_reference_with_target(resolution.original_question, target)
            resolution.dimension = self.state.get("dimension")
            resolution.metric = self.state.get("metric")
            resolution.filters = _entity_filter(target, self.state)
            resolution.cache_key = _cache_key(resolution)
            return resolution

        if _contains_pronoun(normalized):
            entity = _primary_entity(self.state)
            if entity:
                resolution.resolved_question = _replace_pronouns(resolution.original_question, entity)
                resolution.entities = {"entity": entity}
                resolution.cache_key = _cache_key(resolution)
                return resolution

        if resolution.dimension is None:
            resolution.dimension = self.state.get("dimension")
        if resolution.metric is None:
            resolution.metric = self.state.get("metric")
        if resolution.date_range is None:
            resolution.date_range = self.state.get("date_range")
        resolution.resolved_question = _inherit_topic(resolution.original_question, self.state)
        resolution.cache_key = _cache_key(resolution)
        return resolution


def _empty_state() -> dict[str, Any]:
    return {
        "original_question": None,
        "resolved_question": None,
        "intent": None,
        "metric": None,
        "dimension": None,
        "entities": {},
        "filters": {},
        "generated_sql": None,
        "sql_rows": None,
        "business_findings": [],
        "kpi_summary": None,
        "ranking": [],
        "best_entity": None,
        "worst_entity": None,
        "root_cause": None,
        "recommendations": [],
        "summary": None,
        "date_range": None,
        "last_response": None,
    }


def _semantic_from_response(
    *,
    original_question: str,
    resolved_question: str,
    response: QueryResponse,
    intent: str | None,
    inherited_context: dict[str, Any],
) -> dict[str, Any]:
    rows = response.sql_rows or []
    dimension = response.analysis_dimension or _infer_dimension(_normalize(resolved_question)) or _dimension_from_rows(rows)
    metric = response.business_metric or _infer_metric(_normalize(resolved_question)) or _metric_from_rows(rows)
    ranking = _ranking_from_rows(rows, dimension)
    recommendations = _split_recommendations(response.recommendation)
    findings = _business_findings(response)
    entities = _entities_from_rows(rows)
    entities.update(response.planning_entities or {})
    return {
        "original_question": original_question,
        "resolved_question": resolved_question,
        "intent": intent or response.planning_intent or response.capability,
        "metric": metric,
        "dimension": dimension,
        "entities": entities,
        "filters": deepcopy(response.analysis_filters or {}),
        "generated_sql": response.generated_sql,
        "sql_rows": deepcopy(rows),
        "business_findings": findings,
        "kpi_summary": deepcopy(response.kpi_summary),
        "ranking": ranking,
        "best_entity": ranking[0] if ranking else _best_from_answer(response.answer),
        "worst_entity": ranking[-1] if ranking else _worst_from_answer(response.answer),
        "root_cause": deepcopy(response.root_cause),
        "recommendations": recommendations,
        "summary": response.answer,
        "date_range": response.date_range or _infer_date_range(_normalize(resolved_question)),
        "inherited_context": inherited_context,
    }


def _ranking_from_rows(rows: list[dict[str, Any]], dimension: str | None) -> list[str]:
    key = _dimension_key(dimension, rows)
    if not key:
        return []
    ranking: list[str] = []
    for row in rows:
        value = row.get(key)
        if value is not None and str(value) not in ranking:
            ranking.append(str(value))
    return ranking


def _dimension_key(dimension: str | None, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    preferred = {
        "hub": ("hub", "warehouse"),
        "driver": ("driver_name", "driver"),
        "customer": ("customer_name", "customer"),
        "date": ("pickup_date", "date", "period"),
        "status": ("status",),
    }
    keys = set(rows[0].keys())
    for key in preferred.get(dimension or "", ()):
        if key in keys:
            return key
    for key in ("hub", "driver_name", "customer_name", "pickup_date", "status", "period"):
        if key in keys:
            return key
    return None


def _dimension_from_rows(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    if "hub" in keys or "warehouse" in keys:
        return "hub"
    if "driver_name" in keys or "driver" in keys:
        return "driver"
    if "customer_name" in keys or "customer" in keys:
        return "customer"
    if "pickup_date" in keys or "period" in keys:
        return "date"
    if "status" in keys:
        return "status"
    return None


def _metric_from_rows(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for key in ("completion_rate", "delay_rate", "failure_rate", "package_volume", "pickup_count", "pickups"):
        if key in keys:
            return "pickup_count" if key in {"pickups", "pickup_count"} else key
    return None


def _entities_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    first = rows[0]
    entities = {}
    for key in ("hub", "driver_name", "customer_name", "pickup_date", "status"):
        if first.get(key):
            entities[key] = first[key]
    return entities


def _business_findings(response: QueryResponse) -> list[str]:
    findings: list[str] = []
    root_cause = response.root_cause or {}
    issue = root_cause.get("issue")
    if issue:
        findings.append(str(issue))
    findings.extend(str(item) for item in root_cause.get("main_causes") or [])
    anomaly = response.anomaly or {}
    if anomaly.get("is_anomaly"):
        findings.append(str(anomaly.get("message") or "Anomaly detected."))
    return findings


def _split_recommendations(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip(" -") for part in re.split(r";|\n", value) if part.strip(" -")]


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": context.get("intent"),
        "metric": context.get("metric"),
        "dimension": context.get("dimension"),
        "entities": context.get("entities", {}),
        "filters": context.get("filters", {}),
        "ranking": context.get("ranking", []),
        "best_entity": context.get("best_entity"),
        "worst_entity": context.get("worst_entity"),
        "date_range": context.get("date_range"),
        "summary": context.get("summary"),
    }


def _is_repair(normalized: str) -> bool:
    for phrase in _REPAIR_PHRASES:
        if len(phrase) <= 3:
            if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                return True
        elif phrase in normalized:
            return True
    return False


def _strip_repair_prefix(question: str) -> str:
    text = question.strip()
    patterns = (
        r"^(no|nope|wrong|actually|instead)[,.\s:-]+",
        r"^(i mean|i meant|not that|correction)[,.\s:-]+",
    )
    for pattern in patterns:
        updated = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
        if updated != text:
            return updated
    return text


def _is_substantive_replacement(remainder: str, normalized: str) -> bool:
    """True when the correction cue is followed by a complete new analysis request."""
    if not remainder or remainder.lower() == normalized:
        # Prefix strip failed; still treat long "no, …" asks as replacements.
        remainder = re.sub(r"^(no|nope|wrong|actually|instead)[,.\s:-]+", "", normalized, count=1).strip()
    tokens = [token for token in re.split(r"\s+", remainder) if token]
    if len(tokens) >= 6:
        return True
    if re.search(r"[\u4e00-\u9fff]", remainder):
        return True
    if re.search(r"\bcolumns?\b", remainder.lower()) or "列" in remainder or "字段" in remainder:
        return True
    return False


def _is_contextual_followup(normalized: str) -> bool:
    if normalized in {"why", "why?", "explain more", "show details"}:
        return True
    if len(normalized.split()) <= 4 and any(word in normalized for word in ("why", "compare", "details", "recommendations")):
        return True
    return any(
        phrase in normalized
        for phrase in (
            " it",
            " he",
            " she",
            " him",
            " her",
            " them",
            " those",
            " these",
            " that",
            " this",
            "which one",
            "worst",
            "best",
            "previous",
            "same ",
            "compare yesterday",
            "compare last week",
        )
    )


def _asks_for_recommendation(normalized: str) -> bool:
    return (
        "what should operations do" in normalized
        or "recommend" in normalized
        or "recommendation" in normalized
        or "what should we do" in normalized
    )


def _asks_for_explanation(normalized: str) -> bool:
    return normalized in {"explain more", "tell me more", "expand"} or "explain more" in normalized


def _asks_for_drilldown(normalized: str) -> bool:
    return any(phrase in normalized for phrase in ("show details", "details", "drill down", "show records", "expand"))


def _contains_pronoun(normalized: str) -> bool:
    return any(re.search(rf"\b{word}\b", normalized) for word in ("he", "she", "him", "her", "it", "they", "them"))


def _target_from_extreme_reference(normalized: str, state: dict[str, Any]) -> str | None:
    if any(word in normalized for word in ("worst", "lowest", "bottom")):
        return state.get("worst_entity")
    if any(word in normalized for word in ("best", "highest", "top")):
        return state.get("best_entity")
    return None


def _target_from_state(state: dict[str, Any]) -> str:
    return str(state.get("worst_entity") or state.get("best_entity") or state.get("resolved_question") or "the previous analysis")


def _primary_entity(state: dict[str, Any]) -> str | None:
    entities = state.get("entities") or {}
    for key in ("driver_name", "hub", "customer_name", "entity"):
        if entities.get(key):
            return str(entities[key])
    return state.get("best_entity") or state.get("worst_entity")


def _entity_filter(target: str | None, state: dict[str, Any]) -> dict[str, Any]:
    if not target:
        return {}
    dimension = state.get("dimension")
    if dimension == "hub":
        return {"hub": target}
    if dimension == "driver":
        return {"driver_name": target}
    if dimension == "customer":
        return {"customer_name": target}
    return {"entity": target}


def _why_question(topic: Any) -> str:
    topic_text = str(topic or "the previous analysis")
    if topic_text.lower().startswith("why"):
        return topic_text
    return f"Why is {topic_text} performing badly?"


def _recommendation_question(target: str) -> str:
    return f"What should operations do about {target}?"


def _comparison_question(topic: Any, period: str | None, state: dict[str, Any]) -> str:
    metric = str(topic or "pickup performance").replace("pickup_completion_rate", "pickup performance")
    if period:
        return f"Compare {period}'s {metric} with the previous {state.get('date_range') or 'period'}."
    return f"Compare {metric} with the previous period."


def _replace_reference_with_target(question: str, target: str) -> str:
    result = question
    result = re.sub(r"\b(the\s+)?worst\s+(hub|driver|customer|one)?\b", target, result, flags=re.IGNORECASE)
    result = re.sub(r"\b(the\s+)?best\s+(hub|driver|customer|one)?\b", target, result, flags=re.IGNORECASE)
    result = re.sub(r"\b(highest|lowest)\s+(hub|driver|customer|one)?\b", target, result, flags=re.IGNORECASE)
    return _clean_question(result)


def _replace_pronouns(question: str, entity: str) -> str:
    result = question
    for pronoun in ("he", "she", "him", "her", "it", "they", "them"):
        result = re.sub(rf"\b{pronoun}\b", entity, result, flags=re.IGNORECASE)
    return _clean_question(result)


def _inherit_topic(question: str, state: dict[str, Any]) -> str:
    normalized = _normalize(question)
    if normalized.startswith("compare"):
        return _comparison_question(state.get("metric") or "pickup performance", _infer_date_range(normalized), state)
    if normalized in {"why", "why?"}:
        return _why_question(state.get("worst_entity") or state.get("resolved_question"))
    return question


def _replace_dimension(question: str, new_dimension: str) -> str:
    result = question
    replacement = "hubs" if new_dimension == "hub" else f"{new_dimension}s"
    for dimension, words in _DIMENSION_WORDS.items():
        if dimension == new_dimension:
            continue
        for word in sorted(words, key=len, reverse=True):
            if re.search(rf"\b{re.escape(word)}\b", result, flags=re.IGNORECASE):
                return re.sub(rf"\b{re.escape(word)}\b", replacement, result, flags=re.IGNORECASE)
    return f"{question} by {replacement}"


def _replace_date(question: str, new_date: str) -> str:
    result = question
    for phrase in _DATE_PHRASES:
        if phrase == new_date:
            continue
        if re.search(rf"\b{re.escape(phrase)}\b", result, flags=re.IGNORECASE):
            return re.sub(rf"\b{re.escape(phrase)}\b", new_date, result, flags=re.IGNORECASE)
    return f"{question} for {new_date}"


def _remove_filter_for_all_hubs(question: str) -> str:
    result = re.sub(r"\b(?:for|in|at|only)\s+[A-Za-z][A-Za-z\s-]*\s+(?:hub|warehouse)\b", "for all hubs", question, flags=re.IGNORECASE)
    result = re.sub(r"\b[A-Z][A-Za-z]+\s+Hub\b", "all hubs", result)
    if "all hubs" not in result.lower():
        result = f"{result} for all hubs"
    return result


def _remove_named_filter(question: str, name: str) -> str:
    return _clean_question(re.sub(rf"\b(?:for|in|at|only)?\s*{re.escape(name)}\s+(?:hub|warehouse)\b", "", question, flags=re.IGNORECASE))


def _ranking_question(dimension: Any, direction: str, metric: Any) -> str:
    dimension_text = str(dimension or "driver")
    metric_text = "performing" if not metric or "performance" in str(metric) or "completion" in str(metric) else str(metric)
    singular = dimension_text.rstrip("s")
    return f"Show {direction} {metric_text} {singular}"


def _explicit_filters(question: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    hub_match = re.search(r"\b([A-Z]{3}|[A-Z][A-Za-z]+)\s+(?:Hub|hub|Warehouse|warehouse)\b", question)
    if hub_match:
        filters["hub"] = f"{hub_match.group(1)} Hub" if not hub_match.group(0).isupper() else hub_match.group(1)
    return filters


def _extract_entities(question: str) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    hub_filter = _explicit_filters(question).get("hub")
    if hub_filter:
        entities["hub"] = hub_filter
    driver_match = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", question)
    if driver_match and "hub" not in question.lower():
        entities["driver_name"] = driver_match.group(1)
    return entities


def _infer_dimension(normalized: str) -> str | None:
    for dimension, words in _DIMENSION_WORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in words):
            return dimension
    return None


def _infer_metric(normalized: str) -> str | None:
    if "completion" in normalized or "performance" in normalized or "performing" in normalized:
        return "pickup_completion_rate"
    if "delay" in normalized or "delayed" in normalized:
        return "delay_rate"
    if "fail" in normalized or "failed" in normalized or "failure" in normalized:
        return "failure_rate"
    if "package" in normalized:
        return "package_volume"
    if "pickup" in normalized or "volume" in normalized:
        return "pickup_count"
    return None


def _infer_date_range(normalized: str) -> str | None:
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
    if len(iso_dates) >= 2:
        return f"{iso_dates[0]} to {iso_dates[-1]}"
    if iso_dates:
        return iso_dates[0]
    for phrase in _DATE_PHRASES:
        if phrase in normalized:
            return phrase
    return None


def _cache_key(resolution: ConversationResolution) -> tuple[Any, ...]:
    return (
        resolution.intent_hint or "SQL_QUERY",
        _normalize(resolution.resolved_question),
        resolution.metric,
        resolution.dimension,
        tuple(sorted((resolution.filters or {}).items())),
        resolution.date_range,
    )


def _best_from_answer(answer: str | None) -> str | None:
    if not answer:
        return None
    match = re.search(r"([A-Z][A-Za-z ]+ Hub|[A-Z][a-z]+ [A-Z][a-z]+).*?(?:#1|highest|best)", answer)
    return match.group(1).strip() if match else None


def _worst_from_answer(answer: str | None) -> str | None:
    if not answer:
        return None
    match = re.search(r"([A-Z][A-Za-z ]+ Hub|[A-Z][a-z]+ [A-Z][a-z]+).*?(?:worst|lowest|bottom)", answer)
    return match.group(1).strip() if match else None


def _normalize(value: str) -> str:
    return value.lower().strip().rstrip(".?")


def _extract_named_hub(question: str) -> str | None:
    match = re.search(
        r"\b(?:no,?\s+i mean\s+|instead\s+)?([A-Z][A-Za-z]+(?:\s+Hub)?)\b",
        question,
        flags=re.IGNORECASE,
    )
    if not match or "hub" not in question.lower():
        return None
    value = match.group(1).strip()
    if value.lower().endswith("hub"):
        parts = value.split()
        return " ".join(part.capitalize() for part in parts[:-1]) + " Hub"
    return f"{value} Hub"


def _clean_question(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    has_question_mark = compact.endswith("?")
    compact = compact.strip(" .?")
    if compact.lower().startswith(("which ", "what ", "why ", "how ", "show ", "compare ", "find ")):
        compact = compact[:1].upper() + compact[1:]
    return compact + ("?" if has_question_mark else "")
