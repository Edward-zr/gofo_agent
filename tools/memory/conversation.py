"""Robust short-term conversation state for GOFO agent sessions."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from typing import Any
from uuid import uuid4

from core.models import QueryResponse
from tools.memory.summarizer import extract_entities, summarize_result_context

MAX_HISTORY = 10
MAX_RESULT_ROWS = 100


class ConversationMemory:
    """In-memory session state that keeps the latest conversation context."""

    def __init__(
        self,
        session_id: str | None = None,
        max_history: int = MAX_HISTORY,
    ) -> None:
        self.session_id = session_id or str(uuid4())
        self.max_history = max_history
        self.turns: list[dict[str, Any]] = []
        self.global_state: dict[str, Any] = _empty_global_state()

    def add_turn(
        self,
        *,
        user_question: str,
        resolved_question: str,
        response: QueryResponse,
        extracted_entities: dict[str, Any] | None = None,
    ) -> None:
        """Add a completed turn and refresh global conversation state."""
        entities = dict(response.planning_entities or {})
        answer_entities = _without_null_values(
            extracted_entities
            if extracted_entities is not None
            else _safe_extract_entities(resolved_question, response.answer or "")
        )
        merged_entities = _without_null_values({**entities, **answer_entities})

        metric = _infer_metric(resolved_question, response)
        dimension = _infer_dimension(resolved_question, response)
        date_range = _infer_date_range(resolved_question, response)
        filters = _infer_filters(merged_entities, dimension, date_range)

        self.global_state["previous_question"] = resolved_question
        self.global_state["previous_sql"] = response.generated_sql
        self.global_state["previous_answer"] = response.answer
        self.global_state["previous_rows"] = deepcopy(response.sql_rows)
        self.global_state["current_metric"] = metric
        self.global_state["analysis_dimension"] = dimension
        self.global_state["date_range"] = date_range
        self.global_state["filters"] = filters
        self.global_state["current_topic"] = _topic_from_response(response)
        self.global_state["current_intent"] = response.planning_intent
        self.global_state["active_entities"].update(merged_entities)
        self.global_state["active_filters"].update({**entities, **filters})
        self.global_state["active_metrics"].update(_extract_metrics(merged_entities, metric))

        if response.generated_sql:
            self.global_state["last_sql_context"] = {
                "intent": response.planning_intent,
                "sql": response.generated_sql,
                "question": resolved_question,
            }

        if response.sql_rows:
            self.global_state["last_result_context"] = _build_result_context(
                resolved_question,
                response,
            )

        self.turns.append(
            {
                "user_question": user_question,
                "resolved_question": resolved_question,
                "assistant_answer": response.answer,
                "capability": response.capability,
                "intent": response.planning_intent,
                "entities": merged_entities,
                "sql": response.generated_sql,
                "sql_rows": deepcopy(response.sql_rows),
                "result_summary": (
                    self.global_state["last_result_context"] or {}
                ).get("summary"),
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        )

        if len(self.turns) > self.max_history:
            self.turns = self.turns[-self.max_history :]

    def get_recent_history(self, limit: int = MAX_HISTORY) -> list[dict[str, Any]]:
        """Return recent turns, newest last."""
        return deepcopy(self.turns[-limit:])

    def get_current_state(self) -> dict[str, Any]:
        """Return current session-level state."""
        return deepcopy(self.global_state)

    def clear(self) -> None:
        """Clear all short-term state for this session."""
        self.turns.clear()
        self.global_state = _empty_global_state()


def _empty_global_state() -> dict[str, Any]:
    """Return the default in-memory conversation state."""
    return {
        "previous_question": None,
        "previous_sql": None,
        "previous_answer": None,
        "previous_rows": None,
        "current_metric": None,
        "analysis_dimension": None,
        "date_range": None,
        "filters": {},
        "current_topic": None,
        "current_intent": None,
        "active_entities": {},
        "active_metrics": {},
        "active_filters": {},
        "last_sql_context": None,
        "last_result_context": None,
    }


def _build_result_context(
    resolved_question: str,
    response: QueryResponse,
) -> dict[str, Any]:
    """Build a compact context object for the latest SQL result."""
    rows = (response.sql_rows or [])[:MAX_RESULT_ROWS]
    columns = list(rows[0].keys()) if rows else []
    return {
        "type": "sql_result",
        "description": summarize_result_context(
            question=resolved_question,
            intent=response.planning_intent,
            columns=columns,
            rows=rows,
        ),
        "intent": response.planning_intent,
        "columns": columns,
        "rows": rows,
        "generated_sql": response.generated_sql,
        "summary": response.answer,
    }


def _extract_metrics(entities: dict[str, Any], inferred_metric: str | None = None) -> dict[str, Any]:
    """Extract metric-like entities into active_metrics."""
    metric = entities.get("metric") or inferred_metric
    return {"metric": metric} if metric else {}


def _infer_metric(question: str, response: QueryResponse) -> str | None:
    """Infer the active operational metric from planner metadata and text."""
    if response.business_metric:
        return str(response.business_metric)
    entities = response.planning_entities or {}
    if entities.get("metric"):
        return str(entities["metric"])
    intent = (response.planning_intent or "").lower()
    normalized = question.lower()
    if "completion" in intent or "completion" in normalized or "performance" in normalized:
        return "pickup_completion_rate"
    if "delay" in intent or "delayed" in normalized:
        return "delay_rate"
    if "fail" in intent or "failed" in normalized:
        return "failure_rate"
    if "package" in intent or "package" in normalized:
        return "package_volume"
    if "pickup" in intent or "pickup" in normalized:
        return "pickup_count"
    return None


def _infer_dimension(question: str, response: QueryResponse) -> str | None:
    """Infer the primary analysis dimension."""
    entities = response.planning_entities or {}
    for key in ("analysis_dimension", "dimension"):
        if entities.get(key):
            return str(entities[key])
    text = f"{question} {response.generated_sql or ''}".lower()
    if re.search(r"\b(driver|drivers|driver_name)\b", text):
        return "driver"
    if re.search(r"\b(hub|hubs|warehouse|warehouses|d\.hub)\b", text):
        return "hub"
    if re.search(r"\b(customer|customers|customer_name)\b", text):
        return "customer"
    if re.search(r"\b(date|pickup_date|today|yesterday|month|week)\b", text):
        return "date"
    if re.search(r"\b(status|completed|delayed|failed)\b", text):
        return "status"
    if re.search(r"\b(exception|reason)\b", text):
        return "exception"
    return None


def _infer_date_range(question: str, response: QueryResponse) -> str | None:
    """Infer the active operational date range."""
    entities = response.planning_entities or {}
    for key in ("date_range", "date", "time_period"):
        if entities.get(key):
            return str(entities[key])
    text = f"{question} {response.rewritten_question or ''}".lower()
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if len(iso_dates) >= 2:
        return f"{iso_dates[0]} to {iso_dates[-1]}"
    if iso_dates:
        return iso_dates[0]
    for phrase in ("today", "yesterday", "this week", "last week", "this month", "last month"):
        if phrase in text:
            return phrase
    return None


def _infer_filters(
    entities: dict[str, Any],
    dimension: str | None,
    date_range: str | None,
) -> dict[str, Any]:
    """Build compact filters from known entities."""
    filters: dict[str, Any] = {}
    for key in ("hub", "warehouse", "driver", "driver_name", "customer", "customer_name", "status"):
        if entities.get(key):
            filters[key] = entities[key]
    if dimension:
        filters["analysis_dimension"] = dimension
    if date_range:
        filters["date_range"] = date_range
    return filters


def _safe_extract_entities(question: str, answer: str) -> dict[str, Any]:
    """Extract entities without letting memory updates block a response."""
    try:
        return extract_entities(question, answer)
    except Exception:
        return {}


def _topic_from_response(response: QueryResponse) -> str:
    """Infer a coarse topic from response metadata."""
    if response.planning_intent:
        return response.planning_intent.replace("_", " ")
    if response.capability == "sql":
        return "pickup analytics"
    if response.capability == "rag":
        return "sop knowledge"
    if response.capability == "multi":
        return "operations intelligence"
    if response.capability == "memory_analysis":
        return "result analysis"
    return "unknown"


def _without_null_values(entities: dict[str, Any]) -> dict[str, Any]:
    """Drop null/empty entity values before merging into memory."""
    return {
        key: value
        for key, value in entities.items()
        if value is not None and value != ""
    }
