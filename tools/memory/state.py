"""Structured conversation state for repair-aware follow-up resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any

from core.models import QueryResponse

_REPAIR_PHRASES = (
    "actually",
    "i mean",
    "i meant",
    "no",
    "wrong",
    "not that",
    "instead",
)


@dataclass
class ConversationState:
    """Intent-level state that complements entity-focused ConversationMemory."""

    last_question: str | None = None
    last_resolved_question: str | None = None
    last_intent: str | None = None
    last_metric: str | None = None
    last_dimension: str | None = None
    last_sort_direction: str | None = None
    last_filters: dict[str, Any] = field(default_factory=dict)
    last_entities: dict[str, Any] = field(default_factory=dict)
    last_sql: str | None = None
    last_result_summary: str | None = None

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable copy of the current state."""
        return {
            "last_question": self.last_question,
            "last_resolved_question": self.last_resolved_question,
            "last_intent": self.last_intent,
            "last_metric": self.last_metric,
            "last_dimension": self.last_dimension,
            "last_sort_direction": self.last_sort_direction,
            "last_filters": deepcopy(self.last_filters),
            "last_entities": deepcopy(self.last_entities),
            "last_sql": self.last_sql,
            "last_result_summary": self.last_result_summary,
        }

    def resolve(self, question: str) -> dict[str, Any]:
        """Resolve state-based repairs and result references deterministically."""
        question = question.strip()
        normalized = question.lower()
        default = {
            "resolved_question": question,
            "repair_detected": False,
            "repair_type": None,
            "changed_dimension": None,
        }
        if not question:
            return default

        if _is_repair(normalized):
            repaired = self._resolve_repair(question, normalized)
            if repaired:
                return repaired

        worst_resolution = self._resolve_worst_reference(question, normalized)
        if worst_resolution:
            return worst_resolution

        return default

    def update(
        self,
        *,
        user_question: str,
        resolved_question: str,
        response: QueryResponse,
    ) -> None:
        """Update structured state from a completed turn."""
        self.last_question = user_question
        self.last_resolved_question = resolved_question
        self.last_intent = _infer_intent(resolved_question, response)
        self.last_metric = _infer_metric(resolved_question, response)
        self.last_dimension = _infer_dimension(resolved_question, response)
        self.last_sort_direction = _infer_sort_direction(resolved_question, response.generated_sql)
        self.last_filters = _infer_filters(resolved_question, response)
        self.last_entities = dict(response.planning_entities or {})
        self.last_sql = response.generated_sql
        self.last_result_summary = response.answer

    def _resolve_repair(self, question: str, normalized: str) -> dict[str, Any] | None:
        dimension = _mentioned_dimension(normalized) or self.last_dimension
        if _mentions_all_hubs(normalized):
            return {
                "resolved_question": "Show performance for all hubs",
                "repair_detected": True,
                "repair_type": "scope_replacement",
                "changed_dimension": "hub",
            }

        if "lowest" in normalized or "worst" in normalized:
            if dimension == "driver":
                return _repair_result("Show lowest performing driver", "ranking_direction")
            if dimension == "hub":
                metric = self.last_metric or "performance"
                return _repair_result(f"Show lowest {metric} hub", "ranking_direction")
            return _repair_result("Show lowest performing driver", "ranking_direction")

        if "highest" in normalized or "best" in normalized:
            if dimension == "driver":
                return _repair_result("Show highest performing driver", "ranking_direction")
            if dimension == "hub":
                metric = self.last_metric or "performance"
                return _repair_result(f"Show highest {metric} hub", "ranking_direction")
            return _repair_result("Show highest performing driver", "ranking_direction")

        mentioned_dimension = _mentioned_dimension(normalized)
        if mentioned_dimension and mentioned_dimension != self.last_dimension:
            return {
                "resolved_question": _replace_dimension(
                    self.last_resolved_question or self.last_question or question,
                    mentioned_dimension,
                ),
                "repair_detected": True,
                "repair_type": "dimension_replacement",
                "changed_dimension": mentioned_dimension,
            }

        return None

    def _resolve_worst_reference(self, question: str, normalized: str) -> dict[str, Any] | None:
        if "why" not in normalized or not any(word in normalized for word in ("worst", "lowest")):
            return None

        if "hub" in normalized or "one" in normalized or self.last_dimension == "hub":
            worst_hub = _extract_worst_hub(self.last_result_summary or "")
            if worst_hub:
                resolved = f"Why is {worst_hub} performing badly?"
            else:
                metric = self.last_metric or "completion_rate"
                resolved = f"Why is lowest {metric} hub performing badly?"
            return {
                "resolved_question": resolved,
                "repair_detected": False,
                "repair_type": None,
                "changed_dimension": None,
            }

        return None


def _repair_result(resolved_question: str, repair_type: str) -> dict[str, Any]:
    return {
        "resolved_question": resolved_question,
        "repair_detected": True,
        "repair_type": repair_type,
        "changed_dimension": "ranking_direction",
    }


def _is_repair(normalized: str) -> bool:
    return any(phrase in normalized for phrase in _REPAIR_PHRASES)


def _mentions_all_hubs(normalized: str) -> bool:
    return "all hubs" in normalized or "all warehouses" in normalized


def _mentioned_dimension(normalized: str) -> str | None:
    if re.search(r"\b(driver|drivers)\b", normalized):
        return "driver"
    if re.search(r"\b(hub|hubs|warehouse|warehouses)\b", normalized):
        return "hub"
    if re.search(r"\b(customer|customers)\b", normalized):
        return "customer"
    return None


def _replace_dimension(question: str, dimension: str) -> str:
    replacements = {
        "driver": ("customer", "customers", "hub", "hubs", "warehouse", "warehouses"),
        "hub": ("customer", "customers", "driver", "drivers"),
        "customer": ("hub", "hubs", "driver", "drivers"),
    }
    replacement = f"{dimension}s"
    resolved = question
    for old in replacements.get(dimension, ()):
        resolved = re.sub(rf"\b{old}\b", replacement, resolved, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", resolved).strip()


def _infer_intent(question: str, response: QueryResponse) -> str | None:
    text = f"{question} {response.planning_intent or ''}".lower()
    if any(word in text for word in ("rank", "ranking", "highest", "lowest", "best", "worst", "top", "bottom")):
        return "ranking"
    if "why" in text:
        return "root_cause"
    if "which hub" in text or "belong" in text:
        return "lookup"
    return response.planning_intent


def _infer_metric(question: str, response: QueryResponse) -> str | None:
    if response.business_metric:
        return response.business_metric
    entities = response.planning_entities or {}
    if entities.get("metric"):
        return str(entities["metric"])
    text = f"{question} {response.planning_intent or ''}".lower()
    if "completion" in text:
        return "completion_rate"
    if "performance" in text:
        return "performance"
    if "delay" in text:
        return "delay_rate"
    if "fail" in text:
        return "failure_rate"
    if "package" in text:
        return "package_volume"
    return None


def _infer_dimension(question: str, response: QueryResponse) -> str | None:
    if response.analysis_dimension:
        return response.analysis_dimension
    entities = response.planning_entities or {}
    if entities.get("dimension"):
        return str(entities["dimension"])
    text = f"{question} {response.generated_sql or ''}".lower()
    return _mentioned_dimension(text)


def _infer_sort_direction(question: str, sql: str | None) -> str | None:
    text = f"{question} {sql or ''}".lower()
    if any(word in text for word in ("lowest", "worst", "bottom", " asc")):
        return "ASC"
    if any(word in text for word in ("highest", "best", "top", " desc")):
        return "DESC"
    return None


def _infer_filters(question: str, response: QueryResponse) -> dict[str, Any]:
    filters = dict(response.analysis_filters or {})
    hub_match = re.search(r"\b([A-Z][A-Za-z]+|[A-Z]{3})\s+Hub\b", question)
    if hub_match:
        filters["hub"] = hub_match.group(1)
    return filters


def _extract_worst_hub(summary: str) -> str | None:
    patterns = (
        r"([A-Z][A-Za-z ]+ Hub)\s+(?:is\s+)?worst",
        r"([A-Z][A-Za-z ]+ Hub)\s+(?:is\s+)?(?:#?\d+.*)?(?:lowest|bottom)",
    )
    for pattern in patterns:
        match = re.search(pattern, summary, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None
