"""Business context inference for operational questions."""

from __future__ import annotations

import re
from typing import Any

_DIMENSIONS = {
    "driver": ("driver", "drivers"),
    "hub": ("hub", "hubs", "warehouse", "warehouses", "station", "stations", "ord"),
    "customer": ("customer", "customers"),
    "date": ("date", "today", "yesterday", "week", "month"),
    "status": ("status", "completed", "delayed", "failed"),
    "exception": ("exception", "reason", "reasons"),
}


def analyze_business_context(question: str) -> dict[str, Any]:
    """Infer metric, dimension, date range, and filters from a question."""
    normalized = question.lower()
    return {
        "metric": _infer_metric(normalized),
        "analysis_dimension": _infer_dimension(normalized),
        "date_range": _infer_date_range(normalized),
        "filters": _infer_filters(question),
    }


def is_operations_overview(question: str) -> bool:
    normalized = question.lower().strip()
    return normalized in {
        "how are operations?",
        "how are operations",
        "how is operations",
        "operations summary",
        "operational summary",
    } or "how are operations" in normalized


def is_why_question(question: str) -> bool:
    normalized = question.lower()
    return normalized.startswith("why") or "root cause" in normalized or "decrease" in normalized or "dropped" in normalized


def is_anomaly_question(question: str) -> bool:
    normalized = question.lower()
    return "abnormal" in normalized or "anomaly" in normalized or "anything unusual" in normalized


def _infer_metric(normalized: str) -> str | None:
    if "completion" in normalized or "performance" in normalized or "doing" in normalized:
        return "pickup_completion_rate"
    if "delay" in normalized or "delayed" in normalized:
        return "delay_rate"
    if "fail" in normalized or "failed" in normalized:
        return "failure_rate"
    if "package" in normalized:
        return "package_volume"
    if "pickup" in normalized or "volume" in normalized:
        return "pickup_count"
    return None


def _infer_dimension(normalized: str) -> str | None:
    for dimension, words in _DIMENSIONS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in words):
            return dimension
    return None


def _infer_date_range(normalized: str) -> str | None:
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
    if len(iso_dates) >= 2:
        return f"{iso_dates[0]} to {iso_dates[-1]}"
    if iso_dates:
        return iso_dates[0]
    for phrase in ("today", "yesterday", "this week", "last week", "this month", "last month"):
        if phrase in normalized:
            return phrase
    return None


def _infer_filters(question: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    hub_match = re.search(r"\b([A-Z]{3}|[A-Z][A-Za-z]+)\s+(?:hub|warehouse)\b", question)
    if hub_match:
        filters["hub"] = hub_match.group(1)
    elif re.search(r"\bORD\b", question):
        filters["hub"] = "ORD"
    return filters
