"""Context-aware intent classification for the conversational pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class Intent(StrEnum):
    SQL_QUERY = "SQL_QUERY"
    SUMMARY = "SUMMARY"
    ROOT_CAUSE = "ROOT_CAUSE"
    RECOMMENDATION = "RECOMMENDATION"
    COMPARISON = "COMPARISON"
    TREND = "TREND"
    EXPLANATION = "EXPLANATION"
    DRILLDOWN = "DRILLDOWN"
    FOLLOWUP = "FOLLOWUP"
    CORRECTION = "CORRECTION"
    CLARIFICATION = "CLARIFICATION"
    GREETING = "GREETING"
    FILE_SUMMARY = "FILE_SUMMARY"
    FILE_ANALYSIS = "FILE_ANALYSIS"
    FILE_LOOKUP = "FILE_LOOKUP"
    FILE_COMPARISON = "FILE_COMPARISON"
    FILE_DRILLDOWN = "FILE_DRILLDOWN"
    IMAGE_ANALYSIS = "IMAGE_ANALYSIS"
    FILE_DATABASE_COMPARISON = "FILE_DATABASE_COMPARISON"
    FILE_RAG_COMPARISON = "FILE_RAG_COMPARISON"
    UNKNOWN = "UNKNOWN"


def classify_intent(
    question: str,
    *,
    resolved_question: str | None = None,
    conversation_state: dict[str, Any] | None = None,
    repair_detected: bool = False,
    intent_hint: str | None = None,
) -> Intent:
    """Classify user intent using both text and conversation state."""
    if intent_hint:
        try:
            return Intent(intent_hint)
        except ValueError:
            pass

    normalized = question.lower().strip()
    resolved = (resolved_question or question).lower().strip()
    state = conversation_state or {}
    has_context = bool(state.get("summary") or state.get("sql_rows") or state.get("ranking"))

    if repair_detected:
        return Intent.CORRECTION
    if normalized in {"hello", "hi", "hey", "thanks", "thank you"}:
        return Intent.GREETING
    if normalized in {"why", "why?"} or resolved.startswith("why ") or "root cause" in resolved:
        return Intent.ROOT_CAUSE
    if _asks_for_recommendation(normalized, resolved):
        return Intent.RECOMMENDATION
    if _asks_for_drilldown(normalized, resolved):
        return Intent.DRILLDOWN
    if _asks_for_explanation(normalized, resolved):
        return Intent.EXPLANATION
    if _asks_for_comparison(normalized, resolved):
        return Intent.COMPARISON
    if _asks_for_summary(normalized, resolved):
        return Intent.SUMMARY
    if _asks_for_trend(normalized, resolved):
        return Intent.TREND
    if has_context and _looks_like_followup(normalized):
        return Intent.FOLLOWUP
    if _looks_operational_sql(resolved):
        return Intent.SQL_QUERY
    if _looks_sop_question(resolved):
        return Intent.SQL_QUERY
    return Intent.UNKNOWN


def _asks_for_recommendation(normalized: str, resolved: str) -> bool:
    text = f"{normalized} {resolved}"
    return any(
        phrase in text
        for phrase in (
            "what should operations do",
            "what should we do",
            "recommend",
            "recommendation",
            "next action",
        )
    )


def _asks_for_drilldown(normalized: str, resolved: str) -> bool:
    text = f"{normalized} {resolved}"
    return any(
        phrase in text
        for phrase in (
            "show details",
            "show records",
            "drill down",
            "details",
            "expand",
        )
    )


def _asks_for_explanation(normalized: str, resolved: str) -> bool:
    text = f"{normalized} {resolved}"
    return any(phrase in text for phrase in ("explain more", "tell me more", "explain the result"))


def _asks_for_comparison(normalized: str, resolved: str) -> bool:
    text = f"{normalized} {resolved}"
    return any(word in text for word in ("compare", " vs ", "versus", "yesterday", "last week", "last month"))


def _asks_for_summary(normalized: str, resolved: str) -> bool:
    text = f"{normalized} {resolved}"
    return any(
        phrase in text
        for phrase in (
            "what happened today",
            "today's report",
            "todays report",
            "executive report",
            "executive summary",
            "summarize everything",
            "summarize today",
            "weekly report",
            "monthly report",
            "how are operations",
            "operational summary",
            "operations summary",
        )
    )


def _asks_for_trend(normalized: str, resolved: str) -> bool:
    text = f"{normalized} {resolved}"
    return "trend" in text or "over time" in text


def _looks_like_followup(normalized: str) -> bool:
    return (
        len(normalized.split()) <= 5
        or any(word in normalized for word in ("it", "them", "those", "that", "this", "same", "one", "previous"))
    )


def _looks_operational_sql(resolved: str) -> bool:
    return any(
        word in resolved
        for word in (
            "pickup",
            "hub",
            "driver",
            "customer",
            "performance",
            "completion",
            "delay",
            "failed",
            "package",
            "operations",
            "rank",
            "highest",
            "lowest",
        )
    )


def _looks_sop_question(resolved: str) -> bool:
    return any(word in resolved for word in ("sop", "policy", "procedure", "cbt"))
