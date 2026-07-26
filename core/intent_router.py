"""Centralized intent routing for the GOFO Operations Intelligence Agent."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from core.logger import get_logger
from tools.files.source_router import DataSource, plan_data_sources

logger = get_logger("intent_router")


class RouteIntent:
    GENERAL_CHAT = "GENERAL_CHAT"
    SOP_QA = "SOP_QA"
    SQL_ANALYTICS = "SQL_ANALYTICS"
    ATTACHMENT_ANALYSIS = "ATTACHMENT_ANALYSIS"
    ATTACHMENT_VISUALIZATION = "ATTACHMENT_VISUALIZATION"
    WAIT_FOR_UPLOAD = "WAIT_FOR_UPLOAD"
    FOLLOW_UP = "FOLLOW_UP"
    OPENAI_FALLBACK = "OPENAI_FALLBACK"


class RouteDecision(TypedDict):
    intent: str
    confidence: float
    followup: bool
    target: str
    handler: str
    previous_route: str | None
    attachment_active: bool
    use_attachments: bool
    detach_attachments: bool
    chart_type: str | None
    data_sources: list[str]


_GENERAL_CHAT_PHRASES = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "who are you",
    "tell me a joke",
    "weather",
    "translate this",
    "grammar",
    "meaning of",
    "what does this word mean",
)

_SQL_PHRASES = (
    "rank all",
    "rank hubs",
    "rank drivers",
    "rank customers",
    "all hubs",
    "all drivers",
    "all customers",
    "all warehouses",
    "operational database",
    "today's database",
    "today database",
    "from the database",
    "in the database",
    "live data",
    "current operations",
    "how are operations",
    "operations today",
    "pickups today",
    "today's pickups",
    "today pickup",
    "yesterday",
    "last week",
    "last month",
    "root cause",
    "kpi",
    "completion rate",
    "delay rate",
    "failure rate",
)

_SQL_TERMS = (
    "drivers",
    "customers",
    "hubs",
    "warehouse",
    "performance",
    "ranking",
    "database",
    "pickup",
    "pickups",
    "status",
    "volume",
    "completion",
    "delay",
    "delayed",
    "packages",
    "today",
    "metrics",
    "analytics",
    "statistics",
    "comparison",
    "trend",
    "recommendation",
    "anomaly",
    "throughput",
)

_SOP_PHRASES = (
    "workflow",
    "process",
    "policy",
    "sop",
    "procedure",
    "driver procedure",
    "pickup procedure",
    "exception handling",
    "warehouse process",
    "dispatch process",
    "cbt",
    "how do we handle",
    "what is the process",
    "according to sop",
)

_ATTACHMENT_REFERENCE_PHRASES = (
    "this file",
    "that file",
    "the uploaded file",
    "uploaded file",
    "inspect the file",
    "inspect this file",
    "inspect file",
    "look at the file",
    "review the file",
    "the spreadsheet",
    "spreadsheet",
    "excel file",
    "the excel",
    "csv file",
    "the csv",
    "sheet",
    "attachment",
    "the image",
    "the screenshot",
    "the pdf",
    "uploaded report",
    "those rows",
    "these records",
    "this report",
    "that report",
    "in this file",
    "from this file",
    "analyze it",
    "summarize it",
    "summarise it",
    "continue analysis",
    "generate charts",
    "what is in this",
    "what's in this",
    "contents of this",
)

_ATTACHMENT_FOLLOWUP_PHRASES = (
    "visualize it",
    "plot it",
    "chart it",
    "graph it",
    "analyze it",
    "summarize it",
    "summarise it",
    "continue",
    "continue analysis",
    "what should operations do",
    "biggest problems",
    "biggest risks",
    "which hub is worst",
    "which driver is worst",
    "which hub is best",
    "show his records",
    "show her records",
    "show their records",
    "compare these",
    "these reports",
    "these two",
    "which hub changed",
    "changed the most",
)

_ATTACHMENT_CONTEXT_FOLLOWUP_PHRASES = _ATTACHMENT_FOLLOWUP_PHRASES + (
    "belong to",
)

_FOLLOWUP_PHRASES = (
    "visualize it",
    "plot it",
    "chart it",
    "graph it",
    "why?",
    "why is",
    "why are",
    "explain more",
    "show details",
    "continue",
    "break it down",
    "compare them",
    "what about",
    "which one",
    "what should operations do",
    "why is that",
    "what caused it",
    "tell me more",
    "go on",
    "elaborate",
    "drill down",
    "more detail",
    "generate chart",
    "generate charts",
    "make a chart",
    "create a chart",
    "which hub changed",
    "changed the most",
)

_VISUALIZATION_PHRASES = (
    "visualize",
    "visualise",
    "plot",
    "chart",
    "dashboard",
    "graph",
    "bar chart",
    "line chart",
    "pie chart",
    "heatmap",
    "histogram",
    "scatter",
    "box plot",
    "distribution",
    "trend chart",
    "trend over",
    "over time",
    "time series",
)


class IntentRouter:
    """Determine which subsystem handles each user message."""

    def route(
        self,
        question: str,
        conversation_state: dict[str, Any] | None = None,
        *,
        new_attachment_ids: list[str] | None = None,
        has_stored_attachments: bool = False,
    ) -> RouteDecision:
        """Return a single routing decision for the current message."""
        question = question.strip()
        state = conversation_state or {}
        normalized = question.lower().strip()
        previous_route = state.get("last_route")
        attachment_active = bool(state.get("attachment_active"))
        has_new_upload = bool(new_attachment_ids)

        followup = is_followup(question, state)
        chart_type = _detect_chart_type(normalized, state)
        attachment_context = _attachment_context_available(
            attachment_active=attachment_active,
            previous_route=previous_route,
            has_new_upload=has_new_upload,
            has_stored_attachments=has_stored_attachments,
        )
        references_attachment = _references_attachment(
            normalized,
            attachment_context=attachment_context,
        )

        if _is_wait_for_upload(normalized) and not has_new_upload:
            decision = _build_decision(
                intent=RouteIntent.WAIT_FOR_UPLOAD,
                confidence=0.99,
                followup=False,
                target="Wait For Upload",
                previous_route=previous_route,
                attachment_active=False,
                use_attachments=False,
                detach_attachments=False,
                chart_type=None,
                data_sources=[],
            )
            _log_decision(decision)
            return decision

        # Prefer the uploaded file when the user is clearly analyzing it (column
        # language, "this file", new upload). Ops-DB asks like "rank all hubs"
        # still detach to SQL even during an attachment session.
        file_session = bool(
            has_new_upload
            or attachment_active
            or (has_stored_attachments and _route_is_attachment(previous_route))
        )
        file_focused = bool(
            has_new_upload
            or references_attachment
            or (attachment_context and _mentions_file_column_analysis(normalized))
        )
        prefer_attachment = bool(
            file_focused
            or (
                file_session
                and not _is_sql_analytics(normalized)
                and not (
                    _is_sop_question(normalized)
                    and not _attachment_rag_comparison(normalized)
                )
            )
        )

        if _is_general_chat(normalized):
            decision = _build_decision(
                intent=RouteIntent.GENERAL_CHAT,
                confidence=0.98,
                followup=False,
                target="OpenAI Chat",
                previous_route=previous_route,
                attachment_active=False,
                use_attachments=False,
                detach_attachments=attachment_active,
                chart_type=None,
                data_sources=[],
            )
        elif prefer_attachment and _attachment_sql_comparison(normalized):
            decision = _attachment_decision(
                normalized,
                state=state,
                previous_route=previous_route,
                followup=followup and not has_new_upload,
                confidence=0.94,
                hybrid_sql=True,
                hybrid_rag=False,
                chart_type=chart_type if _is_visualization_request(normalized) else None,
            )
        elif prefer_attachment and _is_sop_question(normalized) and _attachment_rag_comparison(normalized):
            decision = _attachment_decision(
                normalized,
                state=state,
                previous_route=previous_route,
                followup=followup and not has_new_upload,
                confidence=0.94,
                hybrid_sql=False,
                hybrid_rag=True,
                chart_type=None,
            )
        elif prefer_attachment:
            if _is_visualization_request(normalized):
                decision = _build_decision(
                    intent=RouteIntent.ATTACHMENT_VISUALIZATION,
                    confidence=0.93,
                    followup=followup and not has_new_upload,
                    target="Attachment Visualizer",
                    previous_route=previous_route,
                    attachment_active=True,
                    use_attachments=True,
                    detach_attachments=False,
                    chart_type=chart_type,
                    data_sources=_attachment_data_sources(
                        normalized,
                        file_types=state.get("last_attachment_file_types") or [],
                        hybrid_sql=False,
                        hybrid_rag=False,
                    ),
                )
            else:
                decision = _attachment_decision(
                    normalized,
                    state=state,
                    previous_route=previous_route,
                    followup=followup and not has_new_upload,
                    confidence=0.93 if has_new_upload or attachment_active or file_focused else 0.88,
                    hybrid_sql=False,
                    hybrid_rag=_attachment_rag_comparison(normalized),
                    chart_type=chart_type if _is_visualization_request(normalized) else None,
                )
        elif _is_sql_analytics(normalized):
            decision = _build_decision(
                intent=RouteIntent.SQL_ANALYTICS,
                confidence=0.96,
                followup=followup,
                target="SQL Planner",
                previous_route=previous_route,
                attachment_active=False,
                use_attachments=False,
                detach_attachments=True,
                chart_type=chart_type if _is_visualization_request(normalized) else None,
                data_sources=[DataSource.SQLITE.value],
            )
        elif _is_sop_question(normalized):
            decision = _build_decision(
                intent=RouteIntent.SOP_QA,
                confidence=0.94,
                followup=followup,
                target="SOP Retriever",
                previous_route=previous_route,
                attachment_active=False,
                use_attachments=False,
                detach_attachments=attachment_active,
                chart_type=None,
                data_sources=[DataSource.RAG.value],
            )
        elif followup and previous_route:
            inherited = _inherit_route(previous_route, normalized, state)
            decision = _build_decision(
                intent=inherited["intent"],
                confidence=inherited["confidence"],
                followup=True,
                target=inherited["target"],
                previous_route=previous_route,
                attachment_active=inherited["attachment_active"],
                use_attachments=inherited["use_attachments"],
                detach_attachments=inherited["detach_attachments"],
                chart_type=chart_type if _is_visualization_request(normalized) else None,
                data_sources=inherited["data_sources"],
            )
        elif _is_visualization_request(normalized):
            inherited = _inherit_route(previous_route or RouteIntent.SQL_ANALYTICS, normalized, state)
            decision = _build_decision(
                intent=inherited["intent"],
                confidence=0.88,
                followup=bool(previous_route),
                target=inherited["target"],
                previous_route=previous_route,
                attachment_active=inherited["attachment_active"],
                use_attachments=inherited["use_attachments"],
                detach_attachments=inherited["detach_attachments"],
                chart_type=chart_type,
                data_sources=inherited["data_sources"],
            )
        elif has_stored_attachments and not attachment_active:
            decision = _build_decision(
                intent=RouteIntent.OPENAI_FALLBACK,
                confidence=0.55,
                followup=False,
                target="OpenAI Fallback",
                previous_route=previous_route,
                attachment_active=False,
                use_attachments=False,
                detach_attachments=False,
                chart_type=None,
                data_sources=[DataSource.SQLITE.value],
            )
        else:
            decision = _build_decision(
                intent=RouteIntent.SQL_ANALYTICS,
                confidence=0.72,
                followup=False,
                target="SQL Planner",
                previous_route=previous_route,
                attachment_active=False,
                use_attachments=False,
                detach_attachments=attachment_active,
                chart_type=None,
                data_sources=[DataSource.SQLITE.value],
            )

        _log_decision(decision)
        return decision


def is_followup(question: str, conversation_state: dict[str, Any] | None = None) -> bool:
    """Detect short follow-up messages that should inherit the previous route."""
    normalized = question.lower().strip()
    state = conversation_state or {}
    if not state.get("last_route"):
        return False

    if any(phrase in normalized for phrase in _FOLLOWUP_PHRASES):
        return True

    short = len(normalized.split()) <= 6
    pronoun_followup = short and any(
        token in normalized.split()
        for token in ("it", "that", "them", "this", "those", "he", "she", "his", "her")
    )
    if pronoun_followup:
        return True

    if short and normalized in {"why", "why?", "continue", "details", "more"}:
        return True

    return False


def _build_decision(
    *,
    intent: str,
    confidence: float,
    followup: bool,
    target: str,
    previous_route: str | None,
    attachment_active: bool,
    use_attachments: bool,
    detach_attachments: bool,
    chart_type: str | None,
    data_sources: list[str],
) -> RouteDecision:
    return RouteDecision(
        intent=intent,
        confidence=confidence,
        followup=followup,
        target=target,
        handler=target,
        previous_route=previous_route,
        attachment_active=attachment_active,
        use_attachments=use_attachments,
        detach_attachments=detach_attachments,
        chart_type=chart_type,
        data_sources=data_sources,
    )


def _is_wait_for_upload(normalized: str) -> bool:
    phrases = (
        "i will upload",
        "i'll upload",
        "i am going to upload",
        "i'm going to upload",
        "going to upload",
        "about to upload",
        "want you to analyze an excel",
        "want you to analyze a csv",
        "want you to analyze a file",
        "i want to upload",
        "let me upload",
        "i will send a file",
        "i'll send a file",
    )
    return any(phrase in normalized for phrase in phrases)


def _is_general_chat(normalized: str) -> bool:
    if normalized in {"hi", "hello", "hey", "thanks", "thank you"}:
        return True
    for phrase in _GENERAL_CHAT_PHRASES:
        if len(phrase) <= 4:
            if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                return True
        elif phrase in normalized:
            return True
    return False


def _is_sql_analytics(normalized: str) -> bool:
    if any(phrase in normalized for phrase in _SQL_PHRASES):
        return True
    if re.search(r"\brank\b", normalized) and re.search(r"\b(hub|driver|customer|warehouse)s?\b", normalized):
        return True
    matched_terms = sum(1 for term in _SQL_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized))
    return matched_terms >= 2


def _is_sop_question(normalized: str) -> bool:
    return any(phrase in normalized for phrase in _SOP_PHRASES)


def _references_attachment(normalized: str, *, attachment_context: bool = False) -> bool:
    if any(phrase in normalized for phrase in _ATTACHMENT_REFERENCE_PHRASES):
        return True
    if not attachment_context:
        return False
    if "the file" in normalized or "this upload" in normalized or "uploaded" in normalized:
        return True
    if any(phrase in normalized for phrase in _ATTACHMENT_CONTEXT_FOLLOWUP_PHRASES):
        return True
    if _mentions_file_column_analysis(normalized):
        return True
    if re.search(r"\bwhich (hub|driver|record)\b", normalized):
        return True
    if re.search(r"\bexcel\b|\bcsv\b|\bxlsx\b|\bspreadsheet\b|\bpdf\b|\bscreenshot\b|\bimage\b", normalized):
        return True
    return False


def _mentions_file_column_analysis(normalized: str) -> bool:
    """True when the user is asking about columns/values inside an uploaded file."""
    if re.search(r"\bcolumns?\b", normalized):
        return True
    if "基于" in normalized and ("列" in normalized or "字段" in normalized):
        return True
    if "列" in normalized or "字段" in normalized:
        return True
    if any(
        phrase in normalized
        for phrase in (
            "for each",
            "based on each",
            "based on the",
            "group by",
            "grouped by",
            "distribution of",
            "how many packages for each",
            "how many for each",
            "count by",
            "breakdown by",
            "by address",
            "each address",
            "each addresses",
        )
    ):
        return True
    return False


def _attachment_context_available(
    *,
    attachment_active: bool,
    previous_route: str | None,
    has_new_upload: bool,
    has_stored_attachments: bool,
) -> bool:
    return bool(
        has_new_upload
        or has_stored_attachments
        or attachment_active
        or _route_is_attachment(previous_route)
    )


def _attachment_analytical_followup(normalized: str) -> bool:
    return bool(
        re.search(r"\bwhich (hub|driver|record)\b", normalized)
        or "changed the most" in normalized
        or "biggest problems" in normalized
        or "biggest risks" in normalized
    )


def _attachment_decision(
    normalized: str,
    *,
    state: dict[str, Any],
    previous_route: str | None,
    followup: bool,
    confidence: float,
    hybrid_sql: bool,
    hybrid_rag: bool,
    chart_type: str | None,
) -> RouteDecision:
    return _build_decision(
        intent=RouteIntent.ATTACHMENT_ANALYSIS,
        confidence=confidence,
        followup=followup,
        target="Attachment Analyzer",
        previous_route=previous_route,
        attachment_active=True,
        use_attachments=True,
        detach_attachments=False,
        chart_type=chart_type,
        data_sources=_attachment_data_sources(
            normalized,
            file_types=state.get("last_attachment_file_types") or [],
            hybrid_sql=hybrid_sql,
            hybrid_rag=hybrid_rag,
        ),
    )


def _is_visualization_request(normalized: str) -> bool:
    return any(phrase in normalized for phrase in _VISUALIZATION_PHRASES)


def _attachment_sql_comparison(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in ("database", "sqlite", "operational database", "today's database", "compare with today")
    )


def _attachment_rag_comparison(normalized: str) -> bool:
    return any(phrase in normalized for phrase in ("sop", "procedure", "policy", "follow our"))


def _attachment_data_sources(
    normalized: str,
    *,
    file_types: list[str],
    hybrid_sql: bool,
    hybrid_rag: bool,
) -> list[str]:
    if hybrid_sql:
        return [DataSource.ATTACHMENT_AND_SQL.value]
    if hybrid_rag:
        return [DataSource.ATTACHMENT_AND_RAG.value]
    sources = plan_data_sources(
        normalized,
        has_attachments=True,
        has_active_attachment_memory=True,
        file_types=file_types,
    )
    return [source.value for source in sources] or [DataSource.ATTACHMENT.value]


def _route_is_attachment(route: str | None) -> bool:
    return route in {RouteIntent.ATTACHMENT_ANALYSIS, RouteIntent.ATTACHMENT_VISUALIZATION}


def _inherit_route(
    previous_route: str,
    normalized: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    if _route_is_attachment(previous_route):
        return {
            "intent": RouteIntent.ATTACHMENT_ANALYSIS,
            "confidence": 0.88,
            "target": "Attachment Analyzer",
            "attachment_active": True,
            "use_attachments": True,
            "detach_attachments": False,
            "data_sources": _attachment_data_sources(
                normalized,
                file_types=state.get("last_attachment_file_types") or [],
                hybrid_sql=_attachment_sql_comparison(normalized),
                hybrid_rag=_attachment_rag_comparison(normalized),
            ),
        }
    if previous_route == RouteIntent.SOP_QA:
        return {
            "intent": RouteIntent.SOP_QA,
            "confidence": 0.86,
            "target": "SOP Retriever",
            "attachment_active": False,
            "use_attachments": False,
            "detach_attachments": False,
            "data_sources": [DataSource.RAG.value],
        }
    if previous_route in {RouteIntent.GENERAL_CHAT, RouteIntent.OPENAI_FALLBACK}:
        return {
            "intent": RouteIntent.GENERAL_CHAT,
            "confidence": 0.8,
            "target": "OpenAI Chat",
            "attachment_active": False,
            "use_attachments": False,
            "detach_attachments": False,
            "data_sources": [],
        }
    return {
        "intent": RouteIntent.SQL_ANALYTICS,
        "confidence": 0.9,
        "target": "SQL Planner",
        "attachment_active": False,
        "use_attachments": False,
        "detach_attachments": True,
        "data_sources": [DataSource.SQLITE.value],
    }


def _detect_chart_type(normalized: str, state: dict[str, Any]) -> str | None:
    if not _is_visualization_request(normalized):
        return None

    last_topic = (state.get("last_topic") or state.get("last_intent") or "").lower()
    combined = f"{normalized} {last_topic}"

    if any(word in combined for word in ("rank", "ranking", "top", "bottom", "worst", "best")):
        return "horizontal_bar"
    if any(word in combined for word in ("trend", "over time", "time series", "daily", "weekly")):
        return "line"
    if any(word in combined for word in ("distribution", "histogram", "spread")):
        return "histogram"
    if any(word in combined for word in ("percent", "percentage", "share", "proportion")):
        return "pie"
    if any(word in combined for word in ("correlation", "relationship", "versus")):
        return "scatter"
    if any(word in combined for word in ("heatmap", "matrix", "heat map")):
        return "heatmap"
    if any(word in combined for word in ("compare", "comparison", "by hub", "by driver", "category")):
        return "bar"
    return "bar"


def _log_decision(decision: RouteDecision) -> None:
    logger.info("----------------------------------")
    logger.info("Intent: %s", decision["intent"])
    logger.info("Confidence: %s", decision["confidence"])
    logger.info("Follow-up: %s", decision["followup"])
    logger.info("Previous Route: %s", decision["previous_route"])
    logger.info("Attachment Active: %s", decision["attachment_active"])
    logger.info("Selected Handler: %s", decision["handler"])
    if decision.get("chart_type"):
        logger.info("Chart Type: %s", decision["chart_type"])
    logger.info("----------------------------------")
