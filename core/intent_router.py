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
    "collection by tiktok",
    "tiktok collection",
    "collect by tiktok",
    "how do we handle",
    "what is the process",
    "according to sop",
    "responsible for",
    "responsibilities",
    "responsibility",
    "what should driver",
    "what should the driver",
    "what should drivers",
    "driver duty",
    "driver duties",
    "guideline",
    "guidelines",
    "playbook",
    "standard operating",
)

_SOP_ENTITY_TOKENS = (
    "cbt",
    "sop",
    "tiktok",
    "collection",
    "pickup procedure",
    "check-in",
    "check in",
)

_ATTACHMENT_REFERENCE_PHRASES = (
    "this file",
    "that file",
    "the uploaded file",
    "uploaded file",
    "data file",
    "the data file",
    "in the data file",
    "from the data file",
    "i just upload",
    "i just uploaded",
    "i uploaded",
    "upload you",
    "uploaded you",
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
    "in the file",
    "from this file",
    "from the file",
    "analyze it",
    "summarize it",
    "summarise it",
    "continue analysis",
    "generate charts",
    "what is in this",
    "what's in this",
    "contents of this",
)

_META_ATTACHMENT_PHRASES = (
    "which file are you reading",
    "what file are you reading",
    "which file are you using",
    "what file are you using",
    "which file are you analyzing",
    "what file are you analyzing",
    "file are you reading now",
    "file are you reading previously",
    "reading previously",
    "current file",
    "previous file",
    "which attachment",
    "what attachment",
    "which file did you",
    "what file did you",
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
    "address name",
    "which address is it",
    "give me the address",
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
        persisted_attachment = _has_persisted_attachment_metadata(state)
        attachment_context = _attachment_context_available(
            attachment_active=attachment_active,
            previous_route=previous_route,
            has_new_upload=has_new_upload,
            has_stored_attachments=has_stored_attachments,
            has_persisted_metadata=persisted_attachment,
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

        # Meta questions about which file is active must not trigger ADA summaries.
        if is_meta_attachment_question(normalized) and (
            has_stored_attachments or persisted_attachment or attachment_active or has_new_upload
        ):
            decision = _build_decision(
                intent=RouteIntent.GENERAL_CHAT,
                confidence=0.97,
                followup=False,
                target="Attachment Context",
                previous_route=previous_route,
                attachment_active=attachment_active or _route_is_attachment(previous_route),
                use_attachments=True,
                detach_attachments=False,
                chart_type=None,
                data_sources=[DataSource.ATTACHMENT.value],
            )
            _log_decision(decision)
            return decision

        # Prefer the uploaded file when the user is clearly analyzing it (column
        # language, "this file", new upload). Ops-DB asks like "rank all hubs"
        # still detach to SQL even during an attachment session.
        # Persisted metadata + prior attachment route keep the file session alive
        # when attachment_active flickers false between turns (no re-upload).
        # Explicit "in the data file I uploaded…" reactivates ADA even after SOP/SQL.
        explicit_upload_ref = _explicitly_references_uploaded_file(normalized)
        uploaded_file_analytics = bool(
            (has_stored_attachments or persisted_attachment)
            and (
                explicit_upload_ref
                or _mentions_file_column_analysis(normalized)
                or _mentions_uploaded_file_analytics(normalized)
                or _is_attachment_entity_followup(normalized, state)
            )
        )
        file_session = bool(
            has_new_upload
            or attachment_active
            or uploaded_file_analytics
            or (
                (has_stored_attachments or persisted_attachment)
                and _route_is_attachment(previous_route)
            )
        )
        if not attachment_active and persisted_attachment and file_session:
            logger.info(
                "[Router] Using persisted attachment context\n"
                "attachment_active=%s\n"
                "last_attachment_file_types=%s\n"
                "last_attachment_filenames=%s",
                attachment_active,
                state.get("last_attachment_file_types") or [],
                state.get("last_attachment_filenames") or [],
            )
        file_focused = bool(
            has_new_upload
            or references_attachment
            or explicit_upload_ref
            or uploaded_file_analytics
            or (attachment_context and _mentions_file_column_analysis(normalized))
        )
        # Inside a file session, only explicit warehouse/SOP/chat/knowledge asks detach.
        # Weak SQL-term matching ("delayed" + "pickups") must not steal ADA follow-ups.
        explicit_ops_detach = _is_explicit_ops_db_ask(normalized)
        # Pure SOP / glossary / knowledge asks leave the file unless the user
        # also references the upload (hybrid compare stays on ADA).
        knowledge_detach = _should_detach_for_knowledge_ask(
            normalized,
            file_focused=file_focused,
            uploaded_file_analytics=uploaded_file_analytics,
            references_attachment=references_attachment,
        )
        prefer_attachment = bool(
            file_focused
            or (
                file_session
                and not explicit_ops_detach
                and not knowledge_detach
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
        elif knowledge_detach and (
            _is_sop_question(normalized) or _is_definitional_knowledge_ask(normalized)
        ):
            # Independent capability transition: leave ADA for SOP/knowledge asks.
            decision = _build_decision(
                intent=RouteIntent.SOP_QA,
                confidence=0.94,
                followup=False,
                target="SOP Retriever",
                previous_route=previous_route,
                attachment_active=False,
                use_attachments=False,
                detach_attachments=attachment_active or _route_is_attachment(previous_route),
                chart_type=None,
                data_sources=[DataSource.RAG.value],
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
        elif prefer_attachment and _is_sop_question(normalized) and _attachment_rag_comparison(
            normalized
        ):
            # Hybrid only when the user is comparing the upload against SOP/policy.
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
        elif _is_sop_question(normalized) or _is_procedural_knowledge_ask(normalized):
            # SOP / procedural knowledge before SQL so "driver responsible" never
            # falls through to warehouse analytics clarification.
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
        elif _is_knowledge_default_ask(normalized):
            # Prefer SOP search over SQL for open-ended knowledge asks with no
            # ranking / metric / time signals (avoids date-range clarification).
            decision = _build_decision(
                intent=RouteIntent.SOP_QA,
                confidence=0.78,
                followup=False,
                target="SOP Retriever",
                previous_route=previous_route,
                attachment_active=False,
                use_attachments=False,
                detach_attachments=attachment_active,
                chart_type=None,
                data_sources=[DataSource.RAG.value],
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
    if _is_explicit_ops_db_ask(normalized):
        return True
    matched_terms = sum(1 for term in _SQL_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized))
    return matched_terms >= 2


def _is_explicit_ops_db_ask(normalized: str) -> bool:
    """High-confidence live warehouse asks that should detach from file ADA."""
    if any(phrase in normalized for phrase in _SQL_PHRASES):
        return True
    if re.search(r"\brank\b", normalized) and re.search(
        r"\b(hub|driver|customer|warehouse)s?\b", normalized
    ):
        return True
    return False


def _is_sop_question(normalized: str) -> bool:
    if any(phrase in normalized for phrase in _SOP_PHRASES):
        return True
    if _is_procedural_knowledge_ask(normalized):
        return True
    # "tell me more about CBT / the SOP / TikTok Collection"
    if re.search(r"\b(tell me more|more about|explain more)\b", normalized):
        if any(token in normalized for token in _SOP_ENTITY_TOKENS):
            return True
    return False


def _is_procedural_knowledge_ask(normalized: str) -> bool:
    """Role / duty / how-to asks that belong in SOP, not SQL analytics."""
    if re.search(
        r"\b(responsible for|responsibilities|responsibility|duties|duty)\b",
        normalized,
    ):
        return True
    if re.search(
        r"\bwhat should (the )?(driver|drivers|hub|agent|courier|operator)s?\b",
        normalized,
    ):
        return True
    if re.search(
        r"\bhow (should|do|does|to)\b.+\b(driver|drivers|pickup|handle|process)\b",
        normalized,
    ):
        return True
    if re.search(r"\bwhat (is|are) (the )?(driver|drivers).+\b(for|to)\b", normalized):
        return True
    return False


def _is_knowledge_default_ask(normalized: str) -> bool:
    """Open knowledge ask without analytics signals — prefer SOP over SQL default."""
    if _is_explicit_ops_db_ask(normalized) or _is_sql_analytics(normalized):
        return False
    if re.search(
        r"\b(rank|top\s+\d+|bottom\s+\d+|kpi|rate|volume|how many|count|today|"
        r"yesterday|this week|last week|chart|plot|dashboard)\b",
        normalized,
    ):
        return False
    if _is_definitional_knowledge_ask(normalized) or _is_procedural_knowledge_ask(normalized):
        return True
    if re.search(r"^\s*(what|how|why|who|when)\b", normalized) and len(normalized.split()) <= 14:
        return True
    return False


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


def is_meta_attachment_question(normalized: str) -> bool:
    """True for questions about which upload is active — not data analysis."""
    if any(phrase in normalized for phrase in _META_ATTACHMENT_PHRASES):
        return True
    if re.search(r"\bwhich file\b", normalized) and any(
        token in normalized for token in ("reading", "using", "analyzing", "analysing", "previous", "now")
    ):
        return True
    return False


def _explicitly_references_uploaded_file(normalized: str) -> bool:
    """True when the user points at an uploaded file (reactivates ADA after SOP/SQL)."""
    phrases = (
        "data file",
        "uploaded file",
        "the uploaded",
        "i just upload",
        "i just uploaded",
        "i uploaded",
        "upload you",
        "uploaded you",
        "in the file",
        "from the file",
        "in this file",
        "from this file",
        "this spreadsheet",
        "the spreadsheet",
        "this excel",
        "the excel",
        "this csv",
        "the csv",
    )
    return any(phrase in normalized for phrase in phrases)


def _mentions_uploaded_file_analytics(normalized: str) -> bool:
    """Analytics language that should stay on the upload, not the ops warehouse."""
    has_address = "address" in normalized or "地址" in normalized
    has_package = any(
        token in normalized for token in ("package", "packages", "volume", "row count", "how many")
    )
    if has_address and has_package:
        return True
    if has_address and any(
        token in normalized for token in ("most", "highest", "lowest", "name", "which")
    ):
        return True
    return False


def _is_attachment_entity_followup(normalized: str, state: dict[str, Any] | None = None) -> bool:
    """Follow-ups like 'which address is it / give me the address name' after file analysis."""
    _ = state  # caller already gates on stored/persisted attachment context
    if any(
        phrase in normalized
        for phrase in (
            "address name",
            "which address is it",
            "give me the address",
            "what address",
            "the address name",
            "full address",
        )
    ):
        return True
    if re.search(r"\bwhich address\b", normalized) and any(
        token in normalized for token in ("it", "that", "name", "this")
    ):
        return True
    return False


def _attachment_context_available(
    *,
    attachment_active: bool,
    previous_route: str | None,
    has_new_upload: bool,
    has_stored_attachments: bool,
    has_persisted_metadata: bool = False,
) -> bool:
    return bool(
        has_new_upload
        or has_stored_attachments
        or has_persisted_metadata
        or attachment_active
        or _route_is_attachment(previous_route)
    )


def _has_persisted_attachment_metadata(state: dict[str, Any]) -> bool:
    return bool(
        state.get("last_attachment_file_types")
        or state.get("last_attachment_filenames")
        or state.get("last_attachment")
        or state.get("last_active_sheet")
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
    """True when the user wants to compare an upload against SOP/policy."""
    has_policy = any(
        phrase in normalized for phrase in ("sop", "procedure", "policy", "cbt")
    )
    has_compare = any(
        phrase in normalized
        for phrase in (
            "compare",
            "versus",
            " vs ",
            "against",
            "according to",
            "follow our",
            "match the",
            "with the sop",
            "with sop",
        )
    )
    return bool(has_policy and has_compare) or "follow our" in normalized


def _should_detach_for_knowledge_ask(
    normalized: str,
    *,
    file_focused: bool,
    uploaded_file_analytics: bool,
    references_attachment: bool,
) -> bool:
    """Detach ADA when the user asks SOP/knowledge/chat with no file reference."""
    if file_focused or uploaded_file_analytics or references_attachment:
        return False
    if _is_sop_question(normalized):
        return True
    if _is_definitional_knowledge_ask(normalized):
        return True
    return False


def _is_definitional_knowledge_ask(normalized: str) -> bool:
    """Short definitional asks (what is X / define X) that should leave the file."""
    compact = normalized.strip().rstrip(".?!")
    tokens = compact.split()
    if len(tokens) > 16:
        return False
    if re.search(r"^\s*what\s+is\s+\w+", normalized):
        return True
    if re.search(r"^\s*what\s+does\s+\w+\s+mean", normalized):
        return True
    if re.search(r"^\s*define\s+\w+", normalized):
        return True
    if "tiktok" in normalized and "collection" in normalized:
        return True
    if _is_procedural_knowledge_ask(normalized):
        return True
    return False


def _explicit_chart_type(normalized: str) -> str | None:
    """Return an explicitly requested chart type, if any (never inherit previous)."""
    checks = (
        ("treemap", "treemap"),
        ("box plot", "box"),
        ("boxplot", "box"),
        ("heatmap", "heatmap"),
        ("heat map", "heatmap"),
        ("histogram", "histogram"),
        ("scatter", "scatter"),
        ("pie chart", "pie"),
        (" pie", "pie"),
        ("line chart", "line"),
        ("horizontal bar", "horizontal_bar"),
        ("bar chart", "bar"),
        ("bar graph", "bar"),
    )
    # Prefer longer / more specific phrases; also catch bare "use pie" / "pie please".
    if re.search(r"\bpie\b", normalized):
        return "pie"
    if re.search(r"\b(line chart|line graph|trend chart)\b", normalized) or (
        re.search(r"\bline\b", normalized) and "chart" in normalized
    ):
        return "line"
    if re.search(r"\bscatter\b", normalized):
        return "scatter"
    if re.search(r"\bhistogram\b", normalized):
        return "histogram"
    if "heat map" in normalized or "heatmap" in normalized:
        return "heatmap"
    if "box plot" in normalized or "boxplot" in normalized:
        return "box"
    if "treemap" in normalized or "tree map" in normalized:
        return "treemap"
    if "horizontal bar" in normalized:
        return "horizontal_bar"
    if re.search(r"\bbar chart\b|\bbar graph\b", normalized) or (
        re.search(r"\bbar\b", normalized) and "chart" in normalized
    ):
        return "bar"
    for phrase, chart_type in checks:
        if phrase in normalized:
            return chart_type
    return None


def _detect_chart_type(normalized: str, state: dict[str, Any]) -> str | None:
    """Detect chart type for this turn only — explicit requests always win."""
    if not _is_visualization_request(normalized):
        return None

    explicit = _explicit_chart_type(normalized)
    if explicit:
        return explicit

    # Heuristics from the current question only (do not inherit last_topic /
    # last_visualization — that locked conversations onto bar charts).
    if any(word in normalized for word in ("rank", "ranking", "top", "bottom", "worst", "best")):
        return "horizontal_bar"
    if any(word in normalized for word in ("trend", "over time", "time series", "daily", "weekly")):
        return "line"
    if any(word in normalized for word in ("distribution", "histogram", "spread")):
        return "histogram"
    if any(word in normalized for word in ("percent", "percentage", "share", "proportion")):
        return "pie"
    if any(word in normalized for word in ("correlation", "relationship", "versus")):
        return "scatter"
    if any(word in normalized for word in ("heatmap", "matrix", "heat map")):
        return "heatmap"
    if any(word in normalized for word in ("compare", "comparison", "by hub", "by driver", "category")):
        return "bar"
    return "bar"


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
