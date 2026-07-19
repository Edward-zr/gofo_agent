"""Production-style GOFO agent wrapper for CLI, API, and dashboard clients."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from core.error_handler import handle_error
from core.intent_router import IntentRouter
from core.logger import get_logger
from core.models import QueryResponse
from core.route_dispatcher import DispatchContext, RouteDispatcher
from tools.context import build_context
from tools.conversation import ConversationResolver
from tools.files.attachment_memory import AttachmentMemory
from tools.files.service import AttachmentService
from tools.memory import (
    ConversationMemory,
    ConversationState,
    references_previous_result,
    resolve,
)
from tools.orchestration import analyze_request
from tools.planner.intent_classifier import classify_intent

logger = get_logger("agent")


class GOFOAgent:
    """Stateful wrapper around the existing GOFO agent core."""

    def __init__(self) -> None:
        self.memory = ConversationMemory()
        self.state = ConversationState()
        self.conversation = ConversationResolver()
        self.attachment_memory = AttachmentMemory()
        self.attachment_service = AttachmentService()
        self.intent_router = IntentRouter()
        self.route_dispatcher = RouteDispatcher()

    def ask(self, question: str, attachment_ids: list[str] | None = None) -> dict[str, Any]:
        """Ask GOFO a question and return a dashboard-friendly response."""
        started = perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        logger.info("Incoming question")
        attachment_ids = attachment_ids or []
        try:
            if attachment_ids:
                processed = self.attachment_service.get_many_processed(attachment_ids, question=question)
                self.attachment_memory.register_contexts(processed)

            previous_state = self.state.snapshot()
            previous_semantic_state = self.conversation.snapshot()
            route_decision = self.intent_router.route(
                question,
                previous_state,
                new_attachment_ids=attachment_ids or None,
                has_stored_attachments=bool(self.attachment_memory.processed_contexts),
            )

            if route_decision["detach_attachments"]:
                self.attachment_memory.deactivate()
                self.state.attachment_active = False
            elif route_decision["use_attachments"]:
                self.attachment_memory.activate(attachment_ids or None)

            attachment_contexts, file_context = self.attachment_memory.resolve_for_question(
                question,
                attachment_ids=attachment_ids or None,
                use_attachments=route_decision["use_attachments"],
            )

            conversation_resolution = self.conversation.resolve(question)
            semantic = analyze_request(
                question,
                resolved_question=conversation_resolution.resolved_question,
                conversation_state=previous_semantic_state,
                repair_detected=conversation_resolution.repair_detected,
                has_attachments=route_decision["use_attachments"] and bool(attachment_contexts),
            )
            intent = classify_intent(
                question,
                resolved_question=conversation_resolution.resolved_question,
                conversation_state=previous_semantic_state,
                repair_detected=conversation_resolution.repair_detected,
                intent_hint=_routing_intent_hint(route_decision, question)
                or conversation_resolution.intent_hint
                or semantic.intent_hint(),
            )
            context = build_context(conversation_resolution, intent, previous_semantic_state)

            resolved_question = context.resolved_question
            if semantic.confidence >= 0.65 and semantic.resolved_question:
                resolved_question = semantic.resolved_question
            if (
                resolved_question == question
                and self.memory.get_recent_history()
                and not conversation_resolution.repair_detected
                and not conversation_resolution.intent_hint
                and not route_decision["use_attachments"]
            ):
                resolved_question = resolve(question, self.memory)
            logger.info("Resolved question: %s", resolved_question)
            logger.info(
                "Semantic analysis: domain=%s sub_intent=%s capability=%s confidence=%s",
                semantic.domain,
                semantic.sub_intent,
                semantic.capability,
                semantic.confidence,
            )

            result_context = None
            if (
                context.use_previous_result
                or references_previous_result(question)
                or references_previous_result(resolved_question)
                or route_decision["followup"]
            ):
                result_context = self.memory.get_current_state().get("last_result_context")

            cached = self.conversation.cached_response(conversation_resolution)
            sql_cache_hit = cached is not None
            last_analysis = self.attachment_memory.last_analysis or {}
            response = self.route_dispatcher.dispatch(
                DispatchContext(
                    question=question,
                    resolved_question=resolved_question,
                    route=route_decision,
                    attachment_contexts=attachment_contexts,
                    file_context=file_context,
                    attachment_ids=attachment_ids,
                    intent=intent,
                    conversation_state=previous_semantic_state,
                    semantic=semantic,
                    result_context=result_context,
                    cached_response=cached,
                    stored_attachments=self.attachment_memory.get_active_stored()
                    or list(self.attachment_memory.stored_attachments.values()),
                    previous_filter=self.attachment_memory.active_filters or None,
                    last_entity=(last_analysis.get("last_entity") or None),
                )
            )
            if response is None:
                raise RuntimeError("Route dispatcher returned no response.")

            logger.info("Selected capability: %s", response.capability)
            if response.generated_sql:
                logger.info("Generated SQL: %s", response.generated_sql)

            self.memory.add_turn(
                user_question=question,
                resolved_question=resolved_question,
                response=response,
            )
            self.state.update(
                user_question=question,
                resolved_question=resolved_question,
                response=response,
                route_decision=route_decision,
            )
            new_state = self.state.snapshot()
            new_semantic_state = self.conversation.update(
                original_question=question,
                resolved_question=resolved_question,
                response=response,
                intent=intent.value,
                inherited_context=context.inherited_context,
                cache_key=context.cache_key,
            )
            logger.info("Memory updated")
            logger.info("Previous state: %s", previous_state)
            logger.info("New state: %s", new_state)
            response.original_question = question
            response.resolved_question = resolved_question
            response.repair_detected = conversation_resolution.repair_detected
            response.repair_type = conversation_resolution.repair_type
            response.changed_dimension = conversation_resolution.changed_dimension
            response.classifier_intent = intent.value
            response.inherited_context = context.inherited_context
            response.semantic_domain = semantic.domain
            response.semantic_sub_intent = semantic.sub_intent
            response.response_mode = semantic.response_mode
            response.semantic_reasoning = semantic.reasoning
            response.sql_cache_hit = sql_cache_hit
            response.memory_updated = True
            response.attachment_ids = [context.attachment_id for context in attachment_contexts] or []
            response.attachment_filenames = [context.filename for context in attachment_contexts] or []
            response.data_sources = route_decision.get("data_sources") or []
            analysis_summary = dict(response.file_context_summary or {})
            response.file_context_summary = {
                **_compact_file_context(file_context),
                **{key: value for key, value in analysis_summary.items() if value is not None},
            }
            self.attachment_memory.update_analysis(
                {
                    "file_context": file_context,
                    "data_sources": response.data_sources,
                    "answer": response.answer,
                    "route": route_decision,
                    "active_filter": (response.analysis_filters or {})
                    or analysis_summary.get("active_filter")
                    or {},
                    "last_entity": analysis_summary.get("last_entity")
                    or last_analysis.get("last_entity"),
                    "charts": response.charts or [],
                }
            )
            response.memory_history_count = len(self.memory.get_recent_history())
            response.memory_current_state = self.memory.get_current_state()
            response.memory_current_state["conversation_state"] = new_state
            response.memory_current_state["previous_conversation_state"] = previous_state
            response.memory_current_state["semantic_conversation"] = new_semantic_state
            response.memory_current_state["previous_semantic_conversation"] = previous_semantic_state
            response.memory_current_state["attachment_memory"] = self.attachment_memory.snapshot()
            response.memory_current_state["route_decision"] = route_decision
            response.last_result_context = self.memory.get_current_state().get("last_result_context")
            return format_agent_response(response)
        except Exception as exc:
            logger.exception("Agent error: %s", handle_error(exc))
            raise
        finally:
            elapsed_ms = int((perf_counter() - started) * 1000)
            logger.info("Execution time: %sms", elapsed_ms)


def format_agent_response(response: QueryResponse) -> dict[str, Any]:
    """Convert QueryResponse into the API/dashboard contract."""
    memory_state = response.memory_current_state or {}
    route_decision = memory_state.get("route_decision") or {}
    return {
        "answer": response.answer or "",
        "sources": [source.model_dump() for source in response.sources],
        "sql": response.generated_sql or "",
        "data": response.sql_rows or [],
        "analysis": {
            "capability": response.capability,
            "root_cause": response.root_cause or {},
            "anomaly": response.anomaly or {},
            "long_memory_matches": response.long_memory_matches or [],
            "previous_issues_found": response.previous_issues_found or [],
            "metric": response.business_metric,
            "dimension": response.analysis_dimension,
            "date_range": response.date_range,
            "filters": response.analysis_filters or {},
            "intent": response.classifier_intent or response.planning_intent,
            "inherited_context": response.inherited_context or {},
            "business_findings": response.business_findings or [],
            "memory_updated": bool(response.memory_updated),
            "sql_cache_hit": bool(response.sql_cache_hit),
            "repair_detected": response.repair_detected,
            "resolved_question": response.resolved_question or response.question,
            "memory_turns": response.memory_history_count or 0,
            "previous_question": memory_state.get("previous_question"),
            "previous_answer": memory_state.get("previous_answer"),
            "current_entities": memory_state.get("active_entities", {}),
            "previous_state": memory_state.get("previous_conversation_state", {}),
            "new_state": memory_state.get("conversation_state", {}),
            "semantic_state": memory_state.get("semantic_conversation", {}),
            "attachment_ids": response.attachment_ids or [],
            "attachment_filenames": response.attachment_filenames or [],
            "data_sources": response.data_sources or [],
            "file_context_summary": response.file_context_summary or {},
            "attachment_memory": memory_state.get("attachment_memory", {}),
            "semantic_domain": response.semantic_domain,
            "semantic_sub_intent": response.semantic_sub_intent,
            "response_mode": response.response_mode,
            "semantic_reasoning": response.semantic_reasoning,
            "suggested_next_steps": response.suggested_next_steps or [],
            "route_intent": route_decision.get("intent"),
            "route_confidence": route_decision.get("confidence"),
            "route_followup": route_decision.get("followup"),
            "route_handler": route_decision.get("handler"),
            "attachment_active": route_decision.get("attachment_active"),
            "chart_type": route_decision.get("chart_type"),
            "charts": response.charts or [],
        },
        "charts": response.charts or [],
        "recommendations": _merge_recommendations(response),
        "kpi": _kpi_payload(response),
        "raw": response.model_dump(),
    }


def _recommendations(recommendation: str | None) -> list[str]:
    if not recommendation:
        return []
    parts = [part.strip(" -") for part in recommendation.split(";")]
    return [part for part in parts if part]


def _merge_recommendations(response: QueryResponse) -> list[str]:
    recommendations = _recommendations(response.recommendation)
    next_steps = list(response.suggested_next_steps or [])
    merged: list[str] = []
    for item in recommendations + next_steps:
        if item and item not in merged:
            merged.append(item)
    return merged


def _kpi_payload(response: QueryResponse) -> dict[str, Any]:
    kpi = dict(response.kpi_summary or {})
    rows = response.sql_rows or []
    if rows:
        first = rows[0]
        for source_key, target_key in (
            ("pickup_count", "pickup_count"),
            ("completion_rate", "completion_rate"),
            ("failed_pickups", "failed"),
            ("delayed_pickups", "delayed"),
            ("package_volume", "package_volume"),
        ):
            if source_key in first:
                kpi[target_key] = first[source_key]
    return kpi


def _routing_intent_hint(route_decision: dict[str, Any], question: str) -> str | None:
    intent_name = route_decision.get("intent")
    normalized = question.lower()
    if intent_name == "ATTACHMENT_VISUALIZATION":
        return "FILE_VISUALIZATION"
    if intent_name not in {"ATTACHMENT_ANALYSIS", "ATTACHMENT_VISUALIZATION"}:
        return None
    if any(phrase in normalized for phrase in ("summarize", "summary", "what is in")):
        return "FILE_SUMMARY"
    if "compare" in normalized:
        return "FILE_COMPARISON"
    if any(phrase in normalized for phrase in ("visualize", "plot", "chart", "graph")):
        return "FILE_VISUALIZATION"
    if any(phrase in normalized for phrase in ("what should operations do", "recommend")):
        return "FILE_RECOMMENDATION"
    return "FILE_ANALYSIS"


def _compact_file_context(file_context: dict[str, Any]) -> dict[str, Any]:
    if not file_context:
        return {}
    return {
        "attachment_ids": file_context.get("attachment_ids", []),
        "filenames": file_context.get("filenames", []),
        "file_types": file_context.get("file_types", []),
        "summaries": file_context.get("summaries", []),
        "statistics": file_context.get("statistics", []),
        "source_references": file_context.get("source_references", []),
    }
