"""Dispatch routed requests to exactly one GOFO subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent import ask as ask_core
from core.intent_router import RouteDecision, RouteIntent
from core.logger import get_logger
from core.models import QueryResponse
from tools.analyzer.business_reasoner import analyze_business_response, answer_from_conversation
from tools.files.analyzer import analyze_attachments
from tools.files.source_router import DataSource
from tools.orchestration.models import SemanticAnalysis
from tools.planner.intent_classifier import Intent

logger = get_logger("route_dispatcher")


@dataclass
class DispatchContext:
    question: str
    resolved_question: str
    route: RouteDecision
    attachment_contexts: list[Any]
    file_context: dict[str, Any]
    attachment_ids: list[str]
    intent: Intent
    conversation_state: dict[str, Any]
    semantic: SemanticAnalysis
    result_context: dict[str, Any] | None = None
    cached_response: QueryResponse | None = None
    stored_attachments: list[Any] | None = None
    previous_filter: dict[str, Any] | None = None
    last_entity: str | None = None


class RouteDispatcher:
    """Execute one subsystem based on the centralized routing decision."""

    def dispatch(self, context: DispatchContext) -> QueryResponse | None:
        route = context.route
        intent_name = route["intent"]

        if intent_name == RouteIntent.WAIT_FOR_UPLOAD:
            return self._dispatch_wait_for_upload(context)

        if intent_name in {RouteIntent.GENERAL_CHAT, RouteIntent.OPENAI_FALLBACK}:
            return self._dispatch_general_chat(context)

        conversation_response = answer_from_conversation(
            question=context.question,
            intent=context.intent,
            conversation_state=context.conversation_state,
            resolved_question=context.resolved_question,
        )
        if conversation_response is not None:
            return conversation_response

        if intent_name in {RouteIntent.ATTACHMENT_ANALYSIS, RouteIntent.ATTACHMENT_VISUALIZATION}:
            return self._dispatch_attachment(context)

        if context.cached_response is not None:
            return context.cached_response

        if intent_name == RouteIntent.SOP_QA:
            return self._dispatch_sop(context)

        if intent_name in {RouteIntent.SQL_ANALYTICS, RouteIntent.FOLLOW_UP}:
            return self._dispatch_sql(context)

        return self._dispatch_sql(context)

    def _dispatch_general_chat(self, context: DispatchContext) -> QueryResponse:
        semantic = context.semantic.model_copy(
            update={
                "domain": "general_conversation",
                "capability": "conversation",
                "response_mode": "conversational",
                "requires_sql": False,
                "requires_rag": False,
            }
        )
        if not semantic.direct_reply:
            normalized = context.question.lower().strip()
            if normalized in {"hi", "hello", "hey"}:
                semantic.direct_reply = (
                    "Hello. I am the GOFO Operations Intelligence Analyst. "
                    "I can analyze pickup performance, investigate root causes, "
                    "compare hubs and drivers, and answer SOP questions."
                )
            elif "weather" in normalized:
                semantic.direct_reply = (
                    "I focus on GOFO operations intelligence rather than weather. "
                    "Ask me about pickups, hubs, drivers, delays, or SOP guidance."
                )
            elif "joke" in normalized:
                semantic.direct_reply = (
                    "Why did the delayed pickup cross the hub? "
                    "To get to the other dispatch window on time."
                )
            else:
                semantic.direct_reply = (
                    "I can help with GOFO operational analytics, SOP knowledge, "
                    "and uploaded report analysis."
                )
        return ask_core(
            context.resolved_question,
            semantic_context=semantic.model_dump(),
        )

    def _dispatch_wait_for_upload(self, context: DispatchContext) -> QueryResponse:
        from tools.files.analysis_intent import WAIT_FOR_UPLOAD_REPLY

        return QueryResponse(
            question=context.question,
            answer=WAIT_FOR_UPLOAD_REPLY,
            sources=[],
            capability="conversation",
            planning_capability="conversation",
            planning_intent="WAIT_FOR_UPLOAD",
            original_question=context.question,
            resolved_question=context.resolved_question,
        )

    def _dispatch_attachment(self, context: DispatchContext) -> QueryResponse | None:
        data_sources = _to_data_sources(context.route.get("data_sources") or [])
        if not context.attachment_contexts:
            return None
        response = analyze_attachments(
            question=context.question,
            resolved_question=context.resolved_question,
            contexts=context.attachment_contexts,
            file_context=context.file_context,
            data_sources=data_sources,
            intent=context.intent.value,
            stored_attachments=context.stored_attachments,
            previous_filter=context.previous_filter,
            last_entity=context.last_entity,
        )
        if response is None:
            return None
        # Prefer real chart payloads over text-only chart recommendations.
        if context.route.get("chart_type") and not response.charts:
            response.answer = _append_chart_guidance(response.answer or "", context.route["chart_type"])
        return response

    def _dispatch_sop(self, context: DispatchContext) -> QueryResponse:
        semantic = context.semantic.model_copy(
            update={
                "domain": "sop_knowledge",
                "capability": "rag",
                "requires_sql": False,
                "requires_rag": True,
            }
        )
        response = ask_core(
            context.resolved_question,
            result_context=context.result_context,
            semantic_context=semantic.model_dump(),
        )
        if context.route.get("followup") and response.answer:
            response.answer = _append_openai_explanation(response.answer)
        return response

    def _dispatch_sql(self, context: DispatchContext) -> QueryResponse:
        semantic = context.semantic.model_copy(
            update={
                "domain": "data_analytics",
                "capability": "sql",
                "requires_sql": True,
                "requires_rag": False,
            }
        )
        response = ask_core(
            context.resolved_question,
            result_context=context.result_context,
            attachments=context.attachment_ids or None,
            file_context=context.file_context or None,
            semantic_context=semantic.model_dump(),
        )
        if response.capability == "sql" and response.sql_rows:
            response = analyze_business_response(response, context.resolved_question)
        if context.route.get("chart_type") and response.sql_rows:
            response.answer = _append_chart_guidance(response.answer or "", context.route["chart_type"])
        return response


def _to_data_sources(values: list[str]) -> list[DataSource]:
    resolved: list[DataSource] = []
    for value in values:
        try:
            resolved.append(DataSource(value))
        except ValueError:
            continue
    return resolved or [DataSource.ATTACHMENT]


def _append_chart_guidance(answer: str, chart_type: str) -> str:
    chart_labels = {
        "horizontal_bar": "Horizontal Bar Chart",
        "line": "Line Chart",
        "bar": "Bar Chart",
        "histogram": "Histogram",
        "pie": "Pie Chart",
        "scatter": "Scatter Plot",
        "heatmap": "Heatmap",
    }
    label = chart_labels.get(chart_type, "Chart")
    if label.lower() in answer.lower():
        return answer
    return f"{answer.rstrip()}\n\nRecommended visualization: {label}."


def _append_openai_explanation(sop_answer: str) -> str:
    if "General knowledge" in sop_answer:
        return sop_answer
    return (
        f"According to SOP:\n{sop_answer}\n\n"
        "General knowledge:\n"
        "Use the SOP guidance above as the operational source of truth, "
        "then apply local hub or dispatch context when executing the process."
    )
