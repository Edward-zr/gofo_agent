"""Production-style GOFO agent wrapper for CLI, API, and dashboard clients."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from agent import ask as ask_core
from core.error_handler import handle_error
from core.logger import get_logger
from core.models import QueryResponse
from tools.memory import (
    ConversationMemory,
    ConversationState,
    detect_repair,
    references_previous_result,
    resolve,
)

logger = get_logger("agent")


class GOFOAgent:
    """Stateful wrapper around the existing GOFO agent core."""

    def __init__(self) -> None:
        self.memory = ConversationMemory()
        self.state = ConversationState()

    def ask(self, question: str) -> dict[str, Any]:
        """Ask GOFO a question and return a dashboard-friendly response."""
        started = perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        logger.info("Incoming question")
        try:
            previous_state = self.state.snapshot()
            repair = detect_repair(question, self.memory)
            state_resolution = self.state.resolve(question)
            state_resolved_question = str(state_resolution.get("resolved_question") or question)
            state_handled = (
                state_resolution.get("repair_detected")
                or state_resolved_question != question
            )

            if state_handled:
                resolved_question = state_resolved_question
                repair = {
                    "is_repair": bool(state_resolution.get("repair_detected")),
                    "repair_type": state_resolution.get("repair_type"),
                    "corrected_question": resolved_question,
                    "changed_dimension": state_resolution.get("changed_dimension"),
                }
            else:
                question_for_resolver = (
                    str(repair["corrected_question"]) if repair.get("is_repair") else question
                )
                resolved_question = resolve(question_for_resolver, self.memory)
            logger.info("Resolved question: %s", resolved_question)
            result_context = None
            if references_previous_result(question) or references_previous_result(resolved_question):
                result_context = self.memory.get_current_state().get("last_result_context")

            response = ask_core(resolved_question, result_context=result_context)
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
            )
            new_state = self.state.snapshot()
            logger.info("Memory updated")
            logger.info("Previous state: %s", previous_state)
            logger.info("New state: %s", new_state)
            response.original_question = question
            response.resolved_question = resolved_question
            response.repair_detected = bool(repair.get("is_repair"))
            response.repair_type = repair.get("repair_type")
            response.changed_dimension = repair.get("changed_dimension")
            response.memory_history_count = len(self.memory.get_recent_history())
            response.memory_current_state = self.memory.get_current_state()
            response.memory_current_state["conversation_state"] = new_state
            response.memory_current_state["previous_conversation_state"] = previous_state
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
    recommendations = _recommendations(response.recommendation)
    memory_state = response.memory_current_state or {}
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
            "repair_detected": response.repair_detected,
            "resolved_question": response.resolved_question or response.question,
            "memory_turns": response.memory_history_count or 0,
            "previous_question": memory_state.get("previous_question"),
            "previous_answer": memory_state.get("previous_answer"),
            "current_entities": memory_state.get("active_entities", {}),
            "previous_state": memory_state.get("previous_conversation_state", {}),
            "new_state": memory_state.get("conversation_state", {}),
        },
        "recommendations": recommendations,
        "kpi": _kpi_payload(response),
        "raw": response.model_dump(),
    }


def _recommendations(recommendation: str | None) -> list[str]:
    if not recommendation:
        return []
    parts = [part.strip(" -") for part in recommendation.split(";")]
    return [part for part in parts if part]


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
