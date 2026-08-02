"""Shared LangGraph AgentState for GOFO orchestration."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict


def _merge_lists(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Reducer: append right items onto left."""
    return list(left or []) + list(right or [])


def _merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer: shallow-merge dictionaries (right wins)."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict):
    """Strongly typed graph state shared across LangGraph nodes.

    Attachment metadata mirrors ConversationState so ADA follow-ups survive
    turns via checkpoints even when ``attachment_active`` flickers false.
    """

    # Conversation
    messages: Annotated[list[dict[str, Any]], _merge_lists]
    user_question: str
    resolved_question: str
    session_id: str
    attachment_ids: list[str]

    # Intent understanding (repair + route)
    repair_detected: bool
    repair_type: str | None
    changed_dimension: str | None
    conversation_resolution: dict[str, Any]
    intent: str
    route_decision: dict[str, Any]
    current_tool: str
    pending_tools: list[str]
    attachments_routed: bool
    reasoning_trace: Annotated[list[str], _merge_lists]
    pre_routed: bool

    # Intent facets (independent — do not overwrite each other)
    primary_intent: str
    secondary_intent: str | None
    domain: str
    is_continuation: bool
    current_capability: str
    previous_capability: str
    current_topic: str
    visualization_request: str | None
    previous_visualization: str | None
    attachment_session_active: bool

    # Task decomposition / planning / verification
    task_spec: dict[str, Any]
    intent_classification: dict[str, Any]
    execution_plan: dict[str, Any]
    planner_output: dict[str, Any]
    needs_clarification: bool
    verification_report: dict[str, Any]
    verification_passed: bool
    replan_count: int

    # Frontier orchestrator
    frontier_decision: dict[str, Any]
    frontier_problem_kind: str
    not_meaningful: bool

    # Memory / attachments (persisted across turns via checkpointer)
    conversation_summary: str
    attachment_active: bool
    last_attachment_file_types: list[str]
    last_attachment_filenames: list[str]
    previous_attachment_filenames: list[str]
    last_active_sheet: str | None
    attachment_context_timestamp: str | None

    # Tool histories
    tool_history: Annotated[list[dict[str, Any]], _merge_lists]
    tool_outputs: Annotated[dict[str, Any], _merge_dicts]
    sql_history: Annotated[list[dict[str, Any]], _merge_lists]
    rag_history: Annotated[list[dict[str, Any]], _merge_lists]
    sop_history: Annotated[list[dict[str, Any]], _merge_lists]

    # Response
    api_response: dict[str, Any]
    final_answer: str
    error: str | None

    # Internal (not for end users)
    node_timings_ms: Annotated[dict[str, float], _merge_dicts]
    debug: NotRequired[dict[str, Any]]


def empty_agent_state(
    *,
    question: str = "",
    session_id: str = "default",
    attachment_ids: list[str] | None = None,
) -> AgentState:
    """Return a blank AgentState for a new turn (or cold start)."""
    return AgentState(
        messages=[],
        user_question=question,
        resolved_question=question,
        session_id=session_id,
        attachment_ids=list(attachment_ids or []),
        repair_detected=False,
        repair_type=None,
        changed_dimension=None,
        conversation_resolution={},
        intent="",
        route_decision={},
        current_tool="",
        pending_tools=[],
        attachments_routed=False,
        reasoning_trace=[],
        pre_routed=False,
        primary_intent="",
        secondary_intent=None,
        domain="",
        is_continuation=False,
        current_capability="",
        previous_capability="",
        current_topic="",
        visualization_request=None,
        previous_visualization=None,
        attachment_session_active=False,
        task_spec={},
        intent_classification={},
        execution_plan={},
        planner_output={},
        needs_clarification=False,
        verification_report={},
        verification_passed=False,
        replan_count=0,
        frontier_decision={},
        frontier_problem_kind="",
        not_meaningful=False,
        conversation_summary="",
        attachment_active=False,
        last_attachment_file_types=[],
        last_attachment_filenames=[],
        previous_attachment_filenames=[],
        last_active_sheet=None,
        attachment_context_timestamp=None,
        tool_history=[],
        tool_outputs={},
        sql_history=[],
        rag_history=[],
        sop_history=[],
        api_response={},
        final_answer="",
        error=None,
        node_timings_ms={},
    )


# Keep operator import referenced for TypedDict Annotated patterns elsewhere.
_ = operator
