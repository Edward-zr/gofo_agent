"""Thin tool adapters over existing GOFO modules (no business-logic rewrite)."""

from __future__ import annotations

from typing import Any

from core.agent import GOFOAgent
from core.intent_router import RouteIntent
from core.logger import get_logger
from tools.conversation.resolver import resolution_to_dict

logger = get_logger("langgraph.tools")


def sync_state_from_agent(agent: GOFOAgent) -> dict[str, Any]:
    """Copy ConversationState attachment/memory fields into graph state patches."""
    snap = agent.state.snapshot()
    return {
        "attachment_active": bool(snap.get("attachment_active")),
        "last_attachment_file_types": list(snap.get("last_attachment_file_types") or []),
        "last_attachment_filenames": list(snap.get("last_attachment_filenames") or []),
        "previous_attachment_filenames": list(
            snap.get("previous_attachment_filenames") or []
        ),
        "last_active_sheet": snap.get("last_active_sheet"),
        "attachment_context_timestamp": snap.get("attachment_context_timestamp"),
        "conversation_summary": (snap.get("last_result_summary") or "")[:500],
    }


def run_conversation_repair(agent: GOFOAgent, *, question: str) -> dict[str, Any]:
    """Phase 1: ConversationResolver runs exactly once (LangGraph-owned)."""
    logger.info("Conversation Repair")
    logger.info("↓")
    resolution = agent.conversation.resolve(question)
    logger.info(
        "[Phase1] ConversationResolver executed once repair_detected=%s type=%s",
        resolution.repair_detected,
        resolution.repair_type,
    )
    return {
        "resolved_question": resolution.resolved_question or question,
        "repair_detected": bool(resolution.repair_detected),
        "repair_type": resolution.repair_type,
        "changed_dimension": resolution.changed_dimension,
        "conversation_resolution": resolution_to_dict(resolution),
        "reasoning_trace": [
            "Thought: conversation repair"
            if not resolution.repair_detected
            else (
                f"Thought: repair detected ({resolution.repair_type}) → "
                f"{resolution.resolved_question}"
            )
        ],
    }


def apply_attachment_routing(
    agent: GOFOAgent,
    route_decision: dict[str, Any],
    *,
    attachment_ids: list[str] | None = None,
) -> None:
    """Apply activate/detach from an already-computed route_decision (no re-route)."""
    if route_decision.get("detach_attachments"):
        agent.attachment_memory.deactivate()
        agent.state.attachment_active = False
    elif route_decision.get("use_attachments"):
        agent.attachment_memory.activate(attachment_ids or None)


def classify_route(
    agent: GOFOAgent,
    *,
    question: str,
    attachment_ids: list[str] | None = None,
    apply_attachments: bool = True,
) -> dict[str, Any]:
    """Phase 1: IntentRouter runs exactly once (LangGraph-owned)."""
    logger.info("Intent Router")
    logger.info("↓")
    snap = agent.state.snapshot()
    decision = agent.intent_router.route(
        question,
        snap,
        new_attachment_ids=attachment_ids or None,
        has_stored_attachments=bool(agent.attachment_memory.processed_contexts),
    )
    if apply_attachments:
        apply_attachment_routing(agent, decision, attachment_ids=attachment_ids)
    intent = str(decision.get("intent") or "")
    tool = _tool_for_intent(intent)
    thought = f"Thought: route={intent} tool={tool} confidence={decision.get('confidence')}"
    logger.info("[Phase1] IntentRouter executed once intent=%s tool=%s", intent, tool)
    logger.info("[ReAct] %s", thought)
    return {
        "intent": intent,
        "route_decision": dict(decision),
        "current_tool": tool,
        "pending_tools": [tool] if tool else [],
        "reasoning_trace": [thought],
        "attachment_active": bool(decision.get("attachment_active")),
        "attachments_routed": True,
        "pre_routed": True,
    }


def run_legacy_ask(
    agent: GOFOAgent,
    *,
    question: str,
    attachment_ids: list[str] | None = None,
    resolved_question: str | None = None,
    route_decision: dict[str, Any] | None = None,
    conversation_resolution: dict[str, Any] | None = None,
    repair_detected: bool = False,
    repair_type: str | None = None,
    changed_dimension: str | None = None,
    pre_routed: bool = False,
) -> dict[str, Any]:
    """Execute GOFOAgent.ask — pre-routed when LangGraph already repaired+routed."""
    if pre_routed and route_decision is not None:
        logger.info("GOFOAgent (pre-routed)")
        response = agent.ask(
            question,
            attachment_ids=attachment_ids or None,
            pre_routed=True,
            resolved_question=resolved_question,
            route_decision=route_decision,
            conversation_resolution=conversation_resolution,
            repair_detected=repair_detected,
            repair_type=repair_type,
            changed_dimension=changed_dimension,
        )
    else:
        response = agent.ask(question, attachment_ids=attachment_ids or None)
    return {
        "api_response": response,
        "final_answer": str(response.get("answer") or ""),
        "resolved_question": str(
            (response.get("analysis") or {}).get("resolved_question")
            or resolved_question
            or question
        ),
        **sync_state_from_agent(agent),
    }


def _tool_for_intent(intent: str) -> str:
    if intent in {RouteIntent.ATTACHMENT_ANALYSIS, RouteIntent.ATTACHMENT_VISUALIZATION}:
        return "ada"
    if intent == RouteIntent.SOP_QA:
        return "rag"
    if intent in {RouteIntent.SQL_ANALYTICS, RouteIntent.FOLLOW_UP}:
        return "sql"
    if intent in {RouteIntent.GENERAL_CHAT, RouteIntent.OPENAI_FALLBACK}:
        return "chat"
    if intent == RouteIntent.WAIT_FOR_UPLOAD:
        return "wait_upload"
    return "legacy"
