"""Supervisor transition helpers for LangGraph capability switching."""

from __future__ import annotations

from typing import Any

from core.intent_router import RouteIntent
from core.logger import get_logger

logger = get_logger("langgraph.supervisor")

_CAPABILITY_FOR_INTENT = {
    RouteIntent.ATTACHMENT_ANALYSIS: "ADA",
    RouteIntent.ATTACHMENT_VISUALIZATION: "ADA",
    RouteIntent.SOP_QA: "RAG",
    RouteIntent.SQL_ANALYTICS: "SQL",
    RouteIntent.FOLLOW_UP: "SQL",
    RouteIntent.GENERAL_CHAT: "CHAT",
    RouteIntent.OPENAI_FALLBACK: "CHAT",
    RouteIntent.WAIT_FOR_UPLOAD: "WAIT_UPLOAD",
}


def capability_for_intent(intent: str) -> str:
    return _CAPABILITY_FOR_INTENT.get(intent, "LEGACY")


def log_capability_transition(
    *,
    previous_capability: str,
    intent: str,
    route_decision: dict[str, Any],
    previous_visualization: str | None,
) -> list[str]:
    """Log and return reasoning traces for independent capability transitions."""
    new_capability = capability_for_intent(intent)
    traces: list[str] = []

    logger.info("Current Capability: %s", previous_capability or "(none)")
    logger.info("↓")
    logger.info("Detected Intent: %s", intent)
    traces.append(f"Thought: capability {previous_capability or 'none'} → intent {intent}")

    if route_decision.get("detach_attachments"):
        logger.info("↓")
        logger.info("Detach Attachment")
        traces.append("Action: detach attachment session")

    if route_decision.get("use_attachments"):
        logger.info("↓")
        logger.info("Activate ADA")
        traces.append("Action: activate ADA attachment session")
    elif new_capability == "RAG":
        logger.info("↓")
        logger.info("Activate RAG")
        traces.append("Action: activate RAG")
    elif new_capability == "SQL":
        logger.info("↓")
        logger.info("Activate SQL")
        traces.append("Action: activate SQL")
    elif new_capability == "CHAT":
        logger.info("↓")
        logger.info("Activate Chat")
        traces.append("Action: activate Chat")

    requested = route_decision.get("chart_type")
    if requested:
        logger.info("Current Visualization: %s", previous_visualization or "(none)")
        logger.info("↓")
        logger.info("Requested Visualization: %s", requested)
        logger.info("↓")
        logger.info("Regenerate chart")
        traces.append(
            f"Action: visualization {previous_visualization or 'none'} → {requested} (regenerate)"
        )

    logger.info("↓")
    logger.info("Next Capability: %s", new_capability)
    traces.append(f"Observation: next capability={new_capability}")
    return traces
