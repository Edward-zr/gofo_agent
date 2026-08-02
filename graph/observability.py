"""Observability helpers for LangGraph node execution."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import config
from core.logger import get_logger
from graph.state import AgentState

logger = get_logger("langgraph")


def log_node_enter(node_name: str, state: AgentState) -> float:
    """Log node entry and return a high-resolution start timestamp."""
    started = perf_counter()
    if config.LANGGRAPH_DEBUG or config.DEBUG:
        logger.info(
            "[LangGraph] ENTER node=%s intent=%s tool=%s attachment_active=%s",
            node_name,
            state.get("intent"),
            state.get("current_tool"),
            state.get("attachment_active"),
        )
    else:
        logger.info("[LangGraph] ENTER node=%s", node_name)
    return started


def log_node_exit(
    node_name: str,
    started: float,
    *,
    updates: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Log node exit with elapsed ms; return timing patch for AgentState."""
    elapsed_ms = (perf_counter() - started) * 1000.0
    logger.info("[LangGraph] EXIT node=%s elapsed_ms=%.1f", node_name, elapsed_ms)
    if (config.LANGGRAPH_DEBUG or config.DEBUG) and updates:
        keys = sorted(updates.keys())
        logger.info("[LangGraph] UPDATE keys=%s", keys)
    return {node_name: elapsed_ms}


def timed_node(
    node_name: str,
) -> Callable[
    [Callable[[AgentState], dict[str, Any]]],
    Callable[[AgentState], dict[str, Any]],
]:
    """Decorator factory: wrap a node with enter/exit logging and timing."""

    def decorator(fn: Callable[[AgentState], dict[str, Any]]) -> Callable[[AgentState], dict[str, Any]]:
        def _wrapped(state: AgentState) -> dict[str, Any]:
            started = log_node_enter(node_name, state)
            try:
                updates = fn(state) or {}
            except Exception as exc:  # noqa: BLE001 — recover into structured error state
                logger.exception("[LangGraph] ERROR node=%s: %s", node_name, exc)
                timing = log_node_exit(node_name, started, updates={"error": str(exc)})
                return {
                    "error": f"{node_name}: {exc}",
                    "reasoning_trace": [f"error:{node_name}:{exc}"],
                    "node_timings_ms": timing,
                }
            timing = log_node_exit(node_name, started, updates=updates)
            updates = dict(updates)
            updates["node_timings_ms"] = timing
            return updates

        _wrapped.__name__ = getattr(fn, "__name__", node_name)
        return _wrapped

    return decorator
