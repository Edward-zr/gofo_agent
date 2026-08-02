"""Compile the GOFO LangGraph with optional checkpointing."""

from __future__ import annotations

from typing import Any

import config
from core.agent import GOFOAgent
from core.logger import get_logger
from graph.nodes import (
    make_nodes,
    route_after_frontier,
    route_after_planner,
    route_after_verification,
)
from graph.state import AgentState

logger = get_logger("langgraph.builder")


def get_checkpointer() -> Any | None:
    """Return an in-memory checkpointer when checkpointing is enabled."""
    if not config.LANGGRAPH_CHECKPOINTING:
        return None
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover
        from langgraph.checkpoint.memory import InMemorySaver as MemorySaver  # type: ignore
    return MemorySaver()


def build_compiled_graph(agent: GOFOAgent, *, checkpointer: Any | None = None):
    """Build frontier + planner-driven orchestration graph."""
    from langgraph.graph import END, START, StateGraph

    nodes = make_nodes(agent)
    builder = StateGraph(AgentState)

    builder.add_node("prepare", nodes["prepare"])
    builder.add_node("intent_understanding", nodes["intent_understanding"])
    builder.add_node("frontier", nodes["frontier"])
    builder.add_node("task_decomposition", nodes["task_decomposition"])
    builder.add_node("planner", nodes["planner"])
    builder.add_node("tool_execution", nodes["tool_execution"])
    builder.add_node("result_verification", nodes["result_verification"])
    builder.add_node("replan", nodes["replan"])
    builder.add_node("summarize", nodes["summarize"])
    builder.add_node("finalize", nodes["finalize"])

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "intent_understanding")
    builder.add_edge("intent_understanding", "frontier")
    builder.add_conditional_edges(
        "frontier",
        route_after_frontier,
        {
            "finalize": "finalize",
            "task_decomposition": "task_decomposition",
        },
    )
    builder.add_edge("task_decomposition", "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "finalize": "finalize",
            "tool_execution": "tool_execution",
        },
    )
    builder.add_edge("tool_execution", "result_verification")
    builder.add_conditional_edges(
        "result_verification",
        route_after_verification,
        {
            "summarize": "summarize",
            "replan": "replan",
        },
    )
    builder.add_edge("replan", "tool_execution")
    builder.add_edge("summarize", "finalize")
    builder.add_edge("finalize", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info(
        "[LangGraph] Compiled frontier+planner loop checkpointing=%s",
        checkpointer is not None,
    )
    if not config.LANGGRAPH_ROUTED_NODES:
        logger.info(
            "[LangGraph] LANGGRAPH_ROUTED_NODES=false ignored — "
            "frontier planner loop always runs"
        )
    return compiled
