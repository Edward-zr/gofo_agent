"""Tests for LangGraph planner-driven orchestration loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import config
import pytest

from core.agent import GOFOAgent
from core.intent_router import RouteIntent
from core.models import QueryResponse
from core.planner import ExecutionPlan, ExecutionStep
from core.plan_executor import ExecutionResult
from graph.builder import build_compiled_graph, get_checkpointer
from graph.nodes import route_after_planner, route_after_supervisor, route_after_verification
from graph.runtime import LangGraphGOFOAgent, create_session_agent
from graph.state import empty_agent_state
from graph.tools import classify_route


@pytest.fixture
def enable_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LANGGRAPH_ENABLED", True)
    monkeypatch.setattr(config, "LANGGRAPH_ROUTED_NODES", True)
    monkeypatch.setattr(config, "LANGGRAPH_CHECKPOINTING", True)
    monkeypatch.setattr(config, "LANGGRAPH_DEBUG", True)


def test_empty_agent_state_defaults() -> None:
    state = empty_agent_state(question="hello", session_id="s1", attachment_ids=["a"])
    assert state["user_question"] == "hello"
    assert state["session_id"] == "s1"
    assert state["attachment_ids"] == ["a"]
    assert state["replan_count"] == 0
    assert state["task_spec"] == {}
    assert state["error"] is None


def test_route_after_planner_and_verification() -> None:
    from graph.nodes import route_after_frontier

    assert route_after_frontier({"not_meaningful": True}) == "finalize"
    assert route_after_frontier({"not_meaningful": False}) == "task_decomposition"
    assert route_after_planner({"needs_clarification": True}) == "finalize"
    assert route_after_planner({"needs_clarification": False}) == "tool_execution"
    assert route_after_verification({"verification_passed": True, "replan_count": 0}) == "summarize"
    assert route_after_verification({"verification_passed": False, "replan_count": 0}) == "replan"
    assert route_after_verification({"verification_passed": False, "replan_count": 1}) == "summarize"


def test_route_after_supervisor_maps_intents() -> None:
    """Backward-compatible Phase-1 helper still maps intent → tool names."""
    assert route_after_supervisor({"current_tool": "ada", "intent": ""}) == "ada"
    assert (
        route_after_supervisor({"current_tool": "", "intent": RouteIntent.SOP_QA}) == "rag"
    )


def test_classify_route_uses_intent_router(enable_langgraph: None) -> None:
    agent = GOFOAgent()
    routed = classify_route(agent, question="what is the pickup procedure?")
    assert routed["intent"] == RouteIntent.SOP_QA
    assert routed["current_tool"] == "rag"
    assert routed["reasoning_trace"]


def _mock_plan_execute(answer: str = "Hello from GOFO", capability: str = "conversation") -> MagicMock:
    response = QueryResponse(
        question="hi",
        answer=answer,
        capability=capability,
        sources=[],
    )
    plan = ExecutionPlan(
        goal="greet",
        steps=[
            ExecutionStep(
                step_number=1,
                tool="LLM",
                action="greet",
                description="greet",
            )
        ],
    )
    return MagicMock(return_value=ExecutionResult(response=response, step_results={}, plan=plan))


@patch("graph.nodes.execute_plan")
def test_langgraph_agent_ask_returns_api_shape(
    mock_execute: MagicMock,
    enable_langgraph: None,
) -> None:
    mock_execute.return_value = QueryResponse(
        question="hi",
        answer="Hello from GOFO",
        capability="conversation",
    )
    graph_agent = LangGraphGOFOAgent(GOFOAgent(), thread_id="test-thread")
    response = graph_agent.ask("hi")
    assert response["answer"] == "Hello from GOFO"
    assert response["analysis"]["langgraph"] is True
    assert response["analysis"].get("langgraph_loop") == "frontier_planner"
    mock_execute.assert_called()


@patch("graph.nodes.execute_plan")
def test_checkpoint_survives_two_turns(
    mock_execute: MagicMock,
    enable_langgraph: None,
) -> None:
    mock_execute.side_effect = [
        QueryResponse(question="hi", answer="first", capability="conversation"),
        QueryResponse(question="thanks", answer="second", capability="conversation"),
    ]
    base = GOFOAgent()
    base.state.last_attachment_filenames = ["report.xlsx"]
    base.state.last_attachment_file_types = ["excel"]
    graph_agent = LangGraphGOFOAgent(base, thread_id="persist-thread")

    first = graph_agent.ask("hi")
    second = graph_agent.ask("thanks")
    assert first["answer"] == "first"
    assert second["answer"] == "second"
    assert graph_agent.state.last_attachment_filenames == ["report.xlsx"]


@patch("graph.nodes.execute_plan")
def test_legacy_flag_still_uses_planner_loop(
    mock_execute: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LANGGRAPH_ROUTED_NODES=false is ignored; loop still runs PlanExecutor."""
    monkeypatch.setattr(config, "LANGGRAPH_ENABLED", True)
    monkeypatch.setattr(config, "LANGGRAPH_ROUTED_NODES", False)
    monkeypatch.setattr(config, "LANGGRAPH_CHECKPOINTING", False)
    mock_execute.return_value = QueryResponse(
        question="q",
        answer="loop path",
        capability="sql",
        sql_rows=[{"hub": "A"}],
    )
    graph_agent = LangGraphGOFOAgent(GOFOAgent(), thread_id="legacy-mode")
    response = graph_agent.ask("summarize hubs")
    assert response["answer"] == "loop path"
    assert response["analysis"]["langgraph"] is True
    mock_execute.assert_called()


def test_build_compiled_graph_with_checkpointer(enable_langgraph: None) -> None:
    agent = GOFOAgent()
    checkpointer = get_checkpointer()
    graph = build_compiled_graph(agent, checkpointer=checkpointer)
    assert graph is not None


@patch("graph.nodes.execute_plan")
def test_tool_failure_recovers_structured_error(
    mock_execute: MagicMock,
    enable_langgraph: None,
) -> None:
    mock_execute.side_effect = RuntimeError("boom")
    graph_agent = LangGraphGOFOAgent(GOFOAgent(), thread_id="err-thread")
    response = graph_agent.ask("rank hubs")
    assert response.get("answer")
    assert (
        "error" in (response.get("analysis") or {})
        or "boom" in str(response.get("analysis"))
        or "internal orchestration error" in response["answer"].lower()
    )


def test_create_session_agent_respects_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LANGGRAPH_ENABLED", False)
    assert isinstance(create_session_agent("s1"), GOFOAgent)
    monkeypatch.setattr(config, "LANGGRAPH_ENABLED", True)
    monkeypatch.setattr(config, "LANGGRAPH_CHECKPOINTING", False)
    wrapped = create_session_agent("s2")
    assert isinstance(wrapped, LangGraphGOFOAgent)


@patch("graph.nodes.execute_plan")
def test_happy_path_intent_plan_execute_verify(
    mock_execute: MagicMock,
    enable_langgraph: None,
) -> None:
    mock_execute.return_value = QueryResponse(
        question="What is CBT?",
        answer="CBT is Collection by TikTok.",
        capability="rag",
        sources=[],
    )
    # Attach a fake source via model after construction if needed
    agent = GOFOAgent()
    resolve_spy = MagicMock(wraps=agent.conversation.resolve)
    route_spy = MagicMock(wraps=agent.intent_router.route)
    agent.conversation.resolve = resolve_spy
    agent.intent_router.route = route_spy

    graph_agent = LangGraphGOFOAgent(agent, thread_id="cbt-loop")
    response = graph_agent.ask("What is CBT?")

    assert "CBT" in (response.get("answer") or "")
    assert resolve_spy.call_count == 1
    assert route_spy.call_count == 1
    assert mock_execute.call_count == 1
    assert response["analysis"].get("langgraph_loop") == "frontier_planner"
    assert response["analysis"].get("langgraph_verification_passed") is True
    assert response["analysis"].get("frontier_problem_kind") == "sop_searchable"


@patch("graph.nodes.execute_plan")
@patch("graph.nodes.verify_request")
def test_verify_fail_one_replan_then_pass(
    mock_verify: MagicMock,
    mock_execute: MagicMock,
    enable_langgraph: None,
) -> None:
    from core.request_verifier import VerificationReport

    mock_execute.side_effect = [
        QueryResponse(question="q", answer="draft", capability="rag"),
        QueryResponse(question="q", answer="fixed CBT answer", capability="rag"),
    ]
    mock_verify.side_effect = [
        VerificationReport(
            passed=False,
            tool_ok=False,
            feedback=["empty sources"],
            should_retry_retrieval=True,
            action="retry_retrieval",
            suggested_tool_calls=["RAG"],
        ),
        VerificationReport(passed=True, action="approve"),
    ]

    graph_agent = LangGraphGOFOAgent(GOFOAgent(), thread_id="replan-once")
    response = graph_agent.ask("What is CBT?")
    assert "fixed" in (response.get("answer") or "") or "CBT" in (response.get("answer") or "")
    assert mock_execute.call_count == 2
    assert mock_verify.call_count == 2
    assert response["analysis"].get("langgraph_replan_count") == 1


def test_phase1_cli_path_still_repairs_and_routes_inside_ask() -> None:
    """With LangGraph disabled, ask() owns repair+route (backward compatible)."""
    agent = GOFOAgent()
    resolve_spy = MagicMock(wraps=agent.conversation.resolve)
    route_spy = MagicMock(wraps=agent.intent_router.route)
    agent.conversation.resolve = resolve_spy
    agent.intent_router.route = route_spy

    mock_response = QueryResponse(
        question="What is CBT?",
        answer="CBT is Collection by TikTok.",
        capability="rag",
    )
    with patch.object(agent.route_dispatcher, "dispatch", return_value=mock_response):
        response = agent.ask("What is CBT?")

    assert "CBT" in (response.get("answer") or "")
    assert resolve_spy.call_count == 1
    assert route_spy.call_count == 1


def test_pre_routed_ask_skips_resolver_and_router() -> None:
    """Direct pre_routed ask must not call ConversationResolver or IntentRouter."""
    agent = GOFOAgent()
    resolve_spy = MagicMock(wraps=agent.conversation.resolve)
    route_spy = MagicMock(wraps=agent.intent_router.route)
    agent.conversation.resolve = resolve_spy
    agent.intent_router.route = route_spy

    route_decision = {
        "intent": RouteIntent.SOP_QA,
        "confidence": 0.9,
        "followup": False,
        "target": "sop",
        "handler": "RAG",
        "previous_route": None,
        "attachment_active": False,
        "use_attachments": False,
        "detach_attachments": False,
        "chart_type": None,
        "data_sources": ["RAG"],
    }
    mock_response = QueryResponse(
        question="What is CBT?",
        answer="CBT means Collection by TikTok.",
        capability="rag",
    )
    with patch.object(agent.route_dispatcher, "dispatch", return_value=mock_response):
        response = agent.ask(
            "What is CBT?",
            pre_routed=True,
            resolved_question="What is CBT?",
            route_decision=route_decision,
            conversation_resolution={
                "original_question": "What is CBT?",
                "resolved_question": "What is CBT?",
                "repair_detected": False,
            },
        )

    assert "CBT" in (response.get("answer") or "")
    assert resolve_spy.call_count == 0
    assert route_spy.call_count == 0
