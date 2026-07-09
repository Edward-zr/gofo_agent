"""Unit tests for tools.router."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import QueryRequest, QueryResponse
from tools.planner import PlanningDecision
from tools.router import route

GENERATED_SQL = (
    "SELECT COUNT(*)\n"
    "FROM pickups\n"
    "WHERE pickup_date = DATE('now', '-1 day');"
)


@patch("tools.router.synthesize")
@patch("tools.router.rag_answer")
@patch("tools.router.sql_answer")
@patch("tools.router.plan")
def test_router_routes_analytics_to_sql_only(
    mock_plan: MagicMock,
    mock_sql_answer: MagicMock,
    mock_rag_answer: MagicMock,
    mock_synthesize: MagicMock,
) -> None:
    mock_plan.return_value = PlanningDecision(
        capability="sql",
        intent="pickup_count",
        confidence=0.94,
        requires_sql=True,
        requires_rag=False,
        reasoning="Question requires operational analytics from the database.",
        entities={"date": "yesterday"},
    )
    mock_sql_answer.return_value = QueryResponse(
        question="How many pickups yesterday?",
        answer="There were 42 pickups yesterday.",
        capability="sql",
        generated_sql=GENERATED_SQL,
        sql_rows=[{"count": 42}],
        latest_business_date="2026-06-28",
    )

    response = route(QueryRequest(question="How many pickups yesterday?"))

    assert response.capability == "sql"
    assert response.answer == "There were 42 pickups yesterday."
    assert response.planning_capability == "sql"
    assert response.planning_intent == "pickup_count"
    mock_sql_answer.assert_called_once()
    mock_rag_answer.assert_not_called()
    mock_synthesize.assert_not_called()


@patch("tools.router.synthesize")
@patch("tools.router.rag_answer")
@patch("tools.router.sql_answer")
@patch("tools.router.plan")
def test_router_routes_sop_questions_to_rag_only(
    mock_plan: MagicMock,
    mock_sql_answer: MagicMock,
    mock_rag_answer: MagicMock,
    mock_synthesize: MagicMock,
) -> None:
    mock_plan.return_value = PlanningDecision(
        capability="rag",
        intent="explain_cbt",
        confidence=0.91,
        requires_sql=False,
        requires_rag=True,
        reasoning="Question requires SOP knowledge retrieval.",
        entities={"term": "CBT"},
    )
    mock_rag_answer.return_value = QueryResponse(
        question="What is CBT?",
        answer="Collection by TikTok.",
        sources=[],
        capability="rag",
    )

    response = route(QueryRequest(question="What is CBT?"))

    assert response.capability == "rag"
    assert response.planning_capability == "rag"
    mock_rag_answer.assert_called_once()
    mock_sql_answer.assert_not_called()
    mock_synthesize.assert_not_called()


@patch("tools.router.synthesize")
@patch("tools.router.rag_answer")
@patch("tools.router.sql_answer")
@patch("tools.router.plan")
def test_router_routes_hybrid_questions_to_multi(
    mock_plan: MagicMock,
    mock_sql_answer: MagicMock,
    mock_rag_answer: MagicMock,
    mock_synthesize: MagicMock,
) -> None:
    mock_plan.return_value = PlanningDecision(
        capability="multi",
        intent="delayed_pickups",
        confidence=0.96,
        requires_sql=True,
        requires_rag=True,
        reasoning="Question requires both operational analytics and SOP guidance.",
        entities={"threshold": 50},
    )
    mock_sql_answer.return_value = QueryResponse(
        question="What should I do if delayed pickups today exceed 50?",
        answer="There are 55 delayed pickups today.",
        capability="sql",
        generated_sql=GENERATED_SQL,
        sql_rows=[{"count": 55}],
    )
    mock_rag_answer.return_value = QueryResponse(
        question="What should I do if delayed pickups today exceed 50?",
        answer="Follow the delayed pickup escalation SOP.",
        sources=[],
        capability="rag",
    )
    mock_synthesize.return_value = (
        "There are 55 delayed pickups today; follow the delayed pickup escalation SOP."
    )

    response = route(
        QueryRequest(question="What should I do if delayed pickups today exceed 50?")
    )

    assert response.capability == "multi"
    assert response.needs_sql is True
    assert response.needs_rag is True
    assert response.planning_capability == "multi"
    assert response.planning_intent == "delayed_pickups"
    assert response.execution_order == ["sql", "rag"]
    mock_synthesize.assert_called_once()


@patch("tools.router.synthesize")
@patch("tools.router.rag_answer")
@patch("tools.router.sql_answer")
@patch("tools.router.plan")
def test_router_returns_unknown_without_tools(
    mock_plan: MagicMock,
    mock_sql_answer: MagicMock,
    mock_rag_answer: MagicMock,
    mock_synthesize: MagicMock,
) -> None:
    mock_plan.return_value = PlanningDecision(
        capability="unknown",
        intent="greeting",
        confidence=0.88,
        requires_sql=False,
        requires_rag=False,
        reasoning="Greeting is not an operational request.",
        entities={},
    )

    response = route(QueryRequest(question="hello"))

    assert response.capability == "unknown"
    assert response.planning_intent == "greeting"
    mock_sql_answer.assert_not_called()
    mock_rag_answer.assert_not_called()
    mock_synthesize.assert_not_called()
