"""Tests for semantic request analysis and response orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import QueryRequest, QueryResponse
from tools.orchestration.models import SemanticAnalysis
from tools.orchestration.next_steps import generate_next_steps
from tools.orchestration.semantic_analyzer import analyze_request
from tools.planner import PlanningDecision
from tools.router import route



@patch("tools.orchestration.semantic_analyzer.get_llm")
def test_analyze_request_classifies_sql_ranking(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="""
        {
          "domain": "data_analytics",
          "sub_intent": "ranking",
          "capability": "sql",
          "response_mode": "analytical",
          "confidence": 0.93,
          "resolved_question": "Rank all hubs by pickup performance",
          "reasoning": "Operational ranking question.",
          "entities": {},
          "metric": "pickup_completion_rate",
          "dimension": "hub",
          "use_previous_result": false,
          "use_previous_analysis": false,
          "requires_sql": true,
          "requires_rag": false,
          "direct_reply": null,
          "planner_intent": "hub_ranking"
        }
        """
    )

    analysis = analyze_request("Rank all hubs by performance")

    assert analysis.domain == "data_analytics"
    assert analysis.sub_intent == "ranking"
    assert analysis.capability == "sql"
    assert analysis.requires_sql is True
    assert analysis.is_actionable() is True


@patch("tools.orchestration.semantic_analyzer.get_llm")
def test_analyze_request_greeting_returns_conversation_reply(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="""
        {
          "domain": "general_conversation",
          "sub_intent": "greeting",
          "capability": "conversation",
          "response_mode": "conversational",
          "confidence": 0.8,
          "resolved_question": "hello",
          "reasoning": "Greeting",
          "entities": {},
          "metric": null,
          "dimension": null,
          "use_previous_result": false,
          "use_previous_analysis": false,
          "requires_sql": false,
          "requires_rag": false,
          "direct_reply": null,
          "planner_intent": "greeting"
        }
        """
    )

    analysis = analyze_request("hello")

    assert analysis.capability == "conversation"
    assert analysis.should_answer_directly() is True
    assert "GOFO Operations Intelligence Analyst" in (analysis.direct_reply or "")


@patch("tools.router.generate_next_steps")
@patch("tools.router.rag_answer")
@patch("tools.router.sql_answer")
@patch("tools.router.plan")
def test_router_uses_semantic_context_for_conversation(
    mock_plan: MagicMock,
    mock_sql_answer: MagicMock,
    mock_rag_answer: MagicMock,
    mock_next_steps: MagicMock,
) -> None:
    mock_plan.return_value = PlanningDecision(
        capability="unknown",
        intent="greeting",
        confidence=0.5,
        requires_sql=False,
        requires_rag=False,
        reasoning="unused",
        entities={},
    )
    mock_next_steps.return_value = ["Rank all hubs by performance."]

    semantic = SemanticAnalysis(
        domain="general_conversation",
        sub_intent="greeting",
        capability="conversation",
        response_mode="conversational",
        confidence=0.95,
        resolved_question="hello",
        reasoning="Greeting detected.",
        direct_reply="Hello. I can help with GOFO operations analytics.",
        planner_intent="greeting",
    )

    response = route(
        QueryRequest(
            question="hello",
            semantic_context=semantic.model_dump(),
        )
    )

    assert response.capability == "conversation"
    assert "GOFO operations analytics" in (response.answer or "")
    mock_sql_answer.assert_not_called()
    mock_rag_answer.assert_not_called()


@patch("tools.router.generate_next_steps")
@patch("tools.router.rag_answer")
@patch("tools.router.sql_answer")
@patch("tools.router.plan")
def test_router_prefers_semantic_sql_over_unknown_plan(
    mock_plan: MagicMock,
    mock_sql_answer: MagicMock,
    mock_rag_answer: MagicMock,
    mock_next_steps: MagicMock,
) -> None:
    mock_plan.return_value = PlanningDecision(
        capability="unknown",
        intent="unrelated",
        confidence=0.4,
        requires_sql=False,
        requires_rag=False,
        reasoning="Planner uncertain.",
        entities={},
    )
    mock_sql_answer.return_value = QueryResponse(
        question="Which hub is worst?",
        answer="Chicago Hub is worst.",
        capability="sql",
        generated_sql="SELECT 1;",
        sql_rows=[{"hub": "Chicago Hub", "pickups": 10}],
    )
    mock_next_steps.return_value = ["Why is Chicago Hub performing badly?"]

    semantic = SemanticAnalysis(
        domain="data_analytics",
        sub_intent="ranking",
        capability="sql",
        response_mode="analytical",
        confidence=0.88,
        resolved_question="Which hub is worst?",
        reasoning="Ranking question.",
        requires_sql=True,
        requires_rag=False,
        planner_intent="hub_ranking",
        dimension="hub",
    )

    response = route(
        QueryRequest(
            question="Which hub is worst?",
            semantic_context=semantic.model_dump(),
        )
    )

    assert response.capability == "sql"
    assert "Chicago Hub" in (response.answer or "")
    mock_sql_answer.assert_called_once()


@patch("tools.orchestration.next_steps.get_llm")
def test_generate_next_steps_returns_actionable_followups(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content='{"next_steps": ["Why is Chicago Hub performing badly?", "What should operations do?"]}'
    )
    response = QueryResponse(
        question="Rank all hubs",
        answer="Chicago Hub is worst.",
        capability="sql",
        sql_rows=[{"hub": "Chicago Hub", "pickups": 10}],
        business_metric="pickup_count",
        analysis_dimension="hub",
    )
    steps = generate_next_steps(
        question="Rank all hubs",
        response=response,
        semantic={"domain": "data_analytics", "sub_intent": "ranking", "dimension": "hub"},
    )
    assert len(steps) >= 1
    assert any("Chicago" in step or "operations" in step.lower() for step in steps)


def test_analyze_request_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="Question must not be empty"):
        analyze_request("   ")
