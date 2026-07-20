"""Unit tests for the Reflection (self-critique) layer."""

from __future__ import annotations

from types import SimpleNamespace

from core.intent_classifier import IntentClassification, IntentType
from core.planner import Planner
from core.reflection import ReflectionAgent, ReflectionResult


def test_successful_answer_is_approved() -> None:
    result = ReflectionAgent().critique(
        "How many pickups today?",
        "There were 120 pickups today with a 92% completion rate.",
        sql="SELECT COUNT(*) AS pickup_count FROM pickups",
        sql_rows=[{"pickup_count": 120, "completion_rate": 0.92}],
        capability="sql",
    )
    assert result.approved is True
    assert result.confidence >= 0.75
    assert result.should_retry_retrieval is False
    assert result.should_ask_user is False


def test_weak_rag_retrieval_requests_retry() -> None:
    result = ReflectionAgent().critique(
        "What is the pickup SOP?",
        "Drivers should always scan packages before departure according to policy.",
        sources=[],
        capability="rag",
    )
    assert result.approved is False
    assert result.should_retry_retrieval is True
    assert "RAG" in result.suggested_tool_calls


def test_missing_sql_information_requests_sql_or_plan_retry() -> None:
    result = ReflectionAgent().critique(
        "Why did Chicago performance decrease last week?",
        "Chicago had 80 pickups last week.",
        sql="SELECT COUNT(*) FROM pickups",
        sql_rows=[{"pickup_count": 80}],
        capability="sql",
    )
    assert result.approved is False
    assert result.should_retry_sql or result.should_retry_python or result.missing_information
    assert result.needs_retry() is True


def test_ambiguous_user_question_asks_for_clarification() -> None:
    result = ReflectionAgent().critique(
        "check performance",
        "Performance varies by hub.",
        sql_rows=[{"hub": "ORD", "rate": 0.9}],
        capability="sql",
        ambiguous=True,
    )
    assert result.approved is False
    assert result.should_ask_user is True
    assert result.clarification_question
    assert "date" in result.clarification_question.lower() or "week" in result.clarification_question.lower()


def test_retry_success_via_planner_feedback() -> None:
    first = ReflectionAgent().critique(
        "Why did Chicago performance decrease last week?",
        "Chicago declined.",
        sql_rows=[{"hub": "Chicago", "pickups": 10}],
        capability="sql",
    )
    assert first.approved is False

    classification = IntentClassification(
        intent=IntentType.SQL_Analysis,
        confidence=0.9,
        requires_sql=True,
        requires_planner=True,
        reasoning="test",
    )
    plan = Planner().plan_retry(
        "Why did Chicago performance decrease last week?",
        classification,
        first,
        previous_answer="Chicago declined.",
    )
    assert plan.steps
    assert any(step.tool in {"SQL", "PYTHON", "RAG"} for step in plan.steps)

    # Simulate improved answer after retry tools.
    improved = ReflectionAgent().critique_response(
        "Why did Chicago performance decrease last week?",
        SimpleNamespace(
            answer=(
                "Chicago decreased because failed pickups rose due to address exceptions. "
                "Compared with the previous week, completion fell from 90% to 72%."
            ),
            sources=[],
            generated_sql="SELECT ...",
            sql_rows=[
                {"hub": "Chicago", "failed_pickups": 40, "completion_rate": 0.72, "week": "current"},
                {"hub": "Chicago", "failed_pickups": 10, "completion_rate": 0.90, "week": "previous"},
            ],
            capability="sql",
            execution_plan=None,
            intent_classification=None,
            root_cause={"primary_factor": "address exceptions"},
            charts=[],
        ),
        retry_count=1,
        previous_answer="Chicago declined.",
    )
    assert improved.confidence >= first.confidence
    assert improved.approved is True or len(improved.missing_information) < len(first.missing_information)


def test_maximum_retry_limit_forces_accept() -> None:
    result = ReflectionAgent(max_retries=2).critique(
        "Why did Chicago performance decrease last week?",
        "Chicago declined.",
        sql_rows=[{"hub": "Chicago", "pickups": 10}],
        capability="sql",
        retry_count=2,
    )
    assert result.approved is True
    assert result.retry_count == 2
    assert result.should_ask_user is False


def test_reflection_never_executes_tools(monkeypatch) -> None:
    called = {"sql": False}

    def boom(*args, **kwargs):
        called["sql"] = True
        raise AssertionError("Reflection must not execute tools")

    monkeypatch.setattr("tools.sql.service.answer", boom)
    result = ReflectionAgent().critique(
        "Show today's pickups",
        "120 pickups",
        sql_rows=[{"pickup_count": 120}],
        capability="sql",
    )
    assert called["sql"] is False
    assert isinstance(result, ReflectionResult)


def test_reflection_result_needs_retry_helper() -> None:
    approved = ReflectionResult(approved=True, confidence=0.9)
    assert approved.needs_retry() is False

    retry = ReflectionResult(
        approved=False,
        confidence=0.5,
        should_retry_sql=True,
        feedback=["Need failures"],
    )
    assert retry.needs_retry() is True


def test_llm_critique_merge_is_conservative() -> None:
    class _LLM:
        def invoke(self, messages):  # noqa: ANN001
            return SimpleNamespace(
                content=(
                    '{"approved": true, "confidence": 0.99, "should_retry_retrieval": false,'
                    ' "should_retry_sql": false, "should_retry_python": false,'
                    ' "should_ask_user": false, "missing_information": [],'
                    ' "feedback": ["looks fine"], "clarification_question": null,'
                    ' "reasoning": "llm wants approve"}'
                )
            )

    # Baseline rejects weak RAG; LLM approve must not override rejection.
    agent = ReflectionAgent(llm=_LLM(), use_llm=True)
    result = agent.critique(
        "What is the pickup SOP?",
        "I believe drivers probably skip scanning.",
        sources=[],
        capability="rag",
    )
    assert result.approved is False
    assert result.should_retry_retrieval is True
