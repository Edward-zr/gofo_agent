"""Unit tests for short-term conversation memory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import QueryResponse
from tools.memory import ConversationMemory, resolve


def _response(
    *,
    answer: str = "There were 13 pickups yesterday.",
    capability: str = "sql",
    intent: str = "pickup_count",
    entities: dict | None = None,
) -> QueryResponse:
    """Build a response suitable for memory tests."""
    return QueryResponse(
        question="How many pickups yesterday?",
        answer=answer,
        capability=capability,
        planning_intent=intent,
        planning_entities=entities or {"date": "yesterday"},
        generated_sql="SELECT COUNT(*) FROM pickups;",
    )


def test_memory_keeps_last_10_turns() -> None:
    memory = ConversationMemory()

    for index in range(12):
        memory.add_turn(
            user_question=f"question {index}",
            resolved_question=f"resolved {index}",
            response=_response(),
        )

    history = memory.get_recent_history()

    assert len(history) == 10
    assert history[0]["user_question"] == "question 2"
    assert history[-1]["user_question"] == "question 11"


def test_memory_removes_oldest_history() -> None:
    memory = ConversationMemory(max_history=2)

    for index in range(3):
        memory.add_turn(
            user_question=f"question {index}",
            resolved_question=f"resolved {index}",
            response=_response(),
        )

    assert [turn["user_question"] for turn in memory.get_recent_history()] == [
        "question 1",
        "question 2",
    ]


def test_memory_current_state_updates_after_turn() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        user_question="How many pickups yesterday?",
        resolved_question="How many pickups yesterday?",
        response=_response(entities={"date": "yesterday", "metric": "pickup_count"}),
    )

    state = memory.get_current_state()

    assert state["current_topic"] == "pickup count"
    assert state["current_intent"] == "pickup_count"
    assert state["active_filters"]["date"] == "yesterday"
    assert state["active_filters"]["metric"] == "pickup_count"
    assert state["active_entities"]["date"] == "yesterday"
    assert state["active_entities"]["metric"] == "pickup_count"


@patch("tools.memory.resolver.get_llm")
def test_resolve_follow_up_warehouse(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="How many pickups yesterday?",
        resolved_question="How many pickups yesterday?",
        response=_response(entities={"date": "yesterday", "metric": "pickup_count"}),
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="How many pickups yesterday for ORD warehouse?"
    )

    resolved = resolve("What about ORD?", memory)

    assert resolved == "How many pickups yesterday for ORD warehouse?"
    mock_get_llm.return_value.invoke.assert_called_once()


@patch("tools.memory.resolver.get_llm")
def test_resolve_follow_up_date(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="How many pickups for ORD?",
        resolved_question="How many pickups for ORD?",
        response=_response(entities={"warehouse": "ORD", "metric": "pickup_count"}),
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="How many pickups yesterday for ORD warehouse?"
    )

    resolved = resolve("What about yesterday?", memory)

    assert resolved == "How many pickups yesterday for ORD warehouse?"


@patch("tools.memory.resolver.get_llm")
def test_resolve_follow_up_comparison(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="How many pickups yesterday for ORD?",
        resolved_question="How many pickups yesterday for ORD warehouse?",
        response=_response(entities={"date": "yesterday", "warehouse": "ORD"}),
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Compare ORD pickup count yesterday with last week."
    )

    resolved = resolve("Compare with last week.", memory)

    assert resolved == "Compare ORD pickup count yesterday with last week."


@patch("tools.memory.resolver.get_llm")
def test_resolve_new_unrelated_question_returns_original(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="How many pickups yesterday?",
        resolved_question="How many pickups yesterday?",
        response=_response(),
    )

    resolved = resolve("What is CBT?", memory)

    assert resolved == "What is CBT?"
    mock_get_llm.assert_not_called()


def test_clear_memory() -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="How many pickups yesterday?",
        resolved_question="How many pickups yesterday?",
        response=_response(),
    )

    memory.clear()

    assert memory.get_recent_history() == []
    assert memory.get_current_state() == {
        "previous_question": None,
        "previous_sql": None,
        "previous_answer": None,
        "previous_rows": None,
        "current_metric": None,
        "analysis_dimension": None,
        "date_range": None,
        "filters": {},
        "current_topic": None,
        "current_intent": None,
        "active_entities": {},
        "active_metrics": {},
        "active_filters": {},
        "last_sql_context": None,
        "last_result_context": None,
    }
