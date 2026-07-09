"""Tests for rebuilt conversation memory behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import QueryResponse
from tools.memory.conversation import ConversationMemory
from tools.memory.resolver import resolve


def _response(
    *,
    answer: str,
    intent: str,
    entities: dict | None = None,
    sql_rows: list[dict] | None = None,
    sql: str | None = None,
) -> QueryResponse:
    """Build a response for conversation memory tests."""
    return QueryResponse(
        question="test",
        answer=answer,
        capability="sql",
        planning_intent=intent,
        planning_entities=entities or {},
        generated_sql=sql,
        sql_rows=sql_rows,
    )


@patch("tools.memory.conversation.extract_entities")
@patch("tools.memory.resolver.get_llm")
def test_driver_name_carried_into_hub_follow_up(
    mock_get_llm: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {
        "driver_name": "Drew Nguyen",
        "metric": "pickup_count",
    }
    memory = ConversationMemory()
    memory.add_turn(
        user_question="highest performing driver",
        resolved_question="highest performing driver",
        response=_response(
            answer="Drew Nguyen had 56 pickups.",
            intent="highest_performing_driver",
        ),
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Which hub does Drew Nguyen belong to?"
    )

    resolved = resolve("which hub?", memory)

    assert resolved == "Which hub does Drew Nguyen belong to?"
    assert memory.get_current_state()["active_entities"]["driver_name"] == "Drew Nguyen"


@patch("tools.memory.resolver.get_llm")
def test_previous_hub_rows_used_for_rank_them(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="hub performance",
        resolved_question="hub performance",
        response=_response(
            answer="Atlanta 114 pickups. Chicago 71 pickups.",
            intent="hub_performance",
            sql="SELECT hub, COUNT(*) FROM pickups GROUP BY hub",
            sql_rows=[
                {"hub": "Atlanta Hub", "pickup_count": 114},
                {"hub": "Chicago Hub", "pickup_count": 71},
            ],
        ),
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Rank Atlanta Hub and Chicago Hub by pickup performance."
    )

    resolved = resolve("rank them", memory)

    assert resolved == "Rank Atlanta Hub and Chicago Hub by pickup performance."
    context = memory.get_current_state()["last_result_context"]
    assert context["columns"] == ["hub", "pickup_count"]
    assert len(context["rows"]) == 2


@patch("tools.memory.conversation.extract_entities")
@patch("tools.memory.resolver.get_llm")
def test_hub_and_date_carried_to_last_month_follow_up(
    mock_get_llm: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {
        "hub": "ORD",
        "time_period": "this month",
        "metric": "pickup_performance",
    }
    memory = ConversationMemory()
    memory.add_turn(
        user_question="ORD performance this month",
        resolved_question="ORD performance this month",
        response=_response(
            answer="ORD had 88 pickups this month.",
            intent="hub_performance",
        ),
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Show ORD pickup performance for last month."
    )

    resolved = resolve("what about last month?", memory)

    assert resolved == "Show ORD pickup performance for last month."
    state = memory.get_current_state()
    assert state["active_entities"]["hub"] == "ORD"
    assert state["active_entities"]["time_period"] == "this month"


def test_only_newest_10_turns_are_stored() -> None:
    memory = ConversationMemory()

    for index in range(12):
        memory.add_turn(
            user_question=f"question {index}",
            resolved_question=f"question {index}",
            response=_response(answer="ok", intent="test"),
            extracted_entities={},
        )

    turns = memory.get_recent_history()

    assert len(turns) == 10
    assert turns[0]["user_question"] == "question 2"
    assert turns[-1]["user_question"] == "question 11"


@patch("tools.memory.resolver.get_llm")
def test_random_new_question_ignores_old_context(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="hub performance",
        resolved_question="hub performance",
        response=_response(
            answer="Atlanta 114 pickups.",
            intent="hub_performance",
            sql_rows=[{"hub": "Atlanta Hub", "pickup_count": 114}],
        ),
        extracted_entities={"hub": "Atlanta Hub"},
    )

    resolved = resolve("What is CBT?", memory)

    assert resolved == "What is CBT?"
    mock_get_llm.assert_not_called()
