"""Unit tests for entity-aware short-term memory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import QueryResponse
from tools.memory import ConversationMemory, extract_entities, resolve


def _sql_response(
    *,
    answer: str,
    intent: str = "pickup_count",
    entities: dict | None = None,
) -> QueryResponse:
    """Build a SQL response for entity memory tests."""
    return QueryResponse(
        question="Highest performing driver",
        answer=answer,
        capability="sql",
        planning_intent=intent,
        planning_entities=entities or {},
    )


@patch("tools.memory.entity_extractor.get_llm")
def test_extract_entities_driver_metric(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content='{"driver_name": "Drew Nguyen", "metric": "pickup_count"}'
    )

    entities = extract_entities(
        "Highest performing driver",
        "The highest performing driver is Drew Nguyen with 33 completed pickups.",
    )

    assert entities == {
        "driver_name": "Drew Nguyen",
        "metric": "pickup_count",
    }


def test_driver_carry_over() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        user_question="Highest performing driver",
        resolved_question="Highest performing driver",
        response=_sql_response(
            answer="The highest performing driver is Drew Nguyen with 33 completed pickups.",
            intent="highest_performing_driver",
        ),
        extracted_entities={"driver_name": "Drew Nguyen", "metric": "pickup_count"},
    )

    state = memory.get_current_state()

    assert state["active_entities"]["driver_name"] == "Drew Nguyen"
    assert state["active_entities"]["metric"] == "pickup_count"


def test_warehouse_carry_over() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        user_question="How many pickups for ORD?",
        resolved_question="How many pickups for ORD warehouse?",
        response=_sql_response(answer="ORD had 42 pickups."),
        extracted_entities={"warehouse": "ORD"},
    )

    assert memory.get_current_state()["active_entities"]["warehouse"] == "ORD"


def test_customer_carry_over() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        user_question="Show TikTok pickups",
        resolved_question="Show TikTok customer pickups",
        response=_sql_response(answer="TikTok had 18 pickups."),
        extracted_entities={"customer": "TikTok"},
    )

    assert memory.get_current_state()["active_entities"]["customer"] == "TikTok"


def test_date_carry_over() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        user_question="How many pickups yesterday?",
        resolved_question="How many pickups yesterday?",
        response=_sql_response(answer="There were 13 pickups yesterday."),
        extracted_entities={"date": "yesterday"},
    )

    assert memory.get_current_state()["active_entities"]["date"] == "yesterday"


def test_metric_carry_over() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        user_question="Pickup completion rate",
        resolved_question="Pickup completion rate",
        response=_sql_response(answer="The completion rate was 91%."),
        extracted_entities={"metric": "completion_rate"},
    )

    assert memory.get_current_state()["active_entities"]["metric"] == "completion_rate"


@patch("tools.memory.resolver.get_llm")
def test_resolve_this_driver_pronoun(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="Highest performing driver",
        resolved_question="Highest performing driver",
        response=_sql_response(answer="Drew Nguyen had 33 completed pickups."),
        extracted_entities={"driver_name": "Drew Nguyen"},
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Which hub does Drew Nguyen belong to?"
    )

    resolved = resolve("which hub does this driver belong to?", memory)

    assert resolved == "Which hub does Drew Nguyen belong to?"
    prompt = mock_get_llm.return_value.invoke.call_args.args[0][1].content
    assert "Drew Nguyen" in prompt
    assert "active_entities" in prompt


@patch("tools.memory.resolver.get_llm")
def test_resolve_him_pronoun(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="Highest performing driver",
        resolved_question="Highest performing driver",
        response=_sql_response(answer="Drew Nguyen had 33 completed pickups."),
        extracted_entities={"driver_name": "Drew Nguyen"},
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="How many pickups did Drew Nguyen do yesterday?"
    )

    assert resolve("how many pickups did he do yesterday?", memory) == (
        "How many pickups did Drew Nguyen do yesterday?"
    )


@patch("tools.memory.resolver.get_llm")
def test_resolve_her_pronoun(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="Highest performing driver",
        resolved_question="Highest performing driver",
        response=_sql_response(answer="Riley Chen had 31 completed pickups."),
        extracted_entities={"driver_name": "Riley Chen"},
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Compare Riley Chen with last week."
    )

    assert resolve("compare her with last week", memory) == "Compare Riley Chen with last week."


@patch("tools.memory.resolver.get_llm")
def test_resolve_that_warehouse(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="Warehouse performance",
        resolved_question="Warehouse performance",
        response=_sql_response(answer="ORD had the worst performance."),
        extracted_entities={"warehouse": "ORD"},
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Show only failed pickups for ORD warehouse."
    )

    assert resolve("show only failed ones for that warehouse", memory) == (
        "Show only failed pickups for ORD warehouse."
    )
