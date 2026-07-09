"""Tests for structured conversation state and repair behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.agent import GOFOAgent
from core.models import QueryResponse


@patch("tools.memory.conversation.extract_entities")
@patch("core.agent.ask_core")
def test_highest_driver_then_actually_lowest_changes_sql_order(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {"driver_name": "Drew Nguyen"}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("highest performing driver")
    response = agent.ask("actually lowest driver")

    assert mock_ask_core.call_args_list[-1].args[0] == "Show lowest performing driver"
    assert "ORDER BY completed_pickups ASC" in response["sql"]
    assert "ORDER BY completed_pickups DESC" not in response["sql"]


@patch("tools.memory.conversation.extract_entities")
@patch("core.agent.ask_core")
def test_rank_hubs_then_why_worst_uses_previous_hub_result(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {"hub": "Chicago Hub"}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("rank hubs by performance")
    response = agent.ask("why is worst hub bad?")

    assert mock_ask_core.call_args_list[-1].args[0] == "Why is Chicago Hub performing badly?"
    assert "Chicago Hub" in response["answer"]


@patch("tools.memory.resolver.get_llm")
@patch("tools.memory.conversation.extract_entities")
@patch("core.agent.ask_core")
def test_highest_driver_then_which_hub_uses_entity_memory(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
    mock_get_llm: MagicMock,
) -> None:
    mock_extract_entities.return_value = {"driver_name": "Drew Nguyen"}
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Which hub does Drew Nguyen belong to?"
    )
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("highest performing driver")
    response = agent.ask("which hub does he belong to?")

    assert mock_ask_core.call_args_list[-1].args[0] == "Which hub does Drew Nguyen belong to?"
    assert "Los Angeles Hub" in response["answer"]


@patch("tools.memory.conversation.extract_entities")
@patch("core.agent.ask_core")
def test_highest_driver_then_actually_lowest_does_not_mention_previous_driver(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {"driver_name": "Drew Nguyen"}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("highest driver")
    response = agent.ask("actually lowest")

    assert response["analysis"]["repair_detected"] is True
    assert response["analysis"]["resolved_question"] == "Show lowest performing driver"
    assert "Drew Nguyen" not in response["answer"]


def _fake_agent_response(question: str, result_context: dict | None = None) -> QueryResponse:
    normalized = question.lower()
    if "which hub does drew nguyen" in normalized:
        return QueryResponse(
            question=question,
            answer="Drew Nguyen belongs to Los Angeles Hub.",
            capability="sql",
            planning_intent="driver_lookup",
            generated_sql="SELECT d.hub FROM drivers d WHERE d.driver_name='Drew Nguyen';",
            sql_rows=[{"hub": "Los Angeles Hub"}],
        )
    if "lowest" in normalized and "driver" in normalized:
        return QueryResponse(
            question=question,
            answer="Alex Chen is the lowest performing driver.",
            capability="sql",
            planning_intent="driver_ranking",
            generated_sql=(
                "SELECT d.driver_name, COUNT(*) AS completed_pickups "
                "FROM pickups p JOIN drivers d ON p.driver_id=d.driver_id "
                "GROUP BY d.driver_name ORDER BY completed_pickups ASC LIMIT 1;"
            ),
            sql_rows=[{"driver_name": "Alex Chen", "completed_pickups": 3}],
        )
    if "highest" in normalized and "driver" in normalized:
        return QueryResponse(
            question=question,
            answer="Drew Nguyen is the highest performing driver.",
            capability="sql",
            planning_intent="driver_ranking",
            generated_sql=(
                "SELECT d.driver_name, COUNT(*) AS completed_pickups "
                "FROM pickups p JOIN drivers d ON p.driver_id=d.driver_id "
                "GROUP BY d.driver_name ORDER BY completed_pickups DESC LIMIT 1;"
            ),
            sql_rows=[{"driver_name": "Drew Nguyen", "completed_pickups": 56}],
        )
    if "rank hubs" in normalized:
        return QueryResponse(
            question=question,
            answer="New York Hub #1. Chicago Hub worst.",
            capability="sql",
            planning_intent="hub_ranking",
            generated_sql=(
                "SELECT d.hub, COUNT(*) AS pickups FROM pickups p "
                "JOIN drivers d ON p.driver_id=d.driver_id "
                "GROUP BY d.hub ORDER BY pickups DESC;"
            ),
            sql_rows=[
                {"hub": "New York Hub", "pickups": 100},
                {"hub": "Chicago Hub", "pickups": 10},
            ],
        )
    if "chicago hub" in normalized and "why" in normalized:
        return QueryResponse(
            question=question,
            answer="Chicago Hub is performing badly because delayed pickups increased.",
            capability="sql",
            planning_intent="root_cause_drilldown",
            generated_sql="SELECT * FROM pickups WHERE hub='Chicago Hub';",
            sql_rows=[{"hub": "Chicago Hub", "delayed_pickups": 12}],
        )
    return QueryResponse(
        question=question,
        answer="I don't know how to route this request.",
        capability="unknown",
        planning_intent="unknown",
    )
