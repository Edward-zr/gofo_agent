"""Unit tests for analytical result context memory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import QueryRequest, QueryResponse
from tools.memory import ConversationMemory, references_previous_result, resolve
from tools.memory.analyzer import analyze_previous_result
from tools.router import route


def _hub_response(row_count: int = 3) -> QueryResponse:
    """Build a SQL response with hub result rows."""
    rows = [
        {"hub": "Atlanta Hub", "pickup_count": 114},
        {"hub": "Chicago Hub", "pickup_count": 71},
        {"hub": "Dallas Hub", "pickup_count": 90},
    ][:row_count]
    return QueryResponse(
        question="Show pickup conditions by hub",
        answer="Atlanta Hub had 114 pickups.",
        capability="sql",
        planning_intent="hub_performance",
        generated_sql="SELECT hub, COUNT(*) AS pickup_count FROM pickups GROUP BY hub",
        sql_rows=rows,
    )


def test_store_last_result_context_from_sql_rows() -> None:
    memory = ConversationMemory()

    memory.add_turn(
        user_question="Show pickup conditions by hub",
        resolved_question="Show pickup conditions by hub",
        response=_hub_response(),
    )

    context = memory.get_current_state()["last_result_context"]

    assert context["type"] == "sql_result"
    assert context["intent"] == "hub_performance"
    assert context["columns"] == ["hub", "pickup_count"]
    assert len(context["rows"]) == 3
    assert context["summary"] == "Atlanta Hub had 114 pickups."


def test_last_result_context_limits_rows_to_100() -> None:
    memory = ConversationMemory()
    response = QueryResponse(
        question="Show hubs",
        answer="Hub results.",
        capability="sql",
        planning_intent="hub_performance",
        generated_sql="SELECT * FROM hub_results",
        sql_rows=[{"hub": f"Hub {index}", "pickup_count": index} for index in range(120)],
    )

    memory.add_turn(
        user_question="Show hubs",
        resolved_question="Show hubs",
        response=response,
    )

    assert len(memory.get_current_state()["last_result_context"]["rows"]) == 100


@patch("tools.memory.resolver.get_llm")
def test_resolve_rank_them(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(
        user_question="Show pickup conditions by hub",
        resolved_question="Show pickup conditions by hub",
        response=_hub_response(),
    )
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Rank the previous hub performance results by pickup_count descending."
    )

    resolved = resolve("Rank them from highest to lowest", memory)

    assert resolved == "Rank the previous hub performance results by pickup_count descending."
    prompt = mock_get_llm.return_value.invoke.call_args.args[0][1].content
    assert "last_result_context" in prompt
    assert "pickup_count" in prompt


@patch("tools.memory.resolver.get_llm")
def test_resolve_sort_them(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(user_question="Show hubs", resolved_question="Show hubs", response=_hub_response())
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Sort the previous hub results by pickup_count descending."
    )

    assert resolve("sort them", memory) == (
        "Sort the previous hub results by pickup_count descending."
    )


@patch("tools.memory.resolver.get_llm")
def test_resolve_top_5(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(user_question="Show hubs", resolved_question="Show hubs", response=_hub_response())
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Show top 5 hubs from the previous hub performance results."
    )

    assert resolve("show only top 5", memory) == (
        "Show top 5 hubs from the previous hub performance results."
    )


@patch("tools.memory.resolver.get_llm")
def test_resolve_lowest_one(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(user_question="Show hubs", resolved_question="Show hubs", response=_hub_response())
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Show the lowest-performing hub from the previous result."
    )

    assert resolve("lowest one", memory) == "Show the lowest-performing hub from the previous result."


@patch("tools.memory.resolver.get_llm")
def test_resolve_compare_these(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(user_question="Show hubs", resolved_question="Show hubs", response=_hub_response())
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Compare the hubs in the previous result."
    )

    assert resolve("compare these", memory) == "Compare the hubs in the previous result."


@patch("tools.memory.resolver.get_llm")
def test_resolve_first_one(mock_get_llm: MagicMock) -> None:
    memory = ConversationMemory()
    memory.add_turn(user_question="Show hubs", resolved_question="Show hubs", response=_hub_response())
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Explain Atlanta Hub from the previous result."
    )

    assert resolve("what about the first one", memory) == "Explain Atlanta Hub from the previous result."


@patch("tools.memory.analyzer.get_llm")
def test_analyze_previous_result(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="Atlanta Hub ranks first with 114 pickups."
    )

    answer = analyze_previous_result(
        "Rank the previous hub results by pickup_count descending.",
        {
            "type": "sql_result",
            "columns": ["hub", "pickup_count"],
            "rows": [{"hub": "Atlanta Hub", "pickup_count": 114}],
        },
    )

    assert answer == "Atlanta Hub ranks first with 114 pickups."
    mock_get_llm.return_value.invoke.assert_called_once()


@patch("tools.router.analyze_previous_result")
@patch("tools.router.plan")
def test_router_routes_result_context_to_memory_analysis(
    mock_plan: MagicMock,
    mock_analyze: MagicMock,
) -> None:
    context = {
        "type": "sql_result",
        "columns": ["hub", "pickup_count"],
        "rows": [{"hub": "Atlanta Hub", "pickup_count": 114}],
    }
    mock_analyze.return_value = "Atlanta Hub ranks first with 114 pickups."

    response = route(QueryRequest(question="Rank them", result_context=context))

    assert response.capability == "memory_analysis"
    assert response.answer == "Atlanta Hub ranks first with 114 pickups."
    assert response.last_result_context == context
    mock_plan.assert_not_called()


def test_references_previous_result_phrases() -> None:
    assert references_previous_result("rank them")
    assert references_previous_result("sort them")
    assert references_previous_result("show top 5")
    assert references_previous_result("lowest one")
    assert references_previous_result("compare these")
    assert references_previous_result("what about the first one")
