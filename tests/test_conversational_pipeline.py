"""Tests for the conversational operations reasoning pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.agent import GOFOAgent
from core.models import QueryResponse
from tools.planner.intent_classifier import Intent, classify_intent


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_worst_hub_followup_resolves_from_semantic_ranking(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("Rank all hubs by performance")
    response = agent.ask("Why is the worst hub performing badly?")

    assert mock_ask_core.call_args_list[-1].args[0] == "Why is Chicago Hub performing badly?"
    assert "Chicago Hub" in response["answer"]
    assert response["analysis"]["intent"] == Intent.ROOT_CAUSE.value


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_recommendation_followup_uses_previous_analysis_without_new_sql(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("Rank all hubs by performance")
    response = agent.ask("What should operations do?")

    assert mock_ask_core.call_count == 1
    assert response["sql"] == ""
    assert response["analysis"]["capability"] == "business_analysis"
    assert response["analysis"]["intent"] == Intent.RECOMMENDATION.value
    assert "Recommendations" in response["answer"]


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_compare_yesterday_inherits_previous_metric_not_raw_filter(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("Show today's pickup performance")
    agent.ask("Compare yesterday")

    resolved = mock_ask_core.call_args_list[-1].args[0]
    assert "Compare yesterday's pickup performance" in resolved
    assert "Los Angeles" not in resolved


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_global_driver_ranking_does_not_inherit_previous_hub_filter(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("Show Los Angeles Hub performance")
    agent.ask("Which driver has the highest performance?")

    resolved = mock_ask_core.call_args_list[-1].args[0]
    assert "highest" in resolved.lower()
    assert "driver" in resolved.lower()
    assert "Los Angeles" not in resolved


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_show_details_uses_previous_result_context(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("Rank all hubs by performance")
    response = agent.ask("Show details")

    assert mock_ask_core.call_args_list[-1].kwargs["result_context"] is not None
    assert response["analysis"]["intent"] == Intent.DRILLDOWN.value
    assert response["analysis"]["capability"] == "memory_analysis"


def test_intent_classifier_uses_conversation_context_for_short_followups() -> None:
    state = {"summary": "Chicago Hub is worst.", "ranking": ["New York Hub", "Chicago Hub"]}

    assert classify_intent("Why?", conversation_state=state) == Intent.ROOT_CAUSE
    assert classify_intent("What should operations do?", conversation_state=state) == Intent.RECOMMENDATION
    assert classify_intent("Show details", conversation_state=state) == Intent.DRILLDOWN
    assert classify_intent("Compare yesterday", conversation_state=state) == Intent.COMPARISON
    assert classify_intent("What happened today?", conversation_state=state) == Intent.SUMMARY
    assert classify_intent("Summarize everything", conversation_state=state) == Intent.SUMMARY


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_what_happened_today_routes_to_operations_summary(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.return_value = QueryResponse(
        question="Generate today's executive operational summary",
        answer="Executive Summary: Operations are stable today.",
        capability="sql",
        planning_intent="operations_kpi_summary",
        sql_rows=[{"pickup_count": 50, "completion_rate": 95.0}],
        business_metric="operations_health",
        analysis_dimension="hub",
    )
    agent = GOFOAgent()
    response = agent.ask("What happened today?")

    assert mock_ask_core.call_count == 1
    assert "Executive Summary" in response["answer"]
    assert response["analysis"]["intent"] == Intent.SUMMARY.value


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_summarize_everything_uses_previous_analysis_without_new_sql(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("Rank all hubs by performance")
    response = agent.ask("Summarize everything")

    assert mock_ask_core.call_count == 1
    assert response["analysis"]["intent"] == Intent.SUMMARY.value
    assert "Executive Summary" in response["answer"]
    assert "Chicago Hub" in response["answer"]


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_hub_repair_switches_to_named_hub(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("Show Atlanta hub performance")
    response = agent.ask("No, I mean Chicago Hub")

    assert mock_ask_core.call_args_list[-1].args[0] == "Show performance for Chicago Hub"
    assert response["analysis"]["repair_detected"] is True


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_actually_compare_last_week_repair(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = _fake_agent_response
    agent = GOFOAgent()

    agent.ask("Show today's pickup performance")
    response = agent.ask("Actually compare last week")

    resolved = mock_ask_core.call_args_list[-1].args[0]
    assert "last week" in resolved.lower()
    assert response["analysis"]["repair_detected"] is True


def _fake_agent_response(
    question: str,
    result_context: dict | None = None,
    **kwargs,
) -> QueryResponse:
    normalized = question.lower()
    if result_context is not None:
        return QueryResponse(
            question=question,
            answer="Details from the previous result: New York Hub and Chicago Hub were returned.",
            capability="memory_analysis",
            planning_intent="drilldown",
            sql_rows=result_context.get("rows") or [],
            last_result_context=result_context,
        )
    if "rank all hubs" in normalized or "rank hubs" in normalized:
        return QueryResponse(
            question=question,
            answer="New York Hub is best. Chicago Hub is worst.",
            capability="sql",
            planning_intent="hub_ranking",
            generated_sql=(
                "SELECT d.hub, COUNT(*) AS pickup_count FROM pickups p "
                "JOIN drivers d ON p.driver_id=d.driver_id "
                "GROUP BY d.hub ORDER BY pickup_count DESC;"
            ),
            sql_rows=[
                {"hub": "New York Hub", "pickup_count": 100, "delayed_pickups": 2},
                {"hub": "Chicago Hub", "pickup_count": 10, "delayed_pickups": 7},
            ],
            business_metric="pickup_completion_rate",
            analysis_dimension="hub",
            root_cause={
                "issue": "Completion performance needs review.",
                "main_causes": ["Delay volume is highest for Chicago Hub."],
                "affected_dimensions": ["hub: Chicago Hub"],
                "recommendation": "Review driver assignment.",
            },
            recommendation="Review driver assignment.",
        )
    if "chicago hub" in normalized and "why" in normalized:
        return QueryResponse(
            question=question,
            answer="Chicago Hub is performing badly because delayed pickups increased.",
            capability="sql",
            planning_intent="root_cause_drilldown",
            generated_sql="SELECT d.hub, COUNT(*) AS delayed_pickups FROM pickups p JOIN drivers d ON p.driver_id=d.driver_id WHERE d.hub='Chicago Hub';",
            sql_rows=[{"hub": "Chicago Hub", "delayed_pickups": 7}],
            business_metric="delay_rate",
            analysis_dimension="hub",
        )
    if "today" in normalized and "pickup performance" in normalized:
        return QueryResponse(
            question=question,
            answer="Today pickup performance is stable.",
            capability="sql",
            planning_intent="pickup_performance",
            generated_sql="SELECT pickup_date, COUNT(*) AS pickup_count FROM pickups GROUP BY pickup_date;",
            sql_rows=[{"pickup_date": "2026-06-29", "pickup_count": 50, "completion_rate": 95.0}],
            business_metric="pickup_completion_rate",
            analysis_dimension="date",
            date_range="today",
        )
    if "compare yesterday" in normalized:
        return QueryResponse(
            question=question,
            answer="Yesterday pickup performance was lower than the previous period.",
            capability="sql",
            planning_intent="comparison",
            generated_sql="SELECT pickup_date, COUNT(*) AS pickup_count FROM pickups GROUP BY pickup_date;",
            sql_rows=[
                {"period": "yesterday", "pickup_count": 40},
                {"period": "previous_period", "pickup_count": 50},
            ],
            business_metric="pickup_count",
            analysis_dimension="date",
        )
    if "los angeles hub" in normalized:
        return QueryResponse(
            question=question,
            answer="Los Angeles Hub performance was 90%.",
            capability="sql",
            planning_intent="hub_performance",
            generated_sql="SELECT d.hub, COUNT(*) AS pickup_count FROM pickups p JOIN drivers d ON p.driver_id=d.driver_id WHERE d.hub='Los Angeles Hub';",
            sql_rows=[{"hub": "Los Angeles Hub", "pickup_count": 30}],
            business_metric="pickup_completion_rate",
            analysis_dimension="hub",
            analysis_filters={"hub": "Los Angeles Hub"},
        )
    if "highest performance" in normalized and "driver" in normalized:
        return QueryResponse(
            question=question,
            answer="Drew Nguyen is the highest performing driver globally.",
            capability="sql",
            planning_intent="driver_ranking",
            generated_sql=(
                "SELECT d.driver_name, COUNT(*) AS completed_pickups "
                "FROM pickups p JOIN drivers d ON p.driver_id=d.driver_id "
                "GROUP BY d.driver_name ORDER BY completed_pickups DESC LIMIT 1;"
            ),
            sql_rows=[{"driver_name": "Drew Nguyen", "completed_pickups": 33}],
            business_metric="pickup_completion_rate",
            analysis_dimension="driver",
        )
    if "atlanta hub" in normalized:
        return QueryResponse(
            question=question,
            answer="Atlanta Hub performance was weak.",
            capability="sql",
            planning_intent="hub_performance",
            generated_sql="SELECT d.hub, COUNT(*) AS pickup_count FROM pickups p JOIN drivers d ON p.driver_id=d.driver_id WHERE d.hub='Atlanta Hub';",
            sql_rows=[{"hub": "Atlanta Hub", "pickup_count": 5}],
            business_metric="pickup_completion_rate",
            analysis_dimension="hub",
            analysis_filters={"hub": "Atlanta Hub"},
        )
    if "chicago hub" in normalized and "performance" in normalized:
        return QueryResponse(
            question=question,
            answer="Chicago Hub performance was 82%.",
            capability="sql",
            planning_intent="hub_performance",
            generated_sql="SELECT d.hub, COUNT(*) AS pickup_count FROM pickups p JOIN drivers d ON p.driver_id=d.driver_id WHERE d.hub='Chicago Hub';",
            sql_rows=[{"hub": "Chicago Hub", "pickup_count": 10}],
            business_metric="pickup_completion_rate",
            analysis_dimension="hub",
            analysis_filters={"hub": "Chicago Hub"},
        )
    if "last week" in normalized:
        return QueryResponse(
            question=question,
            answer="Last week pickup performance was lower.",
            capability="sql",
            planning_intent="comparison",
            generated_sql="SELECT pickup_date, COUNT(*) AS pickup_count FROM pickups GROUP BY pickup_date;",
            sql_rows=[{"period": "last week", "pickup_count": 35}],
            business_metric="pickup_count",
            analysis_dimension="date",
        )
    return QueryResponse(
        question=question,
        answer="I don't know how to route this request.",
        capability="unknown",
        planning_intent="unknown",
    )
