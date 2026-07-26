"""Unit tests for semantic conversation resolver."""

from __future__ import annotations

from core.models import QueryResponse
from tools.conversation import ConversationResolver


def _ranking_response() -> QueryResponse:
    return QueryResponse(
        question="Rank all hubs by performance",
        answer="New York Hub is best. Chicago Hub is worst.",
        capability="sql",
        planning_intent="hub_ranking",
        generated_sql="SELECT d.hub, COUNT(*) AS pickup_count FROM pickups p JOIN drivers d ON p.driver_id=d.driver_id GROUP BY d.hub ORDER BY pickup_count DESC;",
        sql_rows=[
            {"hub": "New York Hub", "pickup_count": 100, "delayed_pickups": 2},
            {"hub": "Chicago Hub", "pickup_count": 10, "delayed_pickups": 7},
        ],
        business_metric="pickup_completion_rate",
        analysis_dimension="hub",
    )


def test_worst_hub_resolves_from_semantic_ranking() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Rank all hubs by performance",
        resolved_question="Rank all hubs by performance",
        response=_ranking_response(),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("Why is the worst hub performing badly?")

    assert "Chicago Hub" in resolution.resolved_question
    assert resolution.intent_hint == "ROOT_CAUSE"


def test_recommendation_followup_uses_previous_analysis() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Rank all hubs by performance",
        resolved_question="Rank all hubs by performance",
        response=_ranking_response(),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("What should operations do?")

    assert resolution.intent_hint == "RECOMMENDATION"
    assert resolution.use_previous_analysis is True
    assert "Chicago Hub" in resolution.resolved_question


def test_compare_yesterday_inherits_metric_not_hub_filter() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Show today's pickup performance",
        resolved_question="Show today's pickup performance",
        response=QueryResponse(
            question="Show today's pickup performance",
            answer="Today pickup performance is stable.",
            capability="sql",
            business_metric="pickup_completion_rate",
            analysis_dimension="date",
            date_range="today",
            analysis_filters={"hub": "Los Angeles Hub"},
            sql_rows=[{"pickup_date": "2026-06-29", "pickup_count": 50}],
        ),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("Compare yesterday")

    assert "pickup performance" in resolution.resolved_question.lower()
    assert "Los Angeles" not in resolution.resolved_question
    assert resolution.intent_hint == "COMPARISON"


def test_fresh_driver_ranking_does_not_inherit_hub_filter() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Show Los Angeles Hub performance",
        resolved_question="Show Los Angeles Hub performance",
        response=QueryResponse(
            question="Show Los Angeles Hub performance",
            answer="Los Angeles Hub performance was 90%.",
            capability="sql",
            analysis_dimension="hub",
            analysis_filters={"hub": "Los Angeles Hub"},
            sql_rows=[{"hub": "Los Angeles Hub", "pickup_count": 30}],
        ),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("Which driver has the highest performance?")

    assert resolution.resolved_question == "Which driver has the highest performance?"
    assert "Los Angeles" not in resolution.resolved_question
    assert resolution.filters == {}


def test_correction_changes_ranking_direction() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Highest performing driver",
        resolved_question="Highest performing driver",
        response=QueryResponse(
            question="Highest performing driver",
            answer="Drew Nguyen is the highest performing driver.",
            capability="sql",
            analysis_dimension="driver",
            sql_rows=[{"driver_name": "Drew Nguyen", "completed_pickups": 33}],
        ),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("Actually I mean lowest driver")

    assert resolution.repair_detected is True
    assert "lowest" in resolution.resolved_question.lower()
    assert "Drew" not in resolution.resolved_question


def test_no_prefix_keeps_file_column_aggregation_ask() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="inspect this file",
        resolved_question="inspect this file",
        response=QueryResponse(
            question="inspect this file",
            answer="Executive Summary",
            capability="file_analysis",
            planning_intent="EXECUTIVE_SUMMARY",
        ),
        intent="ATTACHMENT_ANALYSIS",
    )

    resolution = resolver.resolve(
        "no, for 发件人详细地址 column, there are various of addresses, "
        "based on each addresses, I want to see how many packages for each addresses."
    )

    assert resolution.repair_detected is True
    assert "发件人详细地址" in resolution.resolved_question
    assert "inspect this file" not in resolution.resolved_question.lower()
    assert "by packages" not in resolution.resolved_question.lower()


def test_why_resolves_from_previous_topic() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Rank all hubs by performance",
        resolved_question="Rank all hubs by performance",
        response=_ranking_response(),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("Why?")

    assert "Chicago Hub" in resolution.resolved_question
    assert resolution.intent_hint == "ROOT_CAUSE"


def test_show_details_uses_previous_result() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Rank all hubs by performance",
        resolved_question="Rank all hubs by performance",
        response=_ranking_response(),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("Show details")

    assert resolution.intent_hint == "DRILLDOWN"
    assert resolution.use_previous_result is True


def test_which_one_inherits_dimension_from_state() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Rank all hubs by performance",
        resolved_question="Rank all hubs by performance",
        response=_ranking_response(),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("Which one has the most delayed pickups?")

    assert "hub" in resolution.resolved_question.lower()
    assert "delayed" in resolution.resolved_question.lower()


def test_what_happened_today_maps_to_executive_summary() -> None:
    resolver = ConversationResolver()

    resolution = resolver.resolve("What happened today?")

    assert resolution.intent_hint == "SUMMARY"
    assert "executive" in resolution.resolved_question.lower()


def test_hub_repair_replaces_target_hub() -> None:
    resolver = ConversationResolver()
    resolver.update(
        original_question="Show Atlanta hub performance",
        resolved_question="Show Atlanta hub performance",
        response=QueryResponse(
            question="Show Atlanta hub performance",
            answer="Atlanta Hub performance was weak.",
            capability="sql",
            analysis_dimension="hub",
            analysis_filters={"hub": "Atlanta Hub"},
            sql_rows=[{"hub": "Atlanta Hub", "pickup_count": 5}],
        ),
        intent="SQL_QUERY",
    )

    resolution = resolver.resolve("No, I mean Chicago Hub")

    assert resolution.repair_detected is True
    assert "Chicago Hub" in resolution.resolved_question
    assert resolution.filters.get("hub") == "Chicago Hub"


def test_sql_cache_reuses_identical_request() -> None:
    resolver = ConversationResolver()
    response = _ranking_response()
    resolver.update(
        original_question="Rank all hubs by performance",
        resolved_question="Rank all hubs by performance",
        response=response,
        intent="SQL_QUERY",
        cache_key=("SQL_QUERY", "rank all hubs by performance", "pickup_completion_rate", "hub", (), None),
    )

    resolution = resolver.resolve("Rank all hubs by performance")
    cached = resolver.cached_response(resolution)

    assert cached is not None
    assert cached.answer == response.answer
