"""Tests for centralized GOFO intent routing."""

from __future__ import annotations

from core.intent_router import IntentRouter, RouteIntent, is_followup


router = IntentRouter()


def _state(**kwargs) -> dict:
    return {
        "last_route": None,
        "attachment_active": False,
        "last_topic": None,
        "last_intent": None,
        "last_attachment_file_types": ["excel"],
        **kwargs,
    }


def test_upload_then_what_is_in_file_routes_attachment() -> None:
    decision = router.route(
        "What is in this file?",
        _state(),
        new_attachment_ids=["file-1"],
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["use_attachments"] is True
    assert decision["handler"] == "Attachment Analyzer"


def test_upload_then_visualize_it_routes_attachment_visualization() -> None:
    decision = router.route(
        "Visualize it.",
        _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS, attachment_active=True),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_VISUALIZATION
    assert decision["followup"] is True
    assert decision["use_attachments"] is True
    assert decision["chart_type"] == "bar"


def test_upload_then_rank_all_hubs_routes_sql() -> None:
    decision = router.route(
        "Rank all hubs.",
        _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS, attachment_active=True),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.SQL_ANALYTICS
    assert decision["use_attachments"] is False
    assert decision["detach_attachments"] is True
    assert decision["attachment_active"] is False
    assert decision["handler"] == "SQL Planner"


def test_attachment_column_followup_stays_on_file() -> None:
    decision = router.route(
        "no, for 发件人详细地址 column, there are various of addresses, "
        "based on each addresses, I want to see how many packages for each addresses.",
        _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS, attachment_active=True),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["use_attachments"] is True
    assert decision["detach_attachments"] is False
    assert decision["attachment_active"] is True
    assert decision["handler"] == "Attachment Analyzer"


def test_upload_then_hi_routes_general_chat() -> None:
    decision = router.route(
        "Hi",
        _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS, attachment_active=True),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.GENERAL_CHAT
    assert decision["use_attachments"] is False
    assert decision["detach_attachments"] is True
    assert decision["handler"] == "OpenAI Chat"


def test_upload_then_weather_routes_general_chat() -> None:
    decision = router.route(
        "Tell me the weather",
        _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS, attachment_active=True),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.GENERAL_CHAT
    assert decision["use_attachments"] is False


def test_sql_then_why_routes_sql_followup() -> None:
    decision = router.route(
        "Why?",
        _state(last_route=RouteIntent.SQL_ANALYTICS, last_topic="ranking"),
    )
    assert decision["intent"] == RouteIntent.SQL_ANALYTICS
    assert decision["followup"] is True
    assert decision["handler"] == "SQL Planner"


def test_sop_then_explain_more_routes_sop_followup() -> None:
    decision = router.route(
        "Explain more",
        _state(last_route=RouteIntent.SOP_QA, last_topic="sop"),
    )
    assert decision["intent"] == RouteIntent.SOP_QA
    assert decision["followup"] is True
    assert decision["handler"] == "SOP Retriever"


def test_attachment_then_operations_recommendation_routes_attachment() -> None:
    decision = router.route(
        "What should operations do?",
        _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS, attachment_active=True),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["followup"] is True
    assert decision["use_attachments"] is True
    assert decision["handler"] == "Attachment Analyzer"


def test_sql_then_generate_chart_routes_sql_with_chart() -> None:
    decision = router.route(
        "Generate chart",
        _state(last_route=RouteIntent.SQL_ANALYTICS, last_topic="ranking"),
    )
    assert decision["intent"] == RouteIntent.SQL_ANALYTICS
    assert decision["followup"] is True
    # Chart type comes from this turn only — do not inherit last_topic="ranking".
    assert decision["chart_type"] == "bar"
    assert decision["handler"] == "SQL Planner"


def test_sop_question_routes_sop_qa() -> None:
    decision = router.route("What is the pickup procedure?", _state())
    assert decision["intent"] == RouteIntent.SOP_QA
    assert decision["handler"] == "SOP Retriever"


def test_is_followup_detects_short_pronoun_questions() -> None:
    assert is_followup("Visualize it", _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS)) is True
    assert is_followup("Rank all hubs", _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS)) is False


def test_ranking_visualization_selects_horizontal_bar() -> None:
    decision = router.route(
        "Plot ranking by hub",
        _state(last_route=RouteIntent.SQL_ANALYTICS, last_topic="ranking"),
    )
    assert decision["chart_type"] == "horizontal_bar"


def test_trend_visualization_selects_line_chart() -> None:
    decision = router.route(
        "Show trend over time",
        _state(last_route=RouteIntent.SQL_ANALYTICS, last_topic="trend"),
    )
    assert decision["chart_type"] == "line"


def test_meta_which_file_does_not_route_to_ada_summary() -> None:
    decision = router.route(
        "which file are you reading now, and which file are you reading previously?",
        _state(
            last_route=RouteIntent.ATTACHMENT_ANALYSIS,
            attachment_active=True,
            last_attachment_filenames=["order_1784414170040.xlsx"],
        ),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.GENERAL_CHAT
    assert decision["handler"] == "Attachment Context"
    assert decision["use_attachments"] is True
    assert decision["detach_attachments"] is False


def test_data_file_after_sop_routes_to_attachment_not_sql() -> None:
    decision = router.route(
        "in the data file, I just upload you, show me the address that has most packages volume.",
        _state(
            last_route=RouteIntent.SOP_QA,
            attachment_active=False,
            last_attachment_file_types=["excel"],
            last_attachment_filenames=["order_1784414170040.xlsx"],
        ),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["use_attachments"] is True
    assert decision["detach_attachments"] is False


def test_address_name_followup_after_sql_returns_to_attachment() -> None:
    decision = router.route(
        "which address is it, give me the address name",
        _state(
            last_route=RouteIntent.SQL_ANALYTICS,
            attachment_active=False,
            last_attachment_file_types=["excel"],
            last_attachment_filenames=["order_1784414170040.xlsx"],
        ),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["use_attachments"] is True


def test_what_is_cbt_during_file_session_routes_sop() -> None:
    decision = router.route(
        "what is cbt",
        _state(last_route=RouteIntent.ATTACHMENT_ANALYSIS, attachment_active=True),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.SOP_QA
    assert decision["use_attachments"] is False
    assert decision["detach_attachments"] is True
