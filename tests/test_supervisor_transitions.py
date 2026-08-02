"""Regression tests for Supervisor capability / chart / attachment transitions."""

from __future__ import annotations

import pandas as pd

from core.intent_router import IntentRouter, RouteIntent
from tools.files.charts import build_charts


router = IntentRouter()


def _ada_state(**kwargs) -> dict:
    base = {
        "last_route": RouteIntent.ATTACHMENT_ANALYSIS,
        "attachment_active": True,
        "last_attachment_file_types": ["excel"],
        "last_attachment_filenames": ["orders.xlsx"],
        "last_visualization": "bar",
        "last_topic": RouteIntent.ATTACHMENT_ANALYSIS,
    }
    base.update(kwargs)
    return base


def test_sop_after_ada_detaches_to_rag() -> None:
    for question in (
        "What is CBT?",
        "What is pickup SOP?",
        "What is TikTok Collection?",
        "What is Collection by TikTok?",
    ):
        decision = router.route(
            question,
            _ada_state(),
            has_stored_attachments=True,
        )
        assert decision["intent"] == RouteIntent.SOP_QA, question
        assert decision["use_attachments"] is False, question
        assert decision["detach_attachments"] is True, question
        assert decision["data_sources"] == ["RAG"], question


def test_capability_transitions_are_independent() -> None:
    """SOP → ADA → SOP → SQL → Chat each evaluate independently."""
    state = {
        "last_route": None,
        "attachment_active": False,
        "last_attachment_file_types": [],
        "last_attachment_filenames": [],
    }

    sop = router.route("What is CBT?", state)
    assert sop["intent"] == RouteIntent.SOP_QA
    state["last_route"] = sop["intent"]

    ada = router.route(
        "Summarize this spreadsheet",
        state,
        new_attachment_ids=["f1"],
        has_stored_attachments=True,
    )
    assert ada["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert ada["use_attachments"] is True
    state.update(
        {
            "last_route": ada["intent"],
            "attachment_active": True,
            "last_attachment_filenames": ["sheet.xlsx"],
            "last_attachment_file_types": ["excel"],
        }
    )

    sop2 = router.route("What is CBT?", state, has_stored_attachments=True)
    assert sop2["intent"] == RouteIntent.SOP_QA
    assert sop2["detach_attachments"] is True
    state["last_route"] = sop2["intent"]
    state["attachment_active"] = False

    sql = router.route("Rank all hubs by pickups today", state)
    assert sql["intent"] == RouteIntent.SQL_ANALYTICS
    state["last_route"] = sql["intent"]

    chat = router.route("Hi", state)
    assert chat["intent"] == RouteIntent.GENERAL_CHAT


def test_explicit_chart_type_overrides_previous_bar() -> None:
    state = _ada_state(last_visualization="bar", last_topic="ranking top hubs")
    pie = router.route("Use pie chart.", state, has_stored_attachments=True)
    assert pie["intent"] == RouteIntent.ATTACHMENT_VISUALIZATION
    assert pie["chart_type"] == "pie"

    line = router.route("Show a line chart.", state, has_stored_attachments=True)
    assert line["chart_type"] == "line"

    bar = router.route("Make a bar chart.", state, has_stored_attachments=True)
    assert bar["chart_type"] == "bar"


def test_build_charts_honors_explicit_pie_over_prior_bar_bias() -> None:
    frame = pd.DataFrame(
        {
            "hub": ["A", "B", "C"],
            "packages": [10, 20, 30],
        }
    )
    charts = build_charts(frame, question="Use pie chart.", max_charts=2)
    assert charts
    assert charts[0]["type"] == "pie"

    charts_line = build_charts(frame, question="line chart please", max_charts=2)
    assert charts_line
    assert charts_line[0]["type"] == "line"


def test_file_column_followup_still_stays_on_ada() -> None:
    decision = router.route(
        "for 发件人详细地址 column, how many packages for each address?",
        _ada_state(),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["use_attachments"] is True
    assert decision["detach_attachments"] is False
