"""Unit tests for frontier problem-type decision tree."""

from __future__ import annotations

from core.intent_router import RouteIntent
from graph.frontier import decide_frontier


def test_frontier_sop_cbt() -> None:
    d = decide_frontier(
        question="What is CBT?",
        route_decision={"intent": RouteIntent.SOP_QA},
    )
    assert d.problem_kind == "sop_searchable"
    assert "RAG" in d.tools


def test_frontier_driver_responsibility_sop() -> None:
    d = decide_frontier(
        question="what is the driver responsible for?",
        route_decision={"intent": RouteIntent.SOP_QA},
    )
    assert d.problem_kind == "sop_searchable"


def test_frontier_sql_analysis_tools() -> None:
    d = decide_frontier(
        question="rank top hubs by pickup rate today",
        route_decision={"intent": RouteIntent.SQL_ANALYTICS},
    )
    assert d.problem_kind == "ada_analysis"
    assert d.requires_analysis is True
    assert "SQL" in d.tools


def test_frontier_attachment_python() -> None:
    d = decide_frontier(
        question="plot a chart of this file by hub",
        route_decision={"intent": RouteIntent.ATTACHMENT_VISUALIZATION, "use_attachments": True},
        attachment_ids=["a1"],
        attachment_active=True,
    )
    assert d.problem_kind == "ada_analysis"
    assert "ATTACHMENT" in d.tools


def test_frontier_not_meaningful() -> None:
    d = decide_frontier(
        question="what is the meaning of life?",
        route_decision={"intent": RouteIntent.OPENAI_FALLBACK},
    )
    # Chat pass-through — not "not_meaningful"
    assert d.problem_kind == "ada_analysis"
    assert "LLM" in d.tools


def test_frontier_true_not_meaningful_without_route() -> None:
    d = decide_frontier(
        question="asdf qwerty zxcv",
        route_decision={"intent": ""},
    )
    assert d.problem_kind == "not_meaningful"
    assert d.not_meaningful_message
