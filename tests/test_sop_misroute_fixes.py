"""Regression tests for SOP misroute / clarification bugs (screenshot cases)."""

from __future__ import annotations

from core.clarification_manager import (
    ClarificationManager,
    ClarificationOption,
    PendingClarification,
    is_clarification_answer,
    looks_like_new_domain_question,
)
from core.data_source_selector import DataSource, DataSourceSelector
from core.intent_classifier import IntentClassification, IntentClassifier, IntentType
from core.intent_router import IntentRouter, RouteIntent


router = IntentRouter()
classifier = IntentClassifier()
selector = DataSourceSelector()


def test_driver_responsibility_routes_to_sop() -> None:
    for question in (
        "what should driver do?",
        "what should the driver do?",
        "what is the driver responsible for?",
        "What are driver responsibilities?",
    ):
        decision = router.route(question, {"last_route": None})
        assert decision["intent"] == RouteIntent.SOP_QA, question
        assert decision["data_sources"] == ["RAG"], question


def test_tell_me_more_about_cbt_routes_to_sop() -> None:
    decision = router.route(
        "tell me more about the CBT",
        {"last_route": RouteIntent.SQL_ANALYTICS},
    )
    assert decision["intent"] == RouteIntent.SOP_QA
    assert "RAG" in decision["data_sources"]


def test_tell_me_more_after_sop_inherits_sop() -> None:
    decision = router.route(
        "tell me more",
        {"last_route": RouteIntent.SOP_QA},
    )
    assert decision["intent"] == RouteIntent.SOP_QA
    assert decision["followup"] is True


def test_pending_clarification_cancelled_for_cbt_ask() -> None:
    pending = PendingClarification(
        original_question="show performance",
        missing_fields=["date_range"],
        pending_question="Which time range?",
        options=[
            ClarificationOption(id="today", label="Today"),
            ClarificationOption(id="this_week", label="This week"),
            ClarificationOption(id="all_historical", label="All historical data"),
        ],
        ambiguity_type="planner",
        resolved_base="show performance",
    )
    assert looks_like_new_domain_question("tell me more about the CBT")
    assert is_clarification_answer(pending, "tell me more about the CBT") is False
    decision = ClarificationManager().apply_user_response(
        pending, "tell me more about the CBT"
    )
    assert decision.resumed_question is None
    assert decision.skip_reason and "cancel" in decision.skip_reason.lower()


def test_date_answer_still_resumes_clarification() -> None:
    pending = PendingClarification(
        original_question="rank hubs by pickup rate",
        missing_fields=["date_range"],
        pending_question="Which time range?",
        options=[
            ClarificationOption(id="today", label="Today"),
            ClarificationOption(id="this_week", label="This week"),
        ],
        ambiguity_type="time_range",
        resolved_base="rank hubs by pickup rate",
    )
    assert is_clarification_answer(pending, "this week") is True
    decision = ClarificationManager().apply_user_response(pending, "this week")
    assert decision.resumed_question
    assert "week" in (decision.resumed_question or "").lower() or decision.filled_slots.get(
        "date_range"
    )


def test_classifier_sop_for_driver_and_cbt_followup() -> None:
    for question in (
        "what should driver do?",
        "what is the driver responsible for?",
        "tell me more about the CBT",
    ):
        result = classifier.classify(question, None)
        assert result.intent in {
            IntentType.SOP_QA,
            IntentType.SOP_Summary,
            IntentType.SOP_Compare,
        }, question


def test_selector_sop_signals_skip_date_clarification() -> None:
    for question, classification in (
        (
            "tell me more about the CBT",
            IntentClassification(intent=IntentType.Unknown, confidence=0.2, requires_clarification=True),
        ),
        (
            "what should driver do?",
            IntentClassification(intent=IntentType.Unknown, confidence=0.3),
        ),
        (
            "What is CBT?",
            IntentClassification(intent=IntentType.SOP_QA, confidence=0.95, requires_rag=True),
        ),
    ):
        selection = selector.select(question, classification)
        assert selection.requires_clarification is False, question
        assert DataSource.RAG.value in selection.selected_sources, (question, selection)


def test_sql_ranking_still_can_clarify() -> None:
    """Ranking analytics should still be allowed to ask for time when appropriate."""
    decision = router.route("rank top hubs by pickup rate", {})
    assert decision["intent"] == RouteIntent.SQL_ANALYTICS
