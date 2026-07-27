"""Tests for persisted attachment metadata across conversation turns."""

from __future__ import annotations

from core.agent import GOFOAgent
from core.intent_router import IntentRouter, RouteIntent
from core.models import QueryResponse
from tools.memory.state import ConversationState


router = IntentRouter()


def _state(**kwargs) -> dict:
    return {
        "last_route": None,
        "attachment_active": False,
        "last_topic": None,
        "last_intent": None,
        "last_attachment_file_types": [],
        "last_attachment_filenames": [],
        "last_active_sheet": None,
        "attachment_context_timestamp": None,
        **kwargs,
    }


def _ada_response(
    *,
    filenames: list[str] | None = None,
    file_types: list[str] | None = None,
    active_sheet: str | None = None,
    attachment_ids: list[str] | None = None,
) -> QueryResponse:
    summary: dict = {}
    if file_types is not None:
        summary["file_types"] = file_types
    if filenames is not None:
        summary["filenames"] = filenames
    if active_sheet is not None:
        summary["active_sheet"] = active_sheet
    return QueryResponse(
        question="inspect this file",
        answer="Executive summary of the uploaded file.",
        capability="file_analysis",
        planning_capability="file_analysis",
        attachment_ids=attachment_ids or ["att-1"],
        attachment_filenames=filenames or ["pickup_data.xlsx"],
        file_context_summary=summary or {
            "file_types": ["excel"],
            "filenames": ["pickup_data.xlsx"],
            "active_sheet": "Pickup_Data",
        },
    )


def test_upload_excel_then_followup_without_upload_routes_ada() -> None:
    state = _state(
        last_route=RouteIntent.ATTACHMENT_ANALYSIS,
        attachment_active=False,
        last_attachment_file_types=["excel"],
        last_attachment_filenames=["pickup_data.xlsx"],
        last_active_sheet="Pickup_Data",
    )
    decision = router.route(
        "What are the delayed pickups?",
        state,
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["use_attachments"] is True
    assert decision["detach_attachments"] is False


def test_upload_pdf_then_followup_routes_ada() -> None:
    state = _state(
        last_route=RouteIntent.ATTACHMENT_ANALYSIS,
        attachment_active=False,
        last_attachment_file_types=["pdf"],
        last_attachment_filenames=["ops_report.pdf"],
    )
    decision = router.route(
        "Summarize the key risks.",
        state,
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["use_attachments"] is True


def test_attachment_active_false_but_metadata_exists_keeps_ada() -> None:
    state = _state(
        last_route=RouteIntent.ATTACHMENT_ANALYSIS,
        attachment_active=False,
        last_attachment_file_types=["excel"],
        last_attachment_filenames=["pickup_data.xlsx"],
    )
    decision = router.route(
        "Continue analysis",
        state,
        has_stored_attachments=False,
    )
    assert decision["intent"] == RouteIntent.ATTACHMENT_ANALYSIS
    assert decision["use_attachments"] is True
    assert decision["attachment_active"] is True


def test_empty_metadata_does_not_overwrite_previous() -> None:
    state = ConversationState()
    state.update(
        user_question="inspect this file",
        resolved_question="inspect this file",
        response=_ada_response(
            filenames=["pickup_data.xlsx"],
            file_types=["excel"],
            active_sheet="Pickup_Data",
        ),
        route_decision={
            "intent": RouteIntent.ATTACHMENT_ANALYSIS,
            "attachment_active": True,
        },
    )
    assert state.last_attachment_file_types == ["excel"]
    assert state.last_attachment_filenames == ["pickup_data.xlsx"]
    assert state.last_active_sheet == "Pickup_Data"
    assert state.attachment_context_timestamp

    empty_response = QueryResponse(
        question="follow up",
        answer="ok",
        capability="file_analysis",
        attachment_ids=[],
        attachment_filenames=[],
        file_context_summary={"file_types": [], "filenames": [], "active_sheet": None},
    )
    state.update(
        user_question="follow up",
        resolved_question="follow up",
        response=empty_response,
        route_decision={
            "intent": RouteIntent.ATTACHMENT_ANALYSIS,
            "attachment_active": False,
        },
    )
    assert state.attachment_active is False
    assert state.last_attachment_file_types == ["excel"]
    assert state.last_attachment_filenames == ["pickup_data.xlsx"]
    assert state.last_active_sheet == "Pickup_Data"


def test_new_upload_replaces_previous_metadata() -> None:
    state = ConversationState()
    state.update(
        user_question="inspect this file",
        resolved_question="inspect this file",
        response=_ada_response(
            filenames=["pickup_data.xlsx"],
            file_types=["excel"],
            active_sheet="Pickup_Data",
            attachment_ids=["att-old"],
        ),
        route_decision={
            "intent": RouteIntent.ATTACHMENT_ANALYSIS,
            "attachment_active": True,
        },
    )
    state.update(
        user_question="inspect this file",
        resolved_question="inspect this file",
        response=_ada_response(
            filenames=["inventory.xlsx"],
            file_types=["excel"],
            active_sheet="Stock",
            attachment_ids=["att-new"],
        ),
        route_decision={
            "intent": RouteIntent.ATTACHMENT_ANALYSIS,
            "attachment_active": True,
        },
    )
    assert state.last_attachment == "att-new"
    assert state.last_attachment_filenames == ["inventory.xlsx"]
    assert state.last_active_sheet == "Stock"


def test_worksheet_switching_updates_last_active_sheet() -> None:
    state = ConversationState()
    state.update(
        user_question="inspect this file",
        resolved_question="inspect this file",
        response=_ada_response(active_sheet="Pickup_Data"),
        route_decision={
            "intent": RouteIntent.ATTACHMENT_ANALYSIS,
            "attachment_active": True,
        },
    )
    assert state.last_active_sheet == "Pickup_Data"

    state.update(
        user_question="Summarize this sheet.",
        resolved_question="Summarize this sheet.",
        response=_ada_response(
            filenames=["pickup_data.xlsx"],
            file_types=["excel"],
            active_sheet="Exceptions",
        ),
        route_decision={
            "intent": RouteIntent.ATTACHMENT_ANALYSIS,
            "attachment_active": True,
        },
    )
    assert state.last_active_sheet == "Exceptions"
    assert state.last_attachment_filenames == ["pickup_data.xlsx"]


def test_conversation_reset_clears_metadata() -> None:
    agent = GOFOAgent()
    agent.state.update(
        user_question="inspect this file",
        resolved_question="inspect this file",
        response=_ada_response(),
        route_decision={
            "intent": RouteIntent.ATTACHMENT_ANALYSIS,
            "attachment_active": True,
        },
    )
    assert agent.state.has_persisted_attachment_metadata() is True

    agent.clear_attachment_context()

    assert agent.state.attachment_active is False
    assert agent.state.last_attachment_file_types == []
    assert agent.state.last_attachment_filenames == []
    assert agent.state.last_active_sheet is None
    assert agent.state.attachment_context_timestamp is None
    assert agent.state.has_persisted_attachment_metadata() is False


def test_explicit_rank_all_hubs_still_detaches_to_sql() -> None:
    decision = router.route(
        "Rank all hubs.",
        _state(
            last_route=RouteIntent.ATTACHMENT_ANALYSIS,
            attachment_active=False,
            last_attachment_file_types=["excel"],
            last_attachment_filenames=["pickup_data.xlsx"],
        ),
        has_stored_attachments=True,
    )
    assert decision["intent"] == RouteIntent.SQL_ANALYTICS
    assert decision["detach_attachments"] is True
    assert decision["use_attachments"] is False


def test_snapshot_includes_persisted_attachment_fields() -> None:
    state = ConversationState(
        last_attachment_file_types=["excel"],
        last_attachment_filenames=["pickup_data.xlsx"],
        last_active_sheet="Pickup_Data",
        attachment_context_timestamp="2026-07-26T12:00:00+00:00",
    )
    snap = state.snapshot()
    assert snap["last_attachment_file_types"] == ["excel"]
    assert snap["last_attachment_filenames"] == ["pickup_data.xlsx"]
    assert snap["last_active_sheet"] == "Pickup_Data"
    assert snap["attachment_context_timestamp"] == "2026-07-26T12:00:00+00:00"
