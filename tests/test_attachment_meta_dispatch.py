"""Tests for meta attachment context answers (not ADA executive summary)."""

from __future__ import annotations

from core.intent_router import IntentRouter, RouteIntent
from core.route_dispatcher import DispatchContext, RouteDispatcher
from tools.files.models import ProcessedFileContext, ProcessingStatus
from tools.orchestration.models import SemanticAnalysis
from tools.planner.intent_classifier import Intent


def test_dispatcher_answers_which_file_from_state() -> None:
    router = IntentRouter()
    route = router.route(
        "which file are you reading now, and which file are you reading previously?",
        {
            "last_route": RouteIntent.ATTACHMENT_ANALYSIS,
            "attachment_active": True,
            "last_attachment_filenames": ["order_1784414170040.xlsx"],
            "previous_attachment_filenames": ["pickup_old.xlsx"],
            "last_active_sheet": "waybill",
            "last_attachment_file_types": ["excel"],
        },
        has_stored_attachments=True,
    )
    context = ProcessedFileContext(
        attachment_id="att-1",
        filename="order_1784414170040.xlsx",
        file_type="excel",
        processing_status=ProcessingStatus.READY,
        summary="orders",
        source_references=["order_1784414170040.xlsx"],
    )
    response = RouteDispatcher().dispatch(
        DispatchContext(
            question="which file are you reading now, and which file are you reading previously?",
            resolved_question="which file are you reading now, and which file are you reading previously?",
            route=route,
            attachment_contexts=[context],
            file_context={"filenames": ["order_1784414170040.xlsx"]},
            attachment_ids=["att-1"],
            intent=Intent.UNKNOWN,
            conversation_state={
                "last_attachment_filenames": ["order_1784414170040.xlsx"],
                "previous_attachment_filenames": ["pickup_old.xlsx"],
                "last_active_sheet": "waybill",
            },
            semantic=SemanticAnalysis(
                domain="general_conversation",
                capability="conversation",
                response_mode="conversational",
            ),
        )
    )
    assert response is not None
    assert "Executive Summary" not in response.answer
    assert "order_1784414170040.xlsx" in response.answer
    assert "pickup_old.xlsx" in response.answer
    assert response.planning_intent == "FILE_CONTEXT"
