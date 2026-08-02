"""Unit tests for TaskSpec decomposition and RequestVerifier."""

from __future__ import annotations

from core.intent_classifier import IntentClassification, IntentType
from core.intent_router import RouteIntent
from core.request_verifier import verify
from core.task_decomposition import TaskSpec, decompose


def test_task_spec_top_n_and_chart_type() -> None:
    spec = decompose(
        question="Show a pie chart of top 5 hubs by pickup rate",
        route_decision={
            "intent": RouteIntent.SQL_ANALYTICS,
            "chart_type": "pie",
            "use_attachments": False,
            "data_sources": ["SQLITE"],
        },
    )
    assert spec.domain == "SQL"
    assert spec.limit == 5
    assert spec.ranking_direction == "top"
    assert spec.chart_type == "pie"
    assert spec.output_type == "chart"
    assert spec.knowledge_source == "SQLITE"


def test_task_spec_sop_detach_no_attachment() -> None:
    spec = decompose(
        question="What is CBT?",
        classification=IntentClassification(
            intent=IntentType.SOP_QA,
            confidence=0.95,
            requires_rag=True,
        ),
        route_decision={
            "intent": RouteIntent.SOP_QA,
            "use_attachments": False,
            "detach_attachments": True,
            "data_sources": ["RAG"],
        },
    )
    assert spec.domain == "SOP"
    assert spec.attachment_relevant is False
    assert spec.knowledge_source == "RAG"
    assert spec.output_type == "knowledge_answer"


def test_verifier_sop_must_not_request_sql_retry() -> None:
    report = verify(
        task_spec=TaskSpec(
            domain="SOP",
            route_intent=RouteIntent.SOP_QA,
            knowledge_source="RAG",
        ),
        response={
            "answer": "No matching operational records were found.",
            "sql": "SELECT 'UNKNOWN' AS status",
            "data": [],
            "sources": [],
            "analysis": {"capability": "sql"},
        },
        route_decision={"intent": RouteIntent.SOP_QA},
    )
    assert report.passed is False
    assert report.should_retry_sql is False
    assert "SQL" not in [t.upper() for t in report.suggested_tool_calls]


def test_verifier_wrong_chart_type_fails() -> None:
    report = verify(
        task_spec=TaskSpec(
            domain="ADA",
            chart_type="pie",
            attachment_relevant=True,
        ),
        response={
            "answer": "Here is a bar chart.",
            "charts": [{"type": "bar", "image_base64": "x"}],
            "analysis": {"capability": "file_analysis", "attachment_ids": ["a1"]},
        },
    )
    assert report.passed is False
    assert report.chart_type_ok is False
    assert "ATTACHMENT" in report.suggested_tool_calls or "VISUALIZATION" in report.suggested_tool_calls


def test_verifier_attachment_irrelevant_sop_flags_ada_bind() -> None:
    report = verify(
        task_spec=TaskSpec(
            domain="SOP",
            attachment_relevant=False,
            route_intent=RouteIntent.SOP_QA,
        ),
        response={
            "answer": "SOP text",
            "sources": [{"chunk": "x"}],
            "analysis": {"capability": "file_analysis", "attachment_ids": ["a1"]},
        },
        route_decision={"intent": RouteIntent.SOP_QA},
    )
    assert report.attachment_ok is False
    assert report.should_retry_sql is False
    assert "RAG" in report.suggested_tool_calls


def test_verifier_happy_sop_passes() -> None:
    report = verify(
        task_spec=TaskSpec(
            domain="SOP",
            route_intent=RouteIntent.SOP_QA,
            attachment_relevant=False,
        ),
        response={
            "answer": "CBT means Collection by TikTok.",
            "sql": None,
            "data": [],
            "sources": [{"source": "sop.md"}],
            "analysis": {"capability": "rag"},
        },
        route_decision={"intent": RouteIntent.SOP_QA},
    )
    assert report.passed is True
    assert report.action == "approve"
