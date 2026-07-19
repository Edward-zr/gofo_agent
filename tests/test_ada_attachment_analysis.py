"""Tests for ChatGPT-ADA style attachment analysis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.agent import GOFOAgent
from core.intent_router import IntentRouter, RouteIntent
from tools.files.analysis_intent import AnalysisIntent, WAIT_FOR_UPLOAD_REPLY, detect_analysis_intent
from tools.files.charts import build_charts
from tools.files.data_analysis import analyze_dataframe
from tools.files.dataframe_store import context_to_stored_attachment
from tools.files.models import AttachmentMetadata, ProcessedFileContext, ProcessingStatus
from tools.files.service import AttachmentService
from tools.files.storage import AttachmentStore


CSV_CONTENT = """hub,pickup_count,package_count,delayed_pickups
Chicago Hub,10,120,7
New York Hub,100,900,2
Atlanta Hub,25,210,4
""".encode()


def _csv_context() -> ProcessedFileContext:
    rows = [
        {"hub": "Chicago Hub", "pickup_count": 10, "package_count": 120, "delayed_pickups": 7},
        {"hub": "New York Hub", "pickup_count": 100, "package_count": 900, "delayed_pickups": 2},
        {"hub": "Atlanta Hub", "pickup_count": 25, "package_count": 210, "delayed_pickups": 4},
    ]
    return ProcessedFileContext(
        attachment_id="att-1",
        filename="pickup_report.csv",
        file_type="csv",
        processing_status=ProcessingStatus.READY,
        summary="CSV file with 3 rows and 4 columns.",
        file_schema={"columns": ["hub", "pickup_count", "package_count", "delayed_pickups"]},
        statistics={"row_count": 3, "column_count": 4},
        sample_rows=rows,
        full_data={"rows": rows, "columns": ["hub", "pickup_count", "package_count", "delayed_pickups"]},
        source_references=["pickup_report.csv"],
    )


def test_wait_for_upload_intent() -> None:
    assert detect_analysis_intent("I will upload a file") == AnalysisIntent.WAIT_FOR_UPLOAD
    assert detect_analysis_intent("I want you to analyze an Excel") == AnalysisIntent.WAIT_FOR_UPLOAD


def test_intent_router_waits_before_file_exists() -> None:
    decision = IntentRouter().route("I will upload a file", {})
    assert decision["intent"] == RouteIntent.WAIT_FOR_UPLOAD
    assert decision["use_attachments"] is False


def test_parse_once_stores_dataframe() -> None:
    stored = context_to_stored_attachment(_csv_context())
    assert stored.is_tabular
    assert list(stored.columns) == ["hub", "pickup_count", "package_count", "delayed_pickups"]
    assert len(stored.dataframe) == 3


def test_executive_summary_is_data_driven() -> None:
    stored = context_to_stored_attachment(_csv_context())
    result = analyze_dataframe(
        question="Summarize",
        stored=stored,
        intent=AnalysisIntent.EXECUTIVE_SUMMARY,
    )
    assert "Rows: 3" in result["answer"]
    assert "Columns: 4" in result["answer"]
    assert "Operational metric requires investigation" not in result["answer"]
    assert result["charts"]


def test_ranking_uses_same_dataframe() -> None:
    stored = context_to_stored_attachment(_csv_context())
    result = analyze_dataframe(
        question="Which hub is worst?",
        stored=stored,
        intent=AnalysisIntent.RANKING,
    )
    assert "Chicago Hub" in result["answer"]
    assert result["rows"][0]["hub"] == "Chicago Hub"


def test_filter_then_rank_uses_filtered_frame() -> None:
    stored = context_to_stored_attachment(_csv_context())
    filtered = analyze_dataframe(
        question="Only Chicago",
        stored=stored,
        intent=AnalysisIntent.FILTER,
    )
    assert filtered["active_filter"]
    assert all("Chicago" in str(row["hub"]) for row in filtered["rows"])


def test_visualize_generates_chart_specs() -> None:
    stored = context_to_stored_attachment(_csv_context())
    result = analyze_dataframe(
        question="Visualize",
        stored=stored,
        intent=AnalysisIntent.VISUALIZE,
    )
    assert result["charts"]
    chart = result["charts"][0]
    assert chart["type"] in {"bar", "line", "pie", "scatter", "histogram", "heatmap"}
    assert chart.get("renderer") == "matplotlib"
    assert chart.get("image_base64")
    assert chart.get("format") == "png"
    assert chart["data"]


def test_build_charts_chooses_bar_for_categorical_numeric() -> None:
    frame = pd.DataFrame(
        {
            "hub": ["Chicago Hub", "New York Hub"],
            "pickup_count": [10, 100],
        }
    )
    charts = build_charts(frame, question="show charts")
    bar_charts = [chart for chart in charts if chart["type"] == "bar"]
    assert bar_charts
    assert all(chart.get("image_base64") for chart in bar_charts)
    assert all(chart.get("renderer") == "matplotlib" for chart in bar_charts)


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_agent_wait_for_upload_reply(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
) -> None:
    mock_extract_entities.return_value = {}
    agent = GOFOAgent()
    response = agent.ask("I will upload a file")
    assert WAIT_FOR_UPLOAD_REPLY.splitlines()[0] in response["answer"]
    assert "Upload the file" in response["answer"]
    mock_ask_core.assert_not_called()


@patch("tools.memory.conversation.extract_entities")
@patch("core.route_dispatcher.ask_core")
def test_agent_fresh_analysis_per_prompt(
    mock_ask_core: MagicMock,
    mock_extract_entities: MagicMock,
    tmp_path,
) -> None:
    mock_extract_entities.return_value = {}
    mock_ask_core.side_effect = AssertionError("ask_core should not run for attachment ADA")
    service = AttachmentService(store=AttachmentStore(base_dir=tmp_path / "uploads"))
    agent = GOFOAgent()
    agent.attachment_service = service

    metadata = service.upload(
        filename="pickup_report.csv",
        content=CSV_CONTENT,
        content_type="text/csv",
        conversation_id="ada-session",
    )
    first = agent.ask("Summarize", attachment_ids=[metadata.attachment_id])
    second = agent.ask("Which hub is worst?")
    third = agent.ask("Visualize")

    assert "Rows: 3" in first["answer"]
    assert "Operational metric requires investigation" not in first["answer"]
    assert "Chicago Hub" in second["answer"]
    assert third.get("charts")
    assert first["answer"] != second["answer"]
    assert second["answer"] != third["answer"]
