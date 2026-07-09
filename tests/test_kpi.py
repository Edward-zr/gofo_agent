"""Tests for KPI-backed operations overview."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import QueryRequest, QueryResponse
from tools.analysis.kpi import load_kpi_dictionary
from tools.router import route


def test_kpi_dictionary_loads_required_definitions() -> None:
    dictionary = load_kpi_dictionary()

    assert "pickup_completion_rate" in dictionary
    assert dictionary["pickup_completion_rate"]["formula"] == "completed pickups / total pickups"
    assert "hub" in dictionary["pickup_completion_rate"]["dimensions"]
    assert "delay_rate" in dictionary
    assert "failure_rate" in dictionary
    assert "package_volume" in dictionary


@patch("tools.router.answer_operations_overview")
def test_how_are_operations_uses_kpi_overview(mock_answer_operations: MagicMock) -> None:
    mock_answer_operations.return_value = QueryResponse(
        question="How are operations?",
        answer="Operational KPI summary.",
        capability="sql",
        planning_intent="operations_kpi_summary",
        kpi_summary={"checked_metrics": ["pickup_completion_rate", "delay_rate"]},
        execution_order=["sql", "kpi", "anomaly", "recommendation"],
    )

    response = route(QueryRequest(question="How are operations?"))

    assert response.answer == "Operational KPI summary."
    assert response.kpi_summary is not None
    assert "pickup_completion_rate" in response.kpi_summary["checked_metrics"]
    mock_answer_operations.assert_called_once_with("How are operations?")
