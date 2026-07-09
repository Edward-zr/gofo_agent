"""Tests for business reasoning routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import QueryRequest, QueryResponse
from tools.router import route


@patch("tools.router.run_drilldown")
def test_why_performance_drop_runs_drilldown_analysis(mock_run_drilldown: MagicMock) -> None:
    mock_run_drilldown.return_value = QueryResponse(
        question="Why did performance drop?",
        answer="Operational performance decreased because delays increased.",
        capability="sql",
        execution_order=["sql", "root_cause", "recommendation"],
        root_cause={
            "issue": "Operational performance decreased.",
            "main_causes": ["Delay increased."],
            "affected_dimensions": ["hub: Chicago"],
            "recommendation": "Review driver assignment.",
        },
        recommendation="Review driver assignment.",
    )

    response = route(QueryRequest(question="Why did performance drop?"))

    assert response.answer == "Operational performance decreased because delays increased."
    assert response.root_cause is not None
    assert response.execution_order == ["sql", "root_cause", "recommendation"]
    mock_run_drilldown.assert_called_once_with("Why did performance drop?")
