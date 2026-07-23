"""Unit tests for tools.sql.service."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from core.models import QueryRequest
from tools.sql.service import answer


@patch("tools.sql.service.summarize")
@patch("tools.sql.service.execute")
@patch("tools.sql.service.plan")
@patch("tools.sql.service.resolve_relative_dates")
@patch("tools.sql.service.get_latest_business_date")
def test_answer_runs_date_resolver_planner_executor_and_summarizer(
    mock_get_latest_business_date: MagicMock,
    mock_resolve_relative_dates: MagicMock,
    mock_plan: MagicMock,
    mock_execute: MagicMock,
    mock_summarize: MagicMock,
) -> None:
    mock_get_latest_business_date.return_value = date(2026, 6, 28)
    mock_resolve_relative_dates.return_value = "Top customers in June 2026"
    mock_plan.return_value = "SELECT customer_name FROM customers"
    mock_execute.return_value = [{"customer_name": "Amazon"}]
    mock_summarize.return_value = "Amazon is a top customer."

    response = answer(QueryRequest(question="Top customers this month"))

    assert response.capability == "sql"
    assert response.sources == []
    assert response.answer == "Amazon is a top customer."
    assert response.latest_business_date == "2026-06-28"
    assert response.rewritten_question == "Top customers in June 2026"
    mock_get_latest_business_date.assert_called_once()
    mock_resolve_relative_dates.assert_called_once_with(
        "Top customers this month",
        date(2026, 6, 28),
    )
    mock_plan.assert_called_once()
    assert mock_plan.call_args.args[0] == "Top customers in June 2026"
    assert mock_plan.call_args.kwargs.get("retrieved_schema") is not None
    mock_execute.assert_called_once_with(mock_plan.return_value)
    mock_summarize.assert_called_once_with(
        "Top customers this month",
        mock_plan.return_value,
        mock_execute.return_value,
    )
    assert response.retrieved_schema is not None
    assert response.candidate_tables is not None


def test_answer_empty_question_raises() -> None:
    request = QueryRequest.model_construct(question="   ")

    with pytest.raises(ValueError, match="Question must not be empty"):
        answer(request)
