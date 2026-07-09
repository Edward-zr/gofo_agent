"""Unit tests for tools.sql.summarizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.sql.summarizer import EMPTY_ANSWER, summarize


def test_summarize_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="Question must not be empty"):
        summarize("   ", "SELECT 1", [{"count": 1}])


def test_summarize_empty_rows_returns_default_message() -> None:
    answer = summarize("How many pickups yesterday?", "SELECT COUNT(*) FROM pickups", [])

    assert answer == EMPTY_ANSWER


@patch("tools.sql.summarizer.get_llm")
def test_summarize_success(mock_get_llm: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.content = "There were 42 pickups yesterday."
    mock_get_llm.return_value.invoke.return_value = mock_response

    rows = [{"total_pickups": 42}]
    answer = summarize(
        "How many pickups yesterday?",
        "SELECT COUNT(*) AS total_pickups FROM pickups",
        rows,
    )

    assert answer == "There were 42 pickups yesterday."
    mock_get_llm.return_value.invoke.assert_called_once()


@patch("tools.sql.summarizer.get_llm")
def test_summarize_llm_exception_propagates(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.side_effect = RuntimeError("API unavailable")

    with pytest.raises(RuntimeError, match="API unavailable"):
        summarize("How many pickups yesterday?", "SELECT COUNT(*) FROM pickups", [{"count": 1}])
