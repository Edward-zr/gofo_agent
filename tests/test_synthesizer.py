"""Unit tests for multi-tool synthesizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import QueryResponse
from tools.synthesizer import synthesize


@patch("tools.synthesizer.get_llm")
def test_synthesize_sql_only(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="There were 42 pickups yesterday."
    )
    sql_result = QueryResponse(
        question="How many pickups yesterday?",
        answer="There were 42 pickups yesterday.",
        capability="sql",
    )

    answer = synthesize("How many pickups yesterday?", sql_result, None)

    assert answer == "There were 42 pickups yesterday."
    mock_get_llm.return_value.invoke.assert_called_once()


@patch("tools.synthesizer.get_llm")
def test_synthesize_rag_only(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="CBT means Collection by TikTok."
    )
    rag_result = QueryResponse(
        question="What is CBT?",
        answer="CBT means Collection by TikTok.",
        capability="rag",
    )

    answer = synthesize("What is CBT?", None, rag_result)

    assert answer == "CBT means Collection by TikTok."


@patch("tools.synthesizer.get_llm")
def test_synthesize_both_tools(mock_get_llm: MagicMock) -> None:
    mock_get_llm.return_value.invoke.return_value = MagicMock(
        content="There are 55 delayed pickups today; follow the delayed pickup SOP."
    )
    sql_result = QueryResponse(
        question="What should I do if delayed pickups today exceed 50?",
        answer="There are 55 delayed pickups today.",
        capability="sql",
    )
    rag_result = QueryResponse(
        question="What should I do if delayed pickups today exceed 50?",
        answer="Escalate using the delayed pickup SOP.",
        capability="rag",
    )

    answer = synthesize(
        "What should I do if delayed pickups today exceed 50?",
        sql_result,
        rag_result,
    )

    assert "55 delayed pickups" in answer
    prompt = mock_get_llm.return_value.invoke.call_args.args[0][0].content
    assert "Operational analytics result:" in prompt
    assert "SOP knowledge result:" in prompt


def test_synthesize_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="Question must not be empty"):
        synthesize("   ", None, None)
