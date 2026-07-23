"""Unit tests for tools.rag.rewriter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.rag.rewriter import rewrite


def test_rewrite_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="Question must not be empty"):
        rewrite("   ")


@patch("tools.rag.rewriter.get_prompt_manager")
def test_rewrite_success(mock_get_manager: MagicMock) -> None:
    mock_get_manager.return_value.invoke.return_value = "What is Collection by TikTok (CBT)?"

    rewritten = rewrite("CBT")

    assert rewritten == "What is Collection by TikTok (CBT)?"
    mock_get_manager.return_value.invoke.assert_called_once()


@patch("tools.rag.rewriter.get_prompt_manager")
def test_rewrite_failure_returns_original(mock_get_manager: MagicMock) -> None:
    mock_get_manager.return_value.invoke.side_effect = RuntimeError("API unavailable")

    rewritten = rewrite("what photos")

    assert rewritten == "what photos"


@patch("tools.rag.rewriter.get_prompt_manager")
def test_rewrite_empty_model_output_returns_original(mock_get_manager: MagicMock) -> None:
    # PromptManager.invoke returns stripped text; empty → keep original question.
    mock_get_manager.return_value.invoke.return_value = ""

    rewritten = rewrite("pickup KPI")

    assert rewritten == "pickup KPI"
