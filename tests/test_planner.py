"""Unit tests for natural-language planner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.planner import plan


def _mock_planner_response(mock_get_manager: MagicMock, content: str) -> None:
    """Configure the mocked PromptManager.invoke response."""
    mock_get_manager.return_value.invoke.return_value = content.strip()


@patch("core.prompt_manager.get_prompt_manager")
def test_plan_sql_question(mock_get_manager: MagicMock) -> None:
    _mock_planner_response(
        mock_get_manager,
        """
        {
          "capability": "sql",
          "intent": "pickup_count",
          "confidence": 0.94,
          "requires_sql": true,
          "requires_rag": false,
          "reasoning": "The question asks for a pickup count by date.",
          "entities": {"date": "yesterday"}
        }
        """,
    )

    decision = plan("How many pickups yesterday?")

    assert decision.capability == "sql"
    assert decision.intent == "pickup_count"
    assert decision.requires_sql is True
    assert decision.requires_rag is False
    assert decision.entities == {"date": "yesterday"}


@patch("core.prompt_manager.get_prompt_manager")
def test_plan_rag_question(mock_get_manager: MagicMock) -> None:
    _mock_planner_response(
        mock_get_manager,
        """
        {
          "capability": "rag",
          "intent": "explain_cbt",
          "confidence": 0.91,
          "requires_sql": false,
          "requires_rag": true,
          "reasoning": "The question asks for SOP/domain explanation.",
          "entities": {"term": "CBT"}
        }
        """,
    )

    decision = plan("What is CBT?")

    assert decision.capability == "rag"
    assert decision.intent == "explain_cbt"
    assert decision.requires_sql is False
    assert decision.requires_rag is True


@patch("core.prompt_manager.get_prompt_manager")
def test_plan_multi_question(mock_get_manager: MagicMock) -> None:
    _mock_planner_response(
        mock_get_manager,
        """
        {
          "capability": "multi",
          "intent": "delayed_pickups",
          "confidence": 0.96,
          "requires_sql": true,
          "requires_rag": true,
          "reasoning": "The question needs current delayed pickup metrics and SOP guidance.",
          "entities": {"metric": "delayed pickups", "threshold": 50, "date": "today"}
        }
        """,
    )

    decision = plan("How should operations respond if delayed pickups exceed 50 today?")

    assert decision.capability == "multi"
    assert decision.intent == "delayed_pickups"
    assert decision.requires_sql is True
    assert decision.requires_rag is True


@patch("core.prompt_manager.get_prompt_manager")
def test_plan_greeting_unknown(mock_get_manager: MagicMock) -> None:
    _mock_planner_response(
        mock_get_manager,
        """
        {
          "capability": "unknown",
          "intent": "greeting",
          "confidence": 0.89,
          "requires_sql": false,
          "requires_rag": false,
          "reasoning": "The input is a greeting and not an operational question.",
          "entities": {}
        }
        """,
    )

    decision = plan("hello")

    assert decision.capability == "unknown"
    assert decision.intent == "greeting"
    assert decision.requires_sql is False
    assert decision.requires_rag is False


@patch("core.prompt_manager.get_prompt_manager")
def test_plan_random_text_unknown(mock_get_manager: MagicMock) -> None:
    _mock_planner_response(
        mock_get_manager,
        """
        {
          "capability": "unknown",
          "intent": "unrelated",
          "confidence": 0.82,
          "requires_sql": false,
          "requires_rag": false,
          "reasoning": "The input is unrelated to logistics operations.",
          "entities": {}
        }
        """,
    )

    decision = plan("blue sandwich moon")

    assert decision.capability == "unknown"
    assert decision.intent == "unrelated"


def test_plan_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="Question must not be empty"):
        plan("   ")
