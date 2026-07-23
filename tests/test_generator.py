"""Unit tests for tools.rag.generator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import SourceChunk
from tools.rag.generator import UNKNOWN_ANSWER, generate


def test_generate_empty_question_raises(sample_chunk: SourceChunk) -> None:
    with pytest.raises(ValueError, match="Question must not be empty"):
        generate("   ", [sample_chunk])


def test_generate_empty_chunks_returns_unknown_answer() -> None:
    answer = generate("How do returns work?", [])
    assert answer == UNKNOWN_ANSWER


@patch("tools.rag.generator.get_prompt_manager")
def test_generate_success(
    mock_get_manager: MagicMock,
    sample_chunk: SourceChunk,
) -> None:
    mock_get_manager.return_value.invoke.return_value = (
        "Process customer returns within 30 days of delivery."
    )

    answer = generate(
        "How do returns work?",
        [sample_chunk],
        confidence_context={
            "confidence_level": "HIGH",
            "confidence_score": 0.9,
            "confidence_reason": "strong",
            "fallback_strategy": "GENERATE_CONFIDENT",
        },
    )

    assert answer == "Process customer returns within 30 days of delivery."
    mock_get_manager.return_value.invoke.assert_called_once()
    args, kwargs = mock_get_manager.return_value.invoke.call_args
    assert args[0] == "rag.generator_prompt"
    assert "How do returns work?" in args[1]["question"]
    assert sample_chunk.text in args[1]["sop_context"]
