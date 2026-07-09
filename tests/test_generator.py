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


@patch("tools.rag.generator.get_llm")
def test_generate_success(
    mock_get_llm: MagicMock,
    sample_chunk: SourceChunk,
    mock_llm_response: MagicMock,
) -> None:
    mock_get_llm.return_value.invoke.return_value = mock_llm_response

    answer = generate("How do returns work?", [sample_chunk])

    assert answer == "Process customer returns within 30 days of delivery."
    mock_get_llm.return_value.invoke.assert_called_once()
    messages = mock_get_llm.return_value.invoke.call_args.args[0]
    assert len(messages) == 2
    assert "GOFO Operations Intelligence Assistant" in messages[0].content
    assert "How do returns work?" in messages[1].content
    assert sample_chunk.text in messages[1].content
