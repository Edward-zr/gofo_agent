"""Unit tests for tools.rag.service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import QueryRequest, SourceChunk
from tools.rag.service import answer, retrieve_only


def test_answer_empty_question_raises() -> None:
    request = QueryRequest.model_construct(question="   ")

    with pytest.raises(ValueError, match="Question must not be empty"):
        answer(request)


@patch("tools.rag.service.rewrite", side_effect=lambda question: question)
@patch("tools.rag.service.retrieve")
def test_answer_empty_retrieval(
    mock_retrieve: MagicMock,
    mock_rewrite: MagicMock,
) -> None:
    mock_retrieve.return_value = []

    response = answer(QueryRequest(question="How do returns work?"))

    assert response.question == "How do returns work?"
    assert response.sources == []
    assert response.capability in {"rag", "clarification"}
    assert response.rewritten_question == "How do returns work?"
    assert response.confidence_level == "LOW"
    assert response.fallback_strategy in {"CLARIFY", "REFUSE", "DOCUMENT_REQUEST"}
    assert response.answer  # clarification / refuse message


@patch("tools.rag.service.rewrite", side_effect=lambda question: question)
@patch("tools.rag.service.generate")
@patch("tools.rag.service.retrieve")
def test_answer_low_confidence_skips_generation(
    mock_retrieve: MagicMock,
    mock_generate: MagicMock,
    mock_rewrite: MagicMock,
    low_confidence_chunk: SourceChunk,
) -> None:
    mock_retrieve.return_value = [low_confidence_chunk]

    response = answer(QueryRequest(question="How do returns work?"))

    assert response.confidence_level == "LOW"
    assert response.fallback_strategy == "CLARIFY"
    assert response.requires_clarification is True
    assert response.sources == [low_confidence_chunk]
    mock_generate.assert_not_called()


@patch("tools.rag.service.rewrite", side_effect=lambda question: question)
@patch("tools.rag.service.generate")
@patch("tools.rag.service.retrieve")
def test_answer_success(
    mock_retrieve: MagicMock,
    mock_generate: MagicMock,
    mock_rewrite: MagicMock,
) -> None:
    # Diverse high-scoring chunks so adaptive confidence is HIGH.
    chunks = [
        SourceChunk(
            id=f"c{i}",
            text="Process customer returns within 30 days of delivery.",
            score=0.90 - i * 0.01,
            metadata={"filename": f"doc{i}.pdf", "page": i},
        )
        for i in range(5)
    ]
    mock_retrieve.return_value = chunks
    mock_generate.return_value = "Process customer returns within 30 days of delivery."

    response = answer(QueryRequest(question="How do returns work?"))

    assert response.question == "How do returns work?"
    assert response.answer == "Process customer returns within 30 days of delivery."
    assert response.sources == chunks
    assert response.capability == "rag"
    assert response.confidence_level == "HIGH"
    mock_generate.assert_called_once()
    assert mock_generate.call_args.args[0] == "How do returns work?"
    assert mock_generate.call_args.args[1] == chunks


@patch("tools.rag.service.generate")
@patch("tools.rag.service.retrieve")
@patch("tools.rag.service.rewrite")
def test_answer_uses_rewritten_question_for_retrieval(
    mock_rewrite: MagicMock,
    mock_retrieve: MagicMock,
    mock_generate: MagicMock,
) -> None:
    chunks = [
        SourceChunk(
            id=f"c{i}",
            text="Return process.",
            score=0.91 - i * 0.01,
            metadata={"filename": f"doc{i}.pdf", "page": 0},
        )
        for i in range(5)
    ]
    mock_rewrite.return_value = "What is the return process according to the GOFO SOP?"
    mock_retrieve.return_value = chunks
    mock_generate.return_value = "Generated answer."

    response = answer(QueryRequest(question="returns?"))

    mock_rewrite.assert_called_once_with("returns?")
    retrieval_request = mock_retrieve.call_args.args[0]
    assert retrieval_request.question == "What is the return process according to the GOFO SOP?"
    mock_generate.assert_called_once()
    assert mock_generate.call_args.args[0] == "returns?"
    assert response.rewritten_question == "What is the return process according to the GOFO SOP?"


@patch("tools.rag.service.retrieve")
def test_retrieve_only_returns_chunks_without_answer(
    mock_retrieve: MagicMock,
    sample_chunk: SourceChunk,
) -> None:
    mock_retrieve.return_value = [sample_chunk]

    response = retrieve_only(QueryRequest(question="How do returns work?"))

    assert response.answer is None
    assert response.sources == [sample_chunk]
    assert response.confidence_score is not None
    assert response.confidence_level in {"HIGH", "MEDIUM", "LOW"}
