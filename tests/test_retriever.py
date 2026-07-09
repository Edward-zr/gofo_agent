"""Unit tests for tools.rag.retriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import QueryRequest
from tools.rag.retriever import _distance_to_score, retrieve


def test_distance_to_score() -> None:
    assert _distance_to_score(0.0) == 1.0
    assert _distance_to_score(1.0) == 0.5


def test_retrieve_empty_question_raises() -> None:
    request = QueryRequest.model_construct(question="   ")

    with pytest.raises(ValueError, match="Question must not be empty"):
        retrieve(request)


@patch("tools.rag.retriever.get_collection")
@patch("tools.rag.retriever._get_embeddings")
def test_retrieve_empty_results(
    mock_get_embeddings: MagicMock,
    mock_get_collection: MagicMock,
) -> None:
    mock_get_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [[]],
        "ids": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_get_collection.return_value = mock_collection

    chunks = retrieve(QueryRequest(question="How do returns work?"))

    assert chunks == []
    mock_get_embeddings.return_value.embed_query.assert_called_once_with("How do returns work?")
    mock_collection.query.assert_called_once()


@patch("tools.rag.retriever.get_collection")
@patch("tools.rag.retriever._get_embeddings")
def test_retrieve_valid_results(
    mock_get_embeddings: MagicMock,
    mock_get_collection: MagicMock,
) -> None:
    mock_get_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Return policy text.", "Shipping policy text."]],
        "ids": [["returns.pdf::chunk-00001", "shipping.pdf::chunk-00002"]],
        "metadatas": [
            [
                {"filename": "returns.pdf", "page": 1},
                {"filename": "shipping.pdf", "page": 0},
            ]
        ],
        "distances": [[0.5, 1.0]],
    }
    mock_get_collection.return_value = mock_collection

    chunks = retrieve(QueryRequest(question="How do returns work?", top_k=2))

    assert len(chunks) == 2
    assert chunks[0].id == "returns.pdf::chunk-00001"
    assert chunks[0].text == "Return policy text."
    assert chunks[0].score == pytest.approx(_distance_to_score(0.5))
    assert chunks[0].metadata["filename"] == "returns.pdf"
    assert chunks[1].score == pytest.approx(_distance_to_score(1.0))
