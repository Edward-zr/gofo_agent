"""Shared pytest fixtures for GOFO agent unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.models import SourceChunk
from tools.llm.client import get_llm
from tools.rag import retriever


@pytest.fixture(autouse=True)
def clear_module_caches() -> None:
    """Clear cached OpenAI/Chroma clients between tests."""
    retriever._get_embeddings.cache_clear()
    retriever._get_chroma_client.cache_clear()
    get_llm.cache_clear()
    yield
    retriever._get_embeddings.cache_clear()
    retriever._get_chroma_client.cache_clear()
    get_llm.cache_clear()


@pytest.fixture
def sample_chunk() -> SourceChunk:
    """A high-confidence retrieved chunk."""
    return SourceChunk(
        id="returns.pdf::chunk-00001",
        text="Process customer returns within 30 days of delivery.",
        score=0.75,
        metadata={"filename": "returns.pdf", "page": 2},
    )


@pytest.fixture
def low_confidence_chunk() -> SourceChunk:
    """A chunk below the default similarity threshold."""
    return SourceChunk(
        id="other.pdf::chunk-00002",
        text="Unrelated operational note.",
        score=0.25,
        metadata={"filename": "other.pdf", "page": 0},
    )


@pytest.fixture
def mock_llm_response() -> MagicMock:
    """Fake LangChain chat response."""
    response = MagicMock()
    response.content = "Process customer returns within 30 days of delivery."
    return response
