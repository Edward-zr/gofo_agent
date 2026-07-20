"""Shared pytest fixtures for GOFO agent unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.models import SourceChunk
from tools.llm.client import get_llm
from tools.orchestration.models import SemanticAnalysis
from tools.rag import retriever
from tools.rag.bm25_index import clear_bm25_cache


@pytest.fixture(autouse=True)
def clear_module_caches() -> None:
    """Clear cached OpenAI/Chroma clients between tests."""
    retriever._get_embeddings.cache_clear()
    retriever._get_chroma_client.cache_clear()
    clear_bm25_cache()
    get_llm.cache_clear()
    yield
    retriever._get_embeddings.cache_clear()
    retriever._get_chroma_client.cache_clear()
    clear_bm25_cache()
    get_llm.cache_clear()


@pytest.fixture(autouse=True)
def disable_planner_execution_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy RouteDispatcher as the agent executor in most tests.

    Unit tests exercise Planner/PlanExecutor directly. Opt into agent-level
    planner execution with monkeypatch.setattr(config, "PLANNER_ENABLED", True).
    """
    monkeypatch.setattr("config.PLANNER_ENABLED", False)


@pytest.fixture(autouse=True)
def disable_quality_assurance_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable QA retries in agent integration tests by default.

    Unit tests cover QualityAssurancePipeline directly.
    """
    monkeypatch.setattr("config.QUALITY_ASSURANCE_ENABLED", False)


@pytest.fixture(autouse=True)
def disable_reflection_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable Reflection retries in agent integration tests by default.

    Unit tests cover ReflectionAgent directly.
    """
    monkeypatch.setattr("config.REFLECTION_ENABLED", False)


@pytest.fixture(autouse=True)
def mock_semantic_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic semantic analysis during agent and router tests."""

    def _default_analyze(
        question: str,
        *,
        resolved_question: str | None = None,
        conversation_state: dict | None = None,
        repair_detected: bool = False,
        has_attachments: bool = False,
    ) -> SemanticAnalysis:
        del conversation_state, repair_detected, has_attachments
        return SemanticAnalysis(
            domain="data_analytics",
            sub_intent="analysis",
            capability="sql",
            response_mode="analytical",
            confidence=0.5,
            resolved_question=resolved_question or question,
            reasoning="Test default semantic analysis.",
            requires_sql=True,
            requires_rag=False,
            planner_intent="operational_analysis",
        )

    monkeypatch.setattr("core.agent.analyze_request", _default_analyze)
    monkeypatch.setattr(
        "tools.router.generate_next_steps",
        lambda **kwargs: [],
    )


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
