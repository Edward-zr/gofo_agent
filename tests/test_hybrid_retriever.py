"""Unit tests for hybrid RAG retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from core.models import QueryRequest, SourceChunk
from tools.rag.bm25_index import BM25Retriever
from tools.rag.corpus import save_bm25_corpus
from tools.rag.hybrid import (
    HybridRetriever,
    RankedCandidate,
    expand_queries,
    reciprocal_rank_fusion,
)
from tools.rag.retriever import _dense_retrieve, _distance_to_score, retrieve


def _sample_records() -> list[dict]:
    return [
        {
            "id": "pickup-sop.pdf::chunk-00001",
            "text": "Section 4.2 CBT cutoff time for ORD hub failed pickup handling.",
            "metadata": {
                "filename": "pickup-sop.pdf",
                "page": 12,
                "section_title": "Failed Pickup Handling",
                "sop_version": "v3.1",
                "hub": "ord",
                "department": "dispatch",
                "language": "en",
            },
        },
        {
            "id": "pickup-sop.pdf::chunk-00002",
            "text": "Drivers must report package volume and completion rate daily.",
            "metadata": {
                "filename": "pickup-sop.pdf",
                "page": 3,
                "section_title": "Daily Reporting",
                "sop_version": "v3.1",
                "hub": "ord",
                "department": "dispatch",
                "language": "en",
            },
        },
        {
            "id": "warehouse-sop.pdf::chunk-00003",
            "text": "Warehouse process for inbound scanning and dispatch handoff.",
            "metadata": {
                "filename": "warehouse-sop.pdf",
                "page": 1,
                "section_title": "Inbound Scanning",
                "sop_version": "v2.0",
                "hub": "chicago",
                "department": "warehouse",
                "language": "en",
            },
        },
    ]


@pytest.fixture
def bm25_corpus(tmp_path, monkeypatch):
    path = save_bm25_corpus(records=_sample_records(), path=tmp_path / "bm25_corpus.json")
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path)
    monkeypatch.setattr("tools.rag.corpus.config.CHROMA_PERSIST_DIR", tmp_path)
    monkeypatch.setattr("tools.rag.retriever.bm25_corpus_path", lambda: path)
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(config, "HYBRID_RERANK_ENABLED", False)
    monkeypatch.setattr(config, "HYBRID_QUERY_EXPANSION", True)
    return path


def test_bm25_exact_keyword_retrieval() -> None:
    retriever = BM25Retriever(_sample_records())
    hits = retriever.search("failed pickup handling", top_k=2)
    assert hits
    assert hits[0].chunk_id == "pickup-sop.pdf::chunk-00001"


def test_bm25_acronym_retrieval() -> None:
    retriever = BM25Retriever(_sample_records())
    hits = retriever.search("CBT", top_k=1)
    assert hits[0].chunk_id == "pickup-sop.pdf::chunk-00001"
    assert "CBT" in hits[0].text


def test_bm25_section_number_lookup() -> None:
    retriever = BM25Retriever(_sample_records())
    hits = retriever.search("Section 4.2", top_k=1)
    assert hits[0].chunk_id == "pickup-sop.pdf::chunk-00001"


def test_metadata_filtering_limits_results() -> None:
    retriever = BM25Retriever(_sample_records())
    hits = retriever.search(
        "dispatch process",
        top_k=5,
        filters={"department": "warehouse"},
    )
    assert len(hits) == 1
    assert hits[0].metadata["department"] == "warehouse"


def test_rrf_merge_prefers_chunks_present_in_both_lists() -> None:
    candidates = {
        "a": RankedCandidate(
            chunk_id="a",
            text="A",
            metadata={},
            dense_rank=1,
            bm25_rank=1,
        ),
        "b": RankedCandidate(
            chunk_id="b",
            text="B",
            metadata={},
            dense_rank=2,
        ),
        "c": RankedCandidate(
            chunk_id="c",
            text="C",
            metadata={},
            bm25_rank=1,
        ),
    }
    fused = reciprocal_rank_fusion(candidates, k=60)
    assert fused[0].chunk_id == "a"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_reranker_orders_by_cross_encoder_score(monkeypatch) -> None:
    monkeypatch.setattr(config, "HYBRID_RERANK_ENABLED", True)
    hybrid = HybridRetriever()
    candidates = [
        RankedCandidate(chunk_id="low", text="unrelated text", metadata={}, rrf_score=0.03),
        RankedCandidate(
            chunk_id="high",
            text="Section 4.2 CBT cutoff time for ORD hub failed pickup handling.",
            metadata={},
            rrf_score=0.02,
        ),
    ]

    class FakeReranker:
        def predict(self, pairs):
            del pairs
            return [0.1, 0.9]

    hybrid._reranker = FakeReranker()
    reranked = hybrid._rerank("ORD hub CBT cutoff", candidates)
    assert reranked[0].chunk_id == "high"
    assert reranked[0].final_score >= config.SIMILARITY_THRESHOLD


def test_expand_queries_adds_acronym_and_section_variants(monkeypatch) -> None:
    monkeypatch.setattr(config, "HYBRID_QUERY_EXPANSION", True)
    queries = expand_queries("CBT section 4.2?")
    assert "CBT section 4.2?" in queries
    assert any("CBT cutoff time procedure" in query for query in queries)
    assert any("Section 4.2" in query for query in queries)


@patch("tools.rag.retriever._dense_retrieve")
def test_hybrid_retrieve_merges_dense_and_bm25(
    mock_dense: MagicMock,
    bm25_corpus,
    monkeypatch,
) -> None:
    monkeypatch.setattr("tools.rag.retriever.get_bm25_retriever", lambda: BM25Retriever(_sample_records()))
    mock_dense.return_value = [
        SourceChunk(
            id="warehouse-sop.pdf::chunk-00003",
            text="Warehouse process for inbound scanning and dispatch handoff.",
            score=_distance_to_score(0.2),
            metadata={"filename": "warehouse-sop.pdf", "hub": "chicago"},
        )
    ]

    chunks = retrieve(QueryRequest(question="ORD hub CBT cutoff", top_k=2))

    assert len(chunks) == 2
    assert {chunk.id for chunk in chunks} == {
        "pickup-sop.pdf::chunk-00001",
        "warehouse-sop.pdf::chunk-00003",
    }
    assert all(chunk.score >= config.SIMILARITY_THRESHOLD for chunk in chunks)


@patch("tools.rag.retriever._dense_retrieve")
def test_hybrid_retrieve_applies_metadata_filters(
    mock_dense: MagicMock,
    bm25_corpus,
    monkeypatch,
) -> None:
    monkeypatch.setattr("tools.rag.retriever.get_bm25_retriever", lambda: BM25Retriever(_sample_records()))
    mock_dense.return_value = []

    chunks = retrieve(
        QueryRequest(
            question="warehouse process for chicago hub",
            top_k=3,
            filters={"hub": "chicago"},
        )
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["hub"] == "chicago"
    dense_request = mock_dense.call_args.args[0]
    assert dense_request.filters == {"hub": "chicago"}


@patch("tools.rag.retriever.bm25_corpus_path")
def test_dense_only_fallback_when_bm25_missing(mock_path: MagicMock, monkeypatch) -> None:
    mock_path.return_value.exists.return_value = False
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL_ENABLED", True)

    with patch("tools.rag.retriever._dense_retrieve") as mock_dense:
        mock_dense.return_value = [
            SourceChunk(id="x", text="dense only", score=0.8, metadata={})
        ]
        chunks = retrieve(QueryRequest(question="returns"))

    assert len(chunks) == 1
    assert chunks[0].text == "dense only"


@patch("tools.rag.retriever.get_collection")
@patch("tools.rag.retriever._get_embeddings")
def test_dense_retrieve_regression(
    mock_get_embeddings: MagicMock,
    mock_get_collection: MagicMock,
) -> None:
    mock_get_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Return policy text."]],
        "ids": [["returns.pdf::chunk-00001"]],
        "metadatas": [[{"filename": "returns.pdf", "page": 1}]],
        "distances": [[0.5]],
    }
    mock_get_collection.return_value = mock_collection

    chunks = _dense_retrieve(QueryRequest(question="How do returns work?", top_k=1))

    assert len(chunks) == 1
    assert chunks[0].id == "returns.pdf::chunk-00001"
    assert chunks[0].score == pytest.approx(_distance_to_score(0.5))
