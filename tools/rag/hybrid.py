"""Hybrid dense + sparse retrieval with RRF fusion and cross-encoder reranking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import config
from core.logger import get_logger
from core.models import QueryRequest, SourceChunk
from tools.rag.bm25_index import BM25Hit, get_bm25_retriever
from tools.rag.corpus import METADATA_FIELDS

logger = get_logger("rag.hybrid")

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional at import time
    CrossEncoder = None  # type: ignore[misc, assignment]


@dataclass
class RankedCandidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    dense_rank: int | None = None
    bm25_score: float = 0.0
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    sources: set[str] = field(default_factory=set)


@dataclass
class RetrievalDebugTrace:
    original_query: str
    expanded_queries: list[str]
    filters: dict[str, Any] | None
    dense_results: list[dict[str, Any]] = field(default_factory=list)
    bm25_results: list[dict[str, Any]] = field(default_factory=list)
    merged_ranking: list[dict[str, Any]] = field(default_factory=list)
    reranked_ranking: list[dict[str, Any]] = field(default_factory=list)
    final_context: list[dict[str, Any]] = field(default_factory=list)


class HybridRetriever:
    """Run dense + BM25 retrieval, fuse with RRF, and rerank candidates."""

    def __init__(self) -> None:
        self._reranker: Any | None = None

    def retrieve(
        self,
        request: QueryRequest,
        *,
        dense_search,
    ) -> list[SourceChunk]:
        question = request.question.strip()
        filters = _merge_filters(request.filters, question)
        expanded_queries = expand_queries(question) if config.HYBRID_QUERY_EXPANSION else [question]
        debug = RetrievalDebugTrace(
            original_query=question,
            expanded_queries=expanded_queries,
            filters=filters,
        )

        candidates: dict[str, RankedCandidate] = {}
        dense_request = request.model_copy(update={"top_k": config.HYBRID_DENSE_TOP_K, "filters": filters})
        bm25_retriever = get_bm25_retriever()

        for query in expanded_queries:
            dense_request = dense_request.model_copy(update={"question": query})
            dense_chunks = dense_search(dense_request)
            for rank, chunk in enumerate(dense_chunks, start=1):
                _update_candidate(
                    candidates,
                    chunk_id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    source="dense",
                    rank=rank,
                    score=chunk.score,
                )
                if query == question:
                    debug.dense_results.append(_serialize_result(chunk.id, chunk.score, rank, chunk.metadata))

            if bm25_retriever.available():
                for hit in bm25_retriever.search(query, top_k=config.HYBRID_BM25_TOP_K, filters=filters):
                    _update_candidate(
                        candidates,
                        chunk_id=hit.chunk_id,
                        text=hit.text,
                        metadata=hit.metadata,
                        source="bm25",
                        rank=hit.rank,
                        score=hit.score,
                    )
                    if query == question:
                        debug.bm25_results.append(
                            _serialize_result(hit.chunk_id, hit.score, hit.rank, hit.metadata)
                        )

        if not candidates:
            _emit_debug(debug)
            return []

        fused = reciprocal_rank_fusion(candidates, k=config.HYBRID_RRF_K)
        debug.merged_ranking = [
            {
                "id": candidate.chunk_id,
                "rrf_score": round(candidate.rrf_score, 6),
                "sources": sorted(candidate.sources),
                "metadata": _public_metadata(candidate.metadata),
            }
            for candidate in fused
        ]

        rerank_pool = fused[: config.HYBRID_RERANK_TOP_N]
        reranked = self._rerank(question, rerank_pool)
        debug.reranked_ranking = [
            {
                "id": candidate.chunk_id,
                "rerank_score": candidate.rerank_score,
                "final_score": round(candidate.final_score, 6),
                "metadata": _public_metadata(candidate.metadata),
            }
            for candidate in reranked
        ]

        final_chunks = [
            SourceChunk(
                id=candidate.chunk_id,
                text=candidate.text,
                score=candidate.final_score,
                metadata=candidate.metadata,
            )
            for candidate in reranked[: request.top_k]
        ]
        debug.final_context = [
            {
                "id": chunk.id,
                "score": round(chunk.score, 6),
                "metadata": _public_metadata(chunk.metadata),
                "text_preview": chunk.text[:180],
            }
            for chunk in final_chunks
        ]
        _emit_debug(debug)
        return final_chunks

    def _rerank(self, query: str, candidates: list[RankedCandidate]) -> list[RankedCandidate]:
        if not candidates:
            return []

        if not config.HYBRID_RERANK_ENABLED:
            for index, candidate in enumerate(candidates):
                candidate.final_score = _fallback_score(candidate, index)
            return sorted(candidates, key=lambda item: item.final_score, reverse=True)

        reranker = self._get_reranker()
        if reranker is None:
            for index, candidate in enumerate(candidates):
                candidate.final_score = _fallback_score(candidate, index)
            return sorted(candidates, key=lambda item: item.final_score, reverse=True)

        pairs = [[query, candidate.text] for candidate in candidates]
        raw_scores = reranker.predict(pairs)
        normalized = _normalize_reranker_scores([float(score) for score in raw_scores])

        for candidate, rerank_score, final_score in zip(candidates, raw_scores, normalized, strict=True):
            candidate.rerank_score = float(rerank_score)
            candidate.final_score = final_score

        return sorted(candidates, key=lambda item: item.final_score, reverse=True)

    def _get_reranker(self) -> Any | None:
        if CrossEncoder is None:
            return None
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(config.HYBRID_RERANKER_MODEL)
            except Exception as exc:  # pragma: no cover - model load failures
                logger.warning("Cross-encoder reranker unavailable: %s", exc)
                return None
        return self._reranker


def expand_queries(question: str) -> list[str]:
    """Generate alternative search formulations for short or ambiguous queries."""
    normalized = question.strip()
    if not normalized:
        return []

    queries = [normalized]
    if len(normalized.split()) >= config.HYBRID_EXPANSION_MIN_WORDS:
        return _dedupe_queries(queries)

    heuristic_alternatives = _heuristic_expansions(normalized)
    queries.extend(heuristic_alternatives)

    if config.HYBRID_EXPANSION_USE_LLM:
        try:
            queries.extend(_llm_expansions(normalized))
        except Exception as exc:
            logger.warning("Query expansion fallback after LLM failure: %s", exc)

    return _dedupe_queries(queries)[: config.HYBRID_EXPANSION_MAX_QUERIES]


def reciprocal_rank_fusion(
    candidates: dict[str, RankedCandidate],
    *,
    k: int,
) -> list[RankedCandidate]:
    """Merge dense and BM25 rankings with Reciprocal Rank Fusion."""
    for candidate in candidates.values():
        score = 0.0
        if candidate.dense_rank is not None:
            score += 1.0 / (k + candidate.dense_rank)
        if candidate.bm25_rank is not None:
            score += 1.0 / (k + candidate.bm25_rank)
        candidate.rrf_score = score
    return sorted(candidates.values(), key=lambda item: item.rrf_score, reverse=True)


def _merge_filters(existing: dict[str, Any] | None, question: str) -> dict[str, Any] | None:
    merged = dict(existing or {})
    inferred = _infer_filters_from_query(question)
    for key, value in inferred.items():
        merged.setdefault(key, value)
    return merged or None


def _infer_filters_from_query(question: str) -> dict[str, str]:
    normalized = question.lower()
    filters: dict[str, str] = {}

    hub_match = re.search(r"\b([a-z][a-z0-9_-]+)\s+hub\b", normalized)
    if hub_match:
        filters["hub"] = hub_match.group(1)

    department_match = re.search(
        r"\b(dispatch|warehouse|operations|logistics|pickup|delivery)\s+department\b",
        normalized,
    )
    if department_match:
        filters["department"] = department_match.group(1)

    document_match = re.search(r"\b([a-z0-9_-]+\.pdf)\b", normalized)
    if document_match:
        filters["filename"] = document_match.group(1)

    language_match = re.search(r"\b(english|spanish|french|chinese)\b", normalized)
    if language_match:
        language = language_match.group(1)
        filters["language"] = "en" if language == "english" else language

    return filters


def _heuristic_expansions(question: str) -> list[str]:
    normalized = question.lower().strip()
    alternatives: list[str] = []

    section_match = re.search(r"\bsection\s+([0-9]+(?:\.[0-9]+)*)", normalized)
    if section_match:
        alternatives.append(f"Section {section_match.group(1)}")
        alternatives.append(f"SOP section {section_match.group(1)} procedure")

    acronym_map = {
        "cbt": "CBT cutoff time procedure",
        "ord": "ORD hub operations",
        "sop": "standard operating procedure",
        "pod": "proof of delivery",
        "kpi": "key performance indicator",
    }
    for acronym, expansion in acronym_map.items():
        if re.search(rf"\b{acronym}\b", normalized):
            alternatives.append(expansion)

    if "pickup" in normalized and "failed" in normalized:
        alternatives.append("failed pickup exception handling procedure")

    return alternatives


def _llm_expansions(question: str) -> list[str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from tools.llm.client import get_llm

    messages = [
        SystemMessage(
            content=(
                "Generate 2 alternative GOFO logistics search queries for retrieval. "
                "Return ONLY JSON: {\"queries\": [\"...\", \"...\"]}"
            )
        ),
        HumanMessage(content=question),
    ]
    response = get_llm().invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    payload = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
    queries = payload.get("queries") or []
    return [str(item).strip() for item in queries if str(item).strip()]


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        key = query.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(query.strip())
    return deduped


def _update_candidate(
    candidates: dict[str, RankedCandidate],
    *,
    chunk_id: str,
    text: str,
    metadata: dict[str, Any],
    source: str,
    rank: int,
    score: float,
) -> None:
    candidate = candidates.get(chunk_id)
    if candidate is None:
        candidate = RankedCandidate(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata or {},
        )
        candidates[chunk_id] = candidate

    candidate.sources.add(source)
    if source == "dense":
        if candidate.dense_rank is None or rank < candidate.dense_rank:
            candidate.dense_rank = rank
            candidate.dense_score = score
    elif source == "bm25":
        if candidate.bm25_rank is None or rank < candidate.bm25_rank:
            candidate.bm25_rank = rank
            candidate.bm25_score = score


def _normalize_reranker_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [0.75 for _ in scores]
    return [0.40 + ((score - minimum) / (maximum - minimum)) * 0.55 for score in scores]


def _fallback_score(candidate: RankedCandidate, index: int) -> float:
    base = candidate.rrf_score * 10.0
    return min(0.95, max(0.40, 0.85 - (index * 0.05) + base))


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: metadata.get(key) for key in METADATA_FIELDS if metadata.get(key) is not None}


def _serialize_result(
    chunk_id: str,
    score: float,
    rank: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "score": round(float(score), 6),
        "rank": rank,
        "metadata": _public_metadata(metadata),
    }


def _emit_debug(debug: RetrievalDebugTrace) -> None:
    if not config.HYBRID_RETRIEVAL_DEBUG:
        return
    logger.info("========== RAG Hybrid Retrieval Debug ==========")
    logger.info("Original query: %s", debug.original_query)
    logger.info("Expanded queries: %s", debug.expanded_queries)
    logger.info("Metadata filters: %s", debug.filters)
    logger.info("Dense results: %s", json.dumps(debug.dense_results, ensure_ascii=False))
    logger.info("BM25 results: %s", json.dumps(debug.bm25_results, ensure_ascii=False))
    logger.info("Merged ranking: %s", json.dumps(debug.merged_ranking, ensure_ascii=False))
    logger.info("Reranked ranking: %s", json.dumps(debug.reranked_ranking, ensure_ascii=False))
    logger.info("Final context: %s", json.dumps(debug.final_context, ensure_ascii=False))
    logger.info("================================================")
