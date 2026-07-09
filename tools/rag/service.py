"""Orchestration layer for GOFO RAG: retrieval and generation only."""

from __future__ import annotations

import config
from core.models import QueryRequest, QueryResponse, SourceChunk
from tools.rag.generator import UNKNOWN_ANSWER, generate
from tools.rag.retriever import retrieve
from tools.rag.rewriter import rewrite


def _best_similarity_score(chunks: list[SourceChunk]) -> float:
    """Return the highest relevance score among retrieved chunks."""
    return max(chunk.score for chunk in chunks)


def answer(request: QueryRequest) -> QueryResponse:
    """
    Run retrieval and generation for a question.

    Does not access ChromaDB or construct prompts directly.
    """
    question = request.question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    rewritten_question = rewrite(request.question)
    retrieval_request = request.model_copy(update={"question": rewritten_question})
    chunks = retrieve(retrieval_request)

    if not chunks:
        return QueryResponse(
            question=request.question,
            answer=UNKNOWN_ANSWER,
            sources=[],
            capability="rag",
            rewritten_question=rewritten_question,
        )

    best_score = _best_similarity_score(chunks)
    if best_score < config.SIMILARITY_THRESHOLD:
        # Skip GPT when retrieval confidence is too low. Weak matches often cause
        # hallucinated SOP answers; returning UNKNOWN_ANSWER is safer and cheaper.
        return QueryResponse(
            question=request.question,
            answer=UNKNOWN_ANSWER,
            sources=chunks,
            capability="rag",
            rewritten_question=rewritten_question,
        )

    generated_answer = generate(request.question, chunks)

    return QueryResponse(
        question=request.question,
        answer=generated_answer,
        sources=chunks,
        capability="rag",
        rewritten_question=rewritten_question,
    )


def retrieve_only(request: QueryRequest) -> QueryResponse:
    """
    Run retrieval without generation.

    Useful for debugging chunk quality and citation inspection.
    """
    question = request.question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    chunks = retrieve(request)

    return QueryResponse(
        question=request.question,
        answer=None,
        sources=chunks,
        capability="rag",
    )
