"""Orchestration layer for GOFO RAG: retrieval, confidence, and generation."""

from __future__ import annotations

from typing import Any

from core.models import QueryRequest, QueryResponse, SourceChunk
from tools.rag.confidence import (
    RetrievalDecision,
    RetrievalPolicy,
    agent_state_from_decision,
    confidence_context_for_generator,
    default_retrieval_policy,
    evaluate_and_decide,
    evaluate_retrieval,
    policy_from_mapping,
)
from tools.rag.generator import UNKNOWN_ANSWER, generate
from tools.rag.retriever import retrieve
from tools.rag.rewriter import rewrite


def _attach_confidence_fields(
    response: QueryResponse,
    decision: RetrievalDecision,
) -> QueryResponse:
    payload = agent_state_from_decision(decision)
    existing_state = dict(response.agent_state or {})
    existing_state.update(payload)
    return response.model_copy(
        update={
            "agent_state": existing_state,
            "retrieval_confidence": decision.confidence.model_dump(),
            "confidence_score": decision.confidence.confidence_score,
            "confidence_level": decision.confidence.confidence_level,
            "confidence_breakdown": decision.confidence.confidence_breakdown.model_dump(),
            "similarity_scores": list(decision.confidence.similarity_scores),
            "retrieved_chunk_count": decision.confidence.retrieved_chunk_count,
            "retrieved_sources": list(decision.confidence.retrieved_sources),
            "confidence_reason": decision.confidence.reason,
            "fallback_strategy": decision.fallback_strategy,
            "retrieval_policy": decision.policy.model_dump(),
        }
    )


def answer(
    request: QueryRequest,
    *,
    retrieval_policy: RetrievalPolicy | dict[str, Any] | None = None,
) -> QueryResponse:
    """
    Run retrieval → Adaptive Confidence → Decision → generation.

    Never answers confidently when retrieval evidence is weak.
    """
    question = request.question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    policy = policy_from_mapping(retrieval_policy) if retrieval_policy is not None else default_retrieval_policy()
    rewritten_question = rewrite(request.question)
    retrieval_request = request.model_copy(update={"question": rewritten_question})
    chunks = retrieve(retrieval_request)

    decision = evaluate_and_decide(
        chunks,
        policy=policy,
        retrieval_metadata={
            "rewritten_question": rewritten_question,
            "top_k": request.top_k,
            "mode": "answer",
        },
        expected_chunk_count=request.top_k or None,
    )

    if not chunks:
        response = QueryResponse(
            question=request.question,
            answer=decision.message or UNKNOWN_ANSWER,
            sources=[],
            capability="rag",
            rewritten_question=rewritten_question,
            requires_clarification=decision.requires_clarification,
            clarification_question=decision.message if decision.requires_clarification else None,
        )
        return _attach_confidence_fields(response, decision)

    if not decision.should_generate:
        response = QueryResponse(
            question=request.question,
            answer=decision.message or UNKNOWN_ANSWER,
            sources=chunks,
            capability="clarification" if decision.requires_clarification else "rag",
            rewritten_question=rewritten_question,
            requires_clarification=decision.requires_clarification,
            clarification_question=decision.message if decision.requires_clarification else None,
            plan_reason=decision.reason,
        )
        return _attach_confidence_fields(response, decision)

    generated_answer = generate(
        request.question,
        chunks if not decision.use_general_knowledge else [],
        confidence_context=confidence_context_for_generator(decision),
    )

    response = QueryResponse(
        question=request.question,
        answer=generated_answer,
        sources=chunks,
        capability="rag",
        rewritten_question=rewritten_question,
        plan_reason=decision.reason,
    )
    return _attach_confidence_fields(response, decision)


def retrieve_only(
    request: QueryRequest,
    *,
    retrieval_policy: RetrievalPolicy | dict[str, Any] | None = None,
) -> QueryResponse:
    """
    Run retrieval + confidence evaluation without generation.

    Useful for planner multi-step SOP flows and citation inspection.
    """
    question = request.question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    policy = policy_from_mapping(retrieval_policy) if retrieval_policy is not None else default_retrieval_policy()
    chunks = retrieve(request)
    decision = evaluate_and_decide(
        chunks,
        policy=policy,
        retrieval_metadata={"mode": "retrieve_only", "top_k": request.top_k},
        expected_chunk_count=request.top_k or None,
    )

    response = QueryResponse(
        question=request.question,
        answer=None,
        sources=chunks,
        capability="rag",
        plan_reason=decision.reason,
    )
    return _attach_confidence_fields(response, decision)


def confidence_for_chunks(
    chunks: list[SourceChunk],
    *,
    retrieval_policy: RetrievalPolicy | dict[str, Any] | None = None,
    retrieval_metadata: dict[str, Any] | None = None,
) -> RetrievalDecision:
    """Evaluate and decide for an existing chunk list (LLM / orchestrator reuse)."""
    return evaluate_and_decide(
        chunks,
        policy=retrieval_policy,
        retrieval_metadata=retrieval_metadata,
    )
