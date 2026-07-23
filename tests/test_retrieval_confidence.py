"""Tests for Adaptive Retrieval Confidence Engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.models import QueryRequest, SourceChunk
from tools.rag.confidence import (
    RetrievalPolicy,
    decide_retrieval_action,
    evaluate_and_decide,
    evaluate_retrieval,
)
from tools.rag.service import answer


def _chunk(
    chunk_id: str,
    score: float,
    *,
    filename: str = "sop.pdf",
    page: int = 0,
    text: str = "Return packages within 30 days.",
) -> SourceChunk:
    return SourceChunk(
        id=chunk_id,
        text=text,
        score=score,
        metadata={"filename": filename, "page": page},
    )


def test_high_confidence_diverse_sources() -> None:
    chunks = [
        _chunk("a", 0.92, filename="returns.pdf"),
        _chunk("b", 0.88, filename="pickup.pdf"),
        _chunk("c", 0.85, filename="hub.pdf"),
        _chunk("d", 0.84, filename="driver.pdf"),
        _chunk("e", 0.83, filename="returns.pdf"),
        _chunk("f", 0.81, filename="pickup.pdf"),
    ]
    result = evaluate_retrieval(chunks)
    assert result.confidence_level == "HIGH"
    assert result.confidence_score >= 0.80
    assert result.confidence_breakdown.source_diversity >= 3
    assert result.confidence_breakdown.metadata_quality == "GOOD"


def test_medium_confidence_partial_evidence() -> None:
    chunks = [
        _chunk("a", 0.72, filename="returns.pdf"),
        _chunk("b", 0.68, filename="returns.pdf"),
    ]
    result = evaluate_retrieval(chunks)
    assert result.confidence_level in {"MEDIUM", "HIGH"}
    assert 0.60 <= result.confidence_score


def test_low_confidence_weak_single_chunk() -> None:
    chunks = [_chunk("weak", 0.25, filename="noise.pdf")]
    result = evaluate_retrieval(chunks)
    assert result.confidence_level == "LOW"
    assert result.confidence_score < 0.60


def test_empty_retrieval_is_low() -> None:
    result = evaluate_retrieval([])
    assert result.confidence_level == "LOW"
    assert result.retrieved_chunk_count == 0


def test_decision_high_generates_confident() -> None:
    chunks = [
        _chunk("a", 0.93, filename="a.pdf"),
        _chunk("b", 0.90, filename="b.pdf"),
        _chunk("c", 0.88, filename="c.pdf"),
        _chunk("d", 0.86, filename="d.pdf"),
        _chunk("e", 0.85, filename="e.pdf"),
    ]
    decision = evaluate_and_decide(chunks)
    assert decision.fallback_strategy == "GENERATE_CONFIDENT"
    assert decision.should_generate is True


def test_decision_low_prefers_clarification() -> None:
    decision = evaluate_and_decide([_chunk("weak", 0.2)])
    assert decision.fallback_strategy == "CLARIFY"
    assert decision.requires_clarification is True
    assert decision.should_generate is False


def test_decision_respects_planner_document_request_policy() -> None:
    policy = RetrievalPolicy(
        allow_general_knowledge=False,
        minimum_confidence="MEDIUM",
        allow_clarification=False,
        allow_document_request=True,
    )
    decision = evaluate_and_decide([_chunk("weak", 0.2)], policy=policy)
    assert decision.fallback_strategy == "DOCUMENT_REQUEST"
    assert "upload" in (decision.message or "").lower()


def test_decision_general_knowledge_only_when_permitted() -> None:
    policy = RetrievalPolicy(
        allow_general_knowledge=True,
        minimum_confidence="MEDIUM",
        allow_clarification=False,
        allow_document_request=False,
    )
    decision = evaluate_and_decide([_chunk("weak", 0.2)], policy=policy)
    assert decision.fallback_strategy == "GENERAL_KNOWLEDGE"
    assert decision.use_general_knowledge is True
    assert decision.should_generate is True


def test_minimum_confidence_high_rejects_medium() -> None:
    chunks = [
        _chunk("a", 0.75, filename="returns.pdf"),
        _chunk("b", 0.70, filename="returns.pdf"),
    ]
    conf = evaluate_retrieval(chunks)
    if conf.confidence_level == "HIGH":
        conf = conf.model_copy(update={"confidence_level": "MEDIUM", "confidence_score": 0.72})
    policy = RetrievalPolicy(minimum_confidence="HIGH", allow_clarification=True)
    decision = decide_retrieval_action(conf, policy)
    assert decision.fallback_strategy == "CLARIFY"
    assert decision.should_generate is False


@patch("tools.rag.service.rewrite", side_effect=lambda question: question)
@patch("tools.rag.service.generate")
@patch("tools.rag.service.retrieve")
def test_service_low_confidence_skips_generation(
    mock_retrieve: MagicMock,
    mock_generate: MagicMock,
    mock_rewrite: MagicMock,
) -> None:
    mock_retrieve.return_value = [_chunk("weak", 0.22)]
    response = answer(QueryRequest(question="How do returns work?"))
    assert response.requires_clarification is True
    assert response.fallback_strategy == "CLARIFY"
    assert response.confidence_level == "LOW"
    mock_generate.assert_not_called()


@patch("tools.rag.service.rewrite", side_effect=lambda question: question)
@patch("tools.rag.service.generate")
@patch("tools.rag.service.retrieve")
def test_service_high_confidence_calls_generate(
    mock_retrieve: MagicMock,
    mock_generate: MagicMock,
    mock_rewrite: MagicMock,
) -> None:
    mock_retrieve.return_value = [
        _chunk("a", 0.93, filename="a.pdf"),
        _chunk("b", 0.91, filename="b.pdf"),
        _chunk("c", 0.89, filename="c.pdf"),
        _chunk("d", 0.87, filename="d.pdf"),
        _chunk("e", 0.86, filename="e.pdf"),
    ]
    mock_generate.return_value = "Process returns within 30 days."
    response = answer(QueryRequest(question="How do returns work?"))
    assert response.answer == "Process returns within 30 days."
    assert response.confidence_level == "HIGH"
    assert response.fallback_strategy == "GENERATE_CONFIDENT"
    mock_generate.assert_called_once()
    kwargs = mock_generate.call_args.kwargs
    assert kwargs["confidence_context"]["confidence_level"] == "HIGH"


@patch("tools.rag.generator.get_prompt_manager")
def test_generator_cautious_prompt_for_medium(mock_get_manager: MagicMock) -> None:
    from tools.rag.generator import generate

    mock_get_manager.return_value.invoke.return_value = (
        "Based on the available SOP documentation, returns are due in 30 days."
    )
    answer_text = generate(
        "How do returns work?",
        [_chunk("a", 0.7)],
        confidence_context={
            "confidence_level": "MEDIUM",
            "confidence_score": 0.7,
            "confidence_reason": "partial",
            "fallback_strategy": "GENERATE_CAUTIOUS",
        },
    )
    assert "30 days" in answer_text
    kwargs = mock_get_manager.return_value.invoke.call_args.kwargs
    assert kwargs.get("version") == "v2"


def test_planner_sop_includes_retrieval_policy() -> None:
    from core.intent_classifier import IntentClassification, IntentType
    from core.planner import Planner

    plan = Planner().plan(
        "What is the return policy?",
        IntentClassification(
            intent=IntentType.SOP_QA,
            confidence=0.95,
            reasoning="sop",
            requires_rag=True,
        ),
    )
    assert plan.retrieval_policy is not None
    assert plan.retrieval_policy.get("allow_clarification") is True
    assert plan.steps[0].inputs.get("retrieval_policy") is not None
