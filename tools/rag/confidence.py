"""Adaptive Retrieval Confidence Engine for GOFO RAG.

Evaluates retrieval quality from multiple signals (no single similarity threshold)
and selects a generation / fallback strategy according to Planner policy.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

import config
from core.models import SourceChunk
from core.logger import get_logger

logger = get_logger("retrieval_confidence")

ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]
FallbackStrategy = Literal[
    "GENERATE_CONFIDENT",
    "GENERATE_CAUTIOUS",
    "CLARIFY",
    "DOCUMENT_REQUEST",
    "GENERAL_KNOWLEDGE",
    "REFUSE",
]
MetadataQualityLabel = Literal["GOOD", "FAIR", "POOR"]

_LEVEL_RANK: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

DEFAULT_WEIGHTS: dict[str, float] = {
    "highest_similarity": 0.40,
    "average_similarity": 0.30,
    "chunk_count": 0.15,
    "source_diversity": 0.10,
    "metadata_quality": 0.05,
}

CLARIFY_MESSAGE = (
    "I couldn't find enough information to answer confidently.\n"
    "Could you specify the warehouse, process, or driver you are referring to?"
)

DOCUMENT_REQUEST_MESSAGE = (
    "I don't currently have enough supporting documentation.\n"
    "Would you like to upload the relevant SOP or policy document?"
)

GENERAL_KNOWLEDGE_DISCLAIMER = (
    "This answer is based on general logistics knowledge rather than your "
    "internal SOP documentation."
)


class RetrievalPolicy(BaseModel):
    """Planner-defined retrieval / fallback policy."""

    allow_general_knowledge: bool = False
    minimum_confidence: ConfidenceLevel = "MEDIUM"
    allow_clarification: bool = True
    allow_document_request: bool = True


class ConfidenceBreakdown(BaseModel):
    highest_similarity: float = 0.0
    average_similarity: float = 0.0
    chunk_count: int = 0
    source_diversity: int = 0
    metadata_quality: MetadataQualityLabel = "POOR"
    # Normalized component scores used in the weighted sum (0–1 each)
    highest_similarity_score: float = 0.0
    average_similarity_score: float = 0.0
    chunk_count_score: float = 0.0
    source_diversity_score: float = 0.0
    metadata_quality_score: float = 0.0


class RetrievalConfidence(BaseModel):
    """Full adaptive confidence evaluation for a retrieval set."""

    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_level: ConfidenceLevel = "LOW"
    confidence_breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    reason: str = ""
    similarity_scores: list[float] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    retrieved_sources: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


class RetrievalDecision(BaseModel):
    """Decision Engine output consumed by the Generator / agent."""

    confidence: RetrievalConfidence
    policy: RetrievalPolicy
    fallback_strategy: FallbackStrategy
    should_generate: bool = False
    use_general_knowledge: bool = False
    requires_clarification: bool = False
    message: str | None = None
    reason: str = ""


def default_retrieval_policy() -> RetrievalPolicy:
    """Default policy when the Planner does not supply one."""
    return RetrievalPolicy(
        allow_general_knowledge=False,
        minimum_confidence="MEDIUM",
        allow_clarification=True,
        allow_document_request=True,
    )


def policy_from_mapping(raw: Any) -> RetrievalPolicy:
    """Parse a RetrievalPolicy from plan inputs / dict / model."""
    if isinstance(raw, RetrievalPolicy):
        return raw
    if isinstance(raw, dict):
        try:
            return RetrievalPolicy.model_validate(raw)
        except Exception:
            return default_retrieval_policy()
    return default_retrieval_policy()


def _weights() -> dict[str, float]:
    """Load tunable weights from config with sane fallbacks."""
    base = dict(DEFAULT_WEIGHTS)
    overrides = getattr(config, "RETRIEVAL_CONFIDENCE_WEIGHTS", None)
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in base:
                try:
                    base[key] = float(value)
                except (TypeError, ValueError):
                    continue
    total = sum(base.values()) or 1.0
    return {key: value / total for key, value in base.items()}


def _source_name(metadata: dict[str, Any] | None) -> str:
    meta = metadata or {}
    for key in ("filename", "source", "document", "doc_id"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _has_page(metadata: dict[str, Any] | None) -> bool:
    meta = metadata or {}
    if meta.get("page_label") is not None and str(meta.get("page_label")).strip():
        return True
    return meta.get("page") is not None


def _chunk_count_score(count: int, *, target: int) -> float:
    if count <= 0:
        return 0.0
    target = max(1, target)
    # Saturate at the expected top_k (or a small floor of 3).
    return min(1.0, count / float(target))


def _source_diversity_score(unique_sources: int, chunk_count: int) -> float:
    if unique_sources <= 0 or chunk_count <= 0:
        return 0.0
    # Reward multiple distinct sources; single-source retrieval stays partial.
    if unique_sources == 1:
        return 0.35 if chunk_count >= 2 else 0.2
    if unique_sources == 2:
        return 0.65
    if unique_sources == 3:
        return 0.85
    return 1.0


def _metadata_quality(
    chunks: list[SourceChunk],
) -> tuple[MetadataQualityLabel, float]:
    if not chunks:
        return "POOR", 0.0
    good = 0
    for chunk in chunks:
        meta = chunk.metadata or {}
        has_source = _source_name(meta) != "unknown"
        has_page = _has_page(meta)
        has_text = bool((chunk.text or "").strip())
        if has_source and has_page and has_text:
            good += 1
        elif has_source and has_text:
            good += 0.5
    ratio = good / len(chunks)
    if ratio >= 0.75:
        return "GOOD", 1.0
    if ratio >= 0.4:
        return "FAIR", 0.6
    return "POOR", 0.25


def _level_for_score(score: float) -> ConfidenceLevel:
    high = float(getattr(config, "RETRIEVAL_CONFIDENCE_HIGH", 0.80))
    medium = float(getattr(config, "RETRIEVAL_CONFIDENCE_MEDIUM", 0.60))
    if score >= high:
        return "HIGH"
    if score >= medium:
        return "MEDIUM"
    return "LOW"


def _reason_for(
    level: ConfidenceLevel,
    *,
    highest: float,
    average: float,
    chunk_count: int,
    source_diversity: int,
    metadata_quality: MetadataQualityLabel,
) -> str:
    if level == "HIGH":
        return (
            f"Multiple highly relevant chunks from diverse sources "
            f"(top={highest:.2f}, avg={average:.2f}, chunks={chunk_count}, "
            f"sources={source_diversity}, metadata={metadata_quality})."
        )
    if level == "MEDIUM":
        return (
            f"Partial evidence with some uncertainty "
            f"(top={highest:.2f}, avg={average:.2f}, chunks={chunk_count}, "
            f"sources={source_diversity}, metadata={metadata_quality})."
        )
    if chunk_count == 0:
        return "No documents were retrieved; high hallucination risk."
    return (
        f"Weak retrieval evidence "
        f"(top={highest:.2f}, avg={average:.2f}, chunks={chunk_count}, "
        f"sources={source_diversity}, metadata={metadata_quality})."
    )


def evaluate_retrieval(
    chunks: list[SourceChunk] | None,
    *,
    retrieval_metadata: dict[str, Any] | None = None,
    expected_chunk_count: int | None = None,
) -> RetrievalConfidence:
    """Compute adaptive confidence from retrieved chunks and metadata."""
    chunks = list(chunks or [])
    weights = _weights()
    scores = [float(chunk.score) for chunk in chunks]
    highest = max(scores) if scores else 0.0
    average = (sum(scores) / len(scores)) if scores else 0.0
    sources = sorted({_source_name(chunk.metadata) for chunk in chunks})
    # Drop placeholder only when nothing else exists.
    if sources == ["unknown"] and not chunks:
        sources = []
    unique_sources = len([name for name in sources if name != "unknown"]) or (
        1 if chunks else 0
    )
    if chunks and all(_source_name(c.metadata) == "unknown" for c in chunks):
        unique_sources = 1

    target = expected_chunk_count or int(
        getattr(config, "TOP_K_DEFAULT", 5) or 5
    )
    meta_label, meta_score = _metadata_quality(chunks)
    chunk_score = _chunk_count_score(len(chunks), target=target)
    diversity_score = _source_diversity_score(unique_sources if chunks else 0, len(chunks))

    # Similarity components are already 0–1-ish (hybrid clamps ~0.40–0.95).
    highest_score = max(0.0, min(1.0, highest))
    average_score = max(0.0, min(1.0, average))

    confidence_score = (
        weights["highest_similarity"] * highest_score
        + weights["average_similarity"] * average_score
        + weights["chunk_count"] * chunk_score
        + weights["source_diversity"] * diversity_score
        + weights["metadata_quality"] * meta_score
    )
    confidence_score = max(0.0, min(1.0, round(confidence_score, 4)))
    level = _level_for_score(confidence_score)
    reason = _reason_for(
        level,
        highest=highest,
        average=average,
        chunk_count=len(chunks),
        source_diversity=unique_sources if chunks else 0,
        metadata_quality=meta_label,
    )

    # Optional retrieval metadata can refine the reason (never a hard gate).
    if retrieval_metadata:
        extra = []
        if retrieval_metadata.get("mode"):
            extra.append(f"mode={retrieval_metadata.get('mode')}")
        if retrieval_metadata.get("rewritten_question"):
            extra.append("query was rewritten")
        if extra:
            reason = f"{reason} ({'; '.join(extra)})"

    breakdown = ConfidenceBreakdown(
        highest_similarity=round(highest, 4),
        average_similarity=round(average, 4),
        chunk_count=len(chunks),
        source_diversity=unique_sources if chunks else 0,
        metadata_quality=meta_label,
        highest_similarity_score=round(highest_score, 4),
        average_similarity_score=round(average_score, 4),
        chunk_count_score=round(chunk_score, 4),
        source_diversity_score=round(diversity_score, 4),
        metadata_quality_score=round(meta_score, 4),
    )

    result = RetrievalConfidence(
        confidence_score=confidence_score,
        confidence_level=level,
        confidence_breakdown=breakdown,
        reason=reason,
        similarity_scores=[round(s, 4) for s in scores],
        retrieved_chunk_count=len(chunks),
        retrieved_sources=sources,
        weights=weights,
    )
    _debug_confidence(result, retrieval_metadata=retrieval_metadata)
    return result


def _meets_minimum(level: ConfidenceLevel, minimum: ConfidenceLevel) -> bool:
    return _LEVEL_RANK.get(level, 0) >= _LEVEL_RANK.get(minimum, 1)


def decide_retrieval_action(
    confidence: RetrievalConfidence,
    policy: RetrievalPolicy | None = None,
) -> RetrievalDecision:
    """Map confidence + Planner policy to a generation / fallback strategy."""
    policy = policy or default_retrieval_policy()
    level = confidence.confidence_level

    if level == "HIGH" and _meets_minimum(level, policy.minimum_confidence):
        return RetrievalDecision(
            confidence=confidence,
            policy=policy,
            fallback_strategy="GENERATE_CONFIDENT",
            should_generate=True,
            reason="Strong retrieval evidence; generate a grounded SOP answer.",
        )

    if level == "MEDIUM" and _meets_minimum(level, policy.minimum_confidence):
        return RetrievalDecision(
            confidence=confidence,
            policy=policy,
            fallback_strategy="GENERATE_CAUTIOUS",
            should_generate=True,
            reason="Partial evidence; generate a cautious grounded answer.",
        )

    # LOW, or below planner minimum — never answer confidently from weak retrieval.
    if policy.allow_clarification:
        return RetrievalDecision(
            confidence=confidence,
            policy=policy,
            fallback_strategy="CLARIFY",
            requires_clarification=True,
            message=CLARIFY_MESSAGE,
            reason="Weak evidence; ask for clarification before answering.",
        )
    if policy.allow_document_request:
        return RetrievalDecision(
            confidence=confidence,
            policy=policy,
            fallback_strategy="DOCUMENT_REQUEST",
            message=DOCUMENT_REQUEST_MESSAGE,
            reason="Weak evidence; request additional documentation.",
        )
    if policy.allow_general_knowledge:
        return RetrievalDecision(
            confidence=confidence,
            policy=policy,
            fallback_strategy="GENERAL_KNOWLEDGE",
            should_generate=True,
            use_general_knowledge=True,
            reason="Weak evidence; Planner permits general-knowledge fallback.",
        )
    return RetrievalDecision(
        confidence=confidence,
        policy=policy,
        fallback_strategy="REFUSE",
        message=(
            "I don't know based on the available SOP documents, and general "
            "knowledge fallback is not permitted for this question."
        ),
        reason="Weak evidence; refuse rather than invent an answer.",
    )


def evaluate_and_decide(
    chunks: list[SourceChunk] | None,
    *,
    policy: RetrievalPolicy | dict[str, Any] | None = None,
    retrieval_metadata: dict[str, Any] | None = None,
    expected_chunk_count: int | None = None,
) -> RetrievalDecision:
    """Convenience: evaluate retrieval then apply the Decision Engine."""
    confidence = evaluate_retrieval(
        chunks,
        retrieval_metadata=retrieval_metadata,
        expected_chunk_count=expected_chunk_count,
    )
    decision = decide_retrieval_action(confidence, policy_from_mapping(policy))
    _debug_decision(decision)
    return decision


def confidence_context_for_generator(decision: RetrievalDecision) -> dict[str, Any]:
    """Payload passed into the Generator."""
    return {
        "confidence_level": decision.confidence.confidence_level,
        "confidence_score": decision.confidence.confidence_score,
        "confidence_reason": decision.confidence.reason,
        "fallback_strategy": decision.fallback_strategy,
        "use_general_knowledge": decision.use_general_knowledge,
    }


def agent_state_from_decision(decision: RetrievalDecision) -> dict[str, Any]:
    """Fields to merge into AgentState / QueryResponse.agent_state."""
    conf = decision.confidence
    breakdown = conf.confidence_breakdown.model_dump()
    return {
        "retrieval_confidence": conf.model_dump(),
        "confidence_score": conf.confidence_score,
        "confidence_level": conf.confidence_level,
        "confidence_breakdown": breakdown,
        "similarity_scores": list(conf.similarity_scores),
        "retrieved_chunk_count": conf.retrieved_chunk_count,
        "retrieved_sources": list(conf.retrieved_sources),
        "confidence_reason": conf.reason,
        "fallback_strategy": decision.fallback_strategy,
        "retrieval_policy": decision.policy.model_dump(),
    }


def _debug_enabled() -> bool:
    return bool(
        getattr(config, "DEBUG", False)
        or getattr(config, "RETRIEVAL_CONFIDENCE_DEBUG", False)
        or getattr(config, "HYBRID_RETRIEVAL_DEBUG", False)
    )


def _debug_confidence(
    confidence: RetrievalConfidence,
    *,
    retrieval_metadata: dict[str, Any] | None = None,
) -> None:
    message = (
        f"score={confidence.confidence_score} level={confidence.confidence_level}\n"
        f"  similarities={confidence.similarity_scores}\n"
        f"  highest={confidence.confidence_breakdown.highest_similarity} "
        f"avg={confidence.confidence_breakdown.average_similarity}\n"
        f"  chunks={confidence.retrieved_chunk_count} "
        f"sources={confidence.retrieved_sources} "
        f"diversity={confidence.confidence_breakdown.source_diversity}\n"
        f"  metadata={confidence.confidence_breakdown.metadata_quality}\n"
        f"  reason={confidence.reason}"
    )
    if retrieval_metadata:
        message += f"\n  retrieval_metadata={retrieval_metadata}"
    if _debug_enabled():
        print("----------------------------------")
        print("RetrievalConfidenceEngine")
        print(message)
        print("----------------------------------")
    logger.info(message)


def _debug_decision(decision: RetrievalDecision) -> None:
    message = (
        f"strategy={decision.fallback_strategy} "
        f"generate={decision.should_generate} "
        f"clarify={decision.requires_clarification}\n"
        f"  policy={decision.policy.model_dump()}\n"
        f"  reason={decision.reason}"
    )
    if _debug_enabled():
        print("----------------------------------")
        print("RetrievalDecisionEngine")
        print(message)
        print("----------------------------------")
    logger.info(message)
