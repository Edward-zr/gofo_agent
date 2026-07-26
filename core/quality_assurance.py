"""Multi-stage Quality Assurance pipeline for GOFO agent answers.

The QA pipeline NEVER generates the final answer and NEVER executes tools.
It only evaluates a generated answer and returns a structured QualityReport
for the Planner / agent loop to act on.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

import config
from core.logger import get_logger

logger = get_logger("quality_assurance")

DEFAULT_SCORE_THRESHOLD = 0.75
APPROVE_THRESHOLD = 0.85
MAX_QA_RETRIES = 2

# Attachment / ADA answers store dataframe previews in sql_rows — not SQL evidence.
_FILE_CAPABILITIES = frozenset(
    {
        "file_analysis",
        "file_comparison",
        "file_rag_comparison",
        "attachment",
        "attachment_analysis",
        "attachment_visualization",
    }
)


def is_file_capability(capability: str | None) -> bool:
    """True when the answer came from uploaded-file analysis (not warehouse SQL)."""
    if not capability:
        return False
    value = capability.lower().strip()
    return value in _FILE_CAPABILITIES or value.startswith("file_")


class QAAction(StrEnum):
    """Next action recommended by the Decision Engine."""

    APPROVE = "approve"
    RETRY_RETRIEVAL = "retry_retrieval"
    RETRY_SQL = "retry_sql"
    RETRY_PYTHON = "retry_python"
    RETRY_PLAN = "retry_plan"
    ASK_USER = "ask_user"
    ACCEPT_WITH_LIMITS = "accept_with_limits"


class ValidatorResult(BaseModel):
    """Structured output from a single QA validator."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    feedback: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    should_retry_retrieval: bool = False
    should_retry_sql: bool = False
    should_retry_python: bool = False
    should_ask_user: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))


class QualityReport(BaseModel):
    """Combined QA decision returned to the agent / planner."""

    approved: bool = False
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning_score: float = Field(ge=0.0, le=1.0, default=0.0)
    completeness_score: float = Field(ge=0.0, le=1.0, default=0.0)
    should_retry_retrieval: bool = False
    should_retry_sql: bool = False
    should_retry_python: bool = False
    should_ask_user: bool = False
    retry_reason: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    action: QAAction = QAAction.APPROVE
    validator_results: list[ValidatorResult] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Debug-only explanation.")
    retry_count: int = 0
    max_retries: int = MAX_QA_RETRIES

    @field_validator(
        "overall_confidence",
        "evidence_score",
        "reasoning_score",
        "completeness_score",
        mode="before",
    )
    @classmethod
    def _clamp(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))


class AnswerContext(BaseModel):
    """Evidence bundle evaluated by QA validators (no tool execution)."""

    question: str
    answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    sql: str | None = None
    sql_rows: list[dict[str, Any]] = Field(default_factory=list)
    capability: str | None = None
    execution_plan: dict[str, Any] | None = None
    intent_classification: dict[str, Any] | None = None
    conversation_memory: dict[str, Any] = Field(default_factory=dict)
    root_cause: dict[str, Any] = Field(default_factory=dict)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    ambiguous: bool = False

    model_config = {"arbitrary_types_allowed": True}


class AnswerValidator(Protocol):
    """Protocol for pluggable QA validators."""

    name: str

    def validate(self, context: AnswerContext) -> ValidatorResult: ...


class BaseValidator(ABC):
    """Base class for extensible QA validators."""

    name: str = "base"

    @abstractmethod
    def validate(self, context: AnswerContext) -> ValidatorResult:
        raise NotImplementedError


_HALLUCINATION_MARKERS = (
    "i believe",
    "i assume",
    "probably",
    "might be",
    "could be around",
    "as an ai",
    "i don't have access",
    "i do not have access",
)

_UNKNOWN_MARKERS = (
    "i don't know",
    "i do not know",
    "unable to determine",
    "no data available",
    "unknown_answer",
    "could not find",
)


def _token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9%]{3,}", (text or "").lower())}


def _overlap_ratio(answer_tokens: set[str], evidence_tokens: set[str]) -> float:
    if not answer_tokens:
        return 0.0
    if not evidence_tokens:
        return 0.0
    overlap = answer_tokens & evidence_tokens
    return len(overlap) / max(1, len(answer_tokens))


def _source_texts(sources: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        text = source.get("text") or source.get("content") or ""
        if text:
            texts.append(str(text))
    return texts


def _row_blob(rows: list[dict[str, Any]], limit: int = 40) -> str:
    parts: list[str] = []
    for row in rows[:limit]:
        if isinstance(row, dict):
            parts.append(" ".join(f"{k} {v}" for k, v in row.items()))
    return " ".join(parts)


class EvidenceValidator(BaseValidator):
    """Stage 1 — are claims supported by retrieved evidence / SQL rows?"""

    name = "evidence"

    def validate(self, context: AnswerContext) -> ValidatorResult:
        answer = (context.answer or "").strip()
        feedback: list[str] = []
        missing: list[str] = []
        problems: list[str] = []
        score = 0.5
        retry_retrieval = False
        retry_sql = False

        if not answer:
            return ValidatorResult(
                name=self.name,
                score=0.1,
                feedback=["Answer is empty."],
                problems=["empty_answer"],
                should_retry_retrieval=True,
                should_retry_sql=bool(context.sql is not None or context.capability == "sql"),
            )

        sources = context.sources or []
        rows = context.sql_rows or []
        source_texts = _source_texts(sources)
        evidence_text = " ".join(source_texts) + " " + _row_blob(rows)
        answer_tokens = _token_set(answer)
        evidence_tokens = _token_set(evidence_text)
        overlap = _overlap_ratio(answer_tokens, evidence_tokens)

        capability = (context.capability or "").lower()
        file_mode = is_file_capability(capability)
        if file_mode:
            # Uploaded-file ADA: sql_rows are dataframe previews, not warehouse SQL.
            needs_rag = False
            needs_sql = False
        else:
            needs_rag = capability in {"rag", "multi"} or bool(sources)
            needs_sql = (
                capability in {"sql", "multi", "planner"}
                or bool(context.sql)
                or (bool(rows) and not sources and capability not in {
                    "rag",
                    "conversation",
                    "llm",
                    "clarification",
                    "general",
                })
            )

        # Source score presence / confidence
        source_scores = [
            float(source.get("score", 0.0))
            for source in sources
            if isinstance(source, dict) and source.get("score") is not None
        ]
        avg_source_score = sum(source_scores) / len(source_scores) if source_scores else 0.0

        if file_mode:
            # Ground attachment answers on preview rows / narrative length.
            if rows:
                row_overlap = _overlap_ratio(answer_tokens, _token_set(_row_blob(rows)))
                score = min(0.95, 0.7 + row_overlap * 0.25)
            else:
                score = 0.85 if len(answer) > 40 else 0.65
            if any(marker in answer.lower() for marker in _HALLUCINATION_MARKERS):
                score = min(score, 0.55)
                problems.append("possible_hallucination_language")
                feedback.append("Answer may rely on unsupported assumptions.")
            return ValidatorResult(
                name=self.name,
                score=round(score, 3),
                feedback=feedback,
                missing_information=missing,
                problems=problems,
                should_retry_retrieval=False,
                should_retry_sql=False,
                details={
                    "overlap": round(overlap, 3),
                    "source_count": len(sources),
                    "sql_row_count": len(rows),
                    "file_mode": True,
                },
            )

        if needs_rag and not sources:
            score = 0.35
            retry_retrieval = True
            missing.append("Supporting SOP / document evidence")
            feedback.append("Retrieved documents are insufficient.")
            problems.append("missing_sources")
        elif needs_rag and avg_source_score and avg_source_score < 0.4:
            score = 0.55
            retry_retrieval = True
            feedback.append("Retrieval confidence is low.")
            problems.append("low_retrieval_confidence")
        elif needs_rag:
            score = min(0.95, 0.55 + overlap * 0.4 + min(avg_source_score, 0.9) * 0.2)

        if needs_sql:
            if not rows and not context.sql:
                score = min(score, 0.4) if needs_rag else 0.4
                retry_sql = True
                missing.append("SQL result rows")
                feedback.append("SQL results are missing or insufficient.")
                problems.append("missing_sql_results")
            elif not rows:
                score = min(score, 0.55) if needs_rag else 0.55
                retry_sql = True
                missing.append("Non-empty SQL rows")
                feedback.append("SQL executed but returned no rows.")
                problems.append("empty_sql_rows")
            else:
                row_text = _row_blob(rows)
                sql_overlap = _overlap_ratio(answer_tokens, _token_set(row_text))
                # Numeric grounding: values mentioned in the answer appear in rows.
                numbers = re.findall(r"\d+(?:\.\d+)?", answer)
                row_numbers = set(re.findall(r"\d+(?:\.\d+)?", row_text))
                grounded_numbers = [num for num in numbers if num in row_numbers]
                number_ratio = (
                    len(grounded_numbers) / len(numbers) if numbers else 1.0
                )
                sql_component = 0.62 + sql_overlap * 0.25 + number_ratio * 0.2
                score = max(score, sql_component) if needs_rag else min(0.96, sql_component)

        if not needs_rag and not needs_sql:
            # Conversational / clarification answers need little evidence.
            score = 0.9 if len(answer) > 10 else 0.5

        lowered = answer.lower()
        if any(marker in lowered for marker in _HALLUCINATION_MARKERS):
            score = min(score, 0.45)
            problems.append("possible_hallucination_language")
            feedback.append("Answer may rely on unsupported assumptions.")
            if needs_rag:
                retry_retrieval = True
            if needs_sql:
                retry_sql = True

        if overlap < 0.08 and (sources or rows) and len(answer_tokens) > 8:
            score = min(score, 0.5)
            problems.append("low_evidence_overlap")
            feedback.append("Important claims are weakly grounded in evidence.")
            if sources:
                retry_retrieval = True
            if rows or context.sql:
                retry_sql = True

        if any(marker in lowered for marker in _UNKNOWN_MARKERS) and (needs_rag or needs_sql):
            score = min(score, 0.4)
            problems.append("unknown_or_no_data")
            feedback.append("Answer indicates missing evidence.")
            retry_retrieval = needs_rag or retry_retrieval
            retry_sql = needs_sql or retry_sql

        return ValidatorResult(
            name=self.name,
            score=round(score, 3),
            feedback=feedback,
            missing_information=missing,
            problems=problems,
            should_retry_retrieval=retry_retrieval,
            should_retry_sql=retry_sql,
            details={
                "overlap": round(overlap, 3),
                "source_count": len(sources),
                "sql_row_count": len(rows),
                "avg_source_score": round(avg_source_score, 3),
            },
        )


class ReasoningValidator(BaseValidator):
    """Stage 2 — does the explanation logically follow from the evidence?"""

    name = "reasoning"

    def validate(self, context: AnswerContext) -> ValidatorResult:
        answer = (context.answer or "").strip()
        feedback: list[str] = []
        problems: list[str] = []
        score = 0.9
        retry_python = False

        if not answer:
            return ValidatorResult(
                name=self.name,
                score=0.2,
                feedback=["No answer to evaluate reasoning."],
                problems=["empty_answer"],
            )

        question = (context.question or "").lower()
        lowered = answer.lower()
        rows = context.sql_rows or []

        asks_why = bool(re.search(r"\bwhy\b|root cause|decrease|increase", question))
        asks_compare = any(word in question for word in ("compare", "vs", "versus", "week over week"))
        asks_rank = bool(re.search(r"\b(rank|worst|best|lowest|highest)\b", question))

        if asks_why:
            has_cause = bool(context.root_cause) or any(
                term in lowered
                for term in ("because", "due to", "caused by", "driven by", "root cause", "failure")
            )
            if not has_cause:
                score = min(score, 0.55)
                problems.append("missing_causal_reasoning")
                feedback.append("The answer explains what happened but not why.")
                retry_python = True

        if asks_compare:
            has_compare = any(
                term in lowered for term in ("compared", "versus", "vs", "higher", "lower", "increase", "decrease", "%")
            )
            if not has_compare and len(rows) < 2:
                score = min(score, 0.5)
                problems.append("invalid_or_missing_comparison")
                feedback.append("Comparison logic is incomplete or not grounded in enough rows.")
                retry_python = True

        if asks_rank and rows:
            # Ranking answers should reference an entity from rows when available.
            entities = []
            for row in rows[:10]:
                if not isinstance(row, dict):
                    continue
                for key, value in row.items():
                    if str(key).lower() in {"hub", "hub_name", "driver", "driver_name", "customer"}:
                        entities.append(str(value).lower())
            if entities and not any(entity and entity in lowered for entity in entities):
                score = min(score, 0.6)
                problems.append("ranking_not_tied_to_data")
                feedback.append("Ranking conclusion is not clearly tied to SQL entities.")

        # Numeric claim check: percentages in answer should appear in rows when SQL is present.
        percents = re.findall(r"(\d+(?:\.\d+)?)\s*%", answer)
        if percents and rows:
            row_text = _row_blob(rows).lower()
            unsupported = [p for p in percents if p not in row_text and f"{p}%" not in row_text]
            if unsupported:
                score = min(score, 0.55)
                problems.append("unsupported_numeric_claims")
                feedback.append("Some numeric claims are not present in SQL results.")
                retry_python = True

        plan = context.execution_plan or {}
        steps = plan.get("steps") or []
        if asks_why and steps:
            tools = {str(step.get("tool", "")).upper() for step in steps if isinstance(step, dict)}
            if "PYTHON" not in tools and "SQL" in tools and not context.root_cause:
                score = min(score, 0.65)
                problems.append("planner_skipped_analysis_step")
                feedback.append("Planner may have skipped important analytical logic.")

        if score >= 0.85 and not feedback:
            feedback.append("Reasoning appears consistent with available evidence.")

        return ValidatorResult(
            name=self.name,
            score=round(score, 3),
            feedback=feedback,
            problems=problems,
            should_retry_python=retry_python,
            details={"asks_why": asks_why, "asks_compare": asks_compare, "asks_rank": asks_rank},
        )


class CompletenessValidator(BaseValidator):
    """Stage 3 — did the answer fully address the user question?"""

    name = "completeness"

    def validate(self, context: AnswerContext) -> ValidatorResult:
        answer = (context.answer or "").strip()
        question = (context.question or "").strip()
        feedback: list[str] = []
        missing: list[str] = []
        problems: list[str] = []
        score = 0.75
        ask_user = False

        if context.ambiguous:
            return ValidatorResult(
                name=self.name,
                score=0.4,
                feedback=["Question appears ambiguous."],
                missing_information=["Clarified time range or metric"],
                problems=["ambiguous_question"],
                should_ask_user=True,
            )

        if not answer:
            return ValidatorResult(
                name=self.name,
                score=0.1,
                feedback=["Answer is empty."],
                problems=["empty_answer"],
                missing_information=["Full answer to the user question"],
            )

        q = question.lower()
        a = answer.lower()

        requested_topics: list[tuple[str, tuple[str, ...]]] = []
        if re.search(r"\bwhy\b|root cause", q):
            requested_topics.append(("Root cause analysis", ("because", "due to", "cause", "failure", "reason")))
        if any(word in q for word in ("compare", "vs", "versus", "week over week", "last week")):
            requested_topics.append(
                ("Previous week / comparison", ("compare", "versus", "vs", "last week", "previous", "higher", "lower"))
            )
        if any(word in q for word in ("recommend", "what should", "next action")):
            requested_topics.append(("Recommendation", ("recommend", "should", "next", "action", "mitigate")))
        if any(word in q for word in ("trend", "over time", "dashboard")):
            requested_topics.append(("Trend / visualization", ("trend", "over time", "chart", "increased", "decreased")))
        if "fail" in q or "exception" in q:
            requested_topics.append(("Top failure reasons", ("fail", "exception", "reason", "delay")))

        covered = 0
        for topic, keywords in requested_topics:
            if any(keyword in a for keyword in keywords) or (
                topic.lower().startswith("root") and context.root_cause
            ):
                covered += 1
            else:
                missing.append(topic)
                problems.append(f"missing_topic:{topic}")

        if requested_topics:
            coverage = covered / len(requested_topics)
            score = 0.35 + coverage * 0.6
            if missing:
                feedback.append("The answer only partially addresses the requested points.")
        else:
            # Generic completeness: short answers to multi-clause questions.
            clauses = [part.strip() for part in re.split(r"[?,]| and | also ", q) if len(part.strip()) > 8]
            if len(clauses) >= 2 and len(answer.split()) < 25:
                score = 0.55
                problems.append("short_answer_for_multi_part_question")
                feedback.append("Answer may be too brief for a multi-part question.")
            else:
                score = 0.88

        # Follow-up awareness
        memory = context.conversation_memory or {}
        if memory.get("previous_question") and len(question.split()) <= 6:
            if not any(token in a for token in _token_set(str(memory.get("previous_question")))) and not rows_have_filter(
                context
            ):
                score = min(score, 0.6)
                problems.append("follow_up_context_ignored")
                feedback.append("Follow-up may ignore prior conversational context.")
                missing.append("Inherited filters from previous turn")

        if any(marker in a for marker in ("which date", "which period", "do you want", "please clarify")):
            ask_user = True
            score = min(score, 0.5)
            problems.append("needs_clarification")

        return ValidatorResult(
            name=self.name,
            score=round(score, 3),
            feedback=feedback,
            missing_information=missing,
            problems=problems,
            should_ask_user=ask_user,
            details={"requested_topic_count": len(requested_topics), "covered": covered},
        )


def rows_have_filter(context: AnswerContext) -> bool:
    filters = (context.conversation_memory or {}).get("filters") or {}
    return bool(filters)


class DecisionEngine:
    """Combine validator results into a single QualityReport / next action."""

    def __init__(
        self,
        *,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        approve_threshold: float = APPROVE_THRESHOLD,
        max_retries: int = MAX_QA_RETRIES,
    ) -> None:
        self.score_threshold = score_threshold
        self.approve_threshold = approve_threshold
        self.max_retries = max_retries

    def decide(
        self,
        results: list[ValidatorResult],
        *,
        retry_count: int = 0,
        ambiguous: bool = False,
        capability: str | None = None,
    ) -> QualityReport:
        by_name = {result.name: result for result in results}
        evidence = by_name.get("evidence")
        reasoning = by_name.get("reasoning")
        completeness = by_name.get("completeness")

        evidence_score = evidence.score if evidence else 1.0
        reasoning_score = reasoning.score if reasoning else 1.0
        completeness_score = completeness.score if completeness else 1.0

        # Future validators contribute to overall confidence automatically.
        scores = [result.score for result in results] or [0.0]
        overall = sum(scores) / len(scores)

        feedback: list[str] = []
        missing: list[str] = []
        for result in results:
            feedback.extend(result.feedback)
            missing.extend(result.missing_information)

        should_ask = ambiguous or any(result.should_ask_user for result in results)
        should_retry_retrieval = any(result.should_retry_retrieval for result in results)
        should_retry_sql = any(result.should_retry_sql for result in results)
        should_retry_python = any(result.should_retry_python for result in results)
        file_mode = is_file_capability(capability)

        # Never send attachment/ADA answers through warehouse SQL retries.
        if file_mode:
            should_retry_sql = False
            should_retry_retrieval = False

        # Completeness failures imply additional planner steps (often SQL/python).
        if completeness_score < self.score_threshold and not file_mode:
            if any("comparison" in item.lower() or "failure" in item.lower() for item in missing):
                should_retry_sql = True
            if any("root cause" in item.lower() for item in missing):
                should_retry_sql = True
                should_retry_python = True

        action = QAAction.APPROVE
        retry_reason = None
        approved = False

        if should_ask:
            action = QAAction.ASK_USER
            retry_reason = "Question is ambiguous or clarification is required."
        elif (
            evidence_score >= self.approve_threshold
            and reasoning_score >= self.approve_threshold
            and completeness_score >= self.approve_threshold
        ):
            approved = True
            action = QAAction.APPROVE
        elif (
            evidence_score >= self.score_threshold
            and reasoning_score >= self.score_threshold
            and completeness_score >= self.score_threshold
        ):
            # Good enough for return even if not all above the strict approve bar.
            approved = True
            action = QAAction.APPROVE
        elif retry_count >= self.max_retries:
            action = QAAction.ACCEPT_WITH_LIMITS
            approved = True
            retry_reason = "Retry limit reached; returning best available answer."
        elif file_mode:
            # Attachment answers are grounded in the upload — accept rather than SQL-retry.
            approved = True
            action = QAAction.ACCEPT_WITH_LIMITS if overall < self.score_threshold else QAAction.APPROVE
            retry_reason = (
                "Attachment analysis accepted without warehouse SQL retry."
                if action == QAAction.ACCEPT_WITH_LIMITS
                else None
            )
        elif evidence_score < self.score_threshold and should_retry_retrieval and not should_retry_sql:
            action = QAAction.RETRY_RETRIEVAL
            retry_reason = "Missing supporting evidence"
        elif reasoning_score < self.score_threshold:
            action = QAAction.RETRY_PLAN
            retry_reason = "Reasoning does not reliably follow from evidence"
            should_retry_python = True
        elif completeness_score < self.score_threshold:
            action = QAAction.RETRY_PLAN
            retry_reason = "Answer is incomplete for the user request"
            if should_retry_sql is False and should_retry_python is False:
                should_retry_sql = True
        elif should_retry_sql or (evidence_score < self.score_threshold and should_retry_sql is False and should_retry_retrieval is False):
            if should_retry_sql or evidence_score < self.score_threshold:
                action = QAAction.RETRY_SQL if should_retry_sql or not should_retry_retrieval else QAAction.RETRY_RETRIEVAL
                retry_reason = "SQL evidence is insufficient" if action == QAAction.RETRY_SQL else "Missing supporting evidence"
                if action == QAAction.RETRY_SQL:
                    should_retry_sql = True
        elif should_retry_python:
            action = QAAction.RETRY_PYTHON
            retry_reason = "Additional calculation / analysis steps needed"
        else:
            approved = overall >= self.score_threshold
            action = QAAction.APPROVE if approved else QAAction.RETRY_PLAN
            if not approved:
                retry_reason = "Overall quality below threshold"

        # Deduplicate lists while preserving order.
        feedback = list(dict.fromkeys(feedback))
        missing = list(dict.fromkeys(missing))

        reasoning_text = (
            f"evidence={evidence_score:.2f}, reasoning={reasoning_score:.2f}, "
            f"completeness={completeness_score:.2f}, action={action.value}"
        )
        report = QualityReport(
            approved=approved,
            overall_confidence=round(overall, 3),
            evidence_score=round(evidence_score, 3),
            reasoning_score=round(reasoning_score, 3),
            completeness_score=round(completeness_score, 3),
            should_retry_retrieval=should_retry_retrieval and action != QAAction.ASK_USER,
            should_retry_sql=should_retry_sql and action not in {QAAction.ASK_USER, QAAction.APPROVE},
            should_retry_python=should_retry_python and action not in {QAAction.ASK_USER, QAAction.APPROVE},
            should_ask_user=action == QAAction.ASK_USER,
            retry_reason=retry_reason,
            missing_information=missing,
            feedback=feedback,
            action=action,
            validator_results=results,
            reasoning=reasoning_text,
            retry_count=retry_count,
            max_retries=self.max_retries,
        )
        _debug_print(report)
        return report


class QualityAssurancePipeline:
    """Run registered validators and produce a QualityReport.

    Validators never execute tools and never generate user-facing answers.
    """

    def __init__(
        self,
        validators: list[BaseValidator] | None = None,
        decision_engine: DecisionEngine | None = None,
    ) -> None:
        self.validators: list[BaseValidator] = list(
            validators
            or [
                EvidenceValidator(),
                ReasoningValidator(),
                CompletenessValidator(),
            ]
        )
        self.decision_engine = decision_engine or DecisionEngine()

    def register_validator(self, validator: BaseValidator) -> None:
        """Append a future validator without changing DecisionEngine structure."""
        self.validators.append(validator)

    def evaluate(
        self,
        context: AnswerContext,
        *,
        retry_count: int = 0,
    ) -> QualityReport:
        results = [validator.validate(context) for validator in self.validators]
        return self.decision_engine.decide(
            results,
            retry_count=retry_count,
            ambiguous=context.ambiguous,
            capability=context.capability,
        )


def build_answer_context_from_response(
    *,
    question: str,
    response: Any,
    conversation_memory: Any = None,
    ambiguous: bool = False,
) -> AnswerContext:
    """Adapter from QueryResponse / dict-like agent outputs."""
    memory_state: dict[str, Any] = {}
    if conversation_memory is not None and hasattr(conversation_memory, "get_current_state"):
        memory_state = dict(conversation_memory.get_current_state() or {})
    elif isinstance(conversation_memory, dict):
        memory_state = conversation_memory

    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        data = {
            "answer": getattr(response, "answer", ""),
            "sources": getattr(response, "sources", []) or [],
            "generated_sql": getattr(response, "generated_sql", None),
            "sql_rows": getattr(response, "sql_rows", []) or [],
            "capability": getattr(response, "capability", None),
            "execution_plan": getattr(response, "execution_plan", None),
            "intent_classification": getattr(response, "intent_classification", None),
            "root_cause": getattr(response, "root_cause", {}) or {},
            "charts": getattr(response, "charts", []) or [],
        }

    sources = data.get("sources") or []
    normalized_sources: list[dict[str, Any]] = []
    for source in sources:
        if hasattr(source, "model_dump"):
            normalized_sources.append(source.model_dump())
        elif isinstance(source, dict):
            normalized_sources.append(source)

    return AnswerContext(
        question=question,
        answer=str(data.get("answer") or ""),
        sources=normalized_sources,
        sql=data.get("generated_sql") or data.get("sql"),
        sql_rows=list(data.get("sql_rows") or []),
        capability=data.get("capability"),
        execution_plan=data.get("execution_plan"),
        intent_classification=data.get("intent_classification"),
        conversation_memory=memory_state,
        root_cause=dict(data.get("root_cause") or {}),
        charts=list(data.get("charts") or []),
        ambiguous=ambiguous,
    )


def _debug_print(report: QualityReport) -> None:
    lines = [
        "----------------------------------",
        "QA Quality Report",
        f"Evidence Score: {report.evidence_score}",
        f"Reasoning Score: {report.reasoning_score}",
        f"Completeness Score: {report.completeness_score}",
        f"Overall Confidence: {report.overall_confidence}",
        f"Retry Decision: {report.action.value}",
        f"Approved: {report.approved}",
        f"Missing Information: {report.missing_information}",
        f"Retry Reason: {report.retry_reason}",
        f"Feedback: {report.feedback}",
    ]
    if config.DEBUG:
        lines.append(f"Reasoning: {report.reasoning}")
    lines.append("----------------------------------")
    message = "\n".join(lines)
    if config.DEBUG:
        print(message)
    logger.info(message)
