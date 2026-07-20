"""Reflection (self-critique) layer for the GOFO Operations Intelligence Agent.

The ReflectionAgent NEVER generates the final answer and NEVER executes tools.
It only critiques a generated answer and returns structured feedback for the
Planner / agent retry loop.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

import config
from core.logger import get_logger
from core.quality_assurance import (
    AnswerContext,
    DecisionEngine,
    QAAction,
    QualityAssurancePipeline,
    QualityReport,
    build_answer_context_from_response,
)
from tools.llm.client import get_llm

logger = get_logger("reflection")

MAX_REFLECTION_RETRIES = 2
REFLECTION_APPROVE_CONFIDENCE = 0.85


class ReflectionResult(BaseModel):
    """Structured self-critique of a generated answer."""

    approved: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    missing_information: list[str] = Field(default_factory=list)
    should_retry_retrieval: bool = False
    should_retry_sql: bool = False
    should_retry_python: bool = False
    should_ask_user: bool = False
    clarification_question: str | None = None
    feedback: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Debug-only explanation.")
    suggested_tool_calls: list[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = MAX_REFLECTION_RETRIES
    quality_report: dict[str, Any] | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    def needs_retry(self) -> bool:
        """Return True when Planner should run additional steps."""
        if self.approved or self.should_ask_user:
            return False
        return (
            self.should_retry_retrieval
            or self.should_retry_sql
            or self.should_retry_python
            or bool(self.feedback)
        )


def _default_clarification(missing: list[str]) -> str:
    missing_block = ""
    if missing:
        missing_block = "\n\nI still need:\n" + "\n".join(f"• {item}" for item in missing[:6])
    return (
        "Which date range would you like to compare?\n\n"
        "For example:\n"
        "• today vs yesterday\n"
        "• this week vs last week\n"
        "• this month vs last month"
        f"{missing_block}"
    )


def _suggested_tools(result_flags: dict[str, bool]) -> list[str]:
    tools: list[str] = []
    if result_flags.get("should_retry_retrieval"):
        tools.append("RAG")
    if result_flags.get("should_retry_sql"):
        tools.append("SQL")
    if result_flags.get("should_retry_python"):
        tools.append("PYTHON")
    if result_flags.get("should_ask_user"):
        tools.append("CLARIFY")
    return tools


def _from_quality_report(
    report: QualityReport,
    *,
    retry_count: int = 0,
    max_retries: int = MAX_REFLECTION_RETRIES,
) -> ReflectionResult:
    """Map a QualityReport into the ReflectionResult contract."""
    approved = bool(report.approved)
    confidence = float(report.overall_confidence)

    # After max retries, force accept regardless of remaining gaps.
    if retry_count >= max_retries and not report.should_ask_user:
        approved = True

    should_ask = bool(report.should_ask_user) or report.action == QAAction.ASK_USER
    clarification = None
    if should_ask:
        clarification = _default_clarification(report.missing_information)

    flags = {
        "should_retry_retrieval": bool(report.should_retry_retrieval) and not should_ask and not approved,
        "should_retry_sql": bool(report.should_retry_sql) and not should_ask and not approved,
        "should_retry_python": bool(report.should_retry_python) and not should_ask and not approved,
        "should_ask_user": should_ask,
    }

    feedback = list(report.feedback or [])
    if report.retry_reason and report.retry_reason not in feedback:
        feedback.append(report.retry_reason)

    return ReflectionResult(
        approved=approved,
        confidence=confidence,
        missing_information=list(report.missing_information or []),
        should_retry_retrieval=flags["should_retry_retrieval"],
        should_retry_sql=flags["should_retry_sql"],
        should_retry_python=flags["should_retry_python"],
        should_ask_user=should_ask,
        clarification_question=clarification,
        feedback=feedback,
        reasoning=report.reasoning or "",
        suggested_tool_calls=_suggested_tools(flags),
        retry_count=retry_count,
        max_retries=max_retries,
        quality_report=report.model_dump(),
    )


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def _llm_system_prompt() -> str:
    return (
        "You are a strict operations answer critic for GOFO logistics.\n"
        "You NEVER write a replacement answer for the user.\n"
        "You ONLY critique the draft answer.\n\n"
        "Evaluate:\n"
        "1. Did the answer directly answer the user's question?\n"
        "2. Was enough evidence retrieved?\n"
        "3. Are important facts missing?\n"
        "4. Are there unsupported claims?\n"
        "5. Are SQL results sufficient?\n"
        "6. Is another retrieval likely to improve the answer?\n"
        "7. Should another tool be executed?\n"
        "8. Is confidence high enough to return this answer?\n\n"
        "Return ONLY JSON with this shape:\n"
        "{\n"
        '  "approved": false,\n'
        '  "confidence": 0.67,\n'
        '  "should_retry_retrieval": true,\n'
        '  "should_retry_sql": false,\n'
        '  "should_retry_python": false,\n'
        '  "should_ask_user": false,\n'
        '  "missing_information": ["..."],\n'
        '  "feedback": ["..."],\n'
        '  "clarification_question": null,\n'
        '  "reasoning": "debug only"\n'
        "}"
    )


class ReflectionAgent:
    """AI reviewer that critiques generated answers without answering the user."""

    def __init__(
        self,
        *,
        qa_pipeline: QualityAssurancePipeline | None = None,
        llm: Any | None = None,
        use_llm: bool = False,
        max_retries: int = MAX_REFLECTION_RETRIES,
        approve_confidence: float = REFLECTION_APPROVE_CONFIDENCE,
    ) -> None:
        self.qa_pipeline = qa_pipeline or QualityAssurancePipeline(
            decision_engine=DecisionEngine(
                score_threshold=getattr(config, "QA_SCORE_THRESHOLD", 0.75),
                approve_threshold=getattr(config, "QA_APPROVE_THRESHOLD", 0.85),
                max_retries=max_retries,
            )
        )
        self._llm = llm
        self.use_llm = use_llm
        self.max_retries = max_retries
        self.approve_confidence = approve_confidence

    def critique(
        self,
        question: str,
        answer: str,
        *,
        sources: list[dict[str, Any]] | None = None,
        sql: str | None = None,
        sql_rows: list[dict[str, Any]] | None = None,
        capability: str | None = None,
        execution_plan: dict[str, Any] | None = None,
        conversation_memory: Any = None,
        previous_answer: str | None = None,
        ambiguous: bool = False,
        retry_count: int = 0,
        response: Any | None = None,
    ) -> ReflectionResult:
        """Critique a draft answer and return structured reflection feedback."""
        if response is not None:
            context = build_answer_context_from_response(
                question=question,
                response=response,
                conversation_memory=conversation_memory,
                ambiguous=ambiguous,
            )
            if answer:
                context = context.model_copy(update={"answer": answer})
        else:
            memory_state: dict[str, Any] = {}
            if conversation_memory is not None and hasattr(conversation_memory, "get_current_state"):
                memory_state = dict(conversation_memory.get_current_state() or {})
            elif isinstance(conversation_memory, dict):
                memory_state = conversation_memory
            context = AnswerContext(
                question=question,
                answer=answer or "",
                sources=sources or [],
                sql=sql,
                sql_rows=sql_rows or [],
                capability=capability,
                execution_plan=execution_plan,
                conversation_memory=memory_state,
                ambiguous=ambiguous,
            )

        qa_report = self.qa_pipeline.evaluate(context, retry_count=retry_count)
        result = _from_quality_report(
            qa_report,
            retry_count=retry_count,
            max_retries=self.max_retries,
        )

        if self.use_llm:
            result = self._merge_llm_critique(
                result,
                question=question,
                answer=context.answer,
                previous_answer=previous_answer,
                context=context,
            )

        # Confidence gate: even if validators lean approve, low confidence retries when allowed.
        if (
            result.approved
            and result.confidence < self.approve_confidence
            and retry_count < self.max_retries
            and not result.should_ask_user
        ):
            result = result.model_copy(
                update={
                    "approved": False,
                    "should_retry_sql": result.should_retry_sql
                    or (context.capability or "").lower() in {"sql", "planner", "multi"},
                    "should_retry_retrieval": result.should_retry_retrieval
                    or (context.capability or "").lower() in {"rag", "multi"},
                    "feedback": list(
                        dict.fromkeys(
                            result.feedback
                            + ["Confidence is below the approval threshold; gather more evidence."]
                        )
                    ),
                    "reasoning": (result.reasoning or "")
                    + f" | confidence {result.confidence:.2f} < {self.approve_confidence:.2f}",
                    "suggested_tool_calls": _suggested_tools(
                        {
                            "should_retry_retrieval": result.should_retry_retrieval
                            or (context.capability or "").lower() in {"rag", "multi"},
                            "should_retry_sql": result.should_retry_sql
                            or (context.capability or "").lower() in {"sql", "planner", "multi"},
                            "should_retry_python": result.should_retry_python,
                            "should_ask_user": False,
                        }
                    ),
                }
            )

        _debug_print(result)
        return result

    def critique_response(
        self,
        question: str,
        response: Any,
        *,
        conversation_memory: Any = None,
        ambiguous: bool = False,
        retry_count: int = 0,
        previous_answer: str | None = None,
    ) -> ReflectionResult:
        """Critique a QueryResponse-like object."""
        answer = getattr(response, "answer", None)
        if answer is None and isinstance(response, dict):
            answer = response.get("answer", "")
        return self.critique(
            question,
            str(answer or ""),
            response=response,
            conversation_memory=conversation_memory,
            ambiguous=ambiguous,
            retry_count=retry_count,
            previous_answer=previous_answer,
        )

    def _merge_llm_critique(
        self,
        base: ReflectionResult,
        *,
        question: str,
        answer: str,
        previous_answer: str | None,
        context: AnswerContext,
    ) -> ReflectionResult:
        """Optionally refine reflection with an LLM critic (never writes user answers)."""
        try:
            llm = self._llm or get_llm()
            payload = {
                "question": question,
                "draft_answer": answer,
                "previous_answer": previous_answer,
                "sql": context.sql,
                "sql_row_count": len(context.sql_rows or []),
                "source_count": len(context.sources or []),
                "baseline_reflection": base.model_dump(exclude={"quality_report", "reasoning"}),
            }
            response = llm.invoke(
                [
                    SystemMessage(content=_llm_system_prompt()),
                    HumanMessage(content=json.dumps(payload, default=str)),
                ]
            )
            content = response.content if hasattr(response, "content") else response
            raw = content if isinstance(content, str) else str(content)
            parsed = _parse_json(raw)
            llm_result = ReflectionResult.model_validate(
                {
                    **base.model_dump(),
                    **parsed,
                    "retry_count": base.retry_count,
                    "max_retries": base.max_retries,
                    "quality_report": base.quality_report,
                }
            )
            # Conservative merge: never approve if baseline rejected; OR retry flags.
            approved = bool(llm_result.approved and base.approved)
            return llm_result.model_copy(
                update={
                    "approved": approved,
                    "should_retry_retrieval": llm_result.should_retry_retrieval or base.should_retry_retrieval,
                    "should_retry_sql": llm_result.should_retry_sql or base.should_retry_sql,
                    "should_retry_python": llm_result.should_retry_python or base.should_retry_python,
                    "should_ask_user": llm_result.should_ask_user or base.should_ask_user,
                    "missing_information": list(
                        dict.fromkeys(base.missing_information + llm_result.missing_information)
                    ),
                    "feedback": list(dict.fromkeys(base.feedback + llm_result.feedback)),
                    "suggested_tool_calls": list(
                        dict.fromkeys(base.suggested_tool_calls + llm_result.suggested_tool_calls)
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reflection LLM critique failed: %s", exc)
            return base.model_copy(
                update={"reasoning": (base.reasoning or "") + f" | llm_critique_failed: {exc}"}
            )


def _debug_print(result: ReflectionResult) -> None:
    lines = [
        "----------------------------------",
        "Reflection Result",
        f"Approved: {result.approved}",
        f"Confidence: {result.confidence}",
        f"Retry Decision: retrieval={result.should_retry_retrieval} "
        f"sql={result.should_retry_sql} python={result.should_retry_python} "
        f"ask_user={result.should_ask_user}",
        f"Missing Information: {result.missing_information}",
        f"Suggested Tool Calls: {result.suggested_tool_calls}",
        f"Feedback: {result.feedback}",
    ]
    if config.DEBUG:
        lines.append(f"Reasoning: {result.reasoning}")
    lines.append("----------------------------------")
    message = "\n".join(lines)
    if config.DEBUG:
        print(message)
    logger.info(message)
