"""Clarification Manager — ask for missing business parameters before tools run.

Sits between Planner and Tool Orchestrator. Never guesses missing metrics,
entities, or time ranges. Supports multiple-choice options and resumes the
original plan after the user answers.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

import config
from core.intent_classifier import IntentClassification, IntentType
from core.logger import get_logger
from core.planner import ExecutionPlan

logger = get_logger("clarification_manager")


class ClarificationOption(BaseModel):
    id: str
    label: str


class ClarificationRequest(BaseModel):
    """Payload shown to the user when clarification is required."""

    original_question: str
    missing_fields: list[str] = Field(default_factory=list)
    pending_question: str
    options: list[ClarificationOption] = Field(default_factory=list)
    reason: str = ""
    ambiguity_type: str = ""
    confidence: float = 0.0


class PendingClarification(BaseModel):
    """Session-scoped pending clarification stored on GOFOAgent / AgentState."""

    original_question: str
    missing_fields: list[str] = Field(default_factory=list)
    pending_question: str
    user_response: str | None = None
    options: list[ClarificationOption] = Field(default_factory=list)
    reason: str = ""
    ambiguity_type: str = ""
    classification: dict[str, Any] | None = None
    draft_plan: dict[str, Any] | None = None
    attachment_context: dict[str, Any] = Field(default_factory=dict)
    resolved_base: str = ""
    filled_slots: dict[str, str] = Field(default_factory=dict)


class ClarificationDecision(BaseModel):
    needs_clarification: bool = False
    request: ClarificationRequest | None = None
    pending: PendingClarification | None = None
    resumed_question: str | None = None
    skip_reason: str | None = None
    filled_slots: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ambiguity patterns
# ---------------------------------------------------------------------------

_METRIC_OPTIONS = [
    ClarificationOption(id="completed_pickups", label="Completed pickups"),
    ClarificationOption(id="pickup_rate", label="Pickup rate"),
    ClarificationOption(id="on_time_rate", label="On-time rate"),
    ClarificationOption(id="customer_rating", label="Customer rating"),
]

_TIME_COMPARE_OPTIONS = [
    ClarificationOption(id="today_vs_yesterday", label="Today vs Yesterday"),
    ClarificationOption(id="this_week_vs_last_week", label="This week vs Last week"),
    ClarificationOption(id="this_month_vs_last_month", label="This month vs Last month"),
    ClarificationOption(id="custom_range", label="Custom range"),
]

_ENTITY_OPTIONS = [
    ClarificationOption(id="by_hub", label="By hub"),
    ClarificationOption(id="by_driver", label="By driver"),
    ClarificationOption(id="by_region", label="By region"),
    ClarificationOption(id="overall", label="Overall network"),
]

_CUSTOMER_METRIC_OPTIONS = [
    ClarificationOption(id="volume", label="Volume (packages)"),
    ClarificationOption(id="pickup_count", label="Pickup count"),
    ClarificationOption(id="revenue", label="Revenue"),
    ClarificationOption(id="failure_rate", label="Failure rate"),
]

_TIME_RANGE_OPTIONS = [
    ClarificationOption(id="today", label="Today"),
    ClarificationOption(id="this_week", label="This week"),
    ClarificationOption(id="this_month", label="This month"),
    ClarificationOption(id="all_historical", label="All historical data"),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _memory_state(conversation_memory: Any) -> dict[str, Any]:
    if conversation_memory is None:
        return {}
    if hasattr(conversation_memory, "get_current_state"):
        return dict(conversation_memory.get_current_state() or {})
    if isinstance(conversation_memory, dict):
        return dict(conversation_memory)
    return {}


def _has_metric(text: str) -> bool:
    return bool(
        re.search(
            r"\b(rate|volume|count|packages?|completion|on[- ]?time|rating|"
            r"failed|delayed|success|kpi|revenue)\b",
            text,
        )
    )


def _has_time(text: str) -> bool:
    return bool(
        re.search(
            r"\b(today|yesterday|this week|last week|this month|last month|"
            r"\d{4}-\d{2}-\d{2}|vs\.?|versus|compared to|between)\b",
            text,
        )
    )


def _has_entity_scope(text: str) -> bool:
    return bool(
        re.search(
            r"\b(hub|driver|region|customer|city|warehouse|station|overall|network)\b",
            text,
        )
    )


def _detect_ambiguity(
    question: str,
    *,
    memory: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[ClarificationOption], str] | None:
    """Return (ambiguity_type, missing_fields, options, reason) or None.

    Detects gaps from the question text only. Conversation memory fills are
    applied later in ``ClarificationManager.evaluate`` so skip reasons stay
    explicit for debug mode.
    """
    del memory  # reserved for future context-aware detectors
    text = _normalize(question)

    # 1) Best/worst/top driver without metric
    if re.search(r"\b(best|worst|top|highest|lowest)\b.*\bdrivers?\b", text) or re.search(
        r"\bdrivers?\b.*\b(best|worst|top|highest|lowest)\b", text
    ):
        if not _has_metric(text):
            return (
                "driver_metric",
                ["metric"],
                list(_METRIC_OPTIONS),
                "Best/worst driver is ambiguous without a ranking metric.",
            )

    # 2) Top customers without metric
    if re.search(r"\b(top|best|worst)\b.*\bcustomers?\b", text) or re.search(
        r"\bcustomers?\b.*\b(top|best|worst)\b", text
    ):
        if not _has_metric(text):
            return (
                "customer_metric",
                ["metric"],
                list(_CUSTOMER_METRIC_OPTIONS),
                "Top customers requires a ranking metric (volume, pickup count, etc.).",
            )

    # 3) Bare "performance" / "show performance"
    if re.search(r"\b(show|check|analyze|review)?\s*performance\b", text) and not _has_entity_scope(
        text
    ):
        return (
            "performance_scope",
            ["entity_scope"],
            list(_ENTITY_OPTIONS),
            "Performance is ambiguous without hub, driver, or region scope.",
        )

    # 4) Compare pickup rates without periods
    if re.search(r"\bcompare\b", text) and re.search(r"\b(pickup\s+rates?|rates?|performance)\b", text):
        if not _has_time(text):
            # "compare X and Y" with two named entities is OK
            if not re.search(r"\b(and|vs\.?|versus)\b.+\b(hub|driver|city|region)\b", text):
                return (
                    "compare_periods",
                    ["time_period"],
                    list(_TIME_COMPARE_OPTIONS),
                    "Comparison needs explicit time periods or peer entities.",
                )

    # 5) Ranking with metric but no time — clarify when the question is short
    if (
        re.search(r"\b(best|worst|top|rank)\b", text)
        and re.search(r"\b(driver|hub|customer)\b", text)
        and not _has_time(text)
        and _has_metric(text)
    ):
        if len(text.split()) <= 8:
            return (
                "time_range",
                ["date_range"],
                list(_TIME_RANGE_OPTIONS),
                "Ranking questions are more accurate with an explicit time range.",
            )

    return None


def _format_pending_question(
    prompt: str,
    options: list[ClarificationOption],
) -> str:
    lines = [prompt.strip(), ""]
    for index, option in enumerate(options, start=1):
        lines.append(f"{index}. {option.label}")
    lines.append("")
    lines.append("Reply with an option number or label.")
    return "\n".join(lines)


def _match_option(user_text: str, options: list[ClarificationOption]) -> ClarificationOption | None:
    text = _normalize(user_text)
    if not text:
        return None
    # Numeric choice
    if text.isdigit():
        index = int(text) - 1
        if 0 <= index < len(options):
            return options[index]
    # "1." / "option 2"
    match = re.match(r"^(?:option\s*)?(\d+)\b", text)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(options):
            return options[index]
    # Label / id match
    for option in options:
        if text == _normalize(option.label) or text == _normalize(option.id):
            return option
        if _normalize(option.label) in text or _normalize(option.id).replace("_", " ") in text:
            return option
    return None


def _enrich_question(
    original: str,
    *,
    ambiguity_type: str,
    option: ClarificationOption | None,
    user_response: str,
) -> str:
    label = option.label if option else user_response.strip()
    option_id = option.id if option else _normalize(user_response)

    if ambiguity_type == "driver_metric":
        return f"{original.rstrip('.!?')} by {label}."
    if ambiguity_type == "customer_metric":
        return f"{original.rstrip('.!?')} by {label}."
    if ambiguity_type == "performance_scope":
        if option_id == "overall":
            return f"Show overall network performance."
        if option_id == "by_hub":
            return f"Show performance by hub."
        if option_id == "by_driver":
            return f"Show performance by driver."
        if option_id == "by_region":
            return f"Show performance by region."
        return f"{original.rstrip('.!?')} ({label})."
    if ambiguity_type == "compare_periods":
        mapping = {
            "today_vs_yesterday": "today vs yesterday",
            "this_week_vs_last_week": "this week vs last week",
            "this_month_vs_last_month": "this month vs last month",
        }
        period = mapping.get(option_id, label)
        if "compare" in _normalize(original):
            return f"{original.rstrip('.!?')} for {period}."
        return f"Compare pickup rates for {period}."
    if ambiguity_type == "time_range":
        return f"{original.rstrip('.!?')} for {label.lower()}."
    return f"{original.rstrip('.!?')} ({label})."


def _debug(message: str) -> None:
    if config.DEBUG or getattr(config, "CLARIFICATION_DEBUG", False):
        print("----------------------------------")
        print("ClarificationManager")
        print(message)
        print("----------------------------------")
    logger.info(message)


class ClarificationManager:
    """Detect missing parameters and resume plans after user answers."""

    def evaluate(
        self,
        question: str,
        *,
        resolved_question: str | None = None,
        classification: IntentClassification | None = None,
        plan: ExecutionPlan | None = None,
        conversation_memory: Any = None,
        attachment_context: dict[str, Any] | None = None,
        confidence_floor: float = 0.85,
    ) -> ClarificationDecision:
        """Decide whether to ask before Tool Orchestration."""
        resolved = (resolved_question or question or "").strip()
        memory = _memory_state(conversation_memory)

        # High-confidence + no ambiguity pattern + plan ready → skip
        plan_wants = bool(plan and plan.requires_clarification)
        detected = _detect_ambiguity(resolved, memory=memory)

        if not plan_wants and detected is None:
            return ClarificationDecision(
                needs_clarification=False,
                skip_reason="No missing business parameters detected.",
            )

        # Skip if memory already fills the gap for detected ambiguity
        if detected is not None:
            ambiguity_type, missing, options, reason = detected
            if "metric" in missing and memory.get("current_metric"):
                return ClarificationDecision(
                    needs_clarification=False,
                    skip_reason="Conversation memory already provides metric.",
                    filled_slots={"metric": str(memory.get("current_metric"))},
                )
            if "date_range" in missing and memory.get("date_range"):
                return ClarificationDecision(
                    needs_clarification=False,
                    skip_reason="Conversation memory already provides date_range.",
                    filled_slots={"date_range": str(memory.get("date_range"))},
                )
            if "time_period" in missing and memory.get("date_range"):
                return ClarificationDecision(
                    needs_clarification=False,
                    skip_reason="Conversation memory already provides comparison periods.",
                    filled_slots={"date_range": str(memory.get("date_range"))},
                )
            if "entity_scope" in missing:
                entities = memory.get("active_entities") or {}
                if entities.get("hub") or entities.get("driver") or entities.get("region"):
                    return ClarificationDecision(
                        needs_clarification=False,
                        skip_reason="Conversation memory already provides entity scope.",
                        filled_slots={k: str(v) for k, v in entities.items() if v},
                    )

            # Skip clarification when confidence is high AND question already rich
            conf = classification.confidence if classification else 0.0
            if (
                conf >= confidence_floor
                and classification is not None
                and not classification.requires_clarification
                and _has_metric(resolved)
                and (_has_time(resolved) or memory.get("date_range"))
            ):
                return ClarificationDecision(
                    needs_clarification=False,
                    skip_reason="High confidence with sufficient metric/time context.",
                )

            prompts = {
                "driver_metric": "Best/worst driver — best by what metric?",
                "customer_metric": "Top customers — top by which metric?",
                "performance_scope": "Show performance — which scope?",
                "compare_periods": "Compare pickup rates — which time periods?",
                "time_range": "Which time range should I use?",
            }
            pending_question = _format_pending_question(
                prompts.get(ambiguity_type, "I need a bit more detail:"),
                options,
            )
            request = ClarificationRequest(
                original_question=question,
                missing_fields=missing,
                pending_question=pending_question,
                options=options,
                reason=reason,
                ambiguity_type=ambiguity_type,
                confidence=classification.confidence if classification else 0.0,
            )
            pending = PendingClarification(
                original_question=question,
                missing_fields=missing,
                pending_question=pending_question,
                options=options,
                reason=reason,
                ambiguity_type=ambiguity_type,
                classification=classification.model_dump() if classification else None,
                draft_plan=plan.model_dump() if plan else None,
                attachment_context=attachment_context or {},
                resolved_base=resolved,
            )
            _debug(
                f"needs_clarification=True\n"
                f"  missing={missing}\n"
                f"  reason={reason}\n"
                f"  type={ambiguity_type}"
            )
            return ClarificationDecision(
                needs_clarification=True,
                request=request,
                pending=pending,
            )

        # Planner-requested clarification (time range default)
        if plan_wants:
            options = list(_TIME_RANGE_OPTIONS)
            pending_question = plan.clarification_question or _format_pending_question(
                "I need a bit more detail to plan this correctly.",
                options,
            )
            # If planner text already has bullets, keep it; else add options
            if plan.clarification_question and "1." not in plan.clarification_question:
                # Keep planner prose; append numeric options when useful
                if "today" in (plan.clarification_question or "").lower():
                    pending_question = plan.clarification_question
                    options = list(_TIME_RANGE_OPTIONS)
            request = ClarificationRequest(
                original_question=question,
                missing_fields=["date_range"],
                pending_question=pending_question,
                options=options,
                reason=plan.reasoning or "Planner requested clarification.",
                ambiguity_type="planner",
                confidence=plan.confidence if plan else 0.0,
            )
            pending = PendingClarification(
                original_question=question,
                missing_fields=["date_range"],
                pending_question=pending_question,
                options=options,
                reason=request.reason,
                ambiguity_type="planner",
                classification=classification.model_dump() if classification else None,
                draft_plan=plan.model_dump() if plan else None,
                attachment_context=attachment_context or {},
                resolved_base=resolved,
            )
            _debug(
                f"needs_clarification=True (planner)\n"
                f"  missing=['date_range']\n"
                f"  reason={request.reason}"
            )
            return ClarificationDecision(
                needs_clarification=True,
                request=request,
                pending=pending,
            )

        return ClarificationDecision(needs_clarification=False, skip_reason="No action.")

    def apply_user_response(
        self,
        pending: PendingClarification,
        user_response: str,
    ) -> ClarificationDecision:
        """Merge the user's clarification answer into a resumed question."""
        user_response = (user_response or "").strip()
        option = _match_option(user_response, pending.options)
        resumed = _enrich_question(
            pending.original_question or pending.resolved_base,
            ambiguity_type=pending.ambiguity_type,
            option=option,
            user_response=user_response,
        )
        slots = dict(pending.filled_slots)
        if option:
            if "metric" in pending.missing_fields:
                slots["metric"] = option.label
            if "date_range" in pending.missing_fields or "time_period" in pending.missing_fields:
                slots["date_range"] = option.label
            if "entity_scope" in pending.missing_fields:
                slots["entity_scope"] = option.label
        else:
            slots["free_text"] = user_response

        updated = pending.model_copy(
            update={
                "user_response": user_response,
                "filled_slots": slots,
            }
        )
        _debug(
            f"resumed_execution\n"
            f"  original={pending.original_question}\n"
            f"  user_response={user_response}\n"
            f"  resumed_question={resumed}\n"
            f"  slots={slots}\n"
            f"  draft_plan_goal={(pending.draft_plan or {}).get('goal')}"
        )
        return ClarificationDecision(
            needs_clarification=False,
            pending=updated,
            resumed_question=resumed,
            filled_slots=slots,
            skip_reason="User clarification applied; resume plan.",
        )


def build_clarification_response(
    decision: ClarificationDecision,
    *,
    question: str,
    classification: IntentClassification | None = None,
) -> "QueryResponse":
    """Build a QueryResponse that asks for clarification without running tools."""
    from core.models import QueryResponse

    request = decision.request
    pending = decision.pending
    answer = (request.pending_question if request else None) or (
        "I need a bit more detail before I run analytics."
    )
    agent_state = {
        "original_question": pending.original_question if pending else question,
        "missing_fields": list(pending.missing_fields) if pending else [],
        "pending_question": pending.pending_question if pending else answer,
        "user_response": pending.user_response if pending else None,
        "clarification_options": [opt.model_dump() for opt in (pending.options if pending else [])],
        "clarification_reason": pending.reason if pending else "",
    }
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="clarification",
        planning_intent="clarification",
        plan_reason=(request.reason if request else "clarification_required"),
        intent_classification=classification.model_dump() if classification else None,
        execution_plan=pending.draft_plan if pending else None,
        suggested_next_steps=[opt.label for opt in (request.options if request else [])],
        agent_state=agent_state,
        requires_clarification=True,
        clarification_question=answer,
        clarification_options=[opt.model_dump() for opt in (request.options if request else [])],
        missing_fields=list(pending.missing_fields) if pending else [],
    )
