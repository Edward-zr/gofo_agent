"""Plan execution helpers for LangGraph — no nested GOFOAgent.ask."""

from __future__ import annotations

from typing import Any

import config
from core.agent import (
    GOFOAgent,
    _active_sheet_from_contexts,
    _align_classification_with_route,
    _compact_file_context,
    format_agent_response,
)
from core.clarification_manager import build_clarification_response
from core.intent_classifier import IntentClassification, IntentType
from core.intent_router import RouteIntent
from core.logger import get_logger
from core.models import QueryResponse
from core.planner import ExecutionPlan, ExecutionStep
from core.request_verifier import VerificationReport
from core.task_decomposition import TaskSpec, task_spec_to_plan_inputs
from tools.context import build_context
from tools.conversation.resolver import resolution_from_dict
from tools.files.analysis_intent import WAIT_FOR_UPLOAD_REPLY
from tools.planner.intent_classifier import classify_intent as classify_business_intent

logger = get_logger("langgraph.execution")

_ROUTE_TO_INTENT: dict[str, IntentType] = {
    RouteIntent.SOP_QA: IntentType.SOP_QA,
    RouteIntent.SQL_ANALYTICS: IntentType.SQL_Query,
    RouteIntent.FOLLOW_UP: IntentType.Follow_Up,
    RouteIntent.ATTACHMENT_ANALYSIS: IntentType.Upload_File,
    RouteIntent.ATTACHMENT_VISUALIZATION: IntentType.Upload_File,
    RouteIntent.WAIT_FOR_UPLOAD: IntentType.Upload_File,
    RouteIntent.GENERAL_CHAT: IntentType.ChitChat,
    RouteIntent.OPENAI_FALLBACK: IntentType.General_Knowledge,
}


def ensure_classification(
    agent: GOFOAgent,
    *,
    question: str,
    route_decision: dict[str, Any],
    attachment_ids: list[str] | None = None,
) -> IntentClassification:
    """Classify then align with the LangGraph-owned route decision."""
    classification: IntentClassification | None = None
    if config.INTENT_CLASSIFIER_ENABLED:
        classification = agent.intent_classifier.classify(question, agent.memory)
    if classification is None:
        classification = IntentClassification(
            intent=IntentType.Unknown,
            confidence=0.5,
            requires_planner=True,
            reasoning="Fallback classification for LangGraph path",
        )

    route_intent = str(route_decision.get("intent") or "")
    mapped = _ROUTE_TO_INTENT.get(route_intent)
    if mapped is not None:
        # Prefer route for attachment/chat/wait; keep SOP_* subclass from classifier.
        sop_family = {
            IntentType.SOP_QA,
            IntentType.SOP_Summary,
            IntentType.SOP_Compare,
        }
        sql_family = {
            IntentType.SQL_Query,
            IntentType.SQL_Analysis,
            IntentType.Dashboard,
        }
        if mapped == IntentType.SOP_QA and classification.intent in sop_family:
            pass
        elif mapped == IntentType.SQL_Query and classification.intent in sql_family:
            pass
        elif classification.intent != mapped:
            classification = classification.model_copy(
                update={
                    "intent": mapped,
                    "requires_planner": True,
                    "requires_sql": mapped
                    in {IntentType.SQL_Query, IntentType.SQL_Analysis, IntentType.Dashboard},
                    "requires_rag": mapped
                    in {IntentType.SOP_QA, IntentType.SOP_Summary, IntentType.SOP_Compare},
                    "requires_memory": mapped
                    in {IntentType.Follow_Up, IntentType.Upload_File, IntentType.Explain_Result},
                    "reasoning": (classification.reasoning or "")
                    + f" | langgraph route→{mapped.value}",
                }
            )

    return _align_classification_with_route(
        classification,
        route_decision,
        has_new_upload=bool(attachment_ids),
    )


def enrich_plan_with_task_spec(plan: ExecutionPlan, task_spec: TaskSpec) -> ExecutionPlan:
    """Merge TaskSpec constraints into every step's inputs."""
    extras = {
        key: value
        for key, value in task_spec_to_plan_inputs(task_spec).items()
        if value is not None and value != {} and value != []
    }
    if not extras:
        return plan
    steps: list[ExecutionStep] = []
    for step in plan.steps:
        merged = dict(step.inputs or {})
        for key, value in extras.items():
            if key not in merged or merged.get(key) in (None, "", {}, []):
                merged[key] = value
        # Always prefer TaskSpec chart_type / limit when present.
        if task_spec.chart_type:
            merged["chart_type"] = task_spec.chart_type
        if task_spec.limit is not None:
            merged["limit"] = task_spec.limit
        steps.append(step.model_copy(update={"inputs": merged}))
    return plan.model_copy(update={"steps": steps})


def build_wait_for_upload_response(*, question: str) -> QueryResponse:
    return QueryResponse(
        question=question,
        answer=WAIT_FOR_UPLOAD_REPLY,
        sources=[],
        capability="wait_for_upload",
        planning_capability="planner",
        planning_intent="WAIT_FOR_UPLOAD",
        original_question=question,
        resolved_question=question,
    )


def commit_and_format(
    agent: GOFOAgent,
    *,
    question: str,
    resolved_question: str,
    response: QueryResponse,
    route_decision: dict[str, Any],
    classification: IntentClassification | None,
    conversation_resolution: dict[str, Any] | None = None,
    repair_detected: bool = False,
    repair_type: str | None = None,
    changed_dimension: str | None = None,
    attachment_ids: list[str] | None = None,
    semantic: Any | None = None,
    business_intent: Any | None = None,
    context: Any | None = None,
    sql_cache_hit: bool = False,
    verification_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update memory/state and return the API-shaped response dict."""
    attachment_ids = attachment_ids or []
    previous_state = agent.state.snapshot()
    previous_semantic_state = agent.conversation.snapshot()

    resolution = resolution_from_dict(
        conversation_resolution,
        question=question,
        resolved_question=resolved_question or question,
        repair_detected=repair_detected,
        repair_type=repair_type,
        changed_dimension=changed_dimension,
    )

    if semantic is None:
        from tools.orchestration.semantic_analyzer import _heuristic_fallback

        semantic = _heuristic_fallback(
            question,
            resolved_question=resolved_question,
            has_attachments=bool(route_decision.get("use_attachments")),
        )
    if business_intent is None:
        business_intent = classify_business_intent(
            question,
            resolved_question=resolved_question,
            conversation_state=previous_semantic_state,
            repair_detected=resolution.repair_detected,
            intent_hint=route_decision.get("intent") or resolution.intent_hint,
        )
    if context is None:
        context = build_context(resolution, business_intent, previous_semantic_state)

    attachment_contexts, file_context = agent.attachment_memory.resolve_for_question(
        question,
        attachment_ids=attachment_ids or None,
        use_attachments=bool(route_decision.get("use_attachments")),
    )
    last_analysis = agent.attachment_memory.last_analysis or {}
    analysis_summary = dict(response.file_context_summary or {})

    response.attachment_ids = [ctx.attachment_id for ctx in attachment_contexts] or list(
        response.attachment_ids or []
    )
    response.attachment_filenames = [ctx.filename for ctx in attachment_contexts] or list(
        response.attachment_filenames or []
    )
    response.data_sources = route_decision.get("data_sources") or list(
        response.data_sources or []
    )
    compact_file_context = _compact_file_context(file_context)
    active_sheet = _active_sheet_from_contexts(attachment_contexts) or analysis_summary.get(
        "active_sheet"
    )
    response.file_context_summary = {
        **compact_file_context,
        **{key: value for key, value in analysis_summary.items() if value is not None},
    }
    if active_sheet:
        response.file_context_summary["active_sheet"] = active_sheet

    if classification is not None and not response.intent_classification:
        response.intent_classification = classification.model_dump()

    agent.memory.add_turn(
        user_question=question,
        resolved_question=resolved_question,
        response=response,
    )
    agent.state.update(
        user_question=question,
        resolved_question=resolved_question,
        response=response,
        route_decision=route_decision,
    )
    new_state = agent.state.snapshot()
    new_semantic_state = agent.conversation.update(
        original_question=question,
        resolved_question=resolved_question,
        response=response,
        intent=(
            classification.intent.value
            if classification is not None
            else business_intent.value
        ),
        inherited_context=context.inherited_context,
        cache_key=context.cache_key,
    )

    response.original_question = question
    response.resolved_question = resolved_question
    response.repair_detected = resolution.repair_detected
    response.repair_type = resolution.repair_type
    response.changed_dimension = resolution.changed_dimension
    response.classifier_intent = business_intent.value
    response.inherited_context = context.inherited_context
    response.semantic_domain = semantic.domain
    response.semantic_sub_intent = semantic.sub_intent
    response.response_mode = semantic.response_mode
    response.semantic_reasoning = semantic.reasoning
    response.sql_cache_hit = sql_cache_hit
    response.memory_updated = True

    agent.attachment_memory.update_analysis(
        {
            "file_context": file_context,
            "data_sources": response.data_sources,
            "answer": response.answer,
            "route": route_decision,
            "active_filter": (response.analysis_filters or {})
            or analysis_summary.get("active_filter")
            or {},
            "last_entity": analysis_summary.get("last_entity")
            or last_analysis.get("last_entity"),
            "charts": response.charts or [],
            "intent_classification": response.intent_classification,
            "execution_plan": response.execution_plan,
        }
    )
    response.memory_history_count = len(agent.memory.get_recent_history())
    response.memory_current_state = agent.memory.get_current_state()
    response.memory_current_state["conversation_state"] = new_state
    response.memory_current_state["previous_conversation_state"] = previous_state
    response.memory_current_state["semantic_conversation"] = new_semantic_state
    response.memory_current_state["previous_semantic_conversation"] = previous_semantic_state
    response.memory_current_state["attachment_memory"] = agent.attachment_memory.snapshot()
    response.memory_current_state["route_decision"] = route_decision
    if classification is not None:
        response.memory_current_state["intent_classification"] = classification.model_dump()
    if response.execution_plan:
        response.memory_current_state["execution_plan"] = response.execution_plan
    if verification_report:
        response.memory_current_state["verification_report"] = verification_report
    if agent.pending_clarification is not None:
        response.memory_current_state["pending_clarification"] = (
            agent.pending_clarification.model_dump()
        )
    response.last_result_context = agent.memory.get_current_state().get("last_result_context")

    api = format_agent_response(response)
    if verification_report:
        analysis = dict(api.get("analysis") or {})
        analysis["verification_report"] = verification_report
        api["analysis"] = analysis
    return api


def run_planner(
    agent: GOFOAgent,
    *,
    question: str,
    resolved_question: str,
    classification: IntentClassification,
    route_decision: dict[str, Any],
    task_spec: TaskSpec,
    attachment_ids: list[str] | None = None,
) -> ExecutionPlan:
    """Always call Planner.plan on the LangGraph path; embed TaskSpec inputs."""
    attachment_contexts, _file_context = agent.attachment_memory.resolve_for_question(
        question,
        attachment_ids=attachment_ids or None,
        use_attachments=bool(route_decision.get("use_attachments")),
    )
    plan = agent.planner.plan(
        resolved_question,
        classification,
        agent.memory,
        attachment_context={
            "attachment_ids": attachment_ids or [],
            "filenames": [ctx.filename for ctx in attachment_contexts],
            "use_attachments": bool(route_decision.get("use_attachments")),
            "chart_type": task_spec.chart_type,
        },
    )
    return enrich_plan_with_task_spec(plan, task_spec)


def maybe_clarify(
    agent: GOFOAgent,
    *,
    question: str,
    resolved_question: str,
    classification: IntentClassification,
    plan: ExecutionPlan,
    route_decision: dict[str, Any],
    attachment_ids: list[str] | None = None,
) -> tuple[bool, QueryResponse | None, ExecutionPlan, str | None]:
    """Run ClarificationManager.

    Returns ``(needs_clarification, response_or_none, plan, resumed_question_or_none)``.
    """
    if not config.CLARIFICATION_MANAGER_ENABLED:
        return False, None, plan, None

    # Resume path: user answered a pending clarification.
    if agent.pending_clarification is not None and not attachment_ids:
        resume = agent.clarification_manager.apply_user_response(
            agent.pending_clarification,
            question,
        )
        if resume.resumed_question:
            slots = dict(resume.filled_slots or {})
            if slots.get("metric"):
                agent.memory.global_state["current_metric"] = slots["metric"]
            if slots.get("date_range"):
                agent.memory.global_state["date_range"] = slots["date_range"]
            agent.pending_clarification = None
            logger.info("Resumed after clarification: %s", resume.resumed_question)
            return False, None, plan, resume.resumed_question
        # New domain/SOP ask — clear pending and continue with fresh planning.
        agent.pending_clarification = None
        logger.info("Cleared pending clarification for new LangGraph question: %s", question)

    attachment_contexts, _ = agent.attachment_memory.resolve_for_question(
        question,
        attachment_ids=attachment_ids or None,
        use_attachments=bool(route_decision.get("use_attachments")),
    )
    clarify = agent.clarification_manager.evaluate(
        question,
        resolved_question=resolved_question,
        classification=classification,
        plan=plan,
        conversation_memory=agent.memory,
        attachment_context={
            "attachment_ids": attachment_ids or [],
            "filenames": [ctx.filename for ctx in attachment_contexts],
            "use_attachments": bool(route_decision.get("use_attachments")),
        },
    )
    if clarify.needs_clarification and clarify.pending is not None:
        agent.pending_clarification = clarify.pending
        response = build_clarification_response(
            clarify,
            question=question,
            classification=classification,
        )
        response.execution_plan = plan.model_dump()
        response.resolved_question = resolved_question
        return True, response, plan, None
    return False, None, plan, None


def execute_plan(
    agent: GOFOAgent,
    *,
    plan: ExecutionPlan,
    question: str,
    resolved_question: str,
    route_decision: dict[str, Any],
    classification: IntentClassification,
    task_spec: TaskSpec | None = None,
    attachment_ids: list[str] | None = None,
) -> QueryResponse:
    """Execute via PlanExecutor / SupervisorAgent — never nested ask()."""
    if task_spec is not None:
        plan = enrich_plan_with_task_spec(plan, task_spec)

    attachment_contexts, file_context = agent.attachment_memory.resolve_for_question(
        question,
        attachment_ids=attachment_ids or None,
        use_attachments=bool(route_decision.get("use_attachments")),
    )
    last_analysis = agent.attachment_memory.last_analysis or {}
    result_context = agent.memory.get_current_state().get("last_result_context")

    execution = agent.plan_executor.execute(
        plan,
        question=question,
        resolved_question=resolved_question,
        conversation_memory=agent.memory,
        attachment_contexts=attachment_contexts,
        file_context=file_context,
        attachment_ids=attachment_ids or [],
        stored_attachments=agent.attachment_memory.get_active_stored()
        or list(agent.attachment_memory.stored_attachments.values()),
        previous_filter=agent.attachment_memory.active_filters or None,
        last_entity=(last_analysis.get("last_entity") or None),
        result_context=result_context,
    )
    response = execution.response
    response.intent_classification = classification.model_dump()
    response.execution_plan = plan.model_dump()
    response.step_results_summary = execution.response.step_results_summary

    if response.capability == "sql" and response.sql_rows:
        from tools.analyzer.business_reasoner import analyze_business_response

        response = analyze_business_response(response, resolved_question)
        response.intent_classification = classification.model_dump()
        response.execution_plan = plan.model_dump()

    return response


def plan_retry_from_verification(
    agent: GOFOAgent,
    *,
    question: str,
    classification: IntentClassification,
    verification: VerificationReport,
    previous_plan: ExecutionPlan | dict[str, Any] | None,
    previous_answer: str | None,
    task_spec: TaskSpec | None = None,
) -> ExecutionPlan:
    """Build a retry plan from VerificationReport (Planner.plan_retry compatible)."""
    plan = agent.planner.plan_retry(
        question,
        classification,
        verification,
        previous_plan=previous_plan,
        previous_answer=previous_answer,
        conversation_memory=agent.memory,
    )
    if task_spec is not None:
        plan = enrich_plan_with_task_spec(plan, task_spec)
    return plan
