"""Route queries to the appropriate GOFO capability handler."""

from __future__ import annotations

import os

from core.models import QueryRequest, QueryResponse
from tools.analysis.anomaly import answer_anomaly_question, detect_anomalies
from tools.analysis.context import (
    analyze_business_context,
    is_anomaly_question,
    is_operations_overview,
    is_why_question,
)
from tools.analysis.drilldown import run_drilldown
from tools.analysis.kpi import answer_operations_overview
from tools.analysis.recommender import recommend
from tools.analysis.root_cause import analyze_root_cause
from tools.memory.findings import remember_finding
from tools.memory.learning import detect_repeated_patterns
from tools.memory.long_memory import LongTermMemory
from tools.memory.result_analyzer import analyze_previous_result
from tools.orchestration.models import SemanticAnalysis
from tools.orchestration.next_steps import generate_next_steps
from tools.planner import PlanningDecision, plan
from tools.rag.service import answer as rag_answer
from tools.sql.service import answer as sql_answer
from tools.synthesizer import synthesize


UNKNOWN_ANSWER = "I don't know how to route this request."
CONVERSATION_FALLBACK = (
    "I can help analyze GOFO operations, investigate performance issues, "
    "answer SOP questions, and review uploaded reports. "
    "Try asking about today's pickups, hub rankings, root causes, or the pickup SOP."
)


def route(request: QueryRequest) -> QueryResponse:
    """
    Route a query through the natural-language planner and selected tools.

    The planner only decides capabilities; this router executes the selected
    SQL/RAG tools and merges outputs for multi-tool requests.
    """
    question = request.question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if request.result_context:
        return _memory_analysis_response(question, request.result_context)

    business_context = analyze_business_context(question)
    long_memory = LongTermMemory()
    historical_context = _retrieve_long_memory(long_memory, question, business_context)
    if is_operations_overview(question):
        return _finalize_response(
            _with_business_context(answer_operations_overview(question), business_context),
            question,
            business_context,
            historical_context,
            long_memory,
        )
    if is_anomaly_question(question):
        return _finalize_response(
            _with_business_context(answer_anomaly_question(question), business_context),
            question,
            business_context,
            historical_context,
            long_memory,
        )
    if is_why_question(question):
        return _finalize_response(
            _with_business_context(run_drilldown(question), business_context),
            question,
            business_context,
            historical_context,
            long_memory,
        )

    semantic = _load_semantic_analysis(request.semantic_context)
    if semantic and semantic.should_answer_directly():
        return _finalize_response(
            _conversation_response(question, semantic),
            question,
            business_context,
            historical_context,
            long_memory,
        )

    decision = _resolve_planning_decision(question, semantic)

    if decision.capability == "unknown":
        if semantic and semantic.direct_reply:
            return _finalize_response(
                _conversation_response(question, semantic),
                question,
                business_context,
                historical_context,
                long_memory,
            )
        return _unknown_response(question, decision, semantic)

    sql_result: QueryResponse | None = None
    rag_result: QueryResponse | None = None
    execution_order: list[str] = []

    if decision.requires_sql:
        sql_result = sql_answer(request)
        execution_order.append("sql")
    if decision.requires_rag:
        rag_result = rag_answer(request)
        execution_order.append("rag")

    if decision.capability == "multi":
        capability = "multi"
        final_answer = synthesize(question, sql_result, rag_result)
    elif decision.capability == "sql":
        capability = "sql"
        final_answer = sql_result.answer if sql_result else None
    elif decision.capability == "rag":
        capability = "rag"
        final_answer = rag_result.answer if rag_result else None
    else:
        return _unknown_response(question, decision, semantic)

    sources = rag_result.sources if rag_result else []
    rewritten_question = None
    if rag_result and rag_result.rewritten_question:
        rewritten_question = rag_result.rewritten_question
    elif sql_result and sql_result.rewritten_question:
        rewritten_question = sql_result.rewritten_question

    response = QueryResponse(
        question=question,
        answer=final_answer,
        sources=sources,
        capability=capability,
        rewritten_question=rewritten_question,
        latest_business_date=sql_result.latest_business_date if sql_result else None,
        generated_sql=sql_result.generated_sql if sql_result else None,
        sql_rows=sql_result.sql_rows if sql_result else None,
        needs_sql=decision.requires_sql,
        needs_rag=decision.requires_rag,
        plan_reason=decision.reasoning,
        sql_output=sql_result.answer if sql_result else None,
        rag_output=rag_result.answer if rag_result else None,
        execution_order=execution_order,
        planning_capability=decision.capability,
        planning_intent=decision.intent,
        planning_confidence=decision.confidence,
        planning_reasoning=decision.reasoning,
        planning_entities=decision.entities,
    )
    response = _with_business_context(response, business_context)
    if sql_result and sql_result.sql_rows:
        _attach_operational_analysis(response, question, sql_result.sql_rows)
    response = _attach_semantic_metadata(response, semantic)
    response = _attach_next_steps(response, question=question, semantic=semantic)
    return _finalize_response(response, question, business_context, historical_context, long_memory)


def _unknown_response(
    question: str,
    decision: PlanningDecision,
    semantic: SemanticAnalysis | None = None,
) -> QueryResponse:
    """Return a helpful response when no GOFO capability is appropriate."""
    answer = semantic.direct_reply if semantic and semantic.direct_reply else CONVERSATION_FALLBACK
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="conversation" if semantic and semantic.direct_reply else "unknown",
        needs_sql=decision.requires_sql,
        needs_rag=decision.requires_rag,
        plan_reason=decision.reasoning,
        execution_order=[],
        planning_capability=decision.capability,
        planning_intent=decision.intent,
        planning_confidence=decision.confidence,
        planning_reasoning=decision.reasoning,
        planning_entities=decision.entities,
        semantic_domain=semantic.domain if semantic else None,
        semantic_sub_intent=semantic.sub_intent if semantic else None,
        response_mode=semantic.response_mode if semantic else None,
        semantic_reasoning=semantic.reasoning if semantic else decision.reasoning,
        suggested_next_steps=_default_next_steps(),
    )


def _conversation_response(question: str, semantic: SemanticAnalysis) -> QueryResponse:
    """Return a conversational analyst response without SQL/RAG execution."""
    return QueryResponse(
        question=question,
        answer=semantic.direct_reply or CONVERSATION_FALLBACK,
        sources=[],
        capability="conversation",
        execution_order=["conversation"],
        planning_capability="conversation",
        planning_intent=semantic.planner_intent,
        planning_confidence=semantic.confidence,
        planning_reasoning=semantic.reasoning,
        planning_entities=semantic.entities,
        semantic_domain=semantic.domain,
        semantic_sub_intent=semantic.sub_intent,
        response_mode=semantic.response_mode,
        semantic_reasoning=semantic.reasoning,
        business_metric=semantic.metric,
        analysis_dimension=semantic.dimension,
        suggested_next_steps=_default_next_steps(),
    )


def _resolve_planning_decision(
    question: str,
    semantic: SemanticAnalysis | None,
) -> PlanningDecision:
    """Choose a planning decision using semantic analysis with LLM fallback."""
    if semantic and semantic.is_actionable() and semantic.confidence >= 0.55:
        return semantic.to_planning_decision()
    llm_decision = plan(question)
    if llm_decision.capability != "unknown":
        return llm_decision
    if semantic and semantic.is_actionable():
        return semantic.to_planning_decision()
    return llm_decision


def _load_semantic_analysis(payload: dict | None) -> SemanticAnalysis | None:
    if not payload:
        return None
    return SemanticAnalysis.model_validate(payload)


def _attach_semantic_metadata(
    response: QueryResponse,
    semantic: SemanticAnalysis | None,
) -> QueryResponse:
    if not semantic:
        return response
    response.semantic_domain = semantic.domain
    response.semantic_sub_intent = semantic.sub_intent
    response.response_mode = semantic.response_mode
    response.semantic_reasoning = semantic.reasoning
    response.business_metric = response.business_metric or semantic.metric
    response.analysis_dimension = response.analysis_dimension or semantic.dimension
    return response


def _attach_next_steps(
    response: QueryResponse,
    *,
    question: str,
    semantic: SemanticAnalysis | None,
) -> QueryResponse:
    if response.suggested_next_steps:
        return response
    response.suggested_next_steps = generate_next_steps(
        question=question,
        response=response,
        semantic=semantic,
    )
    return response


def _default_next_steps() -> list[str]:
    return [
        "How are operations performing today?",
        "Rank all hubs by performance.",
        "What is the pickup SOP?",
    ]


def _memory_analysis_response(
    question: str,
    result_context: dict,
) -> QueryResponse:
    """Analyze a follow-up against the previous analytical result table."""
    answer = analyze_previous_result(question, result_context)
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="memory_analysis",
        execution_order=["memory_analysis"],
        last_result_context=result_context,
    )


def _with_business_context(
    response: QueryResponse,
    business_context: dict,
) -> QueryResponse:
    """Attach inferred business context to a response."""
    response.business_metric = response.business_metric or business_context.get("metric")
    response.analysis_dimension = response.analysis_dimension or business_context.get("analysis_dimension")
    response.date_range = response.date_range or business_context.get("date_range")
    response.analysis_filters = response.analysis_filters or business_context.get("filters")
    return response


def _attach_operational_analysis(
    response: QueryResponse,
    question: str,
    rows: list[dict],
) -> None:
    """Attach root cause, anomaly, and recommendation metadata without changing normal answers."""
    response.root_cause = analyze_root_cause(rows, question)
    response.anomaly = detect_anomalies(rows)
    response.recommendation = response.root_cause.get("recommendation") or recommend(question, rows)
    if response.root_cause.get("main_causes") or response.anomaly.get("is_anomaly"):
        remember_finding(
            question,
            {
                "root_cause": response.root_cause,
                "anomaly": response.anomaly,
                "recommendation": response.recommendation,
            },
        )


def _retrieve_long_memory(
    memory: LongTermMemory,
    question: str,
    business_context: dict,
) -> dict:
    """Retrieve relevant long-term memory before current analysis."""
    if _is_pytest():
        return {"history": [], "issues": []}
    metric = business_context.get("metric")
    dimension = business_context.get("analysis_dimension")
    history = memory.search_similar_history(question, metric=metric, dimension=dimension)
    issues = memory.search_similar_issue(question, metric=metric, dimension=dimension)
    return {"history": history, "issues": issues}


def _finalize_response(
    response: QueryResponse,
    question: str,
    business_context: dict,
    historical_context: dict,
    memory: LongTermMemory,
) -> QueryResponse:
    """Attach long memory context and persist important response details."""
    response.long_memory_matches = historical_context.get("history") or []
    response.previous_issues_found = historical_context.get("issues") or []
    if response.previous_issues_found:
        response.answer = _combine_current_and_historical_analysis(
            response.answer or "",
            response.previous_issues_found,
        )
    saved = _save_long_memory(response, question, business_context, memory)
    response.saved_memory = saved
    response.learned_patterns = detect_repeated_patterns(memory) if saved and not _is_pytest() else []
    return response


def _combine_current_and_historical_analysis(
    current_answer: str,
    previous_issues: list[dict],
) -> str:
    """Combine current analysis with supporting historical memory."""
    issue = previous_issues[0]
    historical = (
        f"Similar issue detected: {issue.get('created_at')}. "
        f"Issue: {issue.get('issue')}. "
        f"Previous root cause: {issue.get('root_cause') or 'not recorded'}."
    )
    return (
        "Current Analysis:\n"
        f"{current_answer}\n\n"
        "Historical Context:\n"
        f"{historical}\n\n"
        "Comparison:\n"
        "Current database results remain the source of truth. Historical memory is supporting evidence only."
    )


def _save_long_memory(
    response: QueryResponse,
    question: str,
    business_context: dict,
    memory: LongTermMemory,
) -> bool:
    """Persist important conversations and findings to SQLite long-term memory."""
    if _is_pytest() or response.capability == "unknown":
        return False

    metric = response.business_metric or business_context.get("metric")
    dimension = response.analysis_dimension or business_context.get("analysis_dimension")
    saved_conversation = memory.save_conversation(
        user_question=response.original_question or question,
        resolved_question=response.resolved_question or question,
        answer=response.answer,
        intent=response.planning_intent,
        metric=metric,
        dimension=dimension,
        sql_query=response.generated_sql,
    )

    saved_finding = False
    if response.root_cause:
        saved_finding = memory.save_finding(
            hub=_dimension_value(response.root_cause, "hub"),
            driver=_dimension_value(response.root_cause, "driver_name"),
            customer=_dimension_value(response.root_cause, "customer_name"),
            issue=str(response.root_cause.get("issue") or ""),
            metric=metric,
            root_cause="; ".join(response.root_cause.get("main_causes") or []),
            recommendation=response.recommendation or response.root_cause.get("recommendation"),
            severity="high" if (response.anomaly or {}).get("is_anomaly") else None,
        )
    return saved_conversation or saved_finding


def _dimension_value(root_cause: dict, key: str) -> str | None:
    prefix = f"{key}:"
    for item in root_cause.get("affected_dimensions") or []:
        text = str(item)
        if text.startswith(prefix):
            return text.split(":", 1)[1].split(",", 1)[0].strip()
    return None


def _is_pytest() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST"))
