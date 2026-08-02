"""LangGraph node implementations — planner-driven six-stage execution loop."""

from __future__ import annotations

from typing import Any, Callable

from core.agent import GOFOAgent
from core.intent_classifier import IntentClassification
from core.intent_router import RouteIntent
from core.planner import ExecutionPlan
from core.request_verifier import verify as verify_request
from core.task_decomposition import TaskSpec, decompose
from graph.execution import (
    build_wait_for_upload_response,
    commit_and_format,
    ensure_classification,
    execute_plan,
    maybe_clarify,
    plan_retry_from_verification,
    run_planner,
)
from graph.frontier import apply_frontier_to_route, decide_frontier
from graph.observability import timed_node
from graph.state import AgentState
from graph.tools import classify_route, run_conversation_repair, sync_state_from_agent
from graph.transitions import capability_for_intent, log_capability_transition


def make_nodes(agent: GOFOAgent) -> dict[str, Callable[[AgentState], dict[str, Any]]]:
    """Build node callables closed over a session-scoped GOFOAgent."""

    @timed_node("prepare")
    def prepare(state: AgentState) -> dict[str, Any]:
        question = (state.get("user_question") or "").strip()
        attachment_ids = list(state.get("attachment_ids") or [])
        if attachment_ids:
            processed = agent.attachment_service.get_many_processed(
                attachment_ids,
                question=question,
            )
            agent.attachment_memory.register_contexts(processed)
        updates = sync_state_from_agent(agent)
        updates.update(
            {
                "resolved_question": question,
                "messages": [{"role": "user", "content": question}],
                "reasoning_trace": ["Thought: prepare session context and attachments"],
                "error": None,
                "pre_routed": False,
                "attachments_routed": False,
                "conversation_resolution": {},
                "repair_detected": False,
                "repair_type": None,
                "changed_dimension": None,
                "needs_clarification": False,
                "verification_passed": False,
                "verification_report": {},
                "replan_count": 0,
                "execution_plan": {},
                "planner_output": {},
                "task_spec": {},
                "api_response": {},
                "final_answer": "",
                "frontier_decision": {},
                "frontier_problem_kind": "",
                "not_meaningful": False,
            }
        )
        return updates

    @timed_node("intent_understanding")
    def intent_understanding(state: AgentState) -> dict[str, Any]:
        """Stage 1: conversation repair + IntentClassifier signals + IntentRouter."""
        question = state.get("user_question") or ""
        repair = run_conversation_repair(agent, question=question)
        resolved = repair.get("resolved_question") or question

        previous_capability = (
            state.get("current_capability")
            or capability_for_intent(str(state.get("intent") or ""))
            or ""
        )
        previous_visualization = state.get("visualization_request")

        routed = classify_route(
            agent,
            question=resolved,
            attachment_ids=list(state.get("attachment_ids") or []),
            apply_attachments=True,
        )
        decision = dict(routed.get("route_decision") or {})
        intent = str(routed.get("intent") or "")

        classification = ensure_classification(
            agent,
            question=resolved,
            route_decision=decision,
            attachment_ids=list(state.get("attachment_ids") or []),
        )

        transition_traces = log_capability_transition(
            previous_capability=previous_capability,
            intent=intent,
            route_decision=decision,
            previous_visualization=previous_visualization,
        )
        new_capability = capability_for_intent(intent)
        chart_type = decision.get("chart_type")

        updates: dict[str, Any] = {
            **repair,
            **routed,
            "resolved_question": resolved,
            "intent_classification": classification.model_dump(),
            "primary_intent": intent or classification.intent.value,
            "previous_capability": previous_capability,
            "current_capability": new_capability,
            "current_topic": intent or new_capability,
            "attachment_session_active": bool(decision.get("use_attachments")),
            "previous_visualization": previous_visualization,
            "pre_routed": True,
            "reasoning_trace": list(repair.get("reasoning_trace") or [])
            + list(routed.get("reasoning_trace") or [])
            + transition_traces
            + ["Thought: intent understanding complete"],
        }
        if chart_type:
            updates["visualization_request"] = chart_type
        return updates

    @timed_node("frontier")
    def frontier(state: AgentState) -> dict[str, Any]:
        """Frontier: SOP searchable? → ADA/analysis? → tools or not meaningful."""
        question = state.get("resolved_question") or state.get("user_question") or ""
        route_decision = dict(state.get("route_decision") or {})
        decision = decide_frontier(
            question=question,
            route_decision=route_decision,
            attachment_ids=list(state.get("attachment_ids") or []),
            attachment_active=bool(state.get("attachment_session_active")),
            domain=state.get("domain"),
        )
        updated_route = apply_frontier_to_route(route_decision, decision)
        updates: dict[str, Any] = {
            "frontier_decision": decision.model_dump(),
            "frontier_problem_kind": decision.problem_kind,
            "not_meaningful": decision.problem_kind == "not_meaningful",
            "route_decision": updated_route,
            "intent": str(updated_route.get("intent") or state.get("intent") or ""),
            "pending_tools": list(decision.tools),
            "reasoning_trace": [
                f"Thought: frontier kind={decision.problem_kind} "
                f"tools={decision.tools} — {decision.reasoning}"
            ],
        }
        if decision.problem_kind == "sop_searchable":
            updates["domain"] = "SOP"
            updates["current_capability"] = "SOP"
            classification = dict(state.get("intent_classification") or {})
            classification["intent"] = "SOP_QA"
            classification["requires_rag"] = True
            classification["requires_sql"] = False
            updates["intent_classification"] = classification
        if decision.problem_kind == "not_meaningful":
            from core.models import QueryResponse

            response = QueryResponse(
                question=question,
                answer=decision.not_meaningful_message or "",
                capability="conversation",
                planning_intent="not_meaningful",
            )
            api = commit_and_format(
                agent,
                question=state.get("user_question") or question,
                resolved_question=question,
                response=response,
                route_decision=updated_route,
                classification=IntentClassification.model_validate(
                    state.get("intent_classification")
                    or {"intent": "ChitChat", "confidence": 0.5}
                ),
                conversation_resolution=state.get("conversation_resolution") or {},
                repair_detected=bool(state.get("repair_detected")),
                repair_type=state.get("repair_type"),
                changed_dimension=state.get("changed_dimension"),
                attachment_ids=list(state.get("attachment_ids") or []),
            )
            updates["api_response"] = api
            updates["final_answer"] = api.get("answer") or ""
            updates["needs_clarification"] = True  # skip tools via planner route
        return updates

    @timed_node("task_decomposition")
    def task_decomposition(state: AgentState) -> dict[str, Any]:
        """Stage 2: extract TaskSpec from classifier/router/conversation signals."""
        question = state.get("user_question") or ""
        resolved = state.get("resolved_question") or question
        classification_data = state.get("intent_classification") or {}
        classification = (
            IntentClassification.model_validate(classification_data)
            if classification_data
            else None
        )
        snap = agent.state.snapshot()
        memory_snap = (
            agent.memory.get_current_state()
            if hasattr(agent.memory, "get_current_state")
            else {}
        )
        conversation_snapshot = {**snap, **memory_snap}

        # v1: no LLM enrichment — heuristic signals only (plan: existing signals).
        semantic_payload: dict[str, Any] = {
            "metric": conversation_snapshot.get("last_metric")
            or conversation_snapshot.get("current_metric"),
            "dimension": conversation_snapshot.get("last_dimension"),
            "date_range": conversation_snapshot.get("last_date_range")
            or conversation_snapshot.get("date_range"),
            "filters": conversation_snapshot.get("last_filters") or {},
            "entities": conversation_snapshot.get("last_entities")
            or conversation_snapshot.get("active_entities")
            or {},
            "requires_memory": bool(state.get("repair_detected")),
        }
        try:
            from tools.orchestration.semantic_analyzer import _heuristic_fallback

            heuristic = _heuristic_fallback(
                question,
                resolved_question=resolved,
                has_attachments=bool(
                    (state.get("route_decision") or {}).get("use_attachments")
                ),
            )
            semantic_payload = {
                "metric": heuristic.metric or semantic_payload.get("metric"),
                "dimension": heuristic.dimension or semantic_payload.get("dimension"),
                "date_range": semantic_payload.get("date_range"),
                "filters": semantic_payload.get("filters") or {},
                "entities": heuristic.entities or semantic_payload.get("entities") or {},
                "requires_memory": bool(
                    heuristic.use_previous_result or semantic_payload.get("requires_memory")
                ),
            }
        except Exception:  # noqa: BLE001 — keep TaskSpec resilient
            pass

        spec = decompose(
            question=question,
            resolved_question=resolved,
            classification=classification,
            route_decision=state.get("route_decision") or {},
            conversation_snapshot=conversation_snapshot,
            repair={
                "repair_detected": state.get("repair_detected"),
                "repair_type": state.get("repair_type"),
                "changed_dimension": state.get("changed_dimension"),
            },
            semantic=semantic_payload,
        )
        return {
            "task_spec": spec.model_dump(),
            "domain": spec.domain,
            "secondary_intent": spec.secondary_intent,
            "is_continuation": spec.is_continuation,
            "primary_intent": spec.primary_intent or state.get("primary_intent") or "",
            "reasoning_trace": [
                f"Thought: task_spec domain={spec.domain} chart={spec.chart_type} "
                f"limit={spec.limit} attachment={spec.attachment_relevant}"
            ],
        }

    @timed_node("planner")
    def planner(state: AgentState) -> dict[str, Any]:
        """Stage 3–4: Planner.plan always; ClarificationManager may skip tools."""
        question = state.get("user_question") or ""
        resolved = state.get("resolved_question") or question
        route_decision = dict(state.get("route_decision") or {})
        attachment_ids = list(state.get("attachment_ids") or [])
        classification = IntentClassification.model_validate(
            state.get("intent_classification") or {"intent": "Unknown", "confidence": 0.5}
        )
        task_spec = TaskSpec.model_validate(state.get("task_spec") or {})

        # WAIT_FOR_UPLOAD: one-step wait reply (no nested ask / no attachment tool).
        if route_decision.get("intent") == RouteIntent.WAIT_FOR_UPLOAD:
            response = build_wait_for_upload_response(question=question)
            response.execution_plan = {
                "goal": "Wait for file upload",
                "steps": [],
                "requires_clarification": True,
            }
            api = commit_and_format(
                agent,
                question=question,
                resolved_question=resolved,
                response=response,
                route_decision=route_decision,
                classification=classification,
                conversation_resolution=state.get("conversation_resolution") or {},
                repair_detected=bool(state.get("repair_detected")),
                repair_type=state.get("repair_type"),
                changed_dimension=state.get("changed_dimension"),
                attachment_ids=attachment_ids,
            )
            return {
                "needs_clarification": True,
                "execution_plan": response.execution_plan,
                "planner_output": response.execution_plan,
                "api_response": api,
                "final_answer": api.get("answer") or "",
                "current_tool": "wait_upload",
                "reasoning_trace": ["Thought: wait_for_upload — skip tool execution"],
            }

        # Resume clarification before planning when a pending question exists.
        resumed_question: str | None = None
        if agent.pending_clarification is not None and not attachment_ids:
            _, _, _, resumed_question = maybe_clarify(
                agent,
                question=question,
                resolved_question=resolved,
                classification=classification,
                plan=ExecutionPlan(goal="pending", steps=[]),
                route_decision=route_decision,
                attachment_ids=attachment_ids,
            )
            if resumed_question:
                resolved = resumed_question

        plan = run_planner(
            agent,
            question=question,
            resolved_question=resolved,
            classification=classification,
            route_decision=route_decision,
            task_spec=task_spec,
            attachment_ids=attachment_ids,
        )

        needs = False
        clarify_response = None
        if resumed_question is None:
            needs, clarify_response, plan, _ = maybe_clarify(
                agent,
                question=question,
                resolved_question=resolved,
                classification=classification,
                plan=plan,
                route_decision=route_decision,
                attachment_ids=attachment_ids,
            )

        plan_dump = plan.model_dump()
        updates_extra: dict[str, Any] = {}
        if resumed_question:
            updates_extra["resolved_question"] = resolved
        if needs and clarify_response is not None:
            api = commit_and_format(
                agent,
                question=question,
                resolved_question=resolved,
                response=clarify_response,
                route_decision=route_decision,
                classification=classification,
                conversation_resolution=state.get("conversation_resolution") or {},
                repair_detected=bool(state.get("repair_detected")),
                repair_type=state.get("repair_type"),
                changed_dimension=state.get("changed_dimension"),
                attachment_ids=attachment_ids,
            )
            return {
                **updates_extra,
                "needs_clarification": True,
                "execution_plan": plan_dump,
                "planner_output": plan_dump,
                "api_response": api,
                "final_answer": api.get("answer") or "",
                "current_tool": "clarification",
                "reasoning_trace": ["Thought: clarification required — skip tools"],
            }

        tools = [str(step.tool) for step in plan.steps]
        return {
            **updates_extra,
            "needs_clarification": False,
            "execution_plan": plan_dump,
            "planner_output": plan_dump,
            "current_tool": tools[0] if tools else "planner",
            "pending_tools": tools,
            "reasoning_trace": [
                f"Thought: plan steps={len(plan.steps)} tools={tools} "
                f"capabilities={plan.required_capabilities}"
            ],
        }

    @timed_node("tool_execution")
    def tool_execution(state: AgentState) -> dict[str, Any]:
        """Stage 5: PlanExecutor only — no nested GOFOAgent.ask."""
        question = state.get("user_question") or ""
        resolved = state.get("resolved_question") or question
        route_decision = dict(state.get("route_decision") or {})
        attachment_ids = list(state.get("attachment_ids") or [])
        classification = IntentClassification.model_validate(
            state.get("intent_classification") or {"intent": "Unknown", "confidence": 0.5}
        )
        task_spec = TaskSpec.model_validate(state.get("task_spec") or {})
        plan = ExecutionPlan.model_validate(state.get("execution_plan") or {"goal": "", "steps": []})

        response = execute_plan(
            agent,
            plan=plan,
            question=question,
            resolved_question=resolved,
            route_decision=route_decision,
            classification=classification,
            task_spec=task_spec,
            attachment_ids=attachment_ids,
        )
        # Defer memory commit until after verification (may replan once).
        # For replan path we still need a draft api_response for the verifier.
        from core.agent import format_agent_response

        draft = format_agent_response(response)
        # Stash raw response fields needed for final commit
        tool_name = state.get("current_tool") or "plan"
        history: dict[str, Any] = {
            "tool_history": [
                {
                    "tool": tool_name,
                    "intent": state.get("intent"),
                    "question": question,
                    "ok": True,
                    "replan_count": state.get("replan_count") or 0,
                }
            ],
            "tool_outputs": {tool_name: {"answer": response.answer}},
            "api_response": draft,
            "final_answer": response.answer or "",
            "reasoning_trace": [
                f"Action: PlanExecutor tools={[s.tool for s in plan.steps]}",
                f"Observation: capability={response.capability}",
            ],
            "debug": {"_pending_query_response": response.model_dump()},
        }
        if response.generated_sql or response.sql_rows:
            history["sql_history"] = [
                {"sql": response.generated_sql, "rows": response.sql_rows or []}
            ]
        if response.sources:
            history["rag_history"] = [
                {"answer": response.answer, "sources": [s.model_dump() for s in response.sources]}
            ]
            history["sop_history"] = history["rag_history"]
        return history

    @timed_node("result_verification")
    def result_verification(state: AgentState) -> dict[str, Any]:
        """Stage 6: verify against TaskSpec; SOP never requests SQL retry."""
        task_spec = state.get("task_spec") or {}
        api = state.get("api_response") or {}
        report = verify_request(
            task_spec=task_spec,
            response=api,
            route_decision=state.get("route_decision") or {},
            execution_plan=state.get("execution_plan") or {},
        )
        report_dump = report.model_dump()
        # Attach report into draft analysis for finalize / replan feedback.
        api = dict(api)
        analysis = dict(api.get("analysis") or {})
        analysis["verification_report"] = report_dump
        api["analysis"] = analysis

        passed = bool(report.passed)
        replan_count = int(state.get("replan_count") or 0)
        will_replan = (not passed) and replan_count < 1

        updates: dict[str, Any] = {
            "verification_report": report_dump,
            "verification_passed": passed,
            "api_response": api,
            "reasoning_trace": [
                f"Thought: verify passed={passed} action={report.action} "
                f"replan={'yes' if will_replan else 'no'}"
            ],
        }

        # Commit memory when we will finalize (pass or exhausted retries).
        if not will_replan:
            pending = (state.get("debug") or {}).get("_pending_query_response")
            if pending:
                from core.models import QueryResponse

                response = QueryResponse.model_validate(pending)
                classification = IntentClassification.model_validate(
                    state.get("intent_classification")
                    or {"intent": "Unknown", "confidence": 0.5}
                )
                api = commit_and_format(
                    agent,
                    question=state.get("user_question") or "",
                    resolved_question=state.get("resolved_question")
                    or state.get("user_question")
                    or "",
                    response=response,
                    route_decision=state.get("route_decision") or {},
                    classification=classification,
                    conversation_resolution=state.get("conversation_resolution") or {},
                    repair_detected=bool(state.get("repair_detected")),
                    repair_type=state.get("repair_type"),
                    changed_dimension=state.get("changed_dimension"),
                    attachment_ids=list(state.get("attachment_ids") or []),
                    verification_report=report_dump,
                )
                updates["api_response"] = api
                updates["final_answer"] = api.get("answer") or ""
                updates["debug"] = {"_pending_query_response": None}
        return updates

    @timed_node("replan")
    def replan(state: AgentState) -> dict[str, Any]:
        """One automatic re-plan from verification feedback."""
        from core.request_verifier import VerificationReport

        question = state.get("resolved_question") or state.get("user_question") or ""
        classification = IntentClassification.model_validate(
            state.get("intent_classification") or {"intent": "Unknown", "confidence": 0.5}
        )
        task_spec = TaskSpec.model_validate(state.get("task_spec") or {})
        verification = VerificationReport.model_validate(
            state.get("verification_report") or {}
        )
        previous_plan = state.get("execution_plan") or {}
        previous_answer = (state.get("api_response") or {}).get("answer")

        new_plan = plan_retry_from_verification(
            agent,
            question=question,
            classification=classification,
            verification=verification,
            previous_plan=previous_plan,
            previous_answer=previous_answer,
            task_spec=task_spec,
        )
        plan_dump = new_plan.model_dump()
        tools = [str(step.tool) for step in new_plan.steps]
        return {
            "execution_plan": plan_dump,
            "planner_output": plan_dump,
            "replan_count": int(state.get("replan_count") or 0) + 1,
            "verification_passed": False,
            "current_tool": tools[0] if tools else "replan",
            "pending_tools": tools,
            "reasoning_trace": [
                f"Thought: replan action={verification.action} steps={len(new_plan.steps)}"
            ],
            "debug": {"_pending_query_response": None},
        }

    @timed_node("summarize")
    def summarize(state: AgentState) -> dict[str, Any]:
        """Frontier summarize: present sub-agent output as the user-facing answer."""
        response = dict(state.get("api_response") or {})
        answer = state.get("final_answer") or response.get("answer") or ""
        frontier = dict(state.get("frontier_decision") or {})
        kind = frontier.get("problem_kind") or state.get("frontier_problem_kind") or ""
        analysis = dict(response.get("analysis") or {})
        # SOP answers keep RAG text; strip accidental SQL UNKNOWN noise.
        if kind == "sop_searchable":
            sql = response.get("sql") or ""
            if sql and "UNKNOWN" in str(sql).upper():
                response["sql"] = ""
                response["data"] = []
                analysis.pop("root_cause", None)
        analysis["frontier_summarized"] = True
        analysis["frontier_problem_kind"] = kind
        analysis["frontier_tools"] = frontier.get("tools") or []
        response["analysis"] = analysis
        response["answer"] = answer
        return {
            "api_response": response,
            "final_answer": answer,
            "reasoning_trace": [f"Thought: frontier summarize kind={kind}"],
        }

    @timed_node("finalize")
    def finalize(state: AgentState) -> dict[str, Any]:
        response = dict(state.get("api_response") or {})
        answer = state.get("final_answer") or response.get("answer") or ""
        if state.get("error") and not answer:
            answer = (
                "I hit an internal orchestration error while processing that request. "
                "Please try again."
            )
            response = {
                "answer": answer,
                "sql": None,
                "data": [],
                "analysis": {"error": state.get("error"), "langgraph": True},
            }
        analysis = dict(response.get("analysis") or {})
        analysis["langgraph"] = True
        analysis["langgraph_phase"] = 2
        analysis["langgraph_loop"] = "frontier_planner"
        analysis["langgraph_pre_routed"] = bool(state.get("pre_routed"))
        analysis["langgraph_intent"] = state.get("intent")
        analysis["langgraph_tool"] = state.get("current_tool")
        analysis["langgraph_timings_ms"] = state.get("node_timings_ms") or {}
        analysis["langgraph_domain"] = state.get("domain")
        analysis["langgraph_replan_count"] = state.get("replan_count") or 0
        analysis["langgraph_verification_passed"] = bool(state.get("verification_passed"))
        analysis["frontier_problem_kind"] = state.get("frontier_problem_kind")
        if state.get("frontier_decision"):
            analysis["frontier_decision"] = state.get("frontier_decision")
        if state.get("verification_report"):
            analysis["verification_report"] = state.get("verification_report")
        if state.get("task_spec"):
            analysis["task_spec"] = state.get("task_spec")
        response["analysis"] = analysis
        response["answer"] = answer
        return {
            "api_response": response,
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "reasoning_trace": ["Thought: finalize response for API/UI"],
            **sync_state_from_agent(agent),
        }

    return {
        "prepare": prepare,
        "intent_understanding": intent_understanding,
        "frontier": frontier,
        "task_decomposition": task_decomposition,
        "planner": planner,
        "tool_execution": tool_execution,
        "result_verification": result_verification,
        "replan": replan,
        "summarize": summarize,
        "finalize": finalize,
    }


def route_after_frontier(state: AgentState) -> str:
    """Not meaningful → finalize; else continue decomposition."""
    if state.get("not_meaningful"):
        return "finalize"
    return "task_decomposition"


def route_after_planner(state: AgentState) -> str:
    """After plan: clarify / not-meaningful → finalize, else tool_execution."""
    if state.get("needs_clarification") or state.get("not_meaningful"):
        return "finalize"
    return "tool_execution"


def route_after_verification(state: AgentState) -> str:
    """After verify: fail with retries → replan; else frontier summarize."""
    if not state.get("verification_passed") and int(state.get("replan_count") or 0) < 1:
        return "replan"
    return "summarize"


# Backward-compatible alias used by older tests.
def route_after_supervisor(state: AgentState) -> str:
    """Deprecated Phase-1 helper — maps to planner-driven tool selection."""
    tool = (state.get("current_tool") or "").strip()
    if tool in {"ada", "sql", "rag", "chat", "wait_upload", "legacy"}:
        return tool if tool != "legacy" else "legacy_execute"
    intent = state.get("intent") or ""
    mapping = {
        RouteIntent.ATTACHMENT_ANALYSIS: "ada",
        RouteIntent.ATTACHMENT_VISUALIZATION: "ada",
        RouteIntent.SOP_QA: "rag",
        RouteIntent.SQL_ANALYTICS: "sql",
        RouteIntent.FOLLOW_UP: "sql",
        RouteIntent.GENERAL_CHAT: "chat",
        RouteIntent.OPENAI_FALLBACK: "chat",
        RouteIntent.WAIT_FOR_UPLOAD: "wait_upload",
    }
    return mapping.get(intent, "legacy_execute")
