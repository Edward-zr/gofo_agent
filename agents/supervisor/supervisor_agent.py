"""Supervisor Agent — capability-based dispatch via Agent Registry.

Contains NO business logic. Only:
  Planner output → Registry lookup → dispatch → collect → merge.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any
from uuid import uuid4

import config
from agents import register_default_agents
from agents.base import (
    TOOL_TO_CAPABILITY,
    AgentCapability,
    AgentResponse,
    AgentStatus,
    AgentTask,
)
from agents.registry.agent_registry import AgentRegistry, get_agent_registry
from core.logger import get_logger
from core.models import QueryResponse, SourceChunk
from core.plan_executor import ExecutionResult
from core.planner import ExecutionPlan, ExecutionStep

logger = get_logger("supervisor")


def _debug_enabled() -> bool:
    return bool(
        getattr(config, "DEBUG", False)
        or getattr(config, "MULTI_AGENT_DEBUG", False)
    )


def _capabilities_from_plan(plan: ExecutionPlan) -> list[str]:
    """Derive required capabilities from plan (never agent names)."""
    caps = list(getattr(plan, "required_capabilities", None) or [])
    if caps:
        return caps
    derived: list[str] = []
    for step in plan.steps:
        cap = TOOL_TO_CAPABILITY.get(step.tool)
        if cap and cap not in derived:
            derived.append(cap)
    # selected_data_sources may use tool names
    for source in plan.selected_data_sources or []:
        cap = TOOL_TO_CAPABILITY.get(str(source), str(source).lower())
        if cap and cap not in derived:
            derived.append(cap)
    return derived


def _task_from_step(
    step: ExecutionStep,
    *,
    question: str,
    resolved_question: str,
    plan: ExecutionPlan,
    runtime: dict[str, Any],
    step_results: dict[int, dict[str, Any]],
) -> AgentTask:
    capability = TOOL_TO_CAPABILITY.get(step.tool) or AgentCapability.CONVERSATION.value
    return AgentTask(
        task_id=f"step-{step.step_number}-{uuid4().hex[:8]}",
        capability=capability,
        tool=step.tool,
        action=step.action,
        description=step.description,
        question=question,
        resolved_question=resolved_question,
        inputs=dict(step.inputs or {}),
        depends_on=[f"step-{dep}" for dep in step.depends_on],
        context={
            **runtime,
            "plan": plan,
            "step_number": step.step_number,
            "step_results": step_results,
        },
    )


def _dependency_ready(step: ExecutionStep, completed: set[int], failed: set[int]) -> bool:
    if any(dep in failed for dep in step.depends_on):
        return False
    return all(dep in completed for dep in step.depends_on)


def _merge_agent_results(
    *,
    question: str,
    resolved_question: str,
    plan: ExecutionPlan,
    step_results: dict[int, dict[str, Any]],
    agent_responses: list[AgentResponse],
    execution_order: list[str],
) -> QueryResponse:
    """Merge standardized agent payloads into a QueryResponse (no domain logic)."""
    answer: str | None = None
    sources: list[SourceChunk] = []
    sql: str | None = None
    rows: list[dict[str, Any]] | None = None
    charts: list[Any] = []
    capability = "multi_agent"
    confidences: list[float] = []

    for result in reversed(list(step_results.values())):
        if not isinstance(result, dict):
            continue
        if result.get("answer") and not answer:
            answer = result["answer"]
        if (result.get("generated_sql") or result.get("sql")) and not sql:
            sql = result.get("generated_sql") or result.get("sql")
        if result.get("sql_rows") and not rows:
            rows = result["sql_rows"]
        for chunk in result.get("sources") or []:
            if isinstance(chunk, SourceChunk):
                sources.append(chunk)
            elif isinstance(chunk, dict):
                try:
                    sources.append(SourceChunk.model_validate(chunk))
                except Exception:  # noqa: BLE001
                    continue
        if result.get("charts"):
            charts.extend(result["charts"] if isinstance(result["charts"], list) else [])
        tool = result.get("tool")
        if tool and capability == "multi_agent":
            capability = str(tool).lower() if isinstance(tool, str) else capability

    for resp in agent_responses:
        confidences.append(resp.confidence)
        if resp.status == AgentStatus.SUCCESS and resp.summary and not answer:
            answer = resp.summary

    if not answer:
        parts = [r.summary for r in agent_responses if r.summary]
        answer = "\n\n".join(parts) if parts else "No agent produced an answer."

    avg_conf = sum(confidences) / len(confidences) if confidences else plan.confidence
    summary = [
        {
            "agent": r.agent,
            "status": r.status.value,
            "confidence": r.confidence,
            "latency_ms": r.latency_ms,
            "summary": r.summary[:200] if r.summary else "",
        }
        for r in agent_responses
    ]

    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        capability=capability,
        generated_sql=sql,
        sql_rows=rows,
        charts=charts or None,
        planning_capability="supervisor",
        planning_intent="multi_agent",
        planning_confidence=avg_conf,
        planning_reasoning=plan.reasoning,
        original_question=question,
        resolved_question=resolved_question,
        execution_order=execution_order,
        execution_plan=plan.model_dump(),
        step_results_summary=summary,
        agent_state={
            "supervisor": {
                "required_capabilities": _capabilities_from_plan(plan),
                "agent_responses": [r.to_dict() for r in agent_responses],
                "execution_order": execution_order,
            }
        },
    )


class SupervisorAgent:
    """Coordinates specialized agents through the Agent Registry only."""

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        *,
        max_workers: int | None = None,
        enable_parallel: bool | None = None,
    ) -> None:
        self.registry = registry or get_agent_registry()
        register_default_agents(self.registry)
        self.max_workers = max_workers or int(
            getattr(config, "MULTI_AGENT_MAX_WORKERS", None)
            or getattr(config, "TOOL_ORCHESTRATOR_MAX_WORKERS", 4)
        )
        self.enable_parallel = (
            enable_parallel
            if enable_parallel is not None
            else bool(getattr(config, "MULTI_AGENT_PARALLEL", True))
        )

    def execute_plan(
        self,
        plan: ExecutionPlan,
        *,
        question: str,
        resolved_question: str | None = None,
        conversation_memory: Any = None,
        attachment_contexts: list[Any] | None = None,
        file_context: dict[str, Any] | None = None,
        attachment_ids: list[str] | None = None,
        stored_attachments: list[Any] | None = None,
        previous_filter: dict[str, Any] | None = None,
        last_entity: str | None = None,
        result_context: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Dispatch plan steps to agents selected by capability."""
        started = perf_counter()
        resolved = resolved_question or question
        required_caps = _capabilities_from_plan(plan)
        selection = self.registry.resolve_capabilities(required_caps)

        runtime = {
            "question": question,
            "resolved_question": resolved,
            "conversation_memory": conversation_memory,
            "attachment_contexts": attachment_contexts or [],
            "file_context": file_context or {},
            "attachment_ids": attachment_ids or [],
            "stored_attachments": stored_attachments or [],
            "previous_filter": previous_filter,
            "last_entity": last_entity,
            "result_context": result_context,
            "extras": extras or {},
        }

        if _debug_enabled():
            print("----------------------------------")
            print("Supervisor.execute_plan")
            print(f"  goal={plan.goal!r}")
            print(f"  required_capabilities={required_caps}")
            print(f"  registered_agents={self.registry.discover()}")
            print(f"  selected_agents={selection}")
            print(f"  parallel={self.enable_parallel} max_workers={self.max_workers}")
            print("----------------------------------")

        if plan.requires_clarification:
            answer = plan.clarification_question or (
                "I need more detail before I can run analysis."
            )
            response = QueryResponse(
                question=question,
                answer=answer,
                sources=[],
                capability="clarification",
                planning_capability="supervisor",
                planning_intent="clarification",
                planning_confidence=plan.confidence,
                planning_reasoning=plan.reasoning,
                original_question=question,
                resolved_question=resolved,
                execution_order=[],
            )
            return ExecutionResult(response=response, step_results={}, plan=plan)

        steps = sorted(plan.steps, key=lambda s: s.step_number)
        step_results: dict[int, dict[str, Any]] = {}
        agent_responses: list[AgentResponse] = []
        completed: set[int] = set()
        failed: set[int] = set()
        execution_order: list[str] = []
        pending = {s.step_number: s for s in steps}

        while pending:
            ready = [
                step
                for step in pending.values()
                if _dependency_ready(step, completed, failed)
            ]
            if not ready:
                # Remaining steps blocked by failures
                for step in list(pending.values()):
                    step.status = "skipped"
                    step_results[step.step_number] = {
                        "tool": step.tool,
                        "skipped": True,
                        "reason": "dependency_failed",
                    }
                    pending.pop(step.step_number, None)
                break

            wave_results = self._run_wave(ready, question, resolved, plan, runtime, step_results)

            for step, response in wave_results:
                pending.pop(step.step_number, None)
                agent_responses.append(response)
                label = f"{response.agent}:{step.action or step.tool}"
                execution_order.append(label)

                payload = response.result if isinstance(response.result, dict) else {
                    "answer": response.summary,
                    "raw": response.result,
                }
                payload = {
                    **payload,
                    "agent": response.agent,
                    "status": response.status.value,
                    "confidence": response.confidence,
                    "tool": step.tool,
                }
                step_results[step.step_number] = payload

                if response.status == AgentStatus.FAILED:
                    step.status = "failed"
                    failed.add(step.step_number)
                else:
                    step.status = "done"
                    completed.add(step.step_number)

                if _debug_enabled():
                    print(
                        f"  step={step.step_number} agent={response.agent} "
                        f"status={response.status.value} latency_ms={response.latency_ms:.1f} "
                        f"confidence={response.confidence:.2f}"
                    )

        # Optional reflection pass when plan requests answer_review capability.
        if AgentCapability.ANSWER_REVIEW.value in required_caps or getattr(
            config, "MULTI_AGENT_REFLECTION", False
        ):
            reflection_resp = self._maybe_reflect(
                question=resolved,
                plan=plan,
                step_results=step_results,
                runtime=runtime,
            )
            if reflection_resp is not None:
                agent_responses.append(reflection_resp)
                execution_order.append(f"{reflection_resp.agent}:review")

        response = _merge_agent_results(
            question=question,
            resolved_question=resolved,
            plan=plan,
            step_results=step_results,
            agent_responses=agent_responses,
            execution_order=execution_order,
        )

        if _debug_enabled():
            elapsed = (perf_counter() - started) * 1000.0
            print(f"  merged_answer_chars={len(response.answer or '')}")
            print(f"  total_latency_ms={elapsed:.1f}")
            print("----------------------------------")

        return ExecutionResult(response=response, step_results=step_results, plan=plan)

    def _run_wave(
        self,
        steps: list[ExecutionStep],
        question: str,
        resolved: str,
        plan: ExecutionPlan,
        runtime: dict[str, Any],
        step_results: dict[int, dict[str, Any]],
    ) -> list[tuple[ExecutionStep, AgentResponse]]:
        def _dispatch(step: ExecutionStep) -> tuple[ExecutionStep, AgentResponse]:
            task = _task_from_step(
                step,
                question=question,
                resolved_question=resolved,
                plan=plan,
                runtime=runtime,
                step_results=step_results,
            )
            agent = self.registry.select_for_task(task, healthy_only=True)
            if agent is None:
                # Retry without health filter once.
                agent = self.registry.select_for_task(task, healthy_only=False)
            if agent is None:
                return step, AgentResponse(
                    agent="Supervisor",
                    status=AgentStatus.FAILED,
                    confidence=0.0,
                    error=f"No agent for capability={task.capability} tool={task.tool}",
                    summary=f"No registered agent for capability '{task.capability}'",
                )
            return step, agent.run(task)

        if not self.enable_parallel or len(steps) == 1:
            return [_dispatch(step) for step in steps]

        results: list[tuple[ExecutionStep, AgentResponse]] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(steps))) as pool:
            futures = {pool.submit(_dispatch, step): step for step in steps}
            for future in as_completed(futures):
                results.append(future.result())
        # Stable order by step number
        results.sort(key=lambda item: item[0].step_number)
        return results

    def _maybe_reflect(
        self,
        *,
        question: str,
        plan: ExecutionPlan,
        step_results: dict[int, dict[str, Any]],
        runtime: dict[str, Any],
    ) -> AgentResponse | None:
        agent = self.registry.select_agent(AgentCapability.ANSWER_REVIEW.value)
        if agent is None:
            return None
        # Build a lightweight draft from latest answers
        answer = None
        for result in reversed(list(step_results.values())):
            if isinstance(result, dict) and result.get("answer"):
                answer = result["answer"]
                break
        if not answer:
            return None
        draft = QueryResponse(
            question=question,
            answer=answer,
            capability="multi_agent",
            generated_sql=next(
                (
                    r.get("generated_sql") or r.get("sql")
                    for r in reversed(list(step_results.values()))
                    if isinstance(r, dict) and (r.get("generated_sql") or r.get("sql"))
                ),
                None,
            ),
            sql_rows=next(
                (
                    r.get("sql_rows")
                    for r in reversed(list(step_results.values()))
                    if isinstance(r, dict) and r.get("sql_rows")
                ),
                None,
            ),
        )
        task = AgentTask(
            task_id=f"reflect-{uuid4().hex[:8]}",
            capability=AgentCapability.ANSWER_REVIEW.value,
            action="critique",
            question=question,
            resolved_question=question,
            context={
                **runtime,
                "plan": plan,
                "draft_response": draft,
            },
        )
        return agent.run(task)
