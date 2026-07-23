"""Bridge specialized agents to existing PlanExecutor tool handlers.

Keeps business logic in tools/; agents stay thin adapters.
"""

from __future__ import annotations

from typing import Any, Callable

from agents.base import AgentTask
from core.plan_executor import StepContext
from core.planner import ExecutionPlan, ExecutionStep


def build_step_context(task: AgentTask) -> StepContext:
    """Materialize a StepContext from an AgentTask + shared runtime context."""
    ctx_data = dict(task.context or {})
    plan = ctx_data.get("plan")
    if not isinstance(plan, ExecutionPlan):
        plan = ExecutionPlan(goal=task.question or task.description or "task", steps=[])

    step = ExecutionStep(
        step_number=int(ctx_data.get("step_number") or 1),
        tool=task.tool or "LLM",
        action=task.action or "execute",
        description=task.description,
        inputs=dict(task.inputs or {}),
        depends_on=[],
    )
    return StepContext(
        question=task.question or ctx_data.get("question") or "",
        resolved_question=task.resolved_question
        or ctx_data.get("resolved_question")
        or task.question
        or "",
        step=step,
        plan=plan,
        step_results=dict(ctx_data.get("step_results") or {}),
        conversation_memory=ctx_data.get("conversation_memory"),
        attachment_contexts=list(ctx_data.get("attachment_contexts") or []),
        file_context=dict(ctx_data.get("file_context") or {}),
        attachment_ids=list(ctx_data.get("attachment_ids") or []),
        stored_attachments=list(ctx_data.get("stored_attachments") or []),
        previous_filter=ctx_data.get("previous_filter"),
        last_entity=ctx_data.get("last_entity"),
        result_context=ctx_data.get("result_context"),
        extras=dict(ctx_data.get("extras") or {}),
    )


def run_tool_handler(
    handler: Callable[[StepContext], dict[str, Any]],
    task: AgentTask,
) -> dict[str, Any]:
    """Invoke a PlanExecutor-compatible tool handler for this task."""
    return handler(build_step_context(task))
