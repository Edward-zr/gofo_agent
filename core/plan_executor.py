"""Execute structured ExecutionPlans against existing GOFO tool modules.

The PlanExecutor runs tools; the Planner never does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from core.logger import get_logger
from core.models import QueryRequest, QueryResponse, SourceChunk
from core.planner import ExecutionPlan, ExecutionStep, ToolName
from tools.llm.client import get_llm

logger = get_logger("plan_executor")

ToolHandler = Callable[["StepContext"], dict[str, Any]]


@dataclass
class StepContext:
    """Runtime context passed to each tool handler."""

    question: str
    resolved_question: str
    step: ExecutionStep
    plan: ExecutionPlan
    step_results: dict[int, dict[str, Any]]
    conversation_memory: Any = None
    attachment_contexts: list[Any] = field(default_factory=list)
    file_context: dict[str, Any] = field(default_factory=dict)
    attachment_ids: list[str] = field(default_factory=list)
    stored_attachments: list[Any] = field(default_factory=list)
    previous_filter: dict[str, Any] | None = None
    last_entity: str | None = None
    result_context: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Collected output from executing a plan."""

    response: QueryResponse
    step_results: dict[int, dict[str, Any]]
    plan: ExecutionPlan


def _dependency_ready(step: ExecutionStep, completed: set[int], failed: set[int]) -> bool:
    if any(dep in failed for dep in step.depends_on):
        return False
    return all(dep in completed for dep in step.depends_on)


def _rows_from_results(step_results: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    for result in reversed(list(step_results.values())):
        rows = result.get("sql_rows") or result.get("rows")
        if isinstance(rows, list) and rows:
            return rows
    return []


def _dataframe_from_results(step_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    for result in reversed(list(step_results.values())):
        frame = result.get("dataframe")
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
    rows = _rows_from_results(step_results)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _handle_sql(ctx: StepContext) -> dict[str, Any]:
    from tools.sql import service as sql_service

    question = ctx.step.inputs.get("question") or ctx.resolved_question or ctx.question
    response = sql_service.answer(QueryRequest(question=question))
    return {
        "tool": ToolName.SQL,
        "answer": response.answer,
        "sql": response.generated_sql,
        "sql_rows": response.sql_rows or [],
        "rewritten_question": response.rewritten_question,
        "capability": "sql",
        "response": response,
    }


def _handle_rag(ctx: StepContext) -> dict[str, Any]:
    from tools.rag import service as rag_service

    question = ctx.step.inputs.get("question") or ctx.resolved_question or ctx.question
    # retrieve-only for intermediate steps; generate on summarize actions
    if ctx.step.action.startswith("retrieve"):
        response = rag_service.retrieve_only(QueryRequest(question=question))
        return {
            "tool": ToolName.RAG,
            "sources": [source.model_dump() for source in response.sources],
            "source_chunks": response.sources,
            "capability": "rag",
            "response": response,
        }
    response = rag_service.answer(QueryRequest(question=question))
    return {
        "tool": ToolName.RAG,
        "answer": response.answer,
        "sources": [source.model_dump() for source in response.sources],
        "source_chunks": response.sources,
        "capability": "rag",
        "response": response,
    }


def _handle_memory(ctx: StepContext) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if ctx.conversation_memory is not None and hasattr(ctx.conversation_memory, "get_current_state"):
        state = dict(ctx.conversation_memory.get_current_state() or {})
    last_result = ctx.result_context or state.get("last_result_context") or {}
    return {
        "tool": ToolName.MEMORY,
        "memory_state": state,
        "last_result_context": last_result,
        "previous_answer": state.get("previous_answer"),
        "previous_question": state.get("previous_question"),
        "sql_rows": state.get("previous_rows") or last_result.get("rows") or [],
        "capability": "memory",
    }


def _handle_python(ctx: StepContext) -> dict[str, Any]:
    frame = _dataframe_from_results(ctx.step_results)
    summary: dict[str, Any] = {"row_count": int(len(frame))}
    if frame.empty:
        return {
            "tool": ToolName.PYTHON,
            "dataframe": frame,
            "summary": summary,
            "capability": "python",
        }

    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if numeric_cols:
        summary["numeric_means"] = {
            col: float(frame[col].mean()) for col in numeric_cols[:8] if frame[col].notna().any()
        }
        for col in numeric_cols:
            lower = str(col).lower()
            if "success" in lower or "completion" in lower or "rate" in lower:
                summary["focus_metric"] = col
                summary["focus_mean"] = float(frame[col].mean())
                break

    # Lightweight WoW-style delta when period-like columns exist.
    period_cols = [col for col in frame.columns if str(col).lower() in {"week", "period", "date_range"}]
    if period_cols and numeric_cols:
        period_col = period_cols[0]
        metric_col = numeric_cols[0]
        grouped = frame.groupby(period_col, dropna=True)[metric_col].mean().dropna()
        if len(grouped) >= 2:
            values = list(grouped.values)
            summary["wow_change"] = float(values[-1] - values[-2])
            summary["wow_change_pct"] = (
                float((values[-1] - values[-2]) / values[-2]) if values[-2] else None
            )

    return {
        "tool": ToolName.PYTHON,
        "dataframe": frame,
        "rows": frame.head(100).to_dict(orient="records"),
        "summary": summary,
        "capability": "python",
    }


def _handle_visualization(ctx: StepContext) -> dict[str, Any]:
    from tools.files.charts import build_charts

    frame = _dataframe_from_results(ctx.step_results)
    if frame.empty:
        rows = _rows_from_results(ctx.step_results)
        frame = pd.DataFrame(rows) if rows else pd.DataFrame()
    charts = build_charts(frame, question=ctx.resolved_question or ctx.question)
    return {
        "tool": ToolName.VISUALIZATION,
        "charts": charts,
        "dataframe": frame,
        "capability": "visualization",
    }


def _handle_attachment(ctx: StepContext) -> dict[str, Any]:
    from tools.files.analyzer import analyze_attachments
    from tools.files.source_router import DataSource

    if not ctx.attachment_contexts:
        return {
            "tool": ToolName.ATTACHMENT,
            "answer": "No attachment is available for analysis. Please upload a file first.",
            "capability": "attachment",
            "skipped": True,
        }
    response = analyze_attachments(
        question=ctx.question,
        resolved_question=ctx.resolved_question,
        contexts=ctx.attachment_contexts,
        file_context=ctx.file_context,
        data_sources=[DataSource.ATTACHMENT],
        intent="FILE_ANALYSIS",
        stored_attachments=ctx.stored_attachments or None,
        previous_filter=ctx.previous_filter,
        last_entity=ctx.last_entity,
    )
    if response is None:
        return {
            "tool": ToolName.ATTACHMENT,
            "answer": "Attachment analysis could not be completed.",
            "capability": "attachment",
            "skipped": True,
        }
    return {
        "tool": ToolName.ATTACHMENT,
        "answer": response.answer,
        "charts": response.charts or [],
        "sql_rows": response.sql_rows or [],
        "capability": "attachment",
        "response": response,
    }


def _collect_evidence(step_results: dict[int, dict[str, Any]]) -> str:
    chunks: list[str] = []
    for step_number, result in sorted(step_results.items()):
        tool = result.get("tool", "unknown")
        if result.get("answer"):
            chunks.append(f"Step {step_number} ({tool}) answer:\n{result['answer']}")
        if result.get("sql"):
            chunks.append(f"Step {step_number} SQL:\n{result['sql']}")
        rows = result.get("sql_rows") or result.get("rows")
        if rows:
            preview = rows[:8]
            chunks.append(f"Step {step_number} rows preview:\n{preview}")
        if result.get("summary"):
            chunks.append(f"Step {step_number} python summary:\n{result['summary']}")
        sources = result.get("sources") or []
        if sources:
            texts = [str(source.get("text", ""))[:200] for source in sources[:3] if isinstance(source, dict)]
            if texts:
                chunks.append(f"Step {step_number} SOP excerpts:\n" + "\n---\n".join(texts))
        memory = result.get("last_result_context")
        if memory:
            chunks.append(f"Step {step_number} memory context:\n{memory}")
    return "\n\n".join(chunks) if chunks else "(no tool evidence)"


def _best_tool_answer(step_results: dict[int, dict[str, Any]]) -> str | None:
    """Reuse a substantive tool answer to avoid redundant LLM calls."""
    for result in reversed(list(step_results.values())):
        if result.get("tool") not in {
            ToolName.SQL.value,
            ToolName.RAG.value,
            ToolName.ATTACHMENT.value,
            ToolName.MEMORY.value,
        }:
            continue
        answer = result.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        response = result.get("response")
        if isinstance(response, QueryResponse) and response.answer:
            return str(response.answer).strip()
    return None


def _handle_llm(ctx: StepContext) -> dict[str, Any]:
    action = (ctx.step.action or "").lower()
    question = ctx.resolved_question or ctx.question

    if action == "greet":
        answer = (
            "Hello. I am the GOFO Operations Intelligence Analyst. "
            "I can analyze pickup performance, investigate root causes, "
            "compare hubs and drivers, and answer SOP questions."
        )
        return {"tool": ToolName.LLM, "answer": answer, "capability": "conversation"}

    reuse = _best_tool_answer(ctx.step_results)
    summarize_actions = {
        "summarize",
        "summarize_results",
        "summarize_findings",
        "summarize_follow_up",
        "summarize_sop",
        "summarize_key_points",
        "summarize_attachment",
        "summarize_dashboard",
        "explain_result",
        "explain",
    }
    if reuse and (action in summarize_actions or action.endswith("summarize") or "summar" in action):
        python_summary = None
        for result in ctx.step_results.values():
            if result.get("tool") == ToolName.PYTHON.value and result.get("summary"):
                python_summary = result["summary"]
                break
        answer = reuse
        if python_summary:
            answer = f"{reuse}\n\nDerived metrics: {python_summary}"
        return {"tool": ToolName.LLM, "answer": answer, "capability": "llm_summary", "reused_tool_answer": True}

    evidence = _collect_evidence(ctx.step_results)
    memory_bits = ""
    if ctx.conversation_memory is not None and hasattr(ctx.conversation_memory, "get_current_state"):
        state = ctx.conversation_memory.get_current_state() or {}
        memory_bits = (
            f"Previous question: {state.get('previous_question')}\n"
            f"Date range: {state.get('date_range')}\n"
            f"Filters: {state.get('filters') or state.get('active_filters')}\n"
        )

    if action in {"general_reply"} or not ctx.step_results:
        system = (
            "You are the GOFO Operations Intelligence Agent. "
            "Answer helpfully. For non-operational chit-chat keep replies brief. "
            "For coding questions, provide clear code guidance."
        )
        user = f"User question:\n{question}"
    else:
        system = (
            "You are a GOFO operations analyst. Using ONLY the tool evidence below, "
            "write a clear operational answer. Do not invent metrics that are not present."
        )
        user = (
            f"User question:\n{question}\n\n"
            f"Conversation memory:\n{memory_bits or '(none)'}\n\n"
            f"Tool evidence:\n{evidence}"
        )

    llm = ctx.extras.get("llm") or get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content if hasattr(response, "content") else response
    answer = content if isinstance(content, str) else str(content)
    return {"tool": ToolName.LLM, "answer": answer, "capability": "llm_summary"}


DEFAULT_TOOL_REGISTRY: dict[str, ToolHandler] = {
    ToolName.SQL.value: _handle_sql,
    ToolName.RAG.value: _handle_rag,
    ToolName.MEMORY.value: _handle_memory,
    ToolName.PYTHON.value: _handle_python,
    ToolName.VISUALIZATION.value: _handle_visualization,
    ToolName.ATTACHMENT.value: _handle_attachment,
    ToolName.LLM.value: _handle_llm,
}


def tool_registry_is_default(registry: dict[str, ToolHandler]) -> bool:
    """True when registry still points at the built-in GOFO tool handlers."""
    if set(registry.keys()) != set(DEFAULT_TOOL_REGISTRY.keys()):
        return False
    return all(registry[key] is DEFAULT_TOOL_REGISTRY[key] for key in DEFAULT_TOOL_REGISTRY)


class PlanExecutor:
    """Run ExecutionPlan steps in dependency order and synthesize a response.

    When TOOL_ORCHESTRATOR_ENABLED is true (default), execution is delegated to
    ToolOrchestrator (parallel waves + retries + AgentState). Legacy sequential
    execution remains as a fallback.
    """

    def __init__(
        self,
        *,
        tool_registry: dict[str, ToolHandler] | None = None,
        llm: Any | None = None,
    ) -> None:
        self.tool_registry = dict(tool_registry or DEFAULT_TOOL_REGISTRY)
        self._llm = llm

    def execute(
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
    ) -> ExecutionResult:
        """Execute the plan and return a QueryResponse plus step traces."""
        import config as _config

        if getattr(_config, "TOOL_ORCHESTRATOR_ENABLED", True) and tool_registry_is_default(
            self.tool_registry
        ):
            return self._execute_via_orchestrator(
                plan,
                question=question,
                resolved_question=resolved_question,
                conversation_memory=conversation_memory,
                attachment_contexts=attachment_contexts,
                file_context=file_context,
                attachment_ids=attachment_ids,
                stored_attachments=stored_attachments,
                previous_filter=previous_filter,
                last_entity=last_entity,
                result_context=result_context,
            )

        resolved = resolved_question or question

        if plan.requires_clarification:
            answer = plan.clarification_question or (
                "I need more detail before I can run analysis. "
                "Please clarify the time range or metric you care about."
            )
            response = QueryResponse(
                question=question,
                answer=answer,
                sources=[],
                capability="clarification",
                planning_capability="planner",
                planning_intent="clarification",
                planning_confidence=plan.confidence,
                planning_reasoning=plan.reasoning,
                original_question=question,
                resolved_question=resolved,
                execution_order=[],
            )
            return ExecutionResult(response=response, step_results={}, plan=plan)

        steps = sorted(plan.steps, key=lambda item: item.step_number)
        step_results: dict[int, dict[str, Any]] = {}
        completed: set[int] = set()
        failed: set[int] = set()
        execution_order: list[str] = []

        pending = {step.step_number: step for step in steps}
        safety = 0
        while pending and safety < len(steps) + 5:
            safety += 1
            progress = False
            for step_number, step in list(pending.items()):
                if not _dependency_ready(step, completed, failed):
                    if any(dep in failed for dep in step.depends_on):
                        step.status = "skipped"
                        step_results[step_number] = {
                            "tool": step.tool,
                            "skipped": True,
                            "reason": "dependency_failed",
                        }
                        completed.add(step_number)
                        del pending[step_number]
                        progress = True
                    continue

                step.status = "running"
                handler = self.tool_registry.get(step.tool)
                if handler is None:
                    logger.warning("Unknown tool %s — skipping step %s", step.tool, step_number)
                    step.status = "skipped"
                    step_results[step_number] = {
                        "tool": step.tool,
                        "skipped": True,
                        "reason": f"unsupported_tool:{step.tool}",
                    }
                    completed.add(step_number)
                    del pending[step_number]
                    progress = True
                    continue

                ctx = StepContext(
                    question=question,
                    resolved_question=resolved,
                    step=step,
                    plan=plan,
                    step_results=step_results,
                    conversation_memory=conversation_memory,
                    attachment_contexts=attachment_contexts or [],
                    file_context=file_context or {},
                    attachment_ids=attachment_ids or [],
                    stored_attachments=stored_attachments or [],
                    previous_filter=previous_filter,
                    last_entity=last_entity,
                    result_context=result_context,
                    extras={"llm": self._llm},
                )
                try:
                    result = handler(ctx)
                    step.status = "done"
                    step_results[step_number] = result
                    execution_order.append(f"{step.tool}:{step.action}")
                    completed.add(step_number)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Step %s (%s) failed: %s", step_number, step.tool, exc)
                    step.status = "failed"
                    step_results[step_number] = {
                        "tool": step.tool,
                        "error": str(exc),
                        "failed": True,
                    }
                    failed.add(step_number)
                    completed.add(step_number)
                del pending[step_number]
                progress = True
            if not progress:
                break

        for step_number, step in pending.items():
            step.status = "skipped"
            step_results[step_number] = {
                "tool": step.tool,
                "skipped": True,
                "reason": "unresolved_dependencies",
            }

        # Ensure a final summary exists when the plan did not end with LLM.
        if steps and not any(
            (step_results.get(step.step_number) or {}).get("tool") == ToolName.LLM.value
            and (step_results.get(step.step_number) or {}).get("answer")
            for step in steps
        ):
            synthetic = ExecutionStep(
                step_number=max(step.step_number for step in steps) + 1,
                tool=ToolName.LLM.value,
                action="summarize_findings",
                description="Synthesize tool results into a final answer",
            )
            ctx = StepContext(
                question=question,
                resolved_question=resolved,
                step=synthetic,
                plan=plan,
                step_results=step_results,
                conversation_memory=conversation_memory,
                attachment_contexts=attachment_contexts or [],
                file_context=file_context or {},
                attachment_ids=attachment_ids or [],
                stored_attachments=stored_attachments or [],
                previous_filter=previous_filter,
                last_entity=last_entity,
                result_context=result_context,
                extras={"llm": self._llm},
            )
            summary_result = _handle_llm(ctx)
            step_results[synthetic.step_number] = summary_result
            execution_order.append(f"{ToolName.LLM.value}:summarize_findings")

        response = self._build_response(
            question=question,
            resolved=resolved,
            plan=plan,
            step_results=step_results,
            execution_order=execution_order,
        )
        return ExecutionResult(response=response, step_results=step_results, plan=plan)

    def _execute_via_orchestrator(
        self,
        plan: ExecutionPlan,
        *,
        question: str,
        resolved_question: str | None,
        conversation_memory: Any,
        attachment_contexts: list[Any] | None,
        file_context: dict[str, Any] | None,
        attachment_ids: list[str] | None,
        stored_attachments: list[Any] | None,
        previous_filter: dict[str, Any] | None,
        last_entity: str | None,
        result_context: dict[str, Any] | None,
    ) -> ExecutionResult:
        """Delegate to ToolOrchestrator for parallel waves, retries, and AgentState."""
        import config as _config
        from core.tool_orchestrator import ToolOrchestrator, create_execution_context

        orchestrator = ToolOrchestrator(
            enable_parallel=getattr(_config, "TOOL_ORCHESTRATOR_PARALLEL", True),
            max_workers=getattr(_config, "TOOL_ORCHESTRATOR_MAX_WORKERS", 4),
        )
        context = create_execution_context(
            question=question,
            resolved_question=resolved_question,
            conversation_memory=conversation_memory,
            attachment_contexts=attachment_contexts,
            file_context=file_context,
            attachment_ids=attachment_ids,
            stored_attachments=stored_attachments,
            previous_filter=previous_filter,
            last_entity=last_entity,
            result_context=result_context,
            extras={"llm": self._llm},
        )
        result = orchestrator.run(plan, context)
        return ExecutionResult(
            response=result.response,
            step_results=result.step_results,
            plan=result.plan,
        )

    def _build_response(
        self,
        *,
        question: str,
        resolved: str,
        plan: ExecutionPlan,
        step_results: dict[int, dict[str, Any]],
        execution_order: list[str],
    ) -> QueryResponse:
        answer = None
        sql = None
        sql_rows: list[dict[str, Any]] | None = None
        charts: list[dict[str, Any]] = []
        sources: list[SourceChunk] = []
        capability = "planner"

        for result in step_results.values():
            if result.get("sql"):
                sql = result["sql"]
            if result.get("sql_rows"):
                sql_rows = result["sql_rows"]
            if result.get("charts"):
                charts.extend(result["charts"])
            if result.get("source_chunks"):
                sources = list(result["source_chunks"])
            elif result.get("sources") and not sources:
                for item in result["sources"]:
                    if isinstance(item, dict) and item.get("id") is not None:
                        sources.append(SourceChunk.model_validate(item))
            if result.get("response") and isinstance(result["response"], QueryResponse):
                # Prefer richer subsystem response when available.
                embedded = result["response"]
                if embedded.kpi_summary:
                    pass
            if result.get("answer"):
                answer = result["answer"]
                capability = str(result.get("capability") or capability)

        # Prefer the last LLM answer as the user-facing response.
        for result in reversed(list(step_results.values())):
            if result.get("tool") == ToolName.LLM.value and result.get("answer"):
                answer = result["answer"]
                capability = "planner"
                break

        # If a full QueryResponse was produced by SQL/RAG/attachment, merge useful fields.
        embedded_response: QueryResponse | None = None
        for result in reversed(list(step_results.values())):
            candidate = result.get("response")
            if isinstance(candidate, QueryResponse):
                embedded_response = candidate
                break

        if embedded_response is not None:
            return embedded_response.model_copy(
                update={
                    "answer": answer or embedded_response.answer,
                    "question": question,
                    "original_question": question,
                    "resolved_question": resolved,
                    "generated_sql": sql or embedded_response.generated_sql,
                    "sql_rows": sql_rows if sql_rows is not None else embedded_response.sql_rows,
                    "charts": charts or embedded_response.charts or [],
                    "sources": sources or embedded_response.sources,
                    "capability": capability or embedded_response.capability,
                    "planning_capability": "planner",
                    "planning_confidence": plan.confidence,
                    "planning_reasoning": plan.reasoning,
                    "plan_reason": plan.goal,
                    "execution_order": execution_order,
                    "execution_plan": plan.model_dump(),
                    "step_results_summary": _summarize_steps(step_results),
                }
            )

        return QueryResponse(
            question=question,
            answer=answer or "I completed the planned steps but could not synthesize an answer.",
            sources=sources,
            capability=capability,
            generated_sql=sql,
            sql_rows=sql_rows,
            charts=charts or None,
            planning_capability="planner",
            planning_confidence=plan.confidence,
            planning_reasoning=plan.reasoning,
            plan_reason=plan.goal,
            original_question=question,
            resolved_question=resolved,
            execution_order=execution_order,
            execution_plan=plan.model_dump(),
            step_results_summary=_summarize_steps(step_results),
        )


def _summarize_steps(step_results: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for step_number, result in sorted(step_results.items()):
        summary.append(
            {
                "step_number": step_number,
                "tool": result.get("tool"),
                "skipped": bool(result.get("skipped")),
                "failed": bool(result.get("failed")),
                "has_answer": bool(result.get("answer")),
                "has_sql": bool(result.get("sql")),
                "row_count": len(result.get("sql_rows") or result.get("rows") or []),
                "chart_count": len(result.get("charts") or []),
                "error": result.get("error"),
                "reason": result.get("reason"),
            }
        )
    return summary
