"""Tool Orchestration layer for executing Planner ExecutionPlans.

The Tool Orchestrator NEVER makes planning decisions. The Planner decides WHAT
to do; this module decides HOW to execute (sequential / parallel waves,
dependencies, retries, logging, AgentState updates).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

import config
from core.logger import get_logger
from core.models import QueryResponse, SourceChunk
from core.planner import ExecutionPlan, ExecutionStep, ToolName

logger = get_logger("tool_orchestrator")

TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "503",
    "502",
    "connection reset",
    "connection error",
    "temporarily unavailable",
    "gateway",
    "econnreset",
    "broken pipe",
)

NON_RETRYABLE_MARKERS = (
    "invalid sql",
    "syntax error",
    "must not be empty",
    "validationerror",
    "permission denied",
    "not authorized",
)


class ToolHandler(Protocol):
    """Callable that executes one tool step against an ExecutionContext."""

    def __call__(self, context: "ExecutionContext", step: ExecutionStep) -> dict[str, Any]: ...


class ExecutionResult(BaseModel):
    """Result of a single executed plan step."""

    tool_name: str
    step_number: int
    status: str  # pending|running|done|failed|skipped|retrying
    execution_time_ms: int = 0
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0


class AgentState(BaseModel):
    """Shared mutable state updated after every completed step."""

    question: str = ""
    resolved_question: str = ""
    sql_results: list[dict[str, Any]] = Field(default_factory=list)
    sql_queries: list[str] = Field(default_factory=list)
    retrieved_schema: dict[str, Any] | None = None
    candidate_tables: list[str] = Field(default_factory=list)
    candidate_columns: dict[str, list[str]] = Field(default_factory=dict)
    dataframe: Any | None = None
    transformed_dataframe: Any | None = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    chart_metadata: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    original_question: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    pending_question: str | None = None
    user_response: str | None = None
    documents: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_confidence: dict[str, Any] | None = None
    confidence_score: float | None = None
    confidence_level: str | None = None
    confidence_breakdown: dict[str, Any] | None = None
    similarity_scores: list[float] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    retrieved_sources: list[str] = Field(default_factory=list)
    confidence_reason: str | None = None
    fallback_strategy: str | None = None
    retrieval_policy: dict[str, Any] | None = None
    python_results: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    intermediate_results: dict[int, dict[str, Any]] = Field(default_factory=dict)
    execution_log: list[dict[str, Any]] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class ExecutionContext(BaseModel):
    """Runtime context for orchestrating a full ExecutionPlan."""

    original_question: str
    conversation_memory: Any = None
    execution_plan: ExecutionPlan | None = None
    current_step: int | None = None
    intermediate_results: dict[int, dict[str, Any]] = Field(default_factory=dict)
    sql_results: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_documents: list[dict[str, Any]] = Field(default_factory=list)
    python_results: list[dict[str, Any]] = Field(default_factory=list)
    generated_charts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    execution_time_ms: int = 0
    agent_state: AgentState = Field(default_factory=AgentState)
    step_results: list[ExecutionResult] = Field(default_factory=list)
    resolved_question: str = ""
    attachment_contexts: list[Any] = Field(default_factory=list)
    file_context: dict[str, Any] = Field(default_factory=dict)
    attachment_ids: list[str] = Field(default_factory=list)
    stored_attachments: list[Any] = Field(default_factory=list)
    previous_filter: dict[str, Any] | None = None
    last_entity: str | None = None
    result_context: dict[str, Any] | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class OrchestrationResult:
    """Aggregate result of orchestrating a plan."""

    response: QueryResponse
    context: ExecutionContext
    plan: ExecutionPlan
    step_results: dict[int, dict[str, Any]] = field(default_factory=dict)


class ToolRegistry:
    """Extensible registry of tool handlers."""

    def __init__(self, handlers: dict[str, ToolHandler] | None = None) -> None:
        self._handlers: dict[str, ToolHandler] = dict(handlers or {})

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        self._handlers[str(tool_name)] = handler

    def get(self, tool_name: str) -> ToolHandler | None:
        return self._handlers.get(str(tool_name))

    def names(self) -> list[str]:
        return sorted(self._handlers)

    def __contains__(self, tool_name: object) -> bool:
        return str(tool_name) in self._handlers


def is_transient_error(error: BaseException | str) -> bool:
    """Return True for retryable infrastructure failures."""
    text = str(error).lower()
    if any(marker in text for marker in NON_RETRYABLE_MARKERS):
        return False
    return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)


def _dependency_failed(step: ExecutionStep, failed: set[int]) -> bool:
    return any(dep in failed for dep in step.depends_on)


def _dependencies_met(step: ExecutionStep, completed: set[int], failed: set[int]) -> bool:
    if _dependency_failed(step, failed):
        return False
    return all(dep in completed for dep in step.depends_on)


def _ready_wave(
    pending: dict[int, ExecutionStep],
    completed: set[int],
    failed: set[int],
) -> list[ExecutionStep]:
    ready: list[ExecutionStep] = []
    for step in pending.values():
        if _dependencies_met(step, completed, failed):
            ready.append(step)
    return sorted(ready, key=lambda item: item.step_number)


def _update_agent_state(context: ExecutionContext, step: ExecutionStep, output: dict[str, Any]) -> None:
    state = context.agent_state
    state.intermediate_results[step.step_number] = output
    context.intermediate_results[step.step_number] = output

    if output.get("sql_rows"):
        state.sql_results.extend(output["sql_rows"])
        context.sql_results = list(state.sql_results)
    if output.get("sql"):
        state.sql_queries.append(str(output["sql"]))
    if output.get("retrieved_schema") is not None:
        state.retrieved_schema = output["retrieved_schema"]
        context.metadata["retrieved_schema"] = output["retrieved_schema"]
    if output.get("candidate_tables") is not None:
        state.candidate_tables = list(output["candidate_tables"] or [])
    if output.get("candidate_columns") is not None:
        state.candidate_columns = dict(output["candidate_columns"] or {})
    if output.get("sources"):
        state.documents.extend(output["sources"])
        context.retrieved_documents = list(state.documents)
    if output.get("source_chunks"):
        for chunk in output["source_chunks"]:
            if hasattr(chunk, "model_dump"):
                state.documents.append(chunk.model_dump())
            elif isinstance(chunk, dict):
                state.documents.append(chunk)
        context.retrieved_documents = list(state.documents)
    if output.get("retrieval_confidence") is not None or output.get("confidence_level"):
        if output.get("retrieval_confidence") is not None:
            state.retrieval_confidence = output.get("retrieval_confidence")
        if output.get("confidence_score") is not None:
            state.confidence_score = output.get("confidence_score")
        if output.get("confidence_level") is not None:
            state.confidence_level = output.get("confidence_level")
        if output.get("confidence_breakdown") is not None:
            state.confidence_breakdown = output.get("confidence_breakdown")
        if output.get("similarity_scores") is not None:
            state.similarity_scores = list(output.get("similarity_scores") or [])
        if output.get("retrieved_chunk_count") is not None:
            state.retrieved_chunk_count = int(output.get("retrieved_chunk_count") or 0)
        if output.get("retrieved_sources") is not None:
            state.retrieved_sources = list(output.get("retrieved_sources") or [])
        if output.get("confidence_reason") is not None:
            state.confidence_reason = output.get("confidence_reason")
        if output.get("fallback_strategy") is not None:
            state.fallback_strategy = output.get("fallback_strategy")
        if output.get("retrieval_policy") is not None:
            state.retrieval_policy = output.get("retrieval_policy")
        context.metadata["retrieval_confidence"] = state.retrieval_confidence
        context.metadata["fallback_strategy"] = state.fallback_strategy
    if output.get("summary") or output.get("tool") in {
        ToolName.PYTHON.value,
        ToolName.STATISTICS.value,
        ToolName.TRANSFORM.value,
    }:
        state.python_results.append(
            {
                "step": step.step_number,
                "tool": output.get("tool"),
                "summary": output.get("summary") or output.get("statistics"),
                "functions_executed": output.get("functions_executed") or [],
                "rows": output.get("rows") or [],
            }
        )
        context.python_results = list(state.python_results)
    if output.get("dataframe") is not None:
        state.dataframe = output.get("dataframe")
    if output.get("transformed_dataframe") is not None:
        state.transformed_dataframe = output.get("transformed_dataframe")
    if output.get("statistics"):
        state.statistics = dict(output.get("statistics") or {})
    if output.get("chart_metadata"):
        state.chart_metadata = list(output.get("chart_metadata") or [])
    if output.get("recommendations"):
        state.recommendations = list(output.get("recommendations") or [])
    if output.get("charts"):
        state.charts.extend(output["charts"])
        context.generated_charts = list(state.charts)
        if not state.chart_metadata:
            state.chart_metadata = [
                {
                    "type": chart.get("type"),
                    "title": chart.get("title"),
                    "format": chart.get("format", "png"),
                }
                for chart in output["charts"]
            ]
    if output.get("answer"):
        state.answers.append(str(output["answer"]))
    if output.get("facts") or output.get("tool") == ToolName.KNOWLEDGE_GRAPH.value:
        state.metadata.setdefault("knowledge_graph_facts", [])
        if isinstance(state.metadata["knowledge_graph_facts"], list):
            state.metadata["knowledge_graph_facts"].extend(output.get("facts") or [])


def _log_step(result: ExecutionResult) -> None:
    entry = {
        "tool_name": result.tool_name,
        "step_number": result.step_number,
        "status": result.status,
        "execution_time_ms": result.execution_time_ms,
        "retry_count": result.retry_count,
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "Tool=%s step=%s status=%s duration_ms=%s retries=%s error=%s",
        result.tool_name,
        result.step_number,
        result.status,
        result.execution_time_ms,
        result.retry_count,
        result.error,
    )
    return entry  # type: ignore[return-value]


class ToolExecutor:
    """Execute a single tool step with optional one-shot transient retry."""

    def __init__(self, registry: ToolRegistry, *, max_transient_retries: int = 1) -> None:
        self.registry = registry
        self.max_transient_retries = max_transient_retries

    def execute_step(
        self,
        context: ExecutionContext,
        step: ExecutionStep,
        *,
        commit_state: bool = True,
    ) -> ExecutionResult:
        handler = self.registry.get(step.tool)
        if handler is None:
            result = ExecutionResult(
                tool_name=step.tool,
                step_number=step.step_number,
                status="skipped",
                error=f"unsupported_tool:{step.tool}",
                metadata={"reason": "unsupported_tool"},
            )
            if commit_state:
                self._commit(context, step, result)
            return result

        attempt = 0
        last_error: str | None = None
        while attempt <= self.max_transient_retries:
            started = time.perf_counter()
            try:
                if config.DEBUG:
                    print(
                        "----------------------------------\n"
                        f"Current Step: {step.step_number}\n"
                        f"Tool: {step.tool}\n"
                        f"Action: {step.action}\n"
                        f"Dependencies: {step.depends_on}\n"
                        f"Retry Count: {attempt}\n"
                        "----------------------------------"
                    )
                output = handler(context, step) or {}
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                result = ExecutionResult(
                    tool_name=step.tool,
                    step_number=step.step_number,
                    status="done",
                    execution_time_ms=elapsed_ms,
                    output=output if isinstance(output, dict) else {"value": output},
                    retry_count=attempt,
                    metadata={"action": step.action, "description": step.description},
                )
                if commit_state:
                    self._commit(context, step, result)
                else:
                    _log_step(result)
                return result
            except Exception as exc:  # noqa: BLE001 - orchestrator must not crash workflow
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                last_error = str(exc)
                if attempt < self.max_transient_retries and is_transient_error(exc):
                    attempt += 1
                    if commit_state:
                        context.retry_count += 1
                    logger.warning(
                        "Transient failure on step %s (%s); retrying (%s/%s): %s",
                        step.step_number,
                        step.tool,
                        attempt,
                        self.max_transient_retries,
                        exc,
                    )
                    continue
                result = ExecutionResult(
                    tool_name=step.tool,
                    step_number=step.step_number,
                    status="failed",
                    execution_time_ms=elapsed_ms,
                    error=last_error,
                    retry_count=attempt,
                    metadata={"action": step.action, "transient": is_transient_error(exc)},
                )
                if commit_state:
                    self._commit(context, step, result)
                else:
                    _log_step(result)
                return result

        return ExecutionResult(
            tool_name=step.tool,
            step_number=step.step_number,
            status="failed",
            error=last_error or "unknown_error",
            retry_count=attempt,
        )

    def _commit(self, context: ExecutionContext, step: ExecutionStep, result: ExecutionResult) -> None:
        if result.status == "done":
            _update_agent_state(context, step, result.output)
        if result.retry_count:
            context.retry_count += result.retry_count
        context.step_results.append(result)
        context.agent_state.execution_log.append(
            {
                "tool_name": result.tool_name,
                "step_number": result.step_number,
                "status": result.status,
                "execution_time_ms": result.execution_time_ms,
                "retry_count": result.retry_count,
                "error": result.error,
            }
        )
        _log_step(result)


class ToolOrchestrator:
    """Execute multi-step plans with dependency waves (sequential or parallel)."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        max_workers: int = 4,
        max_transient_retries: int = 1,
        enable_parallel: bool = True,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.executor = ToolExecutor(self.registry, max_transient_retries=max_transient_retries)
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel

    def register_tool(self, tool_name: str, handler: ToolHandler) -> None:
        """Register a future tool without changing orchestration logic."""
        self.registry.register(tool_name, handler)

    def run(self, plan: ExecutionPlan, context: ExecutionContext) -> OrchestrationResult:
        """Execute the plan and return aggregated results + updated AgentState."""
        started = time.perf_counter()
        context.execution_plan = plan
        context.agent_state.question = context.original_question
        context.agent_state.resolved_question = context.resolved_question or context.original_question

        if config.DEBUG:
            print(
                "----------------------------------\n"
                "Execution Plan\n"
                f"Goal: {plan.goal}\n"
                f"Steps: {len(plan.steps)}\n"
                f"Parallel: {self.enable_parallel}\n"
                "----------------------------------"
            )

        if plan.requires_clarification:
            response = QueryResponse(
                question=context.original_question,
                answer=plan.clarification_question
                or "I need more detail before I can run analysis.",
                sources=[],
                capability="clarification",
                planning_capability="planner",
                planning_intent="clarification",
                planning_confidence=plan.confidence,
                planning_reasoning=plan.reasoning,
                original_question=context.original_question,
                resolved_question=context.resolved_question or context.original_question,
                execution_order=[],
                execution_plan=plan.model_dump(),
            )
            context.execution_time_ms = int((time.perf_counter() - started) * 1000)
            return OrchestrationResult(
                response=response,
                context=context,
                plan=plan,
                step_results={},
            )

        steps = sorted(plan.steps, key=lambda item: item.step_number)
        pending = {step.step_number: step for step in steps}
        completed: set[int] = set()
        failed: set[int] = set()
        raw_outputs: dict[int, dict[str, Any]] = {}
        execution_order: list[str] = []

        safety = 0
        while pending and safety < len(steps) + 5:
            safety += 1
            # Skip steps whose dependencies failed.
            for step_number, step in list(pending.items()):
                if _dependency_failed(step, failed):
                    step.status = "skipped"
                    skipped = ExecutionResult(
                        tool_name=step.tool,
                        step_number=step_number,
                        status="skipped",
                        error="dependency_failed",
                        metadata={"depends_on": step.depends_on},
                    )
                    context.step_results.append(skipped)
                    context.agent_state.execution_log.append(skipped.model_dump())
                    raw_outputs[step_number] = {
                        "tool": step.tool,
                        "skipped": True,
                        "reason": "dependency_failed",
                    }
                    completed.add(step_number)
                    del pending[step_number]

            ready = _ready_wave(pending, completed, failed)
            if not ready:
                break

            context.current_step = ready[0].step_number
            if self.enable_parallel and len(ready) > 1:
                wave_results = self._run_parallel(context, ready)
            else:
                wave_results = [self.executor.execute_step(context, step) for step in ready]

            for result in wave_results:
                step = pending.get(result.step_number)
                if step is None:
                    continue
                if result.status == "done":
                    step.status = "done"
                    raw_outputs[result.step_number] = {
                        **result.output,
                        "tool": result.tool_name,
                        "execution_time_ms": result.execution_time_ms,
                        "retry_count": result.retry_count,
                    }
                    execution_order.append(f"{result.tool_name}:{step.action}")
                    completed.add(result.step_number)
                elif result.status == "skipped":
                    step.status = "skipped"
                    raw_outputs[result.step_number] = {
                        "tool": result.tool_name,
                        "skipped": True,
                        "reason": result.error,
                    }
                    completed.add(result.step_number)
                else:
                    step.status = "failed"
                    raw_outputs[result.step_number] = {
                        "tool": result.tool_name,
                        "failed": True,
                        "error": result.error,
                        "retry_count": result.retry_count,
                    }
                    failed.add(result.step_number)
                    completed.add(result.step_number)
                del pending[result.step_number]

        for step_number, step in pending.items():
            step.status = "skipped"
            raw_outputs[step_number] = {
                "tool": step.tool,
                "skipped": True,
                "reason": "unresolved_dependencies",
            }

        context.execution_time_ms = int((time.perf_counter() - started) * 1000)
        response = self._build_response(context, plan, raw_outputs, execution_order)

        if config.DEBUG:
            print(
                "----------------------------------\n"
                "Final Results\n"
                f"Execution Time ms: {context.execution_time_ms}\n"
                f"Steps completed: {len(raw_outputs)}\n"
                f"SQL rows: {len(context.sql_results)}\n"
                f"Documents: {len(context.retrieved_documents)}\n"
                f"Charts: {len(context.generated_charts)}\n"
                f"Retries: {context.retry_count}\n"
                "----------------------------------"
            )

        return OrchestrationResult(
            response=response,
            context=context,
            plan=plan,
            step_results=raw_outputs,
        )

    def _run_parallel(
        self,
        context: ExecutionContext,
        steps: list[ExecutionStep],
    ) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        # Run handlers without committing shared state; merge on the main thread.
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(steps))) as pool:
            futures = {
                pool.submit(
                    self.executor.execute_step,
                    context,
                    step,
                    commit_state=False,
                ): step
                for step in steps
            }
            for future in as_completed(futures):
                step = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = ExecutionResult(
                        tool_name=step.tool,
                        step_number=step.step_number,
                        status="failed",
                        error=str(exc),
                    )
                self.executor._commit(context, step, result)
                results.append(result)
        return sorted(results, key=lambda item: item.step_number)

    def _build_response(
        self,
        context: ExecutionContext,
        plan: ExecutionPlan,
        raw_outputs: dict[int, dict[str, Any]],
        execution_order: list[str],
    ) -> QueryResponse:
        answer = None
        sql = None
        sql_rows: list[dict[str, Any]] | None = None
        charts: list[dict[str, Any]] = list(context.generated_charts)
        sources: list[SourceChunk] = []
        capability = "orchestrator"

        for result in raw_outputs.values():
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
                        try:
                            sources.append(SourceChunk.model_validate(item))
                        except Exception:  # noqa: BLE001
                            continue
            if result.get("answer"):
                answer = result["answer"]
                capability = str(result.get("capability") or capability)

        for result in reversed(list(raw_outputs.values())):
            if result.get("tool") == ToolName.LLM.value and result.get("answer"):
                answer = result["answer"]
                capability = "orchestrator"
                break

        if context.agent_state.answers and not answer:
            answer = context.agent_state.answers[-1]

        step_summary = [
            {
                "step_number": item.step_number,
                "tool": item.tool_name,
                "status": item.status,
                "execution_time_ms": item.execution_time_ms,
                "retry_count": item.retry_count,
                "error": item.error,
            }
            for item in context.step_results
        ]

        return QueryResponse(
            question=context.original_question,
            answer=answer or "I completed the planned steps but could not synthesize an answer.",
            sources=sources,
            capability=capability,
            generated_sql=sql,
            sql_rows=sql_rows if sql_rows is not None else (context.sql_results or None),
            charts=charts or None,
            planning_capability="planner",
            planning_confidence=plan.confidence,
            planning_reasoning=plan.reasoning,
            plan_reason=plan.goal,
            original_question=context.original_question,
            resolved_question=context.resolved_question or context.original_question,
            execution_order=execution_order,
            execution_plan=plan.model_dump(),
            step_results_summary=step_summary,
            agent_state=context.agent_state.model_dump(),
            retrieval_confidence=context.agent_state.retrieval_confidence,
            confidence_score=context.agent_state.confidence_score,
            confidence_level=context.agent_state.confidence_level,
            confidence_breakdown=context.agent_state.confidence_breakdown,
            similarity_scores=list(context.agent_state.similarity_scores or []) or None,
            retrieved_chunk_count=context.agent_state.retrieved_chunk_count or None,
            retrieved_sources=list(context.agent_state.retrieved_sources or []) or None,
            confidence_reason=context.agent_state.confidence_reason,
            fallback_strategy=context.agent_state.fallback_strategy,
            retrieval_policy=context.agent_state.retrieval_policy or plan.retrieval_policy,
        )


def _legacy_step_context(context: ExecutionContext, step: ExecutionStep):
    """Adapt ExecutionContext to PlanExecutor StepContext for reused handlers."""
    from core.plan_executor import StepContext

    return StepContext(
        question=context.original_question,
        resolved_question=context.resolved_question or context.original_question,
        step=step,
        plan=context.execution_plan or ExecutionPlan(goal="", confidence=0.0),
        step_results=context.intermediate_results,
        conversation_memory=context.conversation_memory,
        attachment_contexts=list(context.attachment_contexts or []),
        file_context=dict(context.file_context or {}),
        attachment_ids=list(context.attachment_ids or []),
        stored_attachments=list(context.stored_attachments or []),
        previous_filter=context.previous_filter,
        last_entity=context.last_entity,
        result_context=context.result_context,
        extras=dict(context.extras or {}),
    )


def _wrap_legacy_handler(legacy_handler: Callable[..., dict[str, Any]]) -> ToolHandler:
    def _handler(context: ExecutionContext, step: ExecutionStep) -> dict[str, Any]:
        return legacy_handler(_legacy_step_context(context, step))

    return _handler


def build_default_registry() -> ToolRegistry:
    """Build a registry wrapping existing GOFO tool handlers."""
    from core import plan_executor as pe

    registry = ToolRegistry()
    mapping = {
        ToolName.SQL.value: pe._handle_sql,
        ToolName.KNOWLEDGE_GRAPH.value: pe._handle_knowledge_graph,
        ToolName.RAG.value: pe._handle_rag,
        ToolName.MEMORY.value: pe._handle_memory,
        ToolName.TRANSFORM.value: pe._handle_transform,
        ToolName.STATISTICS.value: pe._handle_statistics,
        ToolName.PYTHON.value: pe._handle_python,
        ToolName.VISUALIZATION.value: pe._handle_visualization,
        ToolName.RECOMMENDATION.value: pe._handle_recommendation,
        ToolName.ATTACHMENT.value: pe._handle_attachment,
        ToolName.LLM.value: pe._handle_llm,
        # Aliases for future / planner naming
        "SQL_PLANNER": pe._handle_sql,
        "SQL_EXECUTOR": pe._handle_sql,
        "GENERATOR": pe._handle_llm,
        "CHART_GENERATOR": pe._handle_visualization,
        "PYTHON_ANALYTICS": pe._handle_python,
        "PYTHON_TRANSFORM": pe._handle_transform,
        "PYTHON_STATISTICS": pe._handle_statistics,
        "RAG_RETRIEVER": pe._handle_rag,
        "KG": pe._handle_knowledge_graph,
        "KNOWLEDGEGRAPH": pe._handle_knowledge_graph,
        "QA": _qa_noop_handler,
        "QUALITY_ASSURANCE": _qa_noop_handler,
    }
    for name, handler in mapping.items():
        registry.register(name, _wrap_legacy_handler(handler))
    return registry


def _qa_noop_handler(ctx: Any) -> dict[str, Any]:
    """QA is evaluated outside orchestration; keep a no-op tool for plan compatibility."""
    return {
        "tool": "QA",
        "answer": None,
        "capability": "qa_placeholder",
        "note": "Quality assurance runs after orchestration in the agent loop.",
    }


def create_execution_context(
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
) -> ExecutionContext:
    """Factory used by agent / PlanExecutor adapters."""
    return ExecutionContext(
        original_question=question,
        resolved_question=resolved_question or question,
        conversation_memory=conversation_memory,
        attachment_contexts=attachment_contexts or [],
        file_context=file_context or {},
        attachment_ids=attachment_ids or [],
        stored_attachments=stored_attachments or [],
        previous_filter=previous_filter,
        last_entity=last_entity,
        result_context=result_context,
        extras=extras or {},
        agent_state=AgentState(
            question=question,
            resolved_question=resolved_question or question,
        ),
    )
