"""Unit tests for ToolOrchestrator sequential/parallel execution."""

from __future__ import annotations

import time

from core.planner import ExecutionPlan, ExecutionStep
from core.tool_orchestrator import (
    ExecutionContext,
    ToolOrchestrator,
    ToolRegistry,
    create_execution_context,
    is_transient_error,
)


def _plan(steps: list[ExecutionStep], **overrides) -> ExecutionPlan:
    data = {
        "goal": "test orchestration",
        "steps": steps,
        "estimated_tool_calls": len(steps),
        "requires_clarification": False,
        "clarification_question": None,
        "confidence": 0.9,
        "reasoning": "unit test",
    }
    data.update(overrides)
    return ExecutionPlan.model_validate(data)


def _registry_with(handlers: dict) -> ToolRegistry:
    registry = ToolRegistry()
    for name, handler in handlers.items():
        registry.register(name, handler)
    return registry


def test_sequential_workflow() -> None:
    order: list[str] = []

    def sql(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        order.append("SQL")
        return {"tool": "SQL", "sql_rows": [{"n": 1}], "sql": "SELECT 1"}

    def llm(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        order.append("LLM")
        assert 1 in ctx.intermediate_results
        return {"tool": "LLM", "answer": "one row", "capability": "llm_summary"}

    orch = ToolOrchestrator(
        registry=_registry_with({"SQL": sql, "LLM": llm}),
        enable_parallel=False,
    )
    result = orch.run(
        _plan(
            [
                ExecutionStep(step_number=1, tool="SQL", action="query", description="q"),
                ExecutionStep(step_number=2, tool="LLM", action="summarize", description="s", depends_on=[1]),
            ]
        ),
        create_execution_context(question="Show pickups"),
    )
    assert order == ["SQL", "LLM"]
    assert result.response.answer == "one row"
    assert result.context.agent_state.sql_results == [{"n": 1}]
    assert result.context.agent_state.answers[-1] == "one row"


def test_parallel_independent_steps() -> None:
    started: dict[str, float] = {}

    def chicago(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        started["CHI"] = time.time()
        time.sleep(0.05)
        return {"tool": "SQL", "sql_rows": [{"hub": "Chicago", "n": 10}], "sql": "CHI"}

    def new_york(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        started["NY"] = time.time()
        time.sleep(0.05)
        return {"tool": "SQL", "sql_rows": [{"hub": "New York", "n": 12}], "sql": "NY"}

    def compare(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        assert 1 in ctx.intermediate_results and 2 in ctx.intermediate_results
        return {"tool": "PYTHON", "summary": {"delta": 2}, "rows": [{"delta": 2}]}

    def summarize(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        return {"tool": "LLM", "answer": "NY ahead by 2", "capability": "llm_summary"}

    # Same tool name SQL for both steps — register once; handlers distinguished by action.
    def sql_router(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        if step.action == "chicago":
            return chicago(ctx, step)
        return new_york(ctx, step)

    orch = ToolOrchestrator(
        registry=_registry_with(
            {
                "SQL": sql_router,
                "PYTHON": compare,
                "LLM": summarize,
            }
        ),
        enable_parallel=True,
        max_workers=2,
    )
    wall_started = time.time()
    result = orch.run(
        _plan(
            [
                ExecutionStep(step_number=1, tool="SQL", action="chicago", description="Chicago"),
                ExecutionStep(step_number=2, tool="SQL", action="new_york", description="NY"),
                ExecutionStep(step_number=3, tool="PYTHON", action="compare", description="cmp", depends_on=[1, 2]),
                ExecutionStep(step_number=4, tool="LLM", action="summarize", description="sum", depends_on=[3]),
            ]
        ),
        create_execution_context(question="Compare Chicago and New York"),
    )
    elapsed = time.time() - wall_started
    assert result.response.answer == "NY ahead by 2"
    assert "CHI" in started and "NY" in started
    # Parallel wave should overlap rather than fully sequential (~0.10s+).
    assert elapsed < 0.18
    assert len(result.context.agent_state.sql_results) >= 2


def test_dependency_skip_on_failure() -> None:
    def boom(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        raise RuntimeError("invalid sql syntax error near SELECT")

    def downstream(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        return {"tool": "LLM", "answer": "should not run"}

    orch = ToolOrchestrator(
        registry=_registry_with({"SQL": boom, "LLM": downstream}),
        enable_parallel=False,
    )
    result = orch.run(
        _plan(
            [
                ExecutionStep(step_number=1, tool="SQL", action="q", description="q"),
                ExecutionStep(step_number=2, tool="LLM", action="s", description="s", depends_on=[1]),
            ]
        ),
        create_execution_context(question="bad sql"),
    )
    statuses = {item.step_number: item.status for item in result.context.step_results}
    assert statuses[1] == "failed"
    assert statuses[2] == "skipped"
    assert result.step_results[2]["reason"] == "dependency_failed"


def test_transient_retry_then_success() -> None:
    calls = {"n": 0}

    def flaky(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("database timeout")
        return {"tool": "SQL", "sql_rows": [{"ok": 1}], "sql": "SELECT 1"}

    def llm(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        return {"tool": "LLM", "answer": "recovered", "capability": "llm_summary"}

    orch = ToolOrchestrator(
        registry=_registry_with({"SQL": flaky, "LLM": llm}),
        enable_parallel=False,
        max_transient_retries=1,
    )
    result = orch.run(
        _plan(
            [
                ExecutionStep(step_number=1, tool="SQL", action="q", description="q"),
                ExecutionStep(step_number=2, tool="LLM", action="s", description="s", depends_on=[1]),
            ]
        ),
        create_execution_context(question="retry me"),
    )
    assert calls["n"] == 2
    assert result.response.answer == "recovered"
    assert result.context.retry_count >= 1


def test_non_transient_error_not_retried() -> None:
    calls = {"n": 0}

    def bad(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        calls["n"] += 1
        raise ValueError("invalid SQL syntax error")

    orch = ToolOrchestrator(
        registry=_registry_with({"SQL": bad}),
        enable_parallel=False,
        max_transient_retries=1,
    )
    result = orch.run(
        _plan([ExecutionStep(step_number=1, tool="SQL", action="q", description="q")]),
        create_execution_context(question="invalid"),
    )
    assert calls["n"] == 1
    assert result.context.step_results[0].status == "failed"


def test_failure_recovery_does_not_crash() -> None:
    def boom(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        raise ConnectionError("connection reset by peer")

    orch = ToolOrchestrator(
        registry=_registry_with({"SQL": boom}),
        enable_parallel=False,
        max_transient_retries=1,
    )
    result = orch.run(
        _plan([ExecutionStep(step_number=1, tool="SQL", action="q", description="q")]),
        create_execution_context(question="network"),
    )
    assert result.response.answer  # structured fallback answer
    assert result.context.step_results[0].status == "failed"


def test_is_transient_error_helpers() -> None:
    assert is_transient_error(TimeoutError("timeout"))
    assert is_transient_error("429 rate limit exceeded")
    assert not is_transient_error(ValueError("invalid SQL syntax error"))


def test_register_future_tool_without_changing_orchestrator() -> None:
    def snowflake(ctx: ExecutionContext, step: ExecutionStep) -> dict:
        return {"tool": "Snowflake", "sql_rows": [{"cloud": 1}]}

    orch = ToolOrchestrator(registry=ToolRegistry(), enable_parallel=False)
    orch.register_tool("Snowflake", snowflake)
    result = orch.run(
        _plan([ExecutionStep(step_number=1, tool="Snowflake", action="query", description="q")]),
        create_execution_context(question="cloud"),
    )
    assert result.step_results[1]["sql_rows"] == [{"cloud": 1}]
    assert result.context.agent_state.sql_results == [{"cloud": 1}]


def test_clarification_short_circuit() -> None:
    orch = ToolOrchestrator(registry=ToolRegistry(), enable_parallel=False)
    result = orch.run(
        _plan([], requires_clarification=True, clarification_question="Which date?"),
        create_execution_context(question="ambiguous"),
    )
    assert result.response.capability == "clarification"
    assert "date" in (result.response.answer or "").lower()
