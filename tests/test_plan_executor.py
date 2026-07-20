"""Unit tests for PlanExecutor dependency execution."""

from __future__ import annotations

from types import SimpleNamespace

from core.plan_executor import PlanExecutor, StepContext
from core.planner import ExecutionPlan, ExecutionStep, ToolName
from core.models import QueryResponse


def _plan(steps: list[ExecutionStep], **overrides) -> ExecutionPlan:
    data = {
        "goal": "test plan",
        "steps": steps,
        "estimated_tool_calls": len(steps),
        "requires_clarification": False,
        "clarification_question": None,
        "confidence": 0.9,
        "reasoning": "unit test",
    }
    data.update(overrides)
    return ExecutionPlan.model_validate(data)


def test_clarification_short_circuit() -> None:
    plan = _plan(
        [],
        requires_clarification=True,
        clarification_question="Do you want today's data?",
        confidence=0.3,
    )
    result = PlanExecutor(tool_registry={}).execute(plan, question="ambiguous")
    assert result.response.capability == "clarification"
    assert "today" in (result.response.answer or "").lower()
    assert result.step_results == {}


def test_dependency_order_with_mocked_tools() -> None:
    order: list[str] = []

    def sql_handler(ctx: StepContext) -> dict:
        order.append("SQL")
        return {"tool": "SQL", "sql_rows": [{"hub": "ORD", "pickups": 10}], "sql": "SELECT 1"}

    def python_handler(ctx: StepContext) -> dict:
        order.append("PYTHON")
        assert 1 in ctx.step_results
        return {"tool": "PYTHON", "summary": {"row_count": 1}, "rows": [{"hub": "ORD", "pickups": 10}]}

    def llm_handler(ctx: StepContext) -> dict:
        order.append("LLM")
        assert 2 in ctx.step_results
        return {"tool": "LLM", "answer": "ORD has 10 pickups.", "capability": "llm_summary"}

    executor = PlanExecutor(
        tool_registry={
            ToolName.SQL.value: sql_handler,
            ToolName.PYTHON.value: python_handler,
            ToolName.LLM.value: llm_handler,
        }
    )
    plan = _plan(
        [
            ExecutionStep(step_number=1, tool="SQL", action="query", description="q"),
            ExecutionStep(step_number=2, tool="PYTHON", action="calc", description="c", depends_on=[1]),
            ExecutionStep(step_number=3, tool="LLM", action="summarize", description="s", depends_on=[2]),
        ]
    )
    result = executor.execute(plan, question="Show hub pickups")
    assert order == ["SQL", "PYTHON", "LLM"]
    assert result.response.answer == "ORD has 10 pickups."
    assert result.response.generated_sql == "SELECT 1"
    assert result.response.execution_order == ["SQL:query", "PYTHON:calc", "LLM:summarize"]


def test_unknown_tool_is_skipped() -> None:
    def llm_handler(ctx: StepContext) -> dict:
        return {"tool": "LLM", "answer": "Fallback summary", "capability": "llm_summary"}

    executor = PlanExecutor(
        tool_registry={
            ToolName.LLM.value: llm_handler,
        }
    )
    plan = _plan(
        [
            ExecutionStep(step_number=1, tool="Snowflake", action="query", description="future tool"),
            ExecutionStep(step_number=2, tool="LLM", action="summarize", description="summary", depends_on=[1]),
        ]
    )
    result = executor.execute(plan, question="query snowflake")
    assert result.step_results[1]["skipped"] is True
    assert result.response.answer == "Fallback summary"


def test_failed_dependency_skips_children() -> None:
    def boom(ctx: StepContext) -> dict:
        raise RuntimeError("sql down")

    def llm_handler(ctx: StepContext) -> dict:
        return {"tool": "LLM", "answer": "partial", "capability": "llm_summary"}

    class _LLM:
        def invoke(self, messages):  # noqa: ANN001
            return SimpleNamespace(content="Could not complete analysis.")

    executor = PlanExecutor(
        tool_registry={
            ToolName.SQL.value: boom,
            ToolName.LLM.value: llm_handler,
        },
        llm=_LLM(),
    )
    plan = _plan(
        [
            ExecutionStep(step_number=1, tool="SQL", action="query", description="q"),
            ExecutionStep(step_number=2, tool="LLM", action="summarize", description="s", depends_on=[1]),
        ]
    )
    result = executor.execute(plan, question="failing sql")
    assert result.step_results[1]["failed"] is True
    assert result.step_results[2]["skipped"] is True


def test_memory_handler_loads_prior_context() -> None:
    memory = SimpleNamespace(
        get_current_state=lambda: {
            "previous_answer": "120 pickups",
            "previous_question": "Show today's pickups",
            "last_result_context": {"summary": "120 pickups today"},
            "previous_rows": [{"n": 120}],
        }
    )
    plan = _plan(
        [
            ExecutionStep(step_number=1, tool="MEMORY", action="load", description="load"),
            ExecutionStep(step_number=2, tool="LLM", action="explain", description="explain", depends_on=[1]),
        ]
    )

    class _LLM:
        def invoke(self, messages):  # noqa: ANN001
            return SimpleNamespace(content="Because volume was 120.")

    result = PlanExecutor(llm=_LLM()).execute(
        plan,
        question="Explain these numbers",
        conversation_memory=memory,
    )
    assert result.step_results[1]["previous_answer"] == "120 pickups"
    assert "120" in (result.response.answer or "")


def test_executor_builds_query_response_fields() -> None:
    def sql_handler(ctx: StepContext) -> dict:
        return {
            "tool": "SQL",
            "sql": "SELECT 1",
            "sql_rows": [{"x": 1}],
            "response": QueryResponse(
                question=ctx.question,
                answer="raw sql answer",
                capability="sql",
                generated_sql="SELECT 1",
                sql_rows=[{"x": 1}],
            ),
        }

    def llm_handler(ctx: StepContext) -> dict:
        return {"tool": "LLM", "answer": "final summary", "capability": "llm_summary"}

    result = PlanExecutor(
        tool_registry={"SQL": sql_handler, "LLM": llm_handler}
    ).execute(
        _plan(
            [
                ExecutionStep(step_number=1, tool="SQL", action="q", description="q"),
                ExecutionStep(step_number=2, tool="LLM", action="s", description="s", depends_on=[1]),
            ]
        ),
        question="q",
    )
    assert result.response.answer == "final summary"
    assert result.response.execution_plan is not None
    assert result.response.step_results_summary
