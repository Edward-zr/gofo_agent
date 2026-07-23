"""Unit tests for core multi-step Planner."""

from __future__ import annotations

import json
from types import SimpleNamespace

from core.intent_classifier import IntentClassification, IntentType
from core.planner import Planner, ToolName


class _FakeMemory:
    def __init__(self) -> None:
        self._turns = [
            {
                "user_question": "Show today's pickups",
                "assistant_answer": "120 pickups completed today.",
            }
        ]
        self._state = {
            "date_range": "today",
            "filters": {"hub": None},
            "active_entities": {},
            "previous_question": "Show today's pickups",
        }

    def get_recent_history(self, limit: int = 10):
        return self._turns[-limit:]

    def get_current_state(self):
        return self._state


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.payload))


def _classification(intent: IntentType, **overrides) -> IntentClassification:
    base = {
        IntentType.Greeting: dict(confidence=0.99),
        IntentType.SOP_QA: dict(confidence=0.92, requires_rag=True),
        IntentType.SQL_Query: dict(confidence=0.9, requires_sql=True),
        IntentType.SQL_Analysis: dict(
            confidence=0.94, requires_sql=True, requires_planner=True, requires_memory=True
        ),
        IntentType.Dashboard: dict(confidence=0.93, requires_sql=True, requires_planner=True),
        IntentType.Follow_Up: dict(confidence=0.93, requires_memory=True, requires_planner=True),
        IntentType.Unknown: dict(confidence=0.2, requires_clarification=True),
    }[intent]
    base.update(overrides)
    return IntentClassification(intent=intent, reasoning="test", **base)


def test_greeting_plan_is_single_llm_step() -> None:
    plan = Planner().plan("hi", _classification(IntentType.Greeting))
    assert plan.requires_clarification is False
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == ToolName.LLM
    assert plan.estimated_tool_calls == 1


def test_sop_plan_has_rag_then_llm() -> None:
    plan = Planner().plan("Explain the CVG pickup SOP", _classification(IntentType.SOP_QA))
    tools = [step.tool for step in plan.steps]
    assert tools == [ToolName.RAG, ToolName.LLM]
    assert plan.steps[1].depends_on == [1]


def test_sql_query_plan_two_steps() -> None:
    plan = Planner().plan("Show today's pickups", _classification(IntentType.SQL_Query))
    assert [step.tool for step in plan.steps] == [ToolName.SQL, ToolName.LLM]


def test_dashboard_plan_includes_visualization() -> None:
    plan = Planner().plan("Create a dashboard of pickup trends", _classification(IntentType.Dashboard))
    tools = [step.tool for step in plan.steps]
    assert ToolName.SQL in tools
    assert ToolName.VISUALIZATION in tools
    assert ToolName.LLM in tools
    assert plan.estimated_tool_calls >= 3


def test_unknown_requests_clarification() -> None:
    plan = Planner().plan("maybe stuff?", _classification(IntentType.Unknown))
    assert plan.requires_clarification is True
    assert plan.clarification_question
    assert plan.steps == []


def test_sql_analysis_uses_multi_step_template() -> None:
    llm = _FakeLLM({"goal": "unused", "steps": [], "confidence": 0.1, "reasoning": "should not run"})
    plan = Planner(llm=llm).plan(
        "Why did Chicago performance decrease last week?",
        _classification(IntentType.SQL_Analysis),
    )
    assert llm.calls == 0
    # Minimum sources: SQL (+ optional RAG for explain). No Python unless calc/chart.
    assert any(step.tool == ToolName.SQL for step in plan.steps)
    assert plan.steps[-1].tool == ToolName.LLM
    assert ToolName.PYTHON not in [step.tool for step in plan.steps]


def test_comparison_plan_is_multi_step() -> None:
    plan = Planner().plan(
        "Compare Chicago and New York performance this month",
        _classification(IntentType.SQL_Analysis),
    )
    assert len(plan.steps) >= 2
    assert any(step.tool == ToolName.SQL for step in plan.steps)
    assert plan.steps[-1].tool == ToolName.LLM


def test_follow_up_inherits_date_context_in_inputs() -> None:
    memory = _FakeMemory()
    plan = Planner().plan(
        "Show only Chicago",
        _classification(IntentType.Follow_Up),
        memory,
    )
    # Follow-up with non-complex wording uses template Explain/Follow path via LLM or template.
    # Ensure planner returns a usable plan with memory-aware inputs when templated.
    if plan.steps:
        assert any(
            step.inputs.get("date_range") == "today" or step.inputs.get("previous_question")
            for step in plan.steps
        )
    else:
        assert plan.requires_clarification is True


def test_ranking_question_plans_sql_without_python() -> None:
    plan = Planner().plan(
        "Which hub had the lowest pickup success rate last week?",
        _classification(IntentType.SQL_Analysis),
    )
    assert any(step.tool == ToolName.SQL for step in plan.steps)
    assert ToolName.PYTHON not in [step.tool for step in plan.steps]
    assert plan.steps[-1].tool == ToolName.LLM


def test_llm_planner_used_when_no_template(monkeypatch) -> None:
    payload = {
        "goal": "Handle unusual request",
        "steps": [
            {
                "step_number": 1,
                "tool": "LLM",
                "action": "general_reply",
                "description": "Reply",
                "inputs": {},
                "depends_on": [],
            }
        ],
        "estimated_tool_calls": 1,
        "requires_clarification": False,
        "clarification_question": None,
        "confidence": 0.7,
        "reasoning": "fallback",
        "selected_data_sources": ["LLM"],
    }
    llm = _FakeLLM(payload)

    # Force legacy LLM planner path by disabling data-source selection and templates.
    monkeypatch.setattr("core.planner.config.DATA_SOURCE_SELECTION_ENABLED", False)
    monkeypatch.setattr("core.planner._template_plan", lambda *args, **kwargs: None)
    plan = Planner(llm=llm).plan(
        "obscure request xyz",
        _classification(IntentType.SQL_Analysis),
    )
    assert llm.calls == 1
    assert plan.goal == "Handle unusual request"
    assert plan.steps[0].tool == ToolName.LLM
