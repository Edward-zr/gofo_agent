"""Tests for Clarification Manager."""

from __future__ import annotations

from types import SimpleNamespace

from core.clarification_manager import (
    ClarificationManager,
    PendingClarification,
    build_clarification_response,
)
from core.intent_classifier import IntentClassification, IntentType
from core.planner import ExecutionPlan, Planner


class _FakeMemory:
    def __init__(self, state: dict | None = None) -> None:
        self._state = state or {}

    def get_current_state(self):
        return dict(self._state)

    def get_recent_history(self, limit: int = 10):
        return []


def _classification(intent: IntentType = IntentType.SQL_Analysis, **kwargs) -> IntentClassification:
    base = {"confidence": 0.9, "reasoning": "test", "requires_sql": True}
    base.update(kwargs)
    return IntentClassification(intent=intent, **base)


def test_best_driver_asks_for_metric() -> None:
    decision = ClarificationManager().evaluate(
        "Show the best driver.",
        classification=_classification(),
        conversation_memory=_FakeMemory(),
    )
    assert decision.needs_clarification is True
    assert "metric" in (decision.request.missing_fields if decision.request else [])
    assert decision.request and len(decision.request.options) >= 3
    assert "Completed pickups" in decision.request.pending_question


def test_performance_asks_for_scope() -> None:
    decision = ClarificationManager().evaluate(
        "Show performance.",
        classification=_classification(),
        conversation_memory=_FakeMemory(),
    )
    assert decision.needs_clarification is True
    assert decision.request
    assert decision.request.ambiguity_type == "performance_scope"
    labels = [opt.label for opt in decision.request.options]
    assert "By hub" in labels


def test_compare_pickup_rates_asks_for_periods() -> None:
    decision = ClarificationManager().evaluate(
        "Compare pickup rates.",
        classification=_classification(),
        conversation_memory=_FakeMemory(),
    )
    assert decision.needs_clarification is True
    assert decision.request
    assert decision.request.ambiguity_type == "compare_periods"
    assert any("Today vs Yesterday" == opt.label for opt in decision.request.options)


def test_top_customers_asks_for_metric() -> None:
    decision = ClarificationManager().evaluate(
        "Show top customers.",
        classification=_classification(),
        conversation_memory=_FakeMemory(),
    )
    assert decision.needs_clarification is True
    assert decision.request
    assert decision.request.ambiguity_type == "customer_metric"


def test_skips_when_memory_has_metric() -> None:
    decision = ClarificationManager().evaluate(
        "Show the best driver.",
        classification=_classification(),
        conversation_memory=_FakeMemory({"current_metric": "Pickup rate"}),
    )
    assert decision.needs_clarification is False
    assert "memory" in (decision.skip_reason or "").lower()


def test_skips_clear_kpi_question() -> None:
    decision = ClarificationManager().evaluate(
        "What is today's pickup rate?",
        classification=_classification(IntentType.SQL_Query),
        conversation_memory=_FakeMemory(),
        plan=ExecutionPlan(goal="kpi", steps=[], confidence=0.95),
    )
    assert decision.needs_clarification is False


def test_resume_merges_option_into_question() -> None:
    first = ClarificationManager().evaluate(
        "Show the best driver.",
        classification=_classification(),
        conversation_memory=_FakeMemory(),
    )
    assert first.pending is not None
    decision = ClarificationManager().apply_user_response(first.pending, "1")
    assert decision.needs_clarification is False
    assert decision.resumed_question
    assert "Completed pickups" in decision.resumed_question
    assert decision.pending and decision.pending.user_response == "1"


def test_resume_accepts_label_text() -> None:
    first = ClarificationManager().evaluate(
        "Show the best driver.",
        classification=_classification(),
        conversation_memory=_FakeMemory(),
    )
    assert first.pending
    decision = ClarificationManager().apply_user_response(first.pending, "Pickup rate")
    assert "Pickup rate" in (decision.resumed_question or "")


def test_build_clarification_response_capability() -> None:
    decision = ClarificationManager().evaluate(
        "Show the best driver.",
        classification=_classification(),
        conversation_memory=_FakeMemory(),
    )
    response = build_clarification_response(
        decision,
        question="Show the best driver.",
        classification=_classification(),
    )
    assert response.capability == "clarification"
    assert response.requires_clarification is True
    assert response.clarification_options
    assert response.agent_state
    assert response.agent_state.get("missing_fields") == ["metric"]


def test_planner_unknown_still_compatible() -> None:
    plan = Planner().plan("maybe stuff?", _classification(IntentType.Unknown, confidence=0.2, requires_clarification=True))
    decision = ClarificationManager().evaluate(
        "maybe stuff?",
        classification=_classification(IntentType.Unknown, confidence=0.2, requires_clarification=True),
        plan=plan,
        conversation_memory=_FakeMemory(),
    )
    assert decision.needs_clarification is True


def test_agent_asks_then_resumes(monkeypatch) -> None:
    """End-to-end: ambiguous question → clarify → answer option → tools run."""
    import config
    from core.agent import GOFOAgent
    from core.intent_classifier import IntentClassification, IntentType
    from core.models import QueryResponse
    from core.plan_executor import ExecutionResult
    from core.planner import ExecutionPlan

    monkeypatch.setattr(config, "CLARIFICATION_MANAGER_ENABLED", True)
    monkeypatch.setattr(config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(config, "INTENT_CLASSIFIER_ENABLED", True)
    monkeypatch.setattr(config, "REFLECTION_ENABLED", False)
    monkeypatch.setattr(config, "QUALITY_ASSURANCE_ENABLED", False)

    agent = GOFOAgent()
    agent.intent_classifier.classify = lambda *a, **k: IntentClassification(
        intent=IntentType.SQL_Analysis,
        confidence=0.9,
        reasoning="test",
        requires_sql=True,
    )
    agent.planner.plan = lambda *a, **k: ExecutionPlan(
        goal="rank drivers",
        steps=[],
        confidence=0.9,
    )

    executed = {"count": 0}

    def _fake_execute(*a, **k):
        executed["count"] += 1
        resolved = k.get("resolved_question") or ""
        return ExecutionResult(
            response=QueryResponse(
                question=k.get("question") or "",
                answer=f"Executed: {resolved}",
                sources=[],
                capability="sql",
                resolved_question=resolved,
            ),
            plan=ExecutionPlan(goal="rank drivers", steps=[], confidence=0.9),
            step_results={},
        )

    agent.plan_executor.execute = _fake_execute

    first = agent.ask("Show the best driver.")
    first_analysis = first.get("analysis") or {}
    assert first_analysis.get("requires_clarification") is True
    assert first_analysis.get("capability") == "clarification" or (first.get("raw") or {}).get("capability") == "clarification"
    assert executed["count"] == 0
    assert agent.pending_clarification is not None

    second = agent.ask("2")
    second_analysis = second.get("analysis") or {}
    assert not second_analysis.get("requires_clarification")
    assert executed["count"] == 1
    resolved = second_analysis.get("resolved_question") or ""
    answer = second.get("answer") or ""
    assert "Pickup rate" in resolved or "Pickup rate" in answer
    assert agent.pending_clarification is None
    assert agent.memory.global_state.get("current_metric") == "Pickup rate"
