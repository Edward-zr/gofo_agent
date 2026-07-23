"""Tests for Multi-Agent Architecture (Registry + Supervisor)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents import register_default_agents
from agents.base import AgentCapability, AgentResponse, AgentStatus, AgentTask, BaseAgent
from agents.registry.agent_registry import AgentRegistry, reset_agent_registry
from agents.supervisor.supervisor_agent import SupervisorAgent
from core.plan_executor import PlanExecutor, ExecutionResult
from core.planner import ExecutionPlan, ExecutionStep


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_agent_registry()
    yield
    reset_agent_registry()


@pytest.fixture
def enable_multi_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("config.MULTI_AGENT_ENABLED", True)
    monkeypatch.setattr("config.MULTI_AGENT_PARALLEL", True)
    monkeypatch.setattr("config.MULTI_AGENT_REFLECTION", False)


def test_registry_discovers_capabilities() -> None:
    registry = register_default_agents(AgentRegistry())
    caps = registry.list_capabilities()
    assert AgentCapability.DATABASE_QUERY.value in caps
    assert AgentCapability.DOCUMENT_RETRIEVAL.value in caps
    assert AgentCapability.STATISTICS.value in caps
    assert AgentCapability.CONVERSATION.value in caps
    assert "SQL" in caps[AgentCapability.DATABASE_QUERY.value]
    assert "RAG" in caps[AgentCapability.DOCUMENT_RETRIEVAL.value]


def test_supervisor_selects_by_capability_not_name() -> None:
    registry = AgentRegistry()

    class FakeSQL(BaseAgent):
        name = "CustomSQLImpl"
        version = "9.0"
        capabilities = [AgentCapability.DATABASE_QUERY]

        def execute(self, task: AgentTask) -> AgentResponse:
            return AgentResponse(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                confidence=0.99,
                result={"tool": "SQL", "sql": "SELECT 1", "answer": "ok", "sql_rows": [{"n": 1}]},
                summary="ok",
            )

    agent = FakeSQL()
    agent.initialize()
    registry.register(agent)

    supervisor = SupervisorAgent(registry=registry, enable_parallel=False)
    plan = ExecutionPlan(
        goal="count",
        steps=[ExecutionStep(step_number=1, tool="SQL", action="query", description="q")],
        required_capabilities=[AgentCapability.DATABASE_QUERY.value],
        confidence=0.9,
    )
    result = supervisor.execute_plan(plan, question="How many pickups?")
    assert result.response.answer == "ok"
    assert result.step_results[1]["agent"] == "CustomSQLImpl"


def test_parallel_independent_agents(enable_multi_agent: None) -> None:
    registry = AgentRegistry()
    order: list[str] = []

    class A(BaseAgent):
        name = "A"
        capabilities = [AgentCapability.DATABASE_QUERY]

        def execute(self, task: AgentTask) -> AgentResponse:
            order.append("A")
            return AgentResponse(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                confidence=0.9,
                result={"tool": "SQL", "answer": "sql-part", "sql_rows": [{"x": 1}]},
                summary="sql-part",
            )

    class B(BaseAgent):
        name = "B"
        capabilities = [AgentCapability.DOCUMENT_RETRIEVAL]

        def execute(self, task: AgentTask) -> AgentResponse:
            order.append("B")
            return AgentResponse(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                confidence=0.85,
                result={"tool": "RAG", "answer": "rag-part", "sources": []},
                summary="rag-part",
            )

    for cls in (A, B):
        agent = cls()
        agent.initialize()
        registry.register(agent)

    supervisor = SupervisorAgent(registry=registry, enable_parallel=True, max_workers=2)
    plan = ExecutionPlan(
        goal="compare and explain",
        steps=[
            ExecutionStep(step_number=1, tool="SQL", action="query", description="metrics"),
            ExecutionStep(step_number=2, tool="RAG", action="retrieve", description="sop"),
        ],
        required_capabilities=[
            AgentCapability.DATABASE_QUERY.value,
            AgentCapability.DOCUMENT_RETRIEVAL.value,
        ],
        confidence=0.95,
    )
    result = supervisor.execute_plan(plan, question="Compare pickups and explain SOP")
    assert set(order) == {"A", "B"}
    assert len(result.step_results) == 2
    assert result.response.answer  # merged


def test_unhealthy_agent_skipped(enable_multi_agent: None) -> None:
    registry = AgentRegistry()

    class Bad(BaseAgent):
        name = "Bad"
        capabilities = [AgentCapability.DATABASE_QUERY]

        def execute(self, task: AgentTask) -> AgentResponse:
            return AgentResponse(agent=self.name, status=AgentStatus.SUCCESS, result={})

    class Good(BaseAgent):
        name = "Good"
        capabilities = [AgentCapability.DATABASE_QUERY]

        def execute(self, task: AgentTask) -> AgentResponse:
            return AgentResponse(
                agent=self.name,
                status=AgentStatus.SUCCESS,
                confidence=0.9,
                result={"tool": "SQL", "answer": "from-good"},
                summary="from-good",
            )

    bad = Bad()
    bad.initialize()
    bad.mark_unhealthy("broken")
    # Force unhealthy health_check
    bad._healthy = False
    bad._total_calls = 10
    bad._success_calls = 0
    registry.register(bad)

    good = Good()
    good.initialize()
    registry.register(good)

    selected = registry.select_agent(AgentCapability.DATABASE_QUERY.value, healthy_only=True)
    assert selected is not None
    assert selected.name == "Good"


def test_plan_executor_delegates_to_supervisor(enable_multi_agent: None, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_execute_plan(self, plan, **kwargs):  # noqa: ANN001
        called["yes"] = True
        return ExecutionResult(
            response=MagicMock(answer="via-supervisor", capability="multi_agent"),
            step_results={},
            plan=plan,
        )

    monkeypatch.setattr(SupervisorAgent, "execute_plan", fake_execute_plan)
    plan = ExecutionPlan(
        goal="t",
        steps=[ExecutionStep(step_number=1, tool="LLM", action="greet", description="g")],
        confidence=0.8,
    )
    # Default registry → multi-agent path
    result = PlanExecutor().execute(plan, question="hello")
    assert called.get("yes") is True
    assert result.response.answer == "via-supervisor"


def test_planner_attaches_required_capabilities() -> None:
    from core.planner import _finalize_plan

    plan = ExecutionPlan(
        goal="sop",
        steps=[
            ExecutionStep(step_number=1, tool="RAG", action="retrieve", description="r"),
            ExecutionStep(step_number=2, tool="LLM", action="answer", description="a", depends_on=[1]),
        ],
        confidence=0.9,
    )
    finalized = _finalize_plan(plan)
    assert "document_retrieval" in finalized.required_capabilities
    assert "conversation" in finalized.required_capabilities
