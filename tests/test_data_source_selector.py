"""Unit tests for Data Source Selection and source-aware Planner plans."""

from __future__ import annotations

from core.data_source_selector import DataSource, DataSourceSelector
from core.intent_classifier import IntentClassification, IntentType
from core.planner import Planner, ToolName


def _classification(intent: IntentType, **overrides) -> IntentClassification:
    defaults = {
        IntentType.Greeting: dict(confidence=0.99),
        IntentType.SOP_QA: dict(confidence=0.92, requires_rag=True),
        IntentType.SQL_Query: dict(confidence=0.9, requires_sql=True),
        IntentType.SQL_Analysis: dict(
            confidence=0.94, requires_sql=True, requires_planner=True
        ),
        IntentType.Dashboard: dict(confidence=0.93, requires_sql=True, requires_planner=True),
        IntentType.Follow_Up: dict(confidence=0.93, requires_memory=True, requires_planner=True),
        IntentType.Unknown: dict(confidence=0.2, requires_clarification=True),
    }.get(intent, dict(confidence=0.9))
    defaults.update(overrides)
    return IntentClassification(intent=intent, reasoning="test", **defaults)


def _evidence_tools(plan) -> list[str]:
    return [step.tool for step in plan.steps if step.tool != ToolName.LLM]


def test_today_pickup_rate_uses_sql_only() -> None:
    question = "What is today's pickup rate?"
    selection = DataSourceSelector().select(question, _classification(IntentType.SQL_Query))
    assert DataSource.SQL.value in selection.selected_sources
    assert DataSource.RAG.value not in selection.selected_sources
    assert DataSource.KNOWLEDGE_GRAPH.value not in selection.selected_sources
    assert DataSource.PYTHON.value not in selection.selected_sources

    plan = Planner().plan(question, _classification(IntentType.SQL_Query))
    assert _evidence_tools(plan) == [ToolName.SQL]
    assert plan.selected_data_sources
    assert "SQL" in plan.source_reasons
    assert plan.expected_outputs
    assert plan.steps[-1].tool == ToolName.LLM


def test_driver_region_uses_knowledge_graph_only() -> None:
    question = "Which region does Driver John belong to?"
    selection = DataSourceSelector().select(question, _classification(IntentType.SQL_Query))
    assert selection.selected_sources.count(DataSource.KNOWLEDGE_GRAPH.value) == 1
    assert DataSource.SQL.value not in selection.selected_sources
    assert DataSource.RAG.value not in selection.selected_sources

    plan = Planner().plan(question, _classification(IntentType.SQL_Query))
    assert _evidence_tools(plan) == [ToolName.KNOWLEDGE_GRAPH]
    assert plan.source_reasons[ToolName.KNOWLEDGE_GRAPH]


def test_explain_sop_uses_rag_only() -> None:
    question = "Explain the pickup SOP."
    selection = DataSourceSelector().select(question, _classification(IntentType.SOP_QA))
    assert DataSource.RAG.value in selection.selected_sources
    assert DataSource.SQL.value not in selection.selected_sources
    assert DataSource.PYTHON.value not in selection.selected_sources

    plan = Planner().plan(question, _classification(IntentType.SOP_QA))
    assert _evidence_tools(plan) == [ToolName.RAG]


def test_compare_rate_and_explain_uses_sql_plus_rag() -> None:
    question = "Compare Chicago's pickup rate with yesterday and explain possible reasons."
    selection = DataSourceSelector().select(question, _classification(IntentType.SQL_Analysis))
    sources = set(selection.selected_sources) - {DataSource.LLM.value}
    assert sources == {DataSource.SQL.value, DataSource.RAG.value}

    plan = Planner().plan(question, _classification(IntentType.SQL_Analysis))
    assert set(_evidence_tools(plan)) == {ToolName.SQL, ToolName.RAG}
    assert ToolName.PYTHON not in _evidence_tools(plan)


def test_lowest_rate_manager_uses_sql_plus_kg() -> None:
    question = "Who manages the hub with the lowest pickup rate?"
    selection = DataSourceSelector().select(question, _classification(IntentType.SQL_Analysis))
    sources = set(selection.selected_sources) - {DataSource.LLM.value}
    assert sources == {DataSource.SQL.value, DataSource.KNOWLEDGE_GRAPH.value}

    plan = Planner().plan(question, _classification(IntentType.SQL_Analysis))
    tools = _evidence_tools(plan)
    assert tools == [ToolName.SQL, ToolName.KNOWLEDGE_GRAPH]
    # KG depends on SQL so manager can be resolved for the extreme hub.
    kg_step = next(step for step in plan.steps if step.tool == ToolName.KNOWLEDGE_GRAPH)
    assert kg_step.depends_on == [1]


def test_midwest_trend_chart_uses_sql_python() -> None:
    question = "Compare Midwest performance and generate a trend chart."
    selection = DataSourceSelector().select(question, _classification(IntentType.SQL_Analysis))
    sources = set(selection.selected_sources) - {DataSource.LLM.value}
    assert DataSource.SQL.value in sources
    assert DataSource.TRANSFORM.value in sources
    assert DataSource.STATISTICS.value in sources
    assert DataSource.VISUALIZATION.value in sources
    assert DataSource.RAG.value not in sources
    assert DataSource.KNOWLEDGE_GRAPH.value not in sources

    plan = Planner().plan(question, _classification(IntentType.SQL_Analysis))
    tools = set(_evidence_tools(plan))
    assert tools >= {ToolName.SQL, ToolName.TRANSFORM, ToolName.STATISTICS, ToolName.VISUALIZATION}


def test_execution_plan_includes_selection_metadata() -> None:
    plan = Planner().plan(
        "What is today's pickup rate?",
        _classification(IntentType.SQL_Query),
    )
    assert plan.data_source_selection is not None
    assert plan.selected_data_sources
    assert plan.source_reasons
    assert plan.expected_outputs
    assert plan.estimated_tool_calls == len(plan.steps)
