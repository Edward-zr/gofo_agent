"""Unit tests for specialized Python analytics tools."""

from __future__ import annotations

import pandas as pd

from core.data_source_selector import DataSource, DataSourceSelector
from core.intent_classifier import IntentClassification, IntentType
from core.planner import Planner, ToolName
from tools.python.recommendation_tool import run_recommendations
from tools.python.statistics_tool import run_statistics
from tools.python.transformation_tool import run_transformation
from tools.python.visualization_tool import run_visualization


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"hub": "Chicago Hub", "pickup_count": 10, "status": "Completed", "week": "w1"},
            {"hub": "Chicago Hub", "pickup_count": 8, "status": "Failed", "week": "w1"},
            {"hub": "Dallas Hub", "pickup_count": 20, "status": "Completed", "week": "w2"},
            {"hub": "Dallas Hub", "pickup_count": 4, "status": "Delayed", "week": "w2"},
        ]
    )


def test_transformation_groupby_and_shape() -> None:
    result = run_transformation(_frame(), question="Compare performance by hub")
    assert result["tool"] == "TRANSFORM"
    assert result["shape"]["rows"] >= 1
    assert "groupby" in " ".join(result["functions_executed"]) or "identity" in " ".join(
        result["functions_executed"]
    )
    assert isinstance(result["transformed_dataframe"], pd.DataFrame)


def test_statistics_computes_summary_and_ranking() -> None:
    result = run_statistics(_frame(), question="Rank hubs by pickup_count")
    assert result["tool"] == "STATISTICS"
    assert "summary" in result["statistics"] or "numeric_means" in result["statistics"]
    assert "ranking" in result["statistics"]
    assert result["execution_time_ms"] >= 0


def test_statistics_correlation_when_requested() -> None:
    frame = _frame()
    frame["packages"] = [5, 3, 9, 2]
    result = run_statistics(frame, question="Show correlation between metrics", include_correlation=True)
    assert "correlation" in result["statistics"]


def test_visualization_returns_chart_metadata() -> None:
    result = run_visualization(_frame(), question="Create a bar chart of hub performance")
    assert result["tool"] == "VISUALIZATION"
    assert "chart_metadata" in result
    assert isinstance(result["charts"], list)


def test_recommendation_from_statistics() -> None:
    stats = run_statistics(_frame(), question="Worst hub performance")["statistics"]
    result = run_recommendations(question="What should we do about low-performing hubs?", statistics=stats)
    assert result["recommendations"]
    assert result["tool"] == "RECOMMENDATION"


def test_selector_chart_question_uses_specialized_python_tools() -> None:
    classification = IntentClassification(
        intent=IntentType.SQL_Analysis,
        confidence=0.93,
        reasoning="test",
        requires_sql=True,
        requires_planner=True,
    )
    selection = DataSourceSelector().select(
        "Compare Midwest performance and generate a trend chart.",
        classification,
    )
    sources = set(selection.selected_sources)
    assert DataSource.SQL.value in sources
    assert DataSource.TRANSFORM.value in sources
    assert DataSource.STATISTICS.value in sources
    assert DataSource.VISUALIZATION.value in sources
    assert DataSource.PYTHON.value not in sources  # specialized tools, not monolith


def test_planner_emits_transform_statistics_visualization_order() -> None:
    classification = IntentClassification(
        intent=IntentType.SQL_Analysis,
        confidence=0.93,
        reasoning="test",
        requires_sql=True,
        requires_planner=True,
    )
    plan = Planner().plan(
        "Compare Midwest performance and generate a trend chart.",
        classification,
    )
    tools = [step.tool for step in plan.steps if step.tool != ToolName.LLM]
    assert ToolName.SQL in tools
    assert ToolName.TRANSFORM in tools
    assert ToolName.STATISTICS in tools
    assert ToolName.VISUALIZATION in tools
    # Dependency: transform after sql, stats after transform
    transform_step = next(step for step in plan.steps if step.tool == ToolName.TRANSFORM)
    stats_step = next(step for step in plan.steps if step.tool == ToolName.STATISTICS)
    assert transform_step.depends_on
    assert stats_step.depends_on
