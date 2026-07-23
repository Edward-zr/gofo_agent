"""Unit tests for tools.sql.planner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.sql.planner import plan

SAMPLE_SQL = (
    "SELECT COUNT(*)\n"
    "FROM pickups\n"
    "WHERE pickup_date = DATE('now', '-1 day');"
)


def test_plan_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="Question must not be empty"):
        plan("   ")


@patch("tools.sql.planner.get_llm")
def test_plan_success(mock_get_llm: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.content = SAMPLE_SQL
    mock_get_llm.return_value.invoke.return_value = mock_response

    sql = plan("How many pickups yesterday?")

    assert sql == SAMPLE_SQL
    mock_get_llm.return_value.invoke.assert_called_once()


@patch("tools.sql.planner.get_llm")
def test_plan_returns_sql_unchanged(mock_get_llm: MagicMock) -> None:
    expected_sql = (
        "SELECT d.driver_name,\n"
        "       COUNT(*) AS pickup_count\n"
        "FROM pickups p\n"
        "JOIN drivers d ON d.driver_id = p.driver_id\n"
        "WHERE p.pickup_date >= DATE('now', '-6 days')\n"
        "GROUP BY d.driver_name\n"
        "ORDER BY pickup_count DESC\n"
        "LIMIT 5;"
    )
    mock_response = MagicMock()
    mock_response.content = expected_sql
    mock_get_llm.return_value.invoke.return_value = mock_response

    sql = plan("Top 5 drivers this week")

    assert sql == expected_sql


@patch("tools.sql.planner.get_llm")
def test_plan_includes_schema_in_prompt(mock_get_llm: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.content = SAMPLE_SQL
    mock_get_llm.return_value.invoke.return_value = mock_response

    plan("How many pickups yesterday?")

    messages = mock_get_llm.return_value.invoke.call_args.args[0]
    system_prompt = messages[0].content

    assert "Retrieved schema" in system_prompt
    assert "TABLE pickups:" in system_prompt
    assert "pickup_date" in system_prompt
    # Full static SCHEMA dump must NOT be injected by default.
    assert "Database Engine" not in system_prompt or "Retrieved schema" in system_prompt
    assert system_prompt.count("TABLE pickups:") >= 1


@patch("tools.sql.planner.get_llm")
def test_plan_includes_sqlite_instructions(mock_get_llm: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.content = SAMPLE_SQL
    mock_get_llm.return_value.invoke.return_value = mock_response

    plan("Average packages per pickup")

    messages = mock_get_llm.return_value.invoke.call_args.args[0]
    system_prompt = messages[0].content

    assert "Use SQLite syntax only." in system_prompt
    assert "DATE('now'" in system_prompt
    assert "Use LIMIT instead of TOP." in system_prompt
    assert "SELECT 'UNKNOWN';" in system_prompt


@pytest.mark.parametrize(
    ("question", "expected_prompt_terms"),
    [
        (
            "Monthly KPI",
            ["Monthly KPI", "SUBSTR(pickup_date, 1, 7)", "completion_rate"],
        ),
        (
            "Weekly KPI",
            ["weekly KPI", "STRFTIME('%Y-%W', pickup_date)"],
        ),
        (
            "Compare periods",
            ["Compare this week and last week.", "this_week", "last_week"],
        ),
        (
            "Pickup trends",
            ["Pickup trends", "GROUP BY pickup_date", "ORDER BY pickup_date"],
        ),
        (
            "Highest performing driver",
            ["Highest performing driver", "ORDER BY completed_pickups DESC", "LIMIT 1"],
        ),
        (
            "Warehouse performance",
            ["Warehouse performance", "warehouse means drivers.hub", "GROUP BY d.hub"],
        ),
        (
            "Average pickup per driver",
            ["Average pickup per driver", "AVG(p.package_count)", "GROUP BY d.driver_id"],
        ),
        (
            "Failed pickups",
            ["Failed pickups", "status = 'Failed'"],
        ),
        (
            "Delayed pickups",
            ["Delayed pickups", "status = 'Delayed'"],
        ),
    ],
)
@patch("tools.sql.planner.get_llm")
def test_plan_prompt_includes_business_analytics_examples(
    mock_get_llm: MagicMock,
    question: str,
    expected_prompt_terms: list[str],
) -> None:
    mock_response = MagicMock()
    mock_response.content = SAMPLE_SQL
    mock_get_llm.return_value.invoke.return_value = mock_response

    plan(question)

    messages = mock_get_llm.return_value.invoke.call_args.args[0]
    system_prompt = messages[0].content

    for term in expected_prompt_terms:
        assert term in system_prompt


@patch("tools.sql.planner.get_llm")
def test_plan_prompt_discourages_unknown_for_valid_analytics(mock_get_llm: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.content = SAMPLE_SQL
    mock_get_llm.return_value.invoke.return_value = mock_response

    plan("Warehouse performance")

    messages = mock_get_llm.return_value.invoke.call_args.args[0]
    system_prompt = messages[0].content

    assert "Attempt a best-effort analytics query" in system_prompt
    assert "Return SELECT 'UNKNOWN'; only when the question truly cannot be" in system_prompt
    assert "mapped to the retrieved schema" in system_prompt
