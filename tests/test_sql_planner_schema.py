"""Schema-awareness tests for tools.sql.planner."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from tools.sql.planner import plan

KNOWN_TABLES = {"pickups", "drivers", "customers", "addresses", "exceptions"}
TABLE_REFERENCE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _referenced_tables(sql: str) -> set[str]:
    return {match.group(1).lower() for match in TABLE_REFERENCE_RE.finditer(sql)}


def _mock_planner_response(mock_get_llm: MagicMock, sql: str) -> None:
    mock_response = MagicMock()
    mock_response.content = sql
    mock_get_llm.return_value.invoke.return_value = mock_response


def _assert_only_known_tables(sql: str) -> None:
    assert _referenced_tables(sql).issubset(KNOWN_TABLES)


@patch("tools.sql.planner.get_llm")
def test_today_details_uses_pickups_drivers_and_customers(mock_get_llm: MagicMock) -> None:
    expected_sql = (
        "SELECT\n"
        "    p.pickup_id,\n"
        "    p.pickup_date,\n"
        "    p.status,\n"
        "    p.package_count,\n"
        "    d.driver_name,\n"
        "    d.hub,\n"
        "    c.customer_name\n"
        "FROM pickups p\n"
        "LEFT JOIN drivers d\n"
        "ON p.driver_id=d.driver_id\n"
        "LEFT JOIN customers c\n"
        "ON p.customer_id=c.customer_id\n"
        "WHERE p.pickup_date='2026-06-29';"
    )
    _mock_planner_response(mock_get_llm, expected_sql)

    sql = plan("give me pickup details for 2026-06-29")

    assert sql == expected_sql
    assert "p.pickup_date='2026-06-29'" in sql
    assert "LEFT JOIN drivers" in sql
    assert "LEFT JOIN customers" in sql
    _assert_only_known_tables(sql)


@patch("tools.sql.planner.get_llm")
def test_driver_ranking_joins_drivers(mock_get_llm: MagicMock) -> None:
    expected_sql = (
        "SELECT d.driver_name,\n"
        "       COUNT(*) pickup_count\n"
        "FROM pickups p\n"
        "JOIN drivers d\n"
        "ON p.driver_id=d.driver_id\n"
        "WHERE p.status='Completed'\n"
        "GROUP BY d.driver_name\n"
        "ORDER BY pickup_count DESC\n"
        "LIMIT 1;"
    )
    _mock_planner_response(mock_get_llm, expected_sql)

    sql = plan("Highest performing driver")

    assert "JOIN drivers" in sql
    assert "p.status='Completed'" in sql
    assert "ORDER BY pickup_count DESC" in sql
    _assert_only_known_tables(sql)


@patch("tools.sql.planner.get_llm")
def test_hub_performance_uses_drivers_hub(mock_get_llm: MagicMock) -> None:
    expected_sql = (
        "SELECT d.hub,\n"
        "       COUNT(*) AS pickup_count,\n"
        "       SUM(CASE WHEN p.status='Completed' THEN 1 ELSE 0 END) AS completed_pickups\n"
        "FROM pickups p\n"
        "JOIN drivers d\n"
        "ON p.driver_id=d.driver_id\n"
        "GROUP BY d.hub\n"
        "ORDER BY pickup_count DESC;"
    )
    _mock_planner_response(mock_get_llm, expected_sql)

    sql = plan("hub performance")

    assert "JOIN drivers" in sql
    assert "d.hub" in sql
    assert "GROUP BY d.hub" in sql
    _assert_only_known_tables(sql)


@patch("tools.sql.planner.get_llm")
def test_customer_volume_joins_customers(mock_get_llm: MagicMock) -> None:
    expected_sql = (
        "SELECT c.customer_name,\n"
        "       COUNT(*) AS pickup_count\n"
        "FROM pickups p\n"
        "JOIN customers c\n"
        "ON p.customer_id=c.customer_id\n"
        "GROUP BY c.customer_name\n"
        "ORDER BY pickup_count DESC;"
    )
    _mock_planner_response(mock_get_llm, expected_sql)

    sql = plan("customer volume")

    assert "JOIN customers" in sql
    assert "c.customer_name" in sql
    assert "ORDER BY pickup_count DESC" in sql
    _assert_only_known_tables(sql)


@patch("tools.sql.planner.get_llm")
def test_failed_pickups_can_join_exceptions_for_reasons(mock_get_llm: MagicMock) -> None:
    expected_sql = (
        "SELECT p.pickup_id,\n"
        "       p.pickup_date,\n"
        "       p.status,\n"
        "       e.reason\n"
        "FROM pickups p\n"
        "LEFT JOIN exceptions e\n"
        "ON p.pickup_id=e.pickup_id\n"
        "WHERE p.status='Failed';"
    )
    _mock_planner_response(mock_get_llm, expected_sql)

    sql = plan("failed pickup reasons")

    assert "LEFT JOIN exceptions" in sql
    assert "p.status='Failed'" in sql
    assert "e.reason" in sql
    _assert_only_known_tables(sql)


@patch("tools.sql.planner.get_llm")
def test_unknown_table_references_are_not_returned(mock_get_llm: MagicMock) -> None:
    _mock_planner_response(mock_get_llm, "SELECT * FROM hubs;")

    sql = plan("hub performance")

    assert sql == "SELECT 'UNKNOWN';"


@patch("tools.sql.planner.get_llm")
def test_schema_rules_are_in_system_prompt(mock_get_llm: MagicMock) -> None:
    _mock_planner_response(mock_get_llm, "SELECT COUNT(*) FROM pickups;")

    plan("customer volume")

    messages = mock_get_llm.return_value.invoke.call_args.args[0]
    system_prompt = messages[0].content

    assert "TABLE pickups:" in system_prompt
    assert "Each row = one pickup task." in system_prompt
    assert "Always use pickup_date for operational dates." in system_prompt
    assert "For hub or warehouse questions, JOIN drivers and use drivers.hub." in system_prompt
    assert "For details questions, return actual rows, not only COUNT." in system_prompt
    assert "Conversation context is NOT a SQL filter." in system_prompt
    assert "Global driver ranking must GROUP BY driver" in system_prompt
