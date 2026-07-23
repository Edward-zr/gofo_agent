"""Tests for Schema Registry + Schema Retriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tools.sql.planner import plan
from tools.sql.schema_registry import (
    get_schema_registry,
    refresh_schema_registry,
    reset_schema_registry_for_tests,
)
from tools.sql.schema_retriever import SchemaRetriever, retrieve_schema
from tools.sql.validator import validate_sql


def setup_function() -> None:
    reset_schema_registry_for_tests()


def test_schema_registry_loads_tables_columns_and_fks() -> None:
    snapshot = get_schema_registry().get_snapshot()
    assert "pickups" in snapshot.tables
    assert "drivers" in snapshot.tables
    pickups = snapshot.tables["pickups"]
    assert "pickup_date" in pickups.column_names()
    assert pickups.primary_keys
    assert any(fk.to_table == "drivers" for fk in pickups.foreign_keys)
    assert snapshot.fingerprint
    assert pickups.description


def test_schema_registry_refresh_updates_fingerprint() -> None:
    first = refresh_schema_registry()
    second = refresh_schema_registry()
    assert first.fingerprint == second.fingerprint
    assert set(first.table_names()) == set(second.table_names())


def test_retriever_pickup_rate_uses_pickups_only() -> None:
    result = retrieve_schema("What is today's pickup rate?")
    assert result.tables == ["pickups"]
    assert "pickup_date" in result.columns["pickups"]
    assert "status" in result.columns["pickups"]
    assert result.full_schema_requested is False
    # Should not dump unrelated tables.
    assert "customers" not in result.tables
    assert "addresses" not in result.tables


def test_retriever_hub_ranking_includes_drivers_and_join() -> None:
    result = retrieve_schema("Rank hubs by completed pickups")
    assert "pickups" in result.tables
    assert "drivers" in result.tables
    assert any("drivers.driver_id" in rel for rel in result.relationships)
    assert "hub" in result.columns["drivers"]


def test_retriever_failure_reason_includes_exceptions() -> None:
    result = retrieve_schema("Why did pickups fail yesterday? Show failure reasons.")
    assert "exceptions" in result.tables
    assert "pickups" in result.tables
    assert any("exceptions.pickup_id" in rel for rel in result.relationships)


def test_retriever_never_returns_full_schema_by_default() -> None:
    result = retrieve_schema("Top 5 drivers this week")
    registry_tables = set(get_schema_registry().get_snapshot().table_names())
    assert set(result.tables) < registry_tables or len(result.tables) < len(registry_tables)


def test_retriever_full_schema_when_explicitly_requested() -> None:
    result = retrieve_schema("Show me the full schema for all tables")
    assert result.full_schema_requested is True
    assert set(result.tables) == set(get_schema_registry().get_snapshot().table_names())


def test_retrieved_schema_prompt_excludes_unselected_tables() -> None:
    snapshot = get_schema_registry().get_snapshot()
    result = SchemaRetriever().retrieve("How many pickups today?")
    prompt = result.to_prompt_text(snapshot)
    assert "TABLE pickups:" in prompt
    assert "TABLE customers:" not in prompt
    assert "Never invent" in prompt


@patch("tools.sql.planner.get_llm")
def test_plan_prompt_uses_only_retrieved_schema(mock_get_llm: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.content = "SELECT COUNT(*) FROM pickups WHERE pickup_date = DATE('now');"
    mock_get_llm.return_value.invoke.return_value = mock_response

    plan("How many pickups today?")

    system_prompt = mock_get_llm.return_value.invoke.call_args.args[0][0].content
    assert "Retrieved schema" in system_prompt
    assert "TABLE pickups:" in system_prompt
    # Customer table should not appear for a simple pickup count.
    assert "TABLE customers:" not in system_prompt


def test_validator_rejects_unknown_table() -> None:
    sql = validate_sql("SELECT * FROM spaceships;")
    assert sql == "SELECT 'UNKNOWN';"


def test_validator_accepts_known_join() -> None:
    sql = (
        "SELECT d.hub, COUNT(*) AS n "
        "FROM pickups p JOIN drivers d ON p.driver_id = d.driver_id "
        "GROUP BY d.hub;"
    )
    assert validate_sql(sql) == sql
