"""Unit tests for Knowledge Graph relationship lookups."""

from __future__ import annotations

from tools.knowledge_graph.service import answer


def test_driver_john_alias_resolves_region() -> None:
    result = answer("Which region does Driver John belong to?")
    assert "Midwest" in result["answer"]
    assert result["facts"]
    assert result["facts"][0]["hub"] == "Chicago Hub"


def test_manager_for_hub_from_sql_context() -> None:
    result = answer(
        "Who manages the hub with the lowest pickup rate?",
        sql_rows=[{"hub": "Chicago Hub", "pickup_rate": 0.71}],
    )
    assert "Maria Chen" in result["answer"]
    assert result["facts"][0]["type"] == "manager_hub"


def test_sop_ownership() -> None:
    result = answer("Who owns the pickup SOP?")
    assert "Ops Excellence" in result["answer"]
