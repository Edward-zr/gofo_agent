"""Tests for conversation repair detection."""

from __future__ import annotations

from core.models import QueryResponse
from tools.memory.conversation import ConversationMemory
from tools.memory.repair import detect_repair
from tools.memory.resolver import resolve


def _remember(memory: ConversationMemory, question: str, sql: str | None = None) -> None:
    memory.add_turn(
        user_question=question,
        resolved_question=question,
        response=QueryResponse(
            question=question,
            answer="ok",
            capability="sql",
            planning_intent="test",
            generated_sql=sql,
            sql_rows=[{"result": "ok"}],
        ),
        extracted_entities={},
    )


def test_customer_ranking_correction_to_hubs_uses_hub_dimension() -> None:
    memory = ConversationMemory()
    _remember(memory, "rank customers by pickup performance")

    repair = detect_repair("I mean hubs", memory)
    resolved = resolve("I mean hubs", memory)

    assert repair["is_repair"] is True
    assert repair["changed_dimension"] == "hub"
    assert "hubs" in resolved
    assert "customer" not in resolved.lower()


def test_all_hubs_correction_removes_single_hub_filter() -> None:
    memory = ConversationMemory()
    _remember(memory, "Show Atlanta hub performance")

    resolved = resolve("No all hubs", memory)

    assert "all hubs" in resolved.lower()
    assert "atlanta" not in resolved.lower()


def test_lowest_correction_changes_ranking_direction() -> None:
    memory = ConversationMemory()
    _remember(memory, "highest performing driver")

    resolved = resolve("I mean lowest", memory)

    assert "lowest" in resolved.lower()
    assert "highest" not in resolved.lower()


def test_last_month_correction_changes_date_filter() -> None:
    memory = ConversationMemory()
    _remember(memory, "show pickups today")

    resolved = resolve("actually last month", memory)

    assert "last month" in resolved.lower()
    assert "today" not in resolved.lower()
