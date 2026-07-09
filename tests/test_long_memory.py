"""Tests for SQLite-backed long-term memory."""

from __future__ import annotations

from pathlib import Path

from tools.memory.database import get_connection
from tools.memory.learning import detect_repeated_patterns
from tools.memory.long_memory import LongTermMemory


def test_save_conversation_restart_memory_retrieve_conversation(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.db"
    memory = LongTermMemory(database_path=database_path)

    saved = memory.save_conversation(
        user_question="What is CBT?",
        resolved_question="What is CBT?",
        answer="Collection by TikTok.",
        intent="explain_cbt",
        metric=None,
        dimension=None,
        sql_query=None,
    )
    restarted = LongTermMemory(database_path=database_path)
    matches = restarted.search_similar_history("CBT definition")

    assert saved is True
    assert matches
    assert matches[0]["user_question"] == "What is CBT?"
    assert matches[0]["answer"] == "Collection by TikTok."


def test_search_similar_issue_retrieves_previous_chicago_issue(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.db"
    memory = LongTermMemory(database_path=database_path)
    memory.save_finding(
        hub="Chicago",
        issue="High delayed pickups",
        metric="delay_rate",
        root_cause="Driver shortage",
        recommendation="Review driver allocation",
        severity="high",
    )

    matches = LongTermMemory(database_path=database_path).search_similar_issue(
        "Why Chicago bad again?",
        metric="delay_rate",
        dimension="hub",
    )

    assert matches
    assert matches[0]["hub"] == "Chicago"
    assert matches[0]["root_cause"] == "Driver shortage"


def test_repeated_failures_create_learned_pattern(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.db"
    memory = LongTermMemory(database_path=database_path)
    for _ in range(3):
        memory.save_finding(
            driver="Driver X",
            issue="Repeated failed pickups",
            metric="failure_rate",
            root_cause="Driver missed pickup windows",
            recommendation="Review driver performance",
            severity="high",
        )

    created = detect_repeated_patterns(memory, database_path=database_path)
    patterns = memory.list_patterns()

    assert created
    assert patterns
    assert "Driver X recurring failure_rate" == patterns[0]["pattern_name"]


def test_random_hello_does_not_store(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.db"
    memory = LongTermMemory(database_path=database_path)

    saved = memory.save_conversation(user_question="hello")

    with get_connection(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM conversation_history;").fetchone()[0]

    assert saved is False
    assert count == 0
