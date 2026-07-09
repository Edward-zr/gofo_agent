"""Persistent long-term memory for GOFO operational intelligence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

from core.logger import get_logger
from tools.memory.database import get_connection

logger = get_logger("memory")

_STOPWORDS = {
    "a",
    "again",
    "and",
    "are",
    "bad",
    "did",
    "for",
    "how",
    "is",
    "me",
    "of",
    "the",
    "today",
    "what",
    "why",
}

_ISSUE_KEYWORDS = {
    "delay": ("delay", "delayed", "slow", "late"),
    "failure": ("fail", "failed", "failure"),
    "completion": ("completion", "performance", "bad", "drop", "decrease"),
    "volume": ("volume", "spike", "package", "packages"),
    "driver": ("driver", "capacity", "shortage"),
}


class LongTermMemory:
    """SQLite-backed memory for conversations, findings, and learned patterns."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path
        with get_connection(database_path):
            pass

    def save_conversation(
        self,
        *,
        user_question: str,
        resolved_question: str | None = None,
        answer: str | None = None,
        intent: str | None = None,
        metric: str | None = None,
        dimension: str | None = None,
        sql_query: str | None = None,
    ) -> bool:
        """Persist an important conversation turn."""
        if not _should_store_conversation(user_question, intent, sql_query):
            logger.info("Conversation not stored")
            return False
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO conversation_history (
                    timestamp,
                    user_question,
                    resolved_question,
                    answer,
                    intent,
                    metric,
                    dimension,
                    sql_query
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    _now(),
                    user_question,
                    resolved_question,
                    answer,
                    intent,
                    metric,
                    dimension,
                    sql_query,
                ),
            )
            connection.commit()
        logger.info("Conversation stored")
        return True

    def save_finding(
        self,
        *,
        hub: str | None = None,
        driver: str | None = None,
        customer: str | None = None,
        issue: str | None = None,
        metric: str | None = None,
        root_cause: str | None = None,
        recommendation: str | None = None,
        severity: str | None = None,
    ) -> bool:
        """Persist an operational finding worth remembering."""
        if not _is_important_finding(issue, metric, root_cause):
            logger.info("Finding not stored")
            return False
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO operational_findings (
                    created_at,
                    hub,
                    driver,
                    customer,
                    issue,
                    metric,
                    root_cause,
                    recommendation,
                    severity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    _now(),
                    hub,
                    driver,
                    customer,
                    issue,
                    metric,
                    root_cause,
                    recommendation,
                    severity or _severity(issue, metric),
                ),
            )
            connection.commit()
        logger.info("Finding stored")
        return True

    def save_pattern(
        self,
        *,
        pattern_name: str,
        description: str | None = None,
        trigger_condition: str | None = None,
        previous_solution: str | None = None,
    ) -> bool:
        """Persist a learned recurring operational pattern."""
        if not pattern_name.strip():
            return False
        if self._pattern_exists(pattern_name):
            logger.info("Learned pattern already exists")
            return False
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO learned_patterns (
                    created_at,
                    pattern_name,
                    description,
                    trigger_condition,
                    previous_solution
                )
                VALUES (?, ?, ?, ?, ?);
                """,
                (_now(), pattern_name, description, trigger_condition, previous_solution),
            )
            connection.commit()
        logger.info("Learned pattern stored")
        return True

    def search_similar_history(
        self,
        question: str,
        *,
        metric: str | None = None,
        dimension: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find similar stored conversations by keyword, metric, and dimension."""
        tokens = _tokens(question)
        with get_connection(self.database_path) as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT *
                    FROM conversation_history
                    ORDER BY id DESC
                    LIMIT 100;
                    """
                )
            ]
        scored = []
        for row in rows:
            score = _score_text(tokens, row.get("user_question"), row.get("resolved_question"), row.get("answer"))
            if metric and row.get("metric") == metric:
                score += 3
            if dimension and row.get("dimension") == dimension:
                score += 3
            if score > 0:
                row["score"] = score
                scored.append(row)
        matches = sorted(scored, key=lambda row: row["score"], reverse=True)[:limit]
        logger.info("History search returned %s matches", len(matches))
        return matches

    def search_similar_issue(
        self,
        question: str,
        *,
        metric: str | None = None,
        dimension: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find similar operational issues using entity and keyword matching."""
        tokens = _tokens(question)
        issue_tags = _issue_tags(question)
        with get_connection(self.database_path) as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT *
                    FROM operational_findings
                    ORDER BY id DESC
                    LIMIT 100;
                    """
                )
            ]
        scored = []
        for row in rows:
            score = _score_text(
                tokens,
                row.get("hub"),
                row.get("driver"),
                row.get("customer"),
                row.get("issue"),
                row.get("root_cause"),
                row.get("recommendation"),
            )
            if metric and row.get("metric") == metric:
                score += 3
            if dimension and row.get(dimension):
                score += 3
            if issue_tags and issue_tags.intersection(_issue_tags(" ".join(str(row.get(k) or "") for k in ("issue", "root_cause", "metric")))):
                score += 3
            if score > 0:
                row["score"] = score
                scored.append(row)
        matches = sorted(scored, key=lambda row: row["score"], reverse=True)[:limit]
        logger.info("Issue search returned %s matches", len(matches))
        return matches

    def list_patterns(self) -> list[dict[str, Any]]:
        """Return all learned patterns, newest first."""
        with get_connection(self.database_path) as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM learned_patterns ORDER BY id DESC;"
                )
            ]

    def _pattern_exists(self, pattern_name: str) -> bool:
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT id FROM learned_patterns WHERE pattern_name = ? LIMIT 1;",
                (pattern_name,),
            ).fetchone()
        return row is not None


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _score_text(tokens: set[str], *values: Any) -> int:
    haystack = " ".join(str(value or "").lower() for value in values)
    haystack_tokens = _tokens(haystack)
    return len(tokens.intersection(haystack_tokens))


def _issue_tags(text: str) -> set[str]:
    normalized = text.lower()
    return {
        tag
        for tag, keywords in _ISSUE_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    }


def _should_store_conversation(
    question: str,
    intent: str | None,
    sql_query: str | None,
) -> bool:
    normalized = question.strip().lower()
    if normalized in {"hello", "hi", "hey", "thanks", "thank you"}:
        return False
    if intent == "greeting":
        return False
    return bool(sql_query or intent or _issue_tags(normalized) or "sop" in normalized)


def _is_important_finding(
    issue: str | None,
    metric: str | None,
    root_cause: str | None,
) -> bool:
    text = " ".join(value or "" for value in (issue, metric, root_cause)).lower()
    triggers = (
        "low completion",
        "high delay",
        "delay",
        "abnormal",
        "failed",
        "failure",
        "driver",
        "hub",
        "volume",
    )
    return any(trigger in text for trigger in triggers)


def _severity(issue: str | None, metric: str | None) -> str:
    text = f"{issue or ''} {metric or ''}".lower()
    if any(word in text for word in ("abnormal", "failed", "failure", "high delay", "low completion")):
        return "high"
    if any(word in text for word in ("delay", "driver", "hub")):
        return "medium"
    return "low"
