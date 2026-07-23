"""Natural-language summarization for SQL query results."""

from __future__ import annotations

from typing import Any

from core.prompt_manager import get_prompt_manager

EMPTY_ANSWER = "No matching operational records were found."


def summarize(question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    """
    Convert SQL result rows into a concise business answer.

    Does not execute SQL or connect to a database.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if not rows:
        return EMPTY_ANSWER

    try:
        return get_prompt_manager().invoke(
            "sql.summarizer_prompt",
            {
                "question": question,
                "sql": sql,
                "rows": rows,
            },
        )
    except Exception:
        return EMPTY_ANSWER
