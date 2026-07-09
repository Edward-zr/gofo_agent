"""Natural-language summarization for SQL query results."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from tools.llm.client import get_llm

EMPTY_ANSWER = "No matching operational records were found."


def _build_prompt(question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    """Build the summarization prompt for the LLM."""
    return (
        "You are an operational analytics assistant.\n\n"
        f"The user asked:\n{question}\n\n"
        f"The executed SQL was:\n{sql}\n\n"
        f"The database returned:\n{rows}\n\n"
        "Write a concise business answer.\n\n"
        "If there are no rows,\n"
        f'reply\n\n"{EMPTY_ANSWER}"\n\n'
        "Do not mention SQL.\n"
        "Do not explain how the answer was generated."
    )


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

    messages = [HumanMessage(content=_build_prompt(question, sql, rows))]
    response = get_llm().invoke(messages)
    content = response.content

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()
