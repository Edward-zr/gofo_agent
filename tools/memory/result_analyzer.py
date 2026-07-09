"""Analyze previous analytical results stored in conversation memory."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from tools.llm.client import get_llm


def _build_system_prompt() -> str:
    """Return result-analysis instructions."""
    return (
        "You are analyzing an existing result table.\n\n"
        "Do not query database.\n"
        "Use only provided rows.\n"
        "Answer user's follow-up.\n"
        "Handle requests such as rank, sort, top, bottom, average, difference, "
        "and percentage using only the provided rows.\n"
        "Be concise and operational."
    )


def _build_user_prompt(question: str, previous_rows: list[dict[str, Any]]) -> str:
    """Return the result-analysis prompt."""
    return (
        f"User follow-up:\n{question}\n\n"
        f"Previous result rows:\n{previous_rows}\n\n"
        "Answer:"
    )


def analyze_previous_result(
    question: str,
    previous_rows: list[dict[str, Any]] | dict[str, Any],
) -> str:
    """Answer a follow-up using only the previous result rows."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    rows = (
        previous_rows.get("rows", [])
        if isinstance(previous_rows, dict)
        else previous_rows
    )
    if not rows:
        raise ValueError("Previous result rows are required.")

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(question, rows)),
    ]
    response = get_llm().invoke(messages)
    content = response.content

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()
