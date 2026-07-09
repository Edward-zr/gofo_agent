"""Summarization and entity extraction helpers for conversation memory."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from tools.llm.client import get_llm

SUPPORTED_ENTITIES = (
    "driver_name",
    "customer",
    "warehouse",
    "hub",
    "date",
    "time_period",
    "pickup_status",
    "metric",
)


def _entity_system_prompt() -> str:
    """Return entity extraction instructions."""
    return (
        "You extract operational entities.\n\n"
        "Return JSON only.\n\n"
        "Supported entities:\n"
        "driver_name\n"
        "customer\n"
        "warehouse\n"
        "hub\n"
        "date\n"
        "time_period\n"
        "pickup_status\n"
        "metric\n\n"
        "Examples:\n\n"
        "Input:\n"
        '"Drew Nguyen completed 56 pickups"\n\n'
        "Output:\n"
        "{\n"
        '  "driver_name": "Drew Nguyen",\n'
        '  "metric": "pickup_count"\n'
        "}\n\n"
        "Input:\n"
        '"Atlanta Hub completed 94 pickups"\n\n'
        "Output:\n"
        "{\n"
        '  "hub": "Atlanta Hub"\n'
        "}\n\n"
        "Do not include unsupported keys.\n"
        "Do not include null values."
    )


def extract_entities(question: str, answer: str) -> dict[str, Any]:
    """Extract operational entities from a question and answer."""
    question = question.strip()
    answer = answer.strip()
    if not question or not answer:
        return {}

    messages = [
        SystemMessage(content=_entity_system_prompt()),
        HumanMessage(content=f"Question:\n{question}\n\nAssistant answer:\n{answer}\n\nJSON:"),
    ]
    response = get_llm().invoke(messages)
    raw_content = response.content if isinstance(response.content, str) else str(response.content)
    entities = _parse_json(raw_content)

    return {
        key: value
        for key, value in entities.items()
        if key in SUPPORTED_ENTITIES and value is not None and value != ""
    }


def summarize_result_context(
    *,
    question: str,
    intent: str | None,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> str:
    """Create a compact human-readable description of a result table."""
    if not rows:
        return "empty analytical result"

    column_text = ", ".join(columns)
    intent_text = intent or "analytics"
    return (
        f"{intent_text} result for '{question}' with columns "
        f"{column_text} and {len(rows)} row(s)"
    )


def _parse_json(content: str) -> dict[str, Any]:
    """Parse JSON, tolerating accidental fenced output."""
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    parsed = json.loads(content)
    return parsed if isinstance(parsed, dict) else {}
