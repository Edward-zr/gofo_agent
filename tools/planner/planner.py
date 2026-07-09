"""Natural-language planning for GOFO capability selection."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from tools.llm.client import get_llm
from tools.planner.models import PlanningDecision


def _build_system_prompt() -> str:
    """Return the planner system prompt."""
    return (
        "You are an AI planner for a logistics operations assistant.\n\n"
        "Your job is NOT to answer questions.\n\n"
        "Your job is ONLY to decide:\n\n"
        "1.\n"
        "Does this require\n\n"
        "- SQL analytics\n\n"
        "- SOP retrieval\n\n"
        "- Both\n\n"
        "- Neither\n\n"
        "2.\n\n"
        "Extract business intent.\n\n"
        "Capability rules:\n\n"
        "SQL questions involve counts, KPIs, volume, trend, performance, ranking, "
        "customers, drivers, warehouse, addresses, packages, dates, statistics, "
        "comparison, aggregation, and lookups of operational entities such as "
        "which hub a driver belongs to.\n\n"
        "RAG questions involve SOP, policy, procedure, how, workflow, scan, "
        "pickup process, CBT, exception handling, or documentation.\n\n"
        "MULTI questions need current metrics plus operational guidance.\n\n"
        "UNKNOWN questions include greetings, random text, and unrelated questions.\n\n"
        "Examples\n\n"
        "How many pickups yesterday?\n"
        "capability = sql\n"
        "intent = pickup_count\n\n"
        "Which hub does Drew Nguyen belong to?\n"
        "capability = sql\n"
        "intent = driver_lookup\n\n"
        "What is CBT?\n"
        "capability = rag\n"
        "intent = explain_cbt\n\n"
        "How should operations respond if delayed pickups exceed 50 today?\n"
        "capability = multi\n"
        "intent = delayed_pickups\n\n"
        "Show pickup trends.\n"
        "capability = sql\n"
        "intent = trend\n\n"
        "What is the scanning SOP?\n"
        "capability = rag\n"
        "intent = scanning\n\n"
        "Compare this week with last week and recommend actions.\n"
        "capability = multi\n"
        "intent = weekly_comparison\n\n"
        "Return ONLY JSON with this exact shape:\n"
        "{\n"
        '  "capability": "sql|rag|multi|unknown",\n'
        '  "intent": "short_snake_case_intent",\n'
        '  "confidence": 0.0,\n'
        '  "requires_sql": true,\n'
        '  "requires_rag": false,\n'
        '  "reasoning": "brief reason",\n'
        '  "entities": {}\n'
        "}"
    )


def _build_user_prompt(question: str) -> str:
    """Return the user message for planning."""
    return f"Question:\n{question}"


def _parse_json(content: str) -> dict:
    """Parse planner JSON, allowing accidental fenced JSON output."""
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def plan(question: str) -> PlanningDecision:
    """
    Decide whether a question requires SQL, RAG, both, or neither.

    Does not generate SQL and does not answer the user's question.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(question)),
    ]
    response = get_llm().invoke(messages)
    content = response.content
    raw_decision = content if isinstance(content, str) else str(content)
    return PlanningDecision.model_validate(_parse_json(raw_decision))
