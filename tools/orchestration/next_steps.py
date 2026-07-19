"""Generate relevant next-step recommendations after an analyst response."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.models import QueryResponse
from tools.llm.client import get_llm
from tools.orchestration.models import SemanticAnalysis


def generate_next_steps(
    *,
    question: str,
    response: QueryResponse,
    semantic: SemanticAnalysis | dict[str, Any] | None = None,
) -> list[str]:
    """Suggest actionable follow-up questions based on the answer context."""
    if response.capability in {"unknown", "conversation"}:
        return _default_conversation_next_steps()

    semantic_data = _as_semantic_dict(semantic)
    domain = semantic_data.get("domain")
    sub_intent = semantic_data.get("sub_intent")
    rows = response.sql_rows or []
    answer = response.answer or ""

    if not answer.strip():
        return []

    try:
        payload = _llm_next_steps(
            question=question,
            answer=answer,
            capability=response.capability,
            domain=domain,
            sub_intent=sub_intent,
            metric=response.business_metric or semantic_data.get("metric"),
            dimension=response.analysis_dimension or semantic_data.get("dimension"),
            row_preview=rows[:5],
        )
        steps = payload.get("next_steps") or []
        return [str(step).strip() for step in steps if str(step).strip()][:4]
    except Exception:
        return _heuristic_next_steps(response, semantic_data)


def _llm_next_steps(
    *,
    question: str,
    answer: str,
    capability: str,
    domain: str | None,
    sub_intent: str | None,
    metric: str | None,
    dimension: str | None,
    row_preview: list[dict[str, Any]],
) -> dict[str, Any]:
    messages = [
        SystemMessage(
            content=(
                "You generate concise follow-up questions for a GOFO operations analyst.\n"
                "Return ONLY JSON: {\"next_steps\": [\"question 1\", \"question 2\", \"question 3\"]}\n"
                "Each step must be a short, actionable user question relevant to the current answer.\n"
                "Prefer operational drilldowns, comparisons, root cause, and recommendations.\n"
                "Do not repeat the original question."
            )
        ),
        HumanMessage(
            content=(
                f"Original question: {question}\n"
                f"Capability: {capability}\n"
                f"Domain: {domain}\n"
                f"Sub-intent: {sub_intent}\n"
                f"Metric: {metric}\n"
                f"Dimension: {dimension}\n"
                f"Answer:\n{answer[:2500]}\n"
                f"Row preview: {row_preview}\n"
            )
        ),
    ]
    response = get_llm().invoke(messages)
    content = response.content
    raw = content if isinstance(content, str) else str(content)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


def _heuristic_next_steps(
    response: QueryResponse,
    semantic: dict[str, Any],
) -> list[str]:
    steps: list[str] = []
    dimension = response.analysis_dimension or semantic.get("dimension")
    sub_intent = semantic.get("sub_intent")
    capability = response.capability

    if capability == "sql":
        if sub_intent == "ranking" and dimension:
            steps.append(f"Why is the worst {dimension} performing badly?")
            steps.append(f"What should operations do about the worst {dimension}?")
        elif sub_intent in {"summary", "overview"}:
            steps.append("Which hub is performing worst today?")
            steps.append("Are there any abnormal pickup patterns today?")
        elif sub_intent == "lookup":
            steps.append("Show more detail from the previous result.")
        else:
            steps.append("What should operations do next?")
            steps.append("Why is performance changing?")
    elif capability == "rag":
        steps.append("How does this apply to today's pickup operations?")
        steps.append("What should operations do if this process fails?")
    elif capability == "multi":
        steps.append("Which hubs are driving the issue?")
        steps.append("What immediate action should operations take?")

    if response.root_cause and "Why" not in " ".join(steps):
        steps.append("Why is this happening?")
    return steps[:4]


def _default_conversation_next_steps() -> list[str]:
    return [
        "How are operations performing today?",
        "Rank all hubs by performance.",
        "What is the pickup SOP?",
    ]


def _as_semantic_dict(semantic: SemanticAnalysis | dict[str, Any] | None) -> dict[str, Any]:
    if semantic is None:
        return {}
    if isinstance(semantic, SemanticAnalysis):
        return semantic.model_dump()
    return semantic
