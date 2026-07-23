"""Natural-language planning for GOFO capability selection."""

from __future__ import annotations

import json

from tools.planner.models import PlanningDecision


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
    Loads the router prompt exclusively through the Prompt Registry.
    """
    from core.prompt_manager import get_prompt_manager

    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    raw_decision = get_prompt_manager().invoke(
        "router.router_prompt",
        {
            "question": question,
            "memory": "(none)",
        },
        user_content=f"Question:\n{question}",
    )
    return PlanningDecision.model_validate(_parse_json(raw_decision))
