"""Query rewriting for improved GOFO SOP semantic retrieval."""

from __future__ import annotations

from core.prompt_manager import get_prompt_manager


def rewrite(question: str) -> str:
    """
    Rewrite a user question for semantic retrieval.

    Returns the original question unchanged if rewriting fails.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    try:
        rewritten = get_prompt_manager().invoke(
            "rag.retrieval_prompt",
            {"question": question},
        )
        if not rewritten:
            return question
        return rewritten
    except Exception:
        return question
