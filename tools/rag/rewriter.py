"""Query rewriting for improved GOFO SOP semantic retrieval."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tools.llm.client import get_llm


def _build_system_prompt() -> str:
    """Return static instructions for retrieval-oriented query rewriting."""
    return (
        "You rewrite user questions for semantic search over GOFO operations SOP documents.\n"
        "Do NOT answer the question.\n"
        "Preserve the original meaning.\n"
        "Expand abbreviations when possible (for example, CBT -> Collection by TikTok (CBT)).\n"
        "Make implicit subjects explicit and reference GOFO SOP context when helpful.\n"
        "Keep the rewritten query concise as one clear question sentence.\n"
        "Return only the rewritten question with no preamble or explanation."
    )


def _build_user_prompt(question: str) -> str:
    """Return the user message containing the original question."""
    return f"Original question:\n{question}\n\nRewritten question:"


def rewrite(question: str) -> str:
    """
    Rewrite a user question for semantic retrieval.

    Returns the original question unchanged if rewriting fails.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    try:
        messages = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content=_build_user_prompt(question)),
        ]
        response = get_llm().invoke(messages)
        content = response.content
        rewritten = content.strip() if isinstance(content, str) else str(content).strip()
        if not rewritten:
            return question
        return rewritten
    except Exception:
        return question
