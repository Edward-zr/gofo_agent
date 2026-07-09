"""Combine SQL analytics and RAG knowledge into one operational answer."""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage

from core.models import QueryResponse
from tools.llm.client import get_llm


def _build_prompt(
    question: str,
    sql_result: Optional[QueryResponse],
    rag_result: Optional[QueryResponse],
) -> str:
    """Build the multi-tool synthesis prompt."""
    sql_output = sql_result.answer if sql_result and sql_result.answer else "None"
    rag_output = rag_result.answer if rag_result and rag_result.answer else "None"

    return (
        "You are the GOFO Operations Intelligence Assistant.\n\n"
        "Combine the operational analytics results and the SOP knowledge.\n\n"
        f"User question:\n{question}\n\n"
        f"Operational analytics result:\n{sql_output}\n\n"
        f"SOP knowledge result:\n{rag_output}\n\n"
        "If only one tool returned results, answer using only that tool.\n"
        "If both returned results, produce one integrated answer.\n"
        "Do not mention SQL.\n"
        "Do not mention ChromaDB.\n"
        "Do not mention retrieval.\n"
        "Write one concise operational recommendation."
    )


def synthesize(
    question: str,
    sql_result: Optional[QueryResponse],
    rag_result: Optional[QueryResponse],
) -> str:
    """Merge SQL and RAG tool outputs into a single business answer."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    messages = [HumanMessage(content=_build_prompt(question, sql_result, rag_result))]
    response = get_llm().invoke(messages)
    content = response.content

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()
