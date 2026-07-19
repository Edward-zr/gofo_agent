"""Public application API for the GOFO Operations Intelligence Agent."""

from __future__ import annotations

import config
from typing import Any

from core.models import QueryRequest, QueryResponse
from tools.router import route


def ask(
    question: str,
    result_context: dict[str, Any] | None = None,
    *,
    attachments: list[str] | None = None,
    file_context: dict[str, Any] | None = None,
    semantic_context: dict[str, Any] | None = None,
) -> QueryResponse:
    """Ask the agent a question and return a structured response."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    request = QueryRequest(
        question=question,
        top_k=config.TOP_K_DEFAULT,
        result_context=result_context,
        attachments=attachments,
        file_context=file_context,
        semantic_context=semantic_context,
    )
    return route(request)
