"""SQL capability for GOFO operational analytics.

Pipeline:
  Question
    → Business Understanding
    → Schema Retriever
    → SQL Planner / Generator
    → Validator
    → Executor
    → Summarizer
"""

from __future__ import annotations

from typing import Any

from core.models import QueryRequest, QueryResponse
from tools.sql.business_date import get_latest_business_date, resolve_relative_dates
from tools.sql.executor import execute
from tools.sql.planner import plan
from tools.sql.schema_retriever import RetrievedSchema, SchemaRetriever
from tools.sql.summarizer import summarize


def understand_business(
    question: str,
    *,
    semantic_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lightweight business understanding before schema retrieval."""
    semantic_context = semantic_context or {}
    text = (question or "").lower()
    intent = (
        semantic_context.get("sub_intent")
        or semantic_context.get("intent")
        or semantic_context.get("business_intent")
    )
    if not intent:
        if any(token in text for token in ("root cause", "why failed", "failure reason")):
            intent = "root_cause"
        elif any(token in text for token in ("rank", "top", "worst", "best", "highest", "lowest")):
            intent = "ranking"
        elif any(token in text for token in ("compare", "vs ", "versus")):
            intent = "comparison"
        elif "hub" in text or "warehouse" in text:
            intent = "hub"
        elif "driver" in text:
            intent = "driver"
        else:
            intent = "sql_query"
    return {
        "business_intent": str(intent),
        "metric": semantic_context.get("metric"),
        "dimension": semantic_context.get("dimension"),
        "entities": semantic_context.get("entities") or {},
    }


def answer(request: QueryRequest) -> QueryResponse:
    """
    Answer operational analytics questions by retrieving schema, planning,
    validating, executing, and summarizing SQL.
    """
    question = request.question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    semantic_context = request.semantic_context if isinstance(request.semantic_context, dict) else None

    # 1) Business understanding + relative date rewrite
    latest_business_date = get_latest_business_date()
    rewritten_question = resolve_relative_dates(request.question, latest_business_date)
    understanding = understand_business(rewritten_question, semantic_context=semantic_context)

    # 2) Schema retrieval (never full schema unless explicitly requested)
    retrieved_schema: RetrievedSchema = SchemaRetriever().retrieve(
        rewritten_question,
        business_intent=understanding.get("business_intent"),
        semantic_context=semantic_context,
    )

    # 3) SQL generation using ONLY retrieved schema + 4) validation inside plan()
    sql = plan(
        rewritten_question,
        retrieved_schema=retrieved_schema,
        business_intent=understanding.get("business_intent"),
        semantic_context=semantic_context,
    )

    # 5) Execute
    rows = execute(sql)
    summary = summarize(request.question, sql, rows)

    return QueryResponse(
        question=request.question,
        answer=summary,
        sources=[],
        capability="sql",
        rewritten_question=rewritten_question,
        latest_business_date=latest_business_date.isoformat(),
        generated_sql=sql,
        sql_rows=rows,
        retrieved_schema=retrieved_schema.model_dump(),
        candidate_tables=list(retrieved_schema.candidate_tables),
        candidate_columns=dict(retrieved_schema.candidate_columns),
        plan_reason=retrieved_schema.reasoning,
        semantic_sub_intent=understanding.get("business_intent"),
    )
