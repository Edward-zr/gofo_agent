"""SQL capability for GOFO operational analytics."""

from __future__ import annotations

from core.models import QueryRequest, QueryResponse
from tools.sql.business_date import get_latest_business_date, resolve_relative_dates
from tools.sql.executor import execute
from tools.sql.planner import plan
from tools.sql.summarizer import summarize


def answer(request: QueryRequest) -> QueryResponse:
    """
    Answer operational analytics questions by planning, executing, and summarizing SQL.
    """
    question = request.question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    latest_business_date = get_latest_business_date()
    rewritten_question = resolve_relative_dates(request.question, latest_business_date)

    sql = plan(rewritten_question)
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
    )
