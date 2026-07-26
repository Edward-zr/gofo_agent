"""Attachment-aware analysis, comparison, and response generation."""

from __future__ import annotations

import re
from typing import Any

from core.models import QueryResponse, SourceChunk
from tools.files.analysis_intent import AnalysisIntent, detect_analysis_intent
from tools.files.data_analysis import analyze_dataframe
from tools.files.dataframe_store import StoredAttachment, context_to_stored_attachment
from tools.files.models import ProcessedFileContext, ProcessingStatus
from tools.files.source_router import DataSource
from tools.rag.service import answer as rag_answer
from tools.sql.business_date import get_latest_business_date
from tools.sql.executor import execute

_HUB_COLUMNS = ("hub", "warehouse")
_DRIVER_COLUMNS = ("driver_name", "driver")
_CUSTOMER_COLUMNS = ("customer_name", "customer")
_METRIC_COLUMNS = ("package_count", "packages", "pickup_count", "pickups", "completion_rate", "delay_rate", "failure_rate")


def analyze_attachments(
    *,
    question: str,
    resolved_question: str,
    contexts: list[ProcessedFileContext],
    file_context: dict[str, Any],
    data_sources: list[DataSource],
    intent: str,
    stored_attachments: list[StoredAttachment] | None = None,
    previous_filter: dict[str, Any] | None = None,
    last_entity: str | None = None,
) -> QueryResponse | None:
    """Answer a question with a fresh analysis of the stored attachment context."""
    if not contexts:
        return None
    if any(context.processing_status == ProcessingStatus.FAILED for context in contexts):
        failed = next(context for context in contexts if context.processing_status == ProcessingStatus.FAILED)
        return _error_response(question, resolved_question, failed.error_message or "Attachment processing failed.")

    if DataSource.ATTACHMENT_AND_SQL in data_sources:
        return _attachment_sql_comparison(question, resolved_question, contexts, file_context)
    if DataSource.ATTACHMENT_AND_RAG in data_sources:
        return _attachment_rag_comparison(question, resolved_question, contexts, file_context)
    if len(contexts) >= 2 and _asks_multi_file_comparison(question):
        return _multi_attachment_comparison(question, resolved_question, contexts, file_context)
    if any(context.file_type == "image" for context in contexts):
        return _image_analysis_response(question, resolved_question, contexts, file_context, intent)
    if _is_structured_data(contexts):
        return _ada_structured_analysis(
            question=question,
            resolved_question=resolved_question,
            contexts=contexts,
            file_context=file_context,
            stored_attachments=stored_attachments,
            previous_filter=previous_filter,
            last_entity=last_entity,
        )
    return _document_summary_response(question, resolved_question, contexts, file_context, intent)


def _ada_structured_analysis(
    *,
    question: str,
    resolved_question: str,
    contexts: list[ProcessedFileContext],
    file_context: dict[str, Any],
    stored_attachments: list[StoredAttachment] | None,
    previous_filter: dict[str, Any] | None,
    last_entity: str | None = None,
) -> QueryResponse:
    """Run ChatGPT-ADA style analysis against the parsed dataframe for this prompt."""
    # Prefer the user's literal wording: conversation repair can rewrite follow-ups
    # into prior summary prompts and drop named file columns.
    analysis_question = _prefer_file_analysis_question(question, resolved_question)
    analysis_intent = detect_analysis_intent(analysis_question)

    if len(contexts) >= 2 and (
        analysis_intent == AnalysisIntent.COMPARISON or _asks_hub_change(analysis_question)
    ):
        return _multi_attachment_comparison(question, analysis_question, contexts, file_context)

    stored = _select_stored_attachment(contexts, stored_attachments)
    if stored is None or not stored.is_tabular:
        primary = contexts[-1]
        stored = context_to_stored_attachment(primary)

    result = analyze_dataframe(
        question=analysis_question,
        stored=stored,
        intent=analysis_intent,
        previous_filter=previous_filter,
        last_entity=last_entity,
    )
    recommendation = "; ".join(result.get("recommendations") or [])
    sources = file_context.get("source_references") or [stored.filename]
    answer = f"{result['answer']}\n\nSources:\n" + "\n".join(sources)
    response = QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="file_analysis",
        planning_capability="file_analysis",
        planning_intent=result.get("intent") or analysis_intent.value,
        execution_order=["attachment", "dataframe_analysis"],
        sql_rows=result.get("rows") or [],
        business_metric=result.get("metric"),
        analysis_dimension=result.get("dimension"),
        analysis_filters=result.get("active_filter") or {},
        root_cause=result.get("root_cause") or {},
        recommendation=recommendation or None,
        business_findings=result.get("findings") or [],
        charts=result.get("charts") or [],
        original_question=question,
        resolved_question=resolved_question,
    )
    # Persist ephemeral follow-up entity for pronoun-style questions.
    response.file_context_summary = {
        "last_entity": result.get("last_entity"),
        "active_filter": result.get("active_filter") or {},
        "analysis_intent": result.get("intent"),
    }
    return response


def _select_stored_attachment(
    contexts: list[ProcessedFileContext],
    stored_attachments: list[StoredAttachment] | None,
) -> StoredAttachment | None:
    if not stored_attachments:
        return None
    by_id = {item.attachment_id: item for item in stored_attachments}
    for context in reversed(contexts):
        if context.attachment_id in by_id:
            return by_id[context.attachment_id]
    return stored_attachments[-1]


def _image_analysis_response(
    question: str,
    resolved_question: str,
    contexts: list[ProcessedFileContext],
    file_context: dict[str, Any],
    intent: str,
) -> QueryResponse:
    primary = contexts[-1]
    analysis = primary.image_analysis or {}
    answer = (
        "Current Analysis:\n"
        f"Based on {primary.filename}...\n\n"
        f"{analysis.get('description') or analysis.get('analysis') or 'No visual analysis was returned.'}\n\n"
        "Sources:\n"
        f"{primary.filename}"
    )
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="image_analysis",
        planning_capability="image_analysis",
        planning_intent=intent or "IMAGE_ANALYSIS",
        execution_order=["attachment", "image_analysis"],
        business_findings=list(analysis.get("observed_metrics") or []),
        original_question=question,
        resolved_question=resolved_question,
    )


def _document_summary_response(
    question: str,
    resolved_question: str,
    contexts: list[ProcessedFileContext],
    file_context: dict[str, Any],
    intent: str,
) -> QueryResponse:
    primary = contexts[-1]
    text = "\n".join(primary.paragraphs[:8]) or (primary.pages[0]["text"] if primary.pages else primary.summary)
    risks = _extract_risks(text or "")
    answer = (
        "Current Analysis:\n"
        f"Based on {primary.filename}...\n\n"
        f"{primary.summary or 'Document analyzed.'}\n\n"
        "Operational Risk:\n"
        f"{'; '.join(risks) if risks else 'No explicit operational risk language was detected.'}\n\n"
        "Sources:\n"
        + "\n".join(file_context.get("source_references", [primary.filename]))
    )
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="file_analysis",
        planning_capability="file_analysis",
        planning_intent=intent or "FILE_SUMMARY",
        execution_order=["attachment", "document_analysis"],
        business_findings=risks,
        original_question=question,
        resolved_question=resolved_question,
    )


def _attachment_sql_comparison(
    question: str,
    resolved_question: str,
    contexts: list[ProcessedFileContext],
    file_context: dict[str, Any],
) -> QueryResponse:
    primary = contexts[-1]
    rows = _all_rows(primary)
    db_rows = _database_rows_for_comparison(resolved_question)
    comparison = _compare_rows(rows, db_rows)
    answer = (
        "Current Analysis:\n"
        f"Compared {primary.filename} with today's GOFO operational database.\n\n"
        "Matching Metrics:\n"
        f"{'; '.join(comparison['matching']) if comparison['matching'] else 'No exact metric matches were found.'}\n\n"
        "Differences:\n"
        f"{'; '.join(comparison['differences']) if comparison['differences'] else 'No major differences detected in available aggregates.'}\n\n"
        "Business Impact:\n"
        f"{comparison['impact']}\n\n"
        "Sources:\n"
        f"{primary.filename}\nGOFO operational database"
    )
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="file_comparison",
        planning_capability="file_comparison",
        planning_intent="FILE_DATABASE_COMPARISON",
        execution_order=["attachment", "sqlite", "comparison"],
        sql_rows=db_rows[:10],
        business_findings=comparison["differences"],
        original_question=question,
        resolved_question=resolved_question,
    )


def _attachment_rag_comparison(
    question: str,
    resolved_question: str,
    contexts: list[ProcessedFileContext],
    file_context: dict[str, Any],
) -> QueryResponse:
    primary = contexts[-1]
    document_text = "\n".join(primary.paragraphs[:10]) or (primary.pages[0]["text"] if primary.pages else "")
    rag_result = rag_answer(_rag_request(resolved_question))
    answer = (
        "Current Analysis:\n"
        f"Compared {primary.filename} with available GOFO SOP knowledge.\n\n"
        "Uploaded Document Summary:\n"
        f"{primary.summary or 'Uploaded document analyzed.'}\n\n"
        "SOP Alignment:\n"
        f"{rag_result.answer or 'No SOP alignment summary available.'}\n\n"
        "Potential Gaps:\n"
        f"{_sop_gaps(document_text, rag_result.answer or '')}\n\n"
        "Sources:\n"
        f"{primary.filename}"
    )
    sources = rag_result.sources or []
    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        capability="file_rag_comparison",
        planning_capability="file_rag_comparison",
        planning_intent="FILE_RAG_COMPARISON",
        execution_order=["attachment", "rag", "comparison"],
        rag_output=rag_result.answer,
        original_question=question,
        resolved_question=resolved_question,
    )


def _multi_attachment_comparison(
    question: str,
    resolved_question: str,
    contexts: list[ProcessedFileContext],
    file_context: dict[str, Any],
) -> QueryResponse:
    left, right = contexts[0], contexts[1]
    left_rows = _all_rows(left)
    right_rows = _all_rows(right)

    if _asks_hub_change(resolved_question) or _asks_hub_change(question):
        metric = _metric_column(left_rows, resolved_question) or _metric_column(right_rows, resolved_question)
        change = _largest_hub_change(left_rows, right_rows, metric=metric)
        if change:
            answer = (
                "Current Analysis:\n"
                f"Compared {left.filename} and {right.filename}.\n\n"
                f"{change['hub']} changed the most for {change['metric']}: "
                f"{change['left_value']} -> {change['right_value']} ({change['delta']} change).\n\n"
                "Sources:\n"
                f"{left.filename}\n{right.filename}"
            )
            return QueryResponse(
                question=question,
                answer=answer,
                sources=[],
                capability="file_comparison",
                planning_capability="file_comparison",
                planning_intent="FILE_COMPARISON",
                execution_order=["attachment", "comparison"],
                sql_rows=[change],
                business_findings=[f"{change['hub']} changed the most for {change['metric']}."],
                original_question=question,
                resolved_question=resolved_question,
            )

    comparison = _compare_rows(left_rows, right_rows, left_name=left.filename, right_name=right.filename)
    answer = (
        "Current Analysis:\n"
        f"Compared {left.filename} and {right.filename}.\n\n"
        "Differences:\n"
        f"{'; '.join(comparison['differences']) if comparison['differences'] else 'No major differences detected.'}\n\n"
        "Sources:\n"
        f"{left.filename}\n{right.filename}"
    )
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="file_comparison",
        planning_capability="file_comparison",
        planning_intent="FILE_COMPARISON",
        execution_order=["attachment", "comparison"],
        business_findings=comparison["differences"],
        original_question=question,
        resolved_question=resolved_question,
    )


def _error_response(question: str, resolved_question: str, message: str) -> QueryResponse:
    return QueryResponse(
        question=question,
        answer=message,
        sources=[],
        capability="file_analysis",
        planning_capability="file_analysis",
        planning_intent="FILE_ANALYSIS",
        original_question=question,
        resolved_question=resolved_question,
    )


def _all_rows(context: ProcessedFileContext) -> list[dict[str, Any]]:
    if context.file_type == "csv":
        return list(context.full_data.get("rows") or [])
    if context.file_type == "excel":
        sheets = context.full_data.get("sheets") or context.sheets
        if not sheets:
            return []
        active = context.file_schema.get("active_sheet")
        sheet = next((item for item in sheets if item.get("name") == active), sheets[0])
        return list(sheet.get("rows") or [])
    return []


def _database_rows_for_comparison(question: str) -> list[dict[str, Any]]:
    latest_date = get_latest_business_date().isoformat()
    sql = f"""
SELECT d.hub,
       COUNT(*) AS pickup_count,
       SUM(p.package_count) AS package_volume,
       ROUND(100.0 * SUM(CASE WHEN p.status='Completed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS completion_rate
FROM pickups p
JOIN drivers d ON p.driver_id = d.driver_id
WHERE p.pickup_date = '{latest_date}'
GROUP BY d.hub
ORDER BY pickup_count DESC;
""".strip()
    return execute(sql)


def _compare_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    left_name: str = "uploaded file",
    right_name: str = "database",
) -> dict[str, Any]:
    if _has_dimension(left_rows, "hub") and _has_dimension(right_rows, "hub"):
        return _compare_hub_rows(left_rows, right_rows, left_name=left_name, right_name=right_name)

    left_metric = _aggregate_rows(left_rows)
    right_metric = _aggregate_rows(right_rows)
    matching = []
    differences = []
    for key in set(left_metric) | set(right_metric):
        left_value = left_metric.get(key)
        right_value = right_metric.get(key)
        if left_value is None or right_value is None:
            continue
        if abs(left_value - right_value) <= max(1.0, abs(right_value) * 0.05):
            matching.append(f"{key}: {left_value} ~= {right_value}")
        else:
            pct = round(abs(left_value - right_value) / max(right_value, 1) * 100, 2)
            differences.append(f"{key}: {left_name}={left_value}, {right_name}={right_value} ({pct}% difference)")
    impact = (
        "Uploaded data diverges from live operations and should be reviewed before action."
        if differences
        else "Uploaded data is broadly aligned with available operational aggregates."
    )
    return {"matching": matching, "differences": differences, "impact": impact}


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if key in _METRIC_COLUMNS:
                try:
                    totals[key] = totals.get(key, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue
    return totals


def _rank_rows(
    rows: list[dict[str, Any]],
    dimension: str,
    metric: str | None,
    *,
    ascending: bool,
) -> list[dict[str, Any]]:
    if not rows or not metric:
        return rows
    key = _dimension_key(rows, dimension)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        entity = str(row.get(key) or "unknown")
        current = grouped.setdefault(entity, {key: entity, metric: 0})
        try:
            current[metric] += float(row.get(metric) or 0)
        except (TypeError, ValueError):
            pass
    ranked = sorted(grouped.values(), key=lambda item: float(item.get(metric) or 0), reverse=not ascending)
    return ranked


def _extreme_value(rows: list[dict[str, Any]], metric: str | None, *, ascending: bool) -> dict[str, Any] | None:
    if not rows or not metric:
        return None
    valid = [row for row in rows if row.get(metric) is not None]
    if not valid:
        return None
    return sorted(valid, key=lambda row: float(row.get(metric) or 0), reverse=not ascending)[0]


def _metric_column(rows: list[dict[str, Any]], question: str) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    normalized = question.lower()
    for candidate in _METRIC_COLUMNS:
        if candidate in keys:
            if candidate.replace("_", " ") in normalized or candidate in normalized:
                return candidate
    for candidate in _METRIC_COLUMNS:
        if candidate in keys:
            return candidate
    return next(iter(keys), None)


def _infer_dimension(question: str) -> str:
    normalized = question.lower()
    if "driver" in normalized:
        return "driver"
    if "customer" in normalized:
        return "customer"
    if "hub" in normalized or "warehouse" in normalized:
        return "hub"
    return "hub"


def _dimension_key(rows: list[dict[str, Any]], dimension: str) -> str:
    keys = set(rows[0].keys()) if rows else set()
    preferred = {
        "hub": _HUB_COLUMNS,
        "driver": _DRIVER_COLUMNS,
        "customer": _CUSTOMER_COLUMNS,
    }
    for candidate in preferred.get(dimension, ()):
        if candidate in keys:
            return candidate
    return next(iter(keys), "entity")


def _entity_value(row: dict[str, Any], dimension: str) -> str:
    return str(row.get(_dimension_key([row], dimension)) or "unknown")


def _asks_ranking(question: str) -> bool:
    return any(word in question.lower() for word in ("rank", "worst", "best", "highest", "lowest", "top", "bottom"))


def _asks_lowest(question: str) -> bool:
    return any(word in question.lower() for word in ("worst", "lowest", "bottom"))


def _asks_numeric_lookup(question: str) -> bool:
    return any(word in question.lower() for word in ("highest", "lowest", "most", "maximum", "minimum"))


def _asks_multi_file_comparison(question: str) -> bool:
    normalized = question.lower()
    return any(
        phrase in normalized
        for phrase in ("compare these", "compare the reports", "two reports", "compare these reports")
    )


def _asks_hub_change(question: str) -> bool:
    normalized = question.lower()
    return "changed" in normalized or "change the most" in normalized


def _asks_entity_lookup(question: str) -> bool:
    normalized = question.lower()
    return "belong" in normalized or ("which hub" in normalized and "driver" in normalized)


def _asks_record_drilldown(question: str) -> bool:
    normalized = question.lower()
    return any(
        phrase in normalized
        for phrase in ("show his records", "show her records", "show their records", "show records")
    )


def _infer_lookup_target(question: str) -> str:
    normalized = question.lower()
    if "hub" in normalized:
        return "hub"
    if "driver" in normalized:
        return "driver"
    if "customer" in normalized:
        return "customer"
    return "hub"


def _extract_named_entity(question: str) -> str | None:
    patterns = (
        r"which hub does (.+?) belong",
        r"show records for (.+)",
        r"records for (.+)",
        r"for (.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ?.")
    return None


def _lookup_entity_rows(
    rows: list[dict[str, Any]],
    entity_name: str | None,
    source_dimension: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    if not entity_name:
        return rows[:10]
    source_key = _dimension_key(rows, source_dimension if source_dimension != "hub" else "driver")
    normalized_name = entity_name.lower()
    return [
        row
        for row in rows
        if normalized_name in str(row.get(source_key) or "").lower()
    ]


def _has_dimension(rows: list[dict[str, Any]], dimension: str) -> bool:
    if not rows:
        return False
    return _dimension_key(rows, dimension) in rows[0]


def _compare_hub_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    metric = _metric_column(left_rows, "") or _metric_column(right_rows, "")
    if not metric:
        return {"matching": [], "differences": [], "impact": "No comparable hub metrics were found."}

    left_by_hub = {str(row.get(_dimension_key(left_rows, "hub"))): row for row in left_rows}
    right_by_hub = {str(row.get(_dimension_key(right_rows, "hub"))): row for row in right_rows}
    matching: list[str] = []
    differences: list[str] = []
    for hub in sorted(set(left_by_hub) | set(right_by_hub)):
        left_value = float(left_by_hub.get(hub, {}).get(metric) or 0)
        right_value = float(right_by_hub.get(hub, {}).get(metric) or 0)
        if hub not in left_by_hub or hub not in right_by_hub:
            differences.append(f"{hub}: missing in one report ({left_name}={left_value}, {right_name}={right_value})")
            continue
        if abs(left_value - right_value) <= max(1.0, abs(right_value) * 0.05):
            matching.append(f"{hub} {metric}: {left_value} ~= {right_value}")
        else:
            pct = round(abs(left_value - right_value) / max(right_value, 1) * 100, 2)
            differences.append(
                f"{hub}: {left_name}={left_value}, {right_name}={right_value} ({pct}% difference in {metric})"
            )
    impact = (
        "Uploaded reports diverge by hub and should be reviewed before action."
        if differences
        else "Uploaded reports are broadly aligned across hubs."
    )
    return {"matching": matching, "differences": differences, "impact": impact}


def _largest_hub_change(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    metric: str | None,
) -> dict[str, Any] | None:
    if not metric or not _has_dimension(left_rows, "hub") or not _has_dimension(right_rows, "hub"):
        return None
    left_by_hub = {str(row.get(_dimension_key(left_rows, "hub"))): row for row in left_rows}
    right_by_hub = {str(row.get(_dimension_key(right_rows, "hub"))): row for row in right_rows}
    largest: dict[str, Any] | None = None
    for hub in set(left_by_hub) & set(right_by_hub):
        left_value = float(left_by_hub[hub].get(metric) or 0)
        right_value = float(right_by_hub[hub].get(metric) or 0)
        delta = abs(right_value - left_value)
        if largest is None or delta > largest["delta"]:
            largest = {
                "hub": hub,
                "metric": metric,
                "left_value": left_value,
                "right_value": right_value,
                "delta": delta,
            }
    return largest


def _is_structured_data(contexts: list[ProcessedFileContext]) -> bool:
    return any(context.file_type in {"csv", "excel"} for context in contexts)


def _prefer_file_analysis_question(question: str, resolved_question: str | None) -> str:
    """Keep named file columns / aggregation asks from being overwritten by repair."""
    original = (question or "").strip()
    resolved = (resolved_question or "").strip()
    if not resolved or resolved == original:
        return original or resolved
    original_intent = detect_analysis_intent(original)
    resolved_intent = detect_analysis_intent(resolved)
    if original_intent in {
        AnalysisIntent.AGGREGATION,
        AnalysisIntent.RANKING,
        AnalysisIntent.COMPARISON,
        AnalysisIntent.FILTER,
        AnalysisIntent.VISUALIZE,
        AnalysisIntent.ANOMALY,
    }:
        return original
    if resolved_intent == AnalysisIntent.EXECUTIVE_SUMMARY and original_intent != AnalysisIntent.EXECUTIVE_SUMMARY:
        return original
    if re.search(r"\bcolumns?\b", original.lower()) or "列" in original or "字段" in original:
        return original
    if re.search(r"[\u4e00-\u9fff]", original) and original_intent != AnalysisIntent.GENERAL:
        return original
    return resolved


def _extract_risks(text: str) -> list[str]:
    keywords = ("risk", "delay", "failure", "issue", "problem", "non-compliance", "missing")
    return [line.strip() for line in text.splitlines() if any(keyword in line.lower() for keyword in keywords)][:5]


def _sop_gaps(document_text: str, sop_answer: str) -> str:
    if not document_text or not sop_answer:
        return "Insufficient evidence to determine SOP gaps."
    if "don't know" in sop_answer.lower():
        return "No matching SOP evidence was found for part of the uploaded process."
    return "Review uploaded steps against the cited SOP guidance for missing or conflicting steps."


def _rag_request(question: str):
    from core.models import QueryRequest

    return QueryRequest(question=question)
