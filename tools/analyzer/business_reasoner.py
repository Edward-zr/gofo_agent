"""Evidence-based business reasoning for conversational operations analysis."""

from __future__ import annotations

from typing import Any

from core.models import QueryResponse
from tools.analysis.anomaly import detect_anomalies
from tools.analysis.recommender import recommend
from tools.analysis.root_cause import analyze_root_cause
from tools.planner.intent_classifier import Intent


def answer_from_conversation(
    *,
    question: str,
    intent: Intent,
    conversation_state: dict[str, Any],
    resolved_question: str,
) -> QueryResponse | None:
    """Answer from previous SQL-backed analysis when no new SQL is required."""
    rows = conversation_state.get("sql_rows") or []
    summary = conversation_state.get("summary") or ""
    findings = conversation_state.get("business_findings") or []
    root_cause = conversation_state.get("root_cause") or analyze_root_cause(rows, question)
    recommendations = conversation_state.get("recommendations") or _recommendations_from_rows(question, rows)

    if intent == Intent.RECOMMENDATION and (rows or findings or recommendations):
        answer = _format_recommendation_answer(
            target=_target(conversation_state),
            findings=findings,
            recommendations=recommendations,
        )
        return _memory_response(
            question=question,
            resolved_question=resolved_question,
            answer=answer,
            intent=intent,
            rows=rows,
            root_cause=root_cause,
            recommendations=recommendations,
            conversation_state=conversation_state,
        )

    if intent == Intent.EXPLANATION and (summary or findings or rows):
        answer = _format_explanation_answer(summary, findings, root_cause)
        return _memory_response(
            question=question,
            resolved_question=resolved_question,
            answer=answer,
            intent=intent,
            rows=rows,
            root_cause=root_cause,
            recommendations=recommendations,
            conversation_state=conversation_state,
        )

    if intent == Intent.SUMMARY and (summary or findings or rows or conversation_state.get("ranking")):
        answer = _format_summary_answer(conversation_state)
        return _memory_response(
            question=question,
            resolved_question=resolved_question,
            answer=answer,
            intent=intent,
            rows=rows,
            root_cause=root_cause,
            recommendations=recommendations,
            conversation_state=conversation_state,
        )

    return None


def analyze_business_response(response: QueryResponse, question: str) -> QueryResponse:
    """Attach executive/business reasoning metadata to a SQL-backed response."""
    rows = response.sql_rows or []
    if not rows:
        response.root_cause = response.root_cause or analyze_root_cause(rows, question)
        response.recommendation = response.recommendation or recommend(question, rows)
        return response

    root_cause = response.root_cause or analyze_root_cause(rows, question)
    anomaly = response.anomaly or detect_anomalies(rows)
    recommendation = response.recommendation or root_cause.get("recommendation") or recommend(question, rows)
    findings = _findings_from_rows(rows, root_cause, anomaly)

    response.root_cause = root_cause
    response.anomaly = anomaly
    response.recommendation = recommendation
    response.business_findings = findings
    if not response.kpi_summary:
        response.kpi_summary = _kpis_from_rows(rows)

    evidence_summary = _executive_summary(response.answer or "", rows, findings)
    if response.answer and not response.answer.startswith("Executive Summary:"):
        response.answer = evidence_summary
    return response


def _memory_response(
    *,
    question: str,
    resolved_question: str,
    answer: str,
    intent: Intent,
    rows: list[dict[str, Any]],
    root_cause: dict[str, Any],
    recommendations: list[str],
    conversation_state: dict[str, Any],
) -> QueryResponse:
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[],
        capability="business_analysis",
        needs_sql=False,
        needs_rag=False,
        execution_order=["conversation_memory", "business_reasoner"],
        planning_capability="business_analysis",
        planning_intent=intent.value,
        generated_sql=None,
        sql_rows=rows,
        business_metric=conversation_state.get("metric"),
        analysis_dimension=conversation_state.get("dimension"),
        date_range=conversation_state.get("date_range"),
        analysis_filters=conversation_state.get("filters") or {},
        root_cause=root_cause,
        recommendation="; ".join(recommendations),
        kpi_summary=conversation_state.get("kpi_summary"),
        original_question=question,
        resolved_question=resolved_question,
    )


def _format_recommendation_answer(
    *,
    target: str,
    findings: list[str],
    recommendations: list[str],
) -> str:
    finding_text = "; ".join(findings[:3]) if findings else "The previous SQL-backed analysis identified operational performance that needs attention."
    recommendation_text = "\n".join(f"- {item}" for item in recommendations[:5])
    return (
        "Executive Summary:\n"
        f"Operations should focus on {target} based on the previous analysis.\n\n"
        "Business Findings:\n"
        f"{finding_text}\n\n"
        "Recommendations:\n"
        f"{recommendation_text or '- Continue monitoring pickup volume, completion, delays, and failures before changing operations.'}\n\n"
        "Suggested Next Investigation:\n"
        "Drill down by hub, driver, status, and failure reason if the team needs record-level evidence."
    )


def _format_summary_answer(conversation_state: dict[str, Any]) -> str:
    summary = conversation_state.get("summary") or "No prior analysis summary is available."
    findings = conversation_state.get("business_findings") or []
    recommendations = conversation_state.get("recommendations") or []
    ranking = conversation_state.get("ranking") or []
    best = conversation_state.get("best_entity")
    worst = conversation_state.get("worst_entity")
    ranking_text = ", ".join(ranking[:5]) if ranking else "No ranking was stored."
    recommendation_text = "\n".join(f"- {item}" for item in recommendations[:5])
    return (
        "Executive Summary:\n"
        f"{summary}\n\n"
        "Operational KPIs:\n"
        f"{_kpi_sentence(conversation_state.get('sql_rows') or [])}\n\n"
        "Business Findings:\n"
        f"{'; '.join(findings[:4]) if findings else 'No additional findings were stored.'}\n\n"
        "Ranking Context:\n"
        f"Best: {best or 'N/A'}. Worst: {worst or 'N/A'}. Order: {ranking_text}.\n\n"
        "Recommendations:\n"
        f"{recommendation_text or '- Continue monitoring operational KPIs and drill down if needed.'}\n\n"
        "Suggested Next Investigation:\n"
        "Ask for details, compare another period, or request root-cause analysis on the worst-performing entity."
    )


def _format_explanation_answer(
    summary: str,
    findings: list[str],
    root_cause: dict[str, Any],
) -> str:
    causes = root_cause.get("main_causes") or []
    return (
        "Executive Summary:\n"
        f"{summary or 'The previous operational analysis is the active topic.'}\n\n"
        "Business Findings:\n"
        f"{'; '.join(findings[:4]) if findings else 'No additional finding was stored beyond the previous SQL result.'}\n\n"
        "Root Cause:\n"
        f"{'; '.join(causes[:4]) if causes else root_cause.get('issue', 'No dominant root cause was supported by the available rows.')}\n\n"
        "Suggested Next Investigation:\n"
        "Ask for details, compare another period, or break the result down by hub, driver, customer, or status."
    )


def _executive_summary(
    base_answer: str,
    rows: list[dict[str, Any]],
    findings: list[str],
) -> str:
    return (
        "Executive Summary:\n"
        f"{base_answer}\n\n"
        "Operational KPIs:\n"
        f"{_kpi_sentence(rows)}\n\n"
        "Business Findings:\n"
        f"{'; '.join(findings[:4]) if findings else 'No major exception pattern was identified from the returned rows.'}\n\n"
        "Suggested Next Investigation:\n"
        "If this result is unexpected, drill down by hub, driver, customer, status, or pickup records."
    )


def _findings_from_rows(
    rows: list[dict[str, Any]],
    root_cause: dict[str, Any],
    anomaly: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    issue = root_cause.get("issue")
    if issue:
        findings.append(str(issue))
    findings.extend(str(item) for item in root_cause.get("main_causes") or [])
    if anomaly.get("is_anomaly"):
        findings.append(str(anomaly.get("message") or "Anomaly detected."))
    if not findings:
        top_dimension = _top_dimension(rows)
        if top_dimension:
            findings.append(f"The returned rows are concentrated around {top_dimension}.")
    return findings


def _kpis_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    first = rows[0]
    return {
        key: first[key]
        for key in (
            "pickup_count",
            "pickups",
            "package_volume",
            "package_count",
            "completion_rate",
            "delay_rate",
            "failure_rate",
            "failed_pickups",
            "delayed_pickups",
            "completed_pickups",
        )
        if key in first
    }


def _kpi_sentence(rows: list[dict[str, Any]]) -> str:
    kpis = _kpis_from_rows(rows)
    if not kpis:
        return f"{len(rows)} row(s) returned for analysis."
    return ", ".join(f"{key}: {value}" for key, value in kpis.items())


def _recommendations_from_rows(question: str, rows: list[dict[str, Any]]) -> list[str]:
    base = recommend(question, rows)
    recommendations = [part.strip(" -") for part in base.split(";") if part.strip(" -")]
    if not recommendations:
        recommendations = [base]
    if _has_column(rows, "delayed_pickups") or _has_status(rows, "Delayed"):
        recommendations.append("Investigate staffing, route load, and affected hub capacity for delayed pickups.")
    if _has_column(rows, "failed_pickups") or _has_status(rows, "Failed"):
        recommendations.append("Review failed pickup reasons and SOP adherence for concentrated failures.")
    if _has_column(rows, "completion_rate"):
        recommendations.append("Prioritize coaching or capacity balancing where completion rate is below target.")
    if _has_column(rows, "package_count") or _has_column(rows, "package_volume"):
        recommendations.append("Check whether package concentration requires additional loading or driver resources.")
    return _dedupe(recommendations)


def _target(conversation_state: dict[str, Any]) -> str:
    return str(
        conversation_state.get("worst_entity")
        or conversation_state.get("best_entity")
        or conversation_state.get("resolved_question")
        or "the current operational issue"
    )


def _top_dimension(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    first = rows[0]
    for key in ("hub", "driver_name", "customer_name", "pickup_date", "status", "reason"):
        if first.get(key):
            return f"{key}={first[key]}"
    return None


def _has_column(rows: list[dict[str, Any]], column: str) -> bool:
    return any(column in row for row in rows)


def _has_status(rows: list[dict[str, Any]], status: str) -> bool:
    return any(str(row.get("status", "")).lower() == status.lower() for row in rows)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
