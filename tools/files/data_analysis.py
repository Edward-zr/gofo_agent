"""True per-prompt data analysis against stored attachment DataFrames."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from tools.files.analysis_intent import AnalysisIntent
from tools.files.charts import build_charts
from tools.files.dataframe_store import StoredAttachment


def analyze_dataframe(
    *,
    question: str,
    stored: StoredAttachment,
    intent: AnalysisIntent,
    previous_filter: dict[str, Any] | None = None,
    last_entity: str | None = None,
) -> dict[str, Any]:
    """Run a fresh analysis for the current prompt against the stored dataframe."""
    if not stored.is_tabular or stored.dataframe is None:
        raise ValueError("Stored attachment is not tabular.")

    frame = stored.dataframe.copy()
    active_filter = dict(previous_filter or {})
    if intent == AnalysisIntent.FILTER or _mentions_filter(question):
        frame, active_filter = apply_filters(frame, question, active_filter)
    elif active_filter and intent not in {
        AnalysisIntent.RANKING,
        AnalysisIntent.AGGREGATION,
        AnalysisIntent.EXECUTIVE_SUMMARY,
        AnalysisIntent.BUSINESS_REPORT,
        AnalysisIntent.VISUALIZE,
    }:
        # Keep filters for follow-up drilldowns, but do not force them on fresh global asks.
        frame = _apply_filter_dict(frame, active_filter)

    if intent == AnalysisIntent.VISUALIZE:
        return _visualize(question, stored, frame, active_filter)
    if intent == AnalysisIntent.RANKING:
        return _ranking(question, stored, frame, active_filter)
    if intent == AnalysisIntent.AGGREGATION:
        return _aggregation(question, stored, frame, active_filter)
    if intent == AnalysisIntent.ANOMALY:
        return _anomalies(question, stored, frame, active_filter)
    if intent == AnalysisIntent.COMPARISON:
        return _comparison(question, stored, frame, active_filter)
    if intent == AnalysisIntent.LOOKUP:
        return _lookup(question, stored, frame, active_filter, last_entity=last_entity)
    if intent == AnalysisIntent.RECORDS:
        return _records(question, stored, frame, active_filter, last_entity=last_entity)
    if intent == AnalysisIntent.ROOT_CAUSE:
        return _root_cause(question, stored, frame, active_filter)
    if intent == AnalysisIntent.RECOMMENDATION:
        return _recommendation(question, stored, frame, active_filter)
    if intent == AnalysisIntent.FORECAST:
        return _forecast(question, stored, frame, active_filter)
    if intent == AnalysisIntent.NARRATIVE:
        return _narrative(question, stored, frame, active_filter)
    if intent == AnalysisIntent.BUSINESS_REPORT:
        return _business_report(question, stored, frame, active_filter)
    return _executive_summary(question, stored, frame, active_filter)


def apply_filters(
    frame: pd.DataFrame,
    question: str,
    existing: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter the dataframe from natural-language constraints."""
    filters = dict(existing or {})
    normalized = question.lower()

    for column in frame.columns:
        column_l = column.lower()
        if column_l in {"hub", "warehouse"}:
            match = re.search(r"\b([a-z][a-z0-9 _-]{1,40})\s+hub\b", normalized)
            if match:
                filters[column] = match.group(1).strip().title() + " Hub" if "hub" not in match.group(1).lower() else match.group(1).strip().title()
            for token in ("chicago", "ord", "atlanta", "new york", "los angeles"):
                if token in normalized:
                    filters[column] = "ORD Hub" if token == "ord" else f"{token.title()} Hub"
        if column_l in {"driver", "driver_name"}:
            match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question)
            if match and "hub" not in match.group(1).lower():
                filters[column] = match.group(1)

    only_match = re.search(r"\bonly\s+([a-z0-9 _-]+)", normalized)
    if only_match and frame.columns.size:
        value = only_match.group(1).strip()
        for column in frame.columns:
            if frame[column].astype(str).str.contains(value, case=False, na=False).any():
                filters[column] = value
                break

    return _apply_filter_dict(frame, filters), filters


def _apply_filter_dict(frame: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    result = frame
    for column, value in filters.items():
        if column not in result.columns:
            continue
        result = result[result[column].astype(str).str.contains(str(value), case=False, na=False)]
    return result


def _executive_summary(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    overview = _dataset_overview(stored, frame, active_filter)
    metrics = _important_metrics(frame)
    patterns = _patterns(frame)
    outliers = _outlier_findings(frame)
    insights = patterns + outliers
    recommendations = _data_recommendations(frame, insights)

    answer = (
        f"Executive Summary — {stored.filename}\n\n"
        f"Dataset overview:\n{overview}\n\n"
        f"Important metrics:\n{_bullets(metrics)}\n\n"
        f"Patterns and outliers:\n{_bullets(insights) if insights else '- No strong outlier pattern detected in the uploaded data.'}\n\n"
        f"Business insights:\n{_bullets(_business_insights(frame))}\n\n"
        f"Recommendations:\n{_bullets(recommendations)}"
    )
    return _result(
        answer=answer,
        rows=_preview_rows(frame),
        findings=metrics + insights,
        recommendations=recommendations,
        charts=build_charts(frame, question=question, max_charts=2),
        active_filter=active_filter,
        intent=AnalysisIntent.EXECUTIVE_SUMMARY,
    )


def _business_report(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    summary = _executive_summary(question, stored, frame, active_filter)
    ranking = _ranking(question, stored, frame, active_filter)
    answer = (
        f"Business Report — {stored.filename}\n\n"
        f"{summary['answer']}\n\n"
        f"Ranking highlight:\n{ranking['answer']}"
    )
    charts = summary["charts"] + [chart for chart in ranking["charts"] if chart not in summary["charts"]]
    return _result(
        answer=answer,
        rows=ranking["rows"] or summary["rows"],
        findings=summary["findings"] + ranking["findings"],
        recommendations=summary["recommendations"],
        charts=charts[:3],
        active_filter=active_filter,
        intent=AnalysisIntent.BUSINESS_REPORT,
    )


def _visualize(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    preferred_dimension = _infer_dimension(question, frame)
    preferred_metric = None if _wants_row_count(question, frame) else _infer_metric(question, frame)
    chart_frame = frame
    if preferred_dimension in frame.columns and _wants_row_count(question, frame):
        chart_frame = (
            frame.groupby(preferred_dimension, dropna=False)
            .size()
            .reset_index(name="package_count")
            .sort_values("package_count", ascending=False)
        )
        preferred_metric = "package_count"
    charts = build_charts(
        chart_frame,
        question=question,
        max_charts=4,
        preferred_dimension=preferred_dimension if preferred_dimension in chart_frame.columns else None,
        preferred_metric=preferred_metric if preferred_metric in chart_frame.columns else None,
    )
    if not charts:
        answer = (
            f"I inspected {stored.filename}, but could not infer a chartable "
            "categorical/numeric combination. Provide a metric and dimension to plot."
        )
    else:
        titles = ", ".join(chart["title"] for chart in charts)
        answer = (
            f"Generated {len(charts)} chart(s) from {stored.filename}: {titles}.\n"
            "Charts are rendered below from the active dataframe."
        )
    return _result(
        answer=answer,
        rows=_preview_rows(frame),
        findings=[f"Rendered {len(charts)} chart(s) from the uploaded dataframe."],
        recommendations=["Use filters like 'Only Chicago' to refine the charts."],
        charts=charts,
        active_filter=active_filter,
        intent=AnalysisIntent.VISUALIZE,
    )


def _ranking(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    dimension = _infer_dimension(question, frame)
    metric = _infer_metric(question, frame)
    ascending = any(word in question.lower() for word in ("lowest", "worst", "bottom"))
    if dimension not in frame.columns or metric not in frame.columns:
        return _executive_summary(question, stored, frame, active_filter)

    grouped = (
        frame.groupby(dimension, dropna=False)[metric]
        .sum(numeric_only=True)
        .reset_index()
        .sort_values(metric, ascending=ascending)
    )
    if grouped.empty:
        answer = f"No ranking could be computed for {dimension} in {stored.filename}."
        rows: list[dict[str, Any]] = []
        findings: list[str] = []
    else:
        best = grouped.iloc[0]
        worst = grouped.iloc[-1]
        label = "lowest" if ascending else "highest"
        answer = (
            f"Ranking by {metric} across {dimension} in {stored.filename}:\n"
            f"- {label.title()} {dimension}: {best[dimension]} ({best[metric]})\n"
            f"- Opposite end: {worst[dimension]} ({worst[metric]})\n"
            f"Showing top {min(10, len(grouped))} rows."
        )
        rows = _preview_rows(grouped.head(10))
        findings = [
            f"{best[dimension]} is {label} for {metric} ({best[metric]}).",
            f"{worst[dimension]} is at the opposite end ({worst[metric]}).",
        ]
    charts = build_charts(grouped, question=f"rank {dimension} {metric}", max_charts=1)
    result = _result(
        answer=answer,
        rows=rows,
        findings=findings,
        recommendations=_data_recommendations(frame, findings),
        charts=charts,
        active_filter=active_filter,
        intent=AnalysisIntent.RANKING,
        dimension=dimension,
        metric=metric,
    )
    if rows:
        result["last_entity"] = str(rows[0].get(dimension) or "")
    return result


def _aggregation(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    dimension = _infer_dimension(question, frame)
    top_n = 10
    match = re.search(r"top\s+(\d+)", question.lower())
    if match:
        top_n = int(match.group(1))
    if dimension not in frame.columns:
        return _ranking(question, stored, frame, active_filter)

    use_row_count = _wants_row_count(question, frame)
    if use_row_count:
        metric = "package_count"
        grouped = (
            frame.groupby(dimension, dropna=False)
            .size()
            .reset_index(name=metric)
            .sort_values(metric, ascending=False)
            .head(top_n)
        )
        answer = (
            f"Package volume by {dimension} from {stored.filename} "
            f"(row count per address/group).\n"
            f"Showing top {len(grouped)} of {frame[dimension].nunique(dropna=False)} distinct values."
        )
        findings = [
            f"Aggregated {len(frame)} rows into {frame[dimension].nunique(dropna=False)} "
            f"{dimension} groups using row counts.",
        ]
        if not grouped.empty:
            findings.append(
                f"Highest volume: {grouped.iloc[0][dimension]} "
                f"({int(grouped.iloc[0][metric])} packages)."
            )
    else:
        metric = _infer_metric(question, frame)
        if metric not in frame.columns:
            return _ranking(question, stored, frame, active_filter)
        grouped = (
            frame.groupby(dimension, dropna=False)[metric]
            .sum(numeric_only=True)
            .reset_index()
            .sort_values(metric, ascending=False)
            .head(top_n)
        )
        answer = f"Top {len(grouped)} {dimension}(s) by {metric} from {stored.filename}."
        findings = [f"Aggregated {len(frame)} rows into {len(grouped)} {dimension} groups."]

    return _result(
        answer=answer,
        rows=_preview_rows(grouped),
        findings=findings,
        recommendations=["Ask 'Visualize' to chart this aggregation."],
        charts=build_charts(
            grouped,
            question=question,
            max_charts=1,
            preferred_dimension=dimension,
            preferred_metric=metric,
        ),
        active_filter=active_filter,
        intent=AnalysisIntent.AGGREGATION,
        dimension=dimension,
        metric=metric,
    )


def _anomalies(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    findings = _outlier_findings(frame)
    metric = _infer_metric(question, frame)
    rows: list[dict[str, Any]] = []
    if metric in frame.columns and pd.api.types.is_numeric_dtype(frame[metric]):
        series = pd.to_numeric(frame[metric], errors="coerce")
        mean = series.mean()
        std = series.std(ddof=0)
        if pd.notna(std) and std > 0:
            mask = (series - mean).abs() > (2 * std)
            rows = _preview_rows(frame.loc[mask])
    answer = (
        f"Anomaly detection on {stored.filename}:\n"
        f"{_bullets(findings) if findings else '- No statistical outliers above 2 standard deviations were found.'}"
    )
    return _result(
        answer=answer,
        rows=rows or _preview_rows(frame),
        findings=findings,
        recommendations=_data_recommendations(frame, findings),
        charts=build_charts(frame, question="distribution histogram", max_charts=2),
        active_filter=active_filter,
        intent=AnalysisIntent.ANOMALY,
        metric=metric,
    )


def _comparison(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    dimension = _infer_dimension(question, frame)
    metric = _infer_metric(question, frame)
    if dimension not in frame.columns or metric not in frame.columns:
        return _executive_summary(question, stored, frame, active_filter)
    grouped = (
        frame.groupby(dimension, dropna=False)[metric]
        .sum(numeric_only=True)
        .reset_index()
        .sort_values(metric, ascending=False)
    )
    if len(grouped) < 2:
        answer = f"Not enough {dimension} values in {stored.filename} to compare."
        findings: list[str] = []
    else:
        top = grouped.iloc[0]
        bottom = grouped.iloc[-1]
        delta = top[metric] - bottom[metric]
        answer = (
            f"Comparison of {metric} by {dimension} in {stored.filename}:\n"
            f"- Leading: {top[dimension]} ({top[metric]})\n"
            f"- Trailing: {bottom[dimension]} ({bottom[metric]})\n"
            f"- Gap: {delta}"
        )
        findings = [f"{top[dimension]} leads {bottom[dimension]} by {delta} on {metric}."]
    return _result(
        answer=answer,
        rows=_preview_rows(grouped.head(10)),
        findings=findings,
        recommendations=_data_recommendations(frame, findings),
        charts=build_charts(grouped, question=question, max_charts=1),
        active_filter=active_filter,
        intent=AnalysisIntent.COMPARISON,
        dimension=dimension,
        metric=metric,
    )


def _lookup(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
    *,
    last_entity: str | None = None,
) -> dict[str, Any]:
    entity = _extract_entity_name(question) or last_entity
    driver_col = _find_column(frame, ("driver_name", "driver"))
    hub_col = _find_column(frame, ("hub", "warehouse"))
    matched = frame
    if entity and driver_col:
        matched = frame[frame[driver_col].astype(str).str.contains(entity, case=False, na=False)]
    if matched.empty and entity and hub_col:
        matched = frame[frame[hub_col].astype(str).str.contains(entity, case=False, na=False)]
    if matched.empty and last_entity and driver_col:
        matched = frame[frame[driver_col].astype(str).str.contains(last_entity, case=False, na=False)]
        entity = last_entity
    if matched.empty:
        answer = f"No matching entity was found in {stored.filename}."
        findings: list[str] = []
        entity_out = entity
    else:
        row = matched.iloc[0]
        hub_value = row[hub_col] if hub_col else "unknown hub"
        driver_value = row[driver_col] if driver_col else entity
        answer = f"{driver_value or entity} belongs to {hub_value}."
        findings = [answer]
        entity_out = str(driver_value or entity)
    result = _result(
        answer=answer,
        rows=_preview_rows(matched),
        findings=findings,
        recommendations=[],
        charts=[],
        active_filter=active_filter,
        intent=AnalysisIntent.LOOKUP,
    )
    result["last_entity"] = entity_out
    return result


def _records(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
    *,
    last_entity: str | None = None,
) -> dict[str, Any]:
    entity = _extract_entity_name(question) or last_entity
    driver_col = _find_column(frame, ("driver_name", "driver"))
    matched = frame
    if entity and driver_col:
        matched = frame[frame[driver_col].astype(str).str.contains(entity, case=False, na=False)]
    answer = (
        f"Records from {stored.filename}"
        + (f" for {entity}" if entity else "")
        + f" ({len(matched)} row(s))."
    )
    result = _result(
        answer=answer,
        rows=_preview_rows(matched, limit=20),
        findings=[f"Returned {len(matched)} record(s) from the uploaded dataframe."],
        recommendations=[],
        charts=[],
        active_filter=active_filter,
        intent=AnalysisIntent.RECORDS,
    )
    result["last_entity"] = entity
    return result


def _root_cause(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    findings = _outlier_findings(frame) + _business_insights(frame)
    dimension = _infer_dimension(question, frame)
    metric = _infer_metric(question, frame)
    evidence = []
    if dimension in frame.columns and metric in frame.columns:
        grouped = (
            frame.groupby(dimension, dropna=False)[metric]
            .sum(numeric_only=True)
            .reset_index()
            .sort_values(metric, ascending=True)
        )
        if not grouped.empty:
            target = grouped.iloc[0]
            evidence.append(f"{target[dimension]} has the weakest {metric} ({target[metric]}).")
            rows = _preview_rows(grouped.head(5))
        else:
            rows = _preview_rows(frame)
    else:
        rows = _preview_rows(frame)

    if not findings and not evidence:
        answer = (
            f"Root-cause review of {stored.filename}:\n"
            "- The uploaded data does not show a clear dominant failure signature.\n"
            "- Inspect delayed/failed metrics or filter to a specific hub/driver for deeper diagnosis."
        )
        findings = ["No dominant failure signature was found in the uploaded dataframe."]
    else:
        answer = (
            f"Root-cause review of {stored.filename}:\n"
            f"{_bullets(evidence + findings)}"
        )
    return _result(
        answer=answer,
        rows=rows,
        findings=evidence + findings,
        recommendations=_data_recommendations(frame, evidence + findings),
        charts=build_charts(frame, question=question, max_charts=1),
        active_filter=active_filter,
        intent=AnalysisIntent.ROOT_CAUSE,
        root_cause={"issue": (evidence + findings)[0] if (evidence or findings) else None, "main_causes": evidence + findings},
    )


def _recommendation(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    findings = _business_insights(frame) + _outlier_findings(frame)
    recommendations = _data_recommendations(frame, findings)
    answer = (
        f"Recommendations based on {stored.filename}:\n"
        f"{_bullets(recommendations)}\n\n"
        f"Evidence:\n{_bullets(findings) if findings else '- Review the uploaded operational metrics for local exceptions.'}"
    )
    return _result(
        answer=answer,
        rows=_preview_rows(frame),
        findings=findings,
        recommendations=recommendations,
        charts=build_charts(frame, question=question, max_charts=1),
        active_filter=active_filter,
        intent=AnalysisIntent.RECOMMENDATION,
    )


def _forecast(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    metric = _infer_metric(question, frame)
    if metric not in frame.columns or not pd.api.types.is_numeric_dtype(frame[metric]):
        answer = f"Forecasting requires a numeric metric. Available columns: {', '.join(map(str, frame.columns))}."
        return _result(answer=answer, rows=_preview_rows(frame), findings=[], recommendations=[], charts=[], active_filter=active_filter, intent=AnalysisIntent.FORECAST)

    series = pd.to_numeric(frame[metric], errors="coerce").dropna()
    mean = float(series.mean()) if not series.empty else 0.0
    recent = float(series.tail(max(1, len(series) // 5)).mean()) if not series.empty else 0.0
    projection = round((mean * 0.4) + (recent * 0.6), 2)
    answer = (
        f"Simple forecast for {metric} using {stored.filename}:\n"
        f"- Historical average: {round(mean, 2)}\n"
        f"- Recent average: {round(recent, 2)}\n"
        f"- Near-term projection: {projection}\n"
        "This is a lightweight trend estimate from the uploaded dataframe, not a trained model."
    )
    return _result(
        answer=answer,
        rows=_preview_rows(frame),
        findings=[f"Projected {metric} near-term value: {projection}."],
        recommendations=["Validate the projection against the next operational day before staffing changes."],
        charts=build_charts(frame, question="trend line chart", max_charts=1),
        active_filter=active_filter,
        intent=AnalysisIntent.FORECAST,
        metric=metric,
    )


def _narrative(
    question: str,
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> dict[str, Any]:
    insights = _business_insights(frame)
    patterns = _patterns(frame)
    answer = (
        f"What happened in {stored.filename}:\n"
        f"{_bullets(insights + patterns) if (insights or patterns) else '- The uploaded dataset is available, but no strong narrative signal was found.'}"
    )
    return _result(
        answer=answer,
        rows=_preview_rows(frame),
        findings=insights + patterns,
        recommendations=_data_recommendations(frame, insights + patterns),
        charts=build_charts(frame, question=question, max_charts=1),
        active_filter=active_filter,
        intent=AnalysisIntent.NARRATIVE,
    )


def _dataset_overview(
    stored: StoredAttachment,
    frame: pd.DataFrame,
    active_filter: dict[str, Any],
) -> str:
    missing = int(frame.isna().sum().sum())
    lines = [
        f"- File: {stored.filename}",
        f"- Rows: {len(frame)}",
        f"- Columns: {len(frame.columns)} ({', '.join(map(str, frame.columns[:8]))}{'...' if len(frame.columns) > 8 else ''})",
        f"- Missing values: {missing}",
    ]
    if stored.sheet_names:
        lines.append(f"- Sheets: {', '.join(stored.sheet_names)} (active: {stored.active_sheet})")
    if active_filter:
        lines.append(f"- Active filters: {active_filter}")
    return "\n".join(lines)


def _important_metrics(frame: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    for column in _numeric_columns(frame)[:4]:
        series = pd.to_numeric(frame[column], errors="coerce")
        findings.append(
            f"{column}: min={_fmt(series.min())}, max={_fmt(series.max())}, mean={_fmt(series.mean())}, sum={_fmt(series.sum())}"
        )
    return findings


def _patterns(frame: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    dimension = _find_column(frame, ("hub", "warehouse", "driver_name", "driver"))
    metric = _infer_metric("", frame)
    if dimension and metric in frame.columns:
        grouped = frame.groupby(dimension, dropna=False)[metric].sum(numeric_only=True).sort_values(ascending=False)
        if not grouped.empty:
            findings.append(f"{grouped.index[0]} leads on {metric} ({_fmt(grouped.iloc[0])}).")
            if len(grouped) > 1:
                findings.append(f"{grouped.index[-1]} trails on {metric} ({_fmt(grouped.iloc[-1])}).")
    return findings


def _outlier_findings(frame: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    for column in _numeric_columns(frame)[:3]:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(series) < 3:
            continue
        mean = series.mean()
        std = series.std(ddof=0)
        if not std or pd.isna(std):
            continue
        outliers = series[(series - mean).abs() > (2 * std)]
        if not outliers.empty:
            findings.append(
                f"{column} has {len(outliers)} outlier value(s) beyond 2σ (mean={_fmt(mean)}, std={_fmt(std)})."
            )
    return findings


def _business_insights(frame: pd.DataFrame) -> list[str]:
    insights = _patterns(frame)
    delay_col = _find_column(frame, ("delayed_pickups", "delay_rate", "delayed"))
    fail_col = _find_column(frame, ("failed_pickups", "failure_rate", "failed"))
    if delay_col and pd.api.types.is_numeric_dtype(frame[delay_col]):
        total = pd.to_numeric(frame[delay_col], errors="coerce").sum()
        insights.append(f"Total {delay_col}: {_fmt(total)}")
    if fail_col and pd.api.types.is_numeric_dtype(frame[fail_col]):
        total = pd.to_numeric(frame[fail_col], errors="coerce").sum()
        insights.append(f"Total {fail_col}: {_fmt(total)}")
    return insights


def _data_recommendations(frame: pd.DataFrame, findings: list[str]) -> list[str]:
    recommendations: list[str] = []
    dimension = _find_column(frame, ("hub", "warehouse"))
    metric = _infer_metric("", frame)
    if dimension and metric in frame.columns:
        grouped = frame.groupby(dimension, dropna=False)[metric].sum(numeric_only=True).sort_values()
        if not grouped.empty:
            recommendations.append(
                f"Prioritize coaching and capacity review at {grouped.index[0]} based on {metric}."
            )
    if any("outlier" in finding.lower() for finding in findings):
        recommendations.append("Investigate outlier rows before changing staffing or routing plans.")
    if not recommendations:
        recommendations.append("Validate the uploaded report against live hub operations and continue monitoring key metrics.")
    return recommendations


def _normalize_column_key(value: Any) -> str:
    text = str(value).replace("\n", "").replace("\r", "").replace(" ", "")
    return text.casefold()


def _match_columns_in_question(question: str, frame: pd.DataFrame) -> list[str]:
    """Return dataframe columns whose names appear in the user question."""
    if frame is None or frame.empty:
        return []
    q_raw = question.replace("\n", "").replace("\r", "")
    q_compact = _normalize_column_key(question)
    matches: list[tuple[int, str]] = []
    for column in frame.columns:
        col_str = str(column)
        col_compact = _normalize_column_key(column)
        if not col_compact:
            continue
        if col_str in q_raw or col_compact in q_compact:
            matches.append((len(col_compact), col_str))
            continue
        # Match meaningful substrings (e.g. Chinese headers without full phrase).
        if len(col_compact) >= 2 and col_compact in q_compact:
            matches.append((len(col_compact), col_str))
    matches.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    ordered: list[str] = []
    for _, column in matches:
        if column not in seen:
            seen.add(column)
            ordered.append(column)
    return ordered


def _wants_row_count(question: str, frame: pd.DataFrame) -> bool:
    """Prefer COUNT(*) when the ask is package/volume/count and no package metric exists."""
    normalized = question.lower()
    count_signals = (
        "how many",
        "count",
        "number of",
        "packages",
        "package volume",
        "volume",
        "for each",
        "distribution",
    )
    if not any(signal in normalized for signal in count_signals):
        return False
    if _find_column(frame, ("package_count", "packages")):
        return False
    return True


def _infer_dimension(question: str, frame: pd.DataFrame) -> str:
    matched = _match_columns_in_question(question, frame)
    if matched:
        # Prefer non-numeric / high-cardinality address-like columns for grouping.
        for column in matched:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                return column
        return matched[0]

    normalized = question.lower()
    if "driver" in normalized:
        return _find_column(frame, ("driver_name", "driver")) or str(frame.columns[0])
    if "customer" in normalized:
        return _find_column(frame, ("customer_name", "customer")) or str(frame.columns[0])
    if "hub" in normalized or "warehouse" in normalized:
        return _find_column(frame, ("hub", "warehouse")) or str(frame.columns[0])
    if "address" in normalized or "地址" in question:
        address_col = _find_column(
            frame,
            ("发件人详细地址", "收件人详细地址", "详细地址", "address", "sender_address", "receiver_address"),
        )
        if address_col:
            return address_col
        for column in frame.columns:
            col_l = str(column).lower()
            if "地址" in str(column) or "address" in col_l:
                return str(column)
    return (
        _find_column(frame, ("hub", "warehouse", "driver_name", "driver", "customer_name", "customer"))
        or str(frame.columns[0])
    )


def _infer_metric(question: str, frame: pd.DataFrame) -> str:
    normalized = question.lower()
    preferred = (
        ("completion", ("completion_rate",)),
        ("delay", ("delayed_pickups", "delay_rate", "delayed")),
        ("fail", ("failed_pickups", "failure_rate", "failed")),
        ("package", ("package_count", "packages")),
        ("pickup", ("pickup_count", "pickups")),
        ("performance", ("package_count", "pickup_count", "completion_rate")),
        ("weight", ("总重量(kg)", "总重量", "weight", "重量")),
    )
    for token, columns in preferred:
        if token in normalized or (token == "weight" and "重量" in question):
            found = _find_column(frame, columns)
            if found:
                return found
    matched = _match_columns_in_question(question, frame)
    for column in matched:
        if pd.api.types.is_numeric_dtype(frame[column]):
            return column
    numeric = _numeric_columns(frame)
    return numeric[0] if numeric else str(frame.columns[-1])


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    preferred = (
        "package_count",
        "packages",
        "pickup_count",
        "pickups",
        "completion_rate",
        "delay_rate",
        "failure_rate",
        "delayed_pickups",
        "failed_pickups",
    )
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    ordered = [column for column in preferred if column in numeric]
    ordered.extend(column for column in numeric if column not in ordered)
    return ordered


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized_map = {_normalize_column_key(column): column for column in frame.columns}
    for candidate in candidates:
        key = _normalize_column_key(candidate)
        if key in normalized_map:
            return str(normalized_map[key])
    for candidate in candidates:
        key = _normalize_column_key(candidate)
        for column_key, column in normalized_map.items():
            if key and key in column_key:
                return str(column)
    return None


def _extract_entity_name(question: str) -> str | None:
    match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question)
    if match:
        return match.group(1)
    for token in ("chicago", "atlanta", "ord", "new york", "los angeles"):
        if token in question.lower():
            return "ORD" if token == "ord" else token.title()
    return None


def _mentions_filter(question: str) -> bool:
    normalized = question.lower()
    return any(phrase in normalized for phrase in ("only ", "filter", " for chicago", " for ord", " in chicago", " in ord"))


def _preview_rows(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    cleaned = frame.head(limit).copy()
    for column in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].astype(str)
    return cleaned.where(pd.notnull(cleaned), None).to_dict(orient="records")


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _result(
    *,
    answer: str,
    rows: list[dict[str, Any]],
    findings: list[str],
    recommendations: list[str],
    charts: list[dict[str, Any]],
    active_filter: dict[str, Any],
    intent: AnalysisIntent,
    dimension: str | None = None,
    metric: str | None = None,
    root_cause: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "rows": rows,
        "findings": findings,
        "recommendations": recommendations,
        "charts": charts,
        "active_filter": active_filter,
        "intent": intent.value,
        "dimension": dimension,
        "metric": metric,
        "root_cause": root_cause or {},
    }
