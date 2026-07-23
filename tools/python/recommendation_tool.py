"""RecommendationTool — business insights from computed analytics (never raw SQL)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

import config
from core.logger import get_logger
from tools.analysis.recommender import recommend

logger = get_logger("python.recommendation")


def _debug(payload: dict[str, Any]) -> None:
    if config.DEBUG or getattr(config, "PYTHON_TOOLS_DEBUG", False):
        print("----------------------------------")
        print("RecommendationTool")
        for key, value in payload.items():
            print(f"  {key}: {value}")
        print("----------------------------------")
    logger.info("RecommendationTool %s", payload)


def _rows_from_frame(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return frame.head(100).to_dict(orient="records")


def _insights_from_statistics(statistics: dict[str, Any] | None) -> list[str]:
    stats = statistics or {}
    insights: list[str] = []

    ranking = stats.get("ranking") or {}
    top = ranking.get("top") or []
    bottom = ranking.get("bottom") or []
    by = ranking.get("by")
    metric = ranking.get("metric")
    if top:
        leader = top[0]
        insights.append(
            f"High-performing {by or 'entity'}: {leader.get('entity')} "
            f"({metric or 'metric'}={leader.get('value')})."
        )
    if bottom:
        trailer = bottom[0]
        insights.append(
            f"Low-performing {by or 'entity'}: {trailer.get('entity')} "
            f"({metric or 'metric'}={trailer.get('value')}) — prioritize coaching / capacity review."
        )

    completion = stats.get("completion_rate")
    if isinstance(completion, (int, float)):
        pct = float(completion) * 100
        if pct < 85:
            insights.append(
                f"KPI observation: completion rate is {pct:.1f}% — below an 85% operational target."
            )
        else:
            insights.append(f"KPI observation: completion rate is healthy at {pct:.1f}%.")

    growth = stats.get("growth") or {}
    change_pct = growth.get("change_pct")
    if isinstance(change_pct, (int, float)):
        direction = "increased" if change_pct >= 0 else "decreased"
        insights.append(
            f"Growth observation: {growth.get('metric')} {direction} by {abs(change_pct) * 100:.1f}% "
            f"versus the prior period."
        )

    status_pct = stats.get("status_percentages") or {}
    failed = next((v for k, v in status_pct.items() if str(k).lower() == "failed"), None)
    delayed = next((v for k, v in status_pct.items() if str(k).lower() == "delayed"), None)
    if isinstance(failed, (int, float)) and failed >= 0.1:
        insights.append(
            f"Capacity suggestion: failure share is {failed * 100:.1f}% — review failed pickup reasons "
            "and rebalance overloaded hubs/drivers."
        )
    if isinstance(delayed, (int, float)) and delayed >= 0.15:
        insights.append(
            f"Operational insight: delay share is {delayed * 100:.1f}% — check route load and driver assignment."
        )

    return insights


class RecommendationTool:
    """Generate business recommendations from statistics / transformed results."""

    def recommend(
        self,
        *,
        question: str = "",
        statistics: dict[str, Any] | None = None,
        frame: pd.DataFrame | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        executed: list[str] = []
        row_data = rows if rows is not None else _rows_from_frame(frame)

        recommendations = _insights_from_statistics(statistics)
        if recommendations:
            executed.append("insights_from_statistics")

        template = recommend(issue=question, rows=row_data)
        executed.append("template_recommend")
        if template and template not in recommendations:
            recommendations.append(template)

        # Deduplicate while preserving order
        deduped: list[str] = []
        for item in recommendations:
            text = str(item).strip()
            if text and text not in deduped:
                deduped.append(text)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        answer = " ".join(deduped) if deduped else None
        result = {
            "tool": "RECOMMENDATION",
            "recommendations": deduped,
            "recommendation": deduped[0] if deduped else None,
            "answer": answer,
            "functions_executed": executed,
            "execution_time_ms": elapsed_ms,
            "capability": "python_recommendation",
        }
        _debug(
            {
                "recommendations": deduped,
                "functions_executed": executed,
                "execution_time_ms": elapsed_ms,
            }
        )
        return result


def run_recommendations(
    *,
    question: str = "",
    statistics: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return RecommendationTool().recommend(
        question=question,
        statistics=statistics,
        frame=frame,
        rows=rows,
    )
