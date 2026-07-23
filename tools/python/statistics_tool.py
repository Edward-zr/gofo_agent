"""StatisticsTool — compute metrics from DataFrames (LLM never calculates)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

import config
from core.logger import get_logger

logger = get_logger("python.statistics")


def _debug(payload: dict[str, Any]) -> None:
    if config.DEBUG or getattr(config, "PYTHON_TOOLS_DEBUG", False):
        print("----------------------------------")
        print("StatisticsTool")
        for key, value in payload.items():
            print(f"  {key}: {value}")
        print("----------------------------------")
    logger.info("StatisticsTool %s", payload)


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if pd.api.types.is_object_dtype(frame[col]) or isinstance(frame[col].dtype, pd.CategoricalDtype):
            cols.append(col)
        elif pd.api.types.is_bool_dtype(frame[col]):
            cols.append(col)
    return cols


class StatisticsTool:
    """Summary stats, ratios, growth, rankings, groupby analysis, optional correlation."""

    def compute(
        self,
        frame: pd.DataFrame | None,
        *,
        question: str = "",
        include_correlation: bool | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        working = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        text = (question or "").lower()
        executed: list[str] = []
        statistics: dict[str, Any] = {
            "row_count": int(len(working)),
            "column_count": int(len(working.columns)) if not working.empty else 0,
        }

        if working.empty:
            result = {
                "tool": "STATISTICS",
                "statistics": statistics,
                "summary": statistics,
                "functions_executed": [],
                "execution_time_ms": int((time.perf_counter() - started) * 1000),
                "capability": "python_statistics",
            }
            _debug(result)
            return result

        numeric = _numeric_columns(working)
        categorical = _categorical_columns(working)

        # Summary statistics
        summary_stats: dict[str, Any] = {}
        for col in numeric[:10]:
            series = working[col].dropna()
            if series.empty:
                continue
            summary_stats[col] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "sum": float(series.sum()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "count": int(series.count()),
            }
            executed.append(f"summary:{col}")
        if summary_stats:
            statistics["summary"] = summary_stats
            statistics["numeric_means"] = {
                col: values["mean"] for col, values in summary_stats.items()
            }

        # Percentages / ratios for status-like columns
        if "status" in working.columns:
            counts = working["status"].astype(str).value_counts(dropna=False)
            total = float(counts.sum()) or 1.0
            percentages = {str(k): float(v / total) for k, v in counts.items()}
            statistics["status_percentages"] = percentages
            if "Completed" in counts.index or any(str(i).lower() == "completed" for i in counts.index):
                completed = sum(
                    int(v) for k, v in counts.items() if str(k).lower() == "completed"
                )
                statistics["completion_rate"] = float(completed / total)
                executed.append("ratio:completion_rate")
            executed.append("percentages:status")

        # Focus metric
        for col in numeric:
            lower = str(col).lower()
            if any(token in lower for token in ("success", "completion", "rate", "kpi")):
                statistics["focus_metric"] = col
                statistics["focus_mean"] = float(working[col].mean())
                executed.append(f"focus:{col}")
                break

        # Growth / WoW
        period_cols = [
            col
            for col in working.columns
            if str(col).lower() in {"week", "period", "date_range", "pickup_date", "date"}
        ]
        if period_cols and numeric and any(
            token in text for token in ("growth", "wow", "week", "trend", "increase", "decrease", "compare")
        ):
            period_col = period_cols[0]
            metric_col = numeric[0]
            grouped = working.groupby(period_col, dropna=True)[metric_col].mean().dropna()
            if len(grouped) >= 2:
                values = list(grouped.values)
                prev, curr = float(values[-2]), float(values[-1])
                statistics["growth"] = {
                    "period_col": period_col,
                    "metric": metric_col,
                    "previous": prev,
                    "current": curr,
                    "change": curr - prev,
                    "change_pct": ((curr - prev) / prev) if prev else None,
                }
                statistics["wow_change"] = statistics["growth"]["change"]
                statistics["wow_change_pct"] = statistics["growth"]["change_pct"]
                executed.append("growth_rate")

        # Ranking / groupby analysis
        if categorical and numeric:
            group_col = categorical[0]
            for preferred in ("hub", "driver_name", "driver", "customer_name", "city", "status"):
                matches = [c for c in categorical if preferred in str(c).lower()]
                if matches:
                    group_col = matches[0]
                    break
            metric_col = numeric[0]
            ranked = (
                working.groupby(group_col, dropna=False)[metric_col]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            statistics["ranking"] = {
                "by": group_col,
                "metric": metric_col,
                "top": [
                    {"entity": str(idx), "value": float(val)}
                    for idx, val in ranked.items()
                ],
            }
            executed.append(f"ranking:{group_col}")
            if "lowest" in text or "worst" in text:
                bottom = (
                    working.groupby(group_col, dropna=False)[metric_col]
                    .mean()
                    .sort_values(ascending=True)
                    .head(5)
                )
                statistics["ranking"]["bottom"] = [
                    {"entity": str(idx), "value": float(val)} for idx, val in bottom.items()
                ]
                executed.append("ranking:bottom")

        # Correlation when requested
        want_corr = include_correlation
        if want_corr is None:
            want_corr = "correlation" in text or "correlate" in text
        if want_corr and len(numeric) >= 2:
            corr = working[numeric[:8]].corr(numeric_only=True)
            statistics["correlation"] = {
                str(a): {str(b): float(corr.loc[a, b]) for b in corr.columns}
                for a in corr.index
            }
            executed.append("correlation")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = {
            "tool": "STATISTICS",
            "statistics": statistics,
            "summary": statistics,  # backward compatible with PYTHON consumers
            "dataframe": working,
            "rows": working.head(100).to_dict(orient="records"),
            "shape": {"rows": int(len(working)), "columns": int(len(working.columns))},
            "functions_executed": executed,
            "execution_time_ms": elapsed_ms,
            "capability": "python_statistics",
        }
        _debug(
            {
                "functions_executed": executed,
                "statistics_keys": list(statistics.keys()),
                "shape": result["shape"],
                "execution_time_ms": elapsed_ms,
            }
        )
        return result


def run_statistics(
    frame: pd.DataFrame | None,
    *,
    question: str = "",
    include_correlation: bool | None = None,
) -> dict[str, Any]:
    return StatisticsTool().compute(
        frame, question=question, include_correlation=include_correlation
    )
