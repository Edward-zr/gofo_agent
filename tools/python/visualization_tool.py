"""VisualizationTool — generate charts from DataFrames via matplotlib."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

import config
from core.logger import get_logger

logger = get_logger("python.visualization")


def _debug(payload: dict[str, Any]) -> None:
    if config.DEBUG or getattr(config, "PYTHON_TOOLS_DEBUG", False):
        print("----------------------------------")
        print("VisualizationTool")
        for key, value in payload.items():
            print(f"  {key}: {value}")
        print("----------------------------------")
    logger.info("VisualizationTool %s", payload)


def _chart_metadata(charts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact metadata for AgentState / Generator (no huge base64 dump in logs)."""
    meta: list[dict[str, Any]] = []
    for chart in charts:
        meta.append(
            {
                "type": chart.get("type"),
                "title": chart.get("title"),
                "format": chart.get("format", "png"),
                "renderer": chart.get("renderer", "matplotlib"),
                "has_image": bool(chart.get("image_base64")),
                "x": chart.get("x"),
                "y": chart.get("y"),
            }
        )
    return meta


class VisualizationTool:
    """Create line / bar / pie / scatter / histogram charts from a DataFrame."""

    def visualize(
        self,
        frame: pd.DataFrame | None,
        *,
        question: str = "",
        max_charts: int = 3,
        chart_types: list[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        from tools.files.charts import build_charts

        working = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        charts = build_charts(working, question=question, max_charts=max_charts)

        if chart_types:
            wanted = {item.lower() for item in chart_types}
            filtered = [c for c in charts if str(c.get("type", "")).lower() in wanted]
            if filtered:
                charts = filtered

        metadata = _chart_metadata(charts)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = {
            "tool": "VISUALIZATION",
            "charts": charts,
            "chart_metadata": metadata,
            "dataframe": working,
            "shape": {"rows": int(len(working)), "columns": int(len(working.columns))},
            "functions_executed": [f"build_charts:{c.get('type')}" for c in charts] or ["build_charts:none"],
            "execution_time_ms": elapsed_ms,
            "capability": "visualization",
        }
        _debug(
            {
                "charts_created": [m.get("type") for m in metadata],
                "chart_metadata": metadata,
                "shape": result["shape"],
                "execution_time_ms": elapsed_ms,
            }
        )
        return result


def run_visualization(
    frame: pd.DataFrame | None,
    *,
    question: str = "",
    max_charts: int = 3,
    chart_types: list[str] | None = None,
) -> dict[str, Any]:
    return VisualizationTool().visualize(
        frame, question=question, max_charts=max_charts, chart_types=chart_types
    )
