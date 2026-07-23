"""TransformationTool — prepare DataFrames for analytics (never invents numbers)."""

from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd

import config
from core.logger import get_logger

logger = get_logger("python.transformation")


def _debug(payload: dict[str, Any]) -> None:
    if config.DEBUG or getattr(config, "PYTHON_TOOLS_DEBUG", False):
        print("----------------------------------")
        print("TransformationTool")
        for key, value in payload.items():
            print(f"  {key}: {value}")
        print("----------------------------------")
    logger.info("TransformationTool %s", payload)


def _infer_groupby_cols(frame: pd.DataFrame, question: str) -> list[str]:
    text = question.lower()
    preferred = ("hub", "driver", "driver_name", "customer", "customer_name", "city", "status", "week", "period")
    cols: list[str] = []
    for name in preferred:
        if name in text:
            for col in frame.columns:
                if str(col).lower() == name or name in str(col).lower():
                    cols.append(col)
                    break
    if cols:
        return cols[:2]
    # Heuristic: first low-cardinality object column
    for col in frame.columns:
        if pd.api.types.is_object_dtype(frame[col]) or isinstance(frame[col].dtype, pd.CategoricalDtype):
            nunique = int(frame[col].nunique(dropna=True))
            if 1 < nunique <= max(50, len(frame) // 2):
                return [col]
    return []


def _infer_metric_cols(frame: pd.DataFrame) -> list[str]:
    numeric = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    preferred = []
    for col in numeric:
        lower = str(col).lower()
        if any(token in lower for token in ("rate", "count", "pickup", "package", "volume", "fail", "complete")):
            preferred.append(col)
    return preferred or numeric[:3]


class TransformationTool:
    """Filter / sort / group / pivot / merge / aggregate / clean DataFrames."""

    def transform(
        self,
        frame: pd.DataFrame | None,
        *,
        question: str = "",
        operations: list[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        source = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        executed: list[str] = []
        text = (question or "").lower()
        ops = {op.lower() for op in (operations or [])}

        if source.empty:
            result = {
                "tool": "TRANSFORM",
                "dataframe": source,
                "transformed_dataframe": source,
                "rows": [],
                "shape": {"rows": 0, "columns": 0},
                "functions_executed": [],
                "execution_time_ms": int((time.perf_counter() - started) * 1000),
                "capability": "python_transform",
            }
            _debug(result)
            return result

        working = source.copy()

        # Missing values
        if "fillna" in ops or "missing" in text or working.isna().any().any():
            for col in working.columns:
                if pd.api.types.is_numeric_dtype(working[col]):
                    working[col] = working[col].fillna(0)
                else:
                    working[col] = working[col].fillna("")
            executed.append("fillna")

        # Date conversion only when asked or clear date columns exist (skip week labels).
        date_like = [
            col
            for col in working.columns
            if any(token in str(col).lower() for token in ("date", "created_at", "timestamp"))
        ]
        if "date" in ops or re.search(r"\b(date|datetime|timestamp)\b", text) or date_like:
            for col in date_like:
                converted = pd.to_datetime(working[col], errors="coerce")
                if converted.notna().any():
                    working[col] = converted
                    executed.append(f"to_datetime:{col}")

        # Rename common aliases
        rename_map = {}
        for col in list(working.columns):
            lower = str(col).lower()
            if lower in {"hub_name", "warehouse", "station"} and "hub" not in working.columns:
                rename_map[col] = "hub"
            if lower in {"driver", "name"} and "driver_name" not in working.columns and "driver" in text:
                rename_map[col] = "driver_name"
        if rename_map:
            working = working.rename(columns=rename_map)
            executed.append(f"rename:{rename_map}")

        # Filter by status keywords in question
        if "status" in working.columns:
            for status in ("Failed", "Delayed", "Completed"):
                if status.lower() in text:
                    working = working[working["status"].astype(str).str.lower() == status.lower()]
                    executed.append(f"filter:status={status}")
                    break

        # Sort
        metric_cols = _infer_metric_cols(working)
        if metric_cols and ("sort" in ops or "rank" in text or "top" in text or "lowest" in text or "highest" in text):
            ascending = any(token in text for token in ("lowest", "worst", "bottom", "ascending"))
            working = working.sort_values(by=metric_cols[0], ascending=ascending, na_position="last")
            executed.append(f"sort:{metric_cols[0]}:{'asc' if ascending else 'desc'}")

        # GroupBy aggregate when question implies ranking/performance by entity
        group_cols = _infer_groupby_cols(working, text)
        if group_cols and metric_cols and (
            "groupby" in ops
            or "group" in text
            or any(token in text for token in ("by hub", "by driver", "rank", "performance", "compare"))
        ):
            agg = {col: "sum" if "rate" not in str(col).lower() else "mean" for col in metric_cols}
            grouped = working.groupby(group_cols, dropna=False).agg(agg).reset_index()
            working = grouped
            executed.append(f"groupby:{group_cols}->agg({list(agg)})")

        # Pivot if requested
        if "pivot" in ops or "pivot" in text:
            if len(working.columns) >= 3:
                index_col = working.columns[0]
                columns_col = working.columns[1]
                values_col = metric_cols[0] if metric_cols else working.columns[2]
                try:
                    pivoted = working.pivot_table(
                        index=index_col,
                        columns=columns_col,
                        values=values_col,
                        aggfunc="sum",
                        fill_value=0,
                    ).reset_index()
                    working = pivoted
                    executed.append(f"pivot:{index_col}/{columns_col}/{values_col}")
                except Exception as exc:  # noqa: BLE001
                    executed.append(f"pivot_skipped:{exc}")

        # Limit huge frames for downstream tools
        if len(working) > 5000:
            working = working.head(5000)
            executed.append("limit:5000")

        if not executed:
            executed.append("identity_copy")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = {
            "tool": "TRANSFORM",
            "dataframe": source,
            "transformed_dataframe": working,
            "rows": working.head(200).to_dict(orient="records"),
            "shape": {"rows": int(len(working)), "columns": int(len(working.columns))},
            "source_shape": {"rows": int(len(source)), "columns": int(len(source.columns))},
            "functions_executed": executed,
            "execution_time_ms": elapsed_ms,
            "capability": "python_transform",
        }
        _debug(
            {
                "functions_executed": executed,
                "shape": result["shape"],
                "execution_time_ms": elapsed_ms,
            }
        )
        return result


def run_transformation(
    frame: pd.DataFrame | None,
    *,
    question: str = "",
    operations: list[str] | None = None,
) -> dict[str, Any]:
    return TransformationTool().transform(frame, question=question, operations=operations)
