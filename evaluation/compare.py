"""Regression comparison against a previous evaluation run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Higher is better except for these inverted metrics.
LOWER_IS_BETTER = {
    "avg_latency_ms",
    "latency_sec",
    "avg_tokens",
    "average_tokens",
    "avg_cost_usd",
    "estimated_cost_per_query",
    "exception_rate",
    "hallucination_rate",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_previous_combined(current_path: Path | None = None) -> Path | None:
    """Find the newest combined results file older than current (if any)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RESULTS_DIR.glob("combined_*.json"))
    if not files:
        return None
    if current_path is None:
        return files[-2] if len(files) >= 2 else None
    prior = [f for f in files if f.resolve() != current_path.resolve()]
    return prior[-1] if prior else None


def _delta(old: float, new: float, *, lower_is_better: bool = False) -> dict[str, Any]:
    change = new - old
    improved = change < 0 if lower_is_better else change > 0
    regressing = change > 0 if lower_is_better else change < 0
    arrow = "↑" if improved else ("↓" if regressing else "→")
    return {
        "previous": old,
        "current": new,
        "delta": round(change, 4),
        "arrow": arrow,
        "improved": improved,
        "regressed": regressing,
    }


def compare_scorecards(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare overall/benchmark/scenario scorecards."""
    if not previous:
        return {
            "has_previous": False,
            "improvements": [],
            "regressions": [],
            "metrics": {},
        }

    cur_overall = current.get("overall") or {}
    prev_overall = previous.get("overall") or {}
    metrics: dict[str, Any] = {}
    keys = sorted(set(cur_overall) | set(prev_overall))
    improvements: list[str] = []
    regressions: list[str] = []

    for key in keys:
        if not isinstance(cur_overall.get(key), (int, float)):
            continue
        if not isinstance(prev_overall.get(key), (int, float)):
            continue
        lower = key in LOWER_IS_BETTER
        row = _delta(float(prev_overall[key]), float(cur_overall[key]), lower_is_better=lower)
        metrics[key] = row
        label = f"{key}: {row['previous']} → {row['current']} ({row['arrow']})"
        if row["improved"] and abs(row["delta"]) >= 0.01:
            improvements.append(label)
        elif row["regressed"] and abs(row["delta"]) >= 0.01:
            regressions.append(label)

    # Suite-level benchmark accuracy
    cur_suites = (current.get("benchmark") or {}).get("summary", {}).get("suite_accuracy") or {}
    prev_suites = (previous.get("benchmark") or {}).get("summary", {}).get("suite_accuracy") or {}
    for suite, value in cur_suites.items():
        old = prev_suites.get(suite)
        if old is None:
            continue
        row = _delta(float(old), float(value))
        metrics[f"suite_{suite}"] = row
        label = f"{suite} accuracy: {row['previous']} → {row['current']} ({row['arrow']})"
        if row["improved"] and abs(row["delta"]) >= 0.01:
            improvements.append(label)
        elif row["regressed"] and abs(row["delta"]) >= 0.01:
            regressions.append(label)

    return {
        "has_previous": True,
        "improvements": improvements,
        "regressions": regressions,
        "metrics": metrics,
    }


def load_latest_combined() -> dict[str, Any] | None:
    path = RESULTS_DIR / "latest_combined.json"
    if not path.exists():
        return None
    return _load(path)
