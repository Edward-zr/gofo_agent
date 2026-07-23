"""Load benchmark and scenario datasets from evaluation/datasets/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_ROOT / "datasets"

BENCHMARK_FILES = {
    "sop": "sop_questions.json",
    "sql": "sql_questions.json",
    "general": "general_questions.json",
    "followup": "followup_questions.json",
    "memory": "memory_questions.json",
    "upload": "upload_questions.json",
}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_benchmark_suite(
    *,
    suites: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load independent benchmark questions. Each item gets a ``suite`` field."""
    selected = suites or list(BENCHMARK_FILES.keys())
    items: list[dict[str, Any]] = []
    for suite in selected:
        filename = BENCHMARK_FILES.get(suite)
        if not filename:
            continue
        path = DATASETS_DIR / filename
        if not path.exists():
            continue
        payload = _load_json(path)
        rows = payload if isinstance(payload, list) else payload.get("questions") or []
        for row in rows:
            item = dict(row)
            item["suite"] = suite
            items.append(item)
            if limit is not None and len(items) >= limit:
                return items
    return items


def load_scenarios(*, limit: int | None = None) -> list[dict[str, Any]]:
    """Load multi-turn scenario definitions."""
    path = DATASETS_DIR / "scenarios.json"
    if not path.exists():
        return []
    payload = _load_json(path)
    rows = payload if isinstance(payload, list) else payload.get("scenarios") or []
    if limit is not None:
        return list(rows)[:limit]
    return list(rows)


def dataset_counts() -> dict[str, int]:
    """Return counts per dataset file for reporting."""
    counts: dict[str, int] = {}
    for suite, filename in BENCHMARK_FILES.items():
        path = DATASETS_DIR / filename
        if not path.exists():
            counts[suite] = 0
            continue
        payload = _load_json(path)
        rows = payload if isinstance(payload, list) else payload.get("questions") or []
        counts[suite] = len(rows)
    scenarios = load_scenarios()
    counts["scenarios"] = len(scenarios)
    return counts
