"""Prompt Experiment Framework — benchmark multiple prompt versions automatically."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

import config
from core.logger import get_logger
from core.prompt_registry import get_prompt_registry, reset_prompt_registry
from core.prompt_manager import reset_prompt_manager

logger = get_logger("prompt_experiments")

DEFAULT_PRIORITIES = [
    "accuracy",
    "sql_success",
    "scenario_success",
    "latency",
    "cost",
]

LOWER_IS_BETTER = {"latency", "cost", "latency_sec", "estimated_cost_usd", "avg_latency_ms"}


@dataclass
class PromptExperiment:
    """Definition of a prompt A/B (or A/B/C) experiment."""

    id: str
    prompt_key: str
    versions: list[str]
    description: str = ""
    priorities: list[str] = field(default_factory=lambda: list(DEFAULT_PRIORITIES))
    benchmark_limit: int | None = 30
    scenario_limit: int | None = 10
    mode: str = "mock"


def load_experiments(path: Path | str | None = None) -> list[PromptExperiment]:
    """Load experiments from prompts/experiments.yaml if present."""
    root = Path(getattr(config, "PROMPT_REGISTRY_DIR", "prompts"))
    if not root.is_absolute():
        root = Path(config.PROJECT_ROOT) / root
    path = Path(path) if path else root / "experiments.yaml"
    if not path.exists():
        return default_experiments()
    if yaml is None:
        return default_experiments()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("experiments") or []
    experiments: list[PromptExperiment] = []
    for row in rows:
        experiments.append(
            PromptExperiment(
                id=str(row.get("id") or row.get("prompt_key")),
                prompt_key=str(row["prompt_key"]),
                versions=[str(v) for v in row.get("versions") or []],
                description=str(row.get("description") or ""),
                priorities=list(row.get("priorities") or DEFAULT_PRIORITIES),
                benchmark_limit=row.get("benchmark_limit", 30),
                scenario_limit=row.get("scenario_limit", 10),
                mode=str(row.get("mode") or "mock"),
            )
        )
    return experiments or default_experiments()


def default_experiments() -> list[PromptExperiment]:
    registry = get_prompt_registry()
    experiments: list[PromptExperiment] = []
    for key in ("planner.planner_prompt", "rag.generator_prompt", "sql.generator_prompt"):
        try:
            versions = registry.list_versions(key)
        except Exception:  # noqa: BLE001
            continue
        if len(versions) >= 2:
            experiments.append(
                PromptExperiment(
                    id=f"exp_{key.replace('.', '_')}",
                    prompt_key=key,
                    versions=versions,
                    description=f"Compare versions of {key}",
                )
            )
    if not experiments:
        # Always provide a planner experiment skeleton when v1/v2 exist or will exist
        experiments.append(
            PromptExperiment(
                id="exp_planner_prompt",
                prompt_key="planner.planner_prompt",
                versions=["v1", "v2"],
                description="Planner prompt version comparison",
            )
        )
    return experiments


def _scorecard_from_eval(combined: dict[str, Any]) -> dict[str, float]:
    overall = combined.get("overall") or {}
    return {
        "accuracy": float(overall.get("benchmark_accuracy") or 0),
        "planner_accuracy": float(overall.get("planner_accuracy") or 0),
        "tool_selection_accuracy": float(overall.get("tool_orchestration") or 0),
        "sql_success": float(overall.get("sql_success") or 0),
        "retrieval_accuracy": float(overall.get("retrieval_confidence") or 0),
        "scenario_success": float(overall.get("scenario_success_rate") or 0),
        "latency": float(overall.get("latency_sec") or 0),
        "cost": float(overall.get("estimated_cost_per_query") or 0),
        "average_tokens": float(overall.get("average_tokens") or 0),
        "overall_score": float(overall.get("overall_score") or 0),
    }


def recommend_best(
    version_metrics: dict[str, dict[str, float]],
    priorities: list[str] | None = None,
) -> str:
    """Pick the best version using lexicographic priority ranking."""
    priorities = priorities or DEFAULT_PRIORITIES
    versions = list(version_metrics.keys())
    if not versions:
        raise ValueError("No versions to compare.")

    def sort_key(version: str) -> tuple:
        metrics = version_metrics[version]
        key_parts = []
        for name in priorities:
            value = float(metrics.get(name) or 0)
            # Negate higher-is-better so ascending sort picks the best first
            key_parts.append(value if name in LOWER_IS_BETTER else -value)
        return tuple(key_parts)

    return sorted(versions, key=sort_key)[0]


def run_prompt_experiment(
    experiment: PromptExperiment,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """Run evaluation suites once per prompt version and compare."""
    from evaluation.evaluator import overall_agent_score
    from evaluation.runner import run_benchmarks
    from evaluation.scenario_runner import run_scenarios

    mode = mode or experiment.mode
    registry = get_prompt_registry()
    results_by_version: dict[str, Any] = {}

    for version in experiment.versions:
        logger.info(
            "Prompt experiment %s → %s@%s",
            experiment.id,
            experiment.prompt_key,
            version,
        )
        registry.set_active_version(experiment.prompt_key, version, persist=False)
        registry.set_experiment(f"{experiment.prompt_key}:{version}")
        reset_prompt_manager()

        benchmark = run_benchmarks(
            mode=mode,
            limit=experiment.benchmark_limit,
        )
        scenarios = run_scenarios(
            mode=mode,
            limit=experiment.scenario_limit,
        )
        overall = overall_agent_score(
            benchmark.get("summary") or {},
            scenarios.get("summary") or {},
        )
        combined = {
            "overall": overall,
            "benchmark": {"summary": benchmark.get("summary")},
            "scenarios": {"summary": scenarios.get("summary")},
        }
        results_by_version[version] = {
            "metrics": _scorecard_from_eval(combined),
            "overall": overall,
            "benchmark_summary": benchmark.get("summary"),
            "scenario_summary": scenarios.get("summary"),
        }

    registry.clear_overrides()
    registry.set_experiment(None)
    reset_prompt_manager()

    metrics_only = {
        version: payload["metrics"] for version, payload in results_by_version.items()
    }
    best = recommend_best(metrics_only, experiment.priorities)

    return {
        "experiment_id": experiment.id,
        "prompt_key": experiment.prompt_key,
        "versions": experiment.versions,
        "priorities": experiment.priorities,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results_by_version,
        "recommended_version": best,
        "recommendation_reason": (
            f"Selected {best} using priorities: {', '.join(experiment.priorities)}"
        ),
    }


def write_experiment_report(payload: dict[str, Any], *, output_dir: Path | None = None) -> Path:
    """Write markdown + JSON comparison report for a prompt experiment."""
    output_dir = output_dir or (
        Path(config.PROJECT_ROOT) / "evaluation" / "results" / "prompt_experiments"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_id = payload.get("experiment_id") or "experiment"
    json_path = output_dir / f"{exp_id}_{stamp}.json"
    md_path = output_dir / f"{exp_id}_{stamp}.md"
    latest_md = output_dir / "latest_prompt_experiment.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    lines = [
        f"# Prompt Experiment: {payload.get('prompt_key')}",
        "",
        f"Experiment ID: `{payload.get('experiment_id')}`  ",
        f"Mode: `{payload.get('mode')}`  ",
        f"Generated: `{payload.get('generated_at')}`  ",
        "",
        f"**Recommended version:** `{payload.get('recommended_version')}`  ",
        f"Reason: {payload.get('recommendation_reason')}",
        "",
        "## Version Comparison",
        "",
    ]
    for version, row in (payload.get("results") or {}).items():
        metrics = row.get("metrics") or {}
        lines.extend(
            [
                f"### {version}",
                "",
                f"- Accuracy: {metrics.get('accuracy')}%",
                f"- Planner Accuracy: {metrics.get('planner_accuracy')}%",
                f"- Tool Selection: {metrics.get('tool_selection_accuracy')}%",
                f"- SQL Success: {metrics.get('sql_success')}%",
                f"- Retrieval Accuracy: {metrics.get('retrieval_accuracy')}%",
                f"- Scenario Success: {metrics.get('scenario_success')}%",
                f"- Latency: {metrics.get('latency')} sec",
                f"- Avg Tokens: {metrics.get('average_tokens')}",
                f"- Cost: ${metrics.get('cost')}/query",
                f"- Overall Score: {metrics.get('overall_score')}%",
                "",
            ]
        )
    lines.extend(
        [
            "## Priorities",
            "",
            *[f"{i}. {name}" for i, name in enumerate(payload.get("priorities") or [], start=1)],
            "",
            "## Next step",
            "",
            "Activate the recommended version without code changes:",
            "",
            "```bash",
            f"python -m evaluation.prompt_experiments --activate "
            f"{payload.get('prompt_key')}:{payload.get('recommended_version')}",
            "```",
            "",
        ]
    )
    markdown = "\n".join(lines)
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return md_path
