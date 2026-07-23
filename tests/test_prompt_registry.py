"""Tests for the Prompt Registry & Prompt Experiment Framework."""

from __future__ import annotations

import pytest

from core.prompt_experiments import recommend_best
from core.prompt_manager import PromptManager, reset_prompt_manager
from core.prompt_registry import PromptRegistry, reset_prompt_registry
from core.prompt_renderer import PromptRenderer
from core.prompt_validator import PromptValidationError, parse_prompt_document


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    reset_prompt_registry()
    reset_prompt_manager()
    yield
    reset_prompt_registry()
    reset_prompt_manager()


def test_parse_frontmatter_and_render() -> None:
    doc = """---
name: demo
version: "1.0"
owner: test
description: demo
temperature: 0.0
max_tokens: 100
output_format: text
required_variables:
  - question
tags:
  - test
status: active
includes: []
---

Hello {{question}}
"""
    meta, body = parse_prompt_document(doc)
    assert meta.name == "demo"
    assert meta.required_variables == ["question"]
    rendered = PromptRenderer().render(body, {"question": "world"}, metadata=meta)
    assert "Hello world" in rendered


def test_missing_required_variable_raises() -> None:
    doc = """---
name: demo
version: "1.0"
owner: test
description: demo
temperature: 0
max_tokens: 50
output_format: text
required_variables:
  - question
tags: [test]
status: active
includes: []
---
Q: {{question}}
"""
    meta, body = parse_prompt_document(doc)
    with pytest.raises(PromptValidationError, match="Missing required"):
        PromptRenderer().render(body, {}, metadata=meta)


def test_registry_loads_and_lists_prompts() -> None:
    registry = PromptRegistry()
    rows = registry.list_prompts()
    keys = {row["key"] for row in rows}
    assert "planner.planner_prompt" in keys
    assert "rag.generator_prompt" in keys
    assert "sql.generator_prompt" in keys
    versions = registry.list_versions("planner.planner_prompt")
    assert "v1" in versions
    assert "v2" in versions


def test_active_version_switch_without_code_change() -> None:
    manager = PromptManager()
    manager.set_active_version("planner.planner_prompt", "v2", persist=False)
    selection = manager.get("planner.planner_prompt")
    assert selection.version_key == "v2"
    assert "v2 rules" in selection.body or "retrieval_policy" in selection.body


def test_prompt_manager_render_planner() -> None:
    manager = PromptManager()
    selection, text = manager.render(
        "planner.planner_prompt",
        {
            "question": "Show pickup rate",
            "available_tools": "SQL, LLM",
            "memory": "(none)",
            "intent_classification": "{}",
        },
    )
    assert selection.key == "planner.planner_prompt"
    assert "Show pickup rate" in text
    assert "SQL" in text


def test_recommend_best_prioritizes_accuracy() -> None:
    metrics = {
        "v1": {"accuracy": 90.0, "sql_success": 95.0, "scenario_success": 90.0, "latency": 2.0, "cost": 0.003},
        "v2": {"accuracy": 95.0, "sql_success": 90.0, "scenario_success": 88.0, "latency": 1.5, "cost": 0.002},
        "v3": {"accuracy": 94.0, "sql_success": 96.0, "scenario_success": 92.0, "latency": 1.2, "cost": 0.002},
    }
    assert recommend_best(metrics) == "v2"
