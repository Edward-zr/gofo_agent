"""Token and API cost estimation helpers."""

from __future__ import annotations

from typing import Any

# Approximate OpenAI gpt-4o-mini pricing (USD per 1K tokens). Tunable.
DEFAULT_INPUT_COST_PER_1K = 0.00015
DEFAULT_OUTPUT_COST_PER_1K = 0.0006


def estimate_tokens_from_text(text: str | None) -> int:
    """Rough token estimate (~4 chars/token) when provider usage is unavailable."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def estimate_usage(
    *,
    prompt_text: str = "",
    completion_text: str = "",
    usage: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Return prompt/completion/total tokens and estimated USD cost."""
    if usage:
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
    else:
        prompt = estimate_tokens_from_text(prompt_text)
        completion = estimate_tokens_from_text(completion_text)
        total = prompt + completion

    cost = (prompt / 1000.0) * DEFAULT_INPUT_COST_PER_1K + (
        completion / 1000.0
    ) * DEFAULT_OUTPUT_COST_PER_1K
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "estimated_cost_usd": round(cost, 6),
    }
