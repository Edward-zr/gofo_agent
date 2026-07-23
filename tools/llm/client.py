"""Shared OpenAI chat client for GOFO agent tools.

LLM sampling parameters should come from Prompt metadata via PromptManager.
Components must not hardcode temperature / max_tokens locally.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

import config


@lru_cache(maxsize=32)
def get_llm(
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> ChatOpenAI:
    """Return a cached OpenAI chat client for the given sampling config."""
    kwargs: dict = {
        "model": model or config.LLM_MODEL,
        "temperature": temperature,
        "openai_api_key": config.require_openai_api_key(),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def clear_llm_cache() -> None:
    """Clear cached clients (tests / config reloads)."""
    get_llm.cache_clear()
