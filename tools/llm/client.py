"""Shared OpenAI chat client for GOFO agent tools."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

import config


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a cached OpenAI chat client aligned with project settings."""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0,
        openai_api_key=config.require_openai_api_key(),
    )
