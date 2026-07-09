"""Tests for environment-backed configuration."""

from __future__ import annotations

import importlib

import pytest

import config
from core.errors import ConfigurationError


def test_environment_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as env:
        env.setenv("OPENAI_API_KEY", "test-key")
        env.setenv("SQLITE_DATABASE", "data/gofo_demo.db")
        env.setenv("MEMORY_DATABASE", "data/memory.db")
        env.setenv("CHROMA_PERSIST_DIR", "chroma_db")
        env.setenv("LOG_LEVEL", "DEBUG")
        env.setenv("LOG_FILE", "logs/test-agent.log")
        env.setenv("API_HOST", "127.0.0.1")
        env.setenv("API_PORT", "9000")

        reloaded = importlib.reload(config)

        assert reloaded.OPENAI_API_KEY == "test-key"
        assert reloaded.SQLITE_DATABASE.name == "gofo_demo.db"
        assert reloaded.MEMORY_DATABASE.name == "memory.db"
        assert reloaded.CHROMA_PATH.name == "chroma_db"
        assert reloaded.LOG_LEVEL == "DEBUG"
        assert reloaded.LOG_FILE.name == "test-agent.log"
        assert reloaded.API_HOST == "127.0.0.1"
        assert reloaded.API_PORT == 9000
    importlib.reload(config)


def test_missing_openai_config_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as env:
        env.setenv("OPENAI_API_KEY", "")
        reloaded = importlib.reload(config)

        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is missing"):
            reloaded.require_openai_api_key()
    importlib.reload(config)
