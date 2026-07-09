"""Tests for structured logging setup."""

from __future__ import annotations

import logging

import config
from core.logger import get_logger


def test_logger_creation_and_log_file_creation(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "logs" / "agent.log"
    monkeypatch.setattr(config, "LOG_FILE", log_file)
    monkeypatch.setattr(config, "LOG_LEVEL", "INFO")

    logger_name = "tests.production_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    configured = get_logger(logger_name)
    configured.info("logger test message")
    for handler in configured.handlers:
        handler.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "INFO" in content
    assert "tests.production_logger" in content
    assert "logger test message" in content
