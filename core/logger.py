"""Structured logging for GOFO production services."""

from __future__ import annotations

import logging
from pathlib import Path

import config

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(_level())
    logger.propagate = False

    if not _has_gofo_handlers(logger):
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(_level())
        console_handler._gofo_handler = True  # type: ignore[attr-defined]

        log_path = Path(config.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_level())
        file_handler._gofo_handler = True  # type: ignore[attr-defined]

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def _level() -> int:
    return getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO)


def _has_gofo_handlers(logger: logging.Logger) -> bool:
    return any(getattr(handler, "_gofo_handler", False) for handler in logger.handlers)
