"""Centralized friendly error handling for GOFO services."""

from __future__ import annotations

import sqlite3
from typing import Any

from core.errors import (
    AgentError,
    ConfigurationError,
    DatabaseError,
    MemoryError,
    PlannerError,
    RetrievalError,
)


def handle_error(error: Exception) -> str:
    """Convert exceptions into safe user-facing messages."""
    if isinstance(error, ConfigurationError):
        return "System configuration error."
    if isinstance(error, (DatabaseError, sqlite3.Error, FileNotFoundError)):
        return "I could not complete the database query."
    if isinstance(error, RetrievalError):
        return "I could not retrieve the requested SOP information."
    if isinstance(error, PlannerError):
        return "I could not plan this operational query."
    if isinstance(error, MemoryError):
        return "I could not access conversation memory."
    if _looks_like_openai_error(error):
        return "The AI service is temporarily unavailable."
    if isinstance(error, AgentError):
        return "I could not complete the agent request."
    if isinstance(error, ValueError):
        return str(error)
    return "I could not complete the request."


def error_payload(error: Exception) -> dict[str, Any]:
    """Return a standard JSON error payload."""
    return {"success": False, "error": handle_error(error)}


def _looks_like_openai_error(error: Exception) -> bool:
    module = error.__class__.__module__.lower()
    name = error.__class__.__name__.lower()
    text = str(error).lower()
    return "openai" in module or "openai" in name or "api key" in text
