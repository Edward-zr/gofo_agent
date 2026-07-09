"""Custom exceptions for GOFO production error handling."""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for GOFO agent failures."""


class DatabaseError(AgentError):
    """Database access or query failure."""


class RetrievalError(AgentError):
    """RAG retrieval failure."""


class PlannerError(AgentError):
    """Planner or SQL generation failure."""


class MemoryError(AgentError):
    """Short-term or long-term memory failure."""


class ConfigurationError(AgentError):
    """Missing or invalid system configuration."""
