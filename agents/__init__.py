"""Multi-agent package: Registry-backed specialized agents + Supervisor."""

from __future__ import annotations

from agents.base import (
    AgentCapability,
    AgentHealth,
    AgentResponse,
    AgentStatus,
    AgentTask,
    BaseAgent,
)
from agents.registry.agent_registry import AgentRegistry, get_agent_registry, reset_agent_registry

__all__ = [
    "AgentCapability",
    "AgentHealth",
    "AgentRegistry",
    "AgentResponse",
    "AgentStatus",
    "AgentTask",
    "BaseAgent",
    "get_agent_registry",
    "register_default_agents",
    "reset_agent_registry",
]


def register_default_agents(registry: AgentRegistry | None = None) -> AgentRegistry:
    """Register built-in GOFO agents (idempotent)."""
    from agents.analytics.analytics_agent import AnalyticsAgent
    from agents.general.general_agent import GeneralAgent
    from agents.memory.memory_agent import MemoryAgent
    from agents.rag.rag_agent import RAGAgent
    from agents.recommendation.recommendation_agent import RecommendationAgent
    from agents.reflection.reflection_agent import ReflectionAgent
    from agents.sql.sql_agent import SQLAgent

    reg = registry or get_agent_registry()
    for agent in (
        RAGAgent(),
        SQLAgent(),
        AnalyticsAgent(),
        GeneralAgent(),
        MemoryAgent(),
        ReflectionAgent(),
        RecommendationAgent(),
    ):
        if not reg.has(agent.name):
            reg.register(agent)
            agent.initialize()
    return reg
