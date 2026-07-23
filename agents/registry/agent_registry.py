"""Agent Registry — single source of agent discovery for the Supervisor."""

from __future__ import annotations

from typing import Any

from agents.base import AgentCapability, AgentHealth, AgentTask, BaseAgent, TOOL_TO_CAPABILITY
from core.logger import get_logger

logger = get_logger("agent_registry")


class AgentRegistry:
    """Register, discover, validate, and select agents by capability."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent, *, replace: bool = False) -> None:
        """Register an agent instance. Validates the interface contract."""
        self._validate_agent(agent)
        key = agent.name
        if key in self._agents and not replace:
            raise ValueError(f"Agent already registered: {key}")
        self._agents[key] = agent
        logger.info(
            "Registered agent %s v%s caps=%s",
            key,
            agent.version,
            list(agent.capabilities),
        )

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._agents

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def list_agents(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for agent in self._agents.values():
            health = agent.health_check()
            rows.append(
                {
                    "name": agent.name,
                    "version": agent.version,
                    "capabilities": health.capabilities,
                    "healthy": health.healthy,
                    "average_latency_ms": health.average_latency_ms,
                    "success_rate": health.success_rate,
                }
            )
        return rows

    def list_capabilities(self) -> dict[str, list[str]]:
        """Map capability → agent names that provide it."""
        mapping: dict[str, list[str]] = {}
        for agent in self._agents.values():
            for cap in agent.capabilities:
                key = cap.value if isinstance(cap, AgentCapability) else str(cap)
                mapping.setdefault(key, []).append(agent.name)
        return mapping

    def agents_for_capability(
        self,
        capability: str,
        *,
        healthy_only: bool = True,
    ) -> list[BaseAgent]:
        """Return agents that advertise ``capability`` (never by hardcoded name)."""
        matched: list[BaseAgent] = []
        for agent in self._agents.values():
            caps = {
                c.value if isinstance(c, AgentCapability) else str(c)
                for c in agent.capabilities
            }
            if capability not in caps:
                continue
            if healthy_only and not agent.health_check().healthy:
                continue
            matched.append(agent)
        # Prefer higher success rate, then lower latency.
        matched.sort(
            key=lambda a: (
                -a.health_check().success_rate,
                a.health_check().average_latency_ms,
            )
        )
        return matched

    def select_agent(
        self,
        capability: str,
        *,
        healthy_only: bool = True,
    ) -> BaseAgent | None:
        agents = self.agents_for_capability(capability, healthy_only=healthy_only)
        return agents[0] if agents else None

    def select_for_task(self, task: AgentTask, *, healthy_only: bool = True) -> BaseAgent | None:
        agent = self.select_agent(task.capability, healthy_only=healthy_only)
        if agent is None:
            return None
        if not agent.can_handle(task):
            return None
        return agent

    def resolve_capabilities(self, capabilities: list[str]) -> dict[str, str | None]:
        """Map each required capability → selected agent name (or None)."""
        return {
            cap: (agent.name if (agent := self.select_agent(cap)) else None)
            for cap in capabilities
        }

    def capability_for_tool(self, tool: str) -> str | None:
        return TOOL_TO_CAPABILITY.get(tool)

    def health_snapshot(self) -> list[AgentHealth]:
        return [agent.health_check() for agent in self._agents.values()]

    def discover(self) -> list[str]:
        return sorted(self._agents.keys())

    @staticmethod
    def _validate_agent(agent: BaseAgent) -> None:
        if not isinstance(agent, BaseAgent):
            raise TypeError("Agent must subclass BaseAgent")
        if not getattr(agent, "name", None):
            raise ValueError("Agent must define a name")
        required = ("initialize", "can_handle", "execute", "validate", "summarize", "health_check")
        for method in required:
            if not callable(getattr(agent, method, None)):
                raise ValueError(f"Agent {agent.name} missing method: {method}")
        if not agent.capabilities:
            raise ValueError(f"Agent {agent.name} must declare capabilities")


_REGISTRY: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = AgentRegistry()
    return _REGISTRY


def reset_agent_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
