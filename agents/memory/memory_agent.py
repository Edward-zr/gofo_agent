"""Memory Agent — conversation memory, context tracking, slot reuse."""

from __future__ import annotations

from agents._bridge import run_tool_handler
from agents.base import AgentCapability, AgentResponse, AgentStatus, AgentTask, BaseAgent


class MemoryAgent(BaseAgent):
    name = "Memory"
    version = "1.0"
    capabilities = [
        AgentCapability.CONVERSATION_MEMORY,
        AgentCapability.CONTEXT_TRACKING,
        AgentCapability.SLOT_REUSE,
    ]

    def execute(self, task: AgentTask) -> AgentResponse:
        from core.plan_executor import _handle_memory

        result = run_tool_handler(_handle_memory, task)
        # Enrich with slot / state snapshot when conversation memory is present.
        memory = (task.context or {}).get("conversation_memory")
        slots: dict = {}
        if memory is not None and hasattr(memory, "get_current_state"):
            state = memory.get_current_state() or {}
            slots = {
                k: state.get(k)
                for k in ("current_metric", "date_range", "last_entity", "hub", "driver")
                if state.get(k) is not None
            }
            result = {**result, "slots": slots, "memory_state": state}

        summary = result.get("answer") or result.get("summary") or "Memory context loaded"
        return AgentResponse(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            confidence=0.9,
            result=result,
            summary=str(summary),
            metadata={"tool": "MEMORY", "slots": slots, "action": task.action},
        )
