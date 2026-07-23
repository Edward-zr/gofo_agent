"""General Agent — conversation, general knowledge, brainstorming."""

from __future__ import annotations

from agents._bridge import run_tool_handler
from agents.base import AgentCapability, AgentResponse, AgentStatus, AgentTask, BaseAgent


class GeneralAgent(BaseAgent):
    name = "General"
    version = "1.0"
    capabilities = [
        AgentCapability.GENERAL_KNOWLEDGE,
        AgentCapability.BRAINSTORMING,
        AgentCapability.CONVERSATION,
    ]

    def execute(self, task: AgentTask) -> AgentResponse:
        from core.plan_executor import _handle_llm

        result = run_tool_handler(_handle_llm, task)
        answer = result.get("answer") or ""
        return AgentResponse(
            agent=self.name,
            status=AgentStatus.SUCCESS if answer else AgentStatus.PARTIAL,
            confidence=0.75 if answer else 0.4,
            result=result,
            summary=answer if isinstance(answer, str) else str(answer),
            metadata={"tool": "LLM", "action": task.action},
        )
