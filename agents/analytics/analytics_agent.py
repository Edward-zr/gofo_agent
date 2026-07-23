"""Analytics Agent — statistics, charts, and data analysis."""

from __future__ import annotations

from agents._bridge import run_tool_handler
from agents.base import AgentCapability, AgentResponse, AgentStatus, AgentTask, BaseAgent


class AnalyticsAgent(BaseAgent):
    name = "Analytics"
    version = "1.0"
    capabilities = [
        AgentCapability.STATISTICS,
        AgentCapability.CHARTS,
        AgentCapability.DATA_ANALYSIS,
    ]

    def execute(self, task: AgentTask) -> AgentResponse:
        tool = (task.tool or "").upper()
        if tool == "TRANSFORM" or task.action in {"transform", "prepare"}:
            from core.plan_executor import _handle_transform

            result = run_tool_handler(_handle_transform, task)
        elif tool == "VISUALIZATION" or task.capability == AgentCapability.CHARTS.value:
            from core.plan_executor import _handle_visualization

            result = run_tool_handler(_handle_visualization, task)
        elif tool == "ATTACHMENT":
            from core.plan_executor import _handle_attachment

            result = run_tool_handler(_handle_attachment, task)
        elif tool == "PYTHON":
            from core.plan_executor import _handle_python

            result = run_tool_handler(_handle_python, task)
        else:
            from core.plan_executor import _handle_statistics

            result = run_tool_handler(_handle_statistics, task)

        summary = (
            result.get("summary")
            or result.get("answer")
            or result.get("analysis")
            or "Analytics completed"
        )
        return AgentResponse(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            confidence=0.88,
            result=result,
            summary=str(summary),
            metadata={"tool": tool or "STATISTICS", "action": task.action},
        )
