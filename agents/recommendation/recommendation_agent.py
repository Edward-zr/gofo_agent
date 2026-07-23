"""Recommendation Agent — suggested analyses, ops recommendations, next steps."""

from __future__ import annotations

from agents._bridge import run_tool_handler
from agents.base import AgentCapability, AgentResponse, AgentStatus, AgentTask, BaseAgent


class RecommendationAgent(BaseAgent):
    name = "Recommendation"
    version = "1.0"
    capabilities = [
        AgentCapability.SUGGESTED_ANALYSES,
        AgentCapability.OPERATIONAL_RECOMMENDATIONS,
        AgentCapability.NEXT_STEPS,
    ]

    def execute(self, task: AgentTask) -> AgentResponse:
        from core.plan_executor import _handle_recommendation

        result = run_tool_handler(_handle_recommendation, task)
        recommendations = (
            result.get("recommendations")
            or result.get("next_steps")
            or result.get("answer")
            or []
        )
        if isinstance(recommendations, list):
            summary = "; ".join(str(item) for item in recommendations[:5]) or "No recommendations"
        else:
            summary = str(recommendations)
        return AgentResponse(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            confidence=0.8,
            result=result,
            summary=summary,
            metadata={"tool": "RECOMMENDATION", "action": task.action},
        )
