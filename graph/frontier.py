"""Frontier orchestrator — problem-type understanding before tools.

Decision order (locked):
1. Understand problem type
2. SOP-related and searchable in SOPs? → search SOPs
3. Else ADA/data question?
   - No → not meaningful
   - Yes → requires analysis?
     - No → not meaningful
     - Yes → choose SQL and/or Python tools, then summarize
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.intent_router import RouteIntent
from core.logger import get_logger

logger = get_logger("langgraph.frontier")

ProblemKind = Literal["sop_searchable", "ada_analysis", "not_meaningful"]
AnalysisTool = Literal["SQL", "PYTHON", "ATTACHMENT", "RAG"]


class FrontierDecision(BaseModel):
    """Structured frontier output consumed by LangGraph nodes."""

    problem_kind: ProblemKind
    sop_related: bool = False
    sop_searchable: bool = False
    ada_or_data: bool = False
    requires_analysis: bool = False
    tools: list[str] = Field(default_factory=list)
    reasoning: str = ""
    not_meaningful_message: str | None = None

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        if self.problem_kind == "sop_searchable" and "RAG" not in self.tools:
            self.tools = ["RAG", *self.tools]
        if self.problem_kind == "ada_analysis" and not self.tools:
            self.tools = ["SQL"]


_NOT_MEANINGFUL = (
    "That question doesn't look like a GOFO SOP knowledge ask or a data-analysis "
    "request I can run. Try asking about a policy/SOP (for example CBT or driver "
    "procedures), or ask for an operational/file analysis with a clear metric."
)

_SOP_HINTS = (
    "sop",
    "policy",
    "procedure",
    "workflow",
    "guideline",
    "playbook",
    "cbt",
    "tiktok",
    "collection by tiktok",
    "responsible for",
    "responsibilities",
    "what should driver",
    "what should the driver",
    "how should",
    "how do we",
    "according to",
    "standard operating",
)

_ANALYSIS_HINTS = (
    "rank",
    "top ",
    "bottom ",
    "kpi",
    "rate",
    "volume",
    "how many",
    "count",
    "today",
    "yesterday",
    "this week",
    "last week",
    "chart",
    "plot",
    "dashboard",
    "trend",
    "compare",
    "vs ",
    "worst",
    "best",
    "highest",
    "lowest",
    "analyze",
    "analyse",
    "summary of",
    "summarize",
    "group by",
    "breakdown",
)

_ADA_FILE_HINTS = (
    "this file",
    "uploaded",
    "excel",
    "csv",
    "spreadsheet",
    "xlsx",
    "pdf",
    "the file",
    "attachment",
)


def decide_frontier(
    *,
    question: str,
    route_decision: dict[str, Any] | None = None,
    attachment_ids: list[str] | None = None,
    attachment_active: bool = False,
    domain: str | None = None,
) -> FrontierDecision:
    """Heuristic frontier decision (no LLM required for v1; structure is stable)."""
    text = (question or "").strip().lower()
    route = dict(route_decision or {})
    route_intent = str(route.get("intent") or "")
    has_attachments = bool(attachment_ids) or attachment_active or bool(
        route.get("use_attachments")
    )
    domain = (domain or "").upper()

    # Chat / wait / greeting — pass through to planner (not "not meaningful").
    if route_intent in {
        RouteIntent.GENERAL_CHAT,
        RouteIntent.OPENAI_FALLBACK,
        RouteIntent.WAIT_FOR_UPLOAD,
    }:
        decision = FrontierDecision(
            problem_kind="ada_analysis",
            ada_or_data=True,
            requires_analysis=True,
            tools=["LLM"] if route_intent != RouteIntent.WAIT_FOR_UPLOAD else [],
            reasoning=f"Pass-through {route_intent} to planner.",
        )
        logger.info("[Frontier] pass-through %s", route_intent)
        return decision

    sop_related = (
        domain == "SOP"
        or route_intent == RouteIntent.SOP_QA
        or any(h in text for h in _SOP_HINTS)
        or bool(re.search(r"\bwhat is\b|\bdefine\b|\bwhat does\b.*mean", text))
    )
    # Definitional / procedural without analytics → searchable in SOPs
    has_analysis_signal = any(h in text for h in _ANALYSIS_HINTS)
    sop_searchable = sop_related and not (
        has_analysis_signal
        and route_intent in {RouteIntent.SQL_ANALYTICS, RouteIntent.ATTACHMENT_ANALYSIS}
        and not any(h in text for h in ("sop", "policy", "procedure", "cbt", "responsible"))
    )
    # Prefer SOP when intent is SOP_QA even if "driver" appears
    if route_intent == RouteIntent.SOP_QA or domain == "SOP":
        sop_searchable = True
        sop_related = True

    if sop_searchable:
        decision = FrontierDecision(
            problem_kind="sop_searchable",
            sop_related=True,
            sop_searchable=True,
            tools=["RAG"],
            reasoning="SOP-related and answer should be searched in SOP knowledge base.",
        )
        logger.info("[Frontier] %s tools=%s", decision.problem_kind, decision.tools)
        return decision

    ada_or_data = (
        has_attachments
        or route_intent
        in {
            RouteIntent.ATTACHMENT_ANALYSIS,
            RouteIntent.ATTACHMENT_VISUALIZATION,
            RouteIntent.SQL_ANALYTICS,
            RouteIntent.FOLLOW_UP,
            RouteIntent.WAIT_FOR_UPLOAD,
        }
        or domain in {"ADA", "SQL", "HYBRID"}
        or any(h in text for h in _ADA_FILE_HINTS)
        or has_analysis_signal
        or bool(re.search(r"\b(hub|driver|customer|pickup|warehouse)s?\b", text))
    )

    if not ada_or_data:
        decision = FrontierDecision(
            problem_kind="not_meaningful",
            sop_related=False,
            ada_or_data=False,
            requires_analysis=False,
            tools=[],
            reasoning="Neither SOP-searchable nor ADA/data analysis.",
            not_meaningful_message=_NOT_MEANINGFUL,
        )
        logger.info("[Frontier] not_meaningful")
        return decision

    requires_analysis = (
        has_analysis_signal
        or has_attachments
        or route_intent
        in {
            RouteIntent.SQL_ANALYTICS,
            RouteIntent.ATTACHMENT_ANALYSIS,
            RouteIntent.ATTACHMENT_VISUALIZATION,
            RouteIntent.FOLLOW_UP,
        }
        or domain in {"ADA", "SQL", "HYBRID"}
    )

    if not requires_analysis:
        decision = FrontierDecision(
            problem_kind="not_meaningful",
            sop_related=False,
            ada_or_data=True,
            requires_analysis=False,
            tools=[],
            reasoning="Data-related wording but no analysis task to run.",
            not_meaningful_message=_NOT_MEANINGFUL,
        )
        logger.info("[Frontier] ada without analysis → not_meaningful")
        return decision

    tools: list[str] = []
    if has_attachments or route_intent in {
        RouteIntent.ATTACHMENT_ANALYSIS,
        RouteIntent.ATTACHMENT_VISUALIZATION,
    } or any(h in text for h in _ADA_FILE_HINTS):
        tools.append("ATTACHMENT")
        if any(h in text for h in ("chart", "plot", "visual", "python", "group by")):
            tools.append("PYTHON")
    if route_intent in {RouteIntent.SQL_ANALYTICS, RouteIntent.FOLLOW_UP} or (
        has_analysis_signal and not has_attachments
    ):
        tools.append("SQL")
    if "chart" in text or "plot" in text or "trend" in text:
        if "PYTHON" not in tools:
            tools.append("PYTHON")
    if not tools:
        tools = ["SQL"]

    decision = FrontierDecision(
        problem_kind="ada_analysis",
        sop_related=False,
        ada_or_data=True,
        requires_analysis=True,
        tools=list(dict.fromkeys(tools)),
        reasoning="ADA/data analysis required; selected tools for execution.",
    )
    logger.info("[Frontier] ada_analysis tools=%s", decision.tools)
    return decision


def apply_frontier_to_route(
    route_decision: dict[str, Any],
    frontier: FrontierDecision,
) -> dict[str, Any]:
    """Align route_decision with frontier problem kind for downstream planner."""
    updated = dict(route_decision or {})
    if frontier.problem_kind == "sop_searchable":
        updated["intent"] = RouteIntent.SOP_QA
        updated["data_sources"] = ["RAG"]
        updated["use_attachments"] = False
        updated["detach_attachments"] = True
        updated["handler"] = "RAG"
        updated["target"] = "SOP Retriever"
    elif frontier.problem_kind == "ada_analysis":
        if "ATTACHMENT" in frontier.tools:
            if any(t in frontier.tools for t in ("PYTHON",)) and updated.get("chart_type"):
                updated["intent"] = RouteIntent.ATTACHMENT_VISUALIZATION
            else:
                updated["intent"] = RouteIntent.ATTACHMENT_ANALYSIS
            updated["use_attachments"] = True
            updated["data_sources"] = ["ATTACHMENT"]
        elif "SQL" in frontier.tools:
            updated["intent"] = RouteIntent.SQL_ANALYTICS
            updated["data_sources"] = ["SQLITE"]
            updated["use_attachments"] = False
    return updated
