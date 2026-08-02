"""Runtime wrapper: LangGraph entrypoint that preserves GOFOAgent behavior."""

from __future__ import annotations

from typing import Any

import config
from core.agent import GOFOAgent
from core.logger import get_logger
from graph.builder import build_compiled_graph, get_checkpointer
from graph.state import AgentState, empty_agent_state
from graph.transitions import capability_for_intent


def capability_label_from_route(route: str | None) -> str:
    return capability_for_intent(str(route or "")) if route else ""

logger = get_logger("langgraph.runtime")


class LangGraphGOFOAgent:
    """Session-facing agent that orchestrates via LangGraph around GOFOAgent.

    Public ``ask()`` signature matches ``GOFOAgent.ask`` so FastAPI / Streamlit
    require no API changes. Existing tools and prompts remain inside GOFOAgent.
    """

    def __init__(self, agent: GOFOAgent | None = None, *, thread_id: str = "default") -> None:
        self.agent = agent or GOFOAgent()
        self.thread_id = thread_id
        self._checkpointer = get_checkpointer() if config.LANGGRAPH_CHECKPOINTING else None
        self._graph = build_compiled_graph(self.agent, checkpointer=self._checkpointer)
        self._configure_tracing()

    # Compatibility surface used by API helpers / attachment wiring
    @property
    def attachment_service(self):
        return self.agent.attachment_service

    @attachment_service.setter
    def attachment_service(self, value) -> None:
        self.agent.attachment_service = value

    @property
    def attachment_memory(self):
        return self.agent.attachment_memory

    @property
    def state(self):
        return self.agent.state

    @property
    def memory(self):
        return self.agent.memory

    def clear_attachment_context(self) -> None:
        self.agent.clear_attachment_context()

    def ask(self, question: str, attachment_ids: list[str] | None = None) -> dict[str, Any]:
        """Run one user turn through the LangGraph orchestration layer."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        seed = empty_agent_state(
            question=question,
            session_id=self.thread_id,
            attachment_ids=attachment_ids or [],
        )
        # Hydrate attachment metadata from ConversationState for checkpoint continuity.
        snap = self.agent.state.snapshot()
        seed["attachment_active"] = bool(snap.get("attachment_active"))
        seed["last_attachment_file_types"] = list(snap.get("last_attachment_file_types") or [])
        seed["last_attachment_filenames"] = list(snap.get("last_attachment_filenames") or [])
        seed["previous_attachment_filenames"] = list(
            snap.get("previous_attachment_filenames") or []
        )
        seed["last_active_sheet"] = snap.get("last_active_sheet")
        seed["attachment_context_timestamp"] = snap.get("attachment_context_timestamp")
        seed["attachment_session_active"] = bool(snap.get("attachment_active"))
        seed["previous_visualization"] = snap.get("last_visualization")
        seed["visualization_request"] = snap.get("last_visualization")
        seed["previous_capability"] = capability_label_from_route(snap.get("last_route"))
        seed["current_capability"] = capability_label_from_route(snap.get("last_route"))
        seed["current_topic"] = str(snap.get("last_topic") or snap.get("last_route") or "")

        config_dict: dict[str, Any] = {
            "configurable": {"thread_id": self.thread_id},
        }
        logger.info(
            "[LangGraph] invoke thread_id=%s routed=%s attachments=%s",
            self.thread_id,
            config.LANGGRAPH_ROUTED_NODES,
            len(attachment_ids or []),
        )
        if self._checkpointer is not None:
            logger.info("[LangGraph] Checkpoint load/save enabled for thread_id=%s", self.thread_id)

        final_state: AgentState = self._graph.invoke(seed, config=config_dict)
        response = dict(final_state.get("api_response") or {})
        if not response:
            # Safety net — should not happen if finalize ran.
            response = self.agent.ask(question, attachment_ids=attachment_ids or None)
            analysis = dict(response.get("analysis") or {})
            analysis["langgraph"] = True
            analysis["langgraph_fallback"] = "empty_api_response"
            response["analysis"] = analysis
        return response

    def _configure_tracing(self) -> None:
        if not config.LANGGRAPH_TRACING:
            return
        # LangSmith picks up LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY from the env.
        import os

        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "gofo-agent"))
        logger.info("[LangGraph] Tracing requested (LangSmith env must be configured)")


def create_session_agent(session_id: str = "default") -> GOFOAgent | LangGraphGOFOAgent:
    """Factory used by FastAPI SessionManager when LangGraph is enabled."""
    base = GOFOAgent()
    if not config.LANGGRAPH_ENABLED:
        return base
    return LangGraphGOFOAgent(base, thread_id=session_id or "default")
