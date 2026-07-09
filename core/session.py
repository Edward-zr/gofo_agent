"""Session-scoped GOFO agent management for API clients."""

from __future__ import annotations

from threading import Lock
from typing import Callable

from core.agent import GOFOAgent


class SessionManager:
    """Keep one stateful GOFOAgent per API session_id."""

    def __init__(self, agent_factory: Callable[[], GOFOAgent] = GOFOAgent) -> None:
        self._agent_factory = agent_factory
        self._sessions: dict[str, GOFOAgent] = {}
        self._lock = Lock()

    def get_session(self, session_id: str | None = None) -> GOFOAgent:
        """Return the existing session agent or create one once."""
        normalized_session_id = (session_id or "default").strip() or "default"
        with self._lock:
            if normalized_session_id not in self._sessions:
                self._sessions[normalized_session_id] = self._agent_factory()
            return self._sessions[normalized_session_id]

    def session_count(self) -> int:
        """Return number of active in-process sessions."""
        with self._lock:
            return len(self._sessions)
