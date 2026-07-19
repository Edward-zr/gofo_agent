"""Tests for FastAPI session memory persistence."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.server import app


class FakeStatefulAgent:
    """Minimal stateful fake proving API reuses the same session agent."""

    def __init__(self) -> None:
        self.driver: str | None = None
        self.worst_hub: str | None = None
        self.previous_question: str | None = None
        self.previous_answer: str | None = None
        self.turns = 0

    def ask(self, question: str, attachment_ids: list[str] | None = None) -> dict:
        self.turns += 1
        normalized = question.lower()
        repair_detected = False
        resolved_question = question

        if "highest" in normalized and "driver" in normalized:
            self.driver = "Drew Nguyen"
            answer = "Drew Nguyen is the highest performing driver."
        elif "which hub" in normalized and "he" in normalized and self.driver:
            resolved_question = f"Which hub does {self.driver} belong to?"
            answer = f"{self.driver} belongs to ORD Hub."
        elif "rank" in normalized and "hub" in normalized:
            self.worst_hub = "Chicago Hub"
            answer = "New York Hub is #1. Chicago Hub is worst."
        elif "why" in normalized and "worst hub" in normalized and self.worst_hub:
            resolved_question = f"Why is {self.worst_hub} performing badly?"
            answer = f"{self.worst_hub} is performing badly because delayed pickups increased."
        elif "actually" in normalized and "lowest" in normalized:
            repair_detected = True
            resolved_question = "Find lowest performing driver"
            answer = "The lowest performing driver is Alex Chen."
        else:
            answer = "I don't know how to route this request."

        previous_question = self.previous_question
        previous_answer = self.previous_answer
        self.previous_question = question
        self.previous_answer = answer
        return _response(
            answer=answer,
            resolved_question=resolved_question,
            repair_detected=repair_detected,
            memory_turns=self.turns,
            previous_question=previous_question,
            previous_answer=previous_answer,
        )


class FakeSessionManager:
    def __init__(self) -> None:
        self.sessions: dict[str, FakeStatefulAgent] = {}

    def get_session(self, session_id: str) -> FakeStatefulAgent:
        self.sessions.setdefault(session_id, FakeStatefulAgent())
        return self.sessions[session_id]


def test_driver_follow_up_remembers_driver(monkeypatch) -> None:
    manager = FakeSessionManager()
    monkeypatch.setattr("api.server.session_manager", manager)
    client = TestClient(app)

    client.post(
        "/ask",
        json={"question": "Highest performing driver", "session_id": "driver-session"},
    )
    response = client.post(
        "/ask",
        json={"question": "Which hub does he belong to?", "session_id": "driver-session"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert "Drew Nguyen belongs to ORD Hub" in payload["answer"]
    assert payload["analysis"]["resolved_question"] == "Which hub does Drew Nguyen belong to?"


def test_worst_hub_follow_up_remembers_worst_hub(monkeypatch) -> None:
    manager = FakeSessionManager()
    monkeypatch.setattr("api.server.session_manager", manager)
    client = TestClient(app)

    client.post(
        "/ask",
        json={"question": "Rank all hubs by performance", "session_id": "hub-session"},
    )
    response = client.post(
        "/ask",
        json={"question": "Why is the worst hub performing badly?", "session_id": "hub-session"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert "Chicago Hub is performing badly" in payload["answer"]
    assert payload["analysis"]["resolved_question"] == "Why is Chicago Hub performing badly?"


def test_repair_changes_ranking_direction(monkeypatch) -> None:
    manager = FakeSessionManager()
    monkeypatch.setattr("api.server.session_manager", manager)
    client = TestClient(app)

    client.post(
        "/ask",
        json={"question": "Highest driver", "session_id": "repair-session"},
    )
    response = client.post(
        "/ask",
        json={"question": "Actually I mean lowest driver", "session_id": "repair-session"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["analysis"]["repair_detected"] is True
    assert payload["analysis"]["resolved_question"] == "Find lowest performing driver"
    assert "lowest performing driver" in payload["answer"]


def _response(
    *,
    answer: str,
    resolved_question: str,
    repair_detected: bool,
    memory_turns: int,
    previous_question: str | None,
    previous_answer: str | None,
) -> dict:
    return {
        "answer": answer,
        "sources": [],
        "sql": "",
        "data": [],
        "analysis": {
            "capability": "sql",
            "resolved_question": resolved_question,
            "repair_detected": repair_detected,
            "memory_turns": memory_turns,
            "previous_question": previous_question,
            "previous_answer": previous_answer,
            "current_entities": {},
        },
        "recommendations": [],
        "kpi": {},
        "raw": {},
    }
