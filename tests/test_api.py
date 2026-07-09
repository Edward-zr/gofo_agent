"""Tests for the production-style FastAPI API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.server import app


def test_post_ask_returns_answer() -> None:
    client = TestClient(app)
    with patch("api.server.session_manager") as mock_session_manager:
        mock_session_agent = MagicMock()
        mock_session_manager.get_session.return_value = mock_session_agent
        mock_session_agent.ask.return_value = {
            "answer": "GOFO operations are stable.",
            "sources": [],
            "sql": "",
            "data": [],
            "analysis": {},
            "recommendations": [],
            "kpi": {},
            "raw": {},
        }

        response = client.post("/ask", json={"question": "How are operations today?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "GOFO operations are stable."
    assert payload["sources"] == []
    mock_session_manager.get_session.assert_called_once_with("default")


@patch("api.server.sqlite3.connect")
@patch("api.server.config.CHROMA_PERSIST_DIR")
@patch("api.server.get_connection")
@patch("api.server.config.SQLITE_DATABASE")
def test_get_health_returns_running(
    mock_sqlite_database: MagicMock,
    mock_get_connection: MagicMock,
    mock_chroma_dir: MagicMock,
    mock_sqlite_connect: MagicMock,
) -> None:
    mock_sqlite_database.exists.return_value = True
    mock_chroma_dir.exists.return_value = True
    mock_sqlite_connect.return_value = MagicMock()
    mock_connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_connection
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["database"] == "connected"
    assert payload["memory"] == "enabled"
    assert payload["rag"] == "enabled"
