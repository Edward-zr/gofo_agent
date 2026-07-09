"""Tests for FastAPI health endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.server import app


@patch("api.server.sqlite3.connect")
@patch("api.server.config.CHROMA_PERSIST_DIR")
@patch("api.server.get_connection")
@patch("api.server.config.SQLITE_DATABASE")
def test_health_endpoint_returns_running(
    mock_sqlite_database: MagicMock,
    mock_get_connection: MagicMock,
    mock_chroma_dir: MagicMock,
    mock_sqlite_connect: MagicMock,
) -> None:
    mock_sqlite_database.exists.return_value = True
    mock_chroma_dir.exists.return_value = True
    mock_sqlite_connect.return_value = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = MagicMock()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "running",
        "version": "1.0",
        "database": "connected",
        "memory": "enabled",
        "rag": "enabled",
    }
