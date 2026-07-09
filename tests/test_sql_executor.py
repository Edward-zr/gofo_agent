"""Unit tests for tools.sql.executor and formatter."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from tools.sql.executor import execute
from tools.sql.formatter import format_result


class _FakeRow:
    """Minimal sqlite3.Row-like object for executor tests."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __getitem__(self, key: str) -> object:
        return self._data[key]


@patch("tools.sql.executor.sqlite3.connect")
@patch("tools.sql.executor.config.SQLITE_DATABASE")
def test_execute_success(mock_database_path: MagicMock, mock_connect: MagicMock) -> None:
    mock_database_path.exists.return_value = True
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_connection.execute.return_value.fetchall.return_value = [
        _FakeRow({"customer": "Amazon", "shipments": 321}),
        _FakeRow({"customer": "Walmart", "shipments": 119}),
    ]

    rows = execute("SELECT customer, shipments FROM shipments")

    assert rows == [
        {"customer": "Amazon", "shipments": 321},
        {"customer": "Walmart", "shipments": 119},
    ]
    mock_connect.assert_called_once()
    mock_connection.execute.assert_called_once_with("SELECT customer, shipments FROM shipments")
    mock_connection.close.assert_called_once()


@patch("tools.sql.executor.sqlite3.connect")
@patch("tools.sql.executor.config.SQLITE_DATABASE")
def test_execute_empty_result(mock_database_path: MagicMock, mock_connect: MagicMock) -> None:
    mock_database_path.exists.return_value = True
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_connection.execute.return_value.fetchall.return_value = []

    rows = execute("SELECT customer FROM shipments WHERE 1 = 0")

    assert rows == []


@patch("tools.sql.executor.config.SQLITE_DATABASE")
def test_execute_missing_database_raises(mock_database_path: MagicMock) -> None:
    mock_database_path.exists.return_value = False

    with pytest.raises(FileNotFoundError, match="SQLite database not found"):
        execute("SELECT 1")


@patch("tools.sql.executor.sqlite3.connect")
@patch("tools.sql.executor.config.SQLITE_DATABASE")
def test_execute_sql_exception_propagates(
    mock_database_path: MagicMock,
    mock_connect: MagicMock,
) -> None:
    mock_database_path.exists.return_value = True
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_connection.execute.side_effect = sqlite3.Error("syntax error")

    with pytest.raises(sqlite3.Error, match="syntax error"):
        execute("SELECT FROM broken")


def test_format_result_empty() -> None:
    assert format_result([]) == "No records found."


def test_format_result_markdown_table() -> None:
    rows = [
        {"customer": "Amazon", "shipments": 321},
        {"customer": "Walmart", "shipments": 119},
    ]

    formatted = format_result(rows)

    assert formatted == (
        "| customer | shipments |\n"
        "| --- | --- |\n"
        "| Amazon | 321 |\n"
        "| Walmart | 119 |"
    )
