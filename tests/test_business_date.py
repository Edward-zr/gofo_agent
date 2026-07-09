"""Unit tests for tools.sql.business_date."""

from __future__ import annotations

import sqlite3
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from tools.sql.business_date import get_latest_business_date, resolve_relative_dates


@pytest.fixture(autouse=True)
def clear_business_date_cache() -> None:
    """Clear cached latest business date between tests."""
    get_latest_business_date.cache_clear()
    yield
    get_latest_business_date.cache_clear()


@patch("tools.sql.business_date.sqlite3.connect")
@patch("tools.sql.business_date.config.SQLITE_DATABASE")
def test_get_latest_business_date_success(
    mock_database_path: MagicMock,
    mock_connect: MagicMock,
) -> None:
    mock_database_path.exists.return_value = True
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_connection.execute.return_value.fetchone.return_value = ("2026-06-28",)

    latest_date = get_latest_business_date()

    assert latest_date == date(2026, 6, 28)
    mock_connection.execute.assert_called_once_with("SELECT MAX(pickup_date) FROM pickups")
    mock_connection.close.assert_called_once()


@patch("tools.sql.business_date.sqlite3.connect")
@patch("tools.sql.business_date.config.SQLITE_DATABASE")
def test_get_latest_business_date_empty_database_raises(
    mock_database_path: MagicMock,
    mock_connect: MagicMock,
) -> None:
    mock_database_path.exists.return_value = True
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_connection.execute.return_value.fetchone.return_value = (None,)

    with pytest.raises(ValueError, match="No pickup dates found"):
        get_latest_business_date()


@patch("tools.sql.business_date.sqlite3.connect")
@patch("tools.sql.business_date.config.SQLITE_DATABASE")
def test_get_latest_business_date_is_cached(
    mock_database_path: MagicMock,
    mock_connect: MagicMock,
) -> None:
    mock_database_path.exists.return_value = True
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    mock_connection.execute.return_value.fetchone.return_value = ("2026-06-28",)

    first = get_latest_business_date()
    second = get_latest_business_date()

    assert first == second
    mock_connect.assert_called_once()


def test_resolve_relative_dates_rewrites_common_phrases() -> None:
    latest_date = date(2026, 6, 28)

    assert resolve_relative_dates("How many pickups today?", latest_date) == (
        "How many pickups 2026-06-28?"
    )
    assert resolve_relative_dates("How many pickups yesterday?", latest_date) == (
        "How many pickups 2026-06-27?"
    )
    assert resolve_relative_dates("Top drivers this week", latest_date) == (
        "Top drivers week of 2026-06-22 to 2026-06-28"
    )
    assert resolve_relative_dates("Pickup count last week", latest_date) == (
        "Pickup count week of 2026-06-15 to 2026-06-21"
    )
    assert resolve_relative_dates("Shipments this month", latest_date) == (
        "Shipments June 2026"
    )
    assert resolve_relative_dates("Shipments last month", latest_date) == (
        "Shipments May 2026"
    )


def test_resolve_relative_dates_leaves_other_text_unchanged() -> None:
    latest_date = date(2026, 6, 28)
    question = "Average packages per pickup by hub"

    assert resolve_relative_dates(question, latest_date) == question
