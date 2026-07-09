"""SQLite database initialization for persistent GOFO memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import config


def get_memory_database_path() -> Path:
    """Return the configured persistent memory database path."""
    return config.MEMORY_DATABASE


def get_connection(database_path: Path | None = None) -> sqlite3.Connection:
    """Return a SQLite connection with memory tables initialized."""
    path = database_path or get_memory_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    initialize(connection)
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    """Create persistent memory tables if they do not already exist."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            user_question TEXT NOT NULL,
            resolved_question TEXT,
            answer TEXT,
            intent TEXT,
            metric TEXT,
            dimension TEXT,
            sql_query TEXT
        );

        CREATE TABLE IF NOT EXISTS operational_findings (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            hub TEXT,
            driver TEXT,
            customer TEXT,
            issue TEXT,
            metric TEXT,
            root_cause TEXT,
            recommendation TEXT,
            severity TEXT
        );

        CREATE TABLE IF NOT EXISTS learned_patterns (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            pattern_name TEXT NOT NULL,
            description TEXT,
            trigger_condition TEXT,
            previous_solution TEXT
        );
        """
    )
    connection.commit()
