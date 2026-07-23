"""Schema Registry — authoritative metadata for the GOFO analytics SQLite DB.

Loads tables, columns, PKs, FKs, and descriptions from live SQLite PRAGMAs,
enriched with curated business descriptions. Supports auto-refresh when the
database file changes and an explicit refresh() API.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import config
from core.logger import get_logger

logger = get_logger("schema_registry")

# Curated descriptions / aliases so retrieval works even when SQLite has no comments.
_TABLE_DESCRIPTIONS: dict[str, str] = {
    "pickups": (
        "Operational pickup tasks. One row per pickup. "
        "Use for counts, KPIs, rates, status, package volume, and dates."
    ),
    "drivers": (
        "Driver master data including hub/warehouse assignment. "
        "Join for driver name, hub rankings, and warehouse filters."
    ),
    "customers": "Customer master data. Join for customer names and customer-level analysis.",
    "addresses": "Pickup address locations with city and state. Join for geography questions.",
    "exceptions": (
        "Failure / exception reasons linked to pickups. "
        "Join for root-cause, failure reason, and delayed pickup analysis."
    ),
}

_COLUMN_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("pickups", "pickup_id"): "Primary key for a pickup task.",
    ("pickups", "customer_id"): "FK to customers.customer_id.",
    ("pickups", "driver_id"): "FK to drivers.driver_id.",
    ("pickups", "address_id"): "FK to addresses.address_id.",
    ("pickups", "pickup_date"): "ISO operational date (YYYY-MM-DD). Always use for time filters.",
    ("pickups", "package_count"): "Number of packages on the pickup.",
    ("pickups", "status"): "Completed, Delayed, or Failed.",
    ("drivers", "driver_id"): "Primary key for a driver.",
    ("drivers", "driver_name"): "Human-readable driver name.",
    ("drivers", "hub"): "Operational warehouse / station name (NOT a separate hub table).",
    ("customers", "customer_id"): "Primary key for a customer.",
    ("customers", "customer_name"): "Human-readable customer name.",
    ("addresses", "address_id"): "Primary key for an address.",
    ("addresses", "city"): "City name for the pickup location.",
    ("addresses", "state"): "State / region code for the pickup location.",
    ("exceptions", "exception_id"): "Primary key for an exception record.",
    ("exceptions", "pickup_id"): "FK to pickups.pickup_id.",
    ("exceptions", "reason"): "Exception / failure reason text.",
    ("exceptions", "created_at"): "When the exception was recorded.",
}

_TABLE_KEYWORDS: dict[str, list[str]] = {
    "pickups": [
        "pickup",
        "pickups",
        "package",
        "packages",
        "status",
        "completed",
        "failed",
        "delayed",
        "kpi",
        "rate",
        "volume",
        "count",
        "today",
        "yesterday",
        "trend",
        "performance",
    ],
    "drivers": [
        "driver",
        "drivers",
        "hub",
        "hubs",
        "warehouse",
        "warehouses",
        "station",
        "facility",
    ],
    "customers": ["customer", "customers", "shipper", "client"],
    "addresses": ["city", "cities", "state", "location", "address", "geography", "region"],
    "exceptions": [
        "exception",
        "exceptions",
        "reason",
        "reasons",
        "root cause",
        "failure",
        "failed",
        "why failed",
    ],
}

_COLUMN_KEYWORDS: dict[tuple[str, str], list[str]] = {
    ("drivers", "hub"): ["hub", "warehouse", "station", "facility"],
    ("drivers", "driver_name"): ["driver", "driver name", "courier"],
    ("pickups", "status"): ["status", "completed", "failed", "delayed", "success rate"],
    ("pickups", "pickup_date"): ["date", "day", "today", "yesterday", "week", "month"],
    ("pickups", "package_count"): ["package", "packages", "volume", "weight"],
    ("exceptions", "reason"): ["reason", "root cause", "why", "cause"],
    ("addresses", "city"): ["city", "cities", "location"],
    ("addresses", "state"): ["state", "region"],
    ("customers", "customer_name"): ["customer", "client", "shipper"],
}


class ColumnInfo(BaseModel):
    name: str
    data_type: str = "TEXT"
    primary_key: bool = False
    nullable: bool = True
    description: str = ""
    keywords: list[str] = Field(default_factory=list)


class ForeignKeyInfo(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    def as_join(self) -> str:
        return f"{self.from_table}.{self.from_column} = {self.to_table}.{self.to_column}"


class TableInfo(BaseModel):
    name: str
    description: str = ""
    columns: list[ColumnInfo] = Field(default_factory=list)
    primary_keys: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]


class SchemaSnapshot(BaseModel):
    """Immutable snapshot of registry contents."""

    tables: dict[str, TableInfo] = Field(default_factory=dict)
    relationships: list[ForeignKeyInfo] = Field(default_factory=list)
    fingerprint: str = ""
    source_path: str = ""
    loaded_at: str = ""

    def table_names(self) -> list[str]:
        return sorted(self.tables.keys())

    def allowed_tables(self) -> set[str]:
        return set(self.tables.keys())

    def allowed_columns(self) -> dict[str, set[str]]:
        return {name: set(table.column_names()) for name, table in self.tables.items()}


class SchemaRegistry:
    """Process-wide schema registry with auto-refresh on DB change."""

    def __init__(self, database_path: Path | None = None) -> None:
        self._database_path = database_path or config.SQLITE_DATABASE
        self._lock = threading.RLock()
        self._snapshot: SchemaSnapshot | None = None
        self._mtime_ns: int | None = None

    @property
    def database_path(self) -> Path:
        return self._database_path

    def get_snapshot(self, *, force_refresh: bool = False) -> SchemaSnapshot:
        with self._lock:
            if force_refresh or self._needs_refresh():
                self._snapshot = self._load()
            assert self._snapshot is not None
            return self._snapshot

    def refresh(self) -> SchemaSnapshot:
        """Explicit rebuild — use after migrations or admin refresh commands."""
        with self._lock:
            self._snapshot = self._load()
            logger.info(
                "Schema registry refreshed path=%s tables=%s fingerprint=%s",
                self._database_path,
                self._snapshot.table_names(),
                self._snapshot.fingerprint,
            )
            return self._snapshot

    def _needs_refresh(self) -> bool:
        if self._snapshot is None:
            return True
        path = self._database_path
        if not path.exists():
            return self._mtime_ns is not None
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return True
        return self._mtime_ns != mtime_ns

    def _load(self) -> SchemaSnapshot:
        from datetime import datetime, timezone

        path = self._database_path
        if path.exists():
            try:
                self._mtime_ns = path.stat().st_mtime_ns
            except OSError:
                self._mtime_ns = None
            snapshot = self._load_from_sqlite(path)
        else:
            self._mtime_ns = None
            snapshot = self._load_fallback_static()

        snapshot.loaded_at = datetime.now(timezone.utc).isoformat()
        snapshot.source_path = str(path)
        snapshot.fingerprint = self._fingerprint(snapshot)
        return snapshot

    def _load_from_sqlite(self, path: Path) -> SchemaSnapshot:
        connection = sqlite3.connect(str(path))
        try:
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
            ).fetchall()
            tables: dict[str, TableInfo] = {}
            relationships: list[ForeignKeyInfo] = []

            for (table_name,) in table_rows:
                name = str(table_name)
                columns_raw = connection.execute(f"PRAGMA table_info({name});").fetchall()
                fk_raw = connection.execute(f"PRAGMA foreign_key_list({name});").fetchall()

                columns: list[ColumnInfo] = []
                primary_keys: list[str] = []
                for row in columns_raw:
                    # cid, name, type, notnull, dflt_value, pk
                    col_name = str(row[1])
                    col_type = str(row[2] or "TEXT")
                    notnull = bool(row[3])
                    is_pk = bool(row[5])
                    if is_pk:
                        primary_keys.append(col_name)
                    columns.append(
                        ColumnInfo(
                            name=col_name,
                            data_type=col_type,
                            primary_key=is_pk,
                            nullable=not notnull and not is_pk,
                            description=_COLUMN_DESCRIPTIONS.get((name, col_name), ""),
                            keywords=list(_COLUMN_KEYWORDS.get((name, col_name), [])),
                        )
                    )

                foreign_keys: list[ForeignKeyInfo] = []
                for fk in fk_raw:
                    # id, seq, table, from, to, on_update, on_delete, match
                    fk_info = ForeignKeyInfo(
                        from_table=name,
                        from_column=str(fk[3]),
                        to_table=str(fk[2]),
                        to_column=str(fk[4]),
                    )
                    foreign_keys.append(fk_info)
                    relationships.append(fk_info)

                tables[name] = TableInfo(
                    name=name,
                    description=_TABLE_DESCRIPTIONS.get(name, f"Table {name}"),
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    keywords=list(_TABLE_KEYWORDS.get(name, [name])),
                )

            return SchemaSnapshot(tables=tables, relationships=relationships)
        finally:
            connection.close()

    def _load_fallback_static(self) -> SchemaSnapshot:
        """When DB is missing (tests), rebuild from curated allowlists."""
        static_columns = {
            "pickups": [
                ("pickup_id", "INTEGER", True),
                ("customer_id", "INTEGER", False),
                ("driver_id", "INTEGER", False),
                ("address_id", "INTEGER", False),
                ("pickup_date", "TEXT", False),
                ("package_count", "INTEGER", False),
                ("status", "TEXT", False),
            ],
            "drivers": [
                ("driver_id", "INTEGER", True),
                ("driver_name", "TEXT", False),
                ("hub", "TEXT", False),
            ],
            "customers": [
                ("customer_id", "INTEGER", True),
                ("customer_name", "TEXT", False),
            ],
            "addresses": [
                ("address_id", "INTEGER", True),
                ("city", "TEXT", False),
                ("state", "TEXT", False),
            ],
            "exceptions": [
                ("exception_id", "INTEGER", True),
                ("pickup_id", "INTEGER", False),
                ("reason", "TEXT", False),
                ("created_at", "TEXT", False),
            ],
        }
        static_fks = [
            ForeignKeyInfo("pickups", "customer_id", "customers", "customer_id"),
            ForeignKeyInfo("pickups", "driver_id", "drivers", "driver_id"),
            ForeignKeyInfo("pickups", "address_id", "addresses", "address_id"),
            ForeignKeyInfo("exceptions", "pickup_id", "pickups", "pickup_id"),
        ]
        tables: dict[str, TableInfo] = {}
        for table_name, cols in static_columns.items():
            columns = [
                ColumnInfo(
                    name=col_name,
                    data_type=col_type,
                    primary_key=is_pk,
                    nullable=not is_pk,
                    description=_COLUMN_DESCRIPTIONS.get((table_name, col_name), ""),
                    keywords=list(_COLUMN_KEYWORDS.get((table_name, col_name), [])),
                )
                for col_name, col_type, is_pk in cols
            ]
            fks = [fk for fk in static_fks if fk.from_table == table_name]
            tables[table_name] = TableInfo(
                name=table_name,
                description=_TABLE_DESCRIPTIONS.get(table_name, f"Table {table_name}"),
                columns=columns,
                primary_keys=[c.name for c in columns if c.primary_key],
                foreign_keys=fks,
                keywords=list(_TABLE_KEYWORDS.get(table_name, [table_name])),
            )
        return SchemaSnapshot(tables=tables, relationships=list(static_fks))

    @staticmethod
    def _fingerprint(snapshot: SchemaSnapshot) -> str:
        payload = []
        for table_name in sorted(snapshot.tables):
            table = snapshot.tables[table_name]
            cols = ",".join(f"{c.name}:{c.data_type}:{int(c.primary_key)}" for c in table.columns)
            fks = ",".join(sorted(fk.as_join() for fk in table.foreign_keys))
            payload.append(f"{table_name}|{cols}|{fks}")
        digest = hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()
        return digest[:16]


_REGISTRY: SchemaRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_schema_registry() -> SchemaRegistry:
    """Return the process-wide SchemaRegistry singleton."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = SchemaRegistry()
        return _REGISTRY


def refresh_schema_registry() -> SchemaSnapshot:
    """Rebuild registry metadata (admin / migration hook)."""
    return get_schema_registry().refresh()


def reset_schema_registry_for_tests() -> None:
    """Clear singleton — tests only."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = None
