"""Lightweight Knowledge Graph for GOFO operational entity relationships.

Covers:
- Driver → Hub → Region
- Manager → Hub
- SOP ownership

Backed by SQLite drivers + static org mappings. Used by the Planner/Orchestrator
when Data Source Selection chooses KNOWLEDGE_GRAPH.
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from typing import Any

import config
from core.logger import get_logger

logger = get_logger("knowledge_graph")

# Hub → Region (demo org map)
HUB_TO_REGION: dict[str, str] = {
    "Chicago Hub": "Midwest",
    "Dallas Hub": "South Central",
    "Atlanta Hub": "Southeast",
    "New York Hub": "Northeast",
    "Los Angeles Hub": "West",
}

# Hub → Manager (demo org map)
HUB_TO_MANAGER: dict[str, str] = {
    "Chicago Hub": "Maria Chen",
    "Dallas Hub": "James Ortiz",
    "Atlanta Hub": "Aisha Brooks",
    "New York Hub": "Noah Patel",
    "Los Angeles Hub": "Elena Vargas",
}

# SOP topic → owner
SOP_OWNERS: dict[str, str] = {
    "pickup": "Ops Excellence — Pickup SOP Owner",
    "exception": "Exception Desk — Policy Owner",
    "safety": "Safety Compliance — SOP Owner",
    "dispatch": "Dispatch Control — Playbook Owner",
    "customer": "Customer Experience — SOP Owner",
}

# Demo-friendly aliases for relationship examples / docs
DRIVER_ALIASES: dict[str, str] = {
    "john": "Logan Clark",
    "john smith": "Logan Clark",
    "driver john": "Logan Clark",
}


def _resolve_driver_alias(name: str) -> str:
    key = (name or "").strip().lower()
    return DRIVER_ALIASES.get(key, name)


def _normalize_hub(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for hub in HUB_TO_REGION:
        if hub.lower() == lowered or hub.lower().replace(" hub", "") == lowered.replace(" hub", ""):
            return hub
    for hub in HUB_TO_REGION:
        city = hub.replace(" Hub", "").lower()
        if city in lowered or lowered in city:
            return hub
    if "hub" not in lowered:
        candidate = f"{text.title()} Hub"
        if candidate in HUB_TO_REGION:
            return candidate
    return text


@lru_cache(maxsize=1)
def _load_driver_hubs() -> tuple[dict[str, str], ...]:
    path = config.SQLITE_DATABASE
    if not path.exists():
        return tuple()
    try:
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT driver_id, driver_name, hub FROM drivers ORDER BY driver_name"
            ).fetchall()
            return tuple(
                {
                    "driver_id": str(row["driver_id"]),
                    "driver_name": str(row["driver_name"]),
                    "hub": _normalize_hub(str(row["hub"])),
                }
                for row in rows
            )
        finally:
            connection.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load drivers for knowledge graph: %s", exc)
        return tuple()


def _find_drivers(query: str) -> list[dict[str, str]]:
    drivers = list(_load_driver_hubs())
    text = (query or "").strip().lower()
    if not text:
        return []
    matches: list[dict[str, str]] = []
    for driver in drivers:
        name = driver["driver_name"].lower()
        if name == text or text in name or name in text:
            matches.append(driver)
            continue
        tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", text)
            if token and token not in {"driver", "the"}
        ]
        if tokens and all(token in name for token in tokens):
            matches.append(driver)
    return matches


def _extract_hub_from_text(question: str) -> str | None:
    lowered = question.lower()
    for hub in HUB_TO_REGION:
        city = hub.replace(" Hub", "").lower()
        if hub.lower() in lowered or city in lowered:
            return hub
    return None


def _extract_driver_name(question: str) -> str | None:
    patterns = [
        r"driver\s+([A-Za-z][A-Za-z\.\-']+(?:\s+[A-Za-z][A-Za-z\.\-']+){0,2})",
        r"which region does\s+([A-Za-z][A-Za-z\.\-']+(?:\s+[A-Za-z][A-Za-z\.\-']+){0,2})\s+belong",
        r"where does\s+([A-Za-z][A-Za-z\.\-']+(?:\s+[A-Za-z][A-Za-z\.\-']+){0,2})\s+work",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            name = re.sub(
                r"\b(belong|belongs|work|works|manage|manages)\b.*$",
                "",
                name,
                flags=re.I,
            ).strip()
            if name:
                return name
    return None


def _hub_from_sql_context(sql_rows: list[dict[str, Any]] | None) -> str | None:
    if not sql_rows:
        return None
    for row in sql_rows:
        for key in ("hub", "hub_name", "Hub", "station", "facility"):
            if key in row and row[key]:
                return _normalize_hub(str(row[key]))
    return None


def query_relationships(
    question: str,
    *,
    sql_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Answer relationship questions using the knowledge graph."""
    text = question or ""
    lowered = text.lower()
    facts: list[dict[str, Any]] = []
    answer_parts: list[str] = []

    if "manage" in lowered or "manager" in lowered:
        hub = _hub_from_sql_context(sql_rows) or _extract_hub_from_text(text)
        if hub:
            hub = _normalize_hub(hub)
            manager = HUB_TO_MANAGER.get(hub)
            region = HUB_TO_REGION.get(hub)
            if manager:
                facts.append(
                    {
                        "type": "manager_hub",
                        "hub": hub,
                        "manager": manager,
                        "region": region,
                    }
                )
                answer_parts.append(
                    f"{manager} manages {hub}"
                    + (f" ({region})" if region else "")
                    + "."
                )
            else:
                answer_parts.append(f"No manager mapping found for {hub}.")
        else:
            answer_parts.append("I could not identify which hub to look up a manager for.")

    driver_name = _extract_driver_name(text)
    if driver_name or (
        re.search(r"\bdriver\b", lowered)
        and ("region" in lowered or "hub" in lowered or "belong" in lowered)
    ):
        lookup_name = _resolve_driver_alias(driver_name or "")
        matches = _find_drivers(lookup_name)
        if not matches and lookup_name:
            token = lookup_name.split()[0].lower()
            matches = [d for d in _load_driver_hubs() if token in d["driver_name"].lower()]
        if matches:
            for driver in matches[:5]:
                hub = _normalize_hub(driver["hub"])
                region = HUB_TO_REGION.get(hub, "Unknown")
                manager = HUB_TO_MANAGER.get(hub)
                facts.append(
                    {
                        "type": "driver_hub_region",
                        "driver_id": driver["driver_id"],
                        "driver_name": driver["driver_name"],
                        "hub": hub,
                        "region": region,
                        "manager": manager,
                    }
                )
                answer_parts.append(
                    f"Driver {driver['driver_name']} belongs to {hub} in the {region} region"
                    + (f" (manager: {manager})" if manager else "")
                    + "."
                )
        elif driver_name:
            answer_parts.append(
                f"No driver matching '{driver_name}' was found in the knowledge graph."
            )

    if "region" in lowered and not any(f.get("type") == "hub_region" for f in facts) and not any(
        f.get("type") == "driver_hub_region" for f in facts
    ):
        hub = _extract_hub_from_text(text)
        if hub:
            hub = _normalize_hub(hub)
            region = HUB_TO_REGION.get(hub)
            if region:
                facts.append({"type": "hub_region", "hub": hub, "region": region})
                answer_parts.append(f"{hub} belongs to the {region} region.")

    if "sop" in lowered and ("own" in lowered or "ownership" in lowered or "who owns" in lowered):
        topic = "pickup"
        for key in SOP_OWNERS:
            if key in lowered:
                topic = key
                break
        owner = SOP_OWNERS[topic]
        facts.append({"type": "sop_ownership", "topic": topic, "owner": owner})
        answer_parts.append(f"The {topic} SOP is owned by {owner}.")

    if not answer_parts:
        answer_parts = [
            "No matching relationship found in the knowledge graph for this question."
        ]

    return {
        "tool": "KNOWLEDGE_GRAPH",
        "answer": " ".join(answer_parts),
        "facts": facts,
        "capability": "knowledge_graph",
        "graph_nodes": {
            "hubs": list(HUB_TO_REGION.keys()),
            "regions": sorted(set(HUB_TO_REGION.values())),
            "managers": list(HUB_TO_MANAGER.values()),
        },
    }


def answer(question: str, *, sql_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Public service entrypoint."""
    return query_relationships(question, sql_rows=sql_rows)
