"""Streamlit dashboard for GOFO Operations Intelligence."""

from __future__ import annotations

import os
from uuid import uuid4
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_health() -> dict[str, Any]:
    """Fetch backend health, returning a safe offline status on errors."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {
            "status": "offline",
            "database": "unknown",
            "memory": "unknown",
            "rag": "unknown",
        }


def ask_api(question: str, session_id: str) -> dict[str, Any]:
    """Send a chat question to the FastAPI backend."""
    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={"question": question, "session_id": session_id},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def render_sidebar(health: dict[str, Any]) -> None:
    """Render system status in the sidebar."""
    with st.sidebar:
        st.header("System Status")
        st.write(_status_line("Database Connected", health.get("database") == "connected"))
        st.write(_status_line("RAG Ready", health.get("rag") == "enabled"))
        st.write(_status_line("Memory Enabled", health.get("memory") == "enabled"))
        st.caption(f"API status: {health.get('status', 'unknown')}")


def render_kpis(response: dict[str, Any]) -> None:
    """Render KPI metric cards when present."""
    kpi = response.get("kpi") or {}
    data = response.get("data") or []
    first_row = data[0] if data else {}
    values = {
        "Pickup Count": kpi.get("pickup_count", first_row.get("pickup_count")),
        "Completion Rate": _percent(kpi.get("completion_rate", first_row.get("completion_rate"))),
        "Failed": kpi.get("failed", first_row.get("failed_pickups")),
        "Delayed": kpi.get("delayed", first_row.get("delayed_pickups")),
    }
    if not any(value is not None for value in values.values()):
        return

    st.subheader("Operational KPIs")
    columns = st.columns(4)
    for column, (label, value) in zip(columns, values.items()):
        column.metric(label, "N/A" if value is None else value)


def render_structured_sections(response: dict[str, Any]) -> None:
    """Render analysis, recommendations, SQL data, and SOP sources."""
    render_kpis(response)

    analysis = response.get("analysis") or {}
    root_cause = analysis.get("root_cause") or {}
    if root_cause:
        st.subheader("Root Cause Analysis")
        st.write(root_cause.get("issue") or "Operational issue identified.")
        causes = root_cause.get("main_causes") or []
        for index, cause in enumerate(causes, start=1):
            st.write(f"{index}. {cause}")

    recommendations = response.get("recommendations") or []
    if recommendations:
        st.subheader("AI Recommendations")
        for index, recommendation in enumerate(recommendations, start=1):
            st.write(f"{index}. {recommendation}")

    previous_issues = analysis.get("previous_issues_found") or []
    if previous_issues:
        st.subheader("Historical Context")
        for issue in previous_issues[:3]:
            st.write(
                f"{issue.get('created_at', 'Previous')}: "
                f"{issue.get('issue', 'Operational issue')}"
            )
            if issue.get("root_cause"):
                st.caption(f"Root cause: {issue['root_cause']}")

    if response.get("sql"):
        with st.expander("Generated SQL"):
            st.code(response["sql"], language="sql")

    data = response.get("data") or []
    if data:
        with st.expander("SQL Analytics Data"):
            st.dataframe(data, use_container_width=True)

    sources = response.get("sources") or []
    if sources:
        st.subheader("Knowledge Sources")
        for source in sources:
            metadata = source.get("metadata") or {}
            filename = metadata.get("filename") or metadata.get("source") or "unknown"
            page = metadata.get("page_label") or metadata.get("page") or "unknown"
            st.write(f"- {filename}, page {page}")


def render_chat() -> None:
    """Render persistent Streamlit session chat."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", str(uuid4()))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("response"):
                render_structured_sections(message["response"])

    question = st.chat_input("Ask about GOFO operations, SOPs, SQL analytics, or root causes")
    if question is None:
        return

    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing GOFO operations..."):
            try:
                response = ask_api(question, st.session_state.session_id)
                answer = response.get("answer") or ""
                st.write(answer)
                render_structured_sections(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "response": response}
                )
            except requests.ConnectionError:
                _show_error("API offline. Start the backend with `uvicorn api.server:app --reload`.")
            except requests.Timeout:
                _show_error("The API request timed out. Please try again.")
            except requests.HTTPError as exc:
                detail = _http_error_detail(exc)
                _show_error(detail)
            except requests.RequestException as exc:
                _show_error(f"API request failed: {exc}")


def main() -> None:
    """Run the GOFO Operations Intelligence Center dashboard."""
    st.set_page_config(
        page_title="GOFO Operations Intelligence Center",
        layout="wide",
    )
    st.title("GOFO Operations Intelligence Center")
    st.divider()
    health = get_health()
    render_sidebar(health)
    render_chat()


def _status_line(label: str, ok: bool) -> str:
    return f"{label} {'✓' if ok else '✗'}"


def _percent(value: Any) -> Any:
    if value is None or isinstance(value, str) and value.endswith("%"):
        return value
    return f"{value}%"


def _http_error_detail(exc: requests.HTTPError) -> str:
    if exc.response is None:
        return str(exc)
    try:
        detail = exc.response.json().get("detail")
    except ValueError:
        detail = exc.response.text
    return f"API error: {detail}"


def _show_error(message: str) -> None:
    st.error(message)
    st.session_state.messages.append({"role": "assistant", "content": message})


if __name__ == "__main__":
    main()
