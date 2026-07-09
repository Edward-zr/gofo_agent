"""Streamlit frontend for the GOFO Operations Intelligence Agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
CONNECT_ERROR = "Unable to connect to the GOFO API."


def check_api_status() -> bool:
    """Return True when the FastAPI health endpoint is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        response.raise_for_status()
        return response.json().get("status") == "running"
    except requests.RequestException:
        return False


def call_api(question: str) -> dict[str, Any]:
    """Send a question to the FastAPI /ask endpoint."""
    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={"question": question},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _source_filename(metadata: dict[str, Any]) -> str:
    return str(metadata.get("filename") or metadata.get("source") or "unknown")


def _source_page(metadata: dict[str, Any]) -> str:
    page_label = metadata.get("page_label")
    if page_label is not None and str(page_label).strip():
        return str(page_label)
    page = metadata.get("page")
    if page is not None:
        try:
            return str(int(page) + 1)
        except (TypeError, ValueError):
            return str(page)
    return "unknown"


def render_answer(response: dict[str, Any]) -> None:
    """Display the generated answer."""
    if response.get("capability") == "multi":
        st.subheader("Capabilities Used")
        st.markdown("✓ SQL Analytics")
        st.markdown("✓ SOP Knowledge Base")
        st.divider()

    st.subheader("Answer")
    st.write(response.get("answer") or "")


def render_sources(response: dict[str, Any]) -> None:
    """Display retrieved SOP source citations."""
    st.subheader("Sources")
    sources = response.get("sources") or []
    if not sources:
        st.info("No sources retrieved.")
        return

    for source in sources:
        metadata = source.get("metadata") or {}
        filename = _source_filename(metadata)
        page = _source_page(metadata)
        score = float(source.get("score", 0.0))
        st.markdown(f"✓ **{filename}** (Page {page})")
        st.caption(f"Similarity: {score:.2f}")


def render_sidebar(api_connected: bool, last_response: Optional[dict[str, Any]]) -> None:
    """Display system status in the sidebar."""
    with st.sidebar:
        st.header("System Status")
        label = "FastAPI Connected" if api_connected else "FastAPI Disconnected"
        st.write(f"• {label}")
        st.write("• RAG Capability")
        capability = last_response.get("capability") if last_response else None
        if capability == "multi":
            st.write("• SQL Analytics")
            st.write("• SOP Knowledge Base")
        source_count = len(last_response.get("sources", [])) if last_response else 0
        st.write(f"• Number of Sources Returned: {source_count}")
        timestamp = st.session_state.get("last_query_timestamp") or "—"
        st.write(f"• Timestamp of last query: {timestamp}")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="GOFO Operations Intelligence Agent", layout="centered")
    st.title("GOFO Operations Intelligence Agent")

    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("last_query_timestamp", None)

    render_sidebar(check_api_status(), st.session_state.last_response)

    question = st.text_input("Question", placeholder="What is CBT?")
    _, ask_col, _ = st.columns([2, 1, 2])
    with ask_col:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if ask_clicked:
        trimmed = question.strip()
        if not trimmed:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching SOP documents..."):
                try:
                    st.session_state.last_response = call_api(trimmed)
                    st.session_state.last_query_timestamp = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except (requests.ConnectionError, requests.Timeout):
                    st.error(CONNECT_ERROR)
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"API request failed: {detail}")
                except requests.RequestException as exc:
                    st.error(f"API request failed: {exc}")

    if st.session_state.last_response:
        st.divider()
        render_answer(st.session_state.last_response)
        st.divider()
        render_sources(st.session_state.last_response)


if __name__ == "__main__":
    main()
