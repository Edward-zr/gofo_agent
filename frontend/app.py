"""Streamlit dashboard for GOFO Operations Intelligence."""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import requests
import streamlit as st

from frontend.attachment_preview import (
    attachment_identity,
    merge_pending_attachments,
    remove_pending_attachment,
)
from frontend.attachment_ui import (
    render_message_attachments,
    render_pending_attachment_previews,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

SUPPORTED_TYPES = [
    "pdf",
    "csv",
    "xlsx",
    "xls",
    "txt",
    "md",
    "docx",
    "png",
    "jpg",
    "jpeg",
    "webp",
]

COMPOSER_CSS = """
<style>
[data-testid="stBottomBlock"] {
    background: #0e1117;
    border-top: 1px solid rgba(128, 128, 128, 0.35);
    padding: 0.65rem 1rem 1rem;
    box-shadow: 0 -8px 28px rgba(0, 0, 0, 0.35);
}

[data-testid="stBottomBlock"] [data-testid="column"] {
    display: flex;
    align-items: center;
}

[data-testid="stBottomBlock"] h5 {
    margin: 0 0 0.35rem 0;
    font-size: 0.85rem;
    color: rgba(250, 250, 250, 0.72);
}

[data-testid="stBottomBlock"] [data-testid="stChatInput"] {
    flex: 1;
}

[data-testid="stBottomBlock"] [data-testid="stChatInput"] textarea {
    border-radius: 1.25rem;
}
</style>
"""


def _bottom_container():
    """Return Streamlit's bottom-pinned layout container."""
    if hasattr(st, "bottom"):
        return st.bottom
    from streamlit import _bottom

    return _bottom


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


def upload_attachments(files: list[Any], session_id: str) -> list[dict[str, Any]]:
    """Upload files to the FastAPI attachment endpoint."""
    multipart = []
    for upload in files:
        multipart.append(
            ("files", (upload.name, upload.getvalue(), upload.type or "application/octet-stream"))
        )
    response = requests.post(
        f"{API_BASE_URL}/attachments",
        files=multipart,
        data={"session_id": session_id},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("attachments") or []


def ask_api(question: str, session_id: str, attachment_ids: list[str] | None = None) -> dict[str, Any]:
    """Send a chat question to the FastAPI backend."""
    payload: dict[str, Any] = {"question": question, "session_id": session_id}
    if attachment_ids:
        payload["attachments"] = attachment_ids
    response = requests.post(
        f"{API_BASE_URL}/ask",
        json=payload,
        timeout=300,
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


def render_charts(response: dict[str, Any]) -> None:
    """Render Python-generated chart images from attachment analysis."""
    charts = response.get("charts") or (response.get("analysis") or {}).get("charts") or []
    if not charts:
        return

    st.subheader("Visualizations")
    for chart in charts:
        title = chart.get("title") or chart.get("type") or "Chart"
        st.markdown(f"**{title}**")
        image_base64 = chart.get("image_base64")
        if image_base64:
            try:
                import base64

                image_bytes = base64.b64decode(image_base64)
                st.image(image_bytes, use_container_width=True)
                continue
            except Exception as exc:
                st.caption(f"Unable to render chart image: {exc}")

        data = chart.get("data") or []
        if not data:
            st.caption("No chart data available.")
            continue
        chart_type = chart.get("type")
        try:
            import pandas as pd

            frame = pd.DataFrame(data)
            if chart_type in {"bar", "histogram"}:
                x = chart.get("x")
                y = chart.get("y")
                if x and y and x in frame.columns and y in frame.columns:
                    st.bar_chart(frame, x=x, y=y, use_container_width=True)
                else:
                    st.dataframe(frame, use_container_width=True)
            elif chart_type == "line":
                x = chart.get("x")
                y = chart.get("y")
                if x and y and x in frame.columns and y in frame.columns:
                    st.line_chart(frame, x=x, y=y, use_container_width=True)
                else:
                    st.dataframe(frame, use_container_width=True)
            elif chart_type == "scatter":
                x = chart.get("x")
                y = chart.get("y")
                if x and y and x in frame.columns and y in frame.columns:
                    st.scatter_chart(frame, x=x, y=y, use_container_width=True)
                else:
                    st.dataframe(frame, use_container_width=True)
            elif chart_type == "pie":
                label = chart.get("label")
                value = chart.get("value")
                if label and value and label in frame.columns and value in frame.columns:
                    st.bar_chart(frame, x=label, y=value, use_container_width=True)
                else:
                    st.dataframe(frame, use_container_width=True)
            else:
                st.dataframe(frame, use_container_width=True)
        except Exception as exc:
            st.caption(f"Unable to render chart: {exc}")
            st.json({key: value for key, value in chart.items() if key != "image_base64"})


def render_structured_sections(response: dict[str, Any]) -> None:
    """Render analysis, recommendations, SQL data, and SOP sources."""
    render_kpis(response)
    render_charts(response)

    analysis = response.get("analysis") or {}
    root_cause = analysis.get("root_cause") or {}
    issue = root_cause.get("issue")
    # Skip generic canned root-cause text for attachment analyses.
    if root_cause and issue and issue != "Operational metric requires investigation.":
        st.subheader("Root Cause Analysis")
        st.write(issue)
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

    file_summary = analysis.get("file_context_summary") or {}
    sources = file_summary.get("source_references") or analysis.get("attachment_filenames") or []
    if sources:
        st.subheader("Attachment Sources")
        for source in sources:
            st.write(f"- {source}")

    if response.get("sql"):
        with st.expander("Generated SQL"):
            st.code(response["sql"], language="sql")

    data = response.get("data") or []
    if data:
        with st.expander("Analysis Data"):
            st.dataframe(data, use_container_width=True)

    rag_sources = response.get("sources") or []
    if rag_sources:
        st.subheader("Knowledge Sources")
        for source in rag_sources:
            metadata = source.get("metadata") or {}
            filename = metadata.get("filename") or metadata.get("source") or "unknown"
            page = metadata.get("page_label") or metadata.get("page") or "unknown"
            st.write(f"- {filename}, page {page}")


def _init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", str(uuid4()))
    st.session_state.setdefault("pending_attachments", [])
    st.session_state.setdefault("preview_cache", {})
    st.session_state.setdefault("processed_upload_keys", [])


def _remove_pending_attachment(attachment_id: str) -> None:
    st.session_state.pending_attachments = remove_pending_attachment(
        st.session_state.pending_attachments,
        attachment_id,
    )
    st.session_state.preview_cache.pop(attachment_id, None)


def _handle_selected_uploads(uploads: list[Any]) -> None:
    """Upload newly selected files and add them to pending attachments."""
    if not uploads:
        return

    new_files = []
    contents_by_name: dict[str, bytes] = {}
    processed_keys = set(st.session_state.processed_upload_keys)

    for upload in uploads:
        content = upload.getvalue()
        identity = attachment_identity(upload.name, len(content), content)
        if identity in processed_keys:
            continue
        new_files.append(upload)
        contents_by_name[upload.name] = content

    if not new_files:
        return

    try:
        uploaded = upload_attachments(new_files, st.session_state.session_id)
        merged, cache_updates = merge_pending_attachments(
            st.session_state.pending_attachments,
            uploaded,
            contents_by_name,
        )
        st.session_state.pending_attachments = merged
        st.session_state.preview_cache.update(cache_updates)
        for upload in new_files:
            content = upload.getvalue()
            processed_keys.add(attachment_identity(upload.name, len(content), content))
        st.session_state.processed_upload_keys = sorted(processed_keys)
    except requests.RequestException as exc:
        st.error(_http_error_detail(exc) if isinstance(exc, requests.HTTPError) else str(exc))


def render_unified_composer() -> str | None:
    """Render ChatGPT-style composer pinned to the bottom via Streamlit's bottom block."""
    st.markdown(COMPOSER_CSS, unsafe_allow_html=True)
    pending = st.session_state.pending_attachments

    with _bottom_container():
        render_pending_attachment_previews(
            pending,
            st.session_state.preview_cache,
            on_remove=_remove_pending_attachment,
        )

        left_col, right_col = st.columns([0.7, 11.3], vertical_alignment="bottom")
        with left_col:
            with st.popover("＋", use_container_width=True):
                uploads = st.file_uploader(
                    "Add files",
                    type=SUPPORTED_TYPES,
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                )
                if uploads:
                    _handle_selected_uploads(uploads)

        with right_col:
            question = st.chat_input(
                "Ask about GOFO operations...",
                key="gofo_chat_input",
            )

    if not question:
        return None
    question = question.strip()
    return question or None


def render_chat() -> None:
    """Render persistent Streamlit session chat."""
    _init_session_state()

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            render_message_attachments(
                message.get("attachments") or [],
                st.session_state.preview_cache,
                message_key=f"history-{message_index}",
            )
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("response"):
                render_structured_sections(message["response"])

    question = render_unified_composer()
    if question is None:
        return

    pending_attachments = list(st.session_state.pending_attachments)
    attachment_ids = [item["attachment_id"] for item in pending_attachments if item.get("attachment_id")]
    live_message_index = len(st.session_state.messages)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "attachments": pending_attachments,
        }
    )
    with st.chat_message("user"):
        render_message_attachments(
            pending_attachments,
            st.session_state.preview_cache,
            message_key=f"live-{live_message_index}",
        )
        st.write(question)

    with st.chat_message("assistant"):
        status = "Analyzing GOFO operations..."
        if pending_attachments:
            names = ", ".join(item.get("original_filename", "file") for item in pending_attachments)
            status = f"Processing {names}..."
        with st.spinner(status):
            try:
                response = ask_api(question, st.session_state.session_id, attachment_ids=attachment_ids or None)
                answer = response.get("answer") or ""
                st.write(answer)
                render_structured_sections(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "response": response}
                )
                st.session_state.pending_attachments = []
            except requests.ConnectionError:
                _show_error("API offline. Start the backend with `uvicorn api.server:app --reload`.")
            except requests.Timeout:
                _show_error("The API request timed out. Please try again.")
            except requests.HTTPError as exc:
                _show_error(_http_error_detail(exc))
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
        payload = exc.response.json()
        detail = payload.get("detail") or payload.get("error")
    except ValueError:
        detail = exc.response.text
    return f"API error: {detail}"


def _show_error(message: str) -> None:
    st.error(message)
    st.session_state.messages.append({"role": "assistant", "content": message})


if __name__ == "__main__":
    main()
