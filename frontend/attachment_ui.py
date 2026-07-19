"""Streamlit UI rendering for attachment previews."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from frontend.attachment_preview import (
    build_preview_payload,
    file_icon,
    format_file_size,
    get_file_type,
    is_image_type,
    is_previewable,
)


def render_pending_attachment_previews(
    attachments: list[dict[str, Any]],
    preview_cache: dict[str, bytes],
    *,
    on_remove: Callable[[str], None],
) -> None:
    """Render removable attachment preview cards above the composer."""
    if not attachments:
        return

    st.markdown("##### Pending attachments")
    columns_per_row = 3
    for row_start in range(0, len(attachments), columns_per_row):
        row_items = attachments[row_start : row_start + columns_per_row]
        columns = st.columns(len(row_items))
        for offset, (column, attachment) in enumerate(zip(columns, row_items)):
            with column:
                _render_attachment_card(
                    attachment,
                    preview_cache,
                    removable=True,
                    on_remove=on_remove,
                    widget_key=f"pending-{row_start + offset}-{attachment.get('attachment_id', 'unknown')}",
                )


def render_message_attachments(
    attachments: list[dict[str, Any]],
    preview_cache: dict[str, bytes],
    *,
    message_key: str,
) -> None:
    """Render attachment cards for a historical chat message."""
    if not attachments:
        return
    for index, attachment in enumerate(attachments):
        attachment_id = attachment.get("attachment_id") or f"unknown-{index}"
        _render_attachment_card(
            attachment,
            preview_cache,
            removable=False,
            on_remove=None,
            widget_key=f"{message_key}-att-{index}-{attachment_id}",
        )


def _render_attachment_card(
    attachment: dict[str, Any],
    preview_cache: dict[str, bytes],
    *,
    removable: bool,
    on_remove: Callable[[str], None] | None,
    widget_key: str,
) -> None:
    attachment_id = attachment.get("attachment_id") or widget_key
    filename = attachment.get("original_filename") or attachment.get("filename") or "attachment"
    file_type = get_file_type(filename, attachment.get("content_type"))
    file_size = int(attachment.get("file_size") or 0)
    content = preview_cache.get(attachment_id, b"")
    icon = file_icon(file_type)

    with st.container(border=True):
        if is_image_type(file_type) and content:
            _render_image_thumbnail(content, filename)
        else:
            st.markdown(f"### {icon}")
            st.markdown(f"**{filename}**")
            st.caption(f"{file_type.upper()} · {format_file_size(file_size)}")

        action_cols = st.columns([3, 1] if removable else [1])
        with action_cols[0]:
            if is_previewable(file_type) and content:
                if st.button("Preview", key=f"preview-{widget_key}", use_container_width=True):
                    _open_preview_dialog(attachment, content, dialog_key=widget_key)
            elif content:
                st.caption("Preview unavailable for this file type.")
            else:
                st.caption("Preview unavailable.")

        if removable and on_remove is not None:
            with action_cols[1]:
                if st.button("×", key=f"remove-{widget_key}", help="Remove attachment"):
                    on_remove(attachment_id)


def _render_image_thumbnail(content: bytes, filename: str) -> None:
    try:
        st.image(content, caption=filename, use_container_width=True)
    except Exception:
        st.markdown(f"**{filename}**")
        st.caption("Unable to preview this file.")


@st.dialog("Attachment Preview")
def _open_preview_dialog(attachment: dict[str, Any], content: bytes, *, dialog_key: str) -> None:
    filename = attachment.get("original_filename") or attachment.get("filename") or "attachment"
    file_type = get_file_type(filename, attachment.get("content_type"))
    file_size = int(attachment.get("file_size") or 0)
    st.markdown(f"**{filename}**")
    st.caption(f"{file_type.upper()} · {format_file_size(file_size)}")
    render_attachment_preview(attachment, content, widget_key=dialog_key)


def render_attachment_preview(
    attachment: dict[str, Any],
    content: bytes,
    *,
    sheet_name: str | None = None,
    widget_key: str = "preview",
) -> None:
    """Render expanded preview content inside a dialog or container."""
    filename = attachment.get("original_filename") or attachment.get("filename") or "attachment"
    file_type = get_file_type(filename, attachment.get("content_type"))
    payload = build_preview_payload(
        filename=filename,
        file_type=file_type,
        content=content,
        sheet_name=sheet_name,
    )
    preview_type = payload.get("preview_type")

    if payload.get("error"):
        st.warning(payload["error"])
        return

    if preview_type == "image":
        _render_image_preview(payload)
    elif preview_type == "pdf":
        _render_pdf_preview(payload)
    elif preview_type == "table":
        _render_table_preview(payload, attachment, content, widget_key=widget_key)
    elif preview_type == "text":
        _render_text_preview(payload)
    else:
        st.info(payload.get("message") or "Preview unavailable for this file type.")


def _render_image_preview(payload: dict[str, Any]) -> None:
    content = payload.get("content")
    if not content:
        st.warning("Unable to preview this file.")
        return
    st.image(content, use_container_width=True)


def _render_pdf_preview(payload: dict[str, Any]) -> None:
    page_count = payload.get("page_count", 0)
    st.write(f"Pages: {page_count}")
    first_page_text = payload.get("first_page_text") or ""
    if first_page_text:
        st.markdown("**First page excerpt**")
        st.text(first_page_text)
    else:
        st.info("This PDF has little or no extractable text. It may be scanned or image-based.")

    pages = payload.get("pages") or []
    for page in pages[:3]:
        text = page.get("text") or ""
        if text:
            st.markdown(f"**Page {page.get('page')}**")
            st.text(text[:2000])


def _render_table_preview(
    payload: dict[str, Any],
    attachment: dict[str, Any],
    content: bytes,
    *,
    widget_key: str,
) -> None:
    sheet_names = payload.get("sheet_names") or []
    active_sheet = payload.get("active_sheet")
    if sheet_names:
        selected_sheet = st.selectbox(
            "Sheet",
            sheet_names,
            index=sheet_names.index(active_sheet) if active_sheet in sheet_names else 0,
            key=f"sheet-{widget_key}",
        )
        if selected_sheet != active_sheet:
            filename = attachment.get("original_filename") or attachment.get("filename") or "attachment"
            file_type = get_file_type(filename, attachment.get("content_type"))
            payload = build_preview_payload(
                filename=filename,
                file_type=file_type,
                content=content,
                sheet_name=selected_sheet,
            )
            if payload.get("error"):
                st.warning(payload["error"])
                return

    rows = payload.get("rows") or []
    if not rows:
        st.warning("No rows available to preview.")
        return
    st.dataframe(rows, use_container_width=True)
    total_rows = payload.get("total_rows", len(rows))
    if payload.get("truncated"):
        st.caption(f"Preview truncated. Showing {len(rows)} of {total_rows} rows.")
    else:
        st.caption(f"Showing all {total_rows} row(s).")


def _render_text_preview(payload: dict[str, Any]) -> None:
    text = payload.get("text") or ""
    if not text:
        st.warning("No text available to preview.")
        return
    language = "json" if payload.get("file_type") == "json" else None
    st.code(text, language=language)
    if payload.get("truncated"):
        st.caption("Preview truncated.")
