"""Validate uploaded files for type, size, and safety."""

from __future__ import annotations

from pathlib import Path

import config
from core.errors import FileError, UnsupportedFileError


def validate_upload(
    *,
    filename: str,
    content_type: str | None,
    file_size: int,
    content: bytes,
) -> str:
    """Validate an upload and return the normalized file extension."""
    if not filename.strip():
        raise FileError("Uploaded file is missing a filename.")
    if file_size <= 0 or not content:
        raise FileError("The uploaded file is empty.")
    if file_size > config.MAX_UPLOAD_SIZE_BYTES:
        raise FileError(
            f"The uploaded file exceeds the configured size limit of {config.MAX_UPLOAD_SIZE_MB} MB."
        )

    extension = Path(filename).suffix.lower().lstrip(".")
    if not extension:
        raise UnsupportedFileError("This file type is not supported.")
    if extension in config.BLOCKED_UPLOAD_EXTENSIONS:
        raise UnsupportedFileError("This file type is not supported.")
    if extension not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise UnsupportedFileError("This file type is not supported.")

    _validate_content_type(extension, content_type)
    return extension


def _validate_content_type(extension: str, content_type: str | None) -> None:
    if not content_type:
        return
    normalized = content_type.lower()
    if normalized.startswith("application/x-msdownload"):
        raise UnsupportedFileError("This file type is not supported.")
    if normalized.startswith("application/x-executable"):
        raise UnsupportedFileError("This file type is not supported.")
