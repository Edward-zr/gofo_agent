"""Image attachment processing with multimodal analysis."""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

import config
from tools.files.models import AttachmentMetadata, ProcessedFileContext, ProcessingStatus


def process_image(
    metadata: AttachmentMetadata,
    content: bytes,
    *,
    question: str | None = None,
) -> ProcessedFileContext:
    """Extract image metadata and analyze the image with the configured multimodal model."""
    try:
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        image_format = (image.format or metadata.file_type).lower()
    except Exception:
        return _failed(metadata, "The image could not be read.")

    analysis = _analyze_image(content, image_format, question=question)
    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="image",
        processing_status=ProcessingStatus.READY,
        summary=analysis.get("description") or "Image uploaded for operational analysis.",
        image_analysis=analysis,
        file_schema={"width": width, "height": height, "format": image_format, "file_size": metadata.file_size},
        statistics={"dimensions": f"{width}x{height}"},
        full_data={"analysis": analysis},
        source_references=[metadata.original_filename],
    )


def _analyze_image(content: bytes, image_format: str, *, question: str | None = None) -> dict[str, Any]:
    """Use OpenAI vision to analyze an uploaded image."""
    prompt = question or (
        "Analyze this operations-related image. Describe what you see, extract any visible "
        "metrics, entities, tables, errors, or operational issues. Do not invent details."
    )
    mime = _mime_type(image_format)
    encoded = base64.b64encode(content).decode("ascii")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.require_openai_api_key())
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return {
            "description": "Image uploaded, but automated image analysis is unavailable.",
            "analysis": "",
            "error": str(exc),
        }

    return {
        "description": text,
        "analysis": text,
        "extracted_text": _extract_visible_text(text),
        "observed_metrics": _extract_metrics(text),
        "observed_entities": _extract_entities(text),
    }


def _mime_type(image_format: str) -> str:
    mapping = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }
    return mapping.get(image_format.lower(), "image/jpeg")


def _extract_visible_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if any(char.isdigit() for char in line)]
    return "\n".join(lines)


def _extract_metrics(text: str) -> list[str]:
    keywords = ("rate", "count", "volume", "delay", "failed", "completion", "package", "%")
    return [line.strip() for line in text.splitlines() if any(keyword in line.lower() for keyword in keywords)]


def _extract_entities(text: str) -> list[str]:
    entities = []
    for token in text.replace(",", " ").split():
        if token.endswith("Hub") or (" " in token and token[0].isupper()):
            entities.append(token.strip())
    return entities[:10]


def _failed(metadata: AttachmentMetadata, message: str) -> ProcessedFileContext:
    return ProcessedFileContext(
        attachment_id=metadata.attachment_id,
        filename=metadata.original_filename,
        file_type="image",
        processing_status=ProcessingStatus.FAILED,
        error_message=message,
        source_references=[metadata.original_filename],
    )
