"""Interactive CLI prompts."""

from __future__ import annotations

from typing import Optional

from cli.colors import Colors


def prompt_question(colors: Optional[Colors] = None) -> str:
    """Ask the user for a question and return the trimmed input."""
    _ = colors or Colors()
    try:
        question = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return question
