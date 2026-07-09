"""
GOFO Operations Intelligence Agent — query entry point.

Collects user input, delegates to the agent API, and displays results.
"""

from __future__ import annotations

import sys

from cli.repl import run_repl
from core.agent import GOFOAgent
from core.error_handler import handle_error
from core.logger import get_logger

logger = get_logger("query")


def run_query(question: str) -> dict:
    """Run a query through the agent; reusable from app.py and LangGraph."""
    return GOFOAgent().ask(question)


def main() -> None:
    """CLI entry point."""
    debug = "--debug" in sys.argv
    if debug:
        raise SystemExit(run_repl(debug=True))

    agent = GOFOAgent()
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)

        if question.lower() in {"exit", "quit"}:
            raise SystemExit(0)
        if not question:
            continue

        try:
            response = agent.ask(question)
            print(response.get("answer", ""))
        except Exception as exc:
            logger.exception("CLI request failed: %s", handle_error(exc))
            print(f"Error: {handle_error(exc)}", file=sys.stderr)


if __name__ == "__main__":
    main()
