"""Interactive REPL for the GOFO Operations Intelligence Agent."""

from __future__ import annotations

import sys

from cli.colors import Colors
from cli.display import print_query_response
from core.agent import GOFOAgent
from core.error_handler import handle_error
from core.logger import get_logger
from core.models import QueryResponse

_EXIT_COMMANDS = frozenset({"exit", "quit"})
logger = get_logger("cli.repl")


def print_banner(colors: Colors) -> None:
    """Print the REPL welcome banner."""
    print()
    print(colors.header("GOFO Operations Intelligence Agent"))
    print(colors.divider("-", 34))
    print()


def run_repl(*, debug: bool = False) -> int:
    """Run an interactive question-and-answer loop using the production agent pipeline."""
    colors = Colors()
    agent = GOFOAgent()
    print_banner(colors)

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue

        if user_input.lower() in _EXIT_COMMANDS:
            return 0

        try:
            payload = agent.ask(user_input)
            response = QueryResponse.model_validate(payload.get("raw") or payload)
            if payload.get("analysis"):
                response.classifier_intent = payload["analysis"].get("intent")
                response.inherited_context = payload["analysis"].get("inherited_context")
                response.business_findings = payload["analysis"].get("business_findings")
                response.memory_updated = payload["analysis"].get("memory_updated")
                response.sql_cache_hit = payload["analysis"].get("sql_cache_hit")
            print_query_response(response, colors=colors, debug=debug)
        except Exception as exc:
            logger.exception("REPL request failed: %s", handle_error(exc))
            print(colors.wrap(f"Error: {handle_error(exc)}", colors.RED), file=sys.stderr)
            continue
