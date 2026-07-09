"""Interactive REPL for the GOFO Operations Intelligence Agent."""

from __future__ import annotations

import sys

from agent import ask
from cli.colors import Colors
from cli.display import print_query_response
from core.error_handler import handle_error
from core.logger import get_logger
from tools.memory import (
    ConversationMemory,
    detect_repair,
    references_previous_result,
    resolve,
)

_EXIT_COMMANDS = frozenset({"exit", "quit"})
logger = get_logger("cli.repl")


def print_banner(colors: Colors) -> None:
    """Print the REPL welcome banner."""
    print()
    print(colors.header("GOFO Operations Intelligence Agent"))
    print(colors.divider("-", 34))
    print()


def run_repl(*, debug: bool = False) -> int:
    """Run an interactive question-and-answer loop."""
    colors = Colors()
    memory = ConversationMemory()
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
            repair = detect_repair(user_input, memory)
            question_for_resolver = (
                str(repair["corrected_question"]) if repair.get("is_repair") else user_input
            )
            resolved_question = resolve(question_for_resolver, memory)
            result_context = None
            if references_previous_result(user_input) or references_previous_result(resolved_question):
                result_context = memory.get_current_state().get("last_result_context")
            response = ask(resolved_question, result_context=result_context)
            memory.add_turn(
                user_question=user_input,
                resolved_question=resolved_question,
                response=response,
            )
            response.original_question = user_input
            response.resolved_question = resolved_question
            response.repair_detected = bool(repair.get("is_repair"))
            response.repair_type = repair.get("repair_type")
            response.changed_dimension = repair.get("changed_dimension")
            response.memory_history_count = len(memory.get_recent_history())
            response.memory_current_state = memory.get_current_state()
            response.last_result_context = memory.get_current_state().get("last_result_context")
        except Exception as exc:
            logger.exception("REPL request failed: %s", handle_error(exc))
            print(colors.wrap(f"Error: {handle_error(exc)}", colors.RED), file=sys.stderr)
            continue

        print_query_response(response, colors=colors, debug=debug)
