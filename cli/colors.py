"""Terminal color helpers with automatic TTY detection."""

from typing import Optional
import os
import sys

class Colors:
    """ANSI color codes; disabled when output is not a TTY or NO_COLOR is set."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"

    def __init__(self, enabled: Optional[bool] = None) -> None:
        if enabled is None:
            enabled = sys.stdout.isatty() and not os.getenv("NO_COLOR")
        self.enabled = enabled

    def wrap(self, text: str, *codes: str) -> str:
        if not self.enabled:
            return text
        return "".join(codes) + text + self.RESET

    def label(self, text: str) -> str:
        return self.wrap(text, self.BOLD, self.YELLOW)

    def value(self, text: str) -> str:
        return self.wrap(text, self.CYAN)

    def header(self, text: str) -> str:
        return self.wrap(text, self.BOLD, self.GREEN)

    def divider(self, char: str = "=", width: int = 40) -> str:
        line = char * width
        return self.wrap(line, self.DIM) if self.enabled else line
