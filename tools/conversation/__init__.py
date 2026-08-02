"""Semantic conversation state for the GOFO reasoning pipeline."""

from tools.conversation.resolver import ConversationResolver, ConversationResolution, resolution_from_dict, resolution_to_dict

__all__ = [
    "ConversationResolver",
    "ConversationResolution",
    "resolution_from_dict",
    "resolution_to_dict",
]
