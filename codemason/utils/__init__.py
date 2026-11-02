"""Utilities for CodeMason"""
from .state import get_cwd, set_cwd, get_original_cwd, set_original_cwd
from .log import log_error, debug_log
from .config import get_config, save_config
from .messages import (
    create_user_message,
    create_assistant_message,
    normalize_messages,
    get_last_assistant_message_id,
    get_text_from_message,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
)
from .tokens import (
    count_tokens,
)

__all__ = [
    'get_cwd',
    'set_cwd',
    'get_original_cwd',
    'set_original_cwd',
    'log_error',
    'debug_log',
    'get_config',
    'save_config',
    'create_user_message',
    'create_assistant_message',
    'normalize_messages',
    'get_last_assistant_message_id',
    'get_text_from_message',
    'INTERRUPT_MESSAGE',
    'INTERRUPT_MESSAGE_FOR_TOOL_USE',
    'count_tokens',
]
