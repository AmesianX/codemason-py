"""
LlamaCPP service — llama.cpp server variant of the model service.

All OpenAI-compatible logic lives in services/_openai_common.py. llama.cpp and
vLLM differ by exactly one request field: vLLM adds
stream_options.continuous_usage_stats for a running per-chunk token count, while
llama.cpp emits usage only in the final chunk. This shim pins that flag OFF and
otherwise delegates everything.

Selected at runtime by services/backend.py when the server is llama.cpp.
Equivalent to src/services/llamacpp.ts.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..types.query import UserMessage, AssistantMessage
from ._openai_common import query_openai_compatible

# Back-compat re-exports. External code only imports query_model (via
# services.backend), but this module historically exposed these helpers, so keep
# them importable from here to avoid surprising any in-tree caller.
from ._openai_common import (  # noqa: F401
    ChatMessage,
    clean_special_tokens,
    convert_to_chat_messages,
    build_system_prompt_with_cwd_emphasis,
    truncate_tool_result,
    get_simple_description,
    recover_leaked_tool_calls,
)


async def query_llamacpp(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_prompt: List[str],
    tools: List[Any],
    signal: Optional[asyncio.Event] = None,
    options: Optional[Dict[str, Any]] = None,
    on_token_update: Optional[Callable[[int], None]] = None,
    on_content_delta: Optional[Callable[[str], None]] = None,
    on_thinking_update: Optional[Callable[[int], None]] = None,
) -> AssistantMessage:
    """Query a llama.cpp server. Usage arrives only in the final stream chunk,
    so continuous_usage_stats is left off (the live count uses a per-chunk tally
    until the final exact count lands)."""
    return await query_openai_compatible(
        messages,
        system_prompt,
        tools,
        signal=signal,
        options=options,
        on_token_update=on_token_update,
        on_content_delta=on_content_delta,
        on_thinking_update=on_thinking_update,
        continuous_usage_stats=False,
    )
