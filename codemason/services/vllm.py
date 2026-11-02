"""
vLLM service — OpenAI-compatible vLLM server variant of the model service.

All OpenAI-compatible logic lives in services/_openai_common.py. llama.cpp and
vLLM differ by two request fields, both pinned on here:

  - stream_options.continuous_usage_stats — a running completion_tokens on every
    chunk, so the live spinner shows the server's exact count instead of a
    per-chunk tally.
  - repetition_detection — vLLM's own n-gram loop detector, which ends a
    generation that has begun repeating itself and says so in finish_reason.
    llama.cpp has no equivalent field.

Otherwise this shim delegates everything.

Selected at runtime by services/backend.py when the server is vLLM.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ..types.query import UserMessage, AssistantMessage
from ._openai_common import query_openai_compatible
from .vision import enabled_limit

# Back-compat re-exports (see llamacpp.py for the rationale).
from ._openai_common import (  # noqa: F401
    ChatMessage,
    clean_special_tokens,
    convert_to_chat_messages,
    build_system_prompt_with_cwd_emphasis,
    truncate_tool_result,
    get_simple_description,
    recover_leaked_tool_calls,
)


async def query_vllm(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_prompt: List[str],
    tools: List[Any],
    signal: Optional[asyncio.Event] = None,
    options: Optional[Dict[str, Any]] = None,
    on_token_update: Optional[Callable[[int], None]] = None,
    on_content_delta: Optional[Callable[[str], None]] = None,
    on_thinking_update: Optional[Callable[[int], None]] = None,
) -> AssistantMessage:
    """Query a vLLM server. Enables continuous_usage_stats so the live token
    count is driven by the server's exact running completion_tokens (counting
    both answer and reasoning deltas) rather than a per-chunk tally.

    services/backend.py dispatches here (as query_vllm) when the server is vLLM;
    the neutral public entry point is services.backend.query_model."""
    return await query_openai_compatible(
        messages,
        system_prompt,
        tools,
        signal=signal,
        options=options,
        on_token_update=on_token_update,
        on_content_delta=on_content_delta,
        on_thinking_update=on_thinking_update,
        continuous_usage_stats=True,
        repetition_detection=True,
        reasoning_roundtrip=True,
        # Images go out as image_url parts only when the server says it takes
        # them, and never more than it takes (asked once, cached —
        # services/vision.py).
        max_images=enabled_limit(),
    )
