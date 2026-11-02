"""
Message utilities for CodeMason
Equivalent to src/utils/messages.tsx
"""

import uuid
from typing import Any, Dict, List, Optional, Union


# Verbatim from utils/messages.ts:207-209 — the SINGLE source of truth. query.py
# and types/query.py import these (they previously each redefined their own
# divergent strings, so is_interrupt_message silently failed to match
# query-produced interrupts).
INTERRUPT_MESSAGE = "[Request interrupted by user]"
INTERRUPT_MESSAGE_FOR_TOOL_USE = "[Request interrupted by user for tool use]"

# constants/messages.ts NO_CONTENT_MESSAGE.
NO_CONTENT_MESSAGE = "(no content)"

# Verbatim from utils/messages.ts:213 — the tool_result the model receives
# when the USER rejects a tool at the permission prompt. The 'STOP … wait for
# the user' phrasing is load-bearing: a weak 'User denied permission' reads as
# an environment error, and the model keeps retrying the same/adjacent
# commands instead of stopping.
REJECT_MESSAGE = (
    "The user doesn't want to proceed with this tool use. The tool use was "
    "rejected (eg. if it was a file edit, the new_string was NOT written to "
    "the file). STOP what you are doing and wait for the user to tell you "
    "how to proceed."
)


def _task_output_tool_name() -> str:
    """TASK_OUTPUT_TOOL_NAME, resolved late: utils.messages is imported by the
    tools, so naming it at module scope would close an import cycle."""
    from ..tools.task_output_tool import TASK_OUTPUT_TOOL_NAME
    return TASK_OUTPUT_TOOL_NAME


def _send_message_tool_name() -> str:
    """SEND_MESSAGE_TOOL_NAME, resolved late, for the same cycle reason."""
    from ..tools.send_message_tool import SEND_MESSAGE_TOOL_NAME
    return SEND_MESSAGE_TOOL_NAME


def wrap_in_system_reminder(content: str) -> str:
    """Wrap text as a system reminder (wrapInSystemReminder)."""
    return f"<system-reminder>\n{content}\n</system-reminder>"


def wrap_messages_in_system_reminder(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Wrap each message's text in a system reminder (wrapMessagesInSystemReminder).

    String content is wrapped whole; block content has each text block wrapped
    and every other block left alone.
    """
    out = []
    for message in messages:
        inner = message.get("message", {})
        content = inner.get("content")
        if isinstance(content, str):
            wrapped: Any = wrap_in_system_reminder(content)
        elif isinstance(content, list):
            wrapped = [
                {**block, "text": wrap_in_system_reminder(block["text"])}
                if isinstance(block, dict) and block.get("type") == "text"
                else block
                for block in content
            ]
        else:
            out.append(message)
            continue
        out.append({**message, "message": {**inner, "content": wrapped}})
    return out


def wrap_command_text(raw: str, origin: Optional[str]) -> str:
    """Frame a message that arrived mid-turn, by who sent it (wrapCommandText).

    The framing is the instruction: a queued line is not just context, it has to
    say what the reader is expected to do about it and when.
    """
    if origin == "task-notification":
        return f"A background agent completed a task:\n{raw}"
    if origin == "coordinator":
        return (
            f"The coordinator sent a message while you were working:\n{raw}\n\n"
            "Address this before completing your current task."
        )
    return (
        f"The user sent a new message while you were working:\n{raw}\n\n"
        "IMPORTANT: After completing your current task, you MUST address the "
        "user's message above. Do not ignore it."
    )


def render_task_status_attachment(attachment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A task_status attachment, as the messages the model reads
    (the `case 'task_status'` arm of getAttachmentMessages).

    Only emitted after a compaction, where the message that spawned the agent
    is gone — which is why the running branch says not to spawn a duplicate,
    and why it names SendMessage: the model has just lost the record of the
    spawn, so it needs to be told it can still reach the agent.
    """
    status = attachment.get("status")
    task_id = attachment.get("taskId")
    description = attachment.get("description")
    output_file_path = attachment.get("outputFilePath")
    delta_summary = attachment.get("deltaSummary")
    display_status = "stopped" if status == "killed" else status

    def _meta(text: str) -> List[Dict[str, Any]]:
        message = create_user_message(wrap_in_system_reminder(text))
        message["isMeta"] = True
        return [message]

    # For stopped tasks, keep it brief — the work was interrupted and the raw
    # transcript delta isn't useful context.
    if status == "killed":
        return _meta(f'Task "{description}" ({task_id}) was stopped by the user.')

    # For running tasks, warn against spawning a duplicate.
    if status == "running":
        parts = [f'Background agent "{description}" ({task_id}) is still running.']
        if delta_summary:
            parts.append(f"Progress: {delta_summary}")
        if output_file_path:
            parts.append(
                "Do NOT spawn a duplicate. You will be notified when it completes. "
                f"You can read partial output at {output_file_path} or send it a "
                f"message with {_send_message_tool_name()}."
            )
        else:
            parts.append(
                "Do NOT spawn a duplicate. You will be notified when it completes. "
                f"You can check its progress with the {_task_output_tool_name()} "
                f"tool or send it a message with {_send_message_tool_name()}."
            )
        return _meta(" ".join(parts))

    # For completed/failed tasks, include the full delta.
    parts = [
        f"Task {task_id}",
        f"(type: {attachment.get('taskType')})",
        f"(status: {display_status})",
        f"(description: {description})",
    ]
    if delta_summary:
        parts.append(f"Delta: {delta_summary}")
    if output_file_path:
        parts.append(f"Read the output file to retrieve the result: {output_file_path}")
    else:
        parts.append(f"You can check its output using the {_task_output_tool_name()} tool.")
    return _meta(" ".join(parts))


def create_user_message(content: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Create a user message"""
    if isinstance(content, str):
        content_blocks = [{'type': 'text', 'text': content}]
    else:
        content_blocks = content

    return {
        'type': 'user',
        'uuid': str(uuid.uuid4()),
        'message': {
            'role': 'user',
            'content': content_blocks,
        }
    }


def create_assistant_message(
    content: Union[str, List[Dict[str, Any]]],
    usage: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Create an assistant message (createAssistantMessage). Empty string
    content becomes NO_CONTENT_MESSAGE; the model is taken from config rather
    than hardcoded."""
    if isinstance(content, str):
        content_blocks = [{'type': 'text', 'text': content or NO_CONTENT_MESSAGE}]
    else:
        content_blocks = content

    if usage is None:
        usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

    from .config import get_config
    return {
        'type': 'assistant',
        'uuid': str(uuid.uuid4()),
        'message': {
            'id': f"msg_{uuid.uuid4().hex[:24]}",
            'type': 'message',
            'role': 'assistant',
            'content': content_blocks,
            'model': get_config().model_name,
            'stop_reason': 'end_turn',
            'usage': usage,
        }
    }


def derive_uuid(parent_uuid: str, index: int) -> str:
    """deriveUUID (messages.ts:725): parent[:24] + index as zero-padded 12-hex.
    Used to give each split content block a stable, unique uuid."""
    hex_part = format(index, '012x')
    return f"{parent_uuid[:24]}{hex_part}"


def get_last_assistant_message_id(messages: List[Any]) -> Optional[str]:
    """Get the ID of the last assistant message"""
    for message in reversed(messages):
        msg_type = message.type if hasattr(message, 'type') else message.get('type')
        if msg_type == 'assistant':
            msg = message.message if hasattr(message, 'message') else message.get('message', {})
            return msg.id if hasattr(msg, 'id') else msg.get('id')
    return None


def _msg_attr(m: Any, key: str, default: Any = None) -> Any:
    if hasattr(m, key):
        return getattr(m, key)
    if isinstance(m, dict):
        return m.get(key, default)
    return default


def normalize_messages(messages: List[Any]) -> List[Any]:
    """normalizeMessages (messages.ts:741): split each message so EVERY content
    block (text and tool_use, for both assistant and user roles) becomes its own
    single-block message. Once any message has >1 block, all SUBSEQUENT messages
    get fresh derive_uuid()-based uuids (isNewChain) to prevent duplicate uuids.
    progress / system / attachment pass through unchanged.

    (The previous port split only assistant messages with >1 tool_use, dropped
    text blocks, and reused the same uuid for every fragment.)"""
    normalized: List[Any] = []
    is_new_chain = False

    for message in messages:
        msg_type = _msg_attr(message, 'type')
        if msg_type not in ('assistant', 'user'):
            # progress / system / attachment / unknown — passthrough.
            normalized.append(message)
            continue

        msg_obj = _msg_attr(message, 'message', {})
        content = _msg_attr(msg_obj, 'content', None)
        if isinstance(content, str):  # user string content -> one text block
            content = [{'type': 'text', 'text': content}]
        if content is None:
            content = []

        is_new_chain = is_new_chain or len(content) > 1

        # Fast path: single block and no prior split → keep the original object.
        if len(content) <= 1 and not is_new_chain:
            normalized.append(message)
            continue

        # Coerce to str: AssistantMessage/UserMessage objects carry `uuid` as a
        # `UUID` object (field(default_factory=uuid4)), and derive_uuid slices it
        # (parent_uuid[:24]) — `'UUID' object is not subscriptable` otherwise.
        # Surfaces in the Agent path, where a sub-agent's multi-block assistant
        # message trips is_new_chain → derive_uuid.
        parent_uuid = str(_msg_attr(message, 'uuid', '') or '')
        base_msg = msg_obj if isinstance(msg_obj, dict) else {'content': content}
        top = message if isinstance(message, dict) else {}
        for index, block in enumerate(content):
            new_uuid = derive_uuid(parent_uuid, index) if is_new_chain else parent_uuid
            normalized.append({
                **top,
                'type': msg_type,
                'uuid': new_uuid,
                'message': {**base_msg, 'content': [block]},
            })

    return normalized


def _content_blocks(message: Any) -> Optional[List[Any]]:
    """The content list of a message, or None when it is not a list."""
    inner = _msg_attr(message, 'message')
    if inner is None:
        return None
    content = inner.get('content') if isinstance(inner, dict) else getattr(inner, 'content', None)
    return content if isinstance(content, list) else None


def _block_field(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def filter_unresolved_tool_uses(messages: List[Any]) -> List[Any]:
    """Drop assistant turns whose tool calls ALL went unanswered
    (filterUnresolvedToolUses).

    Not the same rule as filter_incomplete_tool_calls, and the difference is
    deliberate in the source: that one drops a turn if ANY of its tool calls is
    unanswered, this one only if EVERY one is. A turn that got some results back
    carries real work, and resume keeps it; a turn that got none is the
    interrupted tail and goes.

    The collection walks content blocks directly rather than normalising first,
    for the reason the source gives: normalising mints new uuids, and returning
    those would defeat the transcript's uuid dedup and grow the file on every
    resume.
    """
    tool_use_ids = set()
    tool_result_ids = set()
    for message in messages:
        if _msg_attr(message, 'type') not in ('user', 'assistant'):
            continue
        for block in _content_blocks(message) or []:
            block_type = _block_field(block, 'type')
            if block_type == 'tool_use':
                tool_use_ids.add(_block_field(block, 'id'))
            elif block_type == 'tool_result':
                tool_result_ids.add(_block_field(block, 'tool_use_id'))

    unresolved = {i for i in tool_use_ids if i not in tool_result_ids}
    if not unresolved:
        return messages

    kept = []
    for message in messages:
        if _msg_attr(message, 'type') != 'assistant':
            kept.append(message)
            continue
        blocks = _content_blocks(message)
        if blocks is None:
            kept.append(message)
            continue
        ids = [_block_field(b, 'id') for b in blocks
               if _block_field(b, 'type') == 'tool_use']
        if not ids or not all(i in unresolved for i in ids):
            kept.append(message)
    return kept


def _has_only_whitespace_text_content(blocks: List[Any]) -> bool:
    """True when every block is text and none of it says anything
    (hasOnlyWhitespaceTextContent)."""
    if not blocks:
        return False
    for block in blocks:
        if _block_field(block, 'type') != 'text':
            return False
        text = _block_field(block, 'text')
        if text is not None and str(text).strip() != '':
            return False
    return True


def filter_whitespace_only_assistant_messages(messages: List[Any]) -> List[Any]:
    """Drop assistant turns that are nothing but blank text
    (filterWhitespaceOnlyAssistantMessages).

    An empty content list is kept: that shape is handled elsewhere and means
    something different from "said only spaces".
    """
    kept = []
    for message in messages:
        if _msg_attr(message, 'type') != 'assistant':
            kept.append(message)
            continue
        blocks = _content_blocks(message)
        if blocks is None or len(blocks) == 0:
            kept.append(message)
            continue
        if not _has_only_whitespace_text_content(blocks):
            kept.append(message)
    return kept


_THINKING_TYPES = ('thinking', 'redacted_thinking')


def filter_orphaned_thinking_only_messages(messages: List[Any]) -> List[Any]:
    """Drop thinking-only assistant turns with nothing to merge into
    (filterOrphanedThinkingOnlyMessages).

    A streamed turn can arrive as several messages sharing one message id — the
    thinking in one, the text and tool calls in another — and the API serializer
    merges them again. So a thinking-only message is only orphaned when no other
    message with the SAME id carries non-thinking content. Dropping it anyway
    would throw away reasoning that still belongs to a turn.
    """
    ids_with_content = set()
    for message in messages:
        if _msg_attr(message, 'type') != 'assistant':
            continue
        blocks = _content_blocks(message)
        if blocks is None:
            continue
        if any(_block_field(b, 'type') not in _THINKING_TYPES for b in blocks):
            inner = _msg_attr(message, 'message')
            msg_id = inner.get('id') if isinstance(inner, dict) else getattr(inner, 'id', None)
            if msg_id:
                ids_with_content.add(msg_id)

    kept = []
    for message in messages:
        if _msg_attr(message, 'type') != 'assistant':
            kept.append(message)
            continue
        blocks = _content_blocks(message)
        if blocks is None or len(blocks) == 0:
            kept.append(message)
            continue
        if any(_block_field(b, 'type') not in _THINKING_TYPES for b in blocks):
            kept.append(message)
            continue
        inner = _msg_attr(message, 'message')
        msg_id = inner.get('id') if isinstance(inner, dict) else getattr(inner, 'id', None)
        if msg_id and msg_id in ids_with_content:
            kept.append(message)
    return kept


def get_text_from_message(message: Any) -> str:
    """Extract text content from a message"""
    msg_obj = message.message if hasattr(message, 'message') else message.get('message', {})
    content = msg_obj.content if hasattr(msg_obj, 'content') else msg_obj.get('content', [])

    texts = []
    for block in content:
        block_type = block.type if hasattr(block, 'type') else block.get('type')
        if block_type == 'text':
            block_text = block.text if hasattr(block, 'text') else block.get('text', '')
            texts.append(block_text)

    return '\n'.join(texts)


def is_interrupt_message(message: Any) -> bool:
    """Check if message is an interrupt message"""
    text = get_text_from_message(message)
    return INTERRUPT_MESSAGE in text
