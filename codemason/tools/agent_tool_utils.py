"""Agent tool helpers — adapted from src/tools/AgentTool/agentToolUtils.ts.

Three things the foreground and background paths both need: pulling the agent's
answer out of its messages, dropping a tool call that never came back before the
conversation is sent again, and (for the background one) driving the whole run
from spawn to the notification that carries the result back.
"""

import asyncio
import time
from typing import Any, Awaitable, Callable, List, Optional

from ..state.app_state import set_app_state as _global_set_app_state
from ..tasks.local_agent_task import (
    append_agent_transcript,
    complete_agent_task,
    derive_agent_summary,
    create_activity_description_resolver,
    create_progress_tracker,
    enqueue_agent_notification,
    fail_agent_task,
    get_progress_update,
    get_token_count_from_tracker,
    is_local_agent_task,
    kill_async_agent,
    update_agent_progress,
    update_agent_summary,
    update_progress_from_message,
)
from ..utils.generators import abortable
from ..utils.log import debug_log


def _text_of(message: Any) -> str:
    """The text blocks of one assistant message, joined (extractTextContent)."""
    msg_type = getattr(message, "type", None) or (
        message.get("type") if isinstance(message, dict) else None
    )
    if msg_type != "assistant":
        return ""
    inner = getattr(message, "message", None)
    if inner is None and isinstance(message, dict):
        inner = message.get("message", {})
    content = getattr(inner, "content", None)
    if content is None and isinstance(inner, dict):
        content = inner.get("content", [])
    texts = []
    for block in content or []:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text", "")
        texts.append(text or "")
    return "\n".join(texts).strip()


def extract_partial_result(messages: List[Any]) -> Optional[str]:
    """The agent's answer: the last assistant message that HAS text
    (extractPartialResult).

    Walks BACKWARD, and that is the point — the loop can end on an assistant
    turn that is nothing but tool_use blocks, and taking the last message
    outright would report an empty answer for a run that produced one. Also
    what makes a killed agent still able to hand back whatever it had reached.
    """
    for message in reversed(messages):
        text = _text_of(message)
        if text:
            return text
    return None


def filter_incomplete_tool_calls(messages: List[Any]) -> List[Any]:
    """Drop assistant messages holding a tool_use with no tool_result
    (filterIncompleteToolCalls).

    An orphaned tool call is an API error, not a wasted token: a conversation
    that ends on an assistant turn whose tool never returned is rejected
    outright. That is the exact shape a foreground agent is in at the instant it
    is backgrounded — the race can fire while a tool is mid-flight — so the
    resumed run has to be handed the conversation without it.

    Collects every tool_use_id that HAS a result first, then keeps an assistant
    message only if none of its tool_use blocks is missing from that set. The
    whole message goes, text blocks and all; a partial assistant turn is not a
    thing the wire format can express.
    """
    satisfied = set()
    for message in messages:
        msg_type = getattr(message, "type", None) or (
            message.get("type") if isinstance(message, dict) else None
        )
        if msg_type != "user":
            continue
        inner = getattr(message, "message", None)
        if inner is None and isinstance(message, dict):
            inner = message.get("message", {})
        content = getattr(inner, "content", None)
        if content is None and isinstance(inner, dict):
            content = inner.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "tool_result":
                continue
            use_id = getattr(block, "tool_use_id", None)
            if use_id is None and isinstance(block, dict):
                use_id = block.get("tool_use_id")
            if use_id:
                satisfied.add(use_id)

    kept: List[Any] = []
    for message in messages:
        msg_type = getattr(message, "type", None) or (
            message.get("type") if isinstance(message, dict) else None
        )
        if msg_type != "assistant":
            kept.append(message)
            continue
        inner = getattr(message, "message", None)
        if inner is None and isinstance(message, dict):
            inner = message.get("message", {})
        content = getattr(inner, "content", None)
        if content is None and isinstance(inner, dict):
            content = inner.get("content", [])
        if not isinstance(content, list):
            kept.append(message)
            continue
        incomplete = False
        for block in content:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "tool_use":
                continue
            use_id = getattr(block, "id", None)
            if use_id is None and isinstance(block, dict):
                use_id = block.get("id")
            if use_id and use_id not in satisfied:
                incomplete = True
                break
        if not incomplete:
            kept.append(message)
    return kept


async def run_async_agent_lifecycle(
    agent_id: str,
    description: str,
    make_stream: Callable[[], Any],
    tool_use_context: Any,
    set_app_state_fn: Callable = _global_set_app_state,
    tools: Any = None,
    tool_use_id: Optional[str] = None,
    start_time: Optional[float] = None,
    on_finished: Optional[Callable[[], None]] = None,
    acquire_slot: Optional[Callable[[], Awaitable[Any]]] = None,
    initial_messages: Optional[List[Any]] = None,
    slot: Any = None,
) -> None:
    """Drive a background agent from spawn to its terminal notification
    (runAsyncAgentLifecycle).

    Detached: the tool call that launched it has already returned, so nothing is
    awaiting this. That is exactly why every exit has to announce itself — the
    notification queue is the only route left back into the conversation, and an
    agent that finishes without enqueueing one has done its work in silence.

    The status transition comes BEFORE the notification on purpose: a
    TaskOutput(block=True) waiting on this agent unblocks on the status, so
    anything slower between the two would hold that caller for no reason.

    `acquire_slot` is codemason's own: the concurrency gate is ours, not the
    source's, and a background launch must not block the parent while it waits
    its turn. So the slot is taken HERE, in the background, rather than in the
    tool call — "launched" means registered and queued, which the task's own
    status reports honestly.

    Two parameters serve the hand-over from a foreground agent that got
    backgrounded mid-run:

      * `initial_messages` is the conversation it had reached. The list is used
        as-is, not copied, exactly as the source keeps pushing into the same
        `agentMessages` array — so the final answer and the notification account
        for the whole run, not just the part after the switch. It also seeds the
        progress tracker, so the counts the parent reads do not restart at zero.
      * `slot` is a gate slot the caller is already holding. A backgrounded
        agent is still running and still occupying the KV pool, so the slot
        moves with it instead of being released and re-acquired — releasing it
        would let a third agent in while this one is still working, and
        re-acquiring could deadlock behind the very agent that handed it over.
    """
    started = start_time if start_time is not None else time.time()
    agent_messages: List[Any] = initial_messages if initial_messages is not None else []
    try:
        if slot is None and acquire_slot is not None:
            slot = await acquire_slot()
            if slot is None:
                # Aborted while queued. Still a terminal transition, still
                # announced — the parent is waiting for word either way.
                raise asyncio.CancelledError()

        tracker = create_progress_tracker()
        resolve_activity = create_activity_description_resolver(tools)
        # Fold what already happened into the tally before the stream opens, so
        # a handed-over agent's token and tool counts carry on from where the
        # foreground left them rather than appearing to rewind.
        for existing in list(agent_messages):
            update_progress_from_message(tracker, existing, resolve_activity, tools)

        # Wrapped in the one abort-aware iterator the rest of the codebase uses
        # (utils.generators.abortable) rather than racing the stop here a second
        # time. Nothing is awaiting this run, so a stop that did not reach it
        # would leave an agent working forever with no one to report to.
        #
        # `abortable` STOPS on the signal, it does not raise — so the check after
        # the loop is what tells a finished run from an interrupted one.
        abort_signal = getattr(tool_use_context, 'abort_controller', None)
        async for message in abortable(make_stream(), abort_signal):
            agent_messages.append(message)
            # Append to the viewed transcript only while the UI holds the task.
            # Buffering for an agent nobody is watching is how a long background
            # run turns into a memory leak.
            def _append(task, message=message):
                if not is_local_agent_task(task) or not task.retain:
                    return task
                from dataclasses import replace as _replace
                return _replace(task, messages=[*(task.messages or []), message])

            from ..utils.task.framework import update_task_state
            update_task_state(agent_id, set_app_state_fn, _append)

            # The agent's own output file, so TaskOutput on a RUNNING agent
            # returns what it is doing rather than an empty string.
            append_agent_transcript(agent_id, message)

            update_progress_from_message(tracker, message, resolve_activity, tools)
            update_agent_progress(agent_id, get_progress_update(tracker), set_app_state_fn)
            # The prose line, after the counts: update_agent_progress preserves
            # an existing summary rather than overwriting it, so the fresh one
            # has to be written through the path that owns it.
            summary = derive_agent_summary(tracker)
            if summary:
                update_agent_summary(agent_id, summary, set_app_state_fn)

        if abort_signal is not None and abort_signal.is_set():
            # Interrupted, not finished. Raised so the one handler that holds
            # the messages settles it and announces the partial result.
            raise asyncio.CancelledError()

        progress = get_progress_update(tracker)
        final_message = extract_partial_result(agent_messages)
        elapsed_ms = int((time.time() - started) * 1000)

        complete_agent_task(
            agent_id,
            {
                "agent_id": agent_id,
                "text": final_message or "",
                "tool_use_count": progress.tool_use_count,
                "duration": time.time() - started,
                "usage": {
                    "total_tokens": get_token_count_from_tracker(tracker),
                    "tool_uses": progress.tool_use_count,
                    "duration_ms": elapsed_ms,
                },
            },
            set_app_state_fn,
        )
        enqueue_agent_notification(
            agent_id,
            description,
            "completed",
            set_app_state_fn,
            final_message=final_message,
            usage={
                "total_tokens": get_token_count_from_tracker(tracker),
                "tool_uses": progress.tool_use_count,
                "duration_ms": elapsed_ms,
            },
            tool_use_id=tool_use_id,
        )

    except asyncio.CancelledError:
        # Stopped, not broken. kill_async_agent is a no-op when TaskStop
        # already marked it killed — but only this handler holds the messages,
        # so the notification fires regardless, carrying whatever the agent had
        # reached. Dropping it would throw away work the user asked to stop, not
        # work that failed.
        kill_async_agent(agent_id, set_app_state_fn)
        enqueue_agent_notification(
            agent_id,
            description,
            "killed",
            set_app_state_fn,
            final_message=extract_partial_result(agent_messages),
            tool_use_id=tool_use_id,
        )
    except Exception as exc:  # noqa: BLE001 — a detached task must not die silently
        message = str(exc) or exc.__class__.__name__
        debug_log("ASYNC_AGENT_FAILED", {"agent_id": agent_id, "error": message})
        fail_agent_task(agent_id, message, set_app_state_fn)
        enqueue_agent_notification(
            agent_id,
            description,
            "failed",
            set_app_state_fn,
            error=message,
            final_message=extract_partial_result(agent_messages),
            tool_use_id=tool_use_id,
        )
    finally:
        if slot is not None:
            slot.release()
        if on_finished is not None:
            try:
                on_finished()
            except Exception:
                pass
