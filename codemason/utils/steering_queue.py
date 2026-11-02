"""Mid-turn user steering queue — codemason substrate for interjecting a line
into a turn that is ALREADY running.

The REPL used to park a line typed mid-turn and process it as a brand-new turn
only after the current one finished (queued-messages UX). This queue is the
faithful equivalent of Claude Code's mid-turn steering: a line the user types
while the model is working is injected into that SAME turn at its next model
step, so the model sees it immediately — no turn abort required.

It deliberately mirrors message_queue_manager (the background-notification
substrate): a process-global FIFO of ready-to-inject strings. The interactive
REPL's Enter handler enqueues here while a turn is running; query()'s mid-turn
gate drains it at the exact point it drains task <task-notification> blocks and
injects each as a user-role message. Anything still queued when a turn ends
before it could inject (e.g. the final model response made no tool call, so
there was no next step) is reconciled by the REPL's turn-boundary handler.
"""

import threading
from typing import List

_pending: List[str] = []
_lock = threading.Lock()


def enqueue_steering_message(text: str) -> None:
    """Append a mid-turn steering line to inject into the running turn."""
    text = (text or '').strip()
    if not text:
        return
    with _lock:
        _pending.append(text)


def drain_steering_messages() -> List[str]:
    """Return and clear all queued steering lines, in FIFO order."""
    with _lock:
        drained = list(_pending)
        _pending.clear()
    return drained


def has_steering_messages() -> bool:
    """True if any steering line is queued (peek, without draining)."""
    with _lock:
        return len(_pending) > 0


def reset_steering_messages() -> None:
    """Clear the queue (Ctrl+C hard-stop / tests)."""
    with _lock:
        _pending.clear()
