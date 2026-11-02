"""Pending-notification queue — codemason substrate for Claude Code's
messageQueueManager (enqueuePendingNotification / drain).

The source enqueues task <task-notification> blocks onto a queue that the Ink
query loop drains and injects into the conversation. codemason has no Ink loop;
this module is the faithful equivalent: a process-global FIFO of ready-to-inject
notification strings. LocalShellTask appends completion notices here on finish,
and the REPL drains them between turns (cli.py) — pushing each as a user-role
<task-notification> message, exactly as the original surfaces them.
"""

import threading
from typing import List

# (value, mode) entries. `mode` mirrors enqueuePendingNotification's mode field
# ('task-notification'); kept for fidelity even though codemason injects all
# modes the same way.
_pending: List[dict] = []
_lock = threading.Lock()


def enqueue_pending_notification(value: str, mode: str = "task-notification") -> None:
    """Append a ready-to-inject notification (enqueuePendingNotification)."""
    with _lock:
        _pending.append({"value": value, "mode": mode})


def drain_pending_notifications() -> List[str]:
    """Return and clear all queued notification strings, in FIFO order."""
    with _lock:
        drained = [entry["value"] for entry in _pending]
        _pending.clear()
    return drained


def has_pending_notifications() -> bool:
    """True if any notification is queued (peek, without draining).

    Used by the headless wait loop to close the drain-vs-check race: a task that
    finishes between a drain and the running-task check still leaves a queued
    notification, so the loop must keep going until the queue is empty too."""
    with _lock:
        return len(_pending) > 0


def reset_pending_notifications() -> None:
    """Clear the queue (tests / `clear`)."""
    with _lock:
        _pending.clear()
