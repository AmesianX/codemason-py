"""Task framework helpers — adapted from src/utils/task/framework.ts.

The AppState.tasks update primitives every task mechanism builds on:
update_task_state / register_task / evict_terminal_task / get_running_tasks.

The poll pipeline (generate_task_attachments / apply_task_offsets_and_evictions
/ poll_tasks) is ported too, including the part that surprises: it produces no
attachments. The source builds the list and never pushes to it, and its own
comment says why — each task type enqueues its own completion notification, so
generating one here would race those callbacks and deliver the same completion
twice. What the pass actually does is keep the books: advance each running
task's output offset past what has been read, and evict terminal tasks whose
notification has been consumed and whose grace period has passed.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from ...state.app_state import AppState
from ...task import is_terminal_task_status, now_ms
from .disk_output import get_task_output_delta

# Standard polling interval for all tasks (POLL_INTERVAL_MS).
POLL_INTERVAL_MS = 1000

# How long a killed task stays visible before eviction (STOPPED_DISPLAY_MS).
STOPPED_DISPLAY_MS = 3_000

# Grace period for a terminal local_agent task (PANEL_GRACE_MS). The source
# names it for the coordinator panel that holds the task on screen; here it is
# the window in which a finished agent's record — status, result, counts — can
# still be read back before eviction.
PANEL_GRACE_MS = 30_000

if TYPE_CHECKING:
    from ...tasks.types import TaskState


def update_task_state(task_id: str, set_app_state: Callable,
                      updater: Callable) -> None:
    """Update a task's state in AppState (updateTaskState).

    `updater(task)` returns the next task. If the task is absent, or the updater
    returns the same object (an early-return no-op), the store is left untouched
    — matching the original's reference-equality skip.
    """
    def _apply(prev: AppState) -> AppState:
        task = prev.tasks.get(task_id)
        if task is None:
            return prev
        updated = updater(task)
        if updated is task:
            return prev
        return replace(prev, tasks={**prev.tasks, task_id: updated})

    set_app_state(_apply)


def register_task(task: "TaskState", set_app_state: Callable) -> None:
    """Register a new task in AppState (registerTask).

    Re-registering an id that is already present is a REPLACEMENT, not a new
    start — resuming an agent builds a fresh task state and registers it over
    the old one. The fields the UI and the message history own are carried
    forward, because the replacement does not know them: `retain` is the user's
    hold, `start_time` keeps the ordering stable, `messages` and `disk_loaded`
    are the transcript as it stands (the just-appended prompt lives in
    `messages` and is not on disk yet), and `pending_messages` is a queue that
    may already hold something for the agent to read. Overwriting them would
    drop a queued message on the floor and reset a hold the user placed.

    The original also emits a `task_started` event here, and skips it on a
    replacement so a resume does not look like a second spawn. codemason routes
    its own notifications through utils/message_queue_manager rather than an SDK
    queue, and nothing subscribes to task starts yet, so there is no emit — but
    the replacement is still distinguished, because the carry-forward depends on
    it.
    """
    def _apply(prev: AppState) -> AppState:
        existing = prev.tasks.get(task.id)
        merged = task
        if existing is not None and hasattr(existing, "retain"):
            carried = {
                "retain": existing.retain,
                "start_time": existing.start_time,
                "pending_messages": existing.pending_messages,
            }
            for field_name in ("messages", "disk_loaded"):
                if hasattr(existing, field_name) and hasattr(task, field_name):
                    carried[field_name] = getattr(existing, field_name)
            merged = replace(task, **carried)
        return replace(prev, tasks={**prev.tasks, task.id: merged})

    set_app_state(_apply)


def evict_terminal_task(task_id: str, set_app_state: Callable) -> None:
    """Eagerly evict a terminal, notified task from AppState (evictTerminalTask).

    Three guards, all from the source:
      * not terminal          -> keep (still running)
      * not notified          -> keep (an unconsumed completion is never dropped)
      * eviction deadline not reached -> keep

    The third guard applies to tasks that carry an `evict_after` deadline, which
    in the source is exactly the agent tasks (it tests `"retain" in task`). A
    terminal agent task gets `evict_after = now + PANEL_GRACE_MS`, so its record
    — status, result, token and tool counts — outlives the run by that grace
    period and the parent can still read it back. An agent task with NO deadline
    is a retained one, and a retained task is never evicted at all; that is why
    an unset deadline means keep, not evict (the source's `evictAfter ?? Infinity`).
    """
    def _apply(prev: AppState) -> AppState:
        task = prev.tasks.get(task_id)
        if task is None:
            return prev
        if not is_terminal_task_status(task.status):
            return prev
        if not task.notified:
            return prev
        if hasattr(task, "retain"):
            deadline = getattr(task, "evict_after", None)
            if deadline is None or deadline > now_ms():
                return prev
        remaining = {k: v for k, v in prev.tasks.items() if k != task_id}
        return replace(prev, tasks=remaining)

    set_app_state(_apply)


def get_running_tasks(state: AppState) -> List["TaskState"]:
    """All running tasks (getRunningTasks)."""
    return [t for t in state.tasks.values() if t.status == "running"]


@dataclass
class TaskAttachment:
    """A task status update, as the attachment pipeline would emit it
    (TaskAttachment).

    Built by nothing today — see generate_task_attachments — but it is the shape
    `render_task_status_attachment` consumes, and the compaction path builds the
    same dict by hand.
    """

    task_id: str
    task_type: str
    status: str
    description: str
    delta_summary: Optional[str] = None
    tool_use_id: Optional[str] = None


def generate_task_attachments(state: AppState) -> dict:
    """Walk the task map, returning what to patch and what to drop
    (generateTaskAttachments).

    Returns {'attachments', 'updated_task_offsets', 'evicted_task_ids'}.

    `attachments` is always empty, exactly as in the source, and the source's
    own comment is the reason: "Completed tasks are NOT notified here — each
    task type handles its own completion notification via
    enqueuePendingNotification(). Generating attachments here would race with
    those per-type callbacks, causing dual delivery (one inline attachment + one
    separate API turn)." The list is kept in the return shape because
    `poll_tasks` iterates it and a future task type may want it.

    Only the offsets are returned, never the tasks: a task can transition to
    completed during the disk read below, and writing back a whole stale
    snapshot would clobber that transition — a zombie task, running forever in
    a map that has already forgotten it finished.
    """
    attachments: List[TaskAttachment] = []
    updated_task_offsets: Dict[str, int] = {}
    evicted_task_ids: List[str] = []

    for task in list((state.tasks or {}).values()):
        if task.notified:
            if is_terminal_task_status(task.status):
                # Consumed — it can go.
                evicted_task_ids.append(task.id)
                continue
            if task.status == "pending":
                # Keep it: it has not run yet, but the parent already knows.
                continue

        if task.status == "running":
            delta = get_task_output_delta(task.id, task.output_offset)
            if delta.get("content"):
                updated_task_offsets[task.id] = delta["newOffset"]

    return {
        "attachments": attachments,
        "updated_task_offsets": updated_task_offsets,
        "evicted_task_ids": evicted_task_ids,
    }


def apply_task_offsets_and_evictions(
    set_app_state: Callable,
    updated_task_offsets: Dict[str, int],
    evicted_task_ids: List[str],
) -> None:
    """Apply what generate_task_attachments found
    (applyTaskOffsetsAndEvictions).

    Every condition is re-checked against the state as it is NOW, not as it was
    when the walk started: the disk read in between is a window in which a task
    can finish, or a resume can replace it. An offset for a task that is no
    longer running is moot, and an eviction decided before a replacement landed
    would delete the replacement.
    """
    if not updated_task_offsets and not evicted_task_ids:
        return

    def _apply(prev: AppState) -> AppState:
        tasks = dict(prev.tasks)
        changed = False

        for task_id, offset in updated_task_offsets.items():
            fresh = tasks.get(task_id)
            if fresh is not None and fresh.status == "running":
                tasks[task_id] = replace(fresh, output_offset=offset)
                changed = True

        for task_id in evicted_task_ids:
            fresh = tasks.get(task_id)
            if fresh is None:
                continue
            if not is_terminal_task_status(fresh.status) or not fresh.notified:
                continue
            if hasattr(fresh, "retain"):
                deadline = getattr(fresh, "evict_after", None)
                if deadline is None or deadline > now_ms():
                    continue
            del tasks[task_id]
            changed = True

        return replace(prev, tasks=tasks) if changed else prev

    set_app_state(_apply)


def get_status_text(status: str) -> str:
    """Human-readable status for a notification (getStatusText)."""
    return {
        "completed": "completed successfully",
        "failed": "failed",
        "killed": "was stopped",
        "running": "is running",
        "pending": "is pending",
    }.get(status, status)


def enqueue_task_notification(attachment: TaskAttachment) -> None:
    """Enqueue a <task-notification> for a task status update
    (enqueueTaskNotification).

    Reachable only through `poll_tasks`, which iterates a list that is always
    empty — see generate_task_attachments. Ported so the pipeline is whole, and
    so a task type that does want inline delivery has the writer already.
    """
    from ..message_queue_manager import enqueue_pending_notification
    from ..task.disk_output import get_task_output_path

    tool_use_id_line = (
        f"\n<tool-use-id>{attachment.tool_use_id}</tool-use-id>"
        if attachment.tool_use_id
        else ""
    )
    message = (
        f"<task-notification>\n"
        f"<task-id>{attachment.task_id}</task-id>{tool_use_id_line}\n"
        f"<task-type>{attachment.task_type}</task-type>\n"
        f"<output-file>{get_task_output_path(attachment.task_id)}</output-file>\n"
        f"<status>{attachment.status}</status>\n"
        f'<summary>Task "{attachment.description}" '
        f"{get_status_text(attachment.status)}</summary>\n"
        f"</task-notification>"
    )
    enqueue_pending_notification(message, mode="task-notification")


def poll_tasks(get_app_state: Callable, set_app_state: Callable) -> None:
    """One bookkeeping pass over every task (pollTasks).

    Advances the running tasks' output offsets and drops the terminal ones that
    have been consumed. Not a notification loop — see
    generate_task_attachments for why the attachment list it walks is empty.
    """
    found = generate_task_attachments(get_app_state())
    apply_task_offsets_and_evictions(
        set_app_state, found["updated_task_offsets"], found["evicted_task_ids"]
    )
    for attachment in found["attachments"]:
        enqueue_task_notification(attachment)
