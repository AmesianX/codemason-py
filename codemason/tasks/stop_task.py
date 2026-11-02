"""Shared stop logic — adapted from src/tasks/stopTask.ts.

Look up a task by id, validate it is running, kill it via its task impl, and
(for shell tasks) suppress the kill notification. Used by TaskStop.
"""

from dataclasses import replace
from typing import Callable, Optional

from ..task import now_ms  # noqa: F401  (kept for parity; not needed directly)
from .local_shell_task import is_local_shell_task
from .registry import get_task_by_type


class StopTaskError(Exception):
    """Raised when a task cannot be stopped. `code` distinguishes the reason."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code  # 'not_found' | 'not_running' | 'unsupported_type'


async def stop_task(
    task_id: str,
    get_app_state: Callable,
    set_app_state: Callable,
) -> dict:
    """Kill a running task, returning {'taskId', 'taskType', 'command'} (stopTask).

    Throws StopTaskError (with `.code`) when the task is not found, not running,
    or of an unsupported type.
    """
    app_state = get_app_state()
    task = app_state.tasks.get(task_id)

    if task is None:
        raise StopTaskError(f"No task found with ID: {task_id}", "not_found")

    if task.status != "running":
        raise StopTaskError(
            f"Task {task_id} is not running (status: {task.status})", "not_running"
        )

    task_impl = get_task_by_type(task.type)
    if task_impl is None:
        raise StopTaskError(f"Unsupported task type: {task.type}", "unsupported_type")

    await task_impl.kill(task_id, set_app_state)

    # Bash: suppress the "exit code 137" notification (noise). kill_task already
    # set notified=True, so the late completion notice is dropped — this block
    # mirrors stopTask's explicit suppression. (The original additionally emits
    # an SDK task_terminated event here; codemason has no SDK event queue —
    # substrate — so that emit is dropped.)
    if is_local_shell_task(task):
        def _suppress(prev):
            prev_task = prev.tasks.get(task_id)
            if prev_task is None or prev_task.notified:
                return prev
            return replace(
                prev,
                tasks={**prev.tasks, task_id: replace(prev_task, notified=True)},
            )

        set_app_state(_suppress)

    command: Optional[str] = task.command if is_local_shell_task(task) else task.description
    return {"taskId": task_id, "taskType": task.type, "command": command}
