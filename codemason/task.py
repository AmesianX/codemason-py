"""Task base types — adapted from src/Task.ts.

Defines the task type/status vocabulary, the shared TaskStateBase fields stored
in AppState.tasks, task-id generation (prefixed, crypto-random), and the Task
dispatch contract (`kill`). Concrete task states (e.g. LocalShellTaskState) live
under tasks/ and extend TaskStateBase.
"""

import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .utils.task.disk_output import get_task_output_path

# TaskType — codemason runs tasks in-process synchronously for agents, so the
# only task that actually lives in AppState.tasks is the background shell
# ('local_bash'). The remaining types are kept in the prefix table for id
# fidelity but no implementation is registered (see tasks/registry.py).
TaskType = str  # 'local_bash' | 'local_agent' | 'remote_agent' | ...

# TaskStatus lifecycle: pending -> running -> completed | failed | killed.
TaskStatus = str


def is_terminal_task_status(status: str) -> bool:
    """True when a task will not transition further (isTerminalTaskStatus)."""
    return status in ("completed", "failed", "killed")


def now_ms() -> int:
    """Millisecond epoch — the Python stand-in for the source's Date.now()."""
    return int(time.time() * 1000)


# Task ID prefixes (TASK_ID_PREFIXES). 'b' for local_bash is kept for backward
# compatibility with persisted/transcript task ids.
TASK_ID_PREFIXES: Dict[str, str] = {
    "local_bash": "b",
    "local_agent": "a",
    "remote_agent": "r",
    "in_process_teammate": "t",
    "local_workflow": "w",
    "monitor_mcp": "m",
    "dream": "d",
}

# Case-insensitive-safe alphabet (digits + lowercase) for task IDs.
# 36^8 ≈ 2.8 trillion combinations, sufficient to resist brute-force symlink
# attacks on the output directory.
TASK_ID_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _get_task_id_prefix(task_type: str) -> str:
    return TASK_ID_PREFIXES.get(task_type, "x")


def generate_task_id(task_type: str) -> str:
    """Generate a prefixed, crypto-random task id (generateTaskId).

    Uses secrets.choice (uniform over the alphabet) rather than urandom-byte %
    len: 256 % 36 != 0, so the modulo biased ids toward the first few symbols,
    shrinking the effective keyspace the docstring claims is brute-force-resistant."""
    prefix = _get_task_id_prefix(task_type)
    return prefix + "".join(secrets.choice(TASK_ID_ALPHABET) for _ in range(8))


@dataclass
class TaskStateBase:
    """Base fields shared by all task states (TaskStateBase).

    Every field carries a default so concrete subclasses can add their own
    fields without tripping dataclass ordering rules; `create_task_state_base`
    supplies the real values.
    """

    id: str = ""
    type: str = ""
    status: str = "pending"
    description: str = ""
    tool_use_id: Optional[str] = None
    start_time: int = 0
    end_time: Optional[int] = None
    total_paused_ms: Optional[int] = None
    output_file: str = ""
    output_offset: int = 0
    notified: bool = False


def create_task_state_base(
    task_id: str,
    task_type: str,
    description: str,
    tool_use_id: Optional[str] = None,
) -> dict:
    """Build the base-field dict for a new task (createTaskStateBase).

    Returned as a dict so callers can splat it into a concrete state's
    constructor and override fields (e.g. status -> 'running'), mirroring the
    source's `{ ...createTaskStateBase(...), status: 'running', ... }`.
    """
    return {
        "id": task_id,
        "type": task_type,
        "status": "pending",
        "description": description,
        "tool_use_id": tool_use_id,
        "start_time": now_ms(),
        "output_file": get_task_output_path(task_id),
        "output_offset": 0,
        "notified": False,
    }


class Task(ABC):
    """Task dispatch contract (Task type in Task.ts).

    `get_task_by_type` dispatches to a Task implementation for `kill`. (spawn /
    render were never called polymorphically in the source either.) Kill takes
    only set_app_state — every implementation drives the registry through it.
    """

    name: str = ""
    type: str = ""

    @abstractmethod
    async def kill(self, task_id: str, set_app_state: Callable) -> None:
        ...
