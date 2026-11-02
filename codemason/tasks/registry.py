"""Task registry — adapted from src/tasks.ts (getAllTasks / getTaskByType).

The source registers six task types (shell, local/remote agent, dream, workflow,
monitor). codemason registers the two with a local implementation: the
background shell and the sub-agent. The dispatch contract (getTaskByType → impl
with .kill) is what lets stopTask stop either one without knowing which it has.
"""

from typing import List, Optional

from ..task import Task
from .local_agent_task import LOCAL_AGENT_TASK
from .local_shell_task import LOCAL_SHELL_TASK


def get_all_tasks() -> List[Task]:
    """All registered task implementations (getAllTasks)."""
    return [LOCAL_SHELL_TASK, LOCAL_AGENT_TASK]


def get_task_by_type(task_type: str) -> Optional[Task]:
    """Look up a task implementation by its type (getTaskByType)."""
    return next((t for t in get_all_tasks() if t.type == task_type), None)
