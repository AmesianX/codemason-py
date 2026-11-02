"""TaskState union — adapted from src/tasks/types.ts.

The source union spans local_bash / local_agent / remote_agent / teammate /
workflow / monitor / dream. codemason registers the two that actually run
locally: the background shell and the sub-agent. The remaining types have no
implementation here (no remote execution, no workflows, no teammates), so they
are absent from the union rather than stubbed.
"""

from typing import Union

from .local_agent_task import LocalAgentTaskState
from .local_shell_task import LocalShellTaskState

TaskState = Union[LocalShellTaskState, LocalAgentTaskState]
