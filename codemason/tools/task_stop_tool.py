"""TaskStop tool — based on src/tools/TaskStopTool/ from Claude Code.

Stops a running background task by id. `shell_id` is accepted as a deprecated
alias for `task_id` (KillShell backward compatibility). Delegates to stop_task,
which kills the task via its task impl and suppresses the kill notification.
"""

import json
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from pydantic import Field

from ..state.app_state import get_app_state as _global_get_app_state
from ..state.app_state import set_app_state as _global_set_app_state
from ..tasks.stop_task import stop_task
from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultFailure,
    ValidationResultSuccess,
)

TASK_STOP_TOOL_NAME = "TaskStop"

DESCRIPTION = """
- Stops a running background task by its ID
- Takes a task_id parameter identifying the task to stop
- Returns a success or failure status
- Use this tool when you need to terminate a long-running task
"""


class TaskStopInput(ToolInputSchema):
    task_id: Optional[str] = Field(
        default=None, description="The ID of the background task to stop"
    )
    # shell_id is accepted for backward compatibility with the deprecated
    # KillShell tool.
    shell_id: Optional[str] = Field(
        default=None, description="Deprecated: use task_id instead"
    )


def _ctx_state(context) -> tuple:
    """Resolve (get_app_state, set_app_state) from the tool context, falling
    back to the module-global store singleton (the source threads these through
    toolUseContext; codemason's store is a singleton, so either works)."""
    get = getattr(context, "get_app_state", None) or _global_get_app_state
    setf = getattr(context, "set_app_state", None) or _global_set_app_state
    return get, setf


class TaskStopTool(Tool):
    name = TASK_STOP_TOOL_NAME
    input_schema = TaskStopInput
    # KillShell is the deprecated name — kept as an alias for backward
    # compatibility with existing transcripts and SDK users.
    aliases = ["KillShell"]

    @classmethod
    def is_concurrency_safe(cls, input_data=None) -> bool:
        # Stopping a task is safe to run alongside other concurrency-safe calls
        # even though it isn't read-only (TaskStopTool.isConcurrencySafe → true).
        return True

    @classmethod
    async def description(cls, **kwargs) -> str:
        return "Stop a running background task by ID"

    @classmethod
    async def prompt(cls) -> str:
        return DESCRIPTION

    @classmethod
    def is_read_only(cls) -> bool:
        return False

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return "Stop Task"

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        # TaskStopTool/UI.tsx renderToolUseMessage returns '': the stopped task is
        # named in the result, so the call itself shows nothing.
        return ''

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """The human line, not the JSON the model gets (jsonStringify in call())."""
        if not isinstance(data, dict):
            return None
        message = data.get('message')
        if message:
            return str(message)
        task_id = data.get('task_id')
        return f"Stopped task: {task_id}" if task_id else None

    @classmethod
    async def check_permissions(cls, input_data, context=None):
        # The original's default checkPermissions auto-allows (no classifier
        # gate for stopping a task the model itself backgrounded). Mirror that:
        # stopping a self-owned background task needs no user prompt.
        from .base import PermissionResult
        return PermissionResult(behavior="allow")

    @classmethod
    async def validate_input(cls, input_data, context) -> ValidationResult:
        # Support both task_id and shell_id (deprecated KillShell compat).
        task_id = input_data.get("task_id")
        shell_id = input_data.get("shell_id")
        id_ = task_id or shell_id
        if not id_:
            return ValidationResultFailure(
                message="Missing required parameter: task_id", meta={"errorCode": 1}
            )

        get_app_state, _ = _ctx_state(context)
        task = get_app_state().tasks.get(id_)
        if task is None:
            return ValidationResultFailure(
                message=f"No task found with ID: {id_}", meta={"errorCode": 1}
            )
        if task.status != "running":
            return ValidationResultFailure(
                message=f"Task {id_} is not running (status: {task.status})",
                meta={"errorCode": 3},
            )
        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        task_id = input_data.get("task_id")
        shell_id = input_data.get("shell_id")
        id_ = task_id or shell_id
        if not id_:
            raise ValueError("Missing required parameter: task_id")

        get_app_state, set_app_state = _ctx_state(context)
        result = await stop_task(id_, get_app_state, set_app_state)

        data = {
            "message": f"Successfully stopped task: {result['taskId']} ({result['command']})",
            "task_id": result["taskId"],
            "task_type": result["taskType"],
            "command": result["command"],
        }
        # mapToolResultToToolResultBlockParam: jsonStringify(output).
        yield ToolResultSuccess(result_for_assistant=json.dumps(data), data=data)
