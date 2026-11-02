"""SendMessage tool — based on src/tools/SendMessageTool/ from Claude Code.

Talk to a sub-agent that is already running. Its plain output is not visible to
another agent and the parent's is not visible to it, so a message between them
has to be an explicit tool call.

Scope. The source's SendMessage is mostly a swarm tool: teammates by name,
`"*"` broadcast, shutdown and plan-approval protocol replies, and cross-session
delivery over a unix socket or the Remote Control bridge. codemason has none of
those subsystems — no teammates, no swarm, no inbox sockets — so none of that is
reproduced. What IS ported is the branch that routes to a local sub-agent, which
is the one the Agent tool's own description points at.

That is also why `is_enabled` is unconditional here while the source gates the
whole tool behind its swarm feature: the gate exists because teammates live
there, and the sub-agent route does not need it.

Delivery is a queue, not a handoff. The agent is mid-turn and the list of
messages it is working from belongs to the loop running it, so the message waits
on the agent's own task until that loop reaches its next tool-round boundary
(the drain in query.py). Nothing else can read it: the queue is per-task, so the
only reader is the agent the message was addressed to.
"""

import json
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from pydantic import Field

from ..state.app_state import get_app_state as _global_get_app_state
from ..state.app_state import set_app_state as _global_set_app_state
from ..tasks.local_agent_task import (
    is_local_agent_task,
    is_panel_agent_task,
    queue_pending_message,
)
from ..utils.agent_id import is_agent_id
from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultFailure,
    ValidationResultSuccess,
)

SEND_MESSAGE_TOOL_NAME = "SendMessage"

DESCRIPTION = """
# SendMessage

Send a message to a running sub-agent.

```json
{"to": "a1b2c3d4e5f60718", "summary": "narrow the search", "message": "skip the vendor directory"}
```

`to` is the name you gave the agent when you launched it, or its ID as
returned by the launch when you did not name it.

An agent that was stopped is resumed from its own transcript and told your
message, so you do not have to re-describe a task it had already started.

Your plain text output is NOT visible to the agent — to tell it something, you
MUST call this tool. The message is delivered at the agent's next tool round;
it is not an interrupt, and the agent keeps working in the meantime.
"""


def _ctx_state(context) -> tuple:
    """Resolve (get_app_state, set_app_state) from the tool context, falling
    back to the module-global store singleton."""
    get = getattr(context, "get_app_state", None) or _global_get_app_state
    setf = getattr(context, "set_app_state", None) or _global_set_app_state
    return get, setf


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class SendMessageInput(ToolInputSchema):
    to: str = Field(
        ...,
        description="Recipient: the name or ID of a running sub-agent",
    )
    summary: Optional[str] = Field(
        default=None,
        description="A 5-10 word summary shown as a preview in the UI",
    )
    message: str = Field(..., description="Plain text message content")


class SendMessageTool(Tool):
    name = SEND_MESSAGE_TOOL_NAME
    input_schema = SendMessageInput

    @classmethod
    async def description(cls, **kwargs) -> str:
        return "Send a message to another agent"

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    def is_read_only(cls) -> bool:
        # SendMessageTool.isReadOnly(input) is `typeof input.message === 'string'`
        # — true for a plain message, false for the structured protocol
        # messages (shutdown / plan approval) that belong to the swarm. Only
        # plain messages exist here, so it is simply true.
        return True

    @classmethod
    def is_concurrency_safe(cls, input_data=None) -> bool:
        return cls.is_read_only()

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def needs_permissions(cls, input_data: Optional[Dict[str, Any]] = None) -> bool:
        return False

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return "SendMessage"

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        to = input_data.get("to", "")
        preview = input_data.get("summary") or _truncate(input_data.get("message", ""), 50)
        return f"{preview} → {to}" if preview else str(to)

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])
        return None

    @classmethod
    async def check_permissions(cls, input_data, context=None):
        from .base import PermissionResult
        return PermissionResult(behavior="allow")

    @classmethod
    async def validate_input(cls, input_data, context) -> ValidationResult:
        to = (input_data.get("to") or "").strip()
        if not to:
            return ValidationResultFailure(
                message="Missing required parameter: to", meta={"errorCode": 1}
            )
        if not input_data.get("message"):
            return ValidationResultFailure(
                message="Missing required parameter: message", meta={"errorCode": 1}
            )
        get_app_state, _ = _ctx_state(context)
        if to not in get_app_state().agent_name_registry and is_agent_id(to) is None:
            return ValidationResultFailure(
                message=(
                    f'"{to}" is neither a name you gave an agent nor an agent '
                    "ID. Pass the name you spawned it under, or the ID the "
                    "Agent tool returned."
                ),
                meta={"errorCode": 1},
            )
        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        to = (input_data.get("to") or "").strip()
        message = input_data.get("message") or ""
        get_app_state, set_app_state = _ctx_state(context)

        # A name the spawn registered wins over reading `to` as a raw id, which
        # is the source's `registered ?? toAgentId(input.to)`. The model named
        # the agent so it would not have to carry a 16-hex id around; making it
        # pass the id anyway would waste the name it was offered.
        state = get_app_state()
        agent_id = state.agent_name_registry.get(to) or is_agent_id(to)
        task = state.tasks.get(agent_id) if agent_id else None

        if is_local_agent_task(task) and is_panel_agent_task(task):
            if task.status == "running":
                queue_pending_message(agent_id, message, set_app_state)
                data = {
                    "success": True,
                    "message": (
                        f"Message queued for delivery to {to} at its next tool round."
                    ),
                }
            else:
                # Stopped, but not gone: its conversation is on disk and its
                # task record says what kind of agent it was, so the message can
                # be the next thing it hears (resumeAgentBackground). In the
                # background, because this call sits inside someone else's turn
                # and must not block it — the agent announces its own outcome
                # through the notification queue.
                from .resume_agent import resume_agent_background

                resumed = await resume_agent_background(
                    agent_id=agent_id,
                    prompt=message,
                    tool_use_context=context,
                    can_use_tool=can_use_tool,
                    get_app_state_fn=get_app_state,
                    set_app_state_fn=set_app_state,
                )
                data = {
                    "success": bool(resumed.get("resumed")),
                    "message": resumed.get("message", ""),
                }
        else:
            data = {
                "success": False,
                "message": (
                    f'No running agent called "{to}". '
                    "Check the name you spawned it under or the ID the Agent "
                    "tool returned, or launch a new agent."
                ),
            }

        yield ToolResultSuccess(
            type="result",
            result_for_assistant=json.dumps(data),
            data=data,
        )
