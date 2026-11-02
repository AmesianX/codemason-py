"""Resume a stopped sub-agent — adapted from src/tools/AgentTool/resumeAgent.ts.

An agent that was killed, or that failed, still has everything needed to carry
on: its conversation is on disk (see tasks/local_agent_task.append_agent_
transcript) and its task record says what kind of agent it was. So a message
addressed to a stopped agent does not have to be refused — it can be the next
thing that agent hears.

This is the path SendMessage used to report as unavailable. Its answer was
honest, but it meant the only way to continue a stopped agent's work was to
spawn a fresh one and re-describe the whole task, throwing away the transcript
that was sitting right there.

What the source does and this does not: worktree revival, fork-agent resume with
the parent's exact system prompt, and content-replacement reconstruction. The
first two are features codemason does not have; the third has no state to
rebuild. The source also reads the agent type and description from a metadata
sidecar it writes per agent. There is no sidecar here, and none is needed: this
path is only reachable while the task record is still registered — SendMessage
has just looked it up — and the record carries both.
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional

from ..tasks.local_agent_task import (
    get_agent_transcript,
    is_local_agent_task,
    register_async_agent,
)
from ..utils.messages import (
    create_user_message,
    filter_orphaned_thinking_only_messages,
    filter_unresolved_tool_uses,
    filter_whitespace_only_assistant_messages,
)


def load_resumable_messages(agent_id: str):
    """The agent's conversation, in a shape that can be sent again.

    Three filters, in the source's order, each removing a different way an
    interrupted conversation is malformed:

      * `filter_unresolved_tool_uses` — the turn whose tool calls ALL went
        unanswered. That is the tail a stop leaves behind, and an orphaned
        tool_use is rejected outright by the API. Note this is the ALL rule,
        not the ANY rule `filter_incomplete_tool_calls` uses; a turn that got
        some results back carries real work and is kept.
      * `filter_orphaned_thinking_only_messages` — reasoning with nothing to
        merge into.
      * `filter_whitespace_only_assistant_messages` — a turn that said nothing.

    None when there is no transcript, or nothing survives the filters, which the
    caller must read as "cannot resume" rather than "resume from empty".
    """
    messages = get_agent_transcript(agent_id)
    if not messages:
        return None
    messages = filter_unresolved_tool_uses(messages)
    messages = filter_orphaned_thinking_only_messages(messages)
    messages = filter_whitespace_only_assistant_messages(messages)
    return messages or None


async def resume_agent_background(
    agent_id: str,
    prompt: str,
    tool_use_context: Any,
    can_use_tool: Optional[Callable] = None,
    get_app_state_fn: Optional[Callable] = None,
    set_app_state_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Restart a stopped agent from its transcript, in the background
    (resumeAgentBackground).

    Returns {'resumed': bool, 'message': str, 'agentId': str}. Background
    because that is the only shape that makes sense here: the caller is a
    SendMessage inside someone else's turn, and it must not block while the
    agent works. The agent announces its own outcome through the notification
    queue, exactly as a run_in_background spawn does.

    Registered under the SAME agent id, which makes this a REPLACEMENT in the
    task store rather than a new task — that is what register_task's
    carry-forward of held state was written for, and it is why the agent keeps
    its name in the SendMessage registry without re-registering it.
    """
    from ..query import query as llm_query
    from ..state.app_state import get_app_state as _default_get
    from ..state.app_state import set_app_state as _default_set
    from ..types.query import ToolUseContext
    from ..constants.prompts import agent_prefix_sections
    from ..utils import subagent_gate
    from ..utils.abort import child_of
    from ..utils.config import get_config
    from .agent_tool_utils import run_async_agent_lifecycle
    from .built_in_agents import get_agent_definition, resolve_agent_tools

    get_state = get_app_state_fn or getattr(tool_use_context, 'get_app_state', None) or _default_get
    set_state = set_app_state_fn or getattr(tool_use_context, 'set_app_state', None) or _default_set

    task = get_state().tasks.get(agent_id)
    if not is_local_agent_task(task):
        return {'resumed': False, 'agentId': agent_id,
                'message': f'No agent task is registered with ID {agent_id}.'}
    if task.status == 'running':
        return {'resumed': False, 'agentId': agent_id,
                'message': f'Agent {agent_id} is still running; nothing to resume.'}

    resumed_messages = load_resumable_messages(agent_id)
    if not resumed_messages:
        return {'resumed': False, 'agentId': agent_id,
                'message': (f'Agent {agent_id} has no transcript to resume from. '
                            'Launch a new agent with the full task instead.')}

    # The agent type the task recorded. An unknown one would raise rather than
    # silently become general-purpose, so it is resolved defensively: a resume
    # must not fail because a custom agent definition has since been removed.
    try:
        agent_def = get_agent_definition(task.agent_type)
    except ValueError:
        agent_def = get_agent_definition(None)

    from . import get_agent_tools
    skip_permissions = False
    if hasattr(tool_use_context, 'options') and isinstance(tool_use_context.options, dict):
        skip_permissions = tool_use_context.options.get('dangerously_skip_permissions', False)
    tools = resolve_agent_tools(agent_def, await get_agent_tools())

    model = get_config().model_name
    agent_prompt = [agent_def.get_system_prompt()] + agent_prefix_sections(model)

    # Its own scope again, and unlinked — this is a background relaunch, and
    # register_async_agent's whole point is that ESC on the main thread cancels
    # the turn without killing an agent the user put in the background. Under
    # the caller's scope it would have died with the very turn that resumed it.
    # Stopping it stays explicit: TaskStop sets this scope.
    agent_abort = child_of(None)
    sub_ctx = ToolUseContext(
        abort_controller=agent_abort,
        options={
            'tools': tools,
            'dangerously_skip_permissions': skip_permissions,
        },
        on_token_update=getattr(tool_use_context, 'on_token_update', None),
        on_thinking_update=getattr(tool_use_context, 'on_thinking_update', None),
        input_router=getattr(tool_use_context, 'input_router', None),
    )
    sub_ctx.agent_id = agent_id

    description = task.description or '(resumed)'
    # The transcript, then what the caller wants done next — the source's
    # `[...resumedMessages, createUserMessage({ content: prompt })]`.
    messages = [*resumed_messages, create_user_message(prompt)]

    register_async_agent(
        agent_id=agent_id,
        description=description,
        prompt=prompt,
        selected_agent=agent_def,
        set_app_state_fn=set_state,
        tool_use_id=getattr(tool_use_context, 'tool_use_id', None),
        abort_controller=agent_abort,
    )

    def _make_stream():
        return llm_query(
            messages=messages,
            system_prompt=agent_prompt,
            context={},
            can_use_tool=can_use_tool,
            tool_use_context=sub_ctx,
        )

    asyncio.ensure_future(run_async_agent_lifecycle(
        agent_id=agent_id,
        description=description,
        make_stream=_make_stream,
        tool_use_context=sub_ctx,
        set_app_state_fn=set_state,
        tools=tools,
        tool_use_id=getattr(tool_use_context, 'tool_use_id', None),
        start_time=time.time(),
        acquire_slot=lambda: subagent_gate.acquire(agent_abort),
        # The transcript is what it continues from, and it is already in
        # `messages`; the tally is seeded from it so the counts do not restart.
        initial_messages=list(resumed_messages),
    ))

    return {
        'resumed': True,
        'agentId': agent_id,
        'message': (f'Agent "{description}" ({agent_id}) was stopped and has been '
                    f'resumed in the background from its transcript '
                    f'({len(resumed_messages)} messages). You will be notified '
                    f'when it completes.'),
    }
