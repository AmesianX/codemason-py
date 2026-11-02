"""
Agent Tool for CodeMason
Equivalent to src/tools/AgentTool/AgentTool.tsx
"""

import asyncio
import contextlib
import json
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from pydantic import Field

from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
)
from ..types.messages import Message, TextBlock
from ..types.query import AssistantMessage
from ..state.app_state import get_app_state, set_app_state
from ..utils.task.disk_output import get_task_output_path
from ..utils.config import get_config
from ..utils.formatting import render_markdown


# === Input Schema ===

class AgentInput(ToolInputSchema):
    """Input schema for Agent tool"""
    description: Optional[str] = Field(
        default=None, description="A short (3-5 word) description of the task"
    )
    prompt: str = Field(..., description="The task for the agent to perform")
    subagent_type: Optional[str] = Field(
        default=None,
        description="The type of specialized agent to use for this task",
    )
    run_in_background: Optional[bool] = Field(
        default=None,
        description=(
            "Set to true to run this agent in the background. You will be "
            "notified when it completes."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "Name for the spawned agent. Makes it addressable via "
            "SendMessage({to: name}) while running."
        ),
    )


TOOL_NAME = 'Agent'
# constants.prompts.LEGACY_AGENT_TOOL_NAME; kept here next to TOOL_NAME because
# importing prompts at module scope would make tools -> prompts -> tools circular.
LEGACY_TOOL_NAME = 'Task'

# Tool-use display: cap the echoed brief and wrap it (the full prompt still runs).
_MAX_PROMPT_CHARS = 800
_WRAP_COLS = 80

# The rule also stated in the target language: an instruction written IN the
# language it is asking for is a far stronger cue than the same sentence in
# English (measured: the model kept writing English briefs otherwise). Only
# languages we can state it in natively are listed; anything else gets the
# English sentence alone, which still names the language.
_NATIVE_PROMPT_RULE = {
    'korean': '이 도구의 prompt와 description은 반드시 한국어로 작성하세요. 영어로 쓰지 마세요.',
    'japanese': 'このツールの prompt と description は必ず日本語で書いてください。英語で書かないでください。',
    'chinese': '此工具的 prompt 和 description 必须用中文书写，不要使用英文。',
}


def _prompt_language_rule(language: str) -> str:
    rule = (
        f"IMPORTANT: write `prompt` and `description` in {language} — the user's "
        f"language — never in English, however technical the task is."
    )
    native = _NATIVE_PROMPT_RULE.get(language.strip().lower())
    return f"{rule} {native}" if native else rule


# === Agent Tool ===

from .built_in_agents import build_agent_description, get_agent_definition, resolve_agent_tools


class AgentTool(Tool):
    """Launch a sub-agent for complex tasks"""

    name = TOOL_NAME
    # The source tool is `name: "Agent", aliases: ["Task"]` (cli.js 2.1.88). The
    # constant was ported and then never wired to anything, so a tool_use naming
    # Task resolved to nothing: matches_name() returned False and the call died as
    # an unknown tool. Models trained on the source emit Task, and the port's own
    # comments call it "a legacy alias" as if it worked.
    aliases = [LEGACY_TOOL_NAME]
    input_schema = AgentInput
    # Model-facing description: lists available subagent_type agents (built dynamically).
    DESCRIPTION = build_agent_description()

    # NO build_description / input_json_schema override.
    #
    # Both used to close with the language rule from effective_language(), and
    # both feed the TOOL BLOCK, which the request carries ahead of the system
    # prompt and the entire message history — the base of the server's prefix
    # cache. effective_language() reads the LAST user message, and inside a tool
    # loop that is a tool RESULT, so the detected language swung with whatever a
    # command happened to print: an English log erased the rule, a file with
    # Korean in it restored it. Each swing rewrote the front of the prompt and
    # threw away every cached token behind it. Measured 2026-09-06 on the Rust
    # port, same code shape: ~123 seconds of silent re-prefill on a 121k-token
    # session, and the only two cold prefills in a 27-request session were
    # exactly the two requests where this string changed.
    #
    # The rule now rides the turn message (cli.py process_prompt), beside cwd and
    # the date, which were moved out for the same reason. Appended, so it cannot
    # move the prefix, and nearer the end, which reads stronger anyway.
    # DESCRIPTION (built once at import) supplies the wire text.

    @classmethod
    async def description(cls, **kwargs) -> str:
        return "Launch a new task"

    @classmethod
    async def prompt(cls, dangerously_skip_permissions: bool = False, **kwargs) -> str:
        # The faithful getPrompt port (built_in_agents.build_agent_description)
        # is static — it does not vary with dangerously_skip_permissions — so the
        # flag is accepted for signature compatibility but unused.
        return build_agent_description()

    @classmethod
    def is_read_only(cls) -> bool:
        # A sub-agent can run write tools (Write/Edit/Bash) when policy allows, so
        # launching one is NOT side-effect-free. Returning True let any
        # concurrency/permission shortcut that keys off is_read_only treat a
        # code-writing agent as harmless. (Individual sub-tool calls are still
        # gated via can_use_tool; this only corrects the tool's own label.)
        return False

    @classmethod
    def is_concurrency_safe(cls, input_data=None) -> bool:
        # [ours] The system prompt tells the model to batch independent agents into
        # one message "so they run concurrently", and the base default (is_read_only)
        # silently serialized every one of them — the model was being told about a
        # capability it did not have. is_read_only stays False: a sub-agent may run
        # write tools, so the permission label must not call it harmless. The
        # scheduler decision is a separate flag by design (base.py; TaskStop is
        # concurrency-safe while not read-only).
        #
        # Server-side the win is bounded and shape-dependent: the per-forward cost
        # of streaming the expert weights (~0.57s on the 2x-Spark lane) is shared
        # only by requests the engine puts in the SAME step, so two agents can come
        # out anywhere from -5% to +17% while three or four reach +14/+28%
        # (measured 2026-10-10). Correctness, not throughput, is why this is True.
        return True

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def needs_permissions(cls) -> bool:
        return False

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return 'Agent'

    @classmethod
    async def validate_input(
        cls,
        input_data: Dict[str, Any],
        context: Any
    ) -> ValidationResult:
        return ValidationResultSuccess()

    @classmethod
    async def check_permissions(cls, input_data, context=None):
        """Spawning a sub-agent is auto-approved (AgentTool checkPermissions).

        The source has one branch above this — route through the auto-mode
        security classifier — and it is guarded by `"external" === 'ant'`, a
        compile-time constant that is false in the shipped external build. So
        dead-code elimination leaves exactly one behaviour there: allow. There
        is no classifier here either, so this is that behaviour, not a
        relaxation of it.

        Unreachable today, and ported anyway. The permission flow is entered
        only when `needs_permissions()` is true (query.py), and this tool
        returns False there — so spawning was already never prompted, by that
        gate rather than this answer. Worth having regardless: the source
        implements the method, and `is_read_only` is deliberately False (a
        sub-agent can run Write/Edit/Bash when policy allows, so launching one
        is not side-effect-free), which means the base default would send this
        tool to the interactive prompt if the `needs_permissions` gate ever
        moved. This is the answer the original gives in that case.
        """
        from .base import PermissionResult
        return PermissionResult(behavior='allow')

    @classmethod
    def get_activity_description(cls, input_data):
        """The task's own description (AgentTool getActivityDescription).

        No verb added: `description` is already a 3-5 word phrase for what the
        sub-agent is doing, which is exactly what this line wants.
        """
        if isinstance(input_data, dict) and input_data.get('description'):
            return str(input_data['description'])
        return 'Running task'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """The brief we hand the sub-agent, wrapped to _WRAP_COLS."""
        prompt = input_data.get('prompt', '') if isinstance(input_data, dict) else ''
        # Models sometimes emit literal "\n" inside the JSON string argument.
        prompt = prompt.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\r\n', '\n').strip()
        if not verbose and len(prompt) > _MAX_PROMPT_CHARS:
            prompt = prompt[:_MAX_PROMPT_CHARS] + '...'
        out: List[str] = []
        for line in prompt.split('\n'):
            line = line.strip()
            while len(line) > _WRAP_COLS:
                out.append(line[:_WRAP_COLS])
                line = line[_WRAP_COLS:]
            if line:
                out.append(line)
        return '\n'.join(out)

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """The sub-agent's report is markdown prose — render it (AgentResponseDisplay)."""
        if not isinstance(data, dict):
            return None
        text = data.get('text') or ''
        return render_markdown(text.strip()) if text.strip() else None

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None
    ) -> AsyncGenerator[ToolResult, None]:
        """Run the agent task"""
        from ..query import query as llm_query
        from ..utils.messages import create_user_message, create_assistant_message, normalize_messages

        prompt = input_data.get('prompt', '')
        start_time = time.time()
        # Mirrors tracker.tool_use_count once the tracker exists; kept as a name
        # because the progress yields and the fallback summary read it.
        tool_use_count = 0

        messages: List[Message] = [create_user_message(prompt)]

        # Yield initial progress
        yield ToolResultSuccess(
            type='progress',
            result_for_assistant='Initializing...',
            data={
                'content': create_assistant_message('Initializing...'),
                'normalized_messages': normalize_messages(messages),
            }
        )

        # Resolve the agent definition for the requested subagent_type
        # (default general-purpose), then build its tool pool + system prompt.
        from ..types.query import ToolUseContext
        from ..constants.prompts import agent_prefix_sections

        subagent_type = input_data.get('subagent_type')
        agent_def = get_agent_definition(subagent_type)

        # Available tool pool (threads the parent's dangerously_skip_permissions so
        # a write-capable agent gets write tools), then filtered by the agent
        # definition's allow/deny policy (resolveAgentTools).
        from . import get_agent_tools
        skip_permissions = context.options.get('dangerously_skip_permissions', False) if hasattr(context, 'options') else False
        available_tools = await get_agent_tools()
        tools = resolve_agent_tools(agent_def, available_tools)

        # Agent system prompt = the agent definition's prompt + agent notes + env.
        _model = (context.options.get('slow_and_capable_model') if hasattr(context, 'options') else None) or get_config().model_name
        # Definition prompt + the session-constant tail (notes, env, language).
        # The tail is pinned per process — see agent_prefix_sections for why it
        # must not be rebuilt per spawn.
        agent_prompt = [agent_def.get_system_prompt()] + agent_prefix_sections(_model)

        # Create ToolUseContext for query
        # Pass on_token_update from parent context so spinner updates during sub-queries
        # Its own abort scope, under the parent's: the parent's stop reaches this
        # agent, this agent's stop does not reach the parent or a sibling. Sharing
        # the parent's event made a stop aimed at one agent take down every other
        # one — see utils/abort.py.
        #
        # ...unless it starts in the background, where the link is exactly what
        # must not exist. registerAsyncAgent is called with no parent under the
        # comment "Don't link to parent's abort controller -- background agents
        # should survive when the user presses ESC to cancel the main thread.
        # They are killed explicitly via chat:killAgents." ESC is a cancel of
        # the TURN; an agent the user deliberately detached is not part of that
        # turn any more. Linking it here meant pressing ESC on the main prompt
        # silently killed every background agent — the opposite of the reason
        # for backgrounding one. Stopping it stays possible and stays explicit:
        # TaskStop and kill_async_agent set this scope directly.
        from ..utils.abort import child_of
        from ..utils.agent_id import create_agent_id
        _is_background = input_data.get('run_in_background') is True
        agent_abort = child_of(
            None if _is_background else getattr(context, 'abort_controller', None)
        )
        # One id for the agent and its task: the source builds the task with
        # `id: agentId`, and that id names the agent's output file.
        agent_id = create_agent_id()
        tool_use_context = ToolUseContext(
            abort_controller=agent_abort,
            options={
                'tools': tools,
                'dangerously_skip_permissions': skip_permissions,
                'slow_and_capable_model': context.options.get('slow_and_capable_model') if hasattr(context, 'options') else None,
            },
            on_token_update=context.on_token_update if hasattr(context, 'on_token_update') else None,
            on_thinking_update=getattr(context, 'on_thinking_update', None),
            # The router the parent is holding stdin with. Dropping it did not
            # merely lose a nicety: a permission prompt for a tool the sub-agent
            # runs then saw no router, took the headless branch, and asked with a
            # bare input() that fights the REPL's PromptSession for bytes — while
            # the remote prompt, gated there behind "no terminal to answer with",
            # was never sent at all. A turn driven from the phone sat blocked on
            # a question the phone was never shown.
            input_router=getattr(context, 'input_router', None),
        )
        # Assign a distinct agent_id so state keyed by agent id (e.g. TodoWrite's
        # todo_store via _todo_key) does not collide with the parent's 'session'
        # key. The original derives a fresh createAgentId() per sub-agent.
        tool_use_context.agent_id = agent_id

        _set_app_state = getattr(context, 'set_app_state', None) or set_app_state
        _description = input_data.get('description') or (prompt[:60] if prompt else 'agent')

        # --- run_in_background: launch it and answer at once -----------------
        # The parent gets the agent's id and output path immediately and carries
        # on; the run is driven by a detached task that announces its own
        # outcome through the notification queue. Registered as backgrounded
        # from the start (registerAsyncAgent), which is also what stops the
        # foreground teardown below from touching it.
        if _is_background:
            from ..tasks.local_agent_task import register_async_agent
            from ..utils import subagent_gate
            from .agent_tool_utils import run_async_agent_lifecycle

            register_async_agent(
                agent_id=agent_id,
                description=_description,
                prompt=prompt,
                selected_agent=agent_def,
                set_app_state_fn=_set_app_state,
                tool_use_id=getattr(context, 'tool_use_id', None),
                abort_controller=agent_abort,
            )

            def _make_stream():
                return llm_query(
                    messages=messages,
                    system_prompt=agent_prompt,
                    context={},
                    can_use_tool=can_use_tool,
                    tool_use_context=tool_use_context,
                )

            # The slot is taken inside the lifecycle, not here: waiting for it
            # here would make a background launch block the parent, which is the
            # one thing it must not do.
            asyncio.ensure_future(run_async_agent_lifecycle(
                agent_id=agent_id,
                description=_description,
                make_stream=_make_stream,
                tool_use_context=tool_use_context,
                set_app_state_fn=_set_app_state,
                tools=tools,
                tool_use_id=getattr(context, 'tool_use_id', None),
                start_time=start_time,
                acquire_slot=lambda: subagent_gate.acquire(agent_abort),
            ))

            # Register name -> agentId for SendMessage routing. AFTER the
            # spawn, exactly as the source does, so a spawn that fails leaves
            # no entry pointing at an agent that does not exist. Foreground
            # agents are skipped on purpose: one is blocking its parent, so
            # there is nobody left to address it.
            _name = (input_data.get('name') or '').strip()
            if _name:
                def _register_name(prev, _name=_name, _id=agent_id):
                    from dataclasses import replace as _replace
                    return _replace(
                        prev,
                        agent_name_registry={**prev.agent_name_registry, _name: _id},
                    )

                _set_app_state(_register_name)

            # Whether the parent can look at the partial output itself, rather
            # than only being told where it is.
            can_read_output_file = any(
                getattr(t, 'name', None) in ('Read', 'Bash') for t in tools
            )
            launched = {
                'isAsync': True,
                'status': 'async_launched',
                'agentId': agent_id,
                'description': _description,
                'prompt': prompt,
                'outputFile': str(get_task_output_path(agent_id)),
                'canReadOutputFile': can_read_output_file,
            }
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=json.dumps(launched),
                data=launched,
            )
            return

        # Register the agent as a `local_agent` task for its whole life (EXK).
        # Deliberately BEFORE the slot wait below: the gate is codemason's own
        # addition, and an agent that is queued on it is already alive from the
        # parent's point of view. Registering after the wait would leave a queued
        # agent invisible to the parent and unstoppable by TaskStop — while
        # registering here makes a stop work, because the gate waits on this
        # agent's own abort scope and kill_async_agent sets it.
        from .agent_tool_utils import (
            extract_partial_result,
            filter_incomplete_tool_calls,
            run_async_agent_lifecycle,
        )
        from ..tasks.local_agent_task import (
            append_agent_transcript,
            complete_agent_task,
            create_activity_description_resolver,
            create_progress_tracker,
            derive_agent_summary,
            fail_agent_task,
            get_progress_update,
            is_local_agent_task,
            kill_async_agent,
            register_agent_foreground,
            unregister_agent_foreground,
            update_agent_progress,
            update_agent_summary,
            update_progress_from_message,
        )
        _agent_task = register_agent_foreground(
            agent_id=agent_id,
            description=_description,
            prompt=prompt,
            selected_agent=agent_def,
            set_app_state_fn=_set_app_state,
            tool_use_id=getattr(context, 'tool_use_id', None),
            abort_controller=agent_abort,
        )
        # The running tally the parent reads this agent by. Stored on the task
        # after every message, which is what lets the parent see an agent's
        # context grow — and stop one that is growing where it should not be.
        #
        # The source gates this store on an SDK opt-in in the foreground branch
        # and stores unconditionally in the backgrounded one. There is no SDK
        # consumer to ask here and the task store IS our consumer, so it is
        # unconditional, matching the branch whose reader is local.
        tracker = create_progress_tracker()
        resolve_activity = create_activity_description_resolver(tools)

        from ..utils import subagent_gate

        # One sub-agent at a time (config.max_concurrent_subagents), across every
        # codemason process on this machine — a second one evicts the parent from
        # a KV pool that holds 1.62 conversations. Say so while waiting: an
        # unattended run must never look hung.
        _slot = subagent_gate.try_acquire()
        if _slot is None:
            # ToolResultSuccess(type='progress'), like the two progress yields
            # above and below. ToolResult is Union[Success, Progress] — a type
            # alias, so calling it raised 'Cannot instantiate typing.Union' the
            # moment a sub-agent had to wait, which is every time two of them
            # overlap. The tool died there: no wait, no queue, nothing in
            # error.log, and the model was told the sub-agent produced no
            # output. The gate's whole waiting branch was unreachable.
            #
            # result_for_assistant is the FIELD, not a key inside data: that is
            # what query and cli read to show the line, so burying it in data
            # left the wait silent even once the call could be constructed.
            yield ToolResultSuccess(
                type='progress',
                result_for_assistant='Waiting for another sub-agent to finish (max_concurrent_subagents)',
                data={'normalized_messages': []},
            )
            _slot = await subagent_gate.acquire(agent_abort)
            if _slot is None:
                raise RuntimeError('Aborted while waiting for a sub-agent slot')

        # Pulled by hand rather than with `async for`, because every pull is
        # raced against this agent's background signal (EXK's backgroundSignal).
        # The signal is set by the auto-background timer or by an explicit
        # background_agent_task; whichever fires, the agent stops being the
        # parent's problem and carries on detached.
        _agent_iterator = llm_query(
            messages=messages,
            system_prompt=agent_prompt,
            context={},
            can_use_tool=can_use_tool,
            tool_use_context=tool_use_context,
        ).__aiter__()
        # One future for the whole run, not one per pull: re-waiting the event
        # each iteration is what the source's comment warns about ("otherwise
        # each iteration adds a new .then() reaction to the same pending
        # promise, accumulating callbacks for the lifetime of the agent").
        _background_wait = asyncio.ensure_future(_agent_task.background_signal.wait())
        was_backgrounded = False
        # The pull in flight, visible to the finally.
        #
        # Measured, not assumed: closing this generator still reaches the query
        # generator's own cleanup in ~0.1ms and leaves no task pending, with or
        # without the cancel below — the frame unwinds, the pull task loses its
        # last reference, and CPython's asyncgen finalizer closes the inner
        # generator. So the cancel is a belt, not a fix: it makes the teardown
        # deterministic instead of resting on that finalizer hook being
        # installed, which a different loop policy need not do. The test that
        # pins the property is what matters (test_auto_background).
        _next = None

        try:
            # Run the query loop
            while True:
                _next = asyncio.ensure_future(_agent_iterator.__anext__())
                _backgrounded_first = False
                if _background_wait is not None:
                    _done, _ = await asyncio.wait(
                        {_next, _background_wait},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    _backgrounded_first = _next not in _done

                if _backgrounded_first:
                    # The background signal won. Re-read the task: the signal
                    # and the flag are set together, so a signal without the
                    # flag is not a backgrounding and must not be treated as
                    # one. Give up on the race rather than re-waiting an event
                    # that stays set — and keep the pull we already have in
                    # flight, so the message it is about to produce is not lost.
                    # `... , None) or fallback`, not `..., fallback)`: the real
                    # ToolUseContext HAS a get_app_state attribute and it is
                    # None, so getattr finds it and the default never applies.
                    # Calling it raised "'NoneType' object is not callable" and
                    # killed the tool the moment the timer fired — the same
                    # idiom two dozen lines up (_set_app_state) gets this right.
                    _get_state = getattr(context, 'get_app_state', None) or get_app_state
                    _bg_task = _get_state().tasks.get(agent_id)
                    if not (is_local_agent_task(_bg_task) and _bg_task.is_backgrounded):
                        _background_wait.cancel()
                        _background_wait = None
                    else:
                        was_backgrounded = True
                        # It is not the parent's agent any more: drop the link
                        # to the parent's abort scope so ESC on the main thread
                        # cancels the turn without taking this agent with it.
                        # The source arrives at the same place by re-launching
                        # the continuation with `task.abortController`, which
                        # registerAgentForeground created unlinked; ours is one
                        # object that the task state and this agent's
                        # ToolUseContext already hold, so it detaches in place.
                        # Stopping it stays explicit — TaskStop sets this very
                        # scope, and that still works.
                        agent_abort.detach()
                        # Close the foreground pull so the query generator's own
                        # finally blocks run — they release the things a run
                        # holds (permission state, tool handles). Bounded,
                        # because a cleanup that hangs must not hold the parent:
                        # the source races the same close against a 1s sleep.
                        _next.cancel()
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await asyncio.wait_for(asyncio.shield(_next), timeout=1.0)
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await asyncio.wait_for(_agent_iterator.aclose(), timeout=1.0)

                        # Hand the conversation over, minus any tool call that
                        # never came back — the switch can land while a tool is
                        # mid-flight, and an orphaned tool_use is rejected
                        # outright when the conversation is sent again.
                        #
                        # The source restarts the agent from its original
                        # prompt here, because its promptMessages array is never
                        # appended to. codemason's `messages` IS the live
                        # transcript (the loop above appends every turn to it)
                        # and query() copies its input, so handing it back
                        # RESUMES the agent instead of making it redo the work
                        # it has already paid for.
                        _resume_messages = filter_incomplete_tool_calls(messages)

                        def _resume_stream(_msgs=_resume_messages):
                            return llm_query(
                                messages=_msgs,
                                system_prompt=agent_prompt,
                                context={},
                                can_use_tool=can_use_tool,
                                tool_use_context=tool_use_context,
                            )

                        # The slot travels with the agent — see
                        # run_async_agent_lifecycle. `_slot` is not released by
                        # the finally below once was_backgrounded is set.
                        asyncio.ensure_future(run_async_agent_lifecycle(
                            agent_id=agent_id,
                            description=_description,
                            make_stream=_resume_stream,
                            tool_use_context=tool_use_context,
                            set_app_state_fn=_set_app_state,
                            tools=tools,
                            tool_use_id=getattr(context, 'tool_use_id', None),
                            start_time=start_time,
                            initial_messages=_resume_messages,
                            slot=_slot,
                        ))

                        can_read_output_file = any(
                            getattr(t, 'name', None) in ('Read', 'Bash') for t in tools
                        )
                        launched = {
                            'isAsync': True,
                            'status': 'async_launched',
                            'agentId': agent_id,
                            'description': _description,
                            'prompt': prompt,
                            'outputFile': str(get_task_output_path(agent_id)),
                            'canReadOutputFile': can_read_output_file,
                        }
                        yield ToolResultSuccess(
                            type='result',
                            result_for_assistant=json.dumps(launched),
                            data=launched,
                        )
                        return

                # Whichever way the race went, there is still a pull in flight:
                # either it finished (and this returns at once) or the signal
                # was spurious and the race has just been dropped. Letting the
                # branch above fall through here is what keeps that message
                # from being swallowed.
                try:
                    message = await _next
                except StopAsyncIteration:
                    break
                messages.append(message)

                # Fold the message into the tally and publish it. Every message,
                # not only the ones carrying a tool use: the token count is what
                # the parent watches, and it moves on messages that use no tool.
                # The agent's own output file, so a parent that reads this
                # agent mid-run with TaskOutput sees what it is doing.
                append_agent_transcript(agent_id, message)

                update_progress_from_message(tracker, message, resolve_activity, tools)
                update_agent_progress(agent_id, get_progress_update(tracker), _set_app_state)
                # The prose line, after the counts: update_agent_progress
                # preserves an existing summary rather than overwriting it, so
                # the fresh one has to be written through the path that owns it.
                _summary = derive_agent_summary(tracker)
                if _summary:
                    update_agent_summary(agent_id, _summary, _set_app_state)
                tool_use_count = tracker.tool_use_count

                msg_type = message.type if hasattr(message, 'type') else message.get('type')
                if msg_type == 'assistant':
                    msg_obj = message.message if hasattr(message, 'message') else message.get('message', {})
                    msg_content = msg_obj.content if hasattr(msg_obj, 'content') else msg_obj.get('content', [])
                    for content in msg_content:
                        content_type = content.type if hasattr(content, 'type') else content.get('type')
                        if content_type == 'tool_use':
                            # Extract tool name and input
                            tool_name = content.name if hasattr(content, 'name') else content.get('name', '')
                            tool_input = content.input if hasattr(content, 'input') else content.get('input', {})

                            # Yield progress with tool details
                            yield ToolResultSuccess(
                                type='progress',
                                result_for_assistant=f'Agent → {tool_name} ({tool_use_count})',
                                data={
                                    'normalized_messages': normalize_messages(messages),
                                    'tool_name': tool_name,
                                    'tool_input': tool_input,
                                    'tool_use_count': tool_use_count,
                                }
                            )

            # Get final result.
            # The original finalizeAgentTool walks BACKWARD over the messages to
            # the most recent assistant message that HAS text blocks and returns
            # that text. Taking only messages[-1] loses the agent's real final
            # answer when the loop exits on a tool_use-only assistant turn.
            elapsed = time.time() - start_time

            # The backward walk lives in agent_tool_utils now: the background
            # lifecycle needs the same answer out of the same messages, and two
            # copies of "which message actually holds the result" is exactly the
            # kind of duplication that drifts.
            combined_text = extract_partial_result(messages) or ''
            text_blocks = [{'type': 'text', 'text': combined_text}] if combined_text else []

            # The run finished: mark the task completed and hand it the counts
            # the parent accounts against (hL8). Done before the final yield so
            # a consumer that reads task state on receiving the result sees a
            # completed task, never a running one.
            progress = get_progress_update(tracker)
            complete_agent_task(
                agent_id,
                {
                    'agent_id': agent_id,
                    'text': combined_text,
                    'tool_use_count': progress.tool_use_count,
                    'duration': elapsed,
                    # What the source puts in the completion's usage block:
                    # the agent's final context size, its tool count and how
                    # long it took.
                    'usage': {
                        'total_tokens': progress.token_count,
                        'tool_uses': progress.tool_use_count,
                        'duration_ms': int(elapsed * 1000),
                    },
                },
                _set_app_state,
            )

            if combined_text:
                yield ToolResultSuccess(
                    type='result',
                    result_for_assistant=combined_text,
                    data={
                        # 'text' is what render_tool_result_message displays; the
                        # Output must carry what it renders (Tool.ts renderToolResultMessage
                        # takes Output, never the assistant-facing string).
                        'text': combined_text,
                        'text_blocks': text_blocks,
                        'tool_use_count': tool_use_count,
                        'duration': elapsed,
                    }
                )
            else:
                # No assistant message had any text — fall back to a synthetic
                # summary so the parent still receives a non-empty result.
                result_text = f"Done ({tool_use_count} tool uses, {elapsed:.1f}s)"
                yield ToolResultSuccess(
                    type='result',
                    result_for_assistant=result_text,
                    data={'tool_use_count': tool_use_count, 'duration': elapsed}
                )

        except asyncio.CancelledError:
            # Stopped, not broken: the source distinguishes a user cancel from an
            # error ("user_cancel_sync") and the task's own terminal status has to
            # say which, because that is what the parent reads back.
            kill_async_agent(agent_id, _set_app_state)
            raise
        except Exception as exc:
            # Do NOT swallow into a ToolResultSuccess — that reports a FAILED
            # sub-agent run as success. Re-raise so the query() tool runner turns
            # it into a proper is_error tool_result (original re-throws when there
            # is no partial content to finalize).
            if agent_abort.is_set():
                # Aborted mid-flight and the abort surfaced as some other
                # exception (a cancelled HTTP read, a gate wait giving up).
                kill_async_agent(agent_id, _set_app_state)
            else:
                fail_agent_task(agent_id, str(exc), _set_app_state)
            raise
        finally:
            # Stop the auto-background timer before anything else: it must not
            # fire against a task that is already finishing.
            if _agent_task.cancel_auto_background is not None:
                _agent_task.cancel_auto_background()
            # The race future outlives nothing: cancel it so a run that ended
            # before the timer fired leaves no waiter behind.
            if _background_wait is not None:
                _background_wait.cancel()
            # And neither does the pull. See the note where `_next` is
            # declared: this is a deterministic belt over a teardown that
            # already works, not a repair of a leak.
            if _next is not None and not _next.done():
                _next.cancel()
            # Drop the finished task (LXK). A BACKGROUNDED task survives this on
            # purpose — its result is collected later through TaskOutput, so
            # removing it here is the "agent vanished before anyone read it" bug.
            # The exit cleanup the spawn registered is released by whichever came
            # first: the terminal transition above, or this removal if the task
            # never reached one.
            unregister_agent_foreground(agent_id, _set_app_state)
            # A backgrounded agent took the slot with it; releasing here would
            # admit a third agent while this one is still running.
            if not was_backgrounded:
                _slot.release()
