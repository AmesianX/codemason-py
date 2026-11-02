"""local_agent task — adapted from the agent-task block of cli.js.

Ported functions (original names in parentheses):

    register_agent_foreground              (EXK)  — spawn a foreground agent task
    register_async_agent (Fm8)  — spawn one already backgrounded
    background_agent_task          (lXK)  — move a running one to background
    unregister_agent_foreground              (LXK)  — drop a finished foreground one
    kill_async_agent                (aq6)  — abort + mark killed
    complete_agent_task            (hL8)  — mark completed with its result
    fail_agent_task                (SL8)  — mark failed with its error
    is_local_agent_task            (EJ)   — type guard

Why this exists at all — the two lifecycle failures it rules out:

  * **An agent vanishing before its result is collected.** `unregister_agent_foreground`
    refuses to delete a task whose `is_backgrounded` is set, because the parent
    collects that one later through TaskOutput rather than from the tool's own
    return value.

  * **An agent lingering after its owner is gone.** Spawning registers a
    shutdown cleanup (`utils/cleanup_registry`) that kills the task; the handle
    it returns is stored on the task, and every terminal transition — killed,
    completed, failed — calls it before it nulls the runtime handles. Task
    removal calls it too, after the delete.

`retain` is not a spawn option. In the source it is flipped on only by entering
the transcript view, and it means two things: the task is exempt from automatic
eviction (`evict_after` stays unset), and the streaming loop buffers the agent's
messages into task state. codemason has no transcript viewer — that is Ink
substrate — so nothing flips it here, which matches the original's behaviour for
an agent nobody is watching: no message buffering, normal eviction. The field
and the eviction arithmetic are ported because the terminal transitions and
`evict_terminal_task` both key off them.
"""

import asyncio
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from ..constants.xml import (
    OUTPUT_FILE_TAG,
    STATUS_TAG,
    SUMMARY_TAG,
    TASK_ID_TAG,
    TASK_NOTIFICATION_TAG,
    TOOL_USE_ID_TAG,
    WORKTREE_BRANCH_TAG,
    WORKTREE_PATH_TAG,
    WORKTREE_TAG,
)
from ..state.app_state import get_app_state, set_app_state
from ..task import Task, TaskStateBase, create_task_state_base, now_ms
from ..types.messages import Message
from ..utils.cleanup_registry import register_cleanup
from ..utils.message_queue_manager import enqueue_pending_notification
from ..utils.task.disk_output import get_task_output_path, init_task_output
from ..utils.task.framework import PANEL_GRACE_MS, register_task, update_task_state

# Auto-background delay (getAutoBackgroundMs). The source returns 120s only when
# CLAUDE_AUTO_BACKGROUND_TASKS is truthy or the tengu_auto_background_agents
# gate is on (default false), and 0 otherwise — so the timer is OFF by default.
# codemason has no gate service; the env var alone decides, under codemason's
# own namespace (as CODEMASON_KEEP_GOING does), with the original name accepted
# as well.
AUTO_BACKGROUND_DELAY_MS = 120_000

_TRUTHY = ("1", "true", "yes", "on")


def auto_background_ms() -> int:
    """Milliseconds before a foreground agent auto-backgrounds, 0 = never."""
    for name in ("CODEMASON_AUTO_BACKGROUND_TASKS", "CLAUDE_AUTO_BACKGROUND_TASKS"):
        value = os.environ.get(name)
        if value is not None and value.strip().lower() in _TRUTHY:
            return AUTO_BACKGROUND_DELAY_MS
    return 0


# How many recent tool activities a tracker keeps (MAX_RECENT_ACTIVITIES).
MAX_RECENT_ACTIVITIES = 5


@dataclass
class ToolActivity:
    """One tool use, as the progress view shows it (ToolActivity)."""

    tool_name: str
    input: Dict[str, Any] = field(default_factory=dict)
    # Pre-computed description from the tool, e.g. "Reading src/foo.ts".
    activity_description: Optional[str] = None
    # Pre-computed: a search (Grep, Glob, ...) / a read (Read, cat, ...).
    is_search: Optional[bool] = None
    is_read: Optional[bool] = None


@dataclass
class AgentProgress:
    """What an agent has done so far, as stored on its task (AgentProgress)."""

    tool_use_count: int = 0
    token_count: int = 0
    last_activity: Optional[ToolActivity] = None
    recent_activities: Optional[List[ToolActivity]] = None
    summary: Optional[str] = None


@dataclass
class ProgressTracker:
    """The running tally an agent's loop keeps (ProgressTracker).

    Input and output are tracked separately to avoid double counting: the
    prompt token count is cumulative per turn (it includes all previous
    context), so only the latest value is kept, while output is per-turn and
    therefore summed. The cache fields belong to the same prompt as
    `input_tokens` and are added to it, not accumulated.
    """

    tool_use_count: int = 0
    latest_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    recent_activities: List[ToolActivity] = field(default_factory=list)


def create_progress_tracker() -> ProgressTracker:
    """A fresh tracker (createProgressTracker)."""
    return ProgressTracker()


def get_token_count_from_tracker(tracker: ProgressTracker) -> int:
    """The agent's context size so far (getTokenCountFromTracker)."""
    return tracker.latest_input_tokens + tracker.cumulative_output_tokens


# Given a tool name and its input, a human-readable line for the progress view
# (ActivityDescriptionResolver). Resolved at record time from the tool's own
# get_activity_description, so the view never has to look a tool up again.
ActivityDescriptionResolver = Callable[[str, Dict[str, Any]], Optional[str]]


def create_activity_description_resolver(tools: Any) -> ActivityDescriptionResolver:
    """Resolve activity descriptions out of a tool pool
    (createActivityDescriptionResolver)."""
    def _resolve(tool_name: str, tool_input: Dict[str, Any]) -> Optional[str]:
        for tool in tools or []:
            name = getattr(tool, "name", None) or getattr(tool, "TOOL_NAME", None)
            if name != tool_name:
                continue
            describe = getattr(tool, "get_activity_description", None)
            if describe is None:
                return None
            try:
                return describe(tool_input)
            except Exception:
                # A tool's display helper must never be able to fail the turn it
                # is only describing.
                return None
        return None

    return _resolve


def update_progress_from_message(
    tracker: ProgressTracker,
    message: Any,
    resolve_activity_description: Optional[ActivityDescriptionResolver] = None,
    tools: Any = None,
) -> None:
    """Fold one assistant message into the tracker (updateProgressFromMessage).

    Only assistant messages carry usage and tool uses, so everything else is
    ignored. The token arithmetic is the reason this is a tracker and not a
    running sum: the prompt count already includes every earlier turn, so
    adding it each time would multiply the context size by the turn count.
    """
    msg_type = getattr(message, "type", None) or (
        message.get("type") if isinstance(message, dict) else None
    )
    if msg_type != "assistant":
        return

    inner = getattr(message, "message", None)
    if inner is None and isinstance(message, dict):
        inner = message.get("message", {})

    usage = getattr(inner, "usage", None)
    if usage is None and isinstance(inner, dict):
        usage = inner.get("usage")

    def _usage(name: str) -> int:
        if usage is None:
            return 0
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        return int(value or 0)

    # Keep the latest prompt count, sum the completions. The two cache fields
    # are part of the same prompt; codemason's own backend fills them in
    # BetaUsage and leaves them 0 when the server does not report a cache, so
    # reading them costs nothing and counts the whole prompt when it does.
    tracker.latest_input_tokens = (
        _usage("input_tokens")
        + _usage("cache_creation_input_tokens")
        + _usage("cache_read_input_tokens")
    )
    tracker.cumulative_output_tokens += _usage("output_tokens")

    content = getattr(inner, "content", None)
    if content is None and isinstance(inner, dict):
        content = inner.get("content", [])
    for block in content or []:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "tool_use":
            continue
        tracker.tool_use_count += 1
        tool_name = getattr(block, "name", None) or (
            block.get("name", "") if isinstance(block, dict) else ""
        )
        tool_input = getattr(block, "input", None)
        if tool_input is None and isinstance(block, dict):
            tool_input = block.get("input", {})
        tracker.recent_activities.append(
            ToolActivity(
                tool_name=tool_name,
                input=tool_input or {},
                activity_description=(
                    resolve_activity_description(tool_name, tool_input or {})
                    if resolve_activity_description is not None
                    else None
                ),
            )
        )

    while len(tracker.recent_activities) > MAX_RECENT_ACTIVITIES:
        tracker.recent_activities.pop(0)


def get_progress_update(tracker: ProgressTracker) -> AgentProgress:
    """The snapshot to store on the task (getProgressUpdate)."""
    return AgentProgress(
        tool_use_count=tracker.tool_use_count,
        token_count=get_token_count_from_tracker(tracker),
        last_activity=tracker.recent_activities[-1] if tracker.recent_activities else None,
        recent_activities=list(tracker.recent_activities),
    )


def derive_agent_summary(tracker: ProgressTracker) -> Optional[str]:
    """A one-line "what is it doing right now", from what we already know.

    codemason's local equivalent of startAgentSummarization. The source forks
    the agent's transcript every 30 seconds and asks the model for the line,
    with a prompt whose own examples are "Reading runAgent.ts" and "Running auth
    module tests" — which is exactly the shape each tool's
    get_activity_description already produces, deterministically and for free.
    So the line is derived rather than generated.

    What that trades away: the model can describe an action its tool arguments do
    not name — "Fixing null check in validate.ts" reads the edit, not just the
    path. What it buys is the reason to prefer it here. A summary fork is another
    tenant in the server's KV pool, every 30 seconds, per running agent; on this
    deployment the pool is the binding constraint (see SUBAGENT_TOTAL_FRACTION in
    services/compaction), and three agents would add six requests a minute to it
    for a line we can already write.

    Returns None when nothing has happened yet, so the attachment that reads
    this simply omits its Progress line rather than printing an empty one.
    """
    for activity in reversed(tracker.recent_activities or []):
        if activity.activity_description:
            return activity.activity_description
    return None


@dataclass
class LocalAgentTaskState(TaskStateBase):
    """State for a sub-agent task (LocalAgentTaskState).

    `abort_controller` is the agent's own abort scope (utils/abort.AbortScope) —
    a child of the parent's, so a stop aimed at this agent spares its siblings.
    `selected_agent` holds the resolved agent definition and, like the two
    runtime handles, is dropped on a terminal transition: the source nulls it so
    a finished task stops pinning the definition and its tool pool.
    """

    agent_id: Optional[str] = None
    prompt: str = ""
    selected_agent: Any = None
    agent_type: str = "general-purpose"
    model: Optional[str] = None
    progress: Optional[AgentProgress] = None
    abort_controller: Any = None
    unregister_cleanup: Optional[Callable[[], None]] = None
    retrieved: bool = False
    last_reported_tool_count: int = 0
    last_reported_token_count: int = 0
    is_backgrounded: bool = False
    pending_messages: List[str] = field(default_factory=list)
    retain: bool = False
    disk_loaded: bool = False
    result: Any = None
    error: Any = None
    evict_after: Optional[int] = None
    # Buffered agent messages — only ever populated for a retained task; see the
    # module docstring.
    messages: Optional[List[Any]] = None


def is_local_agent_task(task: Any) -> bool:
    """Type guard (isLocalAgentTask)."""
    return getattr(task, "type", None) == "local_agent"


def is_panel_agent_task(task: Any) -> bool:
    """A local_agent task that is a spawned agent, not the main session
    (isPanelAgentTask).

    The source reserves `agentType === 'main-session'` for the session
    registering ITSELF as a local_agent task, which must not be listed or
    stopped as if it were a sub-agent. codemason does not register the main
    session yet, so today this agrees with is_local_agent_task on every task —
    it exists so the one predicate every listing must share is already the
    right one when that task appears.
    """
    return is_local_agent_task(task) and getattr(task, "agent_type", None) != "main-session"


def queue_pending_message(
    agent_id: str, msg: str, set_app_state_fn: Callable = set_app_state
) -> None:
    """Queue a message for a running agent to read (queuePendingMessage).

    Queued, not delivered: the agent is mid-turn and its message list belongs
    to the loop that is running it. The loop drains this at its next tool-round
    boundary.
    """
    def _apply(task):
        if not is_local_agent_task(task):
            return task
        return replace(task, pending_messages=[*task.pending_messages, msg])

    update_task_state(agent_id, set_app_state_fn, _apply)


def append_message_to_local_agent(
    agent_id: str, message: Any, set_app_state_fn: Callable = set_app_state
) -> None:
    """Append a message to the agent's displayed transcript
    (appendMessageToLocalAgent).

    Separate from queuing on purpose: queuing routes a prompt to the agent's
    API input, this puts it in what the user sees. The caller builds the
    message, which is what keeps this module out of the messages import cycle.

    A seam with no caller here, deliberately. The source's only call site is
    `onAgentSubmit` — the user typing INTO an agent's transcript while watching
    it in the teammate view, a screen codemason does not have. The two things
    that call site then does, we do have and do reach: a running agent gets the
    line queued (queue_pending_message) and a stopped one is relaunched
    (resume_agent_background), both driven by the SendMessage tool. Kept because
    the half that is missing is a view, not a behaviour.
    """
    def _apply(task):
        if not is_local_agent_task(task):
            return task
        return replace(task, messages=[*(task.messages or []), message])

    update_task_state(agent_id, set_app_state_fn, _apply)


def drain_pending_messages(
    agent_id: str,
    get_app_state_fn: Callable = get_app_state,
    set_app_state_fn: Callable = set_app_state,
) -> List[str]:
    """Take everything queued for this agent, leaving the queue empty
    (drainPendingMessages)."""
    task = get_app_state_fn().tasks.get(agent_id)
    if not is_local_agent_task(task) or not task.pending_messages:
        return []
    drained = list(task.pending_messages)

    def _clear(t):
        if not is_local_agent_task(t):
            return t
        return replace(t, pending_messages=[])

    update_task_state(agent_id, set_app_state_fn, _clear)
    return drained


# agentId -> the uuid of the last transcript entry written for that agent, so
# each new line can name its parent and the file reads as a chain
# (recordSidechainTranscript's lastRecordedUuid).
_agent_transcript_chains: Dict[str, str] = {}


def _last_transcript_uuid(agent_id: str) -> Optional[str]:
    """The uuid of the newest entry already on disk for this agent, if any.

    The leaf by the same rule the reader uses, so the entry written next hangs
    off exactly what a read would walk back from. Best-effort: an unreadable or
    absent file simply means this write starts the chain.
    """
    from pathlib import Path

    from ..utils.session_storage import _find_latest_leaf, load_transcript_file

    path = Path(get_task_output_path(agent_id))
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        messages, order = load_transcript_file(path)
    except (OSError, ValueError):
        return None
    if not messages or not order:
        return None
    leaf = _find_latest_leaf(messages, order)
    uuid_str = str(leaf.get('uuid') or '') if leaf else ''
    return uuid_str or None


def append_agent_transcript(agent_id: str, message: Any) -> None:
    """Append one message to the agent's own output file
    (recordSidechainTranscript, via initTaskOutputAsSymlink).

    This is what makes TaskOutput mean something for a RUNNING agent. The
    source points the task's output path at the agent's transcript JSONL — its
    own comment calls it "the full session transcript (every message, tool use,
    etc.), not just the subagent's answer" — and prefers the in-memory `result`
    once the run is over precisely because the file is the raw transcript.
    Without a writer on this side the path stayed an empty file forever, so a
    parent watching a sub-agent could read its token count and its tool count
    but never a word of what it was actually doing.

    codemason has no per-agent session file to link to, so the output path is
    written directly, in the same JSONL shape session_storage uses. The chain
    is kept per agent: borrowing session_storage's module-level chain would
    splice sub-agent entries into the main transcript's parent links.

    Best-effort, like every other write to this file: an agent's run must not
    fail because its log could not be appended to.
    """
    try:
        from ..utils.session_storage import (
            _serialize_entry,
            _TRANSCRIPT_TYPES,
            append_entry_to_file,
        )
        mtype = message.get('type') if isinstance(message, dict) else getattr(message, 'type', None)
        if mtype not in _TRANSCRIPT_TYPES:
            # progress and the like are not transcript entries, here as there.
            return
        parent = _agent_transcript_chains.get(agent_id)
        if parent is None:
            # No chain in memory, but the file may already hold this agent's
            # earlier turns: a terminal transition drops the chain, and a resume
            # then appends to the same file. Writing a second root there made
            # everything before the resume unreachable from the newest leaf —
            # get_agent_transcript walks parentUuid back from it — so a resumed
            # agent was handed one message instead of its whole history, and a
            # second resume would have lost the first resume's work too.
            parent = _last_transcript_uuid(agent_id)
        entry = _serialize_entry(message, parent, True)
        # The agent this line belongs to, which the main transcript's shape has
        # no field for: a reader of this file is looking at one agent, and the
        # id is what ties a line back to the task it came from.
        entry['agentId'] = agent_id
        from pathlib import Path
        append_entry_to_file(Path(get_task_output_path(agent_id)), entry)
        uuid_str = str(entry.get('uuid') or '')
        if uuid_str:
            _agent_transcript_chains[agent_id] = uuid_str
    except Exception:
        return


def get_agent_transcript(agent_id: str) -> Optional[List[Dict[str, Any]]]:
    """An agent's own conversation, read back off disk (getAgentTranscript).

    The agent's output file IS its transcript — see append_agent_transcript —
    so this is the session reader pointed at that path: parse the JSONL, find
    the latest leaf, and walk its parentUuid chain back to the root. The chain
    walk matters even for a file only one writer ever appended to: it is what
    turns the entries back into an ordered conversation rather than a bag.

    None when there is nothing to read, which the caller must treat as "cannot
    resume" rather than "resume from empty".
    """
    from pathlib import Path

    from ..utils.session_storage import (
        build_conversation_chain,
        load_transcript_file,
    )

    path = Path(get_task_output_path(agent_id))
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        messages, order = load_transcript_file(path)
    except (OSError, ValueError):
        return None
    if not messages or not order:
        return None

    from ..utils.session_storage import _find_latest_leaf
    leaf = _find_latest_leaf(messages, order)
    if leaf is None:
        return None
    chain = build_conversation_chain(messages, leaf)
    # Back to the in-memory message shape: the stored entry carries the
    # transcript's own bookkeeping (parentUuid, cwd, sessionId) around the
    # message itself, and only the message is a conversation turn. Images were
    # moved to the session's image store when the entry was written
    # (_serialize_entry); they come back here as they do for a resumed session
    # (restore_messages), or a resumed agent would see references, not images.
    from ..utils.image_store import hydrate
    out: List[Dict[str, Any]] = []
    for entry in chain:
        if entry.get('type') not in ('user', 'assistant'):
            continue
        out.append({
            'type': entry['type'],
            'uuid': entry.get('uuid', ''),
            'message': hydrate(entry.get('message') or {}, entry.get('sessionId')),
        })
    return out or None


def forget_agent_transcript(agent_id: str) -> None:
    """Drop an agent's chain state (the tail of recordSidechainTranscript).

    Called from the terminal transitions. The chain is one short string per
    agent, but a long-lived session spawning thousands of them would keep every
    one of them, and nothing would ever read them again.
    """
    _agent_transcript_chains.pop(agent_id, None)


# agentId -> the callable that resolves that agent's background signal (gR6).
_background_signal_resolvers: Dict[str, Callable[[], None]] = {}


class AgentTaskHandle(NamedTuple):
    """What a foreground spawn hands back (EXK's return value).

    `background_signal` is set once the task goes to the background, whether by
    the auto-background timer or by an explicit `background_agent_task`; the
    agent loop races it against the next message. `cancel_auto_background` stops
    the timer and is None when no timer was armed.
    """

    task_id: str
    background_signal: asyncio.Event
    cancel_auto_background: Optional[Callable[[], None]]


def _base_agent_task(
    agent_id: str,
    description: str,
    prompt: str,
    selected_agent: Any,
    abort_controller: Any,
    tool_use_id: Optional[str],
    is_backgrounded: bool,
) -> LocalAgentTaskState:
    """The task record both spawn paths build (the shared object literal)."""
    base = create_task_state_base(agent_id, "local_agent", description, tool_use_id)
    # The base helper starts every task 'pending'; the agent spawn paths override
    # it to 'running' (`{...C0(...), status:"running", ...}`). This is not
    # cosmetic: every terminal transition guards on `status == "running"`, so a
    # task left pending could never be completed, failed or killed.
    base["status"] = "running"
    return LocalAgentTaskState(
        **base,
        agent_id=agent_id,
        prompt=prompt,
        selected_agent=selected_agent,
        agent_type=getattr(selected_agent, "agent_type", None) or "general-purpose",
        abort_controller=abort_controller,
        retrieved=False,
        last_reported_tool_count=0,
        last_reported_token_count=0,
        is_backgrounded=is_backgrounded,
        pending_messages=[],
        retain=False,
        disk_loaded=False,
    )


def register_agent_foreground(
    agent_id: str,
    description: str,
    prompt: str,
    selected_agent: Any,
    set_app_state_fn: Callable = set_app_state,
    tool_use_id: Optional[str] = None,
    abort_controller: Any = None,
    auto_background_delay_ms: Optional[int] = None,
) -> AgentTaskHandle:
    """Register a running foreground sub-agent task (EXK).

    `abort_controller` is the scope the caller already built for this agent (the
    source's sync path makes its own here; codemason's agent tool builds a child
    of the parent's scope before this call and passes it in, so TaskStop on this
    task aborts the agent that is actually running).
    """
    init_task_output(agent_id)

    task = _base_agent_task(
        agent_id, description, prompt, selected_agent,
        abort_controller, tool_use_id, is_backgrounded=False,
    )
    # Registered before the task is installed, exactly as the source does: the
    # window in which an agent exists but has no teardown registered must be
    # empty.
    task = replace(
        task,
        unregister_cleanup=register_cleanup(
            lambda: kill_async_agent(agent_id, set_app_state_fn)
        ),
    )

    signal = asyncio.Event()
    _background_signal_resolvers[agent_id] = signal.set
    register_task(task, set_app_state_fn)

    cancel: Optional[Callable[[], None]] = None
    delay_ms = auto_background_ms() if auto_background_delay_ms is None else auto_background_delay_ms
    if delay_ms and delay_ms > 0:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            handle = loop.call_later(
                delay_ms / 1000.0, _auto_background, agent_id, set_app_state_fn
            )
            cancel = handle.cancel

    return AgentTaskHandle(agent_id, signal, cancel)


def _auto_background(agent_id: str, set_app_state_fn: Callable) -> None:
    """The auto-background timer body (EXK's setTimeout callback)."""
    def _flip(task):
        if not is_local_agent_task(task) or task.is_backgrounded:
            return task
        return replace(task, is_backgrounded=True)

    update_task_state(agent_id, set_app_state_fn, _flip)
    resolve = _background_signal_resolvers.pop(agent_id, None)
    if resolve is not None:
        resolve()


def register_async_agent(
    agent_id: str,
    description: str,
    prompt: str,
    selected_agent: Any,
    set_app_state_fn: Callable = set_app_state,
    tool_use_id: Optional[str] = None,
    abort_controller: Any = None,
) -> LocalAgentTaskState:
    """Register a sub-agent task that starts in the background (Fm8).

    No background signal and no auto-background timer: it is already there. The
    caller keeps the returned state so it can report the task id immediately —
    the async spawn path answers the parent before the agent has run.
    """
    init_task_output(agent_id)

    task = _base_agent_task(
        agent_id, description, prompt, selected_agent,
        abort_controller, tool_use_id, is_backgrounded=True,
    )
    task = replace(
        task,
        unregister_cleanup=register_cleanup(
            lambda: kill_async_agent(agent_id, set_app_state_fn)
        ),
    )
    register_task(task, set_app_state_fn)
    return task


def background_agent_task(
    agent_id: str,
    get_app_state_fn: Callable = get_app_state,
    set_app_state_fn: Callable = set_app_state,
) -> bool:
    """Move a running foreground agent task to the background (lXK).

    False when there is no such task or it is already backgrounded — the caller
    uses that to tell "moved" from "nothing to move".

    Also a seam with no caller here: the source reaches it from the UI's
    "background this agent" affordance (the BackgroundHint a long foreground run
    shows), which codemason does not draw. The timer path (_auto_background, off
    unless CODEMASON_AUTO_BACKGROUND_TASKS is set) drives the same transition,
    and the tests use this entry directly.
    """
    task = get_app_state_fn().tasks.get(agent_id)
    if not is_local_agent_task(task) or task.is_backgrounded:
        return False

    def _flip(t):
        if not is_local_agent_task(t):
            return t
        return replace(t, is_backgrounded=True)

    update_task_state(agent_id, set_app_state_fn, _flip)
    resolve = _background_signal_resolvers.pop(agent_id, None)
    if resolve is not None:
        resolve()
    return True


def unregister_agent_foreground(agent_id: str, set_app_state_fn: Callable = set_app_state) -> None:
    """Drop a finished foreground agent task (LXK).

    A BACKGROUNDED task is deliberately left in place: its result is collected
    later through TaskOutput, so deleting it here is the "agent vanished before
    anyone read it" bug. The captured unregister runs after the delete, matching
    the source's ordering — the registry no longer refers to the task before its
    exit cleanup is cancelled.
    """
    _background_signal_resolvers.pop(agent_id, None)
    captured: Dict[str, Optional[Callable[[], None]]] = {"fn": None}

    def _apply(prev):
        task = prev.tasks.get(agent_id)
        if not is_local_agent_task(task) or task.is_backgrounded:
            return prev
        captured["fn"] = task.unregister_cleanup
        remaining = {k: v for k, v in prev.tasks.items() if k != agent_id}
        return replace(prev, tasks=remaining)

    set_app_state_fn(_apply)
    if captured["fn"] is not None:
        captured["fn"]()


def _terminal(task, status: str, **extra):
    """Shared terminal transition (the common tail of the three settle paths).

    The source also calls `evictTaskOutput(taskId)` here, which flushes and
    drops that task's in-memory output writer. codemason writes task output
    straight to disk and keeps no writer map, so there is nothing to evict —
    this is a genuine no-op, not a missing call. What it DOES have to drop is
    the transcript chain, which is per-agent state no one will read again.
    """
    if task.unregister_cleanup is not None:
        task.unregister_cleanup()
    forget_agent_transcript(task.id)
    return replace(
        task,
        status=status,
        end_time=now_ms(),
        evict_after=None if task.retain else now_ms() + PANEL_GRACE_MS,
        abort_controller=None,
        unregister_cleanup=None,
        selected_agent=None,
        **extra,
    )


def kill_async_agent(agent_id: str, set_app_state_fn: Callable = set_app_state) -> None:
    """Abort a running agent task and mark it killed (aq6).

    The abort happens inside the state updater, as in the source, so the task is
    never observable as "killed but still un-aborted".
    """
    def _kill(task):
        if not is_local_agent_task(task) or task.status != "running":
            return task
        controller = task.abort_controller
        if controller is not None:
            try:
                controller.set()
            except Exception:
                pass
        return _terminal(task, "killed")

    update_task_state(agent_id, set_app_state_fn, _kill)


def complete_agent_task(
    agent_id: str, result: Any, set_app_state_fn: Callable = set_app_state
) -> None:
    """Mark a running agent task completed with its result (hL8)."""
    def _complete(task):
        if not is_local_agent_task(task) or task.status != "running":
            return task
        return _terminal(task, "completed", result=result)

    update_task_state(agent_id, set_app_state_fn, _complete)


def fail_agent_task(
    agent_id: str, error: Any, set_app_state_fn: Callable = set_app_state
) -> None:
    """Mark a running agent task failed with its error (SL8)."""
    def _fail(task):
        if not is_local_agent_task(task) or task.status != "running":
            return task
        return _terminal(task, "failed", error=error)

    update_task_state(agent_id, set_app_state_fn, _fail)


def mark_agents_notified(
    agent_id: str, set_app_state_fn: Callable = set_app_state
) -> None:
    """Mark a task notified without announcing it (markAgentsNotified).

    A bulk stop sends one aggregate message instead of one per agent; this is
    what suppresses the per-agent notices without pretending they were sent.
    """
    def _mark(task):
        if not is_local_agent_task(task) or task.notified:
            return task
        return replace(task, notified=True)

    update_task_state(agent_id, set_app_state_fn, _mark)


def kill_all_running_agent_tasks(
    tasks: Dict[str, Any], set_app_state_fn: Callable = set_app_state
) -> None:
    """Stop every running agent task (killAllRunningAgentTasks).

    Explicitly NOT what ESC does. In the source this is reached only from the
    deliberate `chat:killAgents` gesture (and from Ctrl+C while the user is
    looking at a teammate), under the comment "killing agents from the main
    prompt stays a deliberate gesture (chat:killAgents), not a side-effect of
    cancelling a turn." A background agent is meant to outlive the turn that
    spawned it; this is the one thing that ends it in bulk.
    """
    for agent_id, task in list(tasks.items()):
        if is_local_agent_task(task) and task.status == "running":
            kill_async_agent(agent_id, set_app_state_fn)


def kill_all_agents_and_notify(
    set_app_state_fn: Callable = set_app_state,
    get_app_state_fn: Callable = get_app_state,
) -> Optional[str]:
    """The whole bulk-stop gesture (killAllAgentsAndNotify). Returns what the
    model was told, or None when there was nothing running.

    Three steps the source takes in this order, and the order matters:
      1. kill every running agent task,
      2. mark each one notified — which SUPPRESSES its own kill notice,
      3. enqueue ONE aggregate message naming them all.
    Skipping step 2 would tell the model the same thing N+1 times; doing step 2
    without step 3 would stop three agents and never say so.

    Reads the task table itself rather than taking it as an argument: the
    source's handler pulls `store.getState().tasks` for exactly this reason
    ("Reads tasks from the store directly to avoid stale closures"), and a kill
    gesture must act on what is running now, not on what was running when a
    keybinding closure was built.

    The source also emits a per-task SDK `task_terminated` event here. There is
    no SDK consumer in codemason — the notification queue below is the whole
    audience — so that step has no local equivalent to stand in for.
    """
    tasks = get_app_state_fn().tasks
    running = [
        (agent_id, task)
        for agent_id, task in list(tasks.items())
        if is_local_agent_task(task) and task.status == "running"
    ]
    if not running:
        return None

    kill_all_running_agent_tasks(tasks, set_app_state_fn)

    descriptions = []
    for agent_id, task in running:
        mark_agents_notified(agent_id, set_app_state_fn)
        descriptions.append(task.description)

    if len(descriptions) == 1:
        summary = f'Background agent "{descriptions[0]}" was stopped by the user.'
    else:
        quoted = ", ".join(f'"{d}"' for d in descriptions)
        summary = (
            f"{len(descriptions)} background agents were stopped by the user: {quoted}."
        )
    enqueue_pending_notification(summary, mode="task-notification")
    return summary


def update_agent_progress(
    agent_id: str, progress: AgentProgress, set_app_state_fn: Callable = set_app_state
) -> None:
    """Store a progress snapshot on a running agent task (updateAgentProgress).

    An existing `summary` survives: it is written by the summarisation path on
    its own schedule, and a counts update from the agent's next message must not
    wipe prose it did not produce.
    """
    def _apply(task):
        if not is_local_agent_task(task) or task.status != "running":
            return task
        existing_summary = task.progress.summary if task.progress is not None else None
        next_progress = (
            replace(progress, summary=existing_summary)
            if existing_summary
            else progress
        )
        return replace(task, progress=next_progress)

    update_task_state(agent_id, set_app_state_fn, _apply)


def update_agent_summary(
    agent_id: str, summary: str, set_app_state_fn: Callable = set_app_state
) -> None:
    """Store a prose progress summary on a running agent (updateAgentSummary).

    Keeps the counts as they are — this path only knows the prose. The source
    additionally emits the summary to SDK consumers behind an opt-in flag;
    codemason has no such consumer, so the store is the whole of it.
    """
    def _apply(task):
        if not is_local_agent_task(task) or task.status != "running":
            return task
        previous = task.progress
        return replace(
            task,
            progress=AgentProgress(
                tool_use_count=previous.tool_use_count if previous else 0,
                token_count=previous.token_count if previous else 0,
                last_activity=previous.last_activity if previous else None,
                recent_activities=previous.recent_activities if previous else None,
                summary=summary,
            ),
        )

    update_task_state(agent_id, set_app_state_fn, _apply)


def enqueue_agent_notification(
    agent_id: str,
    description: str,
    status: str,  # 'completed' | 'failed' | 'killed'
    set_app_state_fn: Callable = set_app_state,
    error: Optional[str] = None,
    final_message: Optional[str] = None,
    usage: Optional[Dict[str, int]] = None,
    tool_use_id: Optional[str] = None,
    worktree_path: Optional[str] = None,
    worktree_branch: Optional[str] = None,
) -> None:
    """Announce a background agent's outcome (enqueueAgentNotification).

    Announced at most once, and the check-and-set is atomic for a reason: a
    TaskStop marks the task notified to suppress the kill notice, so a
    completion that lands at the same moment must not also be announced. The
    loser of that race is dropped here rather than reaching the model twice.

    A foreground agent needs none of this — its result is the tool's return
    value. This is for the one that was backgrounded, where the tool call
    returned long ago and the only way back to the conversation is the queue.
    """
    should_enqueue = {"v": False}

    def _mark(task):
        if task.notified:
            return task
        should_enqueue["v"] = True
        return replace(task, notified=True)

    update_task_state(agent_id, set_app_state_fn, _mark)
    if not should_enqueue["v"]:
        return

    if status == "completed":
        summary = f'Agent "{description}" completed'
    elif status == "failed":
        summary = f'Agent "{description}" failed: {error or "Unknown error"}'
    else:  # killed
        summary = f'Agent "{description}" was stopped'

    output_path = get_task_output_path(agent_id)
    tool_use_id_line = (
        f"\n<{TOOL_USE_ID_TAG}>{tool_use_id}</{TOOL_USE_ID_TAG}>" if tool_use_id else ""
    )
    # The agent's answer travels IN the notification. For a backgrounded agent
    # this is the only delivery: nothing is going to call the tool back for it.
    result_section = f"\n<result>{final_message}</result>" if final_message else ""
    usage_section = (
        "\n<usage>"
        f"<total_tokens>{usage.get('total_tokens', 0)}</total_tokens>"
        f"<tool_uses>{usage.get('tool_uses', 0)}</tool_uses>"
        f"<duration_ms>{usage.get('duration_ms', 0)}</duration_ms>"
        "</usage>"
        if usage
        else ""
    )
    worktree_section = ""
    if worktree_path:
        branch = (
            f"<{WORKTREE_BRANCH_TAG}>{worktree_branch}</{WORKTREE_BRANCH_TAG}>"
            if worktree_branch
            else ""
        )
        worktree_section = (
            f"\n<{WORKTREE_TAG}><{WORKTREE_PATH_TAG}>{worktree_path}"
            f"</{WORKTREE_PATH_TAG}>{branch}</{WORKTREE_TAG}>"
        )

    message = (
        f"<{TASK_NOTIFICATION_TAG}>\n"
        f"<{TASK_ID_TAG}>{agent_id}</{TASK_ID_TAG}>{tool_use_id_line}\n"
        f"<{OUTPUT_FILE_TAG}>{output_path}</{OUTPUT_FILE_TAG}>\n"
        f"<{STATUS_TAG}>{status}</{STATUS_TAG}>\n"
        f"<{SUMMARY_TAG}>{summary}</{SUMMARY_TAG}>"
        f"{result_section}{usage_section}{worktree_section}\n"
        f"</{TASK_NOTIFICATION_TAG}>"
    )
    enqueue_pending_notification(message, mode="task-notification")


class LocalAgentTask(Task):
    """The Task implementation registered for type 'local_agent'."""

    name = "LocalAgentTask"
    type = "local_agent"

    async def kill(self, task_id: str, set_app_state_fn: Callable) -> None:
        kill_async_agent(task_id, set_app_state_fn)


LOCAL_AGENT_TASK = LocalAgentTask()
