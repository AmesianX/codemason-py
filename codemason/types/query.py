"""
Query-related type definitions for CodeMason
Equivalent to src/query.ts types
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Union, AsyncGenerator, TYPE_CHECKING
from uuid import UUID, uuid4

from .messages import Message as APIMessage, MessageParam, ContentBlock, ToolUseBlock

if TYPE_CHECKING:
    from ..tools.base import Tool


# === Response Types ===

@dataclass
class Response:
    cost_usd: float
    response: str


# === Message Types ===

@dataclass
class FullToolUseResult:
    """Full tool use result with data and assistant message"""
    data: Optional[Any] = None
    result_for_assistant: Optional[str] = None
    error: Optional[str] = None


@dataclass
class UserMessage:
    """User message type"""
    message: MessageParam
    type: str = 'user'
    uuid: UUID = field(default_factory=uuid4)
    tool_use_result: Optional[FullToolUseResult] = None

    @property
    def is_user(self) -> bool:
        return True


@dataclass
class AssistantMessage:
    """Assistant message type"""
    message: APIMessage
    cost_usd: float = 0.0
    duration_ms: int = 0
    type: str = 'assistant'
    uuid: UUID = field(default_factory=uuid4)
    is_api_error_message: bool = False

    @property
    def is_assistant(self) -> bool:
        return True


@dataclass
class NormalizedMessage:
    """Normalized message for API"""
    role: str
    content: Any


@dataclass
class ProgressMessage:
    """Progress message type"""
    content: AssistantMessage
    normalized_messages: List[NormalizedMessage]
    sibling_tool_use_ids: Set[str]
    tools: List['Tool']
    tool_use_id: str
    type: str = 'progress'
    uuid: UUID = field(default_factory=uuid4)

    @property
    def is_progress(self) -> bool:
        return True


# Union type for all message types
Message = Union[UserMessage, AssistantMessage, ProgressMessage]


# === Binary Feedback Types ===

@dataclass
class BinaryFeedbackResultNoSkip:
    message: Optional[AssistantMessage]
    should_skip_permission_check: bool = False


@dataclass
class BinaryFeedbackResultSkip:
    message: AssistantMessage
    should_skip_permission_check: bool = True


BinaryFeedbackResult = Union[BinaryFeedbackResultNoSkip, BinaryFeedbackResultSkip]


# === Read File State ===

@dataclass
class ReadFileState:
    """A readFileState entry (fileStateCache.ts). Recorded by Read, consumed by
    Edit/Write/NotebookEdit for read-before-edit + modified-since-read, and by
    Read itself for same-range dedup.

    - timestamp: file mtime (seconds) at read time.
    - content: the file text at read time, used for the modified-since-read
      content-equality fallback (mtime can change without content changing —
      cloud sync / AV / linter touch). Stored for FULL reads; '' otherwise.
    - offset / limit: the read range (None = unbounded). Recorded for the dedup
      range-match and to compute is_full_read for the equality fallback. NOTE: a
      ranged read is NOT a partial view — see is_partial_view.
    - is_partial_view: a SEPARATE flag (fileStateCache.ts:14 `isPartialView?`),
      true ONLY for auto-injected memory/attachment state whose content may
      differ from disk (attachments.ts:1749). It is what makes Edit/Write demand
      a real Read first. A normal Read — full OR ranged — leaves it False.
      codemason has no such auto-injection, so it is always False here; it exists
      to keep the read-before-edit check faithful (`!state || state.isPartialView`).
    """
    timestamp: float
    content: str = ''
    offset: Optional[int] = None
    limit: Optional[int] = None
    is_partial_view: bool = False

    @property
    def is_full_read(self) -> bool:
        return self.offset is None and self.limit is None


# === Tool Use Context ===

@dataclass
class ToolUseContext:
    """Context for tool use operations"""
    abort_controller: Any = None  # AbortController equivalent (asyncio.Event)
    options: Dict[str, Any] = field(default_factory=dict)  # Options dict
    message_id: Optional[str] = None
    read_file_state: Dict[str, 'ReadFileState'] = field(default_factory=dict)
    on_token_update: Optional[Callable[[int], None]] = None
    on_content_delta: Optional[Callable[[str], None]] = None  # -rt: live safe text
    # Reasoning tokens generated so far this turn, when the server thinks on a
    # separate channel. Fired only while thinking actually happens, so the UI can
    # show the breakdown ONLY on turns that had any (see StreamingSpinner).
    on_thinking_update: Optional[Callable[[int], None]] = None
    # Re-pin the caller's spinner label for work that interrupts the turn, None
    # to hand the label back. Same shape and same wiring point as the two
    # above -- the turn owns the spinner, and nothing below it may know that.
    on_status: Optional[Callable[[Optional[str]], None]] = None
    # AppState.tasks accessors (the source threads { getAppState, setAppState }
    # into every tool). codemason's store is a module singleton, so these point
    # at state.app_state's get_app_state/set_app_state; task tools read them
    # here to match the original tool signature.
    get_app_state: Optional[Callable] = None
    set_app_state: Optional[Callable] = None
    # Per-call tool_use id. The source clones the context with toolUseId per
    # call; check_permissions_and_call_tool sets it so tools (e.g. Bash) can
    # stamp background-task notifications with <tool-use-id>.
    tool_use_id: Optional[str] = None
    # Sub-agent id, set on the context when a sub-agent runs. Declared (not an
    # ad-hoc attribute) so agent-keyed state (todos) and the EnterPlanMode agent
    # guard survive dataclasses.replace and a future slots/frozen dataclass.
    agent_id: Optional[str] = None
    # Auto-compaction circuit-breaker state (consecutive_failures counter).
    # Declared as a real field for the same reason as agent_id: it was set as an
    # ad-hoc attribute behind a silent `except`, so under a future slots/frozen
    # dataclass the assignment would raise-and-be-swallowed and the
    # MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES breaker would silently never trip.
    _autocompact_tracking: Optional[Dict[str, Any]] = None
    # Raised when the repetition recovery's LAST RESORT microcompacted a history
    # the decoder had already collapsed on — a reduction that then had nowhere to
    # go, because it lives in the recursion's arguments and the history belongs
    # to the REPL. cli.py's pre-turn check reads and clears it, passing it as
    # context_proven_fatal so the same reduction is recomputed at the one place
    # whose write-back already works.
    #
    # A DICT, not a bool, for the same reason _autocompact_tracking is one: the
    # flag is raised several recursions deep and read at the turn boundary, and
    # dataclasses.replace builds a new context. A bool would be copied and the
    # write would never arrive; a dict is shared by every shallow copy, so the
    # write lands wherever it is read. default_factory, so each context gets its
    # own — a sub-agent's collapse must not reduce the parent's history.
    #
    # Why a flag and not the rescued message set itself: that set is a mid-turn
    # snapshot, taken before the rescued turn produced its own tool calls and
    # its final answer. Installing it as the next turn's history would discard
    # the work the rescue was spent on. Recomputing over the CURRENT history
    # yields the same reduction (microcompact_messages is deterministic and
    # idempotent) with the rescued turn still in it.
    #
    # Measured 2026-10-02 on the Rust build, 16:03-16:11: two turns collapsed
    # back to back on the same history. Each spent all 7 nudges — prompt moving
    # +46 a time, cached pinned at 287,744, every attempt dying identically
    # (think=26, out=200, 200 characters of '!') — then the last resort reduced
    # 289,659 to 171,554 and the turn completed. The next turn's first request
    # went out at prompt=290,984, cached=287,744: the pre-collapse prefix had
    # never been left, so the history that killed the turn was the one carried
    # forward, and it died in the same place.
    _proven_fatal: Dict[str, bool] = field(default_factory=lambda: {'raised': False})
    # A reduction taken MID-TURN and lost with the recursion that owned it. The
    # REPL must redo it on its own history, so _compact_before_recursion raises
    # this for EVERY microcompaction — microcompaction stands on the threshold
    # alone, and a collapse merely borrows it.
    #
    # Separate from _proven_fatal on purpose: that flag asserts the decoder died
    # on this history and drives collapse-specific reporting, while this one
    # asserts only that a reduction needs re-applying — the ordinary case, which
    # must not depend on a collapse ever happening.
    _needs_reduction: Dict[str, bool] = field(default_factory=lambda: {'raised': False})
    # A mid-turn FULL compaction happened and the REPL must summarize its own
    # history. Separate from _needs_reduction because the two are not
    # interchangeable: microcompaction is a pure, idempotent function of the
    # history and can simply be replayed at the turn boundary, while a summary
    # is an API round trip and cannot be reproduced locally at all.
    _needs_summary: Dict[str, bool] = field(default_factory=lambda: {'raised': False})
    # Interactive REPL only: the ReplInputRouter owning the persistent
    # PromptSession. Anything needing user input mid-turn (permission prompts)
    # must route through it — stdin belongs to the session, a bare input()
    # would fight it for bytes. None in --plain mode and -p mode.
    input_router: Optional[Any] = None


# === Validation Types ===

@dataclass
class ValidationResultSuccess:
    result: bool = True


@dataclass
class ValidationResultFailure:
    result: bool = False
    message: str = ''
    meta: Optional[Dict[str, Any]] = None


ValidationResult = Union[ValidationResultSuccess, ValidationResultFailure]


# === Tool Result Types ===

@dataclass
class ToolResultData:
    """Data returned from a tool"""
    type: str
    result_for_assistant: str
    data: Optional[Any] = None


@dataclass
class ToolProgressData:
    """Progress data from a tool"""
    type: str = 'progress'
    content: Optional[AssistantMessage] = None
    normalized_messages: List[NormalizedMessage] = field(default_factory=list)
    tools: List['Tool'] = field(default_factory=list)


ToolResult = Union[ToolResultData, ToolProgressData]


# === Constants ===

MAX_TOOL_USE_CONCURRENCY = 10

# INTERRUPT_MESSAGE / INTERRUPT_MESSAGE_FOR_TOOL_USE live in utils.messages
# (the single verbatim source); the unused duplicates that were here are gone.
