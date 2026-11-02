"""
Base Tool class for CodeMason
Inferred from BashTool, GrepTool, etc. implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel

from ..utils.formatting import summarize_tool_input

if TYPE_CHECKING:
    from ..types.query import ToolUseContext, AssistantMessage, UserMessage, ProgressMessage


# === Validation Result Types ===

@dataclass
class ValidationResultSuccess:
    result: bool = True


@dataclass
class ValidationResultFailure:
    result: bool = False
    message: str = ''
    meta: Optional[Dict[str, Any]] = None


ValidationResult = Union[ValidationResultSuccess, ValidationResultFailure]


# === Permission Result (Tool.ts checkPermissions -> PermissionResult) ===

@dataclass
class PermissionResult:
    """Port of PermissionResult (types/permissions.ts). 3-state decision a tool
    returns from check_permissions:

    - 'allow' : auto-approve, no user prompt.
    - 'ask'   : defer to the interactive prompt (can_use_tool). The optional
                message is shown so the user knows WHY approval is needed
                (e.g. "Dangerous rm operation on critical path /etc").
    - 'deny'  : hard-block with message, no prompt.

    The whole permission flow is bypassed under dangerously_skip_permissions —
    faithful to the original, where checkPermissions is never consulted in that
    mode.
    """
    behavior: str = 'ask'  # 'allow' | 'ask' | 'deny'
    message: str = ''


# === Tool Result Types ===

@dataclass
class ToolResultSuccess:
    """Successful tool result"""
    type: str = 'result'
    result_for_assistant: str = ''
    data: Optional[Any] = None


@dataclass
class ToolResultProgress:
    """Progress update from tool"""
    type: str = 'progress'
    content: Any = None
    normalized_messages: List[Any] = field(default_factory=list)
    tools: List['Tool'] = field(default_factory=list)


ToolResult = Union[ToolResultSuccess, ToolResultProgress]


# === Base Tool Class ===

class Tool(ABC):
    """
    Base class for all tools in CodeMason.

    Each tool must implement:
    - name: Tool identifier
    - input_schema: Pydantic model for input validation
    - call: Main execution method

    Optional implementations:
    - description: Dynamic description generator
    - prompt: System prompt for the tool
    - is_read_only: Whether tool only reads data
    - is_enabled: Whether tool is available
    - needs_permissions: Whether tool needs user permission
    - validate_input: Additional input validation
    - user_facing_name: Display name for UI
    - render_tool_use_message: Format tool use for display
    - render_result_for_assistant: Format result for assistant
    """

    # Class attributes to be overridden
    name: str = ''
    input_schema: Any = None  # Pydantic BaseModel class
    # Optional aliases for backward compatibility when a tool is renamed
    # (ToolDef.aliases). A tool_use naming any alias resolves to this tool —
    # see matches_name. Old transcripts / SDK users that call the deprecated
    # name (e.g. KillShell -> TaskStop) keep working.
    aliases: List[str] = []
    # Max chars of result before it is persisted to disk and replaced with a
    # <persisted-output> preview (ToolDef.maxResultSizeChars; enforced in query
    # via utils.tool_result_storage). The effective threshold is min(this, 50k);
    # set to float('inf') to opt out entirely (e.g. Read, which self-bounds and
    # whose output would loop through Read again). Most tools declare
    # 100_000, so that is the base default; Bash/Grep declare lower.
    max_result_size_chars: float = 100_000

    @classmethod
    def input_json_schema(cls) -> Dict[str, Any]:
        """The JSON schema the model fills in (the `parameters` of the tool call).

        A hook, not a constant, so a tool can adapt a field's description to the
        session — e.g. Agent tells the model which language to write the sub-agent
        prompt in, right where that field is authored. Backends must call this
        rather than input_schema.model_json_schema() directly."""
        schema = getattr(cls.input_schema, 'model_json_schema', None)
        return schema() if callable(schema) else {}

    @classmethod
    async def description(cls, **kwargs) -> str:
        """Generate a description for this tool use"""
        return f"Execute {cls.name}"

    @classmethod
    async def prompt(cls) -> str:
        """Get the system prompt for this tool"""
        return ''

    @classmethod
    def is_read_only(cls) -> bool:
        """Whether this tool only reads data (doesn't modify anything)"""
        return False

    @classmethod
    def is_concurrency_safe(cls, input_data: Optional[Dict[str, Any]] = None) -> bool:
        """Whether this tool may run in parallel with other concurrency-safe
        tools (ToolDef.isConcurrencySafe; the scheduler in toolOrchestration.ts /
        StreamingToolExecutor.ts batches consecutive safe tools).

        The source buildTool default is `false`, but read-only tools there each
        set isConcurrencySafe = isReadOnly. codemason tools don't all annotate
        this yet, so the base defaults to `is_read_only()` — the same value a
        read-only tool would set — which preserves the existing "all read-only →
        concurrent" behavior without touching every tool. A tool that is safe
        while NOT read-only (e.g. TaskStop) overrides this to return True.
        """
        return cls.is_read_only()

    @classmethod
    def matches_name(cls, name: str) -> bool:
        """True if `name` is this tool's name or one of its aliases
        (toolMatchesName, Tool.ts)."""
        return name == cls.name or name in (cls.aliases or [])

    @classmethod
    async def is_enabled(cls) -> bool:
        """Whether this tool is currently enabled"""
        return True

    @classmethod
    def needs_permissions(cls) -> bool:
        """Whether this tool requires user permission"""
        return True

    @classmethod
    def always_ask(cls) -> bool:
        """True when this tool's prompt is the user's decision, not a grant.

        `dangerously_skip_permissions` says "don't ask me to authorise tools I
        was going to let you run anyway". It does not say "decide on my behalf".
        A tool that presents a *proposal* — a plan to approve, an option to pick
        — is asking the user to choose, and skipping that does not save a
        confirmation, it removes the choice. Those tools set this and keep
        asking regardless of the flag.

        Only ExitPlanMode sets it today. AskUserQuestion needs no flag because
        it is not on the permission path at all (needs_permissions() is False);
        the tool *is* the interaction.
        """
        return False

    @classmethod
    async def check_permissions(
        cls,
        input_data: Dict[str, Any],
        context: 'ToolUseContext' = None,
    ) -> 'PermissionResult':
        """Tool-specific permission decision (Tool.ts checkPermissions).

        Default: read-only tools auto-allow; everything else defers to the
        interactive prompt ('ask'). Tools with their own security model (Bash)
        override this to return 'deny'/'ask' with an explanatory message. The
        original's default is `{behavior:'allow'}` (defer to the rule/classifier
        system); codemason has no such system, so 'ask' is the faithful
        equivalent of "let the general permission flow decide" — which here is
        the y/n prompt."""
        return PermissionResult(behavior='allow' if cls.is_read_only() else 'ask')

    @classmethod
    async def validate_input(
        cls,
        input_data: Dict[str, Any],
        context: 'ToolUseContext'
    ) -> ValidationResult:
        """Validate the input before execution"""
        return ValidationResultSuccess()

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        """Get the user-facing name for this tool"""
        return cls.name

    @classmethod
    def get_tool_use_summary(cls, input_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """The one argument that identifies this call (Tool.ts getToolUseSummary).

        A path, a pattern, a URL — whatever a reader would use to tell two calls
        of the same tool apart, already shortened for display. None when the
        input does not carry it (a partial input mid-stream), which is what lets
        the caller fall back to the tool's name.
        """
        return None

    @classmethod
    def get_activity_description(cls, input_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """Present-tense line for a progress view (Tool.ts getActivityDescription).

        "Reading src/foo.py", "Running bun test", "Searching for pattern" — the
        verb belongs to the tool, the argument comes from get_tool_use_summary.
        None falls back to the tool name.

        This is what a parent watching a sub-agent reads: it is stored on each
        tool use in the agent's progress tally (tasks/local_agent_task.py), so
        "what is it doing right now" has an answer that is not just a tool name.
        """
        return None

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """Render the tool use for display (Tool.ts renderToolUseMessage).

        The tool owns its own display: the CLI prints the header and this body,
        and never inspects the input dict itself. May be multi-line; '' shows the
        header alone (the tool draws its own UI, like AskUserQuestion). Returned
        text may carry colour — the CLI indents it verbatim."""
        return summarize_tool_input(input_data)

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """Render the tool RESULT for display (Tool.ts renderToolResultMessage).

        Takes the tool's structured Output (ToolResultSuccess.data), NOT the
        serialized result_for_assistant the model reads — that one stays faithful
        to the original's mapToolResultToToolResultBlockParam (JSON/XML for some
        tools) and must never be dumped on screen.

        None = no custom rendering, the CLI shows the result text.
        ''   = render nothing (the original's "omit for tools whose results are
               surfaced elsewhere", e.g. TodoWrite's list)."""
        return None

    @classmethod
    def render_result_for_assistant(cls, result: Any) -> str:
        """Render the result for the assistant"""
        return str(result)

    @classmethod
    @abstractmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: 'ToolUseContext',
        can_use_tool: Optional[Callable] = None
    ) -> AsyncGenerator[ToolResult, None]:
        """
        Execute the tool.

        Args:
            input_data: Validated input parameters
            context: Tool use context with options and abort controller
            can_use_tool: Function to check if a tool can be used

        Yields:
            ToolResult: Either a progress update or final result
        """
        yield ToolResultSuccess()


# === Input Schema Base ===

class ToolInputSchema(BaseModel):
    """Base class for tool input schemas"""

    class Config:
        extra = 'forbid'  # strictObject equivalent


# (The create_validation_*/create_tool_* constructor helpers were unused — every
# tool builds ValidationResult*/ToolResult* directly — and have been removed.)
