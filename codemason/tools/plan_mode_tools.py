"""
EnterPlanMode + ExitPlanMode tools for CodeMason.
Port of EnterPlanModeTool / ExitPlanModeV2Tool from Claude Code, adapted to the local
substrate:
- Plan-mode state is a process flag (utils/plan_mode) instead of React app state.
- ExitPlanMode takes the plan as a required `plan` parameter and echoes it back,
  instead of the V2 file-based flow (write plan to disk, read it back) — codemason
  has no plan-file / plan-mode-system-message attachment infra. The V2 SDK-facing
  extended schema (_sdkInputSchema) does expose an optional `plan`, so the param
  itself is precedented; the difference is that here it is the source of truth
  rather than a disk mirror. The tool descriptions and result messages are
  reworded accordingly (no plan-file or TeamCreate references).
- Approval uses codemason's terminal interface instead of the Ink dialog.
"""

from typing import Any, AsyncGenerator, Callable, Dict, Optional

from pydantic import Field

from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
    ValidationResultFailure,
    PermissionResult,
)
from ..utils.formatting import Colors, colorize, render_markdown
from ..utils.plan_mode import enter_plan_mode, exit_plan_mode, is_plan_mode_active


ENTER_PLAN_MODE_TOOL_NAME = 'EnterPlanMode'
EXIT_PLAN_MODE_TOOL_NAME = 'ExitPlanMode'

ENTER_PLAN_MODE_DESCRIPTION = """Use this tool proactively when you're about to start a non-trivial implementation task. Getting user sign-off on your approach before writing code prevents wasted effort and ensures alignment. This tool transitions you into plan mode where you can explore the codebase and design an implementation approach for user approval.

## When to Use This Tool

**Prefer using EnterPlanMode** for implementation tasks unless they're simple. Use it when ANY of these conditions apply:

1. **New Feature Implementation**: Adding meaningful new functionality
   - Example: "Add a logout button" - where should it go? What should happen on click?
   - Example: "Add form validation" - what rules? What error messages?

2. **Multiple Valid Approaches**: The task can be solved in several different ways
   - Example: "Add caching to the API" - could use Redis, in-memory, file-based, etc.
   - Example: "Improve performance" - many optimization strategies possible

3. **Code Modifications**: Changes that affect existing behavior or structure
   - Example: "Update the login flow" - what exactly should change?
   - Example: "Refactor this component" - what's the target architecture?

4. **Architectural Decisions**: The task requires choosing between patterns or technologies
   - Example: "Add real-time updates" - WebSockets vs SSE vs polling
   - Example: "Implement state management" - Redux vs Context vs custom solution

5. **Multi-File Changes**: The task will likely touch more than 2-3 files
   - Example: "Refactor the authentication system"
   - Example: "Add a new API endpoint with tests"

6. **Unclear Requirements**: You need to explore before understanding the full scope
   - Example: "Make the app faster" - need to profile and identify bottlenecks
   - Example: "Fix the bug in checkout" - need to investigate root cause

7. **User Preferences Matter**: The implementation could reasonably go multiple ways
   - If you would use AskUserQuestion to clarify the approach, use EnterPlanMode instead
   - Plan mode lets you explore first, then present options with context

## When NOT to Use This Tool

Only skip EnterPlanMode for simple tasks:
- Single-line or few-line fixes (typos, obvious bugs, small tweaks)
- Adding a single function with clear requirements
- Tasks where the user has given very specific, detailed instructions
- Pure research/exploration tasks (use the Agent tool with explore agent instead)

## What Happens in Plan Mode

In plan mode, you'll:
1. Thoroughly explore the codebase using Glob, Grep, and Read tools
2. Understand existing patterns and architecture
3. Design an implementation approach
4. Present your plan to the user for approval
5. Use AskUserQuestion if you need to clarify approaches
6. Exit plan mode with ExitPlanMode when ready to implement

## Examples

### GOOD - Use EnterPlanMode:
User: "Add user authentication to the app"
- Requires architectural decisions (session vs JWT, where to store tokens, middleware structure)

User: "Optimize the database queries"
- Multiple approaches possible, need to profile first, significant impact

User: "Implement dark mode"
- Architectural decision on theme system, affects many components

User: "Add a delete button to the user profile"
- Seems simple but involves: where to place it, confirmation dialog, API call, error handling, state updates

User: "Update the error handling in the API"
- Affects multiple files, user should approve the approach

### BAD - Don't use EnterPlanMode:
User: "Fix the typo in the README"
- Straightforward, no planning needed

User: "Add a console.log to debug this function"
- Simple, obvious implementation

User: "What files handle routing?"
- Research task, not implementation planning

## Important Notes

- This tool REQUIRES user approval - they must consent to entering plan mode
- If unsure whether to use it, err on the side of planning - it's better to get alignment upfront than to redo work
- Users appreciate being consulted before significant changes are made to their codebase
"""

EXIT_PLAN_MODE_DESCRIPTION = """Use this tool when you are in plan mode and have finished designing your implementation plan and are ready for user approval. Pass the plan as the `plan` parameter; the user will review it and approve or reject before you start coding.

## When to Use This Tool
IMPORTANT: Only use this tool when the task requires planning the implementation steps of a task that requires writing code. For research tasks where you're gathering information, searching files, reading files or in general trying to understand the codebase - do NOT use this tool.

## Before Using This Tool
Ensure your plan is complete and unambiguous:
- If you have unresolved questions about requirements or approach, use AskUserQuestion first
- Once your plan is finalized, use THIS tool to request approval

**Important:** Do NOT use AskUserQuestion to ask "Is this plan okay?" or "Should I proceed?" - that's exactly what THIS tool does. ExitPlanMode inherently requests user approval of your plan.

## Examples

1. Initial task: "Search for and understand the implementation of vim mode in the codebase" - Do not use the exit plan mode tool because you are not planning the implementation steps of a task.
2. Initial task: "Help me implement yank mode for vim" - Use the exit plan mode tool after you have finished planning the implementation steps of the task.
3. Initial task: "Add a new feature to handle user authentication" - If unsure about auth method (OAuth, JWT, etc.), use AskUserQuestion first, then use exit plan mode tool after clarifying the approach.
"""


# Short one-line descriptions — the source's description() returns these; the
# long text above is the tool prompt() (EnterPlanModeTool.ts:41 /
# ExitPlanModeV2Tool.ts description()).
ENTER_PLAN_MODE_SHORT_DESCRIPTION = (
    'Requests permission to enter plan mode for complex tasks requiring '
    'exploration and design'
)
EXIT_PLAN_MODE_SHORT_DESCRIPTION = 'Prompts the user to exit plan mode and start coding'


class EnterPlanModeInput(ToolInputSchema):
    pass


class ExitPlanModeInput(ToolInputSchema):
    plan: str = Field(
        ...,
        description="The implementation plan to present to the user for approval "
        "(markdown). Should be concise and unambiguous.",
    )


class EnterPlanModeTool(Tool):
    name = ENTER_PLAN_MODE_TOOL_NAME
    input_schema = EnterPlanModeInput
    DESCRIPTION = ENTER_PLAN_MODE_DESCRIPTION

    @classmethod
    async def description(cls, **kwargs) -> str:
        return ENTER_PLAN_MODE_SHORT_DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return ENTER_PLAN_MODE_DESCRIPTION

    @classmethod
    def is_read_only(cls) -> bool:
        return True

    @classmethod
    def is_concurrency_safe(cls, input_data=None) -> bool:
        return True

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        # Source userFacingName() returns '' for this meta tool.
        return ''

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        # EnterPlanModeTool/UI.tsx: no arguments to show.
        return ''

    # No needs_permissions / check_permissions override: the tool is read-only,
    # so the base check_permissions returns 'allow' (the source's default for
    # EnterPlanMode) — no approval prompt to enter plan mode.

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        # Cannot be used in agent contexts (matches the original agentId guard).
        if getattr(context, 'agent_id', None):
            return ValidationResultFailure(
                result=False, message='EnterPlanMode tool cannot be used in agent contexts'
            )
        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls, input_data: Dict[str, Any], context: Any, can_use_tool: Optional[Callable] = None
    ) -> AsyncGenerator[ToolResult, None]:
        # handlePlanModeTransition + setAppState(setMode 'plan') — now backed by
        # AppState.toolPermissionContext (stashes prior mode as pre_plan_mode).
        enter_plan_mode()
        # The TS tool builds the model-facing text in two stages: call() returns
        # the short `message`, then mapToolResultToToolResultBlockParam appends the
        # workflow block. codemason has no separate result-mapping stage, so the
        # combined non-interview-phase text (EnterPlanModeTool.ts ~99, 108-118) is
        # assembled here verbatim. (The interview-phase branch is omitted because
        # codemason has no plan-file / interview-phase infra.)
        yield ToolResultSuccess(
            type='result',
            result_for_assistant=(
                'Entered plan mode. You should now focus on exploring the codebase '
                'and designing an implementation approach.\n'
                '\n'
                'In plan mode, you should:\n'
                '1. Thoroughly explore the codebase to understand existing patterns\n'
                '2. Identify similar features and architectural approaches\n'
                '3. Consider multiple approaches and their trade-offs\n'
                '4. Use AskUserQuestion if you need to clarify the approach\n'
                '5. Design a concrete implementation strategy\n'
                '6. When ready, use ExitPlanMode to present your plan for approval\n'
                '\n'
                'Remember: DO NOT write or edit any files yet. This is a read-only '
                'exploration and planning phase.'
            ),
            data={'plan_mode': True},
        )


class ExitPlanModeTool(Tool):
    name = EXIT_PLAN_MODE_TOOL_NAME
    input_schema = ExitPlanModeInput
    DESCRIPTION = EXIT_PLAN_MODE_DESCRIPTION

    @classmethod
    async def description(cls, **kwargs) -> str:
        return EXIT_PLAN_MODE_SHORT_DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return EXIT_PLAN_MODE_DESCRIPTION

    @classmethod
    def is_read_only(cls) -> bool:
        # Source: false ("Now writes to disk"); exiting plan mode mutates the
        # permission context, so it is not read-only.
        return False

    @classmethod
    def is_concurrency_safe(cls, input_data=None) -> bool:
        return True

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        # Source userFacingName() returns '' for this meta tool.
        return ''

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        # ExitPlanModeTool/UI.tsx renderToolUseMessage returns null: the plan is
        # shown as the RESULT (below), not echoed again at call time.
        return ''

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """The approved plan, rendered as the markdown it is (ExitPlanModeTool/UI.tsx)."""
        if not isinstance(data, dict):
            return None
        plan = (data.get('plan') or '').strip()
        if not plan:
            return 'Plan approved — exiting plan mode.'
        lines = [colorize('Plan approved:', Colors.BOLD), '']
        lines += render_markdown(plan).split('\n')
        return '\n'.join(lines)

    @classmethod
    def always_ask(cls) -> bool:
        # A plan is the user's call. Approving one is not the same as
        # authorising a tool the model was going to run anyway, so -dsp must not
        # swallow it — the whole point of plan mode is the look before the leap.
        return True

    @classmethod
    async def check_permissions(cls, input_data, context=None) -> PermissionResult:
        # Source checkPermissions returns { behavior: 'ask', message: 'Exit plan
        # mode?' } for non-teammates — route approval through codemason's real
        # permission flow (can_use_tool), NOT a raw input() in call(). call()
        # then runs only when the user approves.
        return PermissionResult(behavior='ask', message='Exit plan mode?')

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        # Source: reject before the approval dialog when not in plan mode
        # (errorCode 1). The deferred-tool list announces this tool regardless of
        # mode, so the model can call it spuriously.
        if not is_plan_mode_active():
            return ValidationResultFailure(
                result=False,
                message=(
                    'You are not in plan mode. This tool is only for exiting plan '
                    'mode after writing a plan. If your plan was already approved, '
                    'continue with implementation.'
                ),
                meta={'errorCode': 1},
            )
        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls, input_data: Dict[str, Any], context: Any, can_use_tool: Optional[Callable] = None
    ) -> AsyncGenerator[ToolResult, None]:
        # Reaching call() means approval already happened (check_permissions
        # 'ask' -> can_use_tool, or dangerously_skip_permissions). Rejection is
        # handled by the permission flow, which yields its own denial result —
        # so there is no rejection branch here (the source has none either).
        plan = input_data.get('plan', '')

        # Exit plan mode: restore the prior mode (ExitPlanModeV2Tool.ts call()).
        exit_plan_mode()

        if not plan or not plan.strip():
            # Source: empty-plan branch of mapToolResultToToolResultBlockParam.
            yield ToolResultSuccess(
                type='result',
                result_for_assistant='User has approved exiting plan mode. You can now proceed.',
                data={'approved': True, 'plan': None},
            )
            return

        # Mirrors ExitPlanModeV2Tool.ts mapToolResultToToolResultBlockParam
        # (~481-491): the lead "You can now start coding…" line and the echoed
        # plan under an "## Approved Plan:" heading. The file-path and TeamCreate
        # lines are dropped as a substrate adaptation (codemason has no plan-file
        # infra and no TeamCreate tool).
        yield ToolResultSuccess(
            type='result',
            result_for_assistant=(
                'User has approved your plan. You can now start coding. Start '
                'with updating your todo list if applicable\n'
                '\n'
                '## Approved Plan:\n'
                f'{plan}'
            ),
            data={'approved': True, 'plan': plan},
        )
