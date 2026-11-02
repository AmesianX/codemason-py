"""
TodoWrite Tool for CodeMason
Based on src/tools/TodoWriteTool/ from Claude Code

Schema, description, prompt, storage semantics (appState.todos[agentId ?? sessionId],
clear-on-all-done) and the result message are ported verbatim from the original.
The reactive Ink rendering is replaced by a terminal render of the list — the
terminal interface is the preserved local substrate.
"""

from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
)
from ..utils.formatting import colorize, Colors, plain_mode
from ..utils.todo_store import get_todos, set_todos
from ..constants.prompts import FILE_EDIT_TOOL_NAME


TODO_WRITE_TOOL_NAME = 'TodoWrite'

DESCRIPTION = (
    'Update the todo list for the current session. To be used proactively and often '
    'to track progress and pending tasks. Make sure that at least one task is '
    'in_progress at all times. Always provide both content (imperative) and '
    'activeForm (present continuous) for each task.'
)

PROMPT = """Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Examples of When to Use the Todo List

<example>
User: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!
Assistant: *Creates todo list with the following items:*
1. Creating dark mode toggle component in Settings page
2. Adding dark mode state management (context/store)
3. Implementing CSS-in-JS styles for dark theme
4. Updating existing components to support theme switching
5. Running tests and build process, addressing any failures or errors that occur
*Begins working on the first task*

<reasoning>
The assistant used the todo list because:
1. Adding dark mode is a multi-step feature requiring UI, state management, and styling changes
2. The user explicitly requested tests and build be run afterward
3. The assistant inferred that tests and build need to pass by adding "Ensure tests and build succeed" as the final task
</reasoning>
</example>

<example>
User: Help me rename the function getCwd to getCurrentWorkingDirectory across my project
Assistant: *Uses grep or search tools to locate all instances of getCwd in the codebase*
I've found 15 instances of 'getCwd' across 8 different files.
*Creates todo list with specific items for each file that needs updating*

<reasoning>
The assistant used the todo list because:
1. First, the assistant searched to understand the scope of the task
2. Upon finding multiple occurrences across different files, it determined this was a complex task with multiple steps
3. The todo list helps ensure every instance is tracked and updated systematically
4. This approach prevents missing any occurrences and maintains code consistency
</reasoning>
</example>


<example>
User: I need to implement these features for my e-commerce site: user registration, product catalog, shopping cart, and checkout flow.
Assistant: *Creates a todo list breaking down each feature into specific tasks based on the project architecture*
Let's start with user registration. This will involve creating database models, API endpoints, and frontend forms that integrate with your existing authentication system.

<reasoning>
The assistant used the todo list because:
1. The user provided multiple complex features to implement in a comma separated list
2. The todo list helps organize these large features into manageable tasks
3. This approach allows for tracking progress across the entire implementation
</reasoning>
</example>

<example>
User: Can you help optimize my React application? It's rendering slowly and has performance issues.
Assistant: *Reviews component structure, render patterns, state management, and data fetching*
After analyzing your codebase, I've identified several performance issues.
*Creates todo list with items like: 1) Implementing memoization for expensive calculations in ProductList, 2) Adding virtualization for long lists in Dashboard, 3) Optimizing image loading in Gallery component, 4) Fixing state update loops in ShoppingCart, 5) Reviewing bundle size and implementing code splitting*
Let's start by implementing memoization for the expensive calculations in your ProductList component.

<reasoning>
The assistant used the todo list because:
1. First, the assistant examined the codebase to identify specific performance issues
2. Based on this analysis, it identified multiple optimization opportunities
3. Performance optimization is a non-trivial task requiring multiple steps
4. The todo list helps methodically track improvements across different components
5. This systematic approach ensures all performance bottlenecks are addressed
</reasoning>
</example>

## Examples of When NOT to Use the Todo List

<example>
User: How do I print 'Hello World' in Python?
Assistant: In Python, you can print "Hello World" with this simple code:

python
print("Hello World")

This will output the text "Hello World" to the console when executed.

<reasoning>
The assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.
</reasoning>
</example>

<example>
User: What does the git status command do?
Assistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.

<reasoning>
The assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.
</reasoning>
</example>

<example>
User: Can you add a comment to the calculateTotal function to explain what it does?
Assistant: Sure, let me add a comment to the calculateTotal function to explain what it does.
* Uses the {FILE_EDIT_TOOL_NAME} tool to add a comment to the calculateTotal function *

<reasoning>
The assistant did not use the todo list because this is a single, straightforward task confined to one location in the code. Adding a comment doesn't require tracking multiple steps or systematic organization.
</reasoning>
</example>

<example>
User: Run npm install for me and tell me what happens.
Assistant: I'll run the npm install command for you.

*Executes: npm install*

The command completed successfully. Here's the output:
[Output of npm install command]

All dependencies have been installed according to your package.json file.

<reasoning>
The assistant did not use the todo list because this is a single command execution with immediate results. There are no multiple steps to track or organize, making the todo list unnecessary for this straightforward task.
</reasoning>
</example>

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully

   **IMPORTANT**: Task descriptions must have two forms:
   - content: The imperative form describing what needs to be done (e.g., "Run tests", "Build the project")
   - activeForm: The present continuous form shown during execution (e.g., "Running tests", "Building the project")

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Exactly ONE task must be in_progress at any time (not less, not more)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - Tests are failing
     - Implementation is partial
     - You encountered unresolved errors
     - You couldn't find necessary files or dependencies

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names
   - Always provide both forms:
     - content: "Fix authentication bug"
     - activeForm: "Fixing authentication bug"

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.
"""

# 원본은 템플릿 리터럴이라 ${FILE_EDIT_TOOL_NAME} 이 빌드 때 치환된다. 여기서는
# 평범한 문자열이라 치환이 안 돼 그 자리표시자가 그대로 모델에게 나갔다.
PROMPT = PROMPT.replace("{FILE_EDIT_TOOL_NAME}", FILE_EDIT_TOOL_NAME)

RESULT_MESSAGE = (
    'Todos have been modified successfully. Ensure that you continue to use the todo '
    'list to track your progress. Please proceed with the current tasks if applicable'
)


# === Input schema (ported from utils/todo/types.ts TodoListSchema) ===

class TodoItem(BaseModel):
    content: str = Field(..., min_length=1, description='Content cannot be empty')
    status: Literal['pending', 'in_progress', 'completed']
    activeForm: str = Field(..., min_length=1, description='Active form cannot be empty')


class TodoWriteInput(ToolInputSchema):
    todos: List[TodoItem] = Field(..., description='The updated todo list')


def _todo_key(context: Any) -> str:
    from ..utils.todo_store import todo_key
    return todo_key(getattr(context, 'agent_id', None))


class TodoWriteTool(Tool):
    """Create and manage the session task checklist."""

    name = TODO_WRITE_TOOL_NAME
    input_schema = TodoWriteInput
    DESCRIPTION = PROMPT   # full todo prompt (was short blurb; original API sends tool.prompt())

    @classmethod
    async def description(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return PROMPT

    @classmethod
    def is_read_only(cls) -> bool:
        return True  # touches session state only, no filesystem; checkPermissions=allow

    @classmethod
    async def is_enabled(cls) -> bool:
        return True  # original: !isTodoV2Enabled(); TodoV2 off in the local build

    @classmethod
    def needs_permissions(cls) -> bool:
        return False

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return 'TodoWrite'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """The list as checkboxes (TodoWriteTool/UI.tsx), not the raw todos array.

        completed ✓ (dim) · in_progress ◐ (bold) · pending ☐."""
        todos = input_data.get('todos', []) if isinstance(input_data, dict) else []
        plain = plain_mode()
        lines = []
        for todo in todos:
            content = (todo.get('content', '') if isinstance(todo, dict)
                       else getattr(todo, 'content', '')) or ''
            status = (todo.get('status', 'pending') if isinstance(todo, dict)
                      else getattr(todo, 'status', 'pending')) or 'pending'
            content = ' '.join(str(content).split())
            if len(content) > 100:
                content = content[:100] + '...'
            if status == 'completed':
                mark = '[x]' if plain else colorize('✓', Colors.GREEN)
                lines.append(f"{mark} {colorize(content, Colors.DIM)}")
            elif status == 'in_progress':
                mark = '[~]' if plain else colorize('◐', Colors.CYAN)
                lines.append(f"{mark} {colorize(content, Colors.BOLD)}")
            else:
                mark = '[ ]' if plain else '☐'
                lines.append(f"{mark} {content}")
        return '\n'.join(lines)

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        # Tool.ts: "Omit for tools whose results are surfaced elsewhere (e.g.,
        # TodoWrite updates the todo panel, not the transcript)." The list above IS
        # the result; echoing "Todos have been modified successfully…" is noise.
        return ''

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        raw = input_data.get('todos', [])
        todos = [t if isinstance(t, dict) else t.model_dump() for t in raw]

        key = _todo_key(context)
        old_todos = get_todos(key)
        all_done = bool(todos) and all(t.get('status') == 'completed' for t in todos)
        new_todos = [] if all_done else todos
        set_todos(key, new_todos)

        # No printing from here: the list is drawn by render_tool_use_message (the
        # display layer owns display). call() printing straight to stdout also leaked
        # the checkboxes into headless `-p` output, which must stay machine-readable.

        # verificationNudgeNeeded (TodoWriteTool.ts:75-114) is deliberately not
        # ported: it's gated behind feature('VERIFICATION_AGENT') + a GrowthBook
        # flag (both OFF in external/3P builds) AND it instructs the model to
        # spawn subagent_type="verification", an agent codemason does not have
        # (built-ins are general-purpose/Explore/Plan) — emitting it would make
        # the Agent tool raise "Agent type 'verification' not found". The base
        # result text below is the verbatim external-build output.
        yield ToolResultSuccess(
            type='result',
            result_for_assistant=RESULT_MESSAGE,
            data={'oldTodos': old_todos, 'newTodos': todos},
        )
