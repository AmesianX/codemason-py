"""
Skill Tool for CodeMason
Based on src/tools/SkillTool/ (getPrompt description + schema) from Claude Code.

A skill is a markdown file with frontmatter (see utils/skills.py). Invoking a
skill with `context: inline` (default) expands its body into the conversation
(returned as the tool result); `context: fork` runs the body in a sub-agent.

Substrate note: the original lists available skills via per-turn system-reminder
attachments; codemason has no attachment pipeline, so the skill listing is
appended to the tool description (built dynamically).
"""

import re
import shlex
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from pydantic import Field

from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
    ValidationResultFailure,
)
from ..utils.skills import discover_skills, get_skill, model_invocable_skills


SKILL_TOOL_NAME = 'Skill'


def _parse_arguments(args: str) -> List[str]:
    """argumentSubstitution.ts parseArguments: shell-quote split, falling back to
    a whitespace split if parsing fails."""
    if not args or not args.strip():
        return []
    try:
        return shlex.split(args)
    except ValueError:
        return [t for t in args.split() if t]


def substitute_arguments(
    content: str,
    args: Optional[str],
    append_if_no_placeholder: bool = True,
    argument_names: Optional[List[str]] = None,
) -> str:
    """Port of argumentSubstitution.ts substituteArguments (lines 94-145).

    Replaces $name / $ARGUMENTS[n] / $n / $ARGUMENTS placeholders in `content`.
    Only when NO placeholder was found (and args is non-empty) does it append
    "\\n\\nARGUMENTS: {args}"."""
    # undefined/null means no args provided - return content unchanged.
    if args is None:
        return content

    parsed = _parse_arguments(args)
    original = content
    argument_names = argument_names or []

    # Named arguments ($foo, $bar) map to positions: names[i] -> parsed[i].
    for i, name in enumerate(argument_names):
        if not name:
            continue
        val = parsed[i] if i < len(parsed) else ''
        # Match $name but not $name[...] or $nameXxx (word chars after).
        # repl is a function so `val` (arg text: C:\Users\x, \1, ...) is inserted
        # literally — a string repl would parse its backslashes (re.error / group refs).
        content = re.sub(rf'\${re.escape(name)}(?![\[\w])', lambda _m: val, content)

    # Indexed arguments $ARGUMENTS[0], $ARGUMENTS[1], ...
    def _indexed(m: 're.Match') -> str:
        idx = int(m.group(1))
        return parsed[idx] if 0 <= idx < len(parsed) else ''

    content = re.sub(r'\$ARGUMENTS\[(\d+)\]', _indexed, content)

    # Shorthand indexed arguments $0, $1, ... (not followed by word char).
    content = re.sub(r'\$(\d+)(?!\w)', _indexed, content)

    # Full arguments string.
    content = content.replace('$ARGUMENTS', args)

    # If no placeholder was substituted and args is non-empty, append.
    if content == original and append_if_no_placeholder and args:
        content = content + f'\n\nARGUMENTS: {args}'

    return content

_BASE_PROMPT = """Execute a skill within the main conversation

When users ask you to perform tasks, check if any of the available skills match. Skills provide specialized capabilities and domain knowledge.

When users reference a "slash command" or "/<something>" (e.g., "/commit", "/review-pr"), they are referring to a skill. Use this tool to invoke it.

How to invoke:
- Use this tool with the skill name and optional arguments
- Examples:
  - `skill: "pdf"` - invoke the pdf skill
  - `skill: "commit", args: "-m 'Fix bug'"` - invoke with arguments
  - `skill: "review-pr", args: "123"` - invoke with arguments

Important:
- When a skill matches the user's request, this is a BLOCKING REQUIREMENT: invoke the relevant Skill tool BEFORE generating any other response about the task
- NEVER mention a skill without actually calling this tool
- Do not invoke a skill that is already running
- Do not use this tool for built-in CLI commands (like /help, /clear, etc.)
"""


class SkillInput(ToolInputSchema):
    skill: str = Field(
        ..., description='The skill name. E.g., "commit", "review-pr", or "pdf"'
    )
    args: Optional[str] = Field(default=None, description='Optional arguments for the skill')


# Embedded-shell syntax (promptShellExecution.ts): fenced ```! blocks and inline
# !`cmd`. Inline requires start-of-line or preceding whitespace so a literal `!`
# mid-word doesn't match.
_SKILL_BLOCK_RE = re.compile(r'```!\s*\n?([\s\S]*?)\n?```')
_SKILL_INLINE_RE = re.compile(r'(?m)(?:^|(?<=\s))!`([^`]+)`')


async def _execute_shell_commands_in_prompt(text: str, context: Any, can_use_tool) -> str:
    """Port of executeShellCommandsInPrompt: run ```! blocks and !`inline`
    commands via Bash and splice their output into the skill body. Permission is
    gated through can_use_tool (fail-closed: no callback or a deny leaves the
    marker untouched, so a skill file can't silently run shell without approval).
    """
    from .bash_tool import BashTool
    from ..utils.tool_result_storage import process_tool_result_content

    matches = list(_SKILL_BLOCK_RE.finditer(text))
    if '!`' in text:
        matches += list(_SKILL_INLINE_RE.finditer(text))
    if not matches:
        return text

    result = text
    for m in matches:
        command = (m.group(1) or '').strip()
        if not command:
            continue
        allowed = False
        if can_use_tool is not None:
            try:
                perm = await can_use_tool(BashTool, {'command': command}, context, None)
                allowed = bool(perm.get('result')) if isinstance(perm, dict) else bool(perm)
            except Exception:
                allowed = False
        if not allowed:
            continue  # permission denied / no gate -> leave the marker as-is
        out = ''
        async for r in BashTool.call({'command': command}, context, can_use_tool):
            if getattr(r, 'type', None) == 'result':
                rc = r.result_for_assistant
                out = rc if isinstance(rc, str) else str(rc)
        out = process_tool_result_content(
            out, 'Bash', getattr(context, 'tool_use_id', '') or 'skill-inline', 30_000
        )
        result = result.replace(m.group(0), out, 1)
    return result


class SkillTool(Tool):
    name = SKILL_TOOL_NAME
    input_schema = SkillInput

    @classmethod
    def build_description(cls) -> str:
        skills = model_invocable_skills()
        if skills:
            listing = '\n'.join(f"- {s.name}: {s.description}" for s in skills)
        else:
            listing = '(no skills are currently available)'
        return f"{_BASE_PROMPT}\nAvailable skills:\n{listing}\n"

    @classmethod
    async def description(cls, **kwargs) -> str:
        return cls.build_description()

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return cls.build_description()

    @classmethod
    def is_read_only(cls) -> bool:
        return False  # a skill may run write tools

    @classmethod
    def is_concurrency_safe(cls, input_data=None) -> bool:
        # [ours] Same split as AgentTool: not read-only (a skill may write), but safe
        # to schedule alongside other concurrency-safe calls. Without this the base
        # default (is_read_only) serialized every forked skill. See agent_tool.py.
        return True

    @classmethod
    async def is_enabled(cls) -> bool:
        # Only expose the Skill tool when model-invocable skills exist.
        return len(model_invocable_skills()) > 0

    @classmethod
    def needs_permissions(cls) -> bool:
        return False

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return 'Skill'

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        name = (input_data.get('skill', '') or '').strip()
        if name.startswith('/'):
            name = name[1:]   # `/commit` -> `commit` (SkillTool.ts commandName normalization)
        if not name:
            return ValidationResultFailure(result=False, message='skill name is required')
        # Reject disable-model-invocation skills with the original's message
        # (SkillTool.ts:412-418) before the generic unknown-skill path.
        for s in discover_skills():
            if s.name == name and s.disable_model_invocation:
                return ValidationResultFailure(
                    result=False,
                    message=(
                        f'Skill {name} cannot be used with {SKILL_TOOL_NAME} '
                        f'tool due to disable-model-invocation'
                    ),
                )
        if get_skill(name) is None:
            available = ', '.join(s.name for s in model_invocable_skills()) or '(none)'
            return ValidationResultFailure(
                result=False, message=f'Unknown skill "{name}". Available: {available}'
            )
        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls, input_data: Dict[str, Any], context: Any, can_use_tool: Optional[Callable] = None
    ) -> AsyncGenerator[ToolResult, None]:
        name = (input_data.get('skill', '') or '').strip()
        if name.startswith('/'):
            name = name[1:]   # `/commit` -> `commit`
        args = input_data.get('args')
        skill = get_skill(name)
        if skill is None:
            yield ToolResultSuccess(type='result', result_for_assistant=f'Unknown skill "{name}".', data={})
            return

        # Faithful to loadSkillsDir.ts:343-362:
        #   1. prefix "Base directory for this skill: {baseDir}\n\n" (so the
        #      model knows where the skill's bundled files live),
        #   2. substitute $name (named args) / $ARGUMENTS[n] / $n / $ARGUMENTS
        #      placeholders, passing the frontmatter argument names,
        #   3. replace ${CODEMASON_SKILL_DIR} with the skill's own directory.
        content = skill.content
        if skill.base_dir:
            content = f"Base directory for this skill: {skill.base_dir}\n\n{content}"
        body = substitute_arguments(content, args, argument_names=skill.argument_names)
        if skill.base_dir:
            body = body.replace('${CODEMASON_SKILL_DIR}', skill.base_dir)
        # Execute embedded shell commands (```! blocks and !`inline`) and splice
        # their output in — the dynamic-context feature (promptShellExecution.ts).
        body = await _execute_shell_commands_in_prompt(body, context, can_use_tool)

        if skill.context == 'fork':
            report = await cls._run_forked(skill, body, context, can_use_tool)
            yield ToolResultSuccess(type='result', result_for_assistant=report, data={'skill': name, 'context': 'fork'})
        else:
            # Inline: the skill body expands into the conversation; the model
            # follows it directly.
            yield ToolResultSuccess(type='result', result_for_assistant=body, data={'skill': name, 'context': 'inline'})

    @classmethod
    async def _run_forked(cls, skill, body: str, context: Any, can_use_tool: Optional[Callable]) -> str:
        """Run a context:fork skill in a sub-agent, returning its final report."""
        import uuid
        from ..query import query as llm_query
        from ..utils.messages import create_user_message
        from ..constants.prompts import agent_prefix_sections
        from ..types.query import ToolUseContext
        from ..utils.config import get_config
        from . import get_agent_tools
        from .built_in_agents import get_agent_definition, resolve_agent_tools

        skip = context.options.get('dangerously_skip_permissions', False) if hasattr(context, 'options') else False
        agent_def = get_agent_definition(skill.agent)
        tools = resolve_agent_tools(agent_def, await get_agent_tools())
        model = (context.options.get('slow_and_capable_model') if hasattr(context, 'options') else None) or get_config().model_name
        # Definition prompt + the same session-constant tail the Agent tool uses
        # (notes, env, language) — one pinned copy per process, so a forked skill
        # and a sub-agent share the cached prefix instead of each spawn rebuilding
        # it from the current cwd and effective_language().
        system_prompt = [agent_def.get_system_prompt()] + agent_prefix_sections(model)

        # Own abort scope under the parent's, same as AgentTool: a forked skill is
        # a sub-agent tenant and must be stoppable without stopping its siblings.
        from ..utils.abort import child_of
        skill_abort = child_of(getattr(context, 'abort_controller', None))
        sub_ctx = ToolUseContext(
            abort_controller=skill_abort,
            options={
                'tools': tools,
                'dangerously_skip_permissions': skip,
                'slow_and_capable_model': model,
            },
            on_token_update=getattr(context, 'on_token_update', None),
            on_thinking_update=getattr(context, 'on_thinking_update', None),
            # Same reason as the sub-agent context: without the parent's router a
            # permission prompt raised inside the skill's own query loop asks
            # with a bare input() and never reaches the phone.
            input_router=getattr(context, 'input_router', None),
        )
        sub_ctx.agent_id = uuid.uuid4().hex[:8]

        # Same slot the Agent tool takes: a forked skill is a sub-agent tenant.
        from ..utils import subagent_gate
        _slot = await subagent_gate.acquire(skill_abort)
        if _slot is None:
            raise RuntimeError('Aborted while waiting for a sub-agent slot')

        messages: List[Any] = [create_user_message(body)]
        try:
            async for msg in llm_query(messages=messages, system_prompt=system_prompt, context={},
                                       can_use_tool=can_use_tool, tool_use_context=sub_ctx):
                messages.append(msg)
        finally:
            _slot.release()

        # Final assistant text (faithful to finalizeAgentTool: last assistant w/ text).
        for m in reversed(messages):
            mtype = m.type if hasattr(m, 'type') else (m.get('type') if isinstance(m, dict) else None)
            if mtype == 'assistant':
                obj = m.message if hasattr(m, 'message') else m.get('message', {})
                content = obj.content if hasattr(obj, 'content') else obj.get('content', [])
                texts = [(b.text if hasattr(b, 'text') else b.get('text', '')) for b in content
                         if (b.type if hasattr(b, 'type') else b.get('type')) == 'text']
                joined = '\n'.join(t for t in texts if t).strip()
                if joined:
                    return joined
        return f'Skill "{skill.name}" completed.'
