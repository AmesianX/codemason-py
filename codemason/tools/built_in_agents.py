"""
Built-in agent definitions + tool resolution for the Agent tool.
Based on src/tools/AgentTool/built-in/* + agentToolUtils.ts from Claude Code
(filterToolsForAgent / resolveAgentTools) + prompt.ts (getPrompt agent list).

Each subagent_type maps to an AgentDefinition with its own system prompt and an
allow/deny tool policy. Agent prompts are the EXTERNAL-resolved text
(embedded search tools off → steer to Glob/Grep). The original's SDK-only
agents (statusline-setup, claude-code-guide, verification) are omitted — they
have no local substrate; general-purpose / Explore / Plan are the portable set.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Type

from .base import Tool

# Tool-name constants (match the registered tool names).
AGENT_TOOL_NAME = 'Agent'
FILE_READ_TOOL_NAME = 'Read'
FILE_WRITE_TOOL_NAME = 'Write'
# tools.send_message_tool.SEND_MESSAGE_TOOL_NAME; named here for the same reason
# the others are — importing the tool module would close an import cycle.
SEND_MESSAGE_TOOL_NAME = 'SendMessage'
FILE_EDIT_TOOL_NAME = 'Edit'
GLOB_TOOL_NAME = 'Glob'
GREP_TOOL_NAME = 'Grep'
BASH_TOOL_NAME = 'Bash'
NOTEBOOK_EDIT_TOOL_NAME = 'NotebookEdit'
ENTER_PLAN_MODE_TOOL_NAME = 'EnterPlanMode'
EXIT_PLAN_MODE_TOOL_NAME = 'ExitPlanMode'
ASK_USER_QUESTION_TOOL_NAME = 'AskUserQuestion'

# Tools no sub-agent may use. Mirrors constants/tools.ts ALL_AGENT_DISALLOWED_TOOLS,
# restricted to the members that have a REGISTERED codemason tool:
#   - Agent           -> recursion guard (no nested sub-agents)
#   - AskUserQuestion -> blocks forever on terminal input from a sub-agent
#   - EnterPlanMode / ExitPlanMode -> mutate main-thread plan-mode state
#   - TaskOutput / TaskStop -> require main-thread task state; a sub-agent must
#     not read/kill the parent's background tasks (these ARE registered here, so
#     they must be barred explicitly — the original bars them too).
# The original's USER_TYPE === 'ant' carve-out (which keeps Agent enabled for
# nested agents) is not reproduced: codemason has no ant build, so Agent is
# always barred. (Workflow is genuinely not ported, so it's omitted.)
ALL_AGENT_DISALLOWED_TOOLS = {
    AGENT_TOOL_NAME,
    ASK_USER_QUESTION_TOOL_NAME,
    ENTER_PLAN_MODE_TOOL_NAME,
    EXIT_PLAN_MODE_TOOL_NAME,
    'TaskOutput',
    'TaskStop',
    # SendMessage: the source does not bar it — it simply never grants it. A
    # sub-agent's pool there is an ALLOWLIST (ASYNC_AGENT_ALLOWED_TOOLS) that
    # does not contain SendMessage; the tool is granted only to in-process
    # teammates and to the coordinator. codemason's pool is a denylist, so
    # producing the same effect means naming it here. Which is the behaviour
    # that matters: a sub-agent holding SendMessage could steer a sibling, and
    # neither of them is the one the user is talking to.
    'SendMessage',
    # Skill: same recursion guard as Agent. A `context: fork` skill spawns a
    # sub-agent; if that sub-agent still had Skill it could invoke a fork skill
    # again → unbounded nested sub-agent spawning, bypassing the Agent guard.
    #
    # NOTE: this is a deliberate divergence. The source ALLOWS Skill for async
    # agents (it is in ASYNC_AGENT_ALLOWED_TOOLS). It can afford to, because
    # its fork path is gated by a feature flag; codemason's `context: fork`
    # skills always spawn, so the guard is load-bearing here.
    'Skill',
}


@dataclass
class AgentDefinition:
    agent_type: str
    when_to_use: str
    get_system_prompt: Callable[[], str]
    tools: Optional[List[str]] = None          # allowlist; None / ['*'] = all
    disallowed_tools: List[str] = field(default_factory=list)
    source: str = 'built-in'


# === general-purpose (generalPurposeAgent.ts) ===

_SHARED_PREFIX = (
    "You are an agent for CodeMason, an AI-powered coding assistant CLI. Given "
    "the user's message, you should use the tools available to complete the task. "
    "Complete the task fully—don't gold-plate, but don't leave it half-done."
)

_SHARED_GUIDELINES = """Your strengths:
- Searching for code, configurations, and patterns across large codebases
- Analyzing multiple files to understand system architecture
- Investigating complex questions that require exploring many files
- Performing multi-step research tasks

Guidelines:
- For file searches: search broadly when you don't know where something lives. Use Read when you know the specific file path.
- For analysis: Start broad and narrow down. Use multiple search strategies if the first doesn't yield results.
- Be thorough: Check multiple locations, consider different naming conventions, look for related files.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested."""


def _general_purpose_prompt() -> str:
    return (
        f"{_SHARED_PREFIX} When you complete the task, respond with a concise report "
        "covering what was done and any key findings — the caller will relay this to "
        f"the user, so it only needs the essentials.\n\n{_SHARED_GUIDELINES}"
    )


GENERAL_PURPOSE_AGENT = AgentDefinition(
    agent_type='general-purpose',
    when_to_use=(
        'General-purpose agent for researching complex questions, searching for '
        'code, and executing multi-step tasks. When you are searching for a keyword '
        'or file and are not confident that you will find the right match in the '
        'first few tries use this agent to perform the search for you.'
    ),
    tools=['*'],
    get_system_prompt=_general_purpose_prompt,
)


# === Explore (exploreAgent.ts) ===

EXPLORE_AGENT_MIN_QUERIES = 3


def _explore_prompt() -> str:
    return f"""You are a file search specialist for CodeMason. You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools - attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use {GLOB_TOOL_NAME} for broad file pattern matching
- Use {GREP_TOOL_NAME} for searching file contents with regex
- Use {FILE_READ_TOOL_NAME} when you know the specific file path you need to read
- Use {BASH_TOOL_NAME} ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
- NEVER use {BASH_TOOL_NAME} for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification
- Adapt your search approach based on the thoroughness level specified by the caller
- Communicate your final report directly as a regular message - do NOT attempt to create files

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

Complete the user's search request efficiently and report your findings clearly."""


EXPLORE_AGENT = AgentDefinition(
    agent_type='Explore',
    when_to_use=(
        'Fast agent specialized for exploring codebases. Use this when you need to '
        'quickly find files by patterns (eg. "src/components/**/*.tsx"), search code '
        'for keywords (eg. "API endpoints"), or answer questions about the codebase '
        '(eg. "how do API endpoints work?"). When calling this agent, specify the '
        'desired thoroughness level: "quick" for basic searches, "medium" for '
        'moderate exploration, or "very thorough" for comprehensive analysis across '
        'multiple locations and naming conventions.'
    ),
    disallowed_tools=[
        AGENT_TOOL_NAME, EXIT_PLAN_MODE_TOOL_NAME, FILE_EDIT_TOOL_NAME,
        FILE_WRITE_TOOL_NAME, NOTEBOOK_EDIT_TOOL_NAME,
    ],
    get_system_prompt=_explore_prompt,
)


# === Plan (planAgent.ts) ===

def _plan_prompt() -> str:
    search_hint = f"{GLOB_TOOL_NAME}, {GREP_TOOL_NAME}, and {FILE_READ_TOOL_NAME}"
    return f"""You are a software architect and planning specialist for CodeMason. Your role is to explore the codebase and design implementation plans.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to explore the codebase and design implementation plans. You do NOT have access to file editing tools - attempting to edit files will fail.

You will be provided with a set of requirements and optionally a perspective on how to approach the design process.

## Your Process

1. **Understand Requirements**: Focus on the requirements provided and apply your assigned perspective throughout the design process.

2. **Explore Thoroughly**:
   - Read any files provided to you in the initial prompt
   - Find existing patterns and conventions using {search_hint}
   - Understand the current architecture
   - Identify similar features as reference
   - Trace through relevant code paths
   - Use {BASH_TOOL_NAME} ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
   - NEVER use {BASH_TOOL_NAME} for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification

3. **Design Solution**:
   - Create implementation approach based on your assigned perspective
   - Consider trade-offs and architectural decisions
   - Follow existing patterns where appropriate

4. **Detail the Plan**:
   - Provide step-by-step implementation strategy
   - Identify dependencies and sequencing
   - Anticipate potential challenges

## Required Output

End your response with:

### Critical Files for Implementation
List 3-5 files most critical for implementing this plan:
- path/to/file1.ts
- path/to/file2.ts
- path/to/file3.ts

REMEMBER: You can ONLY explore and plan. You CANNOT and MUST NOT write, edit, or modify any files. You do NOT have access to file editing tools."""


PLAN_AGENT = AgentDefinition(
    agent_type='Plan',
    when_to_use=(
        'Software architect agent for designing implementation plans. Use this when '
        'you need to plan the implementation strategy for a task. Returns step-by-step '
        'plans, identifies critical files, and considers architectural trade-offs.'
    ),
    tools=EXPLORE_AGENT.tools,
    disallowed_tools=[
        AGENT_TOOL_NAME, EXIT_PLAN_MODE_TOOL_NAME, FILE_EDIT_TOOL_NAME,
        FILE_WRITE_TOOL_NAME, NOTEBOOK_EDIT_TOOL_NAME,
    ],
    get_system_prompt=_plan_prompt,
)


def get_built_in_agents() -> List[AgentDefinition]:
    """getBuiltInAgents (external portable set)."""
    return [GENERAL_PURPOSE_AGENT, EXPLORE_AGENT, PLAN_AGENT]


def _load_custom_agents() -> List[AgentDefinition]:
    """Load user + project `agents/*.md` as custom sub-agent definitions
    (loadAgentsDir.ts). Frontmatter: name -> agent_type (filename stem fallback),
    description -> when_to_use, tools (comma list or YAML list) -> allowlist;
    the markdown body is the agent's system prompt. First-wins by
    agent_type in user -> project order. The global disallow set still applies via
    resolve_agent_tools, so a custom agent can't grant itself Agent/AskUserQuestion.
    """
    import os
    from pathlib import Path
    from ..utils.skills import parse_frontmatter
    from ..utils.state import get_cwd

    # Our own directories first. Until now ONLY the other product's were read,
    # so a codemason user had nowhere of their own to put a sub-agent — and one
    # dropped in another tool's directory was picked up whether they meant it or
    # not. Importing that layout is now opt-in, matching skill discovery.
    from ..utils.config import CONFIG_DIR, get_config
    cwd = Path(get_cwd())
    dirs = [CONFIG_DIR / 'agents', cwd / '.codemason' / 'agents']
    if bool(getattr(get_config(), 'import_claude_dirs', False)):
        dirs += [Path(os.path.expanduser('~')) / '.claude' / 'agents',
                 cwd / '.claude' / 'agents']
    seen = set()
    out: List[AgentDefinition] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob('*.md')):
            try:
                text = f.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            fm, body = parse_frontmatter(text)
            name = str(fm.get('name') or f.stem or '').strip()
            if not name or name in seen:
                continue
            seen.add(name)
            tv = fm.get('tools')
            if isinstance(tv, str):
                tools_list = [t.strip() for t in tv.split(',') if t.strip()] or None
            elif isinstance(tv, list):
                tools_list = [str(t).strip() for t in tv if str(t).strip()] or None
            else:
                tools_list = None
            out.append(AgentDefinition(
                agent_type=name,
                when_to_use=str(fm.get('description') or '').strip(),
                get_system_prompt=(lambda b=body: b),
                tools=tools_list,
                source='project',
            ))
    return out


def get_all_agents() -> List[AgentDefinition]:
    """Built-in agents plus custom `agents/*.md` definitions."""
    return get_built_in_agents() + _load_custom_agents()


def get_agent_definition(subagent_type: Optional[str]) -> AgentDefinition:
    """Resolve a subagent_type to its definition. An omitted type defaults to
    general-purpose; an explicit but UNKNOWN type is an error (AgentTool.tsx:353
    throws "Agent type '...' not found"), never a silent fall-back to
    general-purpose — running the wrong agent with the wrong tool policy/prompt
    would mask the model's mistake."""
    if not subagent_type:
        return GENERAL_PURPOSE_AGENT
    agents = get_all_agents()
    for a in agents:
        if a.agent_type == subagent_type:
            return a
    available = ', '.join(a.agent_type for a in agents)
    raise ValueError(
        f"Agent type '{subagent_type}' not found. Available agents: {available}"
    )


def resolve_agent_tools(
    agent_def: AgentDefinition,
    available_tools: List[Type[Tool]],
) -> List[Type[Tool]]:
    """Port of resolveAgentTools: ALL_AGENT_DISALLOWED filter -> per-agent
    disallowed -> allowlist (['*']/None = all remaining)."""
    # filterToolsForAgent: drop globally-disallowed (recursion guard).
    filtered = [t for t in available_tools if t.name not in ALL_AGENT_DISALLOWED_TOOLS]
    # Per-agent denylist.
    disallowed = set(agent_def.disallowed_tools or [])
    allowed = [t for t in filtered if t.name not in disallowed]
    # Allowlist.
    if agent_def.tools is None or (len(agent_def.tools) == 1 and agent_def.tools[0] == '*'):
        return allowed
    allow_set = set(agent_def.tools)
    return [t for t in allowed if t.name in allow_set]


# === Agent tool description with the agent-types list (prompt.ts getPrompt) ===

def _tools_description(agent_def: AgentDefinition) -> str:
    """Port of getToolsDescription (prompt.ts:15). Note '*' is treated as a
    wildcard (no allowlist) here, unlike the original which has no '*' handling;
    codemason uses ['*'] to mean 'all tools' (see resolve_agent_tools)."""
    tools = agent_def.tools
    has_allowlist = bool(tools) and not (len(tools) == 1 and tools[0] == '*')
    has_denylist = bool(agent_def.disallowed_tools)

    if has_allowlist and has_denylist:
        # Both defined: filter allowlist by denylist to match runtime behavior.
        deny_set = set(agent_def.disallowed_tools)
        effective_tools = [t for t in tools if t not in deny_set]
        if not effective_tools:
            return 'None'
        return ', '.join(effective_tools)
    if has_allowlist:
        return ', '.join(tools)
    if has_denylist:
        return 'All tools except ' + ', '.join(agent_def.disallowed_tools)
    return 'All tools'


def format_agent_line(agent_def: AgentDefinition) -> str:
    return f"- {agent_def.agent_type}: {agent_def.when_to_use} (Tools: {_tools_description(agent_def)})"


def build_agent_description() -> str:
    """Faithful getPrompt() (non-fork, non-coordinator), scoped to codemason's
    Agent params (prompt + subagent_type + description). Keeps the user-language
    rule. Worktree isolation is still not advertised (not implemented) — a
    bullet describing a capability the tool does not have is a lie to the model,
    so it is dropped while missing and restored verbatim when the local
    implementation lands. Background and SendMessage ARE implemented, so their
    bullets are back; the resume one names the agent's ID rather than its name,
    because the Agent tool's `name` parameter is not ported yet."""
    agent_list = '\n'.join(format_agent_line(a) for a in get_all_agents())
    return f"""Launch a new agent to handle complex, multi-step tasks autonomously.

The {AGENT_TOOL_NAME} tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.

Available agent types and the tools they have access to:
{agent_list}

When using the {AGENT_TOOL_NAME} tool, specify a subagent_type parameter to select which agent type to use. If omitted, the general-purpose agent is used.

When NOT to use the {AGENT_TOOL_NAME} tool:
- If you want to read a specific file path, use the {FILE_READ_TOOL_NAME} tool or the {GLOB_TOOL_NAME} tool instead, to find the match more quickly
- If you are searching for a specific class definition like "class Foo", use the {GLOB_TOOL_NAME} tool instead, to find the match more quickly
- If you are searching for code within a specific file or set of 2-3 files, use the {FILE_READ_TOOL_NAME} tool instead of the {AGENT_TOOL_NAME} tool, to find the match more quickly
- Other tasks that are not related to the agent descriptions above

Usage notes:
- Always include a short `description` (3-5 words) summarizing what the agent will do.
- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses.
- IMPORTANT: Write the agent prompt in the language the user is writing in — whatever that language is. Never default to English because the task is technical; the sub-agent reports back to the same person.
- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
- You can optionally run agents in the background using the run_in_background parameter. When an agent runs in the background, you will be automatically notified when it completes — do NOT sleep, poll, or proactively check on its progress. Continue with other work or respond to the user instead.
- **Foreground vs background**: Use foreground (default) when you need the agent's results before you can proceed — e.g., research agents whose findings inform your next steps. Use background when you have genuinely independent work to do in parallel.
- To send a message to a running agent, use {SEND_MESSAGE_TOOL_NAME} with the agent's ID as the `to` field. The message is delivered at the agent's next tool round. Each Agent invocation starts fresh — provide a complete task description.
- The agent's outputs should generally be trusted.
- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent.
- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple {AGENT_TOOL_NAME} tool use content blocks. For example, if you need to launch both a build-validator agent and a test-runner agent in parallel, send a single message with both tool calls.

## Writing the prompt

Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so ("report in under 200 words").
- Lookups: hand over the exact command. Investigations: hand over the question — prescribed steps become dead weight when the premise is wrong.

Terse command-style prompts produce shallow, generic work.

**Never delegate understanding.** Don't write "based on your findings, fix the bug" or "based on the research, implement it." Those phrases push synthesis onto the agent instead of doing it yourself. Write prompts that prove you understood: include file paths, line numbers, what specifically to change.

Example usage:

<example_agent_descriptions>
"test-runner": use this agent after you are done writing code to run tests
"greeting-responder": use this agent to respond to user greetings with a friendly joke
</example_agent_descriptions>

<example>
user: "Please write a function that checks if a number is prime"
assistant: I'm going to use the {FILE_WRITE_TOOL_NAME} tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since a significant piece of code was written and the task was completed, now use the test-runner agent to run the tests
</commentary>
assistant: Uses the {AGENT_TOOL_NAME} tool to launch the test-runner agent
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the {AGENT_TOOL_NAME} tool to launch the greeting-responder agent"
</example>
"""
