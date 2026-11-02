"""
Tools for CodeMason — aligned to Claude Code's getAllBaseTools().

Only tools that exist in Claude Code are registered. Tools the old port carried
that Claude Code does not have (List/Think/Architect/ReadNotebook/Memory) were
deleted. Remaining Claude Code default tools not yet ported (SendMessage / agent
resume, etc.) are tracked for porting.

SessionSearch is the one deliberate addition: it is claude.ai's
conversation_search brought local — the past chats are JSONL transcripts on
this disk, so the capability exists here that the CLI original could not have.
"""

import os
import re
from typing import List, Type
from ..constants.prompts import ASK_USER_QUESTION_TOOL_NAME

from .base import Tool, ToolInputSchema, ToolResult, ToolResultSuccess
from .bash_tool import BashTool
from .file_read_tool import FileReadTool
from .file_write_tool import FileWriteTool
from .file_edit_tool import FileEditTool
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .agent_tool import AgentTool
from .ask_user_question_tool import AskUserQuestionTool
from .todo_write_tool import TodoWriteTool
from .notebook_edit_tool import NotebookEditTool
from .web_fetch_tool import WebFetchTool
from .web_search_tool import WebSearchTool
from .plan_mode_tools import EnterPlanModeTool, ExitPlanModeTool
from .session_search_tool import SessionSearchTool
from .skill_tool import SkillTool
from .task_output_tool import TaskOutputTool
from .send_message_tool import SendMessageTool
from .task_stop_tool import TaskStopTool
from .powershell_tool.tool import PowerShellTool


# All available tools. PowerShellTool is Windows-only and
# self-gates via is_enabled() (get_all_tools filters on it).
ALL_TOOLS: List[Type[Tool]] = [
    AgentTool,
    BashTool,
    PowerShellTool,
    GlobTool,
    GrepTool,
    FileReadTool,
    FileEditTool,
    FileWriteTool,
    TodoWriteTool,
    AskUserQuestionTool,
    NotebookEditTool,
    WebFetchTool,
    WebSearchTool,
    SessionSearchTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
    SkillTool,
    TaskOutputTool,
    SendMessageTool,
    TaskStopTool,
]

DISABLED_TOOL_NAMES: set = set()

# ── 도구 제한 ────────────────────────────────────────────────
#
# 읽기만 하면 되는 쪽(로그 분석기, 리뷰어)이 모델더러 Write 나 WebFetch 를
# 부르지 말라고 부탁할 이유가 없다. 아예 목록에 넣지 않으면 부를 수가 없다.
# 프롬프트에 묻어두는 것보다 명령행에 드러나 있는 편이 감사도 쉽다.
#
# --allowed-tools / --disallowed-tools 로 켜거나, 환경변수
# CODEMASON_ALLOWED_TOOLS / CODEMASON_DISALLOWED_TOOLS 로 건다.
# 부모 프로세스가 argv 를 건드리지 않고도 묶을 수 있게 둘 다 둔다.
_ALLOWED_TOOL_NAMES: set = set()
_DISALLOWED_TOOL_NAMES: set = set()
_TOOL_FILTER_SET = False


def _split_names(raw: str) -> set:
    return {x.strip() for x in re.split(r"[,\s]+", raw or "") if x.strip()}


def set_tool_filter(allowed: str = "", disallowed: str = "") -> None:
    """도구 제한을 건다. 처음 한 번만 먹는다 —
    나중에 불린 쪽이 범위를 넓히지 못하게 한다."""
    global _ALLOWED_TOOL_NAMES, _DISALLOWED_TOOL_NAMES, _TOOL_FILTER_SET
    if _TOOL_FILTER_SET:
        return
    _ALLOWED_TOOL_NAMES = _split_names(allowed)
    _DISALLOWED_TOOL_NAMES = _split_names(disallowed)
    _TOOL_FILTER_SET = True


def _tool_filter() -> tuple:
    if _TOOL_FILTER_SET:
        return _ALLOWED_TOOL_NAMES, _DISALLOWED_TOOL_NAMES
    # 명령행으로 안 걸었으면 환경변수를 본다.
    return (_split_names(os.getenv("CODEMASON_ALLOWED_TOOLS", "")),
            _split_names(os.getenv("CODEMASON_DISALLOWED_TOOLS", "")))


def _tool_permitted(name: str, allowed: set, disallowed: set) -> bool:
    if name in disallowed:
        return False
    return (not allowed) or (name in allowed)


def _mcp_tools() -> List[Type[Tool]]:
    """Tools discovered from configured MCP servers (empty if none / SDK absent).

    Discovery is cached in tools.mcp_tool and never raises out here, so a
    misconfigured or unreachable MCP server can't stop CodeMason from starting.
    """
    try:
        from .mcp_tool import build_mcp_tools
        return build_mcp_tools()
    except Exception:  # noqa: BLE001
        return []


async def get_all_tools() -> List[Type[Tool]]:
    """Get all available tools (native + any MCP-server tools)."""
    allowed, disallowed = _tool_filter()
    enabled_tools = []
    for tool in ALL_TOOLS:
        if tool.name in DISABLED_TOOL_NAMES:
            continue
        if not _tool_permitted(tool.name, allowed, disallowed):
            continue
        if await tool.is_enabled():
            enabled_tools.append(tool)
    for tool in _mcp_tools():
        if tool.name in DISABLED_TOOL_NAMES:
            continue
        if not _tool_permitted(tool.name, allowed, disallowed):
            continue
        if await tool.is_enabled():
            enabled_tools.append(tool)
    return enabled_tools


async def get_read_only_tools() -> List[Type[Tool]]:
    """Get read-only tools only"""
    enabled_tools = []
    for tool in ALL_TOOLS:
        if tool.name in DISABLED_TOOL_NAMES:
            continue
        if await tool.is_enabled() and tool.is_read_only():
            enabled_tools.append(tool)
    return enabled_tools


async def get_agent_tools() -> List[Type[Tool]]:
    """Tool pool available to sub-agents (no recursive Agent tool, no asking).

    Faithful to resolveAgentTools (agentToolUtils.ts): start from the FULL
    enabled tool pool — write capability is gated at CALL TIME by the permission
    flow (can_use_tool / dangerously_skip_permissions), NOT by pre-filtering the
    pool to read-only. resolve_agent_tools() then applies the agent definition's
    allow/deny policy. (The old read-only pre-filter structurally prevented the
    general-purpose agent from ever implementing anything.)

    AskUserQuestion is withheld on top of that: a sub-agent has no user on the
    other end. Its local half reads a stdin nobody is typing into, and its
    remote half puts the question on the phone and waits out remote_ask_timeout
    (1800s by default) before giving up — half an hour of a parent turn spent on
    a question whose answer was never going to arrive. The agent is supposed to
    come back with a report, so an unanswerable question is not a thing it
    should be able to reach for: it should decide, or say what it could not
    decide, and let the caller ask.
    """
    base_tools = await get_all_tools()
    skip = {'Agent', ASK_USER_QUESTION_TOOL_NAME}
    return [t for t in base_tools if t.name not in skip]


def get_tool_by_name(name: str) -> Type[Tool]:
    """Get a tool by its name or one of its aliases (toolMatchesName)."""
    for tool in ALL_TOOLS:
        if tool.matches_name(name):
            return tool
    for tool in _mcp_tools():
        if tool.matches_name(name):
            return tool
    raise ValueError(f"Unknown tool: {name}")


__all__ = [
    'Tool',
    'ToolInputSchema',
    'ToolResult',
    'ToolResultSuccess',
    'BashTool',
    'FileReadTool',
    'FileWriteTool',
    'FileEditTool',
    'GlobTool',
    'GrepTool',
    'AgentTool',
    'AskUserQuestionTool',
    'TodoWriteTool',
    'NotebookEditTool',
    'WebFetchTool',
    'WebSearchTool',
    'SessionSearchTool',
    'EnterPlanModeTool',
    'ExitPlanModeTool',
    'SkillTool',
    'TaskOutputTool',
    'SendMessageTool',
    'TaskStopTool',
    'ALL_TOOLS',
    'get_all_tools',
    'get_read_only_tools',
    'get_agent_tools',
    'get_tool_by_name',
]
