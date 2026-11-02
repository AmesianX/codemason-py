"""
Wrap tools discovered from MCP servers as CodeMason ``Tool`` classes.

``build_mcp_tools()`` connects to every configured MCP server (see
utils/mcp_client), lists their tools, and returns one dynamically-created
``Tool`` subclass per remote tool. tools/__init__.get_all_tools() appends these
to the native tool set, so from the model's point of view an MCP tool is just
another tool — named ``mcp__<server>__<tool>`` (the same convention Claude Code
uses), described by the server, its parameters taken verbatim from the server's
input schema, and its call forwarded over the live session.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Type

from .base import Tool, ToolResult, ToolResultSuccess
from ..utils import mcp_client
from ..utils.log import log_error


def _tool_schema(mcp_tool: Any) -> Dict[str, Any]:
    schema = (getattr(mcp_tool, "input_schema", None)
              or getattr(mcp_tool, "inputSchema", None))
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(by_alias=True, exclude_none=True)
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _read_only(mcp_tool: Any) -> bool:
    """Always False: an MCP tool is never treated as read-only.

    readOnlyHint is supplied BY the server, and is_read_only() is what makes
    has_permissions_to_use_tool auto-approve without a prompt and what lets a
    tool run in plan mode. Reading it from the wire hands the gate's input to
    the party being gated — a hostile or supply-chained server declares
    readOnlyHint on a tool that deletes files or POSTs ~/.ssh/id_rsa and it
    executes unprompted for the whole session. The MCP spec calls annotations
    untrusted hints; every native tool hardcodes this value.

    The cost is a prompt for genuinely read-only MCP tools, which is the safe
    direction and is what the Rust port already does (mcp_tool.rs, "Divergence
    2"). In dangerously-skip-permissions mode nothing prompts anyway.
    """
    return False


def _render_result(result: Any) -> str:
    """Flatten an MCP CallToolResult into text for the assistant."""
    parts: List[str] = []
    for block in (getattr(result, "content", None) or []):
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
            continue
        btype = getattr(block, "type", None)
        if btype == "image":
            mime = getattr(block, "mimeType", None) or getattr(block, "mime_type", "")
            parts.append(f"[image content{(' ' + mime) if mime else ''} omitted]")
        elif btype == "resource":
            parts.append(f"[resource: {getattr(block, 'resource', '')}]")
        else:
            parts.append(str(block))
    out = "\n".join(p for p in parts if p)
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        return f"MCP tool error:\n{out}" if out else "MCP tool returned an error."
    return out or "(no output)"


def _make_tool_class(server: str, mcp_tool: Any) -> Type[Tool]:
    # Underscore-prefixed so none of these collide with a name assigned inside
    # the class body — in particular the ``description`` method below, which
    # would otherwise shadow a plain ``description`` free variable and raise
    # NameError when the class-level ``DESCRIPTION`` referenced it.
    _tool_name = mcp_tool.name
    _full_name = f"mcp__{server}__{_tool_name}"
    _desc = (getattr(mcp_tool, "description", None) or f"MCP tool {_tool_name} from {server}").strip()
    _schema = _tool_schema(mcp_tool)
    _ro = _read_only(mcp_tool)

    class _MCPTool(Tool):
        name = _full_name
        input_schema = None  # schema is served via input_json_schema() below
        DESCRIPTION = _desc
        # kept for diagnostics / a future `mcp` CLI
        mcp_server = server
        mcp_tool_name = _tool_name

        @classmethod
        def input_json_schema(cls) -> Dict[str, Any]:
            return _schema

        @classmethod
        async def description(cls, **kwargs) -> str:  # noqa: D401
            return _desc

        @classmethod
        def is_read_only(cls) -> bool:
            return _ro

        @classmethod
        async def is_enabled(cls) -> bool:
            return True

        @classmethod
        def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
            return f"{server}:{_tool_name}"

        @classmethod
        async def call(
            cls, input_data: Dict[str, Any], context: Any,
            can_use_tool: Optional[Callable] = None,
        ) -> AsyncGenerator[ToolResult, None]:
            args = input_data if isinstance(input_data, dict) else {}
            try:
                result = await mcp_client.call_tool(server, _tool_name, args)
            except Exception as exc:  # noqa: BLE001
                msg = f"MCP call to {_full_name} failed: {type(exc).__name__}: {exc}"
                log_error(msg)
                yield ToolResultSuccess(result_for_assistant=msg,
                                        data={"error": str(exc)})
                return
            yield ToolResultSuccess(
                result_for_assistant=_render_result(result),
                data={"server": server, "tool": _tool_name},
            )

    _MCPTool.__name__ = f"MCPTool_{server}_{_tool_name}"
    _MCPTool.__qualname__ = _MCPTool.__name__
    return _MCPTool


_cache: Optional[List[Type[Tool]]] = None


def build_mcp_tools(force: bool = False) -> List[Type[Tool]]:
    """Discover MCP tools once and return their wrapper classes (cached)."""
    global _cache
    if _cache is not None and not force:
        return _cache
    classes: List[Type[Tool]] = []
    try:
        for server, mcp_tool in mcp_client.discover():
            try:
                classes.append(_make_tool_class(server, mcp_tool))
            except Exception as exc:  # noqa: BLE001
                log_error(f"MCP: skipped tool {getattr(mcp_tool, 'name', '?')} "
                          f"from {server}: {exc}")
    except Exception as exc:  # noqa: BLE001
        log_error(f"MCP: discovery failed: {exc}")
    _cache = classes
    return classes
