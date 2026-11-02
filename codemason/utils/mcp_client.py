"""
Minimal MCP (Model Context Protocol) client for CodeMason.

CodeMason mirrors Claude Code's *native* tool set (see tools/__init__.py) and
had no way to talk to external MCP servers. This module adds one: it reads the
same ``mcpServers`` configuration Claude Code uses, connects to each server over
stdio / Streamable-HTTP / SSE, discovers its tools, and exposes them so
tools/mcp_tool.py can wrap each as a CodeMason ``Tool``.

Design:

  * All MCP sessions live on ONE dedicated background asyncio loop running in a
    daemon thread. The MCP Python SDK builds its stdio/HTTP transports with
    anyio task groups that must be entered and exited on the same task, so a
    long-lived owner task per server keeps the ``ClientSession`` alive between
    calls. CodeMason's own tool calls happen on a *different* loop (and tool
    discovery runs inside a throwaway ``asyncio.run`` loop), so every hop into
    MCP land goes through ``run_coroutine_threadsafe`` + ``asyncio.wrap_future``
    — loop-agnostic, and the sessions outlive the caller's loop.

  * Discovery and per-server connection failures are non-fatal: a broken server
    is logged and skipped so it can never block CodeMason from starting.

Config sources (later wins on name collision), each an object whose
``mcpServers`` maps a server name to its entry:

    ~/.codemason/mcp.json           →  {"mcpServers": {...}}   (Claude Code .mcp.json shape)
    ~/.codemason/settings.json      →  {"mcpServers": {...}}
    <cwd>/.mcp.json                 →  {"mcpServers": {...}}   (project-scoped)

Server entry shapes (Claude Code compatible):

    stdio:  {"command": "npx", "args": ["-y", "pkg"], "env": {"K": "v"}}
    http :  {"type": "http", "url": "https://host/mcp", "headers": {"Authorization": "Bearer .."}}
    sse  :  {"type": "sse",  "url": "https://host/sse", "headers": {...}}

Transport is inferred when ``type`` is absent: ``command`` → stdio, ``url`` →
http. ``"disabled": true`` skips an entry.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import CONFIG_DIR, SETTINGS_FILE
from .log import log_error

try:  # SDK is an optional dependency; absence just means "no MCP".
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
except Exception:  # noqa: BLE001
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore
    _MCP_AVAILABLE = False

# Streamable-HTTP client moved / was renamed across SDK versions
# (mcp<2: ``streamablehttp_client``; mcp>=2: ``streamable_http_client``).
_http_client = None
if _MCP_AVAILABLE:
    try:
        from mcp.client.streamable_http import streamable_http_client as _http_client  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            from mcp.client.streamable_http import streamablehttp_client as _http_client  # type: ignore
        except Exception:  # noqa: BLE001
            _http_client = None
_sse_client = None
if _MCP_AVAILABLE:
    try:
        from mcp.client.sse import sse_client as _sse_client  # type: ignore
    except Exception:  # noqa: BLE001
        _sse_client = None


# ── background loop ─────────────────────────────────────────────────────────
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Start (once) the daemon thread that owns every MCP session's loop."""
    global _loop, _thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(
            target=_loop.run_forever, name="codemason-mcp-loop", daemon=True
        )
        _thread.start()
        return _loop


async def _await_on_loop(coro, timeout: Optional[float]):
    """Run ``coro`` on the MCP loop, await its result from the *caller's* loop.

    ``asyncio.wrap_future`` binds the returned awaitable to whatever loop is
    running when it's awaited, so this works from CodeMason's main loop and from
    the throwaway discovery loop alike.
    """
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        fut.cancel()
        raise


# ── per-server connection ───────────────────────────────────────────────────
class _Server:
    """A live MCP session plus its discovered tools.

    The transport + session are opened inside ``_run`` and held open by parking
    on ``_stop`` until shutdown; that owner task is the only one that enters and
    exits the anyio context managers, per SDK requirements.
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.session: Optional[Any] = None
        self.tools: List[Any] = []
        self.error: Optional[str] = None
        self._ready = asyncio.Event()   # created on the MCP loop
        self._stop = asyncio.Event()

    def _transport_ctx(self):
        cfg = self.config
        ttype = (cfg.get("type") or ("stdio" if cfg.get("command") else "http")).lower()
        if ttype == "stdio":
            if stdio_client is None:
                raise RuntimeError("stdio transport unavailable")
            # Inherit the environment MINUS this program's own secrets. An MCP
            # server is third-party code; passing it the model API key, the body
            # AES key and the bot token gives every one of them the ability to
            # impersonate the user to the model server, decrypt captured
            # traffic, or take over the remote channel. Nothing in the MCP
            # protocol needs them, and an entry that genuinely wants a
            # credential still gets it through its own explicit `env` below.
            _SECRET_ENV_PREFIXES = ("CODEMASON_", "LLAMACPP_", "TELEGRAM_")
            _SECRET_ENV_SUFFIXES = ("_API_KEY", "_AES_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
            env = {
                k: v for k, v in os.environ.items()
                if not (
                    (k.startswith(_SECRET_ENV_PREFIXES)
                     and k.endswith(_SECRET_ENV_SUFFIXES))
                    or k.endswith(("_API_KEY", "_AES_KEY"))
                    or k in ("TELEGRAM_BOT_TOKEN",)
                )
            }
            env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
            params = StdioServerParameters(
                command=cfg["command"],
                args=list(cfg.get("args") or []),
                env=env,
                cwd=cfg.get("cwd") or None,
            )
            return stdio_client(params)
        if ttype in ("http", "streamable-http", "streamable_http"):
            if _http_client is None:
                raise RuntimeError("http transport unavailable in this mcp SDK")
            return _http_client(cfg["url"], headers=cfg.get("headers") or None)
        if ttype == "sse":
            if _sse_client is None:
                raise RuntimeError("sse transport unavailable in this mcp SDK")
            return _sse_client(cfg["url"], headers=cfg.get("headers") or None)
        raise ValueError(f"unknown MCP transport type: {ttype!r}")

    async def _run(self):
        try:
            async with self._transport_ctx() as streams:
                # stdio/sse yield (read, write); streamable-http yields a 3-tuple
                # (read, write, extra). Take the first two either way.
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    result = await session.list_tools()
                    self.tools = list(getattr(result, "tools", []) or [])
                    self._ready.set()
                    await self._stop.wait()  # hold the session open
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self._ready.set()  # unblock the waiter with an error recorded

    async def call(self, tool_name: str, arguments: dict):
        if self.session is None:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")
        return await self.session.call_tool(tool_name, arguments or {})


# name -> _Server (populated on the MCP loop)
_servers: Dict[str, _Server] = {}
_discovered = False


# ── config loading ──────────────────────────────────────────────────────────
def _read_json(path: Path) -> dict:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        log_error(f"MCP: failed to read {path}: {exc}")
    return {}


def load_server_configs() -> Dict[str, dict]:
    """Merge ``mcpServers`` from all config sources (later source wins).

    The project-local ``.mcp.json`` is the exception to "later wins", and is
    opt-in. discover() runs at tool-registry construction — before the user has
    typed anything — and spawns each entry's ``command``. Reading that file from
    whatever directory codemason was started in therefore made `cd` into a
    cloned repo enough to execute code from it, with the full parent environment
    (including the model API key) inherited. It also sat at the HIGHEST
    precedence, so a repo could reuse a server name from the user's own config
    and silently re-point it, sending the user's bearer token to its own host.

    Now: only read when CODEMASON_TRUST_PROJECT_MCP is set, and never allowed to
    replace a server the user configured themselves.
    """
    merged: Dict[str, dict] = {}
    for src in (_read_json(CONFIG_DIR / "mcp.json"), _read_json(Path(SETTINGS_FILE))):
        servers = src.get("mcpServers") or src.get("mcp_servers") or {}
        if isinstance(servers, dict):
            for name, entry in servers.items():
                if isinstance(entry, dict):
                    merged[name] = entry

    project_file = Path.cwd() / ".mcp.json"
    if not project_file.is_file():
        return merged
    if os.environ.get("CODEMASON_TRUST_PROJECT_MCP", "").strip() not in ("1", "true", "yes"):
        print(
            f"  Skipped {project_file} — a project MCP config runs its commands at "
            f"startup. Set CODEMASON_TRUST_PROJECT_MCP=1 to load it."
        )
        return merged
    project = _read_json(project_file)
    servers = project.get("mcpServers") or project.get("mcp_servers") or {}
    if isinstance(servers, dict):
        for name, entry in servers.items():
            if isinstance(entry, dict) and name not in merged:
                merged[name] = entry
    return merged


# ── public API ──────────────────────────────────────────────────────────────
def discover(connect_timeout: float = 20.0) -> List[Tuple[str, Any]]:
    """Connect to every enabled server and return ``(server_name, mcp_tool)``.

    Safe to call from inside an ``asyncio.run`` loop (tools/__init__.get_all_tools
    is async). Runs the actual connecting on the background MCP loop and blocks
    the *calling thread* only until each server is ready or times out. Idempotent.
    """
    global _discovered
    if not _MCP_AVAILABLE:
        return []
    configs = load_server_configs()
    if not configs:
        _discovered = True
        return []

    _ensure_loop()

    async def _spawn(name: str, cfg: dict) -> _Server:
        srv = _Server(name, cfg)
        _servers[name] = srv
        # launch the long-lived owner task on THIS (the MCP) loop
        asyncio.get_running_loop().create_task(srv._run())
        try:
            await asyncio.wait_for(srv._ready.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError:
            srv.error = f"connect timed out after {connect_timeout:.0f}s"
        return srv

    out: List[Tuple[str, Any]] = []
    loop = _ensure_loop()
    for name, cfg in configs.items():
        if cfg.get("disabled") or cfg.get("enabled") is False:
            continue
        try:
            fut = asyncio.run_coroutine_threadsafe(_spawn(name, cfg), loop)
            srv = fut.result(timeout=connect_timeout + 10)
        except Exception as exc:  # noqa: BLE001
            log_error(f"MCP: server '{name}' failed to start: {exc}")
            continue
        if srv.error:
            log_error(f"MCP: server '{name}' unavailable ({srv.error})")
            continue
        for tool in srv.tools:
            out.append((name, tool))
    _discovered = True
    return out


async def call_tool(server_name: str, tool_name: str, arguments: dict,
                    timeout: float = 120.0):
    """Invoke a tool on a connected server. Awaitable from any loop."""
    srv = _servers.get(server_name)
    if srv is None:
        raise RuntimeError(f"MCP server '{server_name}' is not connected")
    return await _await_on_loop(srv.call(tool_name, arguments), timeout)


def is_available() -> bool:
    return _MCP_AVAILABLE


def server_status() -> List[dict]:
    """Lightweight status snapshot for a `mcp` CLI / diagnostics."""
    rows = []
    for name, cfg in load_server_configs().items():
        srv = _servers.get(name)
        rows.append({
            "name": name,
            "transport": (cfg.get("type") or ("stdio" if cfg.get("command") else "http")),
            "connected": bool(srv and srv.session is not None),
            "tools": len(srv.tools) if srv else 0,
            "error": (srv.error if srv else None),
            "disabled": bool(cfg.get("disabled") or cfg.get("enabled") is False),
        })
    return rows
