"""
``codemason mcp`` management subcommand.

Intercepted at the top of cli.main() (before argparse, which owns the positional
``prompt``). Edits ~/.codemason/mcp.json — the same ``mcpServers`` map the
runtime client (utils/mcp_client) reads — and can live-probe a server.

    codemason mcp list
    codemason mcp add <name> --command npx --args "-y,firecrawl-mcp" --env K=V --env K2=V2
    codemason mcp add <name> --url https://host/mcp [--transport http|sse] --header "Authorization: Bearer .."
    codemason mcp remove <name>
    codemason mcp test <name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

from .config import CONFIG_DIR

_MCP_JSON = CONFIG_DIR / "mcp.json"


def _load() -> dict:
    if _MCP_JSON.is_file():
        try:
            return json.loads(_MCP_JSON.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            print(f"error: could not read {_MCP_JSON}: {exc}", file=sys.stderr)
            sys.exit(1)
    return {}


def _edit(mutate) -> bool:
    """Apply `mutate` to mcp.json as a single step, atomically.

    A shared document, and the edits are per-SERVER: `mcp add a` and `mcp remove
    b` in two terminals both read the whole file and both write it back, so the
    later one reverted the other's server outright. Reading inside the lock —
    rather than passing in a copy loaded earlier — is what makes each edit see
    the other's. The plain write it replaces also truncated first, so a crash or
    a concurrent reader could see an mcp.json with no servers at all.

    `mutate` gets the document (always a dict) and returns it, or None for "no
    change". Returns whether the file was written.
    """
    from .config import lock_path
    from .file import update_json_document
    _MCP_JSON.parent.mkdir(parents=True, exist_ok=True)

    def apply(current):
        return mutate(current if isinstance(current, dict) else {})

    return update_json_document(_MCP_JSON, lock_path(_MCP_JSON.name), apply,
                                trailing_newline=True)


def _servers(cfg: dict) -> dict:
    return cfg.setdefault("mcpServers", {})


def _cmd_list(args: List[str]) -> int:
    from . import mcp_client
    if not mcp_client.is_available():
        print("MCP SDK not installed (pip install 'mcp>=1.9').", file=sys.stderr)
    probe = "--no-probe" not in args
    if probe and mcp_client.is_available() and mcp_client.load_server_configs():
        # Actively connect so the status reflects reality (tool counts, errors),
        # not just what a prior in-process discovery happened to populate.
        print("Probing servers ...", file=sys.stderr)
        mcp_client.discover(connect_timeout=60)
    rows = mcp_client.server_status()
    if not rows:
        print("No MCP servers configured. Add one with: codemason mcp add <name> ...")
        return 0
    for r in rows:
        flags = []
        if r["disabled"]:
            flags.append("disabled")
        state = "disabled" if r["disabled"] else ("connected" if r["connected"] else "not connected")
        line = f"  {r['name']:<20} {r['transport']:<6} {state}"
        if r["tools"]:
            line += f"  ({r['tools']} tools)"
        if r["error"]:
            line += f"  error: {r['error']}"
        print(line)
    return 0


def _parse_kv(items: List[str]) -> dict:
    out = {}
    for it in items:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = v
    return out


def _parse_headers(items: List[str]) -> dict:
    out = {}
    for it in items:
        if ":" in it:
            k, v = it.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _cmd_add(args: List[str]) -> int:
    if not args:
        print("usage: codemason mcp add <name> [--command C --args a,b | --url U] "
              "[--transport http|sse] [--env K=V ...] [--header 'H: v' ...]",
              file=sys.stderr)
        return 2
    name = args[0]
    rest = args[1:]
    command = url = transport = None
    arg_list: List[str] = []
    envs: List[str] = []
    headers: List[str] = []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--command":
            i += 1; command = rest[i]
        elif tok == "--args":
            i += 1; arg_list = [a for a in rest[i].split(",") if a != ""]
        elif tok == "--url":
            i += 1; url = rest[i]
        elif tok == "--transport":
            i += 1; transport = rest[i]
        elif tok == "--env":
            i += 1; envs.append(rest[i])
        elif tok == "--header":
            i += 1; headers.append(rest[i])
        else:
            print(f"unknown option: {tok}", file=sys.stderr); return 2
        i += 1

    entry: dict = {}
    if command:
        entry["command"] = command
        if arg_list:
            entry["args"] = arg_list
        env = _parse_kv(envs)
        if env:
            entry["env"] = env
    elif url:
        entry["type"] = transport or "http"
        entry["url"] = url
        hdr = _parse_headers(headers)
        if hdr:
            entry["headers"] = hdr
    else:
        print("error: provide either --command or --url", file=sys.stderr)
        return 2

    def add(cfg: dict) -> dict:
        _servers(cfg)[name] = entry
        return cfg

    if not _edit(add):
        # _edit refuses to overwrite an mcp.json it could not read, so saying
        # "Added" here would be a lie about a file we left alone.
        print(f"error: {_MCP_JSON} could not be read; nothing was changed",
              file=sys.stderr)
        return 1
    print(f"Added MCP server '{name}' to {_MCP_JSON}")
    print(json.dumps(entry, indent=2))
    print("\nProbe it with: codemason mcp test", name)
    return 0


def _cmd_remove(args: List[str]) -> int:
    if not args:
        print("usage: codemason mcp remove <name>", file=sys.stderr)
        return 2
    name = args[0]
    # "is it there" and "remove it" have to be the same step as the write, or a
    # concurrent edit decides the answer after we looked.
    missing = []

    def remove(cfg: dict):
        servers = _servers(cfg)
        if name not in servers:
            missing.append(name)
            return None
        servers.pop(name)
        return cfg

    wrote = _edit(remove)
    if missing:
        print(f"error: no such MCP server: {name}", file=sys.stderr)
        return 1
    if not wrote:
        print(f"error: {_MCP_JSON} could not be read; nothing was changed",
              file=sys.stderr)
        return 1
    print(f"Removed MCP server '{name}'")
    return 0


def _cmd_test(args: List[str]) -> int:
    if not args:
        print("usage: codemason mcp test <name>", file=sys.stderr)
        return 2
    name = args[0]
    from . import mcp_client
    cfg = _load()
    servers = _servers(cfg)
    if name not in servers:
        print(f"error: no such MCP server: {name}", file=sys.stderr)
        return 1
    # Probe only the requested server by temporarily narrowing the config view.
    print(f"Connecting to '{name}' ...")
    found = [(s, t) for (s, t) in mcp_client.discover(connect_timeout=90) if s == name]
    st = next((r for r in mcp_client.server_status() if r["name"] == name), None)
    if st and st.get("error"):
        print(f"FAILED: {st['error']}", file=sys.stderr)
        return 1
    print(f"OK — {len(found)} tools:")
    for _s, t in found:
        desc = (getattr(t, "description", "") or "").strip().splitlines()
        print(f"  {t.name:<32} {desc[0] if desc else ''}")
    return 0


def handle_mcp_cli(argv: List[str]) -> int:
    """Dispatch ``codemason mcp <sub> ...``. ``argv`` is sys.argv[2:]."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("codemason mcp — manage MCP servers\n\n"
              "  list                     list configured servers and status\n"
              "  add <name> ...           add a server (--command/--args/--env or --url/--header)\n"
              "  remove <name>            remove a server\n"
              "  test <name>              connect and list the server's tools")
        return 0
    sub, rest = argv[0], argv[1:]
    dispatch = {
        "list": _cmd_list, "ls": _cmd_list,
        "add": _cmd_add,
        "remove": _cmd_remove, "rm": _cmd_remove,
        "test": _cmd_test,
    }
    fn = dispatch.get(sub)
    if fn is None:
        print(f"unknown mcp subcommand: {sub}", file=sys.stderr)
        return 2
    return fn(rest)
