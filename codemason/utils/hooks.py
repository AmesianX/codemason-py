"""
Hooks — based on the Stop-hook path from src/utils/hooks.ts in Claude Code
+ src/query/stopHooks.ts.

What this is: the mechanism that lets a user-configured "Stop" hook keep the
model working past a natural turn-end. When the model ends a turn with no tool
call (query.ts:1267), the configured Stop hooks run; if one
"blocks" (exit code 2, or stdout {"decision":"block","reason":...}), that
feedback is injected back into the conversation and the loop CONTINUES — the
model resumes instead of stopping. A hook returning {"continue":false} ends the
turn instead. `stop_hook_active` is passed to the hook and set true on the
continue path so a hook can detect it is already in a stop-hook continuation
and avoid looping forever (the hook owns loop-termination)
documented contract).

SUBSTRATE / SCOPE (NOT ported, deliberately): the original hooks subsystem is
~4000 lines spanning many events (PreToolUse/PostToolUse/UserPromptSubmit/
Notification/SessionStart/…), HTTP- and MCP-transport hooks, elicitation, the
trust/permission UI, analytics, teammate/task hooks, and an Ink progress
display. CodeMason ports ONLY the Stop event with command-type hooks — the
piece relevant to autonomous continuation. The per-hook progress/summary
display (Ink) is dropped; blocking feedback still enters history. Hooks are
read from ~/.codemason/settings.json under ["hooks"]["Stop"], matching Claude
Code's settings.json schema.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import SETTINGS_FILE
from .log import debug_log
from .platform import is_windows as _is_windows
from .windows_shell import (
    shell_argv as _win_shell_argv,
    no_window_kwargs as _no_window_kwargs,
    kill_process_tree as _kill_process_tree,
)


# hooks.ts TOOL_HOOK_EXECUTION_TIMEOUT_MS = 10*60*1000 (600s), in seconds.
# (Was 60 — that killed legitimate 1-10min Stop hooks, e.g. test suites.)
DEFAULT_HOOK_TIMEOUT_SEC = 600


@dataclass
class HookBlockingError:
    """HookBlockingError (hooks.ts:330-ish): a hook's blocking feedback."""
    blocking_error: str
    command: str


def _read_settings() -> Dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with open(SETTINGS_FILE, encoding='utf-8') as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def get_hooks_config(event: str) -> List[Dict[str, Any]]:
    """settings.json["hooks"][event] -> list of matcher entries, each
    {matcher?, hooks: [{type, command, timeout?}]}."""
    hooks = _read_settings().get('hooks') or {}
    entry = hooks.get(event) or []
    return entry if isinstance(entry, list) else []


def has_hook_for_event(event: str) -> bool:
    """hasHookForEvent (hooks.ts:1582) — is any hook configured for this event?"""
    return len(get_hooks_config(event)) > 0


def _matched_command_hooks(event: str) -> List[Dict[str, Any]]:
    """Flatten configured command hooks for an event. Stop hooks carry no match
    query (getMatchingHooks: the Stop case leaves matchQuery undefined, so every
    matcher entry applies), so we run all command-type hooks under the event."""
    out: List[Dict[str, Any]] = []
    for matcher_entry in get_hooks_config(event):
        if not isinstance(matcher_entry, dict):
            continue
        for h in (matcher_entry.get('hooks') or []):
            if isinstance(h, dict) and h.get('type') == 'command' and h.get('command'):
                out.append(h)
    return out


def get_stop_hook_message(blocking_error: str) -> str:
    """getStopHookMessage (hooks.ts:1894)."""
    return f"Stop hook feedback:\n{blocking_error}"


def _process_hook_json(js: Dict[str, Any], command: str) -> Dict[str, Any]:
    """processHookJSONOutput (hooks.ts:518) — Stop event subset. continue:false
    -> preventContinuation (+stopReason); decision:'block' -> blockingError
    (reason); decision:'approve' is a no-op for Stop; any other decision value
    is an error (the original THROWS 'Unknown hook decision type', surfaced as a
    hook failure — here logged and returned as a non-blocking 'error' result, so
    it neither blocks nor prevents continuation)."""
    result: Dict[str, Any] = {}
    # handleSyncJSONOutput (hooks.ts:518): continue === false -> preventContinuation.
    if js.get('continue') is False:
        result['prevent_continuation'] = True
        if js.get('stopReason'):
            result['stop_reason'] = js['stopReason']
    # decision (hooks.ts:528-543).
    decision = js.get('decision')
    if decision is not None:
        if decision == 'block':
            result['blocking_error'] = HookBlockingError(
                blocking_error=js.get('reason') or 'Blocked by hook',
                command=command,
            )
        elif decision != 'approve':
            debug_log('STOP_HOOK_UNKNOWN_DECISION',
                      {'decision': decision, 'command': command})
            result['error'] = (
                f"Unknown hook decision type: {decision}. "
                f"Valid types are: approve, block"
            )
    return result


async def _run_command_hook(
    hook: Dict[str, Any], hook_input: Dict[str, Any]
) -> Dict[str, Any]:
    """Run one command hook: stdin = JSON hook_input. Parse its decision per
    hooks.ts handleSyncJSONOutput (continue:false / decision:'block') plus the
    exit-code-2 blocking convention. Returns a partial result dict with optional
    keys: blocking_error (HookBlockingError), prevent_continuation (bool),
    stop_reason (str), error (str — non-blocking execution failure)."""
    command = hook['command']
    timeout = hook.get('timeout', DEFAULT_HOOK_TIMEOUT_SEC)
    payload = json.dumps(hook_input)

    try:
        # Hook commands are user-authored, usually POSIX-sh one-liners. On native
        # Windows create_subprocess_shell routes through cmd.exe, which breaks
        # them; route through git-bash instead (same as the Bash substrate, and
        # matching the "run hooks via Git Bash, not cmd.exe" fix). POSIX/WSL2
        # keep create_subprocess_shell unchanged.
        _hook_argv = _win_shell_argv(command)
        # no_window_kwargs: suppress the console flash on Windows (#61051) without
        # changing the POSIX process-group behaviour of hooks.
        _spawn_common = dict(
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_no_window_kwargs(),
        )
        if _hook_argv is not None:
            proc = await asyncio.create_subprocess_exec(*_hook_argv, **_spawn_common)
        else:
            proc = await asyncio.create_subprocess_shell(command, **_spawn_common)
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(payload.encode()), timeout=timeout
            )
        except asyncio.TimeoutError:
            # proc is the shell (bash.exe / /bin/sh); the real hook runs as its
            # CHILD. proc.kill() reaps only the shell, orphaning the hook (upstream
            # #77078: a suspended/hung hook survives its timeout for 30-60 min). On
            # Windows taskkill /T kills the whole tree by PID — no process group
            # needed. POSIX keeps proc.kill() (the hook shares our group, so killpg
            # would take CodeMason down too — unchanged, pre-existing behaviour).
            if _is_windows():
                _kill_process_tree(proc)
            else:
                try:
                    proc.kill()
                except Exception:
                    pass
            return {'error': f'Stop hook timed out after {timeout}s: {command}'}
        exit_code = proc.returncode
        stdout = (stdout_b or b'').decode('utf-8', 'replace')
        stderr = (stderr_b or b'').decode('utf-8', 'replace')
    except Exception as e:  # pragma: no cover - spawn failure
        return {'error': f'Stop hook failed to run ({command}): {e}'}

    # Decision precedence mirrors runHook (hooks.ts:2499+): parse stdout JSON
    # FIRST — if a JSON object is present, its decision is honored regardless of
    # exit code; ONLY non-JSON output falls back to the exit-code convention.
    # parseHookOutput (hooks.ts:399): JSON is attempted only when trimmed stdout
    # starts with '{'; a '{' that fails to parse is treated as plain text (falls
    # through), as is anything not starting with '{'.
    trimmed = stdout.strip()
    if trimmed.startswith('{'):
        try:
            js = json.loads(trimmed)
        except json.JSONDecodeError:
            js = None
        if isinstance(js, dict):
            return _process_hook_json(js, command)

    # Non-JSON output -> exit-code convention (hooks.ts:2613+):
    #   exit 2     -> blocking feedback (raw stderr is the message),
    #   other != 0 -> non-blocking error (logged, does not block),
    #   0          -> success / no decision.
    if exit_code == 2:
        return {
            'blocking_error': HookBlockingError(
                blocking_error=f"[{command}]: {stderr or 'No stderr output'}",
                command=command,
            )
        }
    if exit_code not in (0, None):
        debug_log('STOP_HOOK_NONBLOCKING_ERROR', {
            'command': command, 'exit_code': exit_code, 'stderr': stderr[:500],
        })
    return {}


def _last_assistant_text(messages: List[Any], assistant_message: Any) -> Optional[str]:
    """Text of the final assistant message, so hooks can inspect the response
    without reading a transcript file (executeStopHooks: last_assistant_message)."""
    obj = getattr(assistant_message, 'message', None)
    content = getattr(obj, 'content', None) if obj is not None else None
    if content is None:
        return None
    parts = []
    for b in content:
        if (getattr(b, 'type', None) == 'text'):
            parts.append(getattr(b, 'text', '') or '')
    text = '\n'.join(parts).strip()
    return text or None


def _used_tools_since_last_user(messages: List[Any]) -> bool:
    """True iff any tool ran since the last GENUINE user message (i.e. the turn
    did real agentic work rather than a pure conversational reply).

    The engine reports this fact to Stop hooks so a keep-going policy can avoid
    nudging chat turns (a greeting that used no tools shouldn't be pushed to
    "keep working"). Decision stays in the hook; the engine only observes.

    Walk history backward to the boundary user input, counting intermediate
    assistant tool_use blocks. Messages may be dicts (user) or objects
    (assistant), so field access is form-agnostic. Skipped as non-boundary:
    tool_result user messages (part of the chain) and isMeta user messages
    (the max-output recovery nudge / a prior Stop-hook block injection)."""
    def _role(m):
        return m.type if hasattr(m, 'type') else (m.get('type') if isinstance(m, dict) else None)

    def _content(m):
        obj = m.message if hasattr(m, 'message') else (m.get('message') if isinstance(m, dict) else None)
        if obj is None:
            return []
        c = obj.content if hasattr(obj, 'content') else (obj.get('content') if isinstance(obj, dict) else None)
        return c or []

    def _is_meta(m):
        return bool(m.get('isMeta')) if isinstance(m, dict) else bool(getattr(m, 'isMeta', False))

    def _btype(b):
        return b.type if hasattr(b, 'type') else (b.get('type') if isinstance(b, dict) else None)

    for m in reversed(messages):
        types = [_btype(b) for b in _content(m)]
        if _role(m) == 'user':
            if 'tool_result' in types or _is_meta(m):
                continue  # part of the tool chain / injected meta — not a boundary
            return False  # reached the genuine user input with no tool use in between
        if _role(m) == 'assistant' and 'tool_use' in types:
            return True
    return False


async def execute_stop_hooks(
    stop_hook_active: bool,
    last_assistant_text: Optional[str],
    used_tools: bool = False,
) -> List[Dict[str, Any]]:
    """executeStopHooks (hooks.ts:3639) — Stop event, command hooks only. Runs
    all configured Stop hooks concurrently (as the original does) and returns
    their parsed result dicts."""
    if not has_hook_for_event('Stop'):
        return []
    hooks = _matched_command_hooks('Stop')
    if not hooks:
        return []

    # Real session identity + transcript path (hooks.ts payload contract) —
    # session persistence exists now, so hooks can read the conversation.
    from .state import get_session_id
    from .session_storage import get_transcript_path
    hook_input = {
        'hook_event_name': 'Stop',
        'stop_hook_active': stop_hook_active,
        'session_id': get_session_id(),
        'transcript_path': str(get_transcript_path()),
        'cwd': os.getcwd(),
        'last_assistant_message': last_assistant_text,
        # Did this turn do real agentic work? Lets a keep-going hook skip pure
        # conversational replies (greetings/Q&A used no tools -> don't nudge).
        'used_tools': used_tools,
    }
    return list(await asyncio.gather(*[_run_command_hook(h, hook_input) for h in hooks]))


async def handle_stop_hooks(
    messages: List[Any],
    assistant_message: Any,
    tool_use_context: Any,
    stop_hook_active: bool,
) -> Dict[str, Any]:
    """handleStopHooks (query/stopHooks.ts:65) — Stop event subset. Aggregates
    hook results into {blocking_errors: List[user_message], prevent_continuation:
    bool}. preventContinuation wins over blocking (matches the original's early
    returns); blocking errors become isMeta user messages carrying the feedback
    the model resumes from."""
    from .messages import create_user_message

    last_text = _last_assistant_text(messages, assistant_message)
    used_tools = _used_tools_since_last_user(messages)
    results = await execute_stop_hooks(stop_hook_active, last_text, used_tools)

    blocking_errors: List[Any] = []
    prevent_continuation = False
    for r in results:
        if r.get('error'):
            debug_log('STOP_HOOK_ERROR', {'error': r['error']})
            continue
        be = r.get('blocking_error')
        if be is not None:
            user_message = create_user_message(get_stop_hook_message(be.blocking_error))
            user_message['isMeta'] = True  # in history for the model; hidden in UI
            blocking_errors.append(user_message)
        if r.get('prevent_continuation'):
            prevent_continuation = True

    if prevent_continuation:
        return {'blocking_errors': [], 'prevent_continuation': True}
    if blocking_errors:
        return {'blocking_errors': blocking_errors, 'prevent_continuation': False}
    return {'blocking_errors': [], 'prevent_continuation': False}
