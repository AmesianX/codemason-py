"""
Non-interactive print mode (`-p` / `--print`) — adapted from cli/print.ts runHeadless.

Implements the SDK OUTPUT CONTRACT faithfully across the three output formats:

  - text        : the final assistant text, plain (default).
  - json        : a single `result` message object (the SDK result schema).
  - stream-json : NDJSON — a `system/init` message, then each assistant/user
                  message as it streams, then the final `result` message.

The message SHAPES (system/init, the assistant/user envelope, and the result
message) mirror utils/messages/systemInit.ts + the SDK result schema in
cli/print.ts. Fields that depend on infrastructure codemason doesn't have are
adapted at the documented substrate boundary, NOT invented:
  - total_cost_usd : 0 / modelUsage {} (no cost tracking without API billing).
  - mcp_servers/plugins/betas/output_style/apiKeySource : omitted/empty
    (those subsystems don't exist locally).
session_id is the persistent session id (utils/state) — the run is recorded to
its transcript JSONL, and `--resume <id>` continues a stored session.
"""

import asyncio
import json
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Type

from .utils.serialization import to_jsonable as _to_jsonable

VALID_OUTPUT_FORMATS = ('text', 'json', 'stream-json')
VALID_INPUT_FORMATS = ('text', 'stream-json')

EMPTY_USAGE = {
    'input_tokens': 0,
    'output_tokens': 0,
    'cache_creation_input_tokens': 0,
    'cache_read_input_tokens': 0,
}


def _emit(obj: Dict[str, Any]) -> None:
    """Write one NDJSON line to stdout (stream-json / json output)."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=_to_jsonable) + '\n')
    sys.stdout.flush()


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _as_dict(obj: Any) -> Any:
    """Best-effort: a codemason message is a dict {type,uuid,message}; tolerate
    dataclass-ish objects too."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'message'):
        return {
            'type': getattr(obj, 'type', None),
            'uuid': str(getattr(obj, 'uuid', '')) or None,
            'message': getattr(obj, 'message'),
        }
    return obj


def build_init_message(
    tools: List[Type],
    model: str,
    skip_permissions: bool,
    session_id: str,
) -> Dict[str, Any]:
    """system/init (buildSystemInitMessage), adapted to the local substrate."""
    from .tools.built_in_agents import get_built_in_agents
    from .utils.skills import model_invocable_skills
    from .utils.state import get_cwd

    try:
        skills = [s.name for s in model_invocable_skills()]
    except Exception:
        skills = []
    return {
        'type': 'system',
        'subtype': 'init',
        'cwd': get_cwd(),
        'session_id': session_id,
        'tools': [t.name for t in tools],
        'mcp_servers': [],
        'model': model,
        'permissionMode': 'bypassPermissions' if skip_permissions else 'default',
        'slash_commands': [],
        'agents': [a.agent_type for a in get_built_in_agents()],
        'skills': skills,
        'uuid': _new_uuid(),
    }


def _wrap_message(msg: Any, msg_type: str, session_id: str) -> Dict[str, Any]:
    """Wrap a streamed assistant/user message in the SDK envelope."""
    d = _as_dict(msg)
    inner = d.get('message') if isinstance(d, dict) else None
    # inner may be a dataclass (APIMessage) — coerce common fields to a dict.
    if inner is not None and not isinstance(inner, dict):
        inner = {
            'role': getattr(inner, 'role', msg_type),
            'content': getattr(inner, 'content', []),
        }
    return {
        'type': msg_type,
        'message': _to_jsonable(inner),
        'parent_tool_use_id': None,
        'session_id': session_id,
        'uuid': (d.get('uuid') if isinstance(d, dict) else None) or _new_uuid(),
    }


def build_result_message(
    *,
    subtype: str,
    result_text: Optional[str],
    num_turns: int,
    usage: Optional[Dict[str, Any]],
    duration_ms: int,
    session_id: str,
    is_error: bool,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The SDK `result` message (cli/print.ts result schema)."""
    msg: Dict[str, Any] = {
        'type': 'result',
        'subtype': subtype,
        'duration_ms': duration_ms,
        'duration_api_ms': 0,
        'is_error': is_error,
        'num_turns': num_turns,
        'session_id': session_id,
        'total_cost_usd': 0,
        'usage': usage or EMPTY_USAGE,
        'modelUsage': {},
        'permission_denials': [],
        'uuid': _new_uuid(),
    }
    if not is_error:
        msg['result'] = result_text or ''
    if errors:
        msg['errors'] = errors
    return msg


def _extract_text_and_usage(msg: Any):
    """Pull text blocks + usage from a codemason assistant message."""
    d = _as_dict(msg)
    inner = d.get('message') if isinstance(d, dict) else None
    if inner is None:
        return '', None
    content = inner.get('content', []) if isinstance(inner, dict) else getattr(inner, 'content', [])
    usage = inner.get('usage') if isinstance(inner, dict) else getattr(inner, 'usage', None)
    parts: List[str] = []
    for block in content or []:
        bt = block.get('type') if isinstance(block, dict) else getattr(block, 'type', None)
        if bt == 'text':
            txt = block.get('text', '') if isinstance(block, dict) else getattr(block, 'text', '')
            if txt:
                parts.append(txt)
    return '\n'.join(parts), usage


def read_prompt_from_input(prompt_arg: str, input_format: str) -> Optional[str]:
    """Resolve the prompt: from the positional arg, else stdin. For
    input_format='stream-json', stdin is NDJSON of SDK user messages — extract
    and join their text content. Returns None when no input is available."""
    if prompt_arg:
        return prompt_arg
    if sys.stdin.isatty():
        return None
    raw = sys.stdin.read()
    if input_format == 'stream-json':
        texts: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = obj.get('message') if isinstance(obj, dict) else None
            content = message.get('content') if isinstance(message, dict) else None
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get('type') == 'text':
                        texts.append(b.get('text', ''))
        joined = '\n'.join(t for t in texts if t).strip()
        return joined or None
    return raw.strip() or None


async def run_print(
    prompt: str,
    tools: List[Type],
    *,
    output_format: str = 'text',
    verbose: bool = False,
    dangerously_skip_permissions: bool = False,
    resume_session_id: Optional[str] = None,
) -> int:
    """Run one non-interactive turn and emit it in the requested format.
    Returns the process exit code (0 ok, 1 error)."""
    from datetime import datetime
    from .query import query
    from .constants.prompts import get_system_prompt
    from .types.query import ToolUseContext
    from .utils.config import get_config
    from .utils.messages import create_user_message
    from .utils.state import get_cwd, get_session_id
    from .utils import session_storage
    from .utils.tokens import resolve_model
    from .constants.prompts import ASK_USER_QUESTION_TOOL_NAME

    # AskUserQuestion is withheld from -p for the same reason it is withheld
    # from sub-agents: nobody is on the other end. The local half reads a stdin
    # that is normally closed, and the remote half puts the question on the
    # phone and waits out remote_ask_timeout (1800s) before giving up. A -p
    # caller wants an answer or a stated inability, not a half-hour pause on a
    # question it cannot see. The interactive REPL keeps the tool.
    tools = [t for t in tools if getattr(t, 'name', '') != ASK_USER_QUESTION_TOOL_NAME]

    # Self-heal a stale/renamed model id (write-back to config.json) before the
    # first request, so -p runs don't die in-stream on "model does not exist".
    resolve_model()
    model = get_config().model_name
    # Resume: restore the stored conversation and keep appending to the same
    # transcript file (adopt seeds the dedup set so nothing is re-written).
    restored: List[Any] = []
    if resume_session_id:
        try:
            restored = session_storage.load_conversation_for_resume(resume_session_id)
            session_storage.adopt_resumed_session(resume_session_id)
        except (FileNotFoundError, ValueError) as e:
            sys.stderr.write(f"Error: --resume {resume_session_id}: {e}\n")
            return 1
    session_id = get_session_id()
    is_stream = output_format == 'stream-json'
    is_json = output_format == 'json'

    if is_stream:
        _emit(build_init_message(tools, model, dangerously_skip_permissions, session_id))

    system_prompt = await get_system_prompt(tools, model)
    context = {'cwd': get_cwd(), 'date': datetime.now().strftime('%Y-%m-%d')}
    # Inject the persistent goal (if any) so -p runs honor it too, like the REPL.
    from .utils.goal import render_goal_context as _render_goal
    _goal = _render_goal()
    if _goal:
        context['active_goal'] = _goal
    tool_use_context = ToolUseContext(
        abort_controller=asyncio.Event(),
        options={
            'dangerously_skip_permissions': dangerously_skip_permissions,
            'tools': tools,
            'verbose': verbose,
            'commands': [],
        },
        message_id=None,
        read_file_state={},
    )

    from .cli import has_permissions_to_use_tool
    messages: List[Any] = [*restored, create_user_message(prompt)]
    session_storage.record_transcript(messages)

    text_parts: List[str] = []
    total_usage: Optional[Dict[str, Any]] = None
    num_turns = 0
    tool_counts: Dict[str, int] = {}
    is_error = False
    errors: List[str] = []
    start = time.time()

    try:
        async for message in query(
            messages=messages,
            system_prompt=system_prompt,
            context=context,
            can_use_tool=has_permissions_to_use_tool,
            tool_use_context=tool_use_context,
        ):
            mtype = message.type if hasattr(message, 'type') else (
                message.get('type') if isinstance(message, dict) else None)
            if mtype == 'assistant':
                num_turns += 1
                # Tool NAMES only, for the remote turn summary — never arguments.
                try:
                    _msg = getattr(message, 'message', None) or (
                        message.get('message') if isinstance(message, dict) else None)
                    for _b in (getattr(_msg, 'content', None)
                               or (_msg.get('content') if isinstance(_msg, dict) else None)
                               or []):
                        _bt = getattr(_b, 'type', None) if not isinstance(_b, dict) else _b.get('type')
                        if _bt != 'tool_use':
                            continue
                        _tn = getattr(_b, 'name', None) if not isinstance(_b, dict) else _b.get('name')
                        if _tn:
                            tool_counts[_tn] = tool_counts.get(_tn, 0) + 1
                except Exception:
                    pass
                text, usage = _extract_text_and_usage(message)
                if text:
                    text_parts.append(text)
                if usage is not None:
                    # usage is a BetaUsage dataclass for real llama.cpp messages
                    # (rarely a dict) — coerce instead of discarding, else the
                    # result message always reports zero tokens. ACCUMULATE across
                    # turns (the SDK result.usage is a cumulative run total; a
                    # multi-turn -p run previously reported only the final turn).
                    _u = usage if isinstance(usage, dict) else _to_jsonable(usage)
                    if isinstance(_u, dict):
                        if total_usage is None:
                            total_usage = dict(_u)
                        else:
                            for _k, _v in _u.items():
                                if isinstance(_v, (int, float)) and not isinstance(_v, bool):
                                    total_usage[_k] = total_usage.get(_k, 0) + _v
                                else:
                                    total_usage.setdefault(_k, _v)
                messages.append(message)
                session_storage.record_transcript(messages)
                if is_stream:
                    _emit(_wrap_message(message, 'assistant', session_id))
            elif mtype == 'user':
                messages.append(message)
                session_storage.record_transcript(messages)
                if is_stream:
                    _emit(_wrap_message(message, 'user', session_id))
            # progress messages are not part of the SDK stream — skip.
    except Exception as e:  # noqa: BLE001 — surface as a structured result/stderr
        is_error = True
        errors.append(str(e))

    # waitingForAgents (print.ts ~2371-2406): the model's turn loop has ended,
    # but background shell tasks may still be running — a -p run can be
    # long-lived (tens of minutes). Keep draining their completion
    # <task-notification>s (emitting each on the stream) and do NOT exit until no
    # background task is still running AND the queue is empty. This is a
    # correctness requirement, not just streaming nicety: process exit triggers
    # shutdown_all(), which SIGTERMs still-running tasks — so exiting early would
    # kill a background task before it finishes and drop its completion. Runs
    # regardless of output format; only stream-json emits the per-notification
    # lines. (Mirrors the do/while: drain at the top, then continue while a bg
    # task runs OR a notification is queued — the queue check closes the race
    # where a task finishes between the drain and the running-task check.)
    try:
        from .utils.message_queue_manager import (
            drain_pending_notifications,
            has_pending_notifications,
        )
        from .utils.task.framework import get_running_tasks
        from .state.app_state import get_app_state
        while True:
            for _block in drain_pending_notifications():
                _nmsg = create_user_message(_block)
                messages.append(_nmsg)
                session_storage.record_transcript(messages)
                if is_stream:
                    _emit(_wrap_message(_nmsg, 'user', session_id))
            has_running_bg = any(
                getattr(t, 'is_backgrounded', False)
                for t in get_running_tasks(get_app_state())
            )
            pending = has_pending_notifications()
            if not has_running_bg and not pending:
                break
            if not pending:
                await asyncio.sleep(0.1)
    except Exception as e:  # noqa: BLE001 — the drain wait must never crash the run
        is_error = True
        errors.append(str(e))

    duration_ms = int((time.time() - start) * 1000)
    result_text = '\n'.join(text_parts).strip()

    # Headless runs are the ones nobody is watching, so this path matters more
    # than the REPL's, not less. Best-effort; never affects the exit code.
    from .remote.hooks import notify_turn_end
    from .utils.session_status import status_line
    notify_turn_end(
        duration_ms,
        result_text,
        cwd=get_cwd(),
        error=(errors[0] if (is_error and errors) else ('execution error' if is_error else '')),
        tools=tool_counts,
        prompt=prompt,
        status=status_line(messages),
    )

    if is_stream or is_json:
        _emit(build_result_message(
            subtype='error_during_execution' if is_error else 'success',
            result_text=result_text,
            num_turns=num_turns,
            usage=total_usage,
            duration_ms=duration_ms,
            session_id=session_id,
            is_error=is_error,
            errors=errors or None,
        ))
    else:  # text
        if is_error:
            sys.stderr.write(f"Error: {errors[0] if errors else 'execution error'}\n")
        else:
            print(result_text)

    return 1 if is_error else 0
