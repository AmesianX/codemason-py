"""TaskOutput tool — based on src/tools/TaskOutputTool/ from Claude Code.

Retrieves output from a background task. block=true (default) waits for the task
to finish (polling AppState.tasks), block=false returns the current state. The
result is rendered as the same `<retrieval_status>/<task_id>/<status>/<output>`
XML the original maps. Marked deprecated in favor of Read-ing the output file
path surfaced in the tool result and <task-notification>.
"""

import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from pydantic import Field

from ..state.app_state import get_app_state as _global_get_app_state
from ..state.app_state import set_app_state as _global_set_app_state
from ..task import now_ms
from ..tasks.local_agent_task import is_local_agent_task
from ..tasks.local_shell_task import is_local_shell_task
from ..utils.formatting import Colors, colorize
from ..utils.task.disk_output import get_task_output
from ..utils.task.framework import update_task_state
from ..utils.task.output_formatting import format_task_output
from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultFailure,
    ValidationResultSuccess,
)

# Output lines shown in the terminal for a completed task (verbose: all).
_RESULT_PREVIEW_LINES = 10

TASK_OUTPUT_TOOL_NAME = "TaskOutput"

# Returned (as a normal result, NOT an error) when a task_id isn't registered.
# The in-memory task table is empty right after a session resume, so a model
# resuming a turn that had launched tasks would otherwise poll ids that no longer
# exist and get a wall of hard errors. This tells it what happened and what to do
# instead of retrying, and it also corrects the launch-then-poll mental model:
# codemason runs agents synchronously and returns their results inline.
_NOT_FOUND_GUIDANCE = (
    "No task is registered with ID {task_id}. Either it never existed, or its "
    "runtime state was cleared by a session restart — task state is in-memory and "
    "is NOT restored on resume. Do not keep polling this id. In codemason, agents "
    "run synchronously and return their results inline, so there is nothing to "
    "retrieve with TaskOutput; if you were waiting on earlier work, re-run it."
)


class TaskOutputInput(ToolInputSchema):
    task_id: str = Field(..., description="The task ID to get output from")
    # semanticBoolean(default true): also tolerates the strings "true"/"false".
    block: bool = Field(default=True, description="Whether to wait for completion")
    timeout: int = Field(default=30000, ge=0, le=600000, description="Max wait time in ms")


def _semantic_bool(value: Any, default: bool) -> bool:
    """Port of semanticBoolean: accept bool or the strings 'true'/'false',
    falling back to `default` when the value is absent."""
    if value is None:
        return default
    if value is True or value == "true":
        return True
    if value is False or value == "false":
        return False
    return bool(value)


def _ctx_state(context) -> tuple:
    get = getattr(context, "get_app_state", None) or _global_get_app_state
    setf = getattr(context, "set_app_state", None) or _global_set_app_state
    return get, setf


def _get_task_output_data(task) -> dict:
    """Unified output for any task type (getTaskOutputData).

    codemason streams shell output to disk, so the bash path reads the file
    (the original's in-memory ShellCommand.taskOutput is substrate). exitCode is
    pulled from the task's recorded result.
    """
    output = get_task_output(task.id)
    base = {
        "task_id": task.id,
        "task_type": task.type,
        "status": task.status,
        "description": task.description,
        "output": output,
    }
    if is_local_shell_task(task):
        base["exitCode"] = task.result["code"] if task.result else None
        return base

    if is_local_agent_task(task):
        # Prefer the finished answer the task recorded over whatever is on
        # disk. The source's reason holds here too: the output file is the full
        # transcript — every message and tool use — while `result` holds only
        # the agent's final text, which is what the parent asked for. The file
        # is the fallback, and for a still-running agent it is all there is.
        result = task.result or {}
        clean_result = result.get("text") if isinstance(result, dict) else None
        if not clean_result and isinstance(result, dict):
            # A result built from content blocks rather than plain text.
            clean_result = "\n".join(
                (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                for b in (result.get("content") or [])
            ).strip() or None
        base.update(
            prompt=task.prompt,
            result=clean_result or output,
            output=clean_result or output,
            error=task.error,
        )
        # The counts the parent accounts an agent by — its context size and how
        # much work it did. Not in the source's TaskOutput shape, which reports
        # them through an SDK event instead; with no such consumer here, the
        # tool the parent already calls is where they belong.
        if task.progress is not None:
            base["usage"] = {
                "total_tokens": task.progress.token_count,
                "tool_uses": task.progress.tool_use_count,
            }
            if task.progress.summary:
                base["summary"] = task.progress.summary

    return base


async def _wait_for_task_completion(task_id, get_app_state, timeout_ms, abort_controller):
    """Poll until the task leaves running/pending or the timeout elapses
    (waitForTaskCompletion). Returns the task, or None if it vanished."""
    start_time = now_ms()
    # Show the wait. The model deciding to wait here is correct; what went
    # wrong on 2026-09-07 was that nothing on screen said so, the silence read
    # as a hang ("prefill?"), and the wait was removed instead of shown. The
    # spinner already shows the tool and the elapsed time; this adds what it
    # is waiting on and the task's latest line, refreshed twice a second.
    from ..utils.tokens import WaitNoteGuard, set_wait_note
    last_note_at = 0
    with WaitNoteGuard():
        while now_ms() - start_time < timeout_ms:
            # Honor an interrupt: stop waiting and return the current snapshot.
            if abort_controller is not None and abort_controller.is_set():
                break
            task = get_app_state().tasks.get(task_id)
            if task is None:
                return None
            if task.status not in ("running", "pending"):
                return task
            if now_ms() - last_note_at >= 500:
                set_wait_note(_wait_note_for(task))
                last_note_at = now_ms()
            await asyncio.sleep(0.1)
    return get_app_state().tasks.get(task_id)


def _wait_note_for(task) -> str:
    """' waiting on b1a2c3 · python scrape.py · last: 5500 saved' — the task,
    what it is, and the last line it wrote so far."""
    def clip(text: str, max_chars: int) -> str:
        text = text.strip()
        return text if len(text) <= max_chars else text[:max_chars - 1] + "…"
    what = task.description or getattr(task, "command", "") or ""
    note = f"waiting on {task.id} · {clip(what, 40)}"
    out = get_task_output(task.id, 4096) or ""
    last = next((l for l in reversed(out.splitlines()) if l.strip()), None)
    if last:
        note += f" · last: {clip(last, 60)}"
    return note


def _map_result(data: dict) -> str:
    """mapToolResultToToolResultBlockParam: render the result as XML."""
    parts = [f"<retrieval_status>{data['retrieval_status']}</retrieval_status>"]
    if data.get("message"):
        parts.append(f"<message>{data['message']}</message>")
    task = data.get("task")
    if task:
        parts.append(f"<task_id>{task['task_id']}</task_id>")
        parts.append(f"<task_type>{task['task_type']}</task_type>")
        parts.append(f"<status>{task['status']}</status>")
        exit_code = task.get("exitCode")
        if exit_code is not None:
            parts.append(f"<exit_code>{exit_code}</exit_code>")
        output = task.get("output")
        if output and output.strip():
            formatted = format_task_output(output, task["task_id"])
            parts.append(f"<output>\n{formatted['content'].rstrip()}\n</output>")
        if task.get("error"):
            parts.append(f"<error>{task['error']}</error>")
        # The source's renderer stops here and routes an agent's token and tool
        # counts to an SDK consumer (the editor's subagent panel) instead. There
        # is no such consumer here, and the parent is the one that has to decide
        # whether an agent is growing where it should not be — so the counts go
        # into the result the parent already reads. Without this the capability
        # is simply lost rather than substituted.
        usage = task.get("usage")
        if usage:
            parts.append(
                f"<usage><total_tokens>{usage['total_tokens']}</total_tokens>"
                f"<tool_uses>{usage['tool_uses']}</tool_uses></usage>"
            )
        if task.get("summary"):
            parts.append(f"<summary>{task['summary']}</summary>")
    return "\n\n".join(parts)


class TaskOutputTool(Tool):
    name = TASK_OUTPUT_TOOL_NAME
    input_schema = TaskOutputInput
    # Backward-compatible aliases for the renamed AgentOutput/BashOutput tools.
    aliases = ["AgentOutputTool", "BashOutputTool"]

    @classmethod
    async def description(cls, **kwargs) -> str:
        return "[Deprecated] — prefer Read on the task output file path"

    @classmethod
    def is_read_only(cls) -> bool:
        return True

    @classmethod
    def is_concurrency_safe(cls, input_data=None) -> bool:
        # TaskOutputTool.isConcurrencySafe returns isReadOnly (true).
        return cls.is_read_only()

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return "Task Output"

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        if not isinstance(input_data, dict):
            return ''
        task_id = input_data.get('task_id') or input_data.get('shell_id') or ''
        return colorize(str(task_id), Colors.DIM) if task_id else ''

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """Status line + the task's output. The model gets the XML form
        (_map_result); the terminal must not."""
        if not isinstance(data, dict):
            return None
        task = data.get("task")
        if not task:
            if data.get("message"):
                # A not_found (e.g. after resume) reads as a dim note, not a
                # red error — the model still gets the full guidance via the XML.
                return colorize(data["message"], Colors.DIM)
            return str(data.get("retrieval_status", "")) or None

        status = task.get("status", "")
        exit_code = task.get("exitCode")
        head = f"{task.get('task_id', '')} · {status}"
        if exit_code is not None:
            head += f" · exit {exit_code}"
        lines = [colorize(head, Colors.DIM)]

        output = (task.get("output") or "").rstrip()
        if output:
            out_lines = output.split('\n')
            shown = out_lines if verbose else out_lines[:_RESULT_PREVIEW_LINES]
            lines += shown
            if len(out_lines) > len(shown):
                hidden = len(out_lines) - len(shown)
                lines.append(colorize(f"... ({hidden} more lines)", Colors.DIM))
        if task.get("error"):
            lines.append(colorize(f"error: {task['error']}", Colors.RED))
        return '\n'.join(lines)

    @classmethod
    async def prompt(cls) -> str:
        return (
            "DEPRECATED: Prefer using the Read tool on the task's output file path "
            "instead. Background tasks return their output file path in the tool "
            "result, and you receive a <task-notification> with the same path when "
            "the task completes — Read that file directly.\n\n"
            "- Retrieves output from a running or completed task (background shell, "
            "agent, or remote session)\n"
            "- Takes a task_id parameter identifying the task\n"
            "- Returns the task output along with status information\n"
            "- Use block=true (default) to wait for task completion\n"
            "- Use block=false for non-blocking check of current status\n"
            "- Task IDs can be found using the /tasks command\n"
            "- Works with all task types: background shells, async agents, and "
            "remote sessions"
        )

    @classmethod
    async def validate_input(cls, input_data, context) -> ValidationResult:
        task_id = input_data.get("task_id")
        if not task_id:
            return ValidationResultFailure(message="Task ID is required", meta={"errorCode": 1})
        # An unregistered task_id is deliberately NOT a validation failure: after a
        # session resume the in-memory task table is empty, so polling a
        # pre-restart id would otherwise surface as a hard error. call() returns an
        # informative not_found result (with recovery guidance) instead.
        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        task_id = input_data.get("task_id")
        block = _semantic_bool(input_data.get("block"), True)
        # The schema's 0..=600000 is advisory to the model, not enforced here
        # (inputs are not parsed through the pydantic model) — clamp, or one
        # call could hold an unattended -dsp turn for as long as it likes.
        try:
            timeout = int(input_data.get("timeout", 30000))
        except (TypeError, ValueError):
            timeout = 30000
        timeout = max(0, min(timeout, 600000))

        get_app_state, set_app_state = _ctx_state(context)
        task = get_app_state().tasks.get(task_id)
        if task is None:
            # Not an error: return actionable guidance so a resumed session doesn't
            # spew hard failures polling tasks the restart cleared (see
            # _NOT_FOUND_GUIDANCE).
            data = {
                "retrieval_status": "not_found",
                "task": None,
                "message": _NOT_FOUND_GUIDANCE.format(task_id=task_id),
            }
            yield ToolResultSuccess(result_for_assistant=_map_result(data), data=data)
            return

        if not block:
            # Non-blocking: return current state.
            if task.status not in ("running", "pending"):
                update_task_state(task_id, set_app_state, lambda t: _mark_notified(t))
                data = {"retrieval_status": "success", "task": _get_task_output_data(task)}
            else:
                data = {"retrieval_status": "not_ready", "task": _get_task_output_data(task)}
            yield ToolResultSuccess(result_for_assistant=_map_result(data), data=data)
            return

        # Blocking: wait for completion.
        abort_controller = getattr(context, "abort_controller", None)
        completed = await _wait_for_task_completion(
            task_id, get_app_state, timeout, abort_controller
        )
        if completed is None:
            data = {"retrieval_status": "timeout", "task": None}
            yield ToolResultSuccess(result_for_assistant=_map_result(data), data=data)
            return
        if completed.status in ("running", "pending"):
            data = {"retrieval_status": "timeout", "task": _get_task_output_data(completed)}
            yield ToolResultSuccess(result_for_assistant=_map_result(data), data=data)
            return

        update_task_state(task_id, set_app_state, lambda t: _mark_notified(t))
        data = {"retrieval_status": "success", "task": _get_task_output_data(completed)}
        yield ToolResultSuccess(result_for_assistant=_map_result(data), data=data)


def _mark_notified(task):
    from dataclasses import replace
    return replace(task, notified=True)
