"""LocalShellTask — adapted from src/tasks/LocalShellTask/.

This is the one task type that lives in AppState.tasks under codemason: a
background shell command (Bash's run_in_background). It folds three source files
into one module appropriate to codemason's substrate:

  - guards.ts          → LocalShellTaskState + is_local_shell_task
  - killShellTasks.ts  → kill_task
  - LocalShellTask.tsx → spawn_shell_task / enqueue_shell_notification / the
                         LocalShellTask Task impl

Substrate boundary: the original spawns through ShellCommand/Node child_process
and streams via DiskTaskOutput; codemason spawns with asyncio.create_subprocess_
shell (start_new_session=True → own process group) and streams the combined
stdout/stderr to the task's output file with a write cap. Everything else — the
registry writes (register_task / update_task_state), the killed-wins completion
guard, and the completion <task-notification> wording — is ported faithfully.
"""

import asyncio
import atexit
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Tuple

from ..utils.platform import is_windows, reap_children_with_this_process
from ..utils.windows_shell import (
    shell_argv as _win_shell_argv,
    spawn_group_kwargs as _spawn_group_kwargs,
    kill_process_tree as _kill_process_tree,
)

# pty/termios are POSIX-only — importing them on native Windows raises and would
# take the whole module (and CodeMason startup) down. On Windows the background
# substrate uses plain pipes instead of a PTY (see _spawn_with_pty), so these
# are only needed off-Windows.
if not is_windows():
    import pty
    import termios
else:  # pragma: no cover - Windows path
    pty = None
    termios = None

from ..constants.xml import (
    OUTPUT_FILE_TAG,
    STATUS_TAG,
    SUMMARY_TAG,
    TASK_ID_TAG,
    TASK_NOTIFICATION_TAG,
    TOOL_USE_ID_TAG,
)
from ..state.app_state import get_app_state, set_app_state
from ..task import (
    Task,
    TaskStateBase,
    create_task_state_base,
    generate_task_id,
    is_terminal_task_status,
    now_ms,
)
from ..utils.message_queue_manager import enqueue_pending_notification
from ..utils.task.disk_output import ensure_output_dir, get_task_output_path
from ..utils.task.framework import evict_terminal_task, register_task, update_task_state

# Cap per-task log output so a runaway background command can't fill the disk
# (the original's DiskTaskOutput caps disk writes). Past the cap we keep draining
# the pipe — so the child never blocks on a full pipe — but stop writing and
# append a one-time truncation note.
_MAX_BG_OUTPUT_BYTES = 10 * 1024 * 1024  # 10 MB

# Bound the in-memory registry. The source lazily GCs terminal+notified tasks in
# pollTasks; codemason has no poll loop, so instead we evict the oldest
# terminal+notified records once the registry exceeds this many entries (their
# .output file persists on disk for on-demand Read). A just-finished task stays
# queryable via TaskOutput until then.
_MAX_TASKS = 100

BACKGROUND_BASH_SUMMARY_PREFIX = "Background command "

# XML tag names (constants/xml.ts), matched 1:1 so the emitted block is
# identical in shape to enqueueShellNotification's.



@dataclass
class LocalShellTaskState(TaskStateBase):
    """State for a background shell task (LocalShellTaskState, guards.ts).

    type is fixed to 'local_bash' for backward compatibility with persisted
    state. `result` carries the exit code once finished. `proc` and `run_task`
    are codemason's live handles (the equivalent of the source's `shellCommand`)
    — non-serializable runtime references kill_task uses to terminate the
    process group.
    """

    command: str = ""
    result: Optional[dict] = None  # {'code': int, 'interrupted': bool}
    completion_status_sent_in_attachment: bool = False
    is_backgrounded: bool = True
    agent_id: Optional[str] = None
    kind: str = "bash"  # 'bash' | 'monitor'
    # Runtime handles (codemason substrate for ShellCommand).
    proc: Any = None  # asyncio subprocess
    run_task: Any = None  # asyncio.Task running _run


def is_local_shell_task(task: Any) -> bool:
    """Type guard (isLocalShellTask)."""
    return getattr(task, "type", None) == "local_bash"


def _escape_xml(s: str) -> str:
    """Port of utils/xml.ts escapeXml: escape element-text special chars."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def enqueue_shell_notification(
    task_id: str,
    description: str,
    status: str,  # 'completed' | 'failed' | 'killed'
    exit_code: Optional[int],
    tool_use_id: Optional[str] = None,
    kind: str = "bash",
) -> None:
    """Enqueue a completion <task-notification> (enqueueShellNotification).

    Atomically checks-and-sets `notified` so a task is announced at most once: if
    it was already marked notified (e.g. by TaskStop killing it), this is a
    no-op. That is exactly how stopTask suppresses the kill ("exit 137") notice.
    """
    should_enqueue = {"v": False}

    def _mark(task):
        if task.notified:
            return task
        should_enqueue["v"] = True
        return replace(task, notified=True)

    update_task_state(task_id, set_app_state, _mark)
    if not should_enqueue["v"]:
        return

    if status == "completed":
        suffix = f" (exit code {exit_code})" if exit_code is not None else ""
        summary = f'{BACKGROUND_BASH_SUMMARY_PREFIX}"{description}" completed{suffix}'
    elif status == "failed":
        suffix = f" with exit code {exit_code}" if exit_code is not None else ""
        summary = f'{BACKGROUND_BASH_SUMMARY_PREFIX}"{description}" failed{suffix}'
    else:  # killed
        summary = f'{BACKGROUND_BASH_SUMMARY_PREFIX}"{description}" was stopped'

    output_path = get_task_output_path(task_id)
    tool_use_id_line = (
        f"\n<{TOOL_USE_ID_TAG}>{tool_use_id}</{TOOL_USE_ID_TAG}>" if tool_use_id else ""
    )
    message = (
        f"<{TASK_NOTIFICATION_TAG}>\n"
        f"<{TASK_ID_TAG}>{task_id}</{TASK_ID_TAG}>{tool_use_id_line}\n"
        f"<{OUTPUT_FILE_TAG}>{output_path}</{OUTPUT_FILE_TAG}>\n"
        f"<{STATUS_TAG}>{status}</{STATUS_TAG}>\n"
        f"<{SUMMARY_TAG}>{_escape_xml(summary)}</{SUMMARY_TAG}>\n"
        f"</{TASK_NOTIFICATION_TAG}>"
    )
    enqueue_pending_notification(message, mode="task-notification")


def kill_task(task_id: str, set_app_state_fn: Callable = set_app_state) -> None:
    """Kill a running shell task and mark it killed+notified (killTask).

    The OS termination happens inside the state updater (mirroring the source,
    which calls shellCommand.kill() inside updateTaskState). Marking notified
    here is what suppresses the later completion notice. No-op unless the task is
    a running local_bash task.
    """
    def _kill(task):
        if task.status != "running" or not is_local_shell_task(task):
            return task
        proc = task.proc
        if proc is not None and proc.returncode is None:
            # Own process group (start_new_session / CREATE_NEW_PROCESS_GROUP) →
            # signal the whole tree so grandchildren are reaped too. No sig arg:
            # POSIX defaults to SIGKILL inside kill_process_tree (signal.SIGKILL
            # doesn't exist on Windows, so callers must not name it); Windows
            # taskkills the tree.
            _kill_process_tree(proc)
        return replace(
            task,
            status="killed",
            notified=True,
            proc=None,
            end_time=now_ms(),
        )

    update_task_state(task_id, set_app_state_fn, _kill)


def _powershell_argv(command: str) -> list:
    """Argv to run `command` through PowerShell via -EncodedCommand (UTF-16LE).

    Mirrors the foreground PowerShell substrate's flags. If no PowerShell is
    found we fall back to git-bash so the background task still runs something
    rather than crashing; the PowerShell tool's Windows-only gating makes that
    fallback unreachable in practice.
    """
    import base64
    from ..utils.powershell_detect import get_cached_powershell_path

    pwsh = get_cached_powershell_path()
    if not pwsh:
        return _win_shell_argv(command)
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return [pwsh, "-NoProfile", "-NonInteractive", "-NoLogo",
            "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded]


async def _spawn_with_pty(command: str, cwd: str, env: dict, kind: str = "bash"):
    """Spawn a shell command attached to a PTY so children that block-buffer
    stdout when it's a pipe (python, and anything using libc stdio) instead
    line-buffer and stream live to disk. Returns (proc, reader, transport).

    Without this, a long-running background command's .output file stays empty
    until the child exits or fills its ~8KB buffer — the streamed-output-to-disk
    contract silently breaks. The real Claude Code runs commands PTY-backed;
    this restores that. reader has the same StreamReader interface as
    proc.stdout, so the existing drain loop is unchanged.

    Windows has no pty/termios, so there we fall back to plain pipes with stderr
    merged onto stdout. Block-buffered children may stream less eagerly than
    under a PTY, but output is never lost and the (proc, reader, transport)
    contract is identical (transport is None — the caller already tolerates that).
    """
    if is_windows():
        # bash.exe -c command (or powershell.exe -EncodedCommand for the PS tool);
        # stderr merged onto stdout (== PTY single-stream).
        # Everything spawned from here on dies with this process, however this
        # process dies -- taskkill /F runs no atexit handler, and this shell was
        # outliving the session. Before the spawn on purpose: a job member's
        # children join the job at creation, so there is no window for a
        # grandchild to start outside it. No-op off Windows.
        reap_children_with_this_process()

        argv = _powershell_argv(command) if kind == "powershell" else _win_shell_argv(command)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env,
            **_spawn_group_kwargs(),
        )

        return proc, proc.stdout, None

    master_fd, slave_fd = pty.openpty()
    # OPOST off: don't translate the child's '\n' into '\r\n' in the log.
    attrs = termios.tcgetattr(slave_fd)
    attrs[1] &= ~termios.OPOST
    termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    finally:
        os.close(slave_fd)  # parent keeps only the master end
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    master_file = os.fdopen(master_fd, "rb", 0)
    transport, _ = await loop.connect_read_pipe(lambda: protocol, master_file)
    return proc, reader, transport


def _background_timeout() -> Optional[float]:
    """Wall-clock cap for one background task in seconds, or None when off.

    Read per task rather than cached: a session that raises the setting for a
    long job should not have to restart for it to take effect."""
    try:
        from ..utils.config import get_config
        seconds = float(getattr(get_config(), "background_task_timeout", 0.0) or 0.0)
    except Exception:  # noqa: BLE001 - a bad setting must not stop the task
        return None
    return seconds if seconds > 0 else None


def _note_timeout(f, seconds: float) -> None:
    """Say why the output stops, in the file the user actually reads.

    The task settles as failed with the signal's exit code, which on its own
    looks like the command failed rather than like we killed it."""
    f.write(
        f"\n[background task terminated: exceeded {int(seconds)} seconds "
        f"\u2014 set background_task_timeout to change, or 0 to disable]\n".encode("utf-8")
    )
    f.flush()


async def _run(
    task_id: str,
    command: str,
    cwd: str,
    output_path: str,
    description: str,
    tool_use_id: Optional[str],
    kind: str,
) -> None:
    """Spawn the command, stream output to disk, then settle task state.

    On completion, transitions running → completed/failed UNLESS the task was
    already killed (killed wins — same guard as the source's result handler),
    then enqueues the completion notification (suppressed if already notified).
    """
    returncode: Optional[int] = None
    errored = False
    transport = None
    try:
        with open(output_path, "wb") as f:
            # PTY-backed so block-buffered children stream live (see _spawn_with_pty).
            # stderr is merged onto the same PTY (== old stderr=STDOUT behaviour).
            proc, pty_reader, transport = await _spawn_with_pty(
                # PYTHONUNBUFFERED: the Windows fallback feeds the child a
                # pipe, where a python child block-buffers and the output file
                # stays empty while it visibly works. Harmless under the PTY.
                command, cwd, {**os.environ, "TERM": "dumb", "PYTHONUNBUFFERED": "1"}, kind
            )
            # Publish the live handle so kill_task can terminate the group.
            # A TaskStop can land in the window between register_task (status
            # 'running') and here — kill_task would have found proc=None and
            # marked the task 'killed' WITHOUT signalling the OS process. Detect
            # that (never attach proc to a killed task), kill the freshly-spawned
            # group now, and bail without streaming/settling.
            killed_in_window = {"v": False}

            def _publish_proc(t):
                if t.status == "killed":
                    killed_in_window["v"] = True
                    return t
                return replace(t, proc=proc)

            update_task_state(task_id, set_app_state, _publish_proc)
            if killed_in_window["v"]:
                _kill_process_tree(proc)  # no sig: SIGKILL on POSIX
                try:
                    await proc.wait()   # reap so the killed child isn't left a zombie
                except Exception:  # noqa: BLE001
                    pass
                return
            written = 0
            truncated = False
            timeout_s = _background_timeout()
            deadline = (time.monotonic() + timeout_s) if timeout_s else None
            while True:
                try:
                    if deadline is None:
                        chunk = await pty_reader.read(4096)
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        chunk = await asyncio.wait_for(
                            pty_reader.read(4096), remaining)
                except asyncio.TimeoutError:
                    # Kill the GROUP, not just the shell: the runaway is usually
                    # a grandchild, and killing the shell alone reparents it.
                    _note_timeout(f, timeout_s)
                    _kill_process_tree(proc)  # no sig: SIGKILL on POSIX
                    break
                except OSError:
                    # PTY master read() raises EIO when the slave side closes
                    # (child exited) — treat as EOF, same as a pipe's empty read.
                    break
                if not chunk:
                    break
                if written < _MAX_BG_OUTPUT_BYTES:
                    f.write(chunk)
                    written += len(chunk)
                    f.flush()
                elif not truncated:
                    f.write(
                        b"\n[background output truncated: exceeded "
                        + str(_MAX_BG_OUTPUT_BYTES).encode()
                        + b" bytes]\n"
                    )
                    f.flush()
                    truncated = True
                # else: keep draining so the child doesn't block on a full PTY.
            returncode = await proc.wait()
    except Exception as e:  # noqa: BLE001 - record any spawn/IO failure
        errored = True
        try:
            # encoding pinned: a bare open() uses the process locale (cp949 on
            # Korean Windows), so an error message carrying a CJK path or emoji
            # raised UnicodeEncodeError — not an OSError, so it escaped this
            # handler and _run() itself, leaving the task stuck "running".
            with open(output_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"\n[background task error] {e}\n")
        except OSError:
            pass
    finally:
        if transport is not None:
            transport.close()

    # Carry _complete's own decision out to the notification rather than
    # re-reading state (which could be evicted/raced and mis-report a failed
    # task as completed). For the spawn/IO-error path, code=1 so the failed
    # notice includes an exit code, matching the source.
    settled = {"killed": False, "status": "completed", "code": None}

    def _complete(task):
        if task.status == "killed":
            settled["killed"] = True
            return task
        code = returncode if returncode is not None else (1 if errored else 0)
        settled["status"] = "failed" if (errored or code != 0) else "completed"
        settled["code"] = code
        return replace(
            task,
            status=settled["status"],
            result={"code": code, "interrupted": False},
            proc=None,
            end_time=now_ms(),
        )

    update_task_state(task_id, set_app_state, _complete)

    if settled["killed"]:
        status, exit_code = "killed", None
    else:
        status, exit_code = settled["status"], settled["code"]
    enqueue_shell_notification(
        task_id, description, status, exit_code, tool_use_id, kind
    )


def _gc_terminal_tasks() -> None:
    """Bound the registry: once it exceeds _MAX_TASKS, evict the oldest
    terminal+notified records (their .output file stays on disk for Read).
    dict preserves insertion order, so iteration is already oldest-first."""
    state = get_app_state()
    if len(state.tasks) <= _MAX_TASKS:
        return
    evictable = [
        tid
        for tid, task in state.tasks.items()
        if is_terminal_task_status(task.status) and task.notified
    ]
    for tid in evictable[: len(state.tasks) - _MAX_TASKS]:
        evict_terminal_task(tid, set_app_state)


def spawn_shell_task(
    command: str,
    cwd: str,
    description: Optional[str] = None,
    tool_use_id: Optional[str] = None,
    kind: str = "bash",
) -> Tuple[str, str]:
    """Spawn a background shell task; return (task_id, output_path) (spawnShellTask).

    Registers a running LocalShellTaskState in AppState.tasks and launches the
    streaming coroutine. `description` is the human label echoed in the
    completion <summary> (falls back to the command).
    """
    ensure_output_dir()
    _gc_terminal_tasks()
    task_id = generate_task_id("local_bash")
    output_path = get_task_output_path(task_id)

    base = create_task_state_base(task_id, "local_bash", description or command, tool_use_id)
    base["status"] = "running"
    task = LocalShellTaskState(
        **base,
        command=command,
        completion_status_sent_in_attachment=False,
        is_backgrounded=True,
        agent_id=None,
        kind=kind,
    )
    register_task(task, set_app_state)

    run_task = asyncio.create_task(
        _run(task_id, command, cwd, output_path, description or command, tool_use_id, kind)
    )
    # Keep a reference so the task isn't garbage-collected mid-run.
    update_task_state(task_id, set_app_state, lambda t: replace(t, run_task=run_task))
    return task_id, output_path


async def _run_adopted(
    task_id: str,
    proc,
    accumulated: bytes,
    output_path: str,
    description: str,
    tool_use_id: Optional[str],
    kind: str,
) -> None:
    """Like _run, but ADOPTS an already-running process (a foreground command that
    hit its timeout) instead of spawning. Writes the output captured so far, then
    keeps draining the SAME process (both pipes, merged) to disk and settles.
    """
    returncode: Optional[int] = None
    errored = False
    closer: Optional[asyncio.Task] = None

    async def _force_close():
        # Prevent a hang when a `cmd &` grandchild keeps the pipes open: once the
        # MAIN process exits, close the transports so read() returns EOF.
        while proc.returncode is None:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)
        for _fd in (1, 2):
            try:
                _t = proc._transport.get_pipe_transport(_fd)
                if _t and not _t.is_closing():
                    _t.close()
            except (AttributeError, OSError):
                pass

    try:
        with open(output_path, "wb") as f:
            # Write the pre-timeout output, BOUNDED — a command that dumped >10MB
            # before timing out must not bloat the file past the cap either.
            trunc0 = False
            if accumulated:
                if len(accumulated) > _MAX_BG_OUTPUT_BYTES:
                    f.write(accumulated[:_MAX_BG_OUTPUT_BYTES])
                    f.write(b"\n[background output truncated: exceeded "
                            + str(_MAX_BG_OUTPUT_BYTES).encode() + b" bytes]\n")
                    accumulated = accumulated[:_MAX_BG_OUTPUT_BYTES]
                    trunc0 = True
                else:
                    f.write(accumulated)
                f.flush()

            killed_in_window = {"v": False}

            def _publish_proc(t):
                if t.status == "killed":
                    killed_in_window["v"] = True
                    return t
                return replace(t, proc=proc)

            update_task_state(task_id, set_app_state, _publish_proc)
            if killed_in_window["v"]:
                _kill_process_tree(proc)  # no sig: SIGKILL on POSIX
                try:
                    await proc.wait()
                except Exception:  # noqa: BLE001
                    pass
                return

            closer = asyncio.create_task(_force_close())
            written = {"n": len(accumulated), "trunc": trunc0}

            async def _drain(reader):
                if reader is None:
                    return
                while True:
                    chunk = await reader.read(4096)
                    if not chunk:
                        break
                    if written["n"] < _MAX_BG_OUTPUT_BYTES:
                        f.write(chunk)
                        written["n"] += len(chunk)
                        f.flush()
                    elif not written["trunc"]:
                        f.write(b"\n[background output truncated: exceeded "
                                + str(_MAX_BG_OUTPUT_BYTES).encode() + b" bytes]\n")
                        f.flush()
                        written["trunc"] = True

            timeout_s = _background_timeout()
            drains = asyncio.gather(_drain(proc.stdout), _drain(proc.stderr))
            try:
                if timeout_s:
                    await asyncio.wait_for(drains, timeout_s)
                else:
                    await drains
            except asyncio.TimeoutError:
                _note_timeout(f, timeout_s)
                _kill_process_tree(proc)  # no sig: SIGKILL on POSIX
            returncode = await proc.wait()
    except Exception as e:  # noqa: BLE001
        errored = True
        try:
            # encoding pinned: a bare open() uses the process locale (cp949 on
            # Korean Windows), so an error message carrying a CJK path or emoji
            # raised UnicodeEncodeError — not an OSError, so it escaped this
            # handler and _run() itself, leaving the task stuck "running".
            with open(output_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"\n[background task error] {e}\n")
        except OSError:
            pass
    finally:
        # Cancel AND await the pipe-closer on every path (normal, error, or an
        # exception before the drain loop) — previously the normal path cancelled
        # without awaiting (unretrieved CancelledError noise) and the error path
        # never cancelled it at all, leaking the task until it self-terminated.
        if closer is not None and not closer.done():
            closer.cancel()
            try:
                await closer
            except BaseException:
                pass

    settled = {"killed": False, "status": "completed", "code": None}

    def _complete(task):
        if task.status == "killed":
            settled["killed"] = True
            return task
        code = returncode if returncode is not None else (1 if errored else 0)
        settled["status"] = "failed" if (errored or code != 0) else "completed"
        settled["code"] = code
        return replace(task, status=settled["status"],
                       result={"code": code, "interrupted": False}, proc=None, end_time=now_ms())

    update_task_state(task_id, set_app_state, _complete)
    if settled["killed"]:
        status, exit_code = "killed", None
    else:
        status, exit_code = settled["status"], settled["code"]
    enqueue_shell_notification(task_id, description, status, exit_code, tool_use_id, kind)


def adopt_timed_out_process(
    proc,
    accumulated: bytes,
    description: Optional[str] = None,
    tool_use_id: Optional[str] = None,
    kind: str = "bash",
) -> Tuple[str, str]:
    """Register an already-running (timed-out foreground) process as a background
    task and keep streaming it to disk. Returns (task_id, output_path). Mirrors
    spawn_shell_task's registration so kill_task / TaskStop / notifications all
    work identically."""
    ensure_output_dir()
    _gc_terminal_tasks()
    task_id = generate_task_id("local_bash")
    output_path = get_task_output_path(task_id)

    label = description or "background command"
    base = create_task_state_base(task_id, "local_bash", label, tool_use_id)
    base["status"] = "running"
    task = LocalShellTaskState(
        **base,
        command=label,
        completion_status_sent_in_attachment=False,
        is_backgrounded=True,
        agent_id=None,
        kind=kind,
    )
    register_task(task, set_app_state)

    run_task = asyncio.create_task(
        _run_adopted(task_id, proc, accumulated, output_path, label, tool_use_id, kind)
    )
    update_task_state(task_id, set_app_state, lambda t: replace(t, run_task=run_task))
    return task_id, output_path


class LocalShellTask(Task):
    """The Task implementation registered for type 'local_bash'."""

    name = "LocalShellTask"
    type = "local_bash"

    async def kill(self, task_id: str, set_app_state_fn: Callable) -> None:
        kill_task(task_id, set_app_state_fn)


LOCAL_SHELL_TASK = LocalShellTask()


def shutdown_all() -> None:
    """Terminate any still-running background commands on process exit.

    Each is spawned start_new_session=True (own process group), so signal the
    whole group to reap grandchildren. Best-effort; registered with atexit."""
    for task in list(get_app_state().tasks.values()):
        if getattr(task, "status", None) != "running":
            continue
        proc = getattr(task, "proc", None)
        if proc is None or proc.returncode is not None:
            continue
        _kill_process_tree(proc)  # no sig: SIGKILL on POSIX


atexit.register(shutdown_all)
