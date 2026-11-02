"""Disk-backed task output — adapted from src/utils/task/diskOutput.ts.

A background task streams its combined stdout/stderr to a per-task file; the
model retrieves it later via TaskOutput (or by Read-ing the path surfaced in the
tool result / <task-notification>). This module owns the output path and the
bounded tail/delta reads used by TaskOutput and the framework.

Substrate notes:
- The original stores files under `getProjectTempDir()/<sessionId>/tasks/` and
  uses O_NOFOLLOW + a 5GB watchdog. codemason has no getSessionId or sandbox, so
  output lives under a fixed temp dir; the write-side 10MB cap lives in
  LocalShellTask (see tasks/local_shell_task.py), the codemason equivalent of
  DiskTaskOutput's disk cap. Path scheme is substrate; the `.output` suffix and
  the read semantics (tail with an "[NKB omitted]" header, byte-offset delta)
  are ported faithfully.
"""

import os
import tempfile
from ..platform import chmod_posix

# Read at most this many bytes when tailing a task's output file, so a multi-GB
# log never gets pulled fully into memory (diskOutput DEFAULT_MAX_READ_BYTES).
DEFAULT_MAX_READ_BYTES = 8 * 1024 * 1024  # 8MB


def get_task_output_dir() -> str:
    """Directory holding per-task output files (getTaskOutputDir)."""
    return os.path.join(tempfile.gettempdir(), "codemason", "tasks")


def get_task_output_path(task_id: str) -> str:
    """Output file path for a task (getTaskOutputPath)."""
    return os.path.join(get_task_output_dir(), f"{task_id}.output")


def ensure_output_dir() -> None:
    """Ensure the task output directory exists (ensureOutputDir), 0700.

    These files hold every background command's full stdout and stderr — build
    logs, environment dumps, credentials that appear in a tool's error output.
    A bare makedirs lands 0755 in a world-readable /tmp, so any local account
    could read them. tool_result_storage.py already chmods its own directory
    0700 for exactly this reason ("tool outputs may hold sensitive data and /tmp
    is world-readable"); this one was missed.

    The parent (/tmp/codemason) is tightened too. It is created by the same
    makedirs and, left at 0755 in a sticky world-writable /tmp, a squatter who
    pre-creates it owns everything written underneath.
    """
    d = get_task_output_dir()
    os.makedirs(d, exist_ok=True)
    for path in (os.path.dirname(d), d):
        try:
            chmod_posix(path, 0o700)
        except OSError:
            # Not ours to tighten (pre-existing, another owner) — the write
            # itself still succeeds or fails on its own merits.
            pass


def get_task_output(task_id: str, max_bytes: int = DEFAULT_MAX_READ_BYTES) -> str:
    """Read a task's output, tailing the file (getTaskOutput).

    Caps at `max_bytes` to avoid loading huge files; when the file is larger,
    returns the last `max_bytes` prefixed with a one-line note recording how
    much earlier output was omitted. Missing file → empty string (ENOENT).
    """
    path = get_task_output_path(task_id)
    try:
        bytes_total = os.path.getsize(path)
        with open(path, "rb") as f:
            if bytes_total > max_bytes:
                f.seek(bytes_total - max_bytes)
            data = f.read()
    except FileNotFoundError:
        return ""
    except OSError:
        return ""
    content = data.decode("utf-8", "replace")
    bytes_read = len(data)
    if bytes_total > bytes_read:
        omitted_kb = round((bytes_total - bytes_read) / 1024)
        return f"[{omitted_kb}KB of earlier output omitted]\n{content}"
    return content


def get_task_output_delta(task_id: str, from_offset: int,
                          max_bytes: int = DEFAULT_MAX_READ_BYTES) -> dict:
    """New content since `from_offset` (getTaskOutputDelta).

    Reads only from the byte offset, up to `max_bytes` — never loads the whole
    file. Returns {'content', 'newOffset'}. Missing file → empty delta.
    """
    path = get_task_output_path(task_id)
    try:
        with open(path, "rb") as f:
            f.seek(from_offset)
            data = f.read(max_bytes)
    except FileNotFoundError:
        return {"content": "", "newOffset": from_offset}
    except OSError:
        return {"content": "", "newOffset": from_offset}
    return {
        "content": data.decode("utf-8", "replace"),
        "newOffset": from_offset + len(data),
    }


def get_task_output_size(task_id: str) -> int:
    """Current size (byte offset) of a task's output file (getTaskOutputSize)."""
    try:
        return os.path.getsize(get_task_output_path(task_id))
    except FileNotFoundError:
        return 0
    except OSError:
        return 0


def cleanup_task_output(task_id: str) -> None:
    """Delete a task's output file (cleanupTaskOutput)."""
    try:
        os.unlink(get_task_output_path(task_id))
    except FileNotFoundError:
        return
    except OSError:
        return


def init_task_output(task_id: str) -> str:
    """Create a task's output file, empty (initTaskOutput).

    The source has two initializers: this one, and `initTaskOutputAsSymlink`,
    which points the output path at the agent's own transcript JSONL so
    TaskOutput reads the real conversation. codemason does not write per-agent
    transcripts yet, so there is no target to link to and this is the right
    equivalent — it is also what the source falls back to when the symlink
    fails. Creating the file at spawn is what lets a tool report `output_file`
    before the task has produced anything.

    Best-effort: the output file is a convenience for TaskOutput, not a
    precondition for running the task, so a read-only or full disk must not
    fail the spawn.
    """
    path = get_task_output_path(task_id)
    try:
        ensure_output_dir()
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8"):
                pass
    except OSError:
        pass
    return path
