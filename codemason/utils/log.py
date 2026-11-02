"""
Logging utilities for CodeMason
Equivalent to src/utils/log.ts
"""

import asyncio
import json
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# Log file paths — single source of truth in config.py
from .config import LOG_DIR
from .platform import chmod_posix, open_append_fd
# Resolved on EVERY use, not captured at import.
#
# -dbg points logging at /tmp/codemason-dbg-<pid>/ by setting CODEMASON_LOG_DIR
# and updating config.LOG_DIR, and that happens while parsing arguments — long
# after this module is imported. Binding the paths here froze them to the
# default, so -dbg printed the /tmp paths on screen and then wrote nothing to
# them: the directory it announced stayed empty while the lines went to
# ~/.codemason/logs. Observed 2026-09-07.
def _log_dir():
    from . import config as _cfg
    return _cfg.LOG_DIR


def _error_log():
    return _log_dir() / 'error.log'


def _debug_log_path():
    return _log_dir() / 'debug.log'


def _turn_log():
    """One line per model request, ALWAYS on. See log_turn."""
    return _log_dir() / 'turns.log'

# Per-file size cap before rotation (one previous generation is kept as .1).
# Without this the append-only logs grow without bound over a long autonomous
# session; env-overridable. 0 disables rotation.
_MAX_LOG_BYTES = 16 * 1024 * 1024


def debug_enabled() -> bool:
    """debug_log is OFF unless CODEMASON_DEBUG is set to a truthy value.

    Rationale: debug_log serializes its payload (json.dumps) and does a blocking
    disk write on EVERY call, and hot-path callers pass the full request body
    (whole conversation) each turn. Ungated, that is event-loop-blocking I/O plus
    unbounded disk growth plus a local plaintext copy of every conversation. Gate
    it so the default (production/solver) run pays nothing; set CODEMASON_DEBUG=1
    to restore full logging when diagnosing."""
    return os.getenv('CODEMASON_DEBUG', '').strip().lower() not in ('', '0', 'false', 'off', 'no')


def _rotate_if_needed(path: Path) -> None:
    """Rotate `path` to `path.1` (overwriting a prior .1) once it exceeds the cap,
    so a single log file can't fill the disk. Best-effort; never raises.

    Every codemason process on this machine appends to the same log, so the
    rotation is the one part that needs coordinating: two processes that both saw
    the file over the cap both renamed, and the second rename landed on the .1
    the first had just written — 16MB of log gone — or on the fresh file the
    first had just started. The size check stays outside the lock, so the normal
    path (under the cap) costs one stat, as before. Only the rare over-the-cap
    branch asks for the lock, and it asks WITHOUT waiting: if another process is
    rotating, this one has nothing left to do. The re-check inside is what makes
    that correct — by then the rotation may already have happened.
    """
    if _MAX_LOG_BYTES <= 0:
        return
    try:
        if not (path.exists() and path.stat().st_size > _MAX_LOG_BYTES):
            return
        from .config import lock_path
        from .file import FileLock
        with FileLock(lock_path('log-rotate'), wait=False) as held:
            if not held:
                return                      # another process is rotating it
            if path.exists() and path.stat().st_size > _MAX_LOG_BYTES:
                path.replace(path.with_name(path.name + '.1'))
    except OSError:
        pass


def ensure_log_dir() -> None:
    """Ensure log directory exists, 0700.

    These logs carry request bodies, tool results and — on a transport failure —
    whatever an error string happened to contain. A bare mkdir lands 0755 and
    the files inside 0644, so any local account could read them.

    Resolved through _log_dir() like every other path here, not the LOG_DIR this
    module imported: -dbg repoints config.LOG_DIR after import, so the captured
    one created and chmod'd a directory nothing was being written to while the
    real one stayed 0755.
    """
    d = _log_dir()
    d.mkdir(parents=True, exist_ok=True)
    try:
        chmod_posix(d, 0o700)
    except OSError:
        pass


def now_iso() -> str:
    """Local time as the log filenames and log lines spell it.

    The Rust build's utils::log::now_iso is
    `chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f")`, and the dumps named
    with it sit in one shared logs/ directory, so the two builds' files have to
    sort together and read alike.

    strftime rather than datetime.isoformat(): isoformat drops the fractional
    part entirely when the microsecond happens to be 0, which would emit a
    differently shaped name roughly once in a million dumps — and a stall that
    rare is exactly the one worth reading. Rust's %.6f always writes six digits,
    so this does too. Microseconds also keep a burst distinguishable: four
    aborted-turn dumps inside 0.7ms are on record (openai_common.rs).
    """
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')


def _open_log(path):
    """Append handle for a log file: created 0600, always UTF-8.

    Two problems with a bare open(path, 'a'). The mode: with none given the file
    lands 0644 and these logs hold request bodies and tool results. The
    encoding: a text handle without encoding= uses the process locale, which is
    cp949 on Korean Windows, so a log line carrying Hangul or an emoji raised
    UnicodeEncodeError — swallowed by the callers' except, leaving the entry
    silently missing from the very file that exists to explain a failure.

    errors='replace' rather than strict: a diagnostic must not be the thing that
    throws. Only affects files created from here on; an existing log keeps the
    mode it already had.

    A third problem, and only on Windows: the CRT's O_APPEND is a seek followed
    by a write, so two of these handles racing on one log overwrite each other.
    open_append_fd is what makes the append indivisible there; see its docstring
    for the measurement. Every codemason process on the machine appends to this
    same file, so the race is the normal case, not a corner.
    """
    fd = open_append_fd(path, 0o600)
    return os.fdopen(fd, 'a', encoding='utf-8', errors='replace')

def log_panic(exc_type, exc_value, exc_tb, session_id: Optional[str] = None) -> None:
    """Record an uncaught exception where it can still be read tomorrow morning.

    Python's default hook prints a traceback to stderr, which in the run this
    matters most for — started at night, looked at the next day — is scrollback
    that may be gone, scrolled past, or in a closed terminal.

    The point of a crash here is not recovery, it is DIAGNOSIS: the run is over
    either way, and the only thing worth salvaging is why. So the full traceback
    goes to the same error log everything else uses, alongside the session id so
    it can be lined up with the transcript that ends at the same moment.

    Never raises: an exception inside the last-chance handler loses the record.
    """
    try:
        ensure_log_dir()
        stack = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'Panic',
            'message': f'{getattr(exc_type, "__name__", exc_type)}: {exc_value}',
            'context': _panic_location(exc_tb),
            'session_id': session_id or os.environ.get('CODEMASON_SESSION_ID'),
            'stack': stack,
        }
        _rotate_if_needed(_error_log())
        with _open_log(_error_log()) as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:  # noqa: BLE001 - a failure here must not mask the crash
        pass


def _panic_location(exc_tb) -> Optional[str]:
    """`file:line:function` of the innermost frame — where it actually blew up.

    Without this the record says what happened and not where, which is the half
    that turns a log line into something someone can act on.
    """
    try:
        frames = traceback.extract_tb(exc_tb)
        if not frames:
            return None
        last = frames[-1]
        return f'{last.filename}:{last.lineno}:{last.name}'
    except Exception:  # noqa: BLE001
        return None


def install_crash_logger() -> None:
    """Route uncaught exceptions to the error log, then to the normal handler.

    Covers threads as well as the main one: a crash on a worker prints and is
    otherwise lost, and that is exactly the crash nobody is awake to see.
    """
    previous = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        # KeyboardInterrupt is the user stopping the run, not a crash.
        if not issubclass(exc_type, KeyboardInterrupt):
            log_panic(exc_type, exc_value, exc_tb)
        previous(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook

    if hasattr(threading, 'excepthook'):
        previous_thread_hook = threading.excepthook

        def _thread_hook(args):
            if not issubclass(args.exc_type, KeyboardInterrupt):
                log_panic(args.exc_type, args.exc_value, args.exc_traceback)
            previous_thread_hook(args)

        threading.excepthook = _thread_hook


def install_asyncio_crash_logger(loop=None) -> None:
    """Route asyncio task failures to the error log as well.

    A crash inside a task does NOT reach sys.excepthook. asyncio hands it to the
    loop's exception handler, which prints "Task exception was never retrieved"
    and moves on — measured, nothing reached the log at all. This program is
    asyncio throughout and does most of its work in background tasks, so that is
    the likeliest shape of crash there is, and it was the one shape not being
    recorded.

    Must be called with a running loop (the handler belongs to the loop, not the
    process), so it is installed where the loop is created rather than alongside
    the process-wide hooks.
    """
    try:
        loop = loop or asyncio.get_running_loop()
    except RuntimeError:
        return
    previous = loop.get_exception_handler()

    def _handler(lp, context):
        exc = context.get('exception')
        if exc is not None and not isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            log_panic(type(exc), exc, exc.__traceback__)
        if previous is not None:
            previous(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(_handler)


def log_turn(fields: dict) -> None:
    """One line per model request: what was sent, what came back, how long.

    Unconditional, unlike debug_log, and separate from error.log, unlike the
    cold-prefill warning. Both of those were the wrong shape for the question
    that keeps being asked of a slow turn — "was that prefill, or thinking, or
    just a long answer?" — because a threshold only records the bad cases and a
    debug flag records nothing unless it was set BEFORE the run that went wrong.
    By then the run is over. A turn that took 500 seconds has to leave its own
    numbers behind whether or not anyone expected it to.

    One line, so it stays greppable across a long unattended session; rotated
    like the others so it cannot grow without bound.

    Appended through _open_log like every other log here. A bare open(path, 'a')
    landed it 0644 while the directory and its siblings were 0600 — the one
    writer in this module that skipped the helper written for that, and the one
    place the Rust build (which goes through append_line, mode 0600) and this one
    disagreed about a file they both append to.
    """
    ensure_log_dir()
    _rotate_if_needed(_turn_log())
    parts = ['[%s]' % datetime.now().isoformat()]
    for k, v in fields.items():
        parts.append('%s=%s' % (k, v))
    try:
        with _open_log(_turn_log()) as f:
            f.write(' '.join(parts) + '\n')
    except OSError as e:
        print('Failed to write turn log: %s' % e, file=sys.stderr)


def log_error(error: Any, context: Optional[str] = None) -> None:
    """Log an error to the error log file"""
    ensure_log_dir()

    timestamp = datetime.now().isoformat()

    if isinstance(error, Exception):
        error_msg = str(error)
        error_type = type(error).__name__
        stack_trace = traceback.format_exc()
    else:
        error_msg = str(error)
        error_type = 'Unknown'
        stack_trace = ''

    log_entry = {
        'timestamp': timestamp,
        'type': error_type,
        'message': error_msg,
        'context': context,
        'stack': stack_trace
    }

    try:
        _rotate_if_needed(_error_log())
        with _open_log(_error_log()) as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Failed to write error log: {e}", file=sys.stderr)


def debug_log(title: str, data: Any = None) -> None:
    """Write debug log entry (no-op unless CODEMASON_DEBUG is set)."""
    if not debug_enabled():
        return
    ensure_log_dir()
    _rotate_if_needed(_debug_log_path())

    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] {title}"

    if data is not None:
        try:
            log_line += f": {json.dumps(data, indent=2, default=str)}"
        except Exception:
            log_line += f": {str(data)}"

    log_line += '\n'

    try:
        with _open_log(_debug_log_path()) as f:
            f.write(log_line)
    except Exception:
        pass


def log_event(event_name: str, properties: Optional[dict] = None) -> None:
    """Log an analytics event (placeholder for statsig)"""
    debug_log(f"EVENT: {event_name}", properties)


def clear_logs() -> None:
    """Clear all log files"""
    ensure_log_dir()
    for log_file in [_error_log(), _debug_log_path()]:
        try:
            log_file.unlink(missing_ok=True)
        except Exception:
            pass


def run_with_crash_logger(coro):
    """asyncio.run(), with the loop's crash handler installed first.

    The handler belongs to a loop, not to the process, so it cannot be set
    alongside sys.excepthook at startup — every entry point that creates a loop
    has to install it. Wrapping the call is how that stops being something each
    new entry point must remember.
    """
    async def _wrapped():
        install_asyncio_crash_logger()
        return await coro
    return asyncio.run(_wrapped())
