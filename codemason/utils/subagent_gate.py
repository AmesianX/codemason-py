"""How many sub-agents may run at once — across every codemason process on this
machine that talks to the SAME model server.

A sub-agent is a second tenant in the server's prefix cache: its own tool
block, its own system prompt, its own history, none of it shared with the
parent. The pool we serve against holds 1.62 conversations (vLLM, 2026-09-07:
"GPU KV cache size: 847,533 tokens, maximum concurrency for 524,288 tokens per
request: 1.62x"). One parent plus one sub-agent is already tight; a second
sub-agent — or a second codemason session's — evicts the parent outright, and
the parent then re-prefills its whole history when control comes back
(measured: ~200s at 267k, 1,334 tok/s cold).

The limit is a set of file locks under the config dir, one per slot, so it
holds between processes too — and, because the lock is per open handle on both
platforms, between two spawns inside one process. A spawn that finds every slot taken
waits, polling; the caller reports the wait — a silent stall is a defect in
unattended runs.

One pool per endpoint, not per machine (see _slot_path), and keyed by the
base_url ALONE, never the model name: one server serving two models serves
them out of the same KV cache, so those two must keep sharing a slot — that is
the limit doing its job, not a case it fails to tell apart."""
import asyncio
import os
from typing import Optional

# The slot IS the lock. Locking differs by platform, and unlike remote/inbox
# there is no sidecar here: the slot file is empty and carries no state, so
# there is no cursor for a byte-range lock to tangle with.
#
# POSIX takes flock(LOCK_EX | LOCK_NB), which is per open file description, so
# two spawns inside ONE process contend just as two processes do. Windows has no
# flock; msvcrt.locking takes a mandatory byte-range lock at the current
# position, and a range may be locked past end of file, so one byte of the empty
# slot file is the whole lock. Windows enforces it per HANDLE — a locked region
# is reachable only through the handle that locked it, or a duplicate — which is
# the same per-handle contention flock gives us. Both are held by an open
# handle, so a crash or a taskkill frees the slot instead of stranding it.
#
# Measured there, not only read off the documentation (Windows 11 Pro
# 10.0.26300, CPython 3.13.11): locking one byte of the empty slot file
# succeeds; a second handle on it IN THE SAME PROCESS is refused with EACCES,
# as is another process; unlocking hands it to the next taker; and taskkill /F
# on the holder frees it. The whole fallback rests on that same-process
# refusal — without it the gate would count slots and ration nothing — so it is
# the one line here worth re-measuring if msvcrt's behaviour ever moves.
#
# This mirrors the Rust build's cfg(unix)/cfg(not(unix)) split in
# utils/subagent_gate.rs; see its module doc for why the file's EXISTENCE is
# deliberately not the lock.
try:
    import fcntl
    msvcrt = None
except ImportError:                     # Windows
    fcntl = None
    try:
        import msvcrt
    except ImportError:                 # neither: see _lock_nb
        msvcrt = None

from .config import LOCKS_DIR, endpoint_tag, get_config


def _lock_nb(fd: int) -> None:
    """Take this slot, or raise OSError because someone else holds it. Never
    blocks: a caller that wants to wait polls (see acquire), so an abort can
    still be seen while waiting.

    A platform with neither primitive is granted the slot. That is not the same
    as an acquisition failing: there is no advisory locking to ask, and refusing
    would mean sub-agents never run at all rather than running unrationed. Same
    choice, and the same reason, as remote/inbox proceeding unlocked.
    """
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)


def _unlock(fd: int) -> None:
    """Drop the lock. Closing the handle would drop it anyway — this is the tidy
    path, so the slot is free the moment release() returns rather than whenever
    the descriptor happens to close."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


class SubagentSlot:
    """A held slot; release() (or garbage collection) frees the lock."""

    def __init__(self, fd: int):
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            try:
                _unlock(self._fd)
            finally:
                os.close(self._fd)
                self._fd = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def _slots() -> int:
    return max(1, int(getattr(get_config(), 'max_concurrent_subagents', 1) or 1))


def _slot_path(endpoint: str, i: int) -> str:
    """Slot `i` of the pool for THIS endpoint.

    The pool is keyed by the model server, because that is whose capacity it
    rations: the limit exists because one vLLM's KV cache holds 1.62
    conversations. A single global pool got that wrong as soon as two sessions
    pointed at different endpoints — the default is ONE slot, so a sub-agent on
    endpoint A blocked every sub-agent on endpoint B from starting at all, for a
    GPU it was not using. Same correction, same reason, as the Telegram poll lock
    being keyed by bot token instead of by home directory.

    Slot files are empty and are never deleted; one set per endpoint ever used is
    a handful of 0-byte files, and deleting a locked path is how you get two
    holders of one slot. See utils/file for the rules.

    `endpoint` is the tag of the pool — passed in rather than read here, so one
    scan for a free slot cannot straddle an endpoint switch and compare slot 0 of
    one pool with slot 1 of another.
    """
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    return str(LOCKS_DIR / f'subagent.{endpoint}.{i}.lock')


def _try_slot(endpoint: str, i: int) -> Optional[SubagentSlot]:
    # An unscoped `subagent.{i}.lock` from before the pools were split may still
    # sit in LOCKS_DIR. It is not migrated and not removed: it carries no state,
    # and the only holder it could still exclude is a build that predates this
    # one and is gone by the time this runs.
    fd = os.open(_slot_path(endpoint, i), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _lock_nb(fd)
    except OSError:
        os.close(fd)
        return None
    return SubagentSlot(fd)


def try_acquire() -> Optional[SubagentSlot]:
    """A free slot right now, or None without waiting."""
    return _try_acquire_from(endpoint_tag())


def _try_acquire_from(endpoint: str) -> Optional[SubagentSlot]:
    """try_acquire against a named pool. The endpoint is resolved once per scan;
    acquire() re-resolves it on every poll, so switching endpoints while waiting
    moves the wait to the pool the request will actually use."""
    for i in range(_slots()):
        s = _try_slot(endpoint, i)
        if s is not None:
            return s
    return None


async def acquire(abort=None) -> Optional[SubagentSlot]:
    """Take a slot, waiting (polling) for one if all are busy. Returns None
    only if `abort` (an asyncio.Event) is set while waiting."""
    while True:
        s = try_acquire()
        if s is not None:
            return s
        if abort is not None and abort.is_set():
            return None
        await asyncio.sleep(2)
