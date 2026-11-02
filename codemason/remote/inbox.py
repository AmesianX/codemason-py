"""Per-session message inboxes.

Only one process may long-poll Telegram — getUpdates hands each update to exactly
one caller — so with several sessions running, whoever holds the poll lock also
receives messages meant for the others. Rather than adding a daemon to own the
connection, the lock holder demultiplexes: every message is filed under its
target session's id and each session reads only its own inbox. When the holder
exits, the next session reclaims the lock (PID-checked) and takes over polling.

`session_id` is the only routing key, deliberately. A transport may have a richer
native notion of "which conversation" — Telegram forum topics, a browser tab in a
future PWA — but that stays inside the adapter as a mapping to a session id.
Building the core on a transport's own concept would leak it everywhere and make
the second adapter painful.

Message management is the whole job here:

  ordering    append-only JSONL; readers advance a byte cursor, never reorder
  exactly-once  a cursor advances only after messages are handed to the caller,
                and handing over cannot fail (the caller appends to an unbounded
                in-memory queue), so the loss window is microseconds and there is
                no redelivery
  concurrency   writer and readers hold an flock over the inbox for the whole
                read-or-append, so a partial line is never observed
  bounded       an inbox that nobody drains is capped and drops OLDEST first,
                leaving a visible marker — silent truncation would read as "you
                sent nothing"
  reclaimed     inboxes of dead sessions are swept, so a crashed run does not
                leak files forever
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..utils.platform import chmod_posix

# File locking differs by platform. POSIX has flock; Windows has msvcrt.locking,
# which locks a byte range *at the current position* — putting it on the data file
# would tangle with the cursor logic that already owns that position, so Windows
# locks a sidecar file beside it instead. A failure to lock is not fatal: a rarely
# interleaved line beats a session that will not start over a notification channel.
try:
    import fcntl
    msvcrt = None
except ImportError:                     # Windows
    fcntl = None
    try:
        import msvcrt
    except ImportError:                 # neither available: run without locking
        msvcrt = None

ROOT = Path.home() / '.codemason' / 'remote'
INBOX_DIR = ROOT / 'inbox'
SESSION_DIR = ROOT / 'sessions'

# Messages with no identifiable target (the user typed into the chat without
# naming a session). Kept separate rather than guessed at — see `take_unrouted`.
UNROUTED = '_unrouted'

# An inbox nobody drains must not grow without bound. Generous enough that a
# session busy for minutes loses nothing.
MAX_MESSAGES = 200
MAX_BYTES = 256 * 1024

# Retention splits by what a file HOLDS, not by which directory it sits in.
#
# Anything carrying the user's own words — an undrained inbox, a message parked
# on a picker — is content, and deleting content on a timer is destroying data
# nobody asked to lose. A held-over message offered with buttons used to be swept
# an hour later, so a prompt saying "run it now?" quietly answered itself if you
# did not look in time; the whole point of offering it was to leave the decision
# with the sender.
#
# A question claim and a forwarded answer are not content. They are in-flight
# coordination, consumed within one prompt round, and one still on disk after an
# hour outlived whatever made it — keeping those longer preserves nothing.
#
# The long window is a floor, not a schedule: sweeping is a backstop for crashed
# runs, and `codemason` deliberately leaves files alone rather than tidying
# behind the user. Explicit removal is `/remote clean`.
RETENTION_DAYS_DEFAULT = 30.0
STALE_QUESTION = 3600


def retention_seconds() -> float:
    """How long content-bearing files are kept. Config: remote_retention_days."""
    try:
        from ..utils.config import get_config
        days = float(getattr(get_config(), 'remote_retention_days',
                             RETENTION_DAYS_DEFAULT))
    except Exception:
        days = RETENTION_DAYS_DEFAULT
    return max(0.0, days) * 24 * 3600


# Kept as a module attribute because callers pass it explicitly (sweep's default
# argument is resolved at call time from retention_seconds()).
STALE_AFTER = RETENTION_DAYS_DEFAULT * 24 * 3600


@dataclass
class InboundMessage:
    """One message from a remote client, on its way into a session."""
    text: str
    source: str = 'telegram'
    ts: float = 0.0
    meta: Dict[str, Any] = None          # transport extras (message id, topic…)

    def to_json(self) -> str:
        return json.dumps({
            'text': self.text,
            'source': self.source,
            'ts': self.ts or time.time(),
            'meta': self.meta or {},
        }, ensure_ascii=False)

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> 'InboundMessage':
        return cls(
            text=str(obj.get('text', '')),
            source=str(obj.get('source', 'unknown')),
            ts=float(obj.get('ts', 0) or 0),
            meta=obj.get('meta') or {},
        )


def _mkdir_private(path: Path) -> None:
    """mkdir + chmod 0700, for every directory under the remote root.

    exist_ok=True leaves an already-created directory at whatever mode it had,
    and a bare mkdir() lands 0755 under a normal umask — so the question/answer/
    owner/route directories were world-readable with 0644 files inside. Anything
    able to write answers/<id> approves an arbitrary pending tool call, and the
    ids are small sequential integers readable from questions/. They were
    contained only by ROOT's 0700, which telegram.py notes is a no-op on Windows.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        chmod_posix(path, 0o700)
    except OSError:
        pass


def _write_private(path: Path, text: str) -> None:
    """Write one side-store record atomically.

    These files are read by other processes while they are being written, so the
    publish has to be the rename and not the write. See the questions/ comment.
    """
    from ..utils.file import atomic_write_bytes
    atomic_write_bytes(str(path), text.encode('utf-8'))


def _ensure_dirs() -> None:
    _mkdir_private(INBOX_DIR)
    _mkdir_private(SESSION_DIR)
    try:
        chmod_posix(ROOT, 0o700)
    except OSError:
        pass


def _inbox_path(session_id: str) -> Path:
    return INBOX_DIR / f'{session_id}.jsonl'


def _cursor_path(session_id: str) -> Path:
    return INBOX_DIR / f'{session_id}.cursor'


class LockUnavailable(OSError):
    """The inbox could not be locked, so the caller must not touch it."""


# How long acquiring may be retried before giving up on this attempt. Not a
# guess about contention: the critical sections here are a read and an append,
# measured in milliseconds, so anything still blocked after seconds is a wedged
# or stopped process rather than a busy one. Honoured on BOTH platforms now —
# see _lock_posix for why a blocking flock was the wrong half of that.
_LOCK_RETRY_SECONDS = 10.0
_LOCK_RETRY_SLEEP = 0.05


class _Locked:
    """flock a file for the whole read-or-append.

    Always BINARY. The cursor is a byte offset, and text-mode seek() is only
    defined for offsets produced by tell() — feeding it a byte count corrupts the
    decoder the moment a message contains a multi-byte character, which for this
    user is the common case, not the edge case.

    The lock is coarse on purpose. Relying on O_APPEND atomicity instead only
    holds for writes under PIPE_BUF, and a pasted paragraph blows past that; a
    reader seeing half a line is worse than the microseconds this costs.

    Failing to lock raises. It used to carry on unlocked — "best effort" — which
    is the one option that loses data silently: two unsynchronised appends
    interleave, `take` cannot parse the mangled line, and its `except: continue`
    drops it. A message the user sent, gone, with nothing anywhere saying so.
    Raising hands the decision to the caller, and both callers can do something
    better than corrupt the file: a reader retries on its next tick with nothing
    lost, and a writer says out loud that it could not deliver.
    """

    def __init__(self, path: Path, mode: str):
        self.path, self.mode, self.fh = path, mode, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, self.mode)
        self._side = None
        try:
            if fcntl is not None:
                self._lock_posix()
            elif msvcrt is not None:
                self._side = open(str(self.path) + '.lock', 'a+b')
                self._lock_windows()
            # Neither available: the platform offers no advisory locking at all,
            # which is not the same as an acquisition failing. Proceeding is the
            # only behaviour there is, and it was already the documented one.
        except Exception:
            if self._side is not None:
                try:
                    self._side.close()
                except OSError:
                    pass
                self._side = None
            self.fh.close()
            raise LockUnavailable(f'could not lock {self.path}')
        return self.fh

    def _lock_posix(self) -> None:
        """Non-blocking flock in a deadline loop, not one blocking LOCK_EX.

        A bare blocking LOCK_EX is uninterruptible. This is reached from the poll
        loop (poller._on_picker -> inbox.deliver), so one peer holding an inbox —
        including a hung or SIGSTOPped one, since flock survives SIGSTOP — parked
        the whole inbound loop indefinitely, with nothing logged and nothing to
        see but "Telegram stopped working". Same ceiling the Windows path already
        honoured, so a lock freed after 40ms is still taken at 40ms, and the
        caller gets the LockUnavailable it already treats as "do not touch the
        inbox".
        """
        deadline = time.monotonic() + _LOCK_RETRY_SECONDS
        while True:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except InterruptedError:
                continue                    # EINTR: retry at once, no penalty
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_LOCK_RETRY_SLEEP)
            except OSError:
                # EBADF, or a filesystem with no flock at all: fail exactly as a
                # blocking attempt would have.
                raise

    def _lock_windows(self) -> None:
        """Retry a non-blocking lock rather than one long blocking attempt.

        msvcrt.locking's blocking mode retries internally for about ten seconds
        and then raises, so a single attempt gave one all-or-nothing verdict with
        no way to distinguish "busy for a moment" from "wedged". Same ceiling,
        asked repeatedly, so a lock freed after 40ms is taken at 40ms.
        """
        deadline = time.monotonic() + _LOCK_RETRY_SECONDS
        while True:
            try:
                self._side.seek(0)
                msvcrt.locking(self._side.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_LOCK_RETRY_SLEEP)

    def __exit__(self, *exc):
        try:
            if fcntl is not None:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None and self._side is not None:
                try:
                    self._side.seek(0)
                    msvcrt.locking(self._side.fileno(), msvcrt.LK_UNLCK, 1)
                finally:
                    self._side.close()
        finally:
            self.fh.close()
        return False


# --- session registry --------------------------------------------------------

def short_label(cwd: str) -> str:
    """A human-sized name for a working directory.

    The basename alone breaks at exactly the place people run things from: the
    home directory becomes the account name, so a session started in `~` was
    labelled `user` — which says nothing about the work and quietly puts the
    username in every alert. Home is `~`, one level under it keeps the parent for
    context, anything deeper is just its own name.
    """
    try:
        p = Path(cwd or os.getcwd()).resolve()
        home = Path.home().resolve()
        if p == home:
            return '~'
        rel = p.relative_to(home)
        parts = rel.parts
        return '~/' + parts[0] if len(parts) == 1 else parts[-1]
    except (ValueError, OSError):
        pass
    # Outside home. A root has no basename — `Path('C:/').name` and `Path('/').name`
    # are both '' — and the fallback for that was the literal '?', which is not a
    # name at all: every session started at a drive root showed up in the phone's
    # picker as '?', and two of them were indistinguishable. The anchor IS the
    # name there ('C:\', '/'), and it is what the user typed to get there.
    try:
        p = Path(cwd or '.')
        return p.name or p.anchor or '?'
    except (ValueError, OSError):
        return '?'


def register(session_id: str, *, cwd: str = '', label: str = '') -> None:
    """Announce a live session so the poller can route to it.

    Also exports the session id and this program's path to the environment, and
    so to every child it spawns: a watcher script the session writes can address
    the session it belongs to, and re-enter this program, without being told
    either. They exist exactly when a wake can land, since registration is also
    what starts the drain.
    """
    from ..utils.wake_cli import export_self
    export_self(session_id)
    _ensure_dirs()
    path = SESSION_DIR / f'{session_id}.json'
    payload = {
        'session_id': session_id,
        'pid': os.getpid(),
        'cwd': cwd or os.getcwd(),
        'label': label or distinct_label(short_label(cwd), session_id),
        'started': time.time(),
    }
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)                     # atomic — a reader never sees a partial entry


def unregister(session_id: str) -> None:
    # Withdraw the exported id too: anything spawned after this reads the
    # absence and refuses, rather than addressing a session that has stopped
    # taking messages.
    from ..utils.wake_cli import SESSION_ENV, unexport_self
    if os.environ.get(SESSION_ENV) == session_id:
        unexport_self()
    for p in (SESSION_DIR / f'{session_id}.json', _inbox_path(session_id), _cursor_path(session_id)):
        try:
            p.unlink()
        except OSError:
            pass


def unregister_registry_only(session_id: str) -> None:
    """Stop advertising the session but keep whatever is still in its inbox.

    A message that arrived while the session was shutting down is still the
    user's; `sweep` reclaims it later once the process is provably gone. Deleting
    it here would drop it with no trace.
    """
    # Withdraw the exported id too: anything spawned after this reads the
    # absence and refuses, rather than addressing a session that has stopped
    # taking messages.
    from ..utils.wake_cli import SESSION_ENV, unexport_self
    if os.environ.get(SESSION_ENV) == session_id:
        unexport_self()
    try:
        (SESSION_DIR / f'{session_id}.json').unlink()
    except OSError:
        pass


def _alive(pid: int) -> bool:
    from ..utils.platform import pid_alive
    return pid_alive(pid)


def live_sessions() -> List[Dict[str, Any]]:
    """Registered sessions whose process is still running, newest first.

    Labels are settled at registration by `distinct_label`, so the suffix below
    is a backstop for the one case that cannot: two sessions registering at the
    same instant both read a registry without the other in it and pick the same
    name. Rare, and a session-id slice is the right answer when it happens —
    unreadable, but never wrong, and only reached when nothing better exists.
    """
    _ensure_dirs()
    out: List[Dict[str, Any]] = []
    for p in SESSION_DIR.glob('*.json'):
        try:
            entry = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if _alive(int(entry.get('pid', -1))):
            out.append(entry)
    out.sort(key=lambda e: e.get('started', 0), reverse=True)

    seen: Dict[str, int] = {}
    for e in out:
        seen[e.get('label', '')] = seen.get(e.get('label', ''), 0) + 1
    for e in out:
        if seen.get(e.get('label', ''), 0) > 1:
            e['label'] = f'{e.get("label", "")} ({e["session_id"][:6]})'
    return out


def distinct_label(base: str, session_id: str = '') -> str:
    """`base`, or the first free `base (2)`, `base (3)`, … among live sessions.

    Settled once, at registration, rather than patched on when a picker is drawn.
    The two are not the same thing: a name invented at display time is not the
    name the bot already carries in the phone's chat list, so the button and the
    chat it routed to disagreed about what the session was called.

    The first session in a directory keeps the plain name. That matters beyond
    tidiness — naming a bot is rate limited per bot on a cooldown of hours, so a
    name that changed on every restart would spend the quota and take the user's
    own `/rc <name>` down with it.
    """
    taken = {s.get('label', '') for s in live_sessions()
             if s.get('session_id') != session_id}
    if base not in taken:
        return base
    n = 2
    while f'{base} ({n})' in taken:
        n += 1
    return f'{base} ({n})'


def sweep(stale_after: Optional[float] = None, *, force: bool = False) -> int:
    """Reclaim everything left behind by sessions that are gone.

    A backstop for crashed runs, not tidying. `stale_after` defaults to the
    configured retention (30 days) rather than a day, because most of what is
    here is the user's own words and losing those to a timer is not housekeeping.
    `force=True` ignores the window entirely — that is `/remote clean`, where the
    user has said so explicitly and does not need to be second-guessed.

    Sweeping the registry alone is not enough, and used to be the whole of it.
    A clean shutdown removes the registry entry but keeps the inbox (a message
    that arrived mid-shutdown is still the user's), which left the file with
    nothing pointing at it — so the next sweep, walking only the registry, could
    not see it. Orphans accumulated silently and forever. Every directory is now
    swept on its own terms.
    """
    _ensure_dirs()
    if stale_after is None:
        stale_after = retention_seconds()
    if force:
        stale_after = 0.0
    removed = 0
    now = time.time()
    live = {e['session_id'] for e in live_sessions()}

    # 1. Registry entries whose process is gone.
    for p in SESSION_DIR.glob('*.json'):
        try:
            entry = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            entry = {}
        if _alive(int(entry.get('pid', -1))):
            continue
        started = float(entry.get('started', 0) or 0)
        if now - max(started, p.stat().st_mtime) < stale_after:
            continue
        unregister(p.stem)
        removed += 1

    # 2. Inboxes with no live owner — including the ones a clean shutdown left
    #    behind, which the registry pass can no longer reach.
    #
    #    `.lock` is the Windows sidecar (_Locked): the data file's own position is
    #    spoken for by the cursor, so the lock is taken on a file beside it. It is
    #    never unlinked at release — deleting a lock another process may be holding
    #    is how you get two writers — so it retires here with the inbox it guarded.
    #    Left out of this list it would be the one file nothing ever collects.
    for p in (list(INBOX_DIR.glob('*.jsonl')) + list(INBOX_DIR.glob('*.cursor'))
              + list(INBOX_DIR.glob('*.lock'))):
        sid = p.name.split('.')[0]
        if sid in live or sid == UNROUTED:
            continue
        try:
            if now - p.stat().st_mtime < stale_after:
                continue
            p.unlink()
            removed += 1
        except OSError:
            pass

    # 3. In-flight coordination. A question claim and a forwarded answer are
    #    consumed within one prompt round, so one still here outlived whatever
    #    made it and keeping it preserves nothing. These are the only files short
    #    enough to age out in an hour.
    for d in (QUESTION_DIR, ANSWER_DIR):
        if not d.exists():
            continue
        for p in d.iterdir():
            try:
                if not force and now - p.stat().st_mtime < STALE_QUESTION:
                    continue
                p.unlink()
                removed += 1
            except OSError:
                pass

    # 4. Content and routing, on the long window.
    #
    #    ROUTE_DIR holds the user's own text: a message parked on a picker,
    #    waiting to be told where it goes. It was swept with the question claims
    #    above, on an hour — so a held-over message offered with "run it now?"
    #    answered itself if nobody looked in time, which is exactly the decision
    #    the offer exists to leave with the sender.
    #
    #    OWNER_DIR is a message id mapped to the session that posted it, which is
    #    what makes replying to an alert route back. Tiny, and only useful while
    #    the alert is still visible in the chat — which is far longer than an
    #    hour, and costs nothing to keep.
    for d in (ROUTE_DIR, OWNER_DIR):
        if not d.exists():
            continue
        for p in d.iterdir():
            try:
                if now - p.stat().st_mtime < stale_after:
                    continue
                p.unlink()
                removed += 1
            except OSError:
                pass

    return removed


# --- delivery ----------------------------------------------------------------

def deliver(session_id: str, message: InboundMessage) -> bool:
    """File one message for `session_id` (called by whoever holds the poll lock).

    False when the inbox could not be locked and the message was NOT filed. The
    caller must say so rather than assume delivery: an instruction that vanishes
    reads as "codemason ignored me", which is the failure this whole path exists
    to avoid. Never writes unlocked as a fallback — an interleaved line is
    unparseable, and an unparseable line is dropped on read, which is the same
    loss with the evidence destroyed too.
    """
    _ensure_dirs()
    path = _inbox_path(session_id)
    try:
        with _Locked(path, 'ab+') as fh:
            fh.seek(0, os.SEEK_END)
            fh.write((message.to_json() + '\n').encode('utf-8'))
            fh.flush()
            _enforce_caps(fh, session_id)
    except LockUnavailable:
        return False
    return True


def _enforce_caps(fh, session_id: str) -> None:
    """Trim an undrained inbox, oldest first, and say so.

    Dropping the NEWEST would be worse: the message you just sent is the one you
    are waiting on. Dropping silently would be worse still — a vanished
    instruction reads as "codemason ignored me".
    """
    fh.seek(0)
    lines = fh.read().split(b'\n')
    if lines and lines[-1] == b'':
        lines.pop()
    size = sum(len(l) + 1 for l in lines)
    if len(lines) <= MAX_MESSAGES and size <= MAX_BYTES:
        return
    dropped = 0
    while lines and (len(lines) > MAX_MESSAGES or size > MAX_BYTES):
        gone = lines.pop(0)
        size -= len(gone) + 1
        dropped += 1
    if dropped:
        notice = InboundMessage(
            text=f'(inbox overflow — {dropped} older message(s) dropped)',
            source='system',
            ts=time.time(),
        ).to_json().encode('utf-8')
        lines.insert(0, notice)
    fh.seek(0)
    fh.truncate()
    if lines:
        fh.write(b'\n'.join(lines) + b'\n')
    fh.flush()
    # The cursor is a byte offset into a file we just rewrote, so it now points
    # at nothing meaningful. Reset it — re-reading a trimmed inbox is correct,
    # skipping past live messages is not.
    try:
        _cursor_path(session_id).unlink()
    except OSError:
        pass


def take(session_id: str) -> List[InboundMessage]:
    """Return messages queued since the last call, and advance the cursor.

    The cursor moves only after the messages are in the caller's hands, and the
    caller's next step (appending to an in-memory queue) cannot fail — so this is
    effectively exactly-once without the redelivery a crash-safe ack would cause.
    """
    path = _inbox_path(session_id)
    if not path.exists():
        return []
    cursor_file = _cursor_path(session_id)
    try:
        return _take_locked(path, cursor_file)
    except LockUnavailable:
        # Nothing is lost by coming back: the cursor has not moved, so the same
        # messages are still there for the next tick a second from now. Reading
        # unlocked to avoid the wait would be the trade that costs a message.
        return []


def _take_locked(path: Path, cursor_file: Path) -> List[InboundMessage]:
    out: List[InboundMessage] = []
    with _Locked(path, 'rb+') as fh:
        try:
            start = int(cursor_file.read_text().strip() or 0)
        except Exception:
            start = 0
        size = path.stat().st_size
        if start > size:
            start = 0                      # file was trimmed or replaced
        fh.seek(start)
        blob = fh.read()
        end = start + len(blob)
        for raw in blob.split(b'\n'):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(InboundMessage.from_obj(json.loads(raw.decode('utf-8'))))
            except Exception:
                continue                   # a corrupt line must not stall the inbox
        if out or blob:
            try:
                cursor_file.write_text(str(end))
            except OSError:
                pass
        # Fully drained and already large: reclaim the space now that no reader
        # needs the history.
        if end >= size and size > MAX_BYTES // 2:
            fh.seek(0)
            fh.truncate()
            try:
                cursor_file.write_text('0')
            except OSError:
                pass
    return out


# --- question ownership ------------------------------------------------------
#
# A prompt is posted by one session but its answer arrives at whichever process
# holds the poll lock — which is often a different one. Without a shared record
# of who asked, the receiver finds no matching question, discards the tap as
# stale, and the session that actually asked waits forever. That is not a corner
# case: it is what happens to every prompt from the moment a second session
# exists.
#
# One small file per outstanding question, written atomically (temp + rename, via
# utils.file.atomic_write_bytes). No lock needed — a message id is unique, so
# there is exactly one writer — but the write still has to be atomic: the reader
# is ANOTHER process polling for its own claims, and a plain write truncates
# first, so it could read an empty or half-written claim and discard the tap as
# stale. Which is the failure this whole side-store exists to prevent.

QUESTION_DIR = ROOT / 'questions'


def _bot_tag() -> str:
    """Fingerprint of the bot this process polls through.

    Question claims and forwarded answers were keyed on the bare Telegram
    message id, with a comment reasoning that "a message id is unique, so there
    is exactly one writer". That premise does not hold here: the adapter runs a
    POOL of bots, each with its own chat and therefore its own id sequence
    starting at 1, so two concurrent sessions routinely hold prompts with the
    same id. A tap in session A's chat could then be consumed by session B, and
    after _reclaim_bot rotated bots an old answer file could resolve a later
    prompt with no user interaction at all.

    telegram.py already solved exactly this for `last_message` by keying per
    token; questions, answers and owners never got the same treatment. Imported
    lazily because telegram imports this module, not the other way round.
    """
    try:
        from .adapters.telegram import token_tag, _credentials
        token, _ = _credentials()
        return token_tag(token)
    except Exception:
        return 'none'


def _scoped(message_id: int) -> str:
    """Filename for a per-question file, scoped to the bot that asked."""
    return f'{_bot_tag()}-{message_id}'


def claim_question(message_id: int, session_id: str, options: int = 0,
                   free_text: bool = False) -> None:
    """Record that `session_id` is the one waiting on this prompt.

    `options` is how many choices the prompt offered. A yes/no word typed in the
    chat is turned into a POSITIONAL answer ("p:0" / "p:1"), and this machinery
    carries both two-option permission prompts and AskUserQuestion — so without
    the count, "no" on a three-way model-authored question silently selected
    whatever option 1 happened to be.

    `free_text` marks a prompt that takes a typed answer of its own (an
    AskUserQuestion, whose picker always offers "Other"). Recorded here rather
    than inferred from the option count, because whichever process holds the
    poll lock is the one that reads a typed message, and it may not be the one
    that asked.
    """
    try:
        _mkdir_private(QUESTION_DIR)
        _write_private(QUESTION_DIR / _scoped(message_id),
                       f'{session_id}\n{options}\n{1 if free_text else 0}')
    except OSError:
        pass


def question_accepts_text(message_id: int) -> bool:
    """True when the prompt behind `message_id` takes a typed answer.

    False for a claim without the flag — a permission prompt, or one written
    before the flag existed — so a typed sentence can never become a grant.
    """
    try:
        lines = (QUESTION_DIR / _scoped(message_id)).read_text(encoding='utf-8').splitlines()
        return lines[2].strip() == '1'
    except (OSError, IndexError):
        return False


def question_option_count(message_id: int) -> Optional[int]:
    """Options the prompt behind `message_id` offered, or None if not recorded.

    None for a claim written before the count was stored, in which case the
    caller must not assume two.
    """
    try:
        lines = (QUESTION_DIR / _scoped(message_id)).read_text(encoding='utf-8').splitlines()
        return int(lines[1].strip())
    except (OSError, IndexError, ValueError):
        return None


def question_owner(message_id: int) -> Optional[str]:
    try:
        raw = (QUESTION_DIR / _scoped(message_id)).read_text(encoding='utf-8')
        # First line only: the claim also records the prompt's option count.
        return (raw.splitlines() or [''])[0].strip() or None
    except OSError:
        return None


def pending_questions() -> Dict[int, str]:
    """Outstanding prompts, message id -> owning session.

    Claims are released as soon as a prompt resolves, so what remains is exactly
    what is still waiting on someone. Used to let a typed "y" answer a prompt
    instead of being read as conversation.
    """
    out: Dict[int, str] = {}
    if not QUESTION_DIR.exists():
        return out
    prefix = f'{_bot_tag()}-'
    for p in QUESTION_DIR.iterdir():
        # Only this bot's claims. Another bot's ids are a different sequence and
        # answering one of those here is the collision this scoping prevents.
        if not p.name.startswith(prefix):
            continue
        try:
            out[int(p.name[len(prefix):])] = p.read_text(encoding='utf-8').strip()
        except (ValueError, OSError):
            continue
    return out


OWNER_DIR = ROOT / 'owners'


def claim_message(message_id: int, session_id: str) -> None:
    """Record which session posted an outbound message.

    Makes replying a routing signal: answer any alert from a session and the text
    goes back to that session, with no picker and nothing to remember.

    Kept in its own directory rather than sharing the question store, even though
    both map a message id to a session. `pending_questions()` reads that store to
    decide whether a typed "y" is answering something — and every notification
    ever sent would look like an outstanding prompt, so a "y" could resolve
    against an hour-old alert.
    """
    if not (message_id and session_id):
        return
    try:
        _mkdir_private(OWNER_DIR)
        _write_private(OWNER_DIR / str(message_id), session_id)
    except OSError:
        pass


def message_owner(message_id: int) -> Optional[str]:
    try:
        return (OWNER_DIR / str(message_id)).read_text(encoding='utf-8').strip() or None
    except OSError:
        return None


def release_question(message_id: int) -> None:
    try:
        (QUESTION_DIR / _scoped(message_id)).unlink()
    except OSError:
        pass


ANSWER_DIR = ROOT / 'answers'


def deliver_answer(message_id: int, data: str) -> None:
    """Leave a button tap where the session that asked will find it.

    Kept out of the message inbox on purpose. That inbox has exactly one drainer
    per session — the REPL's input loop — and a headless `-p` run has none at
    all, so an answer filed there would either be swallowed by the input path or
    never read. One file per question sidesteps both: no second drainer, no lock,
    and the waiter already knows the id it is looking for.
    """
    try:
        _mkdir_private(ANSWER_DIR)
        _write_private(ANSWER_DIR / _scoped(message_id), data)
    except OSError:
        pass


ROUTE_DIR = ROOT / 'routes'


def park_route(picker_id: int, text: str, meta: Dict[str, Any],
               choices: List[str]) -> None:
    """Hold a message whose target session is being chosen.

    Only reached when the intent is genuinely ambiguous. Parking beats guessing:
    an instruction delivered to the wrong session is acted on immediately and by
    the wrong code, which is far worse than one extra tap.
    """
    try:
        _mkdir_private(ROUTE_DIR)
        _write_private(ROUTE_DIR / str(picker_id),
                       json.dumps({'text': text, 'meta': meta, 'choices': choices},
                                  ensure_ascii=False))
    except OSError:
        pass


def take_route(picker_id: int) -> Optional[Dict[str, Any]]:
    path = ROUTE_DIR / str(picker_id)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    try:
        path.unlink()
    except OSError:
        pass
    return data


def take_answer(message_id: int) -> Optional[str]:
    """Consume a forwarded answer for this question, if one arrived."""
    path = ANSWER_DIR / _scoped(message_id)
    try:
        data = path.read_text(encoding='utf-8').strip()
    except OSError:
        return None
    # An answer must POSTDATE the question it answers. Bot scoping already stops
    # ids colliding across the pool, but a leftover answer from a session that
    # died on this same bot could still sit here and resolve a later prompt that
    # happens to reuse the id — with nobody having tapped anything. Consumption
    # had no time check at all; the only bound was the hour-long sweep.
    try:
        claim = QUESTION_DIR / _scoped(message_id)
        if path.stat().st_mtime < claim.stat().st_mtime:
            path.unlink()
            return None
    except OSError:
        # No claim on disk (already released, or never written): nothing to
        # compare against, so fall through rather than discard a real answer.
        pass
    try:
        path.unlink()
    except OSError:
        pass
    return data or None


def deliver_unrouted(message: InboundMessage) -> Optional[str]:
    """File a message that did not name a session.

    A message that came in through a claimed bot has an owner already — the bot
    is exclusive to one session — so that is consulted before counting anything.
    Failing that, exactly one live session makes the intent unambiguous. With
    none or several and no claim, it is parked instead of guessed at: sending an
    instruction to the wrong session is worse than asking which one.

    Returns the session id it was routed to, or None if parked.

    Currently unreferenced — `_adopt_unrouted` files these itself. Left correct
    rather than left wrong: a helper carrying the machine-wide model is how that
    model spreads back in the next time somebody reaches for it.
    """
    try:
        from .input import _owning_session_id
        owner = _owning_session_id()
    except Exception:
        owner = ''
    if owner:
        deliver(owner, message)
        return owner

    sessions = live_sessions()
    if len(sessions) == 1:
        sid = sessions[0]['session_id']
        deliver(sid, message)
        return sid
    deliver(UNROUTED, message)
    return None


def take_unrouted() -> List[InboundMessage]:
    return take(UNROUTED)
