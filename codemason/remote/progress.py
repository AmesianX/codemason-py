"""A spinner for the phone.

The terminal has one; the phone does not, and that asymmetry is what makes
remote work feel dead. You send an instruction and get nothing back until the
turn ends — a minute of silence that looks the same as a crash, so you sit there
wondering whether to send it again.

Telegram cannot stream: edits to one message are throttled to roughly one a
second, and a message per token would be unusable anyway. But one message that
keeps being rewritten is enough to feel alive, and costs no extra chat lines. So
a remote-initiated turn posts a single placeholder, keeps it current while the
turn runs, and finally becomes the answer itself.

Deliberately one message from start to finish. Sending "working…" and then the
answer separately would double every exchange and leave stale progress notes
above each reply.

One message, but not necessarily one *position*. Editing rewrites a message
where it already is, so anything posted meanwhile — a permission prompt most of
all, since answering one is a normal part of a turn — leaves the progress, and
the answer that eventually replaces it, stranded above. So it moves: while a
prompt is waiting it holds still (`pause`), and once something has landed below
it, the next update deletes and reposts it at the bottom. The invariant is that
the live message is the last one, because that is the only place a phone
reliably shows you.

Not only for turns you asked for from the phone. Work started at the keyboard
shows here too, because the point is being able to walk away mid-turn — that is
exactly when a live message earns its place, and it is the same one message
either way. What keeps it from narrating at you is the hold-off: a local turn
waits out `remote_notify_after` before posting, so a placeholder appears only
for turns the phone was going to hear about anyway, and the ones filtered out
as too short stay silent from beginning to end.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from ..utils.i18n import _

# Minimum gap between edits. Telegram throttles edits to a single message at
# roughly one per second; this leaves headroom while still reading as motion.
EDIT_INTERVAL = 2.5

# Below this, a turn finishes before a placeholder would even be useful — it
# would flash up and be replaced in the same breath.
MIN_DURATION_TO_SHOW = 3.0

_FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

# Held while a prompt is waiting on an answer from the phone.
#
# The bottom of the chat is where you look, so it belongs to whatever needs you
# right now — and while a permission prompt is up, that is the prompt, not a
# spinner claiming "working…" about a turn that is in fact stopped, waiting for you.
# Pausing also keeps the spinner from repeatedly reposting itself underneath a
# question it has no business burying.
#
# A counter, not a flag: two tools can be waiting at once, and the first to
# finish must not un-pause the other.
_pause_lock = threading.Lock()
_paused = 0


def pause() -> None:
    """Stop updating any live progress message until `resume`."""
    global _paused
    with _pause_lock:
        _paused += 1


def resume() -> None:
    global _paused
    with _pause_lock:
        _paused = max(0, _paused - 1)


def _is_paused() -> bool:
    with _pause_lock:
        return _paused > 0


class RemoteProgress:
    """Keeps one Telegram message current for the length of a turn."""

    def __init__(self, cwd: str = '', delay: Optional[float] = None):
        # How long to hold off before posting anything. Raised for local turns
        # so the placeholder appears exactly when the turn becomes one the phone
        # will hear about at all — see the caller in cli.py. Never lowered below
        # the floor: that one is not a filter but a rendering rule, and a
        # placeholder that flashes up and is replaced in the same breath is
        # worse than none whoever asked for the turn.
        self.delay = max(MIN_DURATION_TO_SHOW, delay or 0.0)
        self.cwd = cwd
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._text = ''                 # answer prose so far
        self._tools: Dict[str, int] = {}
        self._message_id: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._adapter = None
        self._frame = 0
        self._break = False             # a tool ran: the next text is a new block
        # Set under _lock by finish(): the turn has taken the message id back and
        # the updater no longer owns it. Correctness rests on this flag, never on
        # the join below completing in time — see _push.
        self._finished = False

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        try:
            from .notify import adapters
            for a in adapters():
                if a.is_configured() and hasattr(a, 'edit_progress'):
                    self._adapter = a
                    break
            if self._adapter is None:
                return
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name='codemason-remote-progress')
            self._thread.start()
        except Exception:
            pass

    def finish(self) -> Optional[int]:
        """Stop updating and hand back the message id to be rewritten as the
        answer, or None if nothing was ever posted.

        None is also the answer when the placeholder is no longer the last
        message — the usual case being a permission prompt answered near the end
        of the turn, or the turn ending while one is still on screen. Rewriting a
        stranded placeholder would file the answer above the prompt you just
        tapped, so it is dropped and the caller posts the answer fresh, at the
        bottom where you are looking.
        """
        # Take ownership of the message id FIRST, under the lock. The updater may
        # be inside a network call right now, and edit_progress can return a new
        # id (it deletes and reposts when something landed below). Whoever else
        # was mid-flight must not write that id back after this point.
        #
        # This used to rest on `t.join(timeout=5)` alone: if the join timed out —
        # a slow edit, a stalled connection — the thread stayed alive and its
        # assignment landed after the caller had already read the id, so the
        # answer was written over a message that was no longer the live one, or
        # the placeholder reappeared underneath the answer. A five-second guess
        # standing in for ownership.
        self._stop.set()
        with self._lock:
            self._finished = True
            mid = self._message_id
        t = self._thread
        if t is not None:
            # Tidiness only now: nothing below depends on this having returned.
            # One edit interval plus the adapter's own request timeout is the
            # longest a cycle can legitimately take.
            t.join(timeout=EDIT_INTERVAL + getattr(self._adapter, 'timeout', 10.0))
        if mid is None or self._adapter is None:
            return mid
        try:
            if not self._adapter.progress_is_last(mid):
                self._adapter.delete_message(mid)
                self._message_id = None
                return None
        except Exception:
            pass
        return mid

    # --- turn events ---------------------------------------------------------

    def on_text(self, delta: str) -> None:
        """Accumulate the answer prose, keeping the blocks apart.

        A turn that calls tools emits its text in separate blocks — a sentence
        before each call, another after the result — and appending the deltas
        raw ran them together: "…let me look at the structure.The project is
        large…". They are different paragraphs and the reader can see that on
        screen, where the tool call sits between them; on the phone the tool
        calls are only a count in the header, so the break has to be put back.
        """
        with self._lock:
            if self._break and delta.strip():
                self._text += '\n\n'
                self._break = False
            self._text += delta

    def on_tool(self, name: str) -> None:
        with self._lock:
            self._tools[name] = self._tools.get(name, 0) + 1
            # Whatever the model says next belongs after this call, not glued to
            # the sentence that introduced it.
            self._break = True

    # --- the updating loop ---------------------------------------------------

    def _run(self) -> None:
        # Hold off at first: most turns are quick, and a placeholder that appears
        # and is instantly replaced is worse than none.
        if self._stop.wait(self.delay):
            return
        while not self._stop.is_set():
            try:
                if not _is_paused():
                    self._push()
            except Exception:
                pass
            if self._stop.wait(EDIT_INTERVAL):
                break

    def _snapshot(self) -> str:
        with self._lock:
            text, tools = self._text, dict(self._tools)
        elapsed = time.monotonic() - self._started
        self._frame = (self._frame + 1) % len(_FRAMES)

        head = f'{_FRAMES[self._frame]} ' + _('working…') + f' ({elapsed:.0f}s'
        if tools:
            head += ' · ' + ' · '.join(
                f'{n}{"" if c <= 1 else f" ×{c}"}'
                for n, c in sorted(tools.items(), key=lambda kv: (-kv[1], kv[0])))
        head += ')'

        # Show the tail, not the head: the newest words are the ones that tell
        # you where it got to.
        if text.strip():
            tail = text.strip()[-500:]
            return f'{head}\n\n{tail}'
        return head

    def _push(self) -> None:
        # The id can change under us: when something has been posted below, the
        # adapter reposts at the bottom rather than editing in place, and the
        # repost is a new message. Taking the returned id back is what makes the
        # next edit — and the final answer — land on the one that is actually
        # there.
        new_id = self._adapter.edit_progress(
            self._message_id, self._snapshot(), self.cwd)
        with self._lock:
            if not self._finished:
                self._message_id = new_id
                return
            # finish() already handed the id to the turn while this call was in
            # flight. Writing it back would point the answer at a message the
            # caller does not know about. If the adapter reposted, the message it
            # created is ours to clean up — nothing else knows it exists, so
            # leaving it would strand a dead "working…" line in the chat.
            orphan = new_id if new_id and new_id != self._message_id else None
        if orphan is not None:
            try:
                self._adapter.delete_message(orphan)
            except Exception:
                pass
