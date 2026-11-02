"""The single inbound loop for a process.

Telegram's getUpdates is exclusive — each update is handed to exactly one caller
— so polling cannot simply be started wherever it is needed. The permission
prompt already long-polls while it waits; adding a second loop for incoming
messages would have the two fight over the same lock inside one process and steal
each other's updates. One loop, many consumers, is the only shape that works.

    callback_query  → resolves a pending question (permission, later choices)
    message         → filed into a session inbox for injection

Which consumer an update belongs to is decided here, so adapters stay dumb and
sessions never see traffic that is not theirs.

Ownership is cooperative, not assigned. The loop runs only while something needs
it and only while it holds the cross-process lock; a process that loses the race
simply has no remote input, and everything it does locally is unchanged. When the
holder exits the lock is released and the next session picks it up — the same
property that let us skip building a daemon.
"""

from __future__ import annotations

import atexit
import json
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ..utils.i18n import _

# One long-poll chunk. Also the worst-case delay before the loop notices it
# should stop, so it bounds how long a finished session keeps the lock from the
# next one.
POLL_CHUNK = 20

# A poll failure is almost always transient — a dropped connection, a blip, a
# Telegram 5xx. Over an overnight run one is close to certain, and treating it as
# fatal used to kill remote input for the rest of the session with no sign that
# anything had stopped: no buttons, no messages, silence that looks exactly like
# "nothing happened yet".
#
# So retry with backoff and keep going, indefinitely.
#
# There used to be a cap of 40 consecutive failures — "roughly half an hour" —
# after which the loop gave up for the rest of the session. The number was
# invented, and the reason given for it was not about failure at all: it was to
# stop a permanently broken setup (revoked token, no network) from holding the
# exclusive poll lock forever and starving the other sessions.
#
# That is a resource statement, so it has a resource answer. A failing loop now
# RELEASES the lock and keeps trying without it, re-taking it only when a poll
# succeeds. A broken session can no longer starve a working one, whatever the
# reason or however long it lasts — and the session that recovers at minute
# thirty-one resumes instead of having quietly become deaf at minute thirty.
RETRY_BASE = 2.0
RETRY_MAX = 60.0

# A 409 is NOT one of those transient failures, and must not be retried like one.
# Telegram answers 409 when a second getUpdates arrives for the same bot: two
# processes are sharing one token, and it hands each update to exactly one of
# them. Nothing is broken enough to notice — both sides keep working, each simply
# receives an arbitrary half of what you sent. That is the worst failure shape
# this channel has: no error, no gap, just messages that "sometimes" do not
# arrive, at whichever end you happen to be looking.
#
# The poll lock cannot prevent it. It lives under Path.home(), so it only ever
# serialises processes that share a home directory — a WSL2 session and a native
# Windows session on the same PC, or two machines, each hold their own lock file
# and neither can see the other's. Telegram's own 409 is the only signal that
# crosses those boundaries, which makes it the authority here, not a symptom.
#
# Two observations, and no clock.
#
# An earlier version also waited out a window of POLL_CHUNK + 5s, on the theory
# that a session which had just exited could leave its long poll open and that a
# successor would collide with that corpse. Measured against the live API, the
# theory is wrong and the window guarded against nothing:
#
#   Telegram sends 409 to the poll it TERMINATES — the one already waiting when a
#   new request arrives. So a 409 reaching us means some other caller issued a NEW
#   request. A dead process issues none; its socket is an open connection nobody
#   will ever write to again. Killing a process mid-getUpdates and polling
#   immediately after returns 200, three times over.
#
# A conflict is therefore positive evidence of a live rival, not a coincidence of
# timing to be waited out — and an arbitrary duration is exactly the wrong shape
# of guard: it delays a true detection by half a minute to protect against a case
# that cannot happen. Two consecutive conflicts, so that a single freak response
# is not enough to act on, and nothing else.
MAX_CONSECUTIVE_CONFLICTS = 2

# Floor on an empty cycle. getUpdates is *supposed* to hold the connection for
# POLL_CHUNK seconds, so the loop is normally self-pacing — but that pacing is
# the server's promise, not ours. Anything that returns "ok, nothing" instantly
# (a proxy short-circuiting, a gateway ignoring the timeout parameter) turns this
# into a spin that hammers the API tens of thousands of times a second and earns
# a rate-limit ban. Measured: a stub returning immediately produced ~1M calls in
# 20s. Cheap insurance against a failure mode we do not control.
MIN_CYCLE = 1.0

# How often a session that lost the lock checks whether it has been freed.
#
# The lock is not fair: there is no queue and no handoff, so whoever calls
# acquire first after a release wins. That is fine here — losing costs a session
# its remote input, not its correctness — but only if losers keep asking.
# Previously a session that started while another held the lock gave up for good,
# so an overnight run left every session started afterwards with no remote
# features at all, permanently, even long after the holder exited.
LOCK_RETRY = 15.0

_LOOP_LOCK = threading.Lock()
_instance: Optional['Poller'] = None


class Question:
    """A prompt awaiting a tap, keyed by the message its buttons live on."""

    def __init__(self, message_id: int, options: List[str], free_text: bool = False):
        self.message_id = message_id
        self.options = options
        self.free_text = free_text
        self.answer: Optional[str] = None
        self._done = threading.Event()

    def resolve(self, answer: str) -> None:
        self.answer = answer
        self._done.set()

    def wait(self, timeout: float, cancel: Optional[threading.Event] = None) -> Optional[str]:
        """Block until answered, cancelled, or out of time.

        Two ways an answer arrives. Usually this process is the one polling and
        resolves the question directly. But with several sessions the tap lands
        wherever the poll lock happens to be, and that receiver files it for us —
        so each slice also checks for a forwarded answer. Without that second
        path, a session that lost the lock could never be answered at all.

        Sliced rather than one long wait so a cancel — the terminal answering
        first — is noticed promptly instead of after the full timeout.
        """
        from . import inbox
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                return None
            if self._done.wait(min(0.5, max(0.0, deadline - time.monotonic()))):
                return self.answer
            data = inbox.take_answer(self.message_id)
            if data and data.startswith('p:'):
                try:
                    return self.options[int(data[2:])]
                except (ValueError, IndexError):
                    pass
            # A typed answer ("t:<text>"), filed by whoever read the message.
            # Only a prompt that offers "Other" takes one; anything else ignores
            # it rather than read a sentence as a choice.
            if data and data.startswith('t:') and self.free_text and data[2:].strip():
                return data[2:].strip()
        return None


# How many times a crashing poll loop is respawned before it stays down.
_MAX_POLL_RESTARTS = 5


class Poller:
    """Owns the inbound connection for this process."""

    def __init__(self, adapter):
        self.adapter = adapter
        # No start-time watermark here any more: what was queued before this
        # poller existed is asked of Telegram directly (_probe_backlog), which
        # answers by update id instead of by comparing clocks.
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._questions: Dict[int, Question] = {}
        self._q_lock = threading.Lock()
        self._users = 0                     # consumers keeping the loop alive
        self._lock_fd = None
        self._conflicts = 0                 # consecutive 409s (shared bot token)
        # update_ids that were already queued when this poller started. Recorded
        # by _probe_backlog, consumed by _dispatch — see both.
        self._backlog_ids: set = set()
        # False until a probe actually answered. When it never does, the first
        # fetch is used to classify held-over updates instead — see _run.
        self._probe_ok: bool = False
        self._first_fetch: bool = True
        # An acquire arrived while the loop was shutting down — see _start_locked.
        self._relaunch: bool = False
        # Times the loop was respawned after a crash (see _run_guarded).
        self._restarts: int = 0
        self.on_message: Optional[Callable[[str, Dict[str, Any]], None]] = None
        # Messages carrying photos are handled on a worker thread (_handoff): a
        # download can take a minute on a bad network, and the loop has to keep
        # answering button taps meanwhile. Anything that arrives while the
        # worker is busy queues behind it, so messages still reach the session
        # in the order they were sent ("photo", then "what is this?").
        self._work: 'queue.Queue' = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._work_lock = threading.Lock()
        self._work_pending = 0              # queued or running on the worker
        self._gen = 0                       # bumped per loop run; stale work is dropped
        # Offset bookkeeping (_commit): handed to the worker and not done yet;
        # dispatched since the offset last passed them; highest id done.
        self._inflight: set = set()
        self._seen: set = set()
        self._done_max: Optional[int] = None
        # How far the offset has been allowed to move. Everything below this was
        # handled — that is the whole rule the offset follows — so _seen only has
        # to remember the ids at or above it.
        #
        # Without this floor a photo was delivered twice, every time. _seen is
        # pruned the moment _commit moves the offset, but the fetch that is
        # ALREADY in flight went out with the old offset, and Telegram answers
        # it with the update again: back in the loop, _seen no longer holds the
        # id, so it looks new and the photo lands in the inbox a second time,
        # with a second turn to answer it. Only the photo path has the window
        # open — there the commit happens on the worker while the loop sits in a
        # long poll, where the other paths commit between two fetches.
        self._passed: int = 0

    # --- lifecycle -----------------------------------------------------------

    def acquire(self) -> None:
        """Register a consumer and start the loop if it is not already running."""
        with _LOOP_LOCK:
            self._users += 1
            self._start_locked()

    def _start_locked(self) -> None:
        """Spawn the loop if it is not already up. Call with _LOOP_LOCK held."""
        if self._thread is not None and self._thread.is_alive():
            # RACE. release() sets the stop event but the loop only notices at
            # its next check, so a release immediately followed by an acquire
            # (every /resume, every sub-agent that takes and gives back the
            # remote channel) lands here with the thread still alive and the
            # event already set. Returning early left a consumer attached to a
            # loop on its way out, and remote input died silently until
            # something else acquired. Hand the relaunch to the thread that is
            # leaving: it is the only one that knows when the old loop is gone.
            if self._stop.is_set():
                self._relaunch = True
            return
        self._relaunch = False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_guarded, name='codemason-remote-poller', daemon=True)
        self._thread.start()

    def _run_guarded(self) -> None:
        """_run, plus the two things a bare thread target does not do.

        threading.excepthook already puts a crash in error.log (utils/log.py),
        but nothing restarted the loop: it stayed down until something happened
        to call acquire() again, which in a remote-only session is never. The
        restart is bounded, because a crash that reproduces on the first poll
        would otherwise respawn forever and fill the log with it.
        """
        crashed = False
        try:
            self._run()
        except BaseException:
            crashed = True
            raise
        finally:
            with _LOOP_LOCK:
                if self._thread is threading.current_thread():
                    self._thread = None
                if crashed and self._users > 0 and not self._stop.is_set():
                    from ..utils.log import log_error
                    if self._restarts >= _MAX_POLL_RESTARTS:
                        log_error(
                            f'remote poll loop crashed {_MAX_POLL_RESTARTS} times — '
                            'giving up; remote input is off for the rest of this session',
                            'remote_poller: restart')
                    else:
                        self._restarts += 1
                        log_error(
                            f'remote poll loop crashed — restarting '
                            f'(attempt {self._restarts})',
                            'remote_poller: restart')
                        self._start_locked()
                elif self._relaunch and self._users > 0:
                    # Asked to come back while it was shutting down (_start_locked).
                    self._relaunch = False
                    self._start_locked()

    def release(self) -> None:
        """Drop a consumer; the loop stops once the last one is gone."""
        with _LOOP_LOCK:
            self._users = max(0, self._users - 1)
            if self._users == 0:
                self._stop.set()

    def stop(self) -> None:
        with _LOOP_LOCK:
            self._users = 0
            self._stop.set()

    # --- questions -----------------------------------------------------------

    def add_question(self, message_id: int, options: List[str],
                     free_text: bool = False) -> Question:
        q = Question(message_id, options, free_text)
        with self._q_lock:
            self._questions[message_id] = q
        return q

    def drop_question(self, message_id: int) -> None:
        with self._q_lock:
            self._questions.pop(message_id, None)

    # --- the loop ------------------------------------------------------------

    def _run(self) -> None:
        fd = self._await_lock()
        if fd is None:
            return                          # asked to stop while waiting
        self._lock_fd = fd
        self._reset_dispatch_state()
        try:
            from .adapters.telegram import prune_attachments
            prune_attachments()
        except Exception:
            pass
        # Ask what was already waiting before serving anything, so the first real
        # fetch can tell a held message from a live one.
        try:
            self._probe_backlog()
        except Exception:
            pass                            # _probe_ok stays False → first fetch decides
        failures = 0
        self._first_fetch = True
        try:
            while not self._stop.is_set():
                cycle_started = time.monotonic()
                updates = self._fetch()
                if updates is None:
                    if self._conflicts >= MAX_CONSECUTIVE_CONFLICTS:
                        if not self._reclaim_bot():
                            break
                        continue
                    failures += 1
                    # Hold the lock only while it is doing something. One failure
                    # can be a blip; a second means this process is not currently
                    # able to serve the bot, and there is no reason for it to keep
                    # every other session off it while that lasts.
                    if failures >= 2 and self._lock_fd is not None:
                        # Not with a photo still on the worker: whoever polls
                        # next would be handed it again.
                        self._drain_worker()
                        self.adapter._release_poll_lock(self._lock_fd)
                        self._lock_fd = None
                        from ..utils.log import debug_log
                        debug_log('REMOTE_POLLER_YIELDED_LOCK', {'failures': failures})
                    # Backoff, but stay responsive to shutdown: waiting on the
                    # stop event rather than sleeping means a session exiting
                    # during a network outage still releases the lock at once.
                    delay = min(RETRY_MAX, RETRY_BASE * (2 ** min(failures - 1, 6)))
                    if self._stop.wait(delay):
                        break
                    # Take it back BEFORE fetching again. Polling without the
                    # lock is never allowed: another session would have picked it
                    # up while we were backing off, and two callers on one bot is
                    # the collision this whole file exists to avoid — we would
                    # have manufactured it ourselves, then reported it. If that
                    # session is still holding, this one waits, which is the
                    # correct outcome and the reason letting go was safe.
                    if self._lock_fd is None:
                        self._lock_fd = self._await_lock()
                        if self._lock_fd is None:
                            break           # asked to stop while waiting
                    continue
                if failures:
                    from ..utils.log import debug_log
                    debug_log('REMOTE_POLLER_RECOVERED', {'after_failures': failures})
                    failures = 0
                # Fail closed when the probe could not answer. Anything in the
                # FIRST fetch was queued before we asked for anything — we had not
                # yet listened — so it is held-over by the same definition
                # _probe_backlog uses, and can be classified without it. Without
                # this the old path treated an unanswerable probe as "everything is
                # live" and a message sent days ago ran the instant a session came
                # up. Running someone's stale instruction unasked is the worse
                # error; offering a live one costs a keypress.
                if self._first_fetch:
                    self._first_fetch = False
                    if not self._probe_ok:
                        for upd in updates:
                            try:
                                self._backlog_ids.add(int(upd.get('update_id')))
                            except (TypeError, ValueError):
                                continue
                        if self._backlog_ids:
                            from ..utils.log import debug_log
                            debug_log('REMOTE_BACKLOG_FROM_FIRST_FETCH',
                                      {'count': len(self._backlog_ids)})
                # While the offset waits for a photo on the worker, getUpdates
                # hands back what was already dispatched; only the rest is new.
                fresh = [u for u in updates if self._is_new(u)]
                for group in _group_albums(fresh):
                    try:
                        if len(group) == 1:
                            self._dispatch(group[0])
                        else:
                            self._dispatch_album(group)
                    except Exception:
                        continue            # one bad update must not kill the loop

                # Only paced when nothing arrived. A burst of real updates should
                # be drained as fast as it comes; it is the empty-and-instant
                # cycle that has to be slowed — including the one that only
                # returns updates already dispatched.
                if not fresh:
                    slack = MIN_CYCLE - (time.monotonic() - cycle_started)
                    if slack > 0 and self._stop.wait(slack):
                        break
        finally:
            # May already be None: a failing loop yields the lock and only takes
            # it back when it is ready to poll again, so the exit path can find
            # nothing to release.
            if self._lock_fd is not None:
                self.adapter._release_poll_lock(self._lock_fd)
                self._lock_fd = None

    def _reclaim_bot(self) -> bool:
        """Another instance took our bot. Move to a free one; True if we did.

        Standing down was right while a bot was something you configured: two
        instances on one token meant a mistake only the user could fix. Now the
        bot is claimed rather than chosen, so a conflict is not a misconfiguration
        — it is a claim that was lost, most likely to an instance that started at
        the same moment and probed the same bot before either was polling. Losing
        a race is not a reason to switch the phone off for the rest of the
        session; it is a reason to take the next bot.

        The current bot is dropped BEFORE claiming so the winner is left alone,
        and the poll lock goes with it: the lock is named after the token, so a
        different bot is a different lock.
        """
        from .adapters import telegram as _tg
        from ..utils.log import debug_log

        self._drain_worker()
        if self._lock_fd is not None:
            self.adapter._release_poll_lock(self._lock_fd)
            self._lock_fd = None
        _tg.release_bot()
        # The offset is per bot: nothing recorded against the old one applies.
        self._reset_dispatch_state()
        debug_log('REMOTE_BOT_LOST', {'conflicts': self._conflicts})
        # log_error too, not only debug_log: debug_log is a no-op unless
        # CODEMASON_DEBUG is set, and this is the moment remote input stops
        # working. Without a record, the morning after looks like "Telegram just
        # stopped" with nothing on disk to say the bot was handed over.
        from ..utils.log import log_error
        log_error(
            f'gave up the Telegram bot after {self._conflicts} conflicts (409) — '
            'another poller is using the same token',
            'remote_poller: bot lost')

        if not _tg.claim_bot():
            self._report_no_free_bot()
            return False
        self._conflicts = 0
        self._backlog_ids.clear()           # a different bot has a different queue
        # ...and a different queue has to be classified again from scratch, by the
        # probe or, if it cannot answer, by the next fetch.
        self._probe_ok = False
        self._first_fetch = True
        self._lock_fd = self._await_lock()
        if self._lock_fd is None:
            return False                    # asked to stop while waiting
        try:
            self._probe_backlog()
        except Exception:
            pass
        return True

    def _report_no_free_bot(self) -> None:
        """Say that the pool is exhausted, and stand down.

        Reached only after every bot in the pool has been tried, so it is a real
        shortfall and not a lost race: there are more instances running than there
        are bots to receive them.

        Printed, not debug-logged. debug_log is off unless CODEMASON_DEBUG is set,
        which is exactly why the shared-token version of this was invisible for
        weeks while every symptom pointed somewhere else. Something the user
        resolves in a minute with BotFather has to reach the user.
        """
        from ..utils.log import debug_log
        from .adapters import telegram as _tg
        try:
            size = len(_tg.bot_pool())
        except Exception:
            size = 0
        debug_log('REMOTE_POOL_EXHAUSTED',
                  {'pool': size, 'verdicts': _tg.last_claim_verdicts()})
        from ..utils.log import log_error
        log_error(f'no free bot in a pool of {size} — remote input is off',
                  'remote_poller: pool exhausted')
        try:
            print()
            # One reporter, so the mid-session failure explains itself the same
            # way the startup one does — including when the cause is a dropped
            # network rather than a full pool.
            from .input import _report_no_free_bot
            _report_no_free_bot(size)
            print(_('  Local input, tools and output are unaffected.'))
        except Exception:
            pass

    def _await_lock(self):
        """Take the poll lock, retrying until we get it or are told to stop.

        It keeps asking because sessions come and go, and the one that started
        second should pick up polling the moment the first exits.

        Standing by used to be documented as normal — "another session is
        polling, this one simply has no remote input meanwhile" — and that was
        true when every session shared one bot. It is not true now. The lock is
        per bot (poll.<token>.lock) and each session claims its own out of the
        pool, so a session that has claimed a bot is the only thing that should
        ever want that bot's lock. Waiting on it is therefore a fault, not a
        queue: a lock file left by a dead run, or two sessions that resolved to
        the same token.

        Classifying it as normal is what made it invisible, and invisible is the
        whole of how it presented: the bot claimed, its name printed, outbound
        working — because sending needs no lock — and inbound never starting, in
        silence, with the only record a debug_log line behind a flag that is off
        by default. So it now says so, on screen and on the phone, once, naming
        the file and whoever the file says is holding it. That is the difference
        between a fault you can act on and one that looks like an idle session.
        """
        waited = False
        while not self._stop.is_set():
            fd = self.adapter._acquire_poll_lock()
            if fd is not None:
                if waited:
                    from ..utils.log import debug_log
                    debug_log('REMOTE_POLLER_TOOK_OVER', {})
                return fd
            if not waited:
                waited = True
                self._report_standby()
            if self._stop.wait(LOCK_RETRY):
                break
        return None

    def _report_standby(self) -> None:
        """Say that inbound is not starting, and why, exactly once."""
        holder, path = '', ''
        try:
            from .adapters import telegram as _tg
            holder = _tg.poll_lock_holder()
            path = str(_tg._lock_path())
        except Exception:
            pass
        from ..utils.log import debug_log
        debug_log('REMOTE_POLLER_STANDBY', {'holder': holder, 'lock': path})
        detail = _('Waiting for the poll lock held by process {pid} ({path}). '
                   'Messages sent from the phone will not arrive until this '
                   'clears.', pid=holder or '?', path=path or '?')
        try:
            from ..utils.formatting import safe_print
            safe_print('\n⚠️  ' + detail)
        except Exception:
            pass
        try:
            from .events import EventKind, Sensitivity
            from .notify import notify
            notify(EventKind.INFO, _('Inbound messages are not being received'),
                   body=detail, sensitivity=Sensitivity.INTERNAL)
        except Exception:
            pass

    def _probe_backlog(self) -> None:
        """Record which updates were already queued before we started listening.

        "Sent while nothing was listening" is a state, not an age, and Telegram
        will answer it directly: getUpdates with timeout=0 returns exactly what is
        pending right now and nothing else. Whatever comes back was queued before
        this poller existed — by definition, since we had not yet asked. Anything
        arriving after this call arrived while we were listening.

        This replaces comparing each message's `date` against the poller's start
        time. That comparison could only ever approximate the answer: it made a
        message sent one second before startup old and one sent one second after
        it fresh, though nothing about either is different to the user, and it
        needed a second age threshold downstream to paper over the cases it got
        wrong. The ids are recorded rather than the timestamps, so the decision is
        exact and does not drift with clocks.

        Deliberately does NOT consume them. The offset is left alone, so the very
        next real fetch returns these same updates and they travel the normal
        path — only now the loop knows which ones need asking about first.
        """
        resp = self.adapter._post('getUpdates', {
            'offset': self.adapter._read_offset(),
            'timeout': 0,
            'allowed_updates': json.dumps(['callback_query', 'message']),
        }, http_timeout=15)
        if not resp or not resp.get('ok'):
            return                          # _probe_ok stays False → first fetch decides
        self._probe_ok = True
        for upd in resp.get('result') or []:
            try:
                self._backlog_ids.add(int(upd.get('update_id')))
            except (TypeError, ValueError):
                continue
        if self._backlog_ids:
            from ..utils.log import debug_log
            debug_log('REMOTE_BACKLOG_PROBED', {'count': len(self._backlog_ids)})

    def _fetch(self) -> Optional[List[dict]]:
        resp = self.adapter._post('getUpdates', {
            'offset': self.adapter._read_offset(),
            'timeout': POLL_CHUNK,
            # Messages are needed too now, not just button taps.
            'allowed_updates': json.dumps(['callback_query', 'message']),
        }, http_timeout=POLL_CHUNK + 10)    # must outlast the long poll
        if not resp or not resp.get('ok'):
            error = (resp or {}).get('error', 'no response')
            # Counted separately from ordinary failures: this one is diagnosed and
            # stopped, not retried. _post has already turned the HTTPError into
            # 'HTTP 409 <description>', so the status is matched rather than the
            # wording, which Telegram is free to change.
            if 'HTTP 409' in str(error):
                self._conflicts += 1
            from ..utils.log import debug_log
            debug_log('REMOTE_POLL_FAILED', {'error': error})
            return None
        self._conflicts = 0
        return resp.get('result') or []

    def _dispatch(self, upd: dict) -> None:
        # A missing/non-int update_id used to fall back to 0, which wrote offset 1
        # and made the NEXT fetch replay everything Telegram still held (~24h) as
        # though it had just been typed — _first_fetch is already False by then, so
        # none of it lands in _backlog_ids either. Skip the update instead: never
        # move the offset somewhere we did not read from.
        try:
            update_id = int(upd['update_id'])
        except (KeyError, TypeError, ValueError):
            from ..utils.log import log_error
            log_error('telegram update without a usable update_id — skipped',
                      'remote_poller: dispatch')
            return

        # ORDER. The offset is what Telegram replays from, so writing it first
        # makes every update at-most-once: a crash or a kill between that write
        # and the inbox write drops the message for good, and the user sees their
        # instruction simply never arrive. Writing it AFTER the handler makes the
        # input path at-least-once instead, which is the right trade here — the
        # handler's job is to put the text in the inbox file, so the only way to
        # get a duplicate is a crash in the window between that file write and
        # this one, while the failure it removes is silent loss of a command
        # nobody knows was sent.
        #
        # The two INTERACTIVE paths keep the early write: both can sit on a
        # Telegram picker for as long as the user takes to tap, and an offset that
        # has not moved by then means getUpdates hands the same update back on the
        # next poll and opens a second picker for it.
        cq = upd.get('callback_query')
        if cq:
            self._commit(update_id)
            self._on_callback(cq)
            return

        msg = upd.get('message')
        if not msg:
            self._commit(update_id)
            return
        # Anything that appears in the chat pushes an in-flight progress message
        # off the bottom — including what you typed yourself. Recorded here
        # rather than in _on_message so it counts even when this process has no
        # inbox attached, and for photos and stickers that are never injected.
        if self.adapter._is_trusted_chat(msg):
            try:
                from .adapters.telegram import note_chat_message
                note_chat_message(msg.get('message_id'))
            except Exception:
                pass
        if self._needs_worker([msg]):
            self._handoff([update_id], lambda live: self._on_message(msg, update_id, live=live))
            return
        self._on_message(msg, update_id)
        self._commit(update_id)

    def _dispatch_album(self, group: list) -> None:
        """Several photos sent together arrive as one update each, sharing a
        media_group_id; usually only the first carries a caption. Delivered one
        by one they became one turn per photo, the later ones steered into a turn
        already answering the first. Handled as one message instead, offset
        moved past the whole group afterwards (same at-least-once order as
        _dispatch)."""
        first = group[0]['message']
        if self.adapter._is_trusted_chat(first):
            try:
                from .adapters.telegram import note_chat_message
                for upd in group:
                    note_chat_message(upd['message'].get('message_id'))
            except Exception:
                pass
        try:
            ids = [int(u['update_id']) for u in group]
        except (KeyError, TypeError, ValueError):
            return                          # same rule as _dispatch: never guess an offset
        extra = [u['message'] for u in group[1:]]
        if self._needs_worker([first, *extra]):
            self._handoff(ids, lambda live: self._on_message(first, ids[0], extra=extra, live=live))
            return
        self._on_message(first, ids[0], extra=extra)
        for i in ids:
            self._commit(i)

    # --- offset and the photo worker -----------------------------------------

    def _reset_dispatch_state(self) -> None:
        with self._work_lock:
            self._gen += 1
            self._inflight.clear()
            self._seen.clear()
            self._done_max = None
            self._passed = 0

    def _is_new(self, upd: dict) -> bool:
        """False for an update already dispatched (and not yet behind the offset)."""
        try:
            uid = int(upd['update_id'])
        except (KeyError, TypeError, ValueError):
            return True                     # _dispatch reports and skips it
        with self._work_lock:
            if uid < self._passed:
                return False                # the offset is already past it: handled
            if uid in self._seen:
                return False
            self._seen.add(uid)
            return True

    def _commit(self, update_id: int) -> None:
        """Record `update_id` as handled and move the offset as far as that allows.

        The offset is what Telegram replays from, and it deletes everything
        below it. So it may pass an update only once every update before it is
        handled: while a photo message is on the worker, later updates are
        handled but the offset waits for it (and _is_new skips them when
        getUpdates returns them again)."""
        with self._work_lock:
            self._inflight.discard(update_id)
            self._done_max = update_id if self._done_max is None else max(self._done_max, update_id)
            target = min(self._inflight) if self._inflight else self._done_max + 1
            self._passed = max(self._passed, target)
            self._seen = {u for u in self._seen if u >= target}
        try:
            if target > self.adapter._read_offset():
                self.adapter._write_offset(target)
        except Exception:
            pass

    def _needs_worker(self, msgs: list) -> bool:
        if any(_image_refs(m) for m in msgs if self.adapter._is_trusted_chat(m)):
            return True
        with self._work_lock:
            return self._work_pending > 0   # keep the order behind a photo

    def _handoff(self, ids: list, handle) -> None:
        """Run `handle(live)` on the worker, then commit `ids`. `live()` turns
        False once this loop run is over (stopped, bot changed): the handler
        must not deliver then — the next run is handed the same update again,
        since its offset was never committed."""
        with self._work_lock:
            self._inflight.update(ids)
            self._work_pending += 1
            gen = self._gen
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work_loop, daemon=True,
                                                name='remote-photo-worker')
                self._worker.start()
        self._work.put((gen, ids, handle))

    def _live(self, gen: int) -> bool:
        return gen == self._gen and not self._stop.is_set()

    def _work_loop(self) -> None:
        while True:
            gen, ids, handle = self._work.get()
            try:
                if not self._live(gen):
                    continue
                try:
                    handle(lambda: self._live(gen))
                except Exception:
                    pass                    # same as the loop: one bad update is skipped
                if self._live(gen):
                    for i in ids:
                        self._commit(i)
            finally:
                with self._work_lock:
                    self._work_pending -= 1

    def _drain_worker(self) -> None:
        """Wait (stop-aware) until nothing is left on the worker."""
        while not self._stop.is_set():
            with self._work_lock:
                if self._work_pending == 0:
                    return
            self._stop.wait(0.2)

    def _on_callback(self, cq: dict) -> None:
        # The sender of a callback query is on the QUERY, not on the message it
        # is attached to — that message was posted by the bot. Passing the
        # message alone meant a button tap was authenticated by "the bot posted
        # this prompt in the configured chat", which is always true.
        if not self.adapter._is_trusted_chat(cq.get('message') or {},
                                             sender=cq.get('from') or {}):
            return
        mid = (cq.get('message') or {}).get('message_id')
        data = cq.get('data') or ''
        if self._on_picker(cq, mid, data):
            return
        if data == OTHER_DATA:
            self._on_other(cq, mid)
            return
        if not data.startswith('p:'):
            return
        with self._q_lock:
            q = self._questions.get(mid)

        if q is not None:
            try:
                answer = q.options[int(data[2:])]
            except (ValueError, IndexError):
                return
            self.adapter._post('answerCallbackQuery',
                               {'callback_query_id': cq.get('id', ''), 'text': answer})
            q.resolve(answer)
            return

        # Not ours. It probably belongs to another session: prompts are posted by
        # whoever needs them, but answers arrive at whoever holds the poll lock,
        # and those are routinely different processes. Discarding here is what
        # made remote approval stop working the moment a second session existed.
        from . import inbox
        owner = inbox.question_owner(mid)
        if owner:
            # Forward the raw choice index, not a label: the option text lives in
            # the process that posted the prompt, and it is the one that maps it
            # back. Guessing here would risk answering with the wrong option.
            self.adapter._post('answerCallbackQuery',
                               {'callback_query_id': cq.get('id', ''), 'text': _('Forwarded')})
            inbox.deliver_answer(mid, data)
            return

        # Genuinely unknown — an old button from a session long gone. Acknowledge
        # so the phone stops spinning, then drop it: a stale tap must never
        # resolve a live question.
        self.adapter._post('answerCallbackQuery',
                           {'callback_query_id': cq.get('id', ''),
                            'text': _('This request has expired')})

    def _on_other(self, cq: dict, mid: int) -> None:
        """The "Other" button: say how to answer, and leave the prompt open.

        A button cannot carry a sentence, so the tap itself answers nothing —
        the typed message that follows does (see questions.answer_by_text). The
        alert is the whole job: without it the button looks broken.
        """
        from . import inbox
        with self._q_lock:
            live = mid in self._questions
        if live or inbox.question_owner(mid):
            text = _('Type your answer and send it as a message')
            alert = 'true'
        else:
            text = _('This request has expired')
            alert = 'false'
        self.adapter._post('answerCallbackQuery',
                           {'callback_query_id': cq.get('id', ''), 'text': text,
                            'show_alert': alert})

    def _with_images(self, text: str, images: list) -> str:
        """Download the images and name them in the message text.

        The session's input is a line of text (queue, mid-turn steering and echo
        all carry strings), so a photo travels as the path it was saved to and
        the model opens it with Read — the one way images enter a conversation,
        which is also what resizes it, numbers it, keeps the original in the
        session's image store and counts it against the server's limit. A
        download that fails is said in the text: the user sent something, and
        dropping it silently reads as "it ignored my photo"."""
        lines = [text] if text else []

        def fetch(ref):
            try:
                return self.adapter.download_image(*ref)
            except Exception as e:
                return '', str(e) or type(e).__name__

        # An album's photos download side by side, not one after another.
        if len(images) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(4, len(images))) as pool:
                results = list(pool.map(fetch, images))
        else:
            results = [fetch(images[0])]
        for path, err in results:
            if path:
                lines.append(f'[Photo from the phone: {path}]')
            else:
                lines.append(f'[A photo was sent from the phone but could not be downloaded: {err}]')
        return '\n'.join(lines)

    def _on_bot_command(self, text: str) -> None:
        """Answer a bot-owned command instead of feeding it to a session.

        Doubles as the one place to see what is running: /start and /help are the
        first thing you send from a phone, so they may as well tell you which
        sessions exist and how to reach them.
        """
        from . import inbox
        sessions = inbox.live_sessions()
        if sessions:
            lines = [f'• <code>{s.get("label") or s["session_id"][:8]}</code>'
                     for s in sessions]
            body = (_('Running sessions:') + '\n' + '\n'.join(lines) + '\n\n' +
                    (_('Just send a message and it goes there.') if len(sessions) == 1 else
                     _('<b>Reply</b> to a session alert to route there.\n'
                       'Send without replying and you will be asked which one.')))
        # Only promise delivery when something will actually deliver it:
        # adopting held messages is opt-in (see input.py), so the old promise
        # holds only when that flag is on. Both strings are already translated.
        elif getattr(__import__('codemason.utils.config', fromlist=['get_config'])
                     .get_config(), 'remote_replay_backlog', False):
            body = _('No sessions are running.\n'
                     'Messages are held and delivered once one starts.')
        else:
            body = _('Messages sent before a session started are not delivered')
        token, chat = _credentials_for(self.adapter)
        if not (token and chat):
            return
        resp = self.adapter._post('sendMessage', {
            'chat_id': chat,
            'text': f'🤖 <b>codemason</b>\n\n{body}',
            'parse_mode': 'HTML',
            'disable_notification': 'true',
        })
        try:
            from .adapters.telegram import note_chat_message
            note_chat_message(((resp or {}).get('result') or {}).get('message_id'))
        except Exception:
            pass

    def _on_picker(self, cq: dict, mid: int, data: str) -> bool:
        """Handle a tap on a "which session?" picker. True if it was one."""
        if not data.startswith('r:'):
            return False
        from . import inbox
        parked = inbox.take_route(mid)
        self.adapter._post('answerCallbackQuery', {'callback_query_id': cq.get('id', '')})
        if not parked:
            return True                     # already routed, or long expired
        try:
            target = parked['choices'][int(data[2:])]
        except (ValueError, IndexError, KeyError):
            return True
        if not target:
            # The discard button on a held-over message. Nothing to deliver and
            # nothing to keep: take_route already removed the parked copy, and
            # that is the whole intent of the tap.
            self.adapter.note_routed(mid, '')
            return True
        # take_route has already consumed the parked copy, so a failed delivery
        # here loses the message outright — the one place in this file where that
        # is possible. Put it back and say so, rather than reporting a route that
        # did not happen: the picker stays answerable and nothing is invented.
        if not inbox.deliver(target, inbox.InboundMessage(
                text=parked.get('text', ''), source='telegram',
                meta=parked.get('meta') or {})):
            inbox.park_route(mid, parked.get('text', ''),
                             parked.get('meta') or {}, parked.get('choices') or [])
            try:
                from .events import EventKind, Sensitivity
                from .notify import notify
                notify(EventKind.ERROR, _('Message not delivered'),
                       body=_('That session\'s inbox is locked by another '
                              'process. Nothing was written — tap again.'),
                       sensitivity=Sensitivity.PUBLIC)
            except Exception:
                pass
            return True
        self.adapter.note_routed(mid, target)
        return True

    def _is_backlog(self, update_id) -> bool:
        """Was this update already queued before we started listening?

        Answered from the probe taken at startup, not from the message's age.
        `_probe_backlog` asked Telegram what was pending before we had asked for
        anything, so membership in that set IS the answer — no clock, no
        comparison, no threshold that is right either side of one second.
        """
        try:
            if getattr(__import__('codemason.utils.config', fromlist=['get_config'])
                       .get_config(), 'remote_replay_backlog', False):
                return False
        except Exception:
            pass
        try:
            return int(update_id) in self._backlog_ids
        except (TypeError, ValueError):
            return False

    def _offer_backlog(self, text: str, meta: dict) -> None:
        """Hold a held-over message and ask, instead of dropping or running it.

        Both of the old answers were guesses about intent that only the sender
        can make. Running it was dangerous — a queued `/clear` wiped a
        conversation the moment a session came up. Dropping it was quieter but
        no better: the count was reported and the contents were not, so you were
        told "something you sent was ignored" and could not see what, and the
        message was gone either way.

        So it is neither run nor discarded: it is put in front of you with its
        text, and a button per live session, and a button to discard. Exactly the
        picker an ambiguous message already gets, which is the same question —
        "where should this go, if anywhere?" — and the same machinery answers it.
        """
        from . import inbox
        # Scope the offer to the session that owns this bot. The question worth
        # asking about a held-over message is whether to run it at all — it is
        # old, and running it blind is how a queued `/clear` wiped a conversation.
        # Which session it belongs to is not open: it arrived through a bot one
        # session claimed exclusively. Listing every session on the machine turned
        # one answerable question into two, and made the second unanswerable
        # whenever the labels matched.
        from .input import _owning_session_id
        owner = _owning_session_id()
        sessions = [s for s in inbox.live_sessions()
                    if not owner or s['session_id'] == owner]
        try:
            from .notify import adapters
            for adapter in adapters():
                if not adapter.is_configured() or not hasattr(adapter, 'pick'):
                    continue
                labels = [s.get('label') or s['session_id'][:8] for s in sessions]
                picker_id = adapter.pick(
                    _('Sent while nothing was listening — run it now?'),
                    (text or '')[:200],
                    labels + [_('Discard')],
                )
                if picker_id:
                    # The discard choice is the one index past the sessions, so
                    # _on_picker resolves it without a second store to consult.
                    inbox.park_route(picker_id, text, meta or {},
                                     [s['session_id'] for s in sessions] + [''])
                    return
        except Exception:
            pass
        # Nothing could ask. Holding it unread is the honest fallback: the offset
        # is only advanced past updates we have dealt with, so it is still queued
        # server-side and the next session that can ask will offer it again.
        from ..utils.log import debug_log
        debug_log('REMOTE_BACKLOG_UNASKED', {'text': (text or '')[:80]})

    def _on_message(self, msg: dict, update_id=None, extra=(), live=None) -> None:
        if not self.adapter._is_trusted_chat(msg):
            return
        # A photo's words are its caption; in an album each photo can have one.
        parts = [(m.get('text') or m.get('caption') or '').strip() for m in (msg, *extra)]
        text = '\n'.join(dict.fromkeys(p for p in parts if p))
        images = [ref for m in (msg, *extra) for ref in _image_refs(m)]
        if not text and not images:
            return                          # stickers, joins — not input
        if images:
            text = self._with_images(text, images)
            if live is not None and not live():
                return                      # this loop run ended during the download

        # Telegram sends /start by itself when you first open a bot, and offers
        # /help in its own UI — neither is something you meant to say to a
        # session, and injecting them means the model is suddenly answering
        # "/start". These belong to the bot; everything else, including
        # codemason's own slash commands like /clear, is genuine input. A photo
        # is never a bot command, whatever its caption starts with.
        if not images and _is_bot_command(text):
            if self.on_message:
                self._on_bot_command(text)
            return

        meta = {
            'message_id': msg.get('message_id'),
            'reply_to': (msg.get('reply_to_message') or {}).get('message_id'),
            'thread_id': msg.get('message_thread_id'),
            # When the user actually sent it, not when we read it. Telegram holds
            # updates for ~24h while nobody polls, so a session starting in the
            # morning receives everything sent overnight — and an instruction
            # written hours ago can mean something very different now. Carrying
            # the send time lets the session show the age instead of injecting it
            # as if it had just been typed.
            'sent_at': msg.get('date'),
        }
        if self._is_backlog(update_id):
            if not self.on_message:
                return
            # The other interactive path (see _dispatch): _offer_backlog waits on
            # a picker, so move the offset before blocking on it.
            if update_id is not None:
                try:
                    self._commit(int(update_id))
                except (TypeError, ValueError):
                    pass
            self._offer_backlog(text, meta)
            return

        # A question is waiting and you typed instead of tapping: that is the
        # answer. Checked here, ahead of the session's input, because the turn
        # is blocked on the question — filed as the next instruction, the text
        # sat unread until a button was tapped. Here rather than in the
        # session's router so a headless -p run, which has no router, gets it
        # too.
        try:
            from .questions import answer_by_text
            if answer_by_text(text, meta):
                return
        except Exception:
            pass
        if not self.on_message:
            return
        try:
            self.on_message(text, meta)
        except Exception:
            pass


# Commands the BOT owns. Deliberately tiny: codemason has its own slash commands
# (/clear, /compact, /export…) and those must still reach the session, so only
# names Telegram itself surfaces are intercepted.
_BOT_COMMANDS = ('/start', '/help', '/status', '/sessions')

# Callback payload of the "Other" button on a question. Not 'p:<n>': it selects
# nothing, and a positional payload would be read as a choice by older code.
OTHER_DATA = 'o'


def _credentials_for(adapter):
    try:
        from .adapters.telegram import _credentials
        return _credentials()
    except Exception:
        return ('', '')


def _image_refs(msg: dict) -> list:
    """(file_id, name hint) of each image a message carries. A photo comes in
    several sizes, largest last; an image sent as a file keeps its original
    bytes and name."""
    refs = []
    photo = msg.get('photo') or []
    if photo:
        best = max(photo, key=lambda p: (p.get('width', 0) * p.get('height', 0), p.get('file_size', 0)))
        if best.get('file_id'):
            refs.append((best['file_id'], 'photo.jpg'))
    doc = msg.get('document') or {}
    if doc.get('file_id') and str(doc.get('mime_type', '')).startswith('image/'):
        refs.append((doc['file_id'], doc.get('file_name') or ''))
    return refs


def _group_albums(updates: list) -> list:
    """Consecutive updates of one media group as one list; everything else alone."""
    groups: list = []
    for upd in updates:
        gid = (upd.get('message') or {}).get('media_group_id')
        prev = groups[-1][-1] if groups else None
        if gid and prev is not None and (prev.get('message') or {}).get('media_group_id') == gid:
            groups[-1].append(upd)
        else:
            groups.append([upd])
    return groups


def _is_bot_command(text: str) -> bool:
    head = text.split()[0].split('@')[0].lower()   # '/help@mybot' → '/help'
    return head in _BOT_COMMANDS


def get(adapter=None) -> Optional['Poller']:
    """The process-wide poller, created on first use."""
    global _instance
    if _instance is None:
        if adapter is None:
            from .adapters.telegram import TelegramAdapter
            adapter = TelegramAdapter()
        _instance = Poller(adapter)
    return _instance


@atexit.register
def _shutdown() -> None:
    if _instance is not None:
        _instance.stop()
