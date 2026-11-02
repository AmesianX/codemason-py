"""Remote input — messages from the phone become session input.

The last link: the poller files inbound messages into per-session inboxes, and a
session drains its own into the queue its prompt already reads from.

That queue is the whole trick. codemason's REPL was already built to accept lines
typed while a turn is running and hold them until the turn ends — the behaviour
you get by typing mid-turn and pressing Esc. Chat is asynchronous by nature, so a
message sent from a phone lands in exactly that queue and inherits the same
semantics for free. No new interrupt path, no Esc equivalent, nothing to explain.

Routing is by session id and nothing else. When only one session is live the
intent is unambiguous and the message goes there; with several it is parked
rather than guessed at, because delivering an instruction to the wrong session is
worse than asking which one. A transport with a richer native notion of "which
conversation" — Telegram forum topics, a browser tab later — maps that to a
session id inside its adapter and the core never learns about it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.i18n import _

from . import inbox

# How often a session checks its inbox. Cheap (a stat and usually no read), and
# well under the threshold where typing feels ignored.
DRAIN_INTERVAL = 1.0

# There is deliberately no age threshold here any more.
#
# There used to be one: a message older than five minutes was offered rather than
# run, because "starting a session delivers everything sent since the last one
# ended". That is a real problem and it is not this layer's to solve — the poller
# now asks Telegram which updates were already queued before it began listening
# (poller._probe_backlog) and offers exactly those, by identity rather than by
# age. Whatever reaches this loop arrived while a session was listening.
#
# What remained was a second guess on top of a settled fact, and its only live
# effect was wrong: the picker parks a message until you tap a session, so a
# message you had just chosen to route arrived "stale" and was refused for being
# older than the time you took to answer. Delivery state is what the cursor
# already records, exactly as Claude Code's mailbox uses a `read` flag and never
# looks at a clock.

# How often a running session reclaims its own leftovers.
#
# sweep() used to run once, at start, on the theory that what it collects is left
# by runs that are already gone. That holds for inboxes; it does not hold for the
# small per-message files. `claim_message` writes one file per outbound alert so a
# reply can be routed back — and with remote_notify_after at its default of 0 that
# is one file per turn, forever, in a session that may stay up for days. Nothing
# collected them until the NEXT session started, which on a long-lived box is not
# a bound at all.
#
# Cheap enough to do on a timer: a few directory listings, and only the entries
# already past their own staleness window are touched.
SWEEP_INTERVAL = 900.0


def _claim_bot_for(cwd: str, label: str = '') -> bool:
    """Claim a bot for this session, preferring the one this session slot used.

    Preferring it is not a cache, it is continuity: reclaiming the same bot keeps
    a window under the same name in the phone's chat list across restarts, which
    is the difference between a stable list of sessions and a shuffle. Costs one
    probe when it works, and falls through to the scan when somebody else has it.

    The preference is remembered per session slot, not per directory. Keyed by
    directory alone it was one record shared by every window opened there, so a
    second session overwrote the first's bot and both came back on restart
    preferring the same one — the continuity this exists for, lost precisely when
    there was more than one thing to keep continuous. The slot is the label, which
    is stable across restarts for the same reason it is stable in the chat list.
    """
    from .adapters import telegram as _tg
    from ..utils import session_storage as _ss

    remembered = ''
    try:
        saved = _ss.read_project_settings(cwd)
        # Directories that only ever ran one session still carry the old
        # unsuffixed key; honour it for the first slot so their bot survives the
        # upgrade instead of being reshuffled once, silently.
        remembered = str(saved.get(f'remote_bot:{label}')
                         or (saved.get('remote_bot', '') if label == inbox.short_label(cwd) else '')
                         or '')
    except Exception:
        pass
    # Stored as a fingerprint, never the token: project settings are ordinary
    # files in the project tree's shadow, and a credential does not belong there.
    pool = _tg.bot_pool()
    preferred = next((t for t in pool if _tg.token_tag(t) == remembered), '')

    token = _tg.claim_bot(preferred)
    if not token:
        return False
    try:
        _ss.write_project_setting(f'remote_bot:{label}', _tg.token_tag(token), cwd=cwd)
    except Exception:
        pass
    # A claimed bot keeps the name it already has. Naming it here derived the
    # name from the working directory, and short_label() answers '~' for the home
    # directory by design -- it is a picker label, where '~' sits among other
    # entries and deliberately avoids printing the account name. As a bot name it
    # is alone in the phone's chat list and says nothing, so every session started
    # in $HOME renamed its bot to '~'.
    #
    # Deriving a better string is the wrong fix: the name belongs to the bot, not
    # to whichever directory happened to claim it, and renaming is rate limited
    # per bot on a cooldown of hours -- an automatic rename on every claim spends
    # that quota on a name nobody asked for and can leave `/rc` unable to set one.
    # `/rc <name>` renames, and nothing else does; it calls set_bot_name directly.
    return True


def _report_no_free_bot(pool_size: int) -> None:
    """Say why the phone is silent, naming the situation that actually occurred.

    Three ways to end up with no bot, and they need three different things done
    about them. Reporting them as one — "all N are in use" — sent the user to
    BotFather to create bots they did not need whenever the wifi had dropped, and
    said nothing at all about a token that had been revoked.

    Printed rather than logged: a session that is quietly unreachable looks
    exactly like one that is broken.
    """
    if pool_size == 0:
        return                              # unconfigured is not a shortfall

    from .adapters import telegram as _tg
    verdicts = _tg.last_claim_verdicts()
    counts = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    taken = counts.get(_tg.TAKEN, 0)
    invalid = counts.get(_tg.INVALID, 0)
    unreachable = counts.get(_tg.UNREACHABLE, 0)

    if unreachable and not taken:
        # Nothing is wrong with the pool; we could not ask. Saying "in use" here
        # is the one answer that makes the user act on a problem they do not have.
        print(_('remote: off — could not reach Telegram to claim a bot.'))
        print(_('  Nothing is wrong with your bots; the channel starts working '
                'again next time the network does.'))
        return

    if invalid and not taken and not unreachable:
        print(_('remote: off — Telegram rejected {n} of {total} token(s) in '
                '~/.codemason/remote.env.', n=invalid, total=pool_size))
        print(_('  A rejected token was revoked, mistyped, or belongs to a deleted '
                'bot. Check it with @BotFather.'))
        return

    print(_('remote: no free bot — every usable one of the {n} in '
            '~/.codemason/remote.env is already taken by another instance.',
            n=pool_size))
    print(_('  Telegram delivers one bot to exactly one process, so each instance '
            'needs its own.'))
    print(_('  Create another with @BotFather and add a TELEGRAM_BOT_TOKEN line.'))
    if invalid:
        print(_('  ({n} token(s) were also rejected — revoked or mistyped.)',
                n=invalid))
    if unreachable:
        print(_('  ({n} could not be checked right now.)', n=unreachable))


class RemoteInput:
    """Wires the poller and one session's inbox to that session's input queue."""

    def __init__(self, session_id: str, *, cwd: str = '', on_message=None):
        self.session_id = session_id
        self.cwd = cwd
        # Settled in start(), once a live registry exists to be distinct against.
        # One string for the bot's name, the picker's button and the ended-session
        # notice, so those three cannot drift apart.
        self.label = ''
        self.on_message = on_message        # callable(text, age, source) -> None
        self._task: Optional[asyncio.Task] = None
        self._poller = None
        self._started = False
        # start() sweeps immediately; the timer paces every sweep after that.
        self._last_sweep = 0.0

    # --- lifecycle -----------------------------------------------------------

    # This process's instance. The reference survives a stop, because turning it
    # back on needs that same object — building a fresh one would register the
    # session twice. And /remote off has to reach further than the outbound gate:
    # the poller checks availability once, at start(), and then loops, so without
    # this a switched-off channel would keep injecting whatever the phone sends.
    _active: Optional['RemoteInput'] = None

    def start(self) -> bool:
        """Arm the inbox, and the phone leg too when there is a bot for it.

        These are two different things and used to be one. A session's inbox is
        local: anything on this machine can drop a message in it and the drain
        loop hands it to the prompt a second later — `codemason inject`, a
        watcher script, a build hook, a cron line. Telegram is one way to reach
        that inbox, not what makes it exist. Tying them together meant a machine
        with no credentials had no way to wake its own session at all, so an
        agent could not arrange to be called back by work it had started, which
        is the plainer half of the feature and the half that needs no account
        anywhere.

        So registration and the drain loop run regardless; the claim and the
        poller are skipped when no bot is free. The return value still answers
        "is the phone leg up", because that is what the caller prints. Never
        raises: a session must run identically with or without a phone attached.
        """
        try:
            # Settle the name before claiming, because the claim uses it twice —
            # to pick which bot to prefer, and to name it. Deriving it separately
            # in each place is how the bot's name and the picker's button came to
            # disagree about what the same session was called.
            self.label = inbox.distinct_label(inbox.short_label(self.cwd),
                                              self.session_id)

            # Claiming is a question for Telegram, not a setting. getUpdates is
            # exclusive per bot, so instances that cannot see each other's
            # filesystem — a VM, a container, another machine — have no way to
            # divide a single bot between them, and no shared file to agree
            # through either. The bot itself is the only rendezvous they share, so
            # "is this one taken?" is asked there and nowhere else.
            from .permissions import is_available
            phone = False
            if is_available():
                from .adapters import telegram as _tg
                phone = _claim_bot_for(self.cwd, self.label)
                if not phone:
                    _report_no_free_bot(len(_tg.bot_pool()))

            inbox.sweep()                   # reclaim files left by crashed runs
            import time as _time
            self._last_sweep = _time.monotonic()
            inbox.register(self.session_id, cwd=self.cwd, label=self.label)

            if phone:
                from . import poller as _poller
                self._poller = _poller.get()
                # File messages by session; the poller may be serving other
                # sessions too, and this callback runs on its thread.
                self._poller.on_message = _route
                self._poller.acquire()

            self._task = asyncio.ensure_future(self._drain_loop())
            self._started = True
            RemoteInput._active = self
            return phone
        except Exception:
            return False

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            if self._task is not None and not self._task.done():
                self._task.cancel()
        except Exception:
            pass
        try:
            if self._poller is not None:
                self._poller.release()
        except Exception:
            pass
        # Anything that arrived in the last moments is still the user's, and this
        # session is the only one that knows it was for them. Say so on the way
        # out rather than leaving it to rot in a file until sweep — a typed
        # instruction that vanishes with no trace reads as "it ignored me", which
        # is the worst way for this to fail.
        try:
            leftover = [m.text for m in inbox.take(self.session_id) if m.text]
            if leftover:
                _tell(_('{count} message(s) not delivered', count=len(leftover)),
                      _('The session ended before these could be handled:') + '\n' +
                      '\n'.join(f'• {t[:120]}' for t in leftover[:5]) +
                      ('\n…' if len(leftover) > 5 else ''))
        except Exception:
            pass
        # The inbox file itself is left for `sweep`: a message can still land
        # between that drain and the process actually exiting.
        try:
            inbox.unregister_registry_only(self.session_id)
        except Exception:
            pass

        # Say the phone has gone deaf, and whether anything else is still
        # listening. Without this a closed session is indistinguishable from a
        # crashed one, a dead network, or a bug — you send a message, nothing
        # happens, and there is no way to tell which. The only cue you get today
        # is silence, and silence is what every other failure looks like too.
        try:
            others = [s for s in inbox.live_sessions()
                      if s['session_id'] != self.session_id]
            if others:
                names = ', '.join(s.get('label') or s['session_id'][:8] for s in others)
                body = _('Other sessions are still running: {names}', names=names)
            else:
                body = _('No sessions are running. Messages will not be delivered '
                         'until you start one.')
            # Sent in both cases. The call used to sit inside the else, so a
            # session that ended while a sibling was still up composed the line
            # naming that sibling and dropped it: its bot said nothing at all.
            # With four bots open, three of them went quiet on exit and only the
            # last one to close ever reported anything — which is the silence
            # this block exists to break, reintroduced by where the call sat.
            #
            # The session's own label, not the directory's basename. That
            # basename is the bug short_label exists to avoid — in the home
            # directory it is the account name, so the notice read "Session
            # ended: user", which names neither the session nor the work.
            _tell(_('Session ended: {name}',
                    name=self.label or inbox.short_label(self.cwd)), body)
        except Exception:
            pass

    def _adopt_unrouted(self) -> None:
        """Take over messages that arrived while no session was registered.

        Gated OFF by default -- `_drain_loop` runs this only under
        `remote_replay_backlog`; see the note there for what it did while it was
        unconditional.

        `_route` files those under UNROUTED and tells the sender they are "held
        and will be delivered when one starts". Nothing ever read that file:
        take_unrouted had no callers and sweep skips UNROUTED by name, so the
        promise was false and the message sat there for good. Being told it was
        kept is worse than being told it was dropped — you do not send it again.

        Now the arrival of a session IS the event the promise named, so this runs
        when one registers. Moved rather than copied: two sessions starting
        together must not both run the same instruction, and whoever gets there
        first has taken responsibility for it.
        """
        try:
            held = inbox.take_unrouted()
        except Exception:
            return
        for message in held:
            if not message.text:
                continue
            if inbox.deliver(self.session_id, message):
                continue
            # Could not file it: put it back rather than lose it, and stop —
            # the next tick will try again.
            try:
                inbox.deliver(inbox.UNROUTED, message)
            except Exception:
                pass
            return

    # --- draining ------------------------------------------------------------

    async def _resync_session_id(self) -> None:
        """Follow the session id if it changed under us.

        `register` used to run once, at start, and the id is not fixed for the
        life of the process: /clear mints a new one (switch_session) and /resume
        adopts the id of the transcript it restored. Nothing re-registered, so
        from that moment the registry advertised an id the session no longer had.

        Inbound kept working by luck — the stale entry is still a live session, so
        `_route` delivers to it and this loop drains the same stale id — which is
        why the damage went unnoticed. What broke is everything that names the
        CURRENT id: claim_message and claim_question file ownership under the new
        id, `_route` and the poller then look it up in `live_sessions()`, do not
        find it, and fall through. Replying to an alert stops routing to the
        session that posted it, and a forwarded button answer never reaches the
        session that asked. /remote showing '(unregistered)' was the visible tip.

        Drain the old inbox before switching so nothing in flight is stranded, and
        move the registration rather than adding one: two entries for one process
        would make the phone's picker offer the same session twice.
        """
        from ..utils.state import get_session_id
        try:
            current = get_session_id()
        except Exception:
            return
        if not current or current == self.session_id:
            return
        previous = self.session_id
        try:
            for message in await asyncio.to_thread(inbox.take, previous):
                if message.text and self.on_message:
                    self.on_message(message.text, age_seconds(message), message.source)
        except Exception:
            pass
        try:
            inbox.unregister(previous)
            # Carry the settled label across: re-deriving it here would let the
            # session be renamed mid-run by whoever else happened to be live at
            # that moment, while the bot it is bound to keeps the old name.
            inbox.register(current, cwd=self.cwd, label=self.label)
        except Exception:
            return                          # keep the old id rather than none
        self.session_id = current

    async def _sweep_if_due(self) -> None:
        """Reclaim aged leftovers periodically, not only at startup."""
        import time as _time
        now = _time.monotonic()
        if now - self._last_sweep < SWEEP_INTERVAL:
            return
        self._last_sweep = now
        try:
            await asyncio.to_thread(inbox.sweep)
        except Exception:
            pass                            # housekeeping must never end a session

    async def _drain_loop(self) -> None:
        # Every filesystem call below is pushed to a worker thread. `take` holds
        # an exclusive lock over this session's inbox for the whole read, and on
        # Windows it may sit through a retry window waiting for one. Called
        # inline that wait is on the event loop, and with it the prompt, the
        # spinner and every keystroke: the terminal goes dead for a reason that
        # is nowhere on screen. permissions.py pushes its blocking wait off the
        # loop for the same reason.
        reported = False
        while self._started:
            try:
                await self._resync_session_id()
                await self._sweep_if_due()
                # Opt-in only, and OFF by default (`remote_replay_backlog`).
                #
                # A session must not inherit messages nobody asked it to run.
                # This sits in the drain loop, so it fired EVERY tick, not just
                # at startup: anything in UNROUTED was pulled into whichever
                # session happened to be live and submitted as a user turn, with
                # no prompt, no confirmation and no size limit. Measured
                # 2026-10-01: a freshly started cm2 whose only typed input was
                # one short line (18,144-token first request, 18:03:40) was
                # followed at 18:05:21 by a 497,039-character turn, and the next
                # request carried 262,342 prompt tokens -- 59% of the window in
                # one step, off a 4-minute cold prefill. From the terminal that
                # is indistinguishable from the session resuming itself.
                #
                # Held messages now stay held and the sender is told they were
                # NOT delivered (poller.py), which is something they can act on.
                if getattr(__import__('codemason.utils.config', fromlist=['get_config'])
                           .get_config(), 'remote_replay_backlog', False):
                    await asyncio.to_thread(self._adopt_unrouted)
                for message in await asyncio.to_thread(inbox.take, self.session_id):
                    if not message.text:
                        continue
                    if self.on_message:
                        # Acknowledge before running it. A turn can take a minute
                        # before it has anything to say, and until then the phone
                        # shows nothing — which looks exactly like the message
                        # never arriving.
                        _ack(message, '👀')
                        self.on_message(message.text, age_seconds(message), message.source)
                reported = False             # drained cleanly; a later fault is news again
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Reported on the FIRST one. Contention does not reach here —
                # take() catches LockUnavailable itself and returns nothing, so
                # the retry that a busy inbox needs has already happened one
                # level down. Anything arriving here is therefore unexpected by
                # construction, and waiting for it to happen again before saying
                # so only buys silence: this loop discarding its exceptions is
                # why a dead inbox was indistinguishable from an idle one, with
                # the phone's messages vanishing while the other direction
                # carried on working.
                #
                # Once per outbreak, not per tick — a fault that repeats every
                # second must not notify every second — and again if it returns
                # after a clean drain, because that is a new fact.
                if not reported:
                    reported = True
                    from ..utils.log import debug_log
                    debug_log('REMOTE_DRAIN_FAILED',
                              {'session': self.session_id, 'error': repr(e)})
                    # On screen too, and not only to the phone: the phone is the
                    # half that is broken, so a fault report that only goes there
                    # can be lost to the same fault it is reporting. Printing
                    # from this task is safe under patch_stdout — it queues above
                    # the prompt like any mid-turn output.
                    try:
                        from ..utils.formatting import safe_print
                        safe_print('\n⚠️  ' + _('This session could not read its '
                                                 'inbox: {error}', error=e))
                    except Exception:
                        pass
                    _tell(_('Messages from the phone are not being delivered'),
                          _('This session could not read its inbox: {error}',
                            error=e))
            await asyncio.sleep(DRAIN_INTERVAL)


def _ack(message, emoji: str) -> None:
    """Mark a received message so it visibly landed. Best-effort, never raises."""
    mid = (message.meta or {}).get('message_id')
    if not mid:
        return
    try:
        from .notify import adapters
        for adapter in adapters():
            if adapter.is_configured() and hasattr(adapter, 'react'):
                adapter.react(int(mid), emoji)
    except Exception:
        pass


def age_seconds(message) -> float:
    """How long ago the user actually sent this, or 0 when unknown.

    Telegram holds updates for ~24h while nobody polls, so a session opened in
    the morning receives everything sent overnight. Injecting a message written
    hours ago as though it had just been typed is the dangerous case: "delete
    that" meant something specific then and something else now. The age is
    surfaced rather than the message dropped — deciding it is stale is the user's
    call, and silently discarding their instruction is its own failure.
    """
    import time
    sent = (message.meta or {}).get('sent_at')
    if not isinstance(sent, (int, float)) or sent <= 0:
        return 0.0
    return max(0.0, time.time() - float(sent))


def _deliver_or_tell(session_id: str, message) -> bool:
    """File a message, and say so on the phone if it could not be filed.

    `deliver` returns False only when the inbox could not be locked, which it now
    refuses to write around — an interleaved line is unparseable and would be
    dropped on read, losing the message with the evidence destroyed too. So the
    one thing that must not happen here is silence: an instruction that vanishes
    reads as "codemason ignored me", and the sender is the only one who can
    decide whether to send it again.
    """
    if inbox.deliver(session_id, message):
        return True
    _tell(_('Message not delivered'),
          _('The session inbox is locked by another process and did not accept '
            'it. Nothing was written, so nothing is half-there — send it again.'))
    return False


def _owning_session_id() -> str:
    """This session's id, when it owns the bot the message came in through.

    A claim binds one bot to one session and Telegram enforces it: getUpdates is
    exclusive per bot, so this process polls the bot it claimed and nothing else.
    Every update reaching `_route` therefore arrived through that bot, and the
    bot has exactly one owner — which makes the owner known, not inferred.

    Empty when nothing is claimed. Outbound works without a claim (a configured
    adapter can send), so "configured" does not imply this process polls
    anything, and the caller must not read absence as ambiguity to guess at.
    """
    try:
        from .adapters import telegram as _tg
        if not _tg.claimed_token():
            return ''
        active = RemoteInput._active
        return active.session_id if active else ''
    except Exception:
        return ''


def _route(text: str, meta: Dict[str, Any]) -> None:
    """Poller-thread callback: decide which session a message belongs to.

    Runs off the event loop, so it only touches the filesystem — the owning
    session picks it up on its next drain.

    Cases, in order of how much they are inferring:

      claimed   the message arrived through a bot this session owns → this
                session. Nothing is being inferred: the claim already decided it.
      reply     you replied to a message a session posted → that session.
      only one  no ambiguity to resolve.
      several   ask. Delivering an instruction to the wrong session gets it acted
                on straight away by the wrong code, which is far worse than one
                extra tap — so this never guesses, not even at "the most recent".
      none      park it and say so, rather than dropping it silently.

    The claimed case used to be missing, and its absence is what put a picker in
    front of every message. Bots are claimed per session out of a pool, but
    routing asked `live_sessions()` — every session on the machine — and so
    offered a choice the binding had already made. Two windows in one directory
    turned that into an unanswerable question: both label from the directory, so
    the buttons read `~ (776b58)` and `~ (4076f1)` and the only thing telling
    them apart was a slice of a session id. The directory was never the unit;
    the bot-to-session claim is.
    """
    message = inbox.InboundMessage(text=text, source='telegram', meta=meta)

    # Typing an answer is at least as natural as tapping one — you are in a chat.
    # Without this, "y" sent while a prompt is up goes into the conversation as a
    # user message and the prompt keeps waiting, which looks like it was ignored.
    if _answer_by_text(text, meta):
        return

    owner = _owning_session_id()
    if owner:
        _deliver_or_tell(owner, message)
        return

    reply_to = meta.get('reply_to')
    if reply_to:
        owner = inbox.message_owner(int(reply_to))
        if owner and owner in {s['session_id'] for s in inbox.live_sessions()}:
            _deliver_or_tell(owner, message)
            return

    sessions = inbox.live_sessions()
    if len(sessions) == 1:
        _deliver_or_tell(sessions[0]['session_id'], message)
        return

    if not sessions:
        if _deliver_or_tell(inbox.UNROUTED, message):
            _tell(_('No sessions are running'),
                  _('Your message is held and will be delivered when one starts.'))
        return

    _ask_which_session(message, sessions)


# Words that mean yes or no and nothing else. Kept short and exact — a message
# is only ever read as an answer when it is *just* one of these, so "y" answers a
# prompt while "yes, and also refactor the parser" stays an instruction.
# These are INPUT, not display: they are what the user may type back, so they are
# not translated but accumulated. A word only counts when the whole message is
# that word, which keeps "no" — English and Spanish alike — from swallowing a
# sentence that merely starts with it.
_YES = {'y', 'yes', 'ok', 'okay', 'yep', 'sure',
        '승인', '허용', 'ㅇ', 'ㅇㅇ',
        'はい', '是', '好',
        'ja', 'oui', 'si', 'sí', 'sim', 'да'}
_NO = {'n', 'no', 'nope', 'nah',
       '거부', '취소', 'ㄴ', 'ㄴㄴ',
       'いいえ', '否', '不',
       'nein', 'non', 'não', 'nao', 'нет'}


def _answer_by_text(text: str, meta: Dict[str, Any]) -> bool:
    """Resolve a prompt from a typed yes/no — but only as a REPLY to it.

    Buttons are the intended way to answer, and they say what they answer. A bare
    "y" does not. An earlier version accepted one whenever a single prompt was
    outstanding, which quietly created a way to approve a tool call by accident:
    the session asks something conversationally, you type "y" meaning that, and
    it lands on the permission request instead. You would have approved a command
    without ever seeing which one.

    Replying removes the ambiguity — the message you replied to *is* the target —
    so that path stays and the guessing one is gone.
    """
    word = text.strip().lower()
    if word not in _YES and word not in _NO:
        return False

    reply_to = meta.get('reply_to')
    if not reply_to:
        return False
    target = int(reply_to)
    if target not in inbox.pending_questions():
        return False                        # not a live prompt — ordinary input

    # 'p:0'/'p:1' are POSITIONAL, and this queue carries both two-option
    # permission prompts and AskUserQuestion, whose options are arbitrary
    # model-authored labels. A typed "no" on a three-way question therefore
    # selected whatever option 1 happened to be — "Drop and recreate the DB"
    # reads as a denial to the user and as a choice to the agent — and the
    # confirmation below then reported it as "denied". Only take the shorthand
    # where the positions are known to mean allow/deny; anything else falls
    # through to ordinary input, and the buttons are still there to tap.
    if inbox.question_option_count(target) != 2:
        return False
    inbox.deliver_answer(target, 'p:0' if word in _YES else 'p:1')
    _tell(_('Answer delivered'),
          f'"{text}" -> ' + (_('approved') if word in _YES else _('denied')))
    return True


def _tell(title: str, body: str = '') -> None:
    """A note back to the phone about routing itself.

    INTERNAL rather than PUBLIC: the body may quote what you typed, which is
    yours, not metadata. Sent blocking because the callers are shutdown paths —
    a daemon thread would be killed before the POST completed.
    """
    try:
        from .events import EventKind, Sensitivity
        from .notify import notify
        notify(EventKind.INFO, title, body=body,
               sensitivity=Sensitivity.INTERNAL, blocking=True)
    except Exception:
        pass


def _ask_which_session(message, sessions) -> None:
    """Post a picker listing the live sessions and park the message on it."""
    try:
        from .notify import adapters
        for adapter in adapters():
            if not adapter.is_configured():
                continue
            labels = [s.get('label') or s['session_id'][:8] for s in sessions]
            picker_id = adapter.pick(
                _('Which session should this go to?'),
                (message.text or '')[:200],
                labels,
            )
            if picker_id:
                inbox.park_route(picker_id, message.text, message.meta or {},
                                 [s['session_id'] for s in sessions])
                return
    except Exception:
        pass
    # No transport could ask — hold it rather than send it somewhere arbitrary.
    _deliver_or_tell(inbox.UNROUTED, message)


def set_enabled(enabled: bool) -> str:
    """Attach or detach this session's inbound remote input at runtime.

    Outbound (alerts, spinner, prompts) is gated by remote.notify.off_reason, but
    inbound is a live poller thread that was started once — flipping a setting
    does not reach it. So the switch has to stop and restart it, or "off" would
    keep injecting whatever the phone sends.

    Returns a short status for the caller to print.
    """
    cur = RemoteInput._active
    if cur is None:
        return _('no inbound link (this session does not use remote input)')
    if enabled:
        if cur._started:
            return _('inbound already attached')
        return _('inbound attached') if cur.start() else _('inbound failed to attach')
    if not cur._started:
        return _('inbound already detached')
    try:
        cur.stop()
        return _('inbound detached (poller stopped, poll lock released)')
    except Exception as e:
        return _('failed to detach inbound: {error}', error=e)
