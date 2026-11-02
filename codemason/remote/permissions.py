"""Remote tool approval.

The hard part of remote control is not moving bytes, it is deciding who answers
a permission prompt when a terminal and a phone are both attached. Claude Code
shipped bugs here twice (2.1.208, 2.1.211), and their print mode sidesteps it
entirely — `print.ts` has no CLI-side dialog at all, so headless runs either
delegate to an SDK consumer or bypass permissions outright.

We avoid the ownership question instead of negotiating it: **both prompts go up
and the first answer wins.** No side owns the decision, so nothing has to be
handed over, and a phone that is closed, offline or uninstalled changes nothing —
the local prompt behaves exactly as it does today. The remote path can only ever
add a way to answer, never remove one.

The other half is the failure mode. `None` from an adapter means "nobody
answered *here*", never "denied". An unanswered prompt must not silently become
an allow or a deny; it must fall through to the local prompt and keep waiting,
which is what a user sitting at the terminal already expects.

Headless (`-p`) is the case this unlocks: today a non-read-only tool there hits
`input()`, takes EOF and is denied, which is why sub-agent runs get
`--dangerously-skip-permissions` and end up bypassing everything. With a phone to
ask, `-p` can run *with* permissions again.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional, Type

from ..utils.i18n import _

# Button labels, in the order they appear on the phone. Mapped back to the
# single-key answers the local prompt already speaks.
#
# 'always' (a) is deliberately NOT offered remotely. It is not a stronger yes —
# it opens a blanket grant for the rest of the session, keyed by tool name, or by
# command word for Bash ('a' on `git status` allows every later `git ...`). A
# terminal answer is given after reading the command on screen; a phone answer is
# a reflex tap on a notification, with no confirmation step and no easy way to
# review what was just granted. The value of remote approval is unblocking a
# single call while you are away, and per-call approval covers that fully.
# Reducing repeat prompts is a sit-at-the-desk concern — answer 'a' there.
#
# Kept as the canonical English here and translated at call time, not at import:
# the config that decides the language may not be loaded yet when this module is
# first imported, which would freeze the buttons in whatever was current then.
_ALLOW = '✅ Allow'
_DENY = '⛔ Deny'


def _buttons() -> List[str]:
    return [_(_ALLOW), _(_DENY)]


def _to_key(answer: Optional[str], options: List[str]) -> Optional[str]:
    """The tapped button back to the single key the local prompt speaks.

    Compared against the list that was actually sent, not against a fresh
    translation: the language can change between posting the buttons and the tap
    landing, and re-deriving the labels would then fail to recognise the answer —
    silently, as "nobody answered", leaving the turn waiting on a question that
    had in fact been answered.
    """
    if not answer:
        return None
    if options and answer == options[0]:
        return 'y'
    if len(options) > 1 and answer == options[1]:
        return 'n'
    return {_(_ALLOW): 'y', _(_DENY): 'n', _ALLOW: 'y', _DENY: 'n'}.get(answer)

# Fallback when config is unreadable; the real value is `remote_ask_timeout`.
#
# This is NOT a display deadline. While the question is live the buttons stay
# usable, because the turn is blocked on them the whole time. Collapsing them
# early left a turn waiting on a prompt that could no longer be answered from the
# phone at all.
#
# What actually ends the wait is the terminal answering (which cancels this) or
# the run being abandoned. The bound exists for the case where neither ever
# happens: a headless -p run has no terminal to fall back to.
#
# Half an hour, not hours. Waiting is not free — the prompt holds the exclusive
# poll lock, so every other session goes without remote input until it resolves.
# And a longer window only pays off if the answer genuinely arrives late; in
# practice it comes within the half hour or not at all, so the extra time buys
# nothing and costs the lock. Expiry is announced with its own push, so a short
# window never leaves you guessing whether it is still waiting.
DEFAULT_TIMEOUT = 1800.0

# Ceiling on the yield between re-ask rounds. NOT the yield itself — the wait
# ends when the lock is observably gone, and this only stops a hang if it never
# is.
#
# It used to be a flat 2-second sleep, "a yield, not a wait". It yielded nothing.
# Releasing the poller only sets its stop event; the loop is inside a long poll
# and does not notice until the cycle ends, which is bounded by POLL_CHUNK — so
# after two seconds this process was still holding the lock every time, and the
# next round took it straight back. A sleep whose whole purpose was to let go of
# something it had not let go of yet.
#
# So wait for the fact instead: poll for the lock actually being released, and
# skip the wait entirely when no other session exists to hand it to — the common
# case, where the delay was pure cost.
RELOCK_CEILING = 30.0
RELOCK_POLL = 0.25


def _timeout() -> float:
    try:
        from ..utils.config import get_config
        return float(getattr(get_config(), 'remote_ask_timeout', DEFAULT_TIMEOUT))
    except Exception:
        return DEFAULT_TIMEOUT


def _summary(tool: Type, input_data: Dict[str, Any]) -> str:
    """A one-glance description of what is about to run.

    Uses the tool's own renderer so the phone shows the same thing the terminal
    does. Falls back to the raw input, truncated — never to nothing, since an
    approval prompt with no detail is worse than useless."""
    try:
        rendered = tool.render_tool_use_message(input_data, verbose=False)
        if rendered and str(rendered).strip():
            return str(rendered).strip()
    except Exception:
        pass
    try:
        import json
        return json.dumps(input_data, ensure_ascii=False)[:400]
    except Exception:
        return str(input_data)[:400]


def _ask_blocking(kind, title: str, body: str, options: List[str],
                  cwd: Optional[str], timeout: float, cancel,
                  meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Run the adapter round-trip. Blocking — callers push it to a thread.

    Returns the option the user picked, verbatim, or None when nobody answered
    here. Deliberately says nothing about what the options mean: a permission
    grant and a multiple-choice question are the same round-trip, and only the
    caller knows how to read the answer.
    """
    from .events import Event, Sensitivity
    from .notify import adapters

    # The body can quote a command line or a file path, so it is INTERNAL, not
    # PUBLIC. Adapters cleared below that get the title and a withheld body —
    # still enough to decide "am I expecting codemason to run Bash right now?".
    event = Event(
        kind=kind,
        title=title,
        body=body,
        sensitivity=Sensitivity.INTERNAL,
        cwd=cwd,
        meta=meta or {},
    )
    for adapter in adapters():
        try:
            if not adapter.is_configured():
                continue
            answer = adapter.ask(
                event.redacted(adapter.max_sensitivity),
                options,
                timeout,
                cancel,
            )
        except NotImplementedError:
            continue
        except Exception:
            continue
        if answer:
            return answer
    return None


async def ask_options(kind, title: str, body: str, options: List[str], *,
                      cwd: Optional[str] = None, timeout: Optional[float] = None,
                      meta: Optional[Dict[str, Any]] = None,
                      repeat_title=None) -> Optional[str]:
    """Put a choice on the phone and wait. Returns the chosen option, or None.

    The waiting is the hard part, and it is the same whatever is being asked, so
    it lives here once: cancellation has to be cooperative because a thread
    cannot be killed, the poll lock has to be released between rounds so one
    blocked session cannot starve every other, and giving up is not an option
    while the turn is still stopped.
    """
    if not is_available():
        return None
    if timeout is None:
        timeout = _timeout()
    if timeout <= 0:
        return None                        # remote prompting disabled
    # A thread cannot be killed, so cancellation has to be cooperative. Without
    # this the worker would keep polling for the full timeout after the terminal
    # already answered — holding the exclusive poll lock and starving every other
    # session's prompt.
    cancel = threading.Event()

    # A live question owns the bottom of the chat. The turn's progress message
    # keeps ticking otherwise, and every tick that reposts itself pushes the
    # prompt you are meant to answer further up — so it holds still until this
    # resolves, then resumes below the answered prompt.
    from . import progress as _progress
    _progress.pause()
    try:
        # Ask again rather than give up. The window closing does not mean the
        # question stopped mattering — the turn is still blocked — so expiring
        # used to leave exactly one way forward: walk to the terminal. For
        # someone who drives this from a phone precisely because sitting down is
        # inconvenient, that is not an answer.
        #
        # Each round posts a fresh prompt, which is a fresh push: a nudge that
        # the work is still stopped, rather than silence. And between rounds the
        # poll lock is released, so a long wait cannot starve another session —
        # which is the only reason to bound a round at all.
        round_no = 0
        while True:
            # Say it is a repeat and how long it has been stuck, so a re-ask
            # reads as "still waiting" rather than a second, different request.
            shown = title if not round_no else (
                repeat_title(int(timeout * round_no / 60)) if repeat_title
                else _('{title} (re-asking · {mins} min so far)',
                       title=title, mins=int(timeout * round_no / 60)))
            try:
                answer = await asyncio.to_thread(
                    _ask_blocking, kind, shown, body, options, cwd, timeout,
                    cancel, meta)
            except asyncio.CancelledError:
                cancel.set()
                raise
            except Exception:
                return None
            if answer is not None:
                return answer
            if cancel.is_set():
                return None                # the terminal answered
            round_no += 1
            await _yield_poll_lock()
    finally:
        _progress.resume()


def _holds_poll_lock() -> bool:
    """True while any configured adapter still owns the inbound lock here."""
    try:
        from .notify import adapters
    except Exception:
        return False
    for a in adapters():
        held = getattr(a, 'holds_poll_lock', None)
        try:
            if callable(held) and held():
                return True
        except Exception:
            continue
    return False


async def _yield_poll_lock() -> None:
    """Between re-ask rounds, actually let go before asking again.

    Two things make this a wait on a fact rather than a duration:

    it is skipped when there is nobody to yield to, and who that could be is
    decided per bot, because the lock is per bot — `poll.<token>.lock`. A claimed
    bot belongs to one session, so nothing else can be waiting on its lock and
    there is nothing to yield. Asking instead how many sessions are alive on the
    machine answered a question about a different resource: two sessions polling
    two different bots contend for neither, and each re-ask round sat here
    burning the ceiling for no one.

    and when there IS someone, it ends the moment this process no longer holds
    the lock, however long that takes — releasing the poller only asks it to
    stop, and it finishes the long poll it is inside first.

    The ceiling is a hang guard, not the mechanism. Reaching it means the lock
    was never released, and going round again is still better than blocking the
    turn forever on a prompt nobody can see.
    """
    try:
        from .adapters import telegram as _tg
        if _tg.claimed_token():
            return                          # ours exclusively; no other claimant
    except Exception:
        pass
    try:
        # No claim: the bot may be shared, and then the session count is the
        # only signal there is.
        from . import inbox
        if len(inbox.live_sessions()) <= 1:
            return                          # no other claimant; nothing to yield
    except Exception:
        pass
    deadline = asyncio.get_running_loop().time() + RELOCK_CEILING
    while _holds_poll_lock() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(RELOCK_POLL)


def is_available() -> bool:
    """True when at least one adapter could actually field a prompt."""
    try:
        from .notify import adapters
        return any(a.is_configured() for a in adapters())
    except Exception:
        return False


async def ask(tool: Type, input_data: Dict[str, Any], *, cwd: Optional[str] = None,
              timeout: Optional[float] = None) -> Optional[str]:
    """Ask the phone. Returns 'y' | 'n', or None if nobody answered.

    None is not a decision. Callers fall back to (or keep waiting on) the local
    prompt.
    """
    from .events import EventKind
    tool_name = getattr(tool, 'name', str(tool))
    options = _buttons()
    answer = await ask_options(
        EventKind.PERMISSION_NEEDED,
        _('Permission needed: {tool}', tool=tool_name),
        _summary(tool, input_data),
        options,
        cwd=cwd,
        timeout=timeout,
        meta={'tool': tool_name},
        repeat_title=lambda mins: _('Permission needed (re-asking · {mins} min so far): {tool}',
                                    mins=mins, tool=tool_name),
    )
    return _to_key(answer, options)
