"""Where codemason itself hands events to the remote channel.

Kept out of cli.py because two entry points need it — the interactive REPL and
the headless `-p` runner — and print_runner importing cli would be circular.
Headless runs matter more here, not less: a background `-p` run is precisely the
case where nobody is watching the screen.

Everything here is best-effort. The session's lifetime belongs to the session;
whether a phone is attached, reachable, or even installed must not change what
the turn does.
"""

from __future__ import annotations

from typing import Optional

from ..utils.i18n import _

# Only errors are capped here. Answers are passed whole and each transport
# truncates to its own limit — Telegram's is 4096, a future in-network client has
# none worth applying.
#
# There used to be a 1500-character cap at this layer too, and it was a silent
# slice: a long answer arrived cut mid-sentence, or mid-code-block, with nothing
# saying so. Two caps in series also meant the visible one never fired, because
# this invisible one always got there first. One cap, at the layer that actually
# has the constraint, and it announces itself.
_MAX_ERROR = 500

# How much of the originating prompt is echoed above an answer. Enough to
# recognise which question this was, not enough to re-read it.
_MAX_PROMPT_ECHO = 200


def _with_prompt(body: str, prompt: str, remote_origin: bool) -> str:
    """Put the question back above the answer, for turns started elsewhere.

    A turn you asked for from the phone already has your own message sitting in
    the chat above it, so echoing it there would only say it twice. A turn
    started at the keyboard has nothing: the phone receives an answer with no
    question attached, and a column of those is unreadable — which is what makes
    it impossible to pick a session back up from the phone, however well the
    inbound side works.
    """
    if not prompt or remote_origin:
        return body
    head = ' '.join(prompt.split())
    if len(head) > _MAX_PROMPT_ECHO:
        head = head[:_MAX_PROMPT_ECHO - 1].rstrip() + '…'
    return f'{_("You asked")}: {head}\n\n{body}' if body else f'{_("You asked")}: {head}'


def turn_report_delay(remote_origin: bool) -> Optional[float]:
    """When a live progress message may start posting, or None for never.

    The single source of truth for "will the phone hear about this turn?", so
    the placeholder and the answer cannot disagree. A turn you asked for from
    the phone is always answered, so its progress starts immediately (the
    class's own floor still applies). A local turn is only reported once it has
    run past `remote_notify_after`, so that is when its progress may appear —
    a turn short enough to be filtered out never posts anything at all, which
    is what keeps the channel from narrating every three-second exchange.

    None when the channel is off or reporting is disabled entirely.
    """
    try:
        from .notify import is_configured
        if not is_configured():
            return None
    except Exception:
        return None
    if remote_origin:
        return 0.0
    after = _threshold()
    return None if after < 0 else after


def _titled(source: str, what: str) -> str:
    """Headline for a turn event.

    The bot *is* codemason, so naming it in the title says nothing — and the
    working directory shown underneath is usually called `codemason` too, which
    put the same word on two consecutive lines of every alert. The line that
    earns its place is which project this was, and that one is already there.
    Other sources (a build, a download) still identify themselves.
    """
    return what if source == 'codemason' else f'{source}: {what}'


def _threshold() -> float:
    """Seconds a local turn must exceed before it is announced. 0 = every turn.

    The fallbacks come from the dataclass field, never from a literal repeated
    here. They used to read 60.0 while Config said 0.0, so a machine where
    get_config() raised — or an install predating the field — silently answered
    "only turns longer than a minute", which is every chat turn there is. The
    channel then looked broken in exactly one direction (local input never
    reaching the phone, remote-origin turns fine, because those bypass the
    threshold), and nothing anywhere said why.
    """
    from ..utils.config import Config
    default = Config.remote_notify_after
    try:
        from ..utils.config import get_config
        return float(getattr(get_config(), 'remote_notify_after', default))
    except Exception:
        return float(default)


def notify_turn_end(
    duration_ms: int,
    answer: str = '',
    *,
    cwd: Optional[str] = None,
    error: str = '',
    source: str = 'codemason',
    tools: Optional[dict] = None,
    remote_origin: bool = False,
    progress=None,
    prompt: str = '',
    status: str = '',
) -> None:
    """Announce that a turn ended.

    `status` is the context/image line (utils/session_status.py). Numbers
    only, so it rides at PUBLIC like the duration.

    Short successful turns are normally skipped (`remote_notify_after`): a
    channel that buzzes on every reply teaches you to ignore it, and the turns
    you care about are the long ones you walked away from. Errors always go out
    regardless of duration — a failure five seconds in is still a failure.

    `remote_origin` overrides the threshold. If you asked from the phone, the
    answer belongs on the phone, however fast it came — otherwise sending "hi"
    and getting nothing back makes the whole channel look broken. The threshold
    is about not nagging you over work you never asked about from there; it was
    never meant to swallow replies to your own messages.
    """
    placeholder = None
    try:
        if progress is not None:
            placeholder = progress.finish()
    except Exception:
        placeholder = None

    try:
        after = _threshold()
        if after < 0 and not remote_origin:
            return
        elapsed = duration_ms / 1000.0
        if not error and not remote_origin and elapsed < after:
            return

        from .events import Event, EventKind, Sensitivity
        from .notify import is_configured, notify
        if not is_configured():
            return

        # A progress message is already sitting in the chat saying "working…".
        # Rewrite it into the answer instead of posting a second message, so one
        # exchange stays one message and no stale status is left above the reply.
        if placeholder is not None:
            done = Event(
                kind=EventKind.ERROR if error else EventKind.TURN_DONE,
                title=_titled(source, _('Turn failed') if error else _('Turn done')),
                body=_with_prompt(
                    error[:_MAX_ERROR] if error else (answer or '').strip(),
                    prompt, remote_origin),
                sensitivity=Sensitivity.INTERNAL,
                cwd=cwd,
                meta={'duration_ms': duration_ms, 'tools': tools or {}, 'status': status},
            )
            from .notify import adapters
            for a in adapters():
                if a.is_configured() and hasattr(a, 'replace'):
                    if a.replace(placeholder, done.redacted(a.max_sensitivity)):
                        return

        if error:
            notify(
                EventKind.ERROR,
                _titled(source, _('Turn failed')),
                body=_with_prompt(error[:_MAX_ERROR], prompt, remote_origin),
                sensitivity=Sensitivity.INTERNAL,
                cwd=cwd,
                duration_ms=duration_ms,
                tools=tools or {},
                status=status,
            )
            return

        # The answer is model prose → INTERNAL. A PUBLIC-only transport receives
        # the headline plus an explicit withheld marker instead of silence.
        notify(
            EventKind.TURN_DONE,
            _titled(source, _('Turn done')),
            body=_with_prompt((answer or '').strip(), prompt, remote_origin),
            sensitivity=Sensitivity.INTERNAL,
            cwd=cwd,
            duration_ms=duration_ms,
            # Names and counts only. Deliberately not the arguments: a Bash
            # command line or a file path is SECRET and never takes this path.
            tools=tools or {},
            status=status,
        )
    except Exception:
        # A notification is never worth a failed turn.
        pass
