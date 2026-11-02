"""The notify API — the one thing the rest of the box calls.

Deliberately not codemason-specific. The gap this project is closing is "I have
to keep looking at the screen to know when something needs me", and codemason
turns are only one source of that: downloads, benchmarks, builds and a server
falling over are the same problem. So this is a box-wide channel with a shell
entry point, and codemason is its first caller rather than its owner.

    from codemason.remote import notify, EventKind
    notify(EventKind.TURN_DONE, 'codemason finished', body=answer_text)

    python -m codemason.remote.notify --kind task_done "168GB download finished"

Contract: never raises, never blocks the caller. A notification is not worth a
failed turn, and a wedged Telegram POST must not stall an agent loop, so
delivery runs on a daemon thread and failures land in the debug log.
"""

from __future__ import annotations

import atexit
import threading
from typing import List, Optional

from .adapters.telegram import TelegramAdapter
from .events import Event, EventKind, Sensitivity

# Registry. A PWA/WebSocket adapter slots in here later and both fire; nothing
# above this line changes.
_ADAPTERS: List = [TelegramAdapter()]


def off_reason() -> str:
    """Why the channel is switched off wholesale, or '' when it is on.

    One gate for the whole feature, checked where transports are handed out, so
    "off" cannot mean off for some paths and on for others. That is exactly what
    it used to mean in --plain: the phone could not answer a prompt there (no
    input router), yet turn-done alerts and the progress spinner still went out —
    a channel that talks at you and ignores you.

    Plain mode is the pure-ASCII path for a dumb console (Hercules, bare VT100,
    dumb log capture). Nobody drives one of those from a phone, so the channel is
    not half-available there; it is not there.
    """
    import os
    if os.getenv('CODEMASON_REMOTE', '').strip().lower() in ('0', 'off', 'false', 'no'):
        return '--no-remote'
    try:
        from ..utils.config import get_config
        if not getattr(get_config(), 'remote_enabled', True):
            return '/remote off'
    except Exception:
        pass
    try:
        from ..utils.formatting import plain_mode
        if plain_mode():
            return '--plain/--ascii'
    except Exception:
        pass
    return ''


def all_adapters() -> List:
    """The registry as configured, ignoring `off_reason`.

    Only for telling the user what would be used — a warning at startup, the
    /remote status line. Never for delivery.
    """
    return list(_ADAPTERS)


def adapters() -> List:
    return [] if off_reason() else list(_ADAPTERS)


def register(adapter) -> None:
    """Add a transport (used by the future local WebSocket adapter)."""
    _ADAPTERS.append(adapter)


def is_configured() -> bool:
    """True when at least one transport can actually deliver."""
    return any(a.is_configured() for a in adapters())


# In-flight delivery threads. They are daemons so a wedged POST can never hold
# the process open, but a daemon is killed outright at interpreter exit — and a
# `-p` run finishes its turn and exits immediately, which silently dropped every
# notification it tried to send. So exit waits for them, briefly and boundedly.
_inflight: List[threading.Thread] = []
_INFLIGHT_LOCK = threading.Lock()
_EXIT_GRACE = 5.0       # total seconds exit will wait for stragglers


def _drain_at_exit() -> None:
    import time as _time
    deadline = _time.monotonic() + _EXIT_GRACE
    with _INFLIGHT_LOCK:
        pending = [t for t in _inflight if t.is_alive()]
    for t in pending:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            break
        t.join(remaining)


atexit.register(_drain_at_exit)


def _deliver(event: Event) -> None:
    for a in adapters():
        try:
            if not a.is_configured():
                continue
            # Redact per transport, not per event: the same event may go out in
            # full locally and stripped to Telegram.
            a.send(event.redacted(a.max_sensitivity))
        except Exception as e:                      # pragma: no cover - defensive
            try:
                from ..utils.log import debug_log
                debug_log('REMOTE_NOTIFY_ERROR', {'adapter': getattr(a, 'name', '?'), 'error': str(e)})
            except Exception:
                pass


def notify(
    kind: EventKind,
    title: str,
    *,
    body: str = '',
    sensitivity: Sensitivity = Sensitivity.PUBLIC,
    session_id: Optional[str] = None,
    cwd: Optional[str] = None,
    blocking: bool = False,
    **meta,
) -> None:
    """Fire an event at every configured transport.

    blocking=True is for the CLI entry point, where the process exits as soon as
    this returns and a daemon thread would be killed mid-POST.
    """
    event = Event(
        kind=kind if isinstance(kind, EventKind) else EventKind(str(kind)),
        title=title,
        body=body,
        sensitivity=sensitivity,
        session_id=session_id,
        cwd=cwd,
        meta=meta,
    )
    if blocking:
        _deliver(event)
        return
    t = threading.Thread(target=_deliver, args=(event,), daemon=True)
    with _INFLIGHT_LOCK:
        # Drop finished threads so a long REPL session doesn't accumulate them.
        _inflight[:] = [x for x in _inflight if x.is_alive()]
        _inflight.append(t)
    t.start()


def _main(argv: Optional[List[str]] = None) -> int:
    """`python -m codemason.remote.notify` — so shell scripts share the channel."""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog='codemason.remote.notify',
        description='Send a notification to the configured remote channels.',
    )
    p.add_argument('title', help='One-line summary (always sent).')
    p.add_argument('--kind', default='info',
                   choices=[k.value for k in EventKind])
    p.add_argument('--body', default='', help='Detail. Read from stdin when "-".')
    p.add_argument('--sensitivity', default='public',
                   choices=[s.name.lower() for s in Sensitivity],
                   help='public=metadata, internal=model prose, secret=never leaves the box.')
    args = p.parse_args(argv)

    body = args.body
    if body == '-':
        body = sys.stdin.read()

    if not is_configured():
        print('remote: no transport configured '
              '(set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, or ~/.codemason/remote.env)',
              file=sys.stderr)
        return 2

    notify(
        EventKind(args.kind),
        args.title,
        body=body,
        sensitivity=Sensitivity[args.sensitivity.upper()],
        blocking=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
