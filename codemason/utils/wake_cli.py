"""``codemason whoami`` / ``inject`` / ``notify`` — the self-wake surface.

Intercepted at the top of cli.main() alongside ``mcp`` and ``skill`` (before
argparse, which owns the positional ``prompt``).

WHY THIS EXISTS. A session can already be woken from outside: something files a
message into ``~/.codemason/remote/inbox/<id>.jsonl`` and the session drains it
into the queue its prompt reads from, once a second (remote/input.py). Outbound
alerts have the same shape through remote/notify.py. Neither was reachable from
a shell, and nothing told the model either existed.

So an agent that set up long work had no way to arrange its own report. One told
the user it had set up monitoring and would speak when the alarm came; three
days of crashes then waited for someone to ask — not because the watcher failed,
but because the model does not run between turns and nothing it "watches" is
watched. Asked about it directly, it answered that no such path existed, in a
turn it was running because a message had arrived through exactly that path.

The inbox is local: it needs no account, no network and no bot. Telegram is one
way to reach it, not what makes it exist.

    codemason whoami                      this session's id, cwd, delivery state
    codemason inject [--session ID] TEXT  wake a session with TEXT
    codemason notify TITLE [--body B]     push an alert, session or no session

`inject` reaches a live session's queue, so the wake carries the session's
context and the model can look before it speaks. `notify` goes out over the
configured channels whether or not a session is running. A watcher wanting both
calls both.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Name of the environment variable a session exports for its children, so a
# script it spawns can address it without being told the id.
SESSION_ENV = 'CODEMASON_SESSION_ID'

# Absolute path of the running program, exported beside SESSION_ENV.
#
# The command's NAME is not knowable from inside: this ships as `codemason` but
# is installed behind whatever symlink the site chose (`cm2` on these boxes),
# and that symlink sits in ~/.local/bin, which a login shell has on PATH and a
# cron line or an ssh command does not. A watcher told to run `codemason inject`
# therefore dies with "command not found" in exactly the unattended context it
# was written for. The absolute path has neither problem.
BIN_ENV = 'CODEMASON_BIN'


def export_self(session_id: str) -> None:
    """Export both, so a child process can address this session and re-enter
    this program without knowing either."""
    os.environ[SESSION_ENV] = session_id
    try:
        os.environ[BIN_ENV] = os.path.abspath(sys.argv[0])
    except Exception:
        pass


def unexport_self() -> None:
    """Withdraw the id when the session stops being reachable.

    `/remote off` unregisters, which stops the drain; a watcher started before
    that holds the id in its own environment and cannot be told. But anything
    spawned AFTER the variable is cleared reads the absence and refuses, instead
    of addressing a session that no longer takes messages.
    """
    os.environ.pop(SESSION_ENV, None)


def at_a_terminal() -> bool:
    """Whether a person is watching this invocation.

    Decides whether "the only live session" is a convenience or a hazard. Typed
    at a prompt it is obviously right and saves an argument nobody wants to look
    up. Run from a watcher, a cron line or a `-p` process it is a guess about
    someone else's conversation, made where no one will see it be wrong — and a
    `-p` run never registers a session, so it has no id of its own to prefer and
    would take whatever interactive session happened to be open.
    """
    try:
        return bool(sys.stdin.isatty() and sys.stderr.isatty())
    except Exception:
        return False

WHOAMI_HELP = """codemason whoami — this session's identity and delivery state

  --session-id    print only the session id (empty if none), for scripts
  --json          machine-readable"""

INJECT_HELP = """codemason inject — wake a session with a message

  codemason inject [--session ID] [--source NAME] TEXT
  codemason inject [--session ID] -            read TEXT from stdin

The message lands in the session's inbox and is picked up within a second,
exactly as a message from the phone is. This is local: no account, no network.
The session must be running — with nothing alive this fails rather than
queueing for a session that may never come back. Pair it with
`codemason notify` when the alert must arrive either way.

Target resolution, in order: --session, $CODEMASON_SESSION_ID, then — only when
run at a terminal, where a wrong guess is seen — the single live session.
Unaddressed from a script it is an error, as is an ambiguous or dead target.

TEXT ARRIVES AS THE USER'S OWN WORDS. The session cannot tell it apart from
something the person typed, so it carries the authority of an instruction.
Piping a log or any other program's output into it hands that
authority to whatever wrote it: say what happened and where to look, rather
than pasting the bytes in.

The queue keeps only the most recent messages (200, or 256 KiB), dropping
oldest-first, so a watcher should send one short message per event rather
than a stream."""

NOTIFY_HELP = """codemason notify — push an alert to the user

  codemason notify [--body TEXT] [--kind KIND] [--sensitivity LEVEL] TITLE
  codemason notify --body - TITLE                 read the body from stdin

  --kind         info (default), task_done, error
  --sensitivity  public (default), internal, secret

Goes out over whatever channels this machine has configured; it does not need a
session to be running and does not need credentials to be read or passed."""


def _wants_help(argv: Sequence[str]) -> bool:
    return bool(argv) and argv[0] in ('-h', '--help', 'help')


def _read_stdin() -> str:
    """Read stdin to end; used for ``-`` so a script can pipe a report in
    rather than fight the shell over quoting it."""
    try:
        return sys.stdin.read()
    except Exception:
        return ''


def _describe_live(live: Sequence[Dict[str, Any]]) -> str:
    if not live:
        return '  (no live sessions)'
    return '\n'.join(
        f'  {e.get("session_id", "")}  pid {e.get("pid", "?")}  {e.get("label", "")}'
        for e in live
    )


def pick_session(
    live: Sequence[Dict[str, Any]],
    explicit: Optional[str] = None,
    env_id: Optional[str] = None,
    interactive: bool = True,
) -> Tuple[Optional[str], str]:
    """Which live session a wake is for: ``(session_id, '')`` or ``(None, why)``.

    The world is passed in so the refusals can be tested — they are the whole
    point of the function, and both are silent failures if they ever turn into
    fallbacks. Delivering a report to the wrong conversation reads as delivered,
    which is worse than not delivering it.
    """
    by_id = {e.get('session_id'): e for e in live}
    if explicit:
        if explicit in by_id:
            return explicit, ''
        return None, f'no live session {explicit}\n{_describe_live(live)}'
    # A session exports its own id to everything it spawns, so a watcher the
    # agent wrote from inside the session needs no argument at all.
    if env_id:
        if env_id in by_id:
            return env_id, ''
        return None, (
            f'{SESSION_ENV} names {env_id}, which is not running\n'
            f'{_describe_live(live)}'
        )
    if not live:
        return None, (
            'no live session to wake (start one, or send an alert with `notify` '
            'instead — it needs none)'
        )
    # Unattended and unaddressed: refuse. The convenience below is only a
    # convenience when someone is there to see it pick wrong.
    if not interactive:
        return None, (
            f'no target: {SESSION_ENV} is not set and this is not a terminal, so there '
            f'is nobody to catch a wrong guess. Pass --session, or run from a process '
            f'that inherited {SESSION_ENV} from the session you mean.'
            f'\n{_describe_live(live)}'
        )
    if len(live) == 1:
        return live[0].get('session_id'), ''
    return None, f'several sessions are live; name one with --session\n{_describe_live(live)}'


def _resolve_session(explicit: Optional[str]) -> Tuple[Optional[str], str]:
    from ..remote import inbox
    return pick_session(inbox.live_sessions(), explicit,
                        os.environ.get(SESSION_ENV) or None, at_a_terminal())


def handle_whoami_cli(argv: Sequence[str]) -> int:
    if _wants_help(argv):
        print(WHOAMI_HELP)
        return 0
    from ..remote import inbox, notify

    id_only = '--session-id' in argv
    as_json = '--json' in argv

    live = inbox.live_sessions()
    env_id = os.environ.get(SESSION_ENV, '')
    # Outside a session there is no "me"; a single live session is still a
    # useful answer for a script, so report it and say where it came from.
    if env_id:
        me, via = env_id, SESSION_ENV
    elif len(live) == 1:
        me, via = live[0].get('session_id', ''), 'the only live session'
    else:
        me, via = '', 'nothing claimed'

    if id_only:
        print(me)
        return 1 if not me else 0

    mine = next((e for e in live if e.get('session_id') == me), None)
    configured = notify.is_configured()
    off = notify.off_reason()
    pool = len(notify.all_adapters())

    if as_json:
        print(json.dumps({
            'session_id': me,
            'resolved_via': via,
            'pid': mine.get('pid') if mine else None,
            'cwd': mine.get('cwd') if mine else None,
            'label': mine.get('label') if mine else None,
            'remote_configured': configured,
            'remote_off_reason': off,
            'bot_pool': pool,
            'live_sessions': live,
        }, ensure_ascii=False, indent=2))
        return 0

    if not me:
        print(f'session: (none — {via})')
    else:
        print(f'session: {me}  ({via})')
        if mine:
            print(f'  pid {mine.get("pid")}  cwd {mine.get("cwd")}  label {mine.get("label")}')
    if configured:
        state = f'on, {pool} channel(s) configured'
    elif not off:
        state = 'not configured (~/.codemason/remote.env)'
    else:
        state = f'off — {off}'
    print(f'remote : {state}')
    print(f'live   :\n{_describe_live(live)}')
    return 0


def handle_inject_cli(argv: Sequence[str]) -> int:
    if not argv or _wants_help(argv):
        print(INJECT_HELP)
        return 2 if not argv else 0
    from ..remote import inbox

    session: Optional[str] = None
    source = 'inject'
    parts: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('--session', '-s'):
            i += 1
            if i >= len(argv):
                print('--session needs a session id', file=sys.stderr)
                return 2
            session = argv[i]
        elif a == '--source':
            i += 1
            if i >= len(argv):
                print('--source needs a name', file=sys.stderr)
                return 2
            source = argv[i]
        else:
            parts.append(a)
        i += 1

    text = _read_stdin() if parts == ['-'] else ' '.join(parts)
    text = text.rstrip('\n')
    if not text.strip():
        print('nothing to inject (give TEXT, or `-` to read stdin)', file=sys.stderr)
        return 2

    target, why = _resolve_session(session)
    if not target:
        print(why, file=sys.stderr)
        return 1

    # `source` rides along so the session can tell a machine's message from a
    # person's; the delivery path is the one the phone already uses.
    msg = inbox.InboundMessage(text=text, source=source, ts=0.0, meta={})
    if inbox.deliver(target, msg):
        print(f'delivered to {target}')
        return 0
    print(f'could not write to the inbox of {target}', file=sys.stderr)
    return 1


def handle_notify_cli(argv: Sequence[str]) -> int:
    if not argv or _wants_help(argv):
        print(NOTIFY_HELP)
        return 2 if not argv else 0
    from ..remote import notify
    from ..remote.events import EventKind, Sensitivity

    kinds = {'info': EventKind.INFO, 'task_done': EventKind.TASK_DONE, 'error': EventKind.ERROR}
    levels = {
        'public': Sensitivity.PUBLIC,
        'internal': Sensitivity.INTERNAL,
        'secret': Sensitivity.SECRET,
    }

    body = ''
    kind = EventKind.INFO
    sensitivity = Sensitivity.PUBLIC
    parts: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('--body', '-b'):
            i += 1
            if i >= len(argv):
                print('--body needs text (or `-` for stdin)', file=sys.stderr)
                return 2
            body = _read_stdin() if argv[i] == '-' else argv[i]
        elif a in ('--kind', '-k'):
            i += 1
            if i >= len(argv) or argv[i] not in kinds:
                print(f'--kind must be one of {", ".join(kinds)}', file=sys.stderr)
                return 2
            kind = kinds[argv[i]]
        elif a == '--sensitivity':
            i += 1
            if i >= len(argv) or argv[i] not in levels:
                print(f'--sensitivity must be one of {", ".join(levels)}', file=sys.stderr)
                return 2
            sensitivity = levels[argv[i]]
        else:
            parts.append(a)
        i += 1

    title = ' '.join(parts).strip()
    if not title:
        print('nothing to send (give TITLE)', file=sys.stderr)
        return 2
    if not notify.is_configured():
        off = notify.off_reason() or 'not configured (~/.codemason/remote.env)'
        print(f'the remote channel cannot deliver: {off}', file=sys.stderr)
        return 1

    # blocking: this process exits the moment the call returns, which would
    # otherwise kill the delivery thread mid-POST.
    notify.notify(kind, title, body=body, sensitivity=sensitivity, blocking=True)
    print('sent')
    return 0
