"""Remote answering for AskUserQuestion.

The permission prompt was only half of "the phone can answer". A turn also stops
when the model asks *you* something — which option, which approach — and that
stop is longer and more common than a permission one. Until now it had no remote
path at all: the question printed to a terminal nobody was sitting at, and a
session driven from a phone hung until someone walked back to it.

Same shape as `permissions`, deliberately: both prompts go up, the first answer
wins, and `None` means "nobody answered *here*", never a choice. The waiting
itself is shared (`permissions.ask_options`) — rounds, the poll lock, pausing the
progress message — because none of it is specific to what is being asked.

Two things the phone does not carry:

*Free text* is carried, through the chat itself. The phone has a keyboard, and
leaving "Other" off it had a worse effect than a missing option: you typed the
answer anyway, it was filed as the session's next instruction, and the turn sat
blocked on the question with that text unread until a button was tapped. So a
message typed while a question waits answers it (`answer_by_text`), and an
"Other" button says so. Permission prompts are excluded — a sentence typed at
one must never become a grant.

*Multi-select.* One tap is one option. A `multiSelect` question answered from the
phone returns a single choice, which is a valid subset of what was asked, rather
than a second mechanism for assembling a set on a lock screen.
"""

from __future__ import annotations

from typing import List, Optional

from ..utils.i18n import _

# Telegram's inline keyboard rejects a callback payload over 64 bytes, and long
# labels wrap into unreadable buttons on a phone. The index is what travels, so
# only the visible text is trimmed.
_MAX_LABEL = 40


def _short(label: str) -> str:
    return label if len(label) <= _MAX_LABEL else label[:_MAX_LABEL - 1] + '…'


def is_available() -> bool:
    from .permissions import is_available as _avail
    return _avail()


async def ask(question: str, header: str, labels: List[str], *,
              descriptions: Optional[List[str]] = None,
              cwd: Optional[str] = None,
              timeout: Optional[float] = None) -> Optional[str]:
    """Put a question on the phone. Returns the chosen label, or None.

    The label is returned verbatim — trimming happens on the button, never on the
    answer, so what comes back is exactly what the local picker would have
    produced and the caller needs no special case for a remote answer.
    """
    if not labels:
        return None
    from .events import EventKind
    from .permissions import ask_options

    # The options are already on the buttons; the body carries what the terminal
    # shows underneath each one, which is usually why you would pick it.
    body_lines = []
    for i, label in enumerate(labels):
        desc = (descriptions or [])[i] if descriptions and i < len(descriptions) else ''
        body_lines.append(f'• {label}' + (f' — {desc}' if desc else ''))
    body = question + ('\n\n' + '\n'.join(body_lines) if body_lines else '')

    title = _('Question: {header}', header=header) if header else _('Question')
    chosen = await ask_options(
        EventKind.QUESTION_ASKED,
        title,
        body,
        [_short(l) for l in labels],
        cwd=cwd,
        timeout=timeout,
        repeat_title=lambda mins: _('{title} (re-asking · {mins} min so far)',
                                    title=title, mins=mins),
    )
    if chosen is None:
        return None
    # Map the button text back to the full label it was trimmed from.
    for label in labels:
        if _short(label) == chosen:
            return label
    return chosen


def answer_by_text(text: str, meta: dict) -> bool:
    """File a typed message as the answer to a waiting question. True if it was.

    Runs in whichever process holds the poll lock, which need not be the one
    that asked — so the target is found through the on-disk claims, and the
    answer goes through the same forwarding file a button tap uses.

    Which question:
      reply     you replied to a question → that one. A reply to anything else
                (a permission prompt, a session alert) is not an answer here.
      only one  the only question taking text → that one.
      several   ask you to reply to the one you mean, rather than guess. The
                text is not filed as an instruction either: that is the path
                that left it unread while the turn stayed blocked.

    A slash command is never an answer — /clear typed at a question is still
    /clear.
    """
    body = (text or '').strip()
    if not body or body.startswith('/'):
        return False
    from . import inbox

    waiting = [mid for mid in inbox.pending_questions()
               if inbox.question_accepts_text(mid)]
    if not waiting:
        return False

    reply_to = (meta or {}).get('reply_to')
    if reply_to:
        try:
            target = int(reply_to)
        except (TypeError, ValueError):
            return False
        if target not in waiting:
            return False
    elif len(waiting) == 1:
        target = waiting[0]
    else:
        _tell(_('Several questions are waiting'),
              _('Reply to the question you are answering, then send it again.'))
        return True

    inbox.deliver_answer(target, 't:' + body)
    return True


def _tell(title: str, body: str) -> None:
    try:
        from .events import EventKind, Sensitivity
        from .notify import notify
        notify(EventKind.INFO, title, body=body,
               sensitivity=Sensitivity.PUBLIC, blocking=True)
    except Exception:
        pass
