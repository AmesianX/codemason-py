"""
Interactive session picker for `resume` — adapted from components/ResumeTask.tsx's
session list UX, adapted from INK to prompt_toolkit (the documented substrate
boundary: RAW-mode React components become prompt_toolkit Applications).

Shows the current project's stored sessions newest-first with modified time,
message count and first prompt; ↑/↓ to move, Enter to pick, d-d to delete the
selected session (double-press confirm), Esc/q to cancel.
Falls back to a numbered text prompt when the terminal can't run a
full-screen-less prompt_toolkit Application (e.g. dumb terminals).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .formatting import display_width
from .session_storage import list_sessions

# Cap the list like the original picker viewport; extension point: paging.
MAX_SESSIONS = 50


def _format_age(iso_ts: str) -> str:
    """Relative modified time, always down to seconds ('42s ago',
    '5m 3s ago', '2h 14m 3s ago', '3d 7h 12m 5s ago') — every row visibly
    ticks with the picker's 1s refresh, day-old sessions included."""
    try:
        ts = datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
    except ValueError:
        return '?'
    delta = datetime.now(timezone.utc) - ts
    secs = max(0, int(delta.total_seconds()))
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or parts:
        parts.append(f"{h}h")
    if m or parts:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return ' '.join(parts) + ' ago'


def _row_text(s: Dict[str, Any], width: int) -> str:
    age = _format_age(s.get('modified', ''))
    count = s.get('message_count', 0)
    prompt = s.get('first_prompt', 'No prompt')
    head = f"{age:>16}  {count:>4} msg  "
    room = max(10, width - display_width(head) - 4)
    if display_width(prompt) > room:
        # display_width-aware truncation (CJK chars are 2 cells wide)
        out, used = [], 0
        for ch in prompt:
            w = display_width(ch)
            if used + w > room - 1:
                break
            out.append(ch)
            used += w
        prompt = ''.join(out) + '…'
    return head + prompt


async def pick_session() -> Optional[Dict[str, Any]]:
    """Arrow-key session picker. Returns the chosen session dict from
    list_sessions(), or None if cancelled / nothing to resume."""
    sessions = list_sessions(limit=MAX_SESSIONS)
    # Don't offer the ACTIVE session — resuming the conversation you're in is
    # a no-op (and adopt would needlessly reload it).
    from .state import get_session_id
    current = get_session_id()
    sessions = [s for s in sessions if s.get('session_id') != current]
    if not sessions:
        print("No stored sessions for this directory.")
        return None

    import sys
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        # No tty (piped/dumb) — the Application would render garbage before
        # failing; go straight to the numbered prompt.
        return _pick_with_numbers(sessions)
    try:
        return await _pick_with_prompt_toolkit(sessions)
    except Exception:
        return _pick_with_numbers(sessions)


async def _pick_with_prompt_toolkit(
    sessions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.output import ColorDepth
    import shutil

    # delete_armed: index awaiting a second 'd' to confirm deletion, or None.
    state = {'index': 0, 'delete_armed': None}
    width = shutil.get_terminal_size((80, 24)).columns

    def render():
        lines = [('bold',
                  'Resume a session (↑/↓ move, Enter select, d delete, Esc cancel)\n')]
        for i, s in enumerate(sessions):
            row = _row_text(s, width)
            if i == state['index']:
                if state['delete_armed'] == i:
                    lines.append(('fg:ansired reverse', f"❯ {row}  (d again to delete)\n"))
                else:
                    lines.append(('reverse', f"❯ {row}\n"))
            else:
                lines.append(('', f"  {row}\n"))
        return lines

    kb = KeyBindings()

    def _move(delta):
        state['index'] = (state['index'] + delta) % len(sessions)
        state['delete_armed'] = None  # moving away disarms a pending delete

    @kb.add('up')
    def _up(event):
        _move(-1)

    @kb.add('down')
    def _down(event):
        _move(1)

    @kb.add('enter')
    def _enter(event):
        event.app.exit(result=sessions[state['index']])

    @kb.add('d')
    def _delete(event):
        i = state['index']
        if state['delete_armed'] != i:
            state['delete_armed'] = i  # first press arms; second confirms
            return
        state['delete_armed'] = None
        from .session_storage import delete_session
        if not delete_session(Path(sessions[i]['full_path'])):
            return  # already gone / unreadable — just leave the row
        sessions.pop(i)
        if not sessions:
            event.app.exit(result=None)
            return
        state['index'] = min(i, len(sessions) - 1)

    @kb.add('escape', eager=True)
    @kb.add('q')
    @kb.add('c-c')
    def _cancel(event):
        event.app.exit(result=None)

    control = FormattedTextControl(render, focusable=True, show_cursor=False)
    # Height = header + rows: render inline (not full-screen) so the REPL
    # scrollback isn't destroyed — matches the original's inline list feel.
    # refresh_interval: redraw once a second even with no input, so the
    # "Ns ago" column ticks in real time instead of only on keypress.
    app: Application = Application(
        layout=Layout(Window(control, height=len(sessions) + 1, wrap_lines=False)),
        key_bindings=kb,
        full_screen=False,
        color_depth=ColorDepth.DEPTH_8_BIT,
        refresh_interval=1.0,
    )
    return await app.run_async()


def _pick_with_numbers(
    sessions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Plain fallback for terminals where the Application can't run."""
    import shutil
    width = shutil.get_terminal_size((80, 24)).columns
    print("Stored sessions:")
    for i, s in enumerate(sessions, 1):
        print(f"  {i:>2}. {_row_text(s, width)}")
    try:
        raw = input("Session number (empty to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    try:
        idx = int(raw)
    except ValueError:
        print("Not a number.")
        return None
    if not (1 <= idx <= len(sessions)):
        print("Out of range.")
        return None
    return sessions[idx - 1]
