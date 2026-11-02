"""Inline arrow-key picker (radio / checkbox), built on prompt_toolkit the same
way as resume_picker: a short-lived Application that renders inline (not
full-screen, so REPL scrollback survives). In the interactive REPL it must run
while the persistent PromptSession is PAUSED (one Application per terminal at a
time) — see ReplInputRouter.pause_prompt/resume_prompt. Falls back to numbered
input when an Application can't run (e.g. non-tty).

Used by AskUserQuestion so the user picks with ↑/↓/enter instead of typing a
number into a raw input() that fights the REPL's prompt_toolkit session.
"""
from typing import List, Optional, Sequence

from .i18n import _


async def arrow_select(
    title: str,
    options: Sequence[str],
    *,
    descriptions: Optional[Sequence[Optional[str]]] = None,
    previews: Optional[Sequence[Optional[str]]] = None,
    multi: bool = False,
    header: Optional[str] = None,
    close_when=None,
) -> Optional[List[int]]:
    """Present an inline picker over `options`.

    Returns the selected option indices (a single-element list for single-select),
    or None if the user cancelled (Esc / Ctrl+C). For multi-select, Enter with
    nothing toggled selects the currently-highlighted row.

    `close_when` is an asyncio.Event that takes the picker down from outside —
    used when the same question was answered somewhere else (the phone). It has
    to exit the Application rather than cancel the task around it: cancelling
    leaves prompt_toolkit mid-render with the terminal never restored, and the
    next thing printed lands inside the prompt line instead of above it.
    """
    if not options:
        return None
    try:
        return await _select_pt(
            title, list(options), descriptions, previews, multi, header, close_when)
    except Exception:
        # Any prompt_toolkit failure (no tty, nested-loop quirk) → plain numbers.
        return _select_numbered(title, list(options), descriptions, multi, header)


async def _select_pt(title, options, descriptions, previews, multi, header, close_when=None):
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.output import ColorDepth

    n = len(options)
    state = {'index': 0, 'selected': set()}

    # Reserve a fixed preview area (max lines over all options) so the inline
    # height doesn't jump as the cursor moves between rows.
    max_prev = 0
    if previews:
        max_prev = max(
            (len(str(p).split('\n')) for p in previews if p), default=0)

    def render():
        lines = []
        if header:
            lines.append(('bold', f"[{header}]\n"))
        lines.append(('bold', title + '\n'))
        for i, opt in enumerate(options):
            cur = i == state['index']
            cursor = '❯' if cur else ' '
            if multi:
                box = '[x]' if i in state['selected'] else '[ ]'
                body = f"{cursor} {box} {opt}"
            else:
                body = f"{cursor} {opt}"
            lines.append(('bold' if cur else '', body))
            # Every option carries its description, not just the highlighted one.
            # Showing one at a time meant three of four choices were unexplained
            # right when the choice is being made — and the single line, printed
            # under the list, read as though something had already been picked.
            desc = descriptions[i] if descriptions and i < len(descriptions) else None
            if desc:
                lines.append(('fg:#8a8580', f"  — {desc}"))
            lines.append(('', '\n'))
        # fixed preview area for the highlighted row
        if max_prev:
            pv = []
            if previews and state['index'] < len(previews) and previews[state['index']]:
                pv = str(previews[state['index']]).split('\n')
            for j in range(max_prev):
                seg = pv[j] if j < len(pv) else ''
                lines.append(('', f"    │ {seg}\n"))
        hint = (_('↑/↓ move · space toggle · enter confirm · esc cancel')
                if multi else
                _('↑/↓ move · enter select · esc cancel'))
        lines.append(('', hint + '\n'))
        return lines

    kb = KeyBindings()

    @kb.add('up')
    def _up(event):
        state['index'] = (state['index'] - 1) % n

    @kb.add('down')
    def _down(event):
        state['index'] = (state['index'] + 1) % n

    if multi:
        @kb.add('space')
        def _toggle(event):
            i = state['index']
            state['selected'].discard(i) if i in state['selected'] else state['selected'].add(i)

        @kb.add('enter')
        def _confirm(event):
            sel = sorted(state['selected']) or [state['index']]
            event.app.exit(result=sel)
    else:
        @kb.add('enter')
        def _confirm(event):
            event.app.exit(result=[state['index']])

    @kb.add('escape', eager=True)
    @kb.add('c-c')
    def _cancel(event):
        event.app.exit(result=None)

    # height = header? + title + rows + desc-line? + preview area + hint
    height = (1 if header else 0) + 1 + n + max_prev + 1
    control = FormattedTextControl(render, focusable=True, show_cursor=False)
    app = Application(
        layout=Layout(Window(control, height=height, wrap_lines=False)),
        key_bindings=kb,
        full_screen=False,
        color_depth=ColorDepth.DEPTH_8_BIT,
    )
    if close_when is None:
        return await app.run_async()

    # Someone else may answer first. Watch for that alongside the keyboard and
    # close the Application the way Esc would, so the screen is put back.
    import asyncio

    async def _watch():
        await close_when.wait()
        if app.is_running:
            app.exit(result=None)

    watcher = asyncio.ensure_future(_watch())
    try:
        return await app.run_async()
    finally:
        watcher.cancel()


def _select_numbered(title, options, descriptions, multi, header):
    """Fallback for terminals where the Application can't run."""
    if header:
        print(f"[{header}]")
    print(title)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
        if descriptions and i - 1 < len(descriptions) and descriptions[i - 1]:
            print(f"     {descriptions[i - 1]}")
    hint = "numbers comma-separated" if multi else "a number"
    while True:
        try:
            raw = input('  ' + _('Select {hint}: ', hint=hint)).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            return None
        picks = [p.strip() for p in raw.split(',')] if multi else [raw]
        idxs = []
        ok = True
        for p in picks:
            if not p.isdigit() or not (1 <= int(p) <= len(options)):
                ok = False
                break
            idxs.append(int(p) - 1)
        if ok and idxs:
            return idxs
