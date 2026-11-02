"""
Optional terminal colour-theme application via OSC escape sequences.

This repaints the CURRENT terminal session (default foreground/background/cursor)
to look like a named theme — e.g. macOS Terminal's "Homebrew" profile (green on
black) — for terminals that don't already use it. It is OPT-IN
(`--theme homebrew` / CODEMASON_THEME=homebrew); by default codemason leaves the
terminal's own colours alone.

Restore strategy (see apply_theme):
  - atexit  -> covers normal exit, `exit`/`quit`, Ctrl+D, and a Ctrl+C that
               propagates out of the program (KeyboardInterrupt -> clean shutdown
               still runs atexit). A Ctrl+C that the REPL merely uses to cancel a
               query does NOT exit, so colours correctly stay applied.
  - SIGTERM / SIGHUP handlers -> cover being killed or an SSH disconnect, which
               otherwise terminate without running atexit.
  - We deliberately do NOT trap SIGINT (Ctrl+C) ourselves — in the REPL that
    often means "cancel the current query", not "quit"; atexit handles the real
    exit case.

HARD LIMIT (honest): a crash or `kill -9` (SIGKILL) kills the process before it
can emit the reset, so the terminal stays themed until the user resets it
manually (reconnect, `tput reset`, or
`printf '\\033]110\\033\\\\\\033]111\\033\\\\\\033]104\\033\\\\'`).
"""

import atexit
import os
import signal
import sys

# macOS Terminal.app "Homebrew" profile: bright lime-green text on black.
# 'prompt'/'input' are NOT terminal OSC colours — they're consumed by the REPL's
# prompt_toolkit Style (cli.py) to keep the '>' marker and the user's typed text
# from disappearing into the all-green body. Amber marker + warm-white input give
# high contrast against homebrew green/black.
_THEMES = {
    'homebrew': {
        'fg': '#28fe14',      # foreground (default text)
        'bg': '#000000',      # background
        'cursor': '#28fe14',  # cursor
        'prompt': '#ffb000',  # REPL '>' marker (amber — stands out from green)
        'input': '#f5f0e8',   # REPL typed text (warm white — distinct from body)
    },
}

_applied = False
_active_name = None


def active_theme_colors() -> dict:
    """Colour map of the theme applied this session (or {} if none). Lets the
    REPL tint its prompt/input to match — see cli.py."""
    return _THEMES.get(_active_name, {}) if _active_name else {}


def _osc(code: str, value: str = '') -> str:
    """Build an OSC sequence terminated with ST (ESC \\)."""
    body = f'{code};{value}' if value else code
    return f'\033]{body}\033\\'


def _write(seq: str) -> None:
    try:
        sys.stdout.write(seq)
        sys.stdout.flush()
    except Exception:
        pass


def restore() -> None:
    """Revert fg/bg/cursor/palette to the terminal's OWN configured defaults via
    OSC reset codes — no need to know the original values. Idempotent."""
    global _applied
    if not _applied:
        return
    global _active_name
    _applied = False
    _active_name = None
    _write(_osc('110'))   # reset default foreground
    _write(_osc('111'))   # reset default background
    _write(_osc('112'))   # reset cursor colour
    _write(_osc('104'))   # reset colour palette


def _on_terminating_signal(signum, _frame) -> None:
    # Restore, then let the signal do its default thing (terminate).
    restore()
    try:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except Exception:
        os._exit(128 + signum)


def apply_theme(name: str) -> bool:
    """Apply a named terminal theme for this session and arrange restoration.

    Returns True if applied. No-op (returns False) for an unknown theme, a
    non-tty stdout, or pure-ASCII/plain mode (dumb terminals don't do OSC).
    """
    global _applied
    if not name:
        return False
    name = name.strip().lower()
    theme = _THEMES.get(name)
    if theme is None:
        return False
    # Don't paint a non-interactive stream or a dumb/plain terminal.
    from .formatting import plain_mode
    if plain_mode() or not sys.stdout.isatty():
        return False

    _write(_osc('11', theme['bg']))      # default background
    _write(_osc('10', theme['fg']))      # default foreground
    _write(_osc('12', theme['cursor']))  # cursor
    global _active_name
    _active_name = name
    _applied = True

    atexit.register(restore)
    # SIGTERM/SIGHUP only — see module docstring for why not SIGINT.
    # SIGHUP does not exist on native Windows (no SSH-disconnect signal there), and
    # naming it would AttributeError before the try below can catch anything.
    _signals = [signal.SIGTERM]
    _sighup = getattr(signal, 'SIGHUP', None)
    if _sighup is not None:
        _signals.append(_sighup)
    for sig in _signals:
        try:
            signal.signal(sig, _on_terminating_signal)
        except (ValueError, OSError):
            pass  # e.g. not in main thread
    return True
