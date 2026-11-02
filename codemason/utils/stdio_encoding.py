"""UTF-8 stdio on Windows.

On a Korean/Japanese/Chinese Windows the process locale encoding is a legacy
codepage (cp949, cp932, cp936). When stdout is a real console Python writes
through WriteConsoleW and Unicode survives, but the moment output is piped or
redirected the streams fall back to that codepage -- and any non-latin1
character in our output (the em dash in argparse help, box-drawing glyphs,
spinner frames) dies with UnicodeEncodeError before the CLI does any work.

Force UTF-8 on the standard streams so redirected output behaves like a console.
"""

import sys

__all__ = ["ensure_utf8_stdio"]


def ensure_utf8_stdio() -> None:
    """Reconfigure stdin/stdout/stderr to UTF-8. No-op off Windows."""
    if sys.platform != "win32":
        return

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        # None under pythonw / a detached process; already-UTF-8 streams are cheap
        # to reconfigure again. Non-TextIOWrapper replacements (pytest capture,
        # some IDE consoles) have no .reconfigure -- leave those alone.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            # errors="replace": a glyph the terminal can't render should degrade to
            # "?" rather than abort the CLI.
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
