"""Cross-platform ``termios`` / ``tty`` access.

POSIX TTY control (``termios``, ``tty``) does not exist on native Windows
Python, so a top-level ``import termios`` crashes the whole CLI at startup
before it reaches any Windows substrate. This module imports the real modules
where they exist and otherwise exposes drop-in stubs whose read operations
raise (so every existing ``except (termios.error, ValueError, OSError)``
fallback fires) and whose write operations no-op.

Callers keep their POSIX code path verbatim; on Windows the raw-mode / echo
control simply degrades to "no terminal control" (the same fallback taken when
stdin is not a TTY), until msvcrt-based keypress handling lands. Import as:

    from ..utils.tty_compat import termios, tty
"""

try:  # POSIX
    import termios  # type: ignore
    import tty  # type: ignore
except ImportError:  # native Windows: no POSIX termios/tty

    class _NoTermios:
        # `.error` and the flag constants are referenced in except-clauses and
        # attribute math; keep them present so those lines don't NameError.
        error = OSError
        TCSADRAIN = TCSANOW = TCIFLUSH = ECHO = ICANON = 0

        def tcgetattr(self, *args):
            raise OSError("termios is unavailable on Windows")

        def tcsetattr(self, *args):
            pass

        def tcflush(self, *args):
            pass

    class _NoTty:
        def setcbreak(self, *args):
            raise OSError("tty is unavailable on Windows")

        def setraw(self, *args):
            raise OSError("tty is unavailable on Windows")

    termios = _NoTermios()  # type: ignore
    tty = _NoTty()  # type: ignore


_saved_attrs = None


def suppress_echo():
    """Stop the kernel echoing keystrokes typed before the prompt exists.

    Startup does real work before prompt_toolkit owns the terminal (model
    resolve, remote channel), and until then the tty is in canonical mode: the
    KERNEL echoes anything typed and queues it. Those keystrokes get painted
    across the startup banner and then land in the first prompt, in a buffer the
    user never saw fill. Suppressing echo for the startup window keeps them off
    the screen; :func:`flush_input` drops the queued bytes. Canonical mode is
    otherwise untouched, so banner output is unaffected.
    """
    global _saved_attrs
    import sys
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
    except (termios.error, ValueError, OSError, AttributeError):
        return  # not a tty (piped/dumb), or Windows
    if _saved_attrs is None:
        _saved_attrs = attrs
    new_attrs = list(attrs)
    new_attrs[3] &= ~termios.ECHO  # lflag
    try:
        termios.tcsetattr(fd, termios.TCSANOW, new_attrs)
    except (termios.error, ValueError, OSError):
        pass


def restore_echo():
    """Put echo back the way it was found (no-op if never suppressed)."""
    global _saved_attrs
    if _saved_attrs is None:
        return
    import sys
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, _saved_attrs)
    except (termios.error, ValueError, OSError, AttributeError):
        pass
    _saved_attrs = None


def flush_input():
    """Discard input buffered before we were ready for it (TCIFLUSH)."""
    import sys
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (termios.error, ValueError, OSError, AttributeError):
        pass


__all__ = ["termios", "tty", "suppress_echo", "restore_echo", "flush_input"]
