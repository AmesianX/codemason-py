#!/usr/bin/env python3
"""
CLI Entry Point for CodeMason
Equivalent to src/entrypoints/cli.tsx
"""

import argparse
import atexit
import asyncio
import json
import os
import random
import sys
import time
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .constants.prompts import get_system_prompt
from .tools import get_all_tools, get_tool_by_name, Tool
from .types.query import ToolUseContext
from .query import query
from .utils.stdio_encoding import ensure_utf8_stdio
from .utils.windows_shell import ensure_utf8_locale
from .utils.tty_compat import termios, tty
from .utils.state import get_cwd, set_cwd, set_original_cwd
from .utils.config import get_config, save_config
from .utils.hooks import has_hook_for_event
from .utils.log import log_error
from .utils.messages import create_user_message, REJECT_MESSAGE
from .utils.paste_store import (
    PasteStore,
    expand_pasted_refs,
    format_pasted_ref,
    pasted_num_lines,
    should_placeholder,
)
from .utils.surrogates import repair_surrogates
from .utils.tokens import token_count_with_estimation


PRODUCT_NAME = "CodeMason"

_BUILD_ID: Optional[str] = None


def build_id() -> str:
    """The commit this source tree is at, with `-dirty` when it has uncommitted
    changes, and the commit's date -- '1a2b3c4 2026-10-31'.

    Mirrors BUILD_ID in the Rust client (src/cli.rs), which build.rs bakes into
    the binary at compile time. WHY it exists at all: the banner printed a
    version string and nothing else, and the version string had not moved in
    197 commits, so two copies from either side of a day of fixes both
    introduced themselves the same way and which one was running had to be
    settled from outside the program.

    There is no compile step here, so it is resolved from git the first time it
    is asked for -- which for an interpreted program is the more honest answer
    anyway: the source on disk IS what is running, so `-dirty` describes the
    code of this very process rather than of some earlier build. Two git
    invocations, once per run, and only when the banner or /help asks.

    Empty when this is not a git checkout (an installed or copied tree). The
    caller then prints the version alone rather than '(unknown unknown)', which
    says less than nothing.
    """
    global _BUILD_ID
    if _BUILD_ID is not None:
        return _BUILD_ID
    _BUILD_ID = ''
    try:
        import subprocess as _sp
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def _git(*args) -> str:
            r = _sp.run(('git', '-C', _root) + args, capture_output=True,
                        text=True, timeout=2)
            return r.stdout.strip() if r.returncode == 0 else ''

        _h = _git('rev-parse', '--short', 'HEAD')
        if _h:
            # Exit status only. `git diff --quiet` writes NOTHING to stdout --
            # that is the point of the flag -- so a helper that treats empty
            # output as failure would call every clean tree dirty.
            _dirty = _sp.run(('git', '-C', _root, 'diff', '--quiet', 'HEAD', '--'),
                             capture_output=True, timeout=2).returncode != 0
            if _dirty:
                _h += '-dirty'
            _d = _git('log', '-1', '--format=%cd', '--date=format:%Y-%m-%d')
            _BUILD_ID = f'{_h} {_d}' if _d else _h
    except Exception:
        pass
    return _BUILD_ID


def _num(v) -> str:
    """Render a number the way Rust's `{}` does, so the two banners read the
    same: 1.0 -> '1', 0.95 -> '0.95'.

    Python's str() keeps the trailing '.0' and Rust's f64 Display drops it, and
    that one character was a visible difference on the Sampling line between the
    two clients for no reason anybody chose.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f or f in (float('inf'), float('-inf')):   # NaN / inf
        return repr(f)
    return str(int(f)) if f == int(f) else repr(f)


from .utils.i18n import _
from .utils.formatting import (
    Colors,
    clip_to_width,
    colorize,
    display_width,
    terminal_columns,
    format_number,
    format_tokens_compact,
    format_token_display,
    safe_print,
    sanitize_terminal_text,
    render_markdown,
    plain_mode,
    summarize_tool_input,
    sym,
    display_width,
)
from .utils.log import debug_log


def get_token_warning(token_usage: int) -> str:
    """Get context warning message if approaching limits (thresholds)."""
    from .services.compaction import calculate_token_warning_state
    state = calculate_token_warning_state(token_usage)
    if not state['is_above_warning_threshold']:
        return ""

    used_pct = 100 - state['percent_left']

    # Progress bar
    bar_width = 20
    filled = int(bar_width * used_pct / 100)
    bar = sym('bar_full') * filled + sym('bar_empty') * (bar_width - filled)

    if state['is_above_error_threshold']:
        return colorize(f"{sym('warn')} Context: [{bar}] {used_pct}% - Run '/compact' to continue", Colors.RED)
    else:
        return colorize(f"{sym('bolt')} Context: [{bar}] {used_pct}%", Colors.YELLOW)


def _usage_field(usage: Any, key: str) -> int:
    """Read one counter off a usage record that may be a dict OR a BetaUsage.

    Both shapes reach the display code (msg_obj.get('usage') hands back a dict,
    the attribute path a BetaUsage), and a plain getattr silently returns 0 for
    the dict — which reads as "this response reported no usage" and sends every
    caller down its fallback path."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return usage.get(key, 0) or 0
    return getattr(usage, key, 0) or 0


def _reported_usage(usage: Any) -> bool:
    """True when the response actually carried token counts."""
    return bool(_usage_field(usage, 'input_tokens')
                or _usage_field(usage, 'output_tokens'))


def _render_token_footer(usage: Any, duration_ms: int, spinner: 'StreamingSpinner',
                         messages: List[Any], verbose: bool) -> str:
    """The dim per-response line ('⎯ ↓ 51k / 232.8k (22%) · 3.2s'), or '' when
    there is no duration to report yet.

    The fallback is the point of having this in one place. When a response
    carries no usage — a server that never sent the final include_usage chunk, a
    stream cut short by Esc — the count fell back to the spinner's streamed
    tokens, which are OUTPUT ONLY. That figure is then printed as context fill
    against the auto-compact threshold, so a conversation genuinely sitting at
    40% reports ~1%: not merely imprecise but reassuring in exactly the
    situation the line exists to warn about. The honest stand-in is the estimate
    compaction itself sizes on — the whole history, images and pending tool
    results included."""
    elapsed, _streamed = spinner.get_final_stats()
    display_duration = duration_ms if duration_ms > 0 else int(elapsed * 1000)
    if display_duration <= 0:
        return ''
    fallback = 0
    if not _reported_usage(usage):
        try:
            # The estimate compaction itself sizes on: it anchors on the last
            # response that reported usage (whose input_tokens also covers the
            # system prompt and tool definitions, invisible to a walk over
            # `messages`) and estimates everything after it. The just-arrived
            # zero-usage response no longer anchors — token_count_with_estimation
            # skips it — so this no longer collapses to 0 the way it did before
            # that guard existed.
            fallback = token_count_with_estimation(messages)
        except Exception:  # noqa: BLE001 - a display line must not end a turn
            fallback = 0
    return colorize(
        format_token_display(usage, display_duration, show_details=verbose,
                             fallback_tokens=fallback), Colors.DIM)


# Taken from the package, never spelled out again. Two literals for one version
# is the same shape of bug as two defaults for one setting: they agree until a
# release forgets one, and then the banner and the package disagree with nobody
# the wiser.
from . import __version__ as VERSION


# === Spinner for Streaming ===

_BRAILLE_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
_ASCII_FRAMES = ['|', '/', '-', '\\']
_cached_spinner_frames = None


def _terminal_supports_braille() -> bool:
    """Best-effort check for braille (U+2800–28FF) glyph rendering.

    HONEST LIMITS — there is NO terminal capability that reports font glyph
    coverage (TERM/COLORTERM report colour DEPTH, not glyphs). So 'auto' can
    only check:
      1. UTF-8 output — necessary, not sufficient. A non-UTF-8 terminal cannot
         render braille at all → ASCII. (reliable)
      2. A cursor-advance (DSR ESC[6n) probe — catches braille drawn at the
         wrong cell width (0 or 2 cells). It does NOT catch the common 'tofu
         box' case, where a missing glyph still occupies exactly 1 cell.
    A font that draws braille as boxes passes this probe; set
    CODEMASON_SPINNER=ascii to force ASCII there.
    """
    enc = (getattr(sys.stdout, 'encoding', '') or '').lower()
    if 'utf' not in enc:
        return False  # reliable: non-UTF-8 can't render braille at all
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        return True  # can't probe (piped); UTF-8 ok, prefer the nicer spinner

    import select
    import re
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except (termios.error, ValueError, OSError):
        return True
    try:
        tty.setcbreak(fd)

        def _col() -> Optional[int]:
            # Device Status Report: terminal replies ESC[<row>;<col>R
            sys.stdout.write('\033[6n')
            sys.stdout.flush()
            r, _, _ = select.select([fd], [], [], 0.15)
            if not r:
                return None
            data = os.read(fd, 32).decode('latin-1', 'replace')
            m = re.search(r'\[(\d+);(\d+)R', data)
            return int(m.group(2)) if m else None

        c0 = _col()
        if c0 is None:
            return True  # terminal didn't answer DSR — don't penalise braille
        sys.stdout.write('⠿')  # a fully-set braille cell
        sys.stdout.flush()
        c1 = _col()
        sys.stdout.write('\b \b')  # erase the probe glyph
        sys.stdout.flush()
        if c1 is None:
            return True
        return (c1 - c0) == 1  # exactly one cell == consistent with a real glyph
    except Exception:
        return True
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def _spinner_frames() -> list:
    """Pick spinner frames. Default braille; CODEMASON_SPINNER overrides with
    'braille' | 'ascii' | 'auto'. 'auto' = braille unless the probe above says
    otherwise. Result is cached (the probe runs at most once per process)."""
    global _cached_spinner_frames
    if _cached_spinner_frames is not None:
        return _cached_spinner_frames
    if plain_mode():
        _cached_spinner_frames = _ASCII_FRAMES  # pure-ASCII: never braille/probe
        return _cached_spinner_frames
    mode = os.environ.get('CODEMASON_SPINNER', 'auto').strip().lower()
    if mode == 'ascii':
        frames = _ASCII_FRAMES
    elif mode == 'braille':
        frames = _BRAILLE_FRAMES
    else:  # auto
        frames = _BRAILLE_FRAMES if _terminal_supports_braille() else _ASCII_FRAMES
    _cached_spinner_frames = frames
    return frames


# The turn a finished background task starts on its own.
#
# Not empty: _run_turn sends what it is given, and a blank line reads as the
# user having pressed enter for no reason. The notification blocks are already
# in the history by the time this runs, so this only has to say why the model
# is being addressed again.
_TASK_WAKE_PROMPT = (
    "A background task you started has finished — its <task-notification> is "
    "above, with the path to its output. Pick the work back up from there."
)


class StreamingSpinner:
    """Animated spinner with real-time token count display.

    Uses an asyncio background task to animate independently of token updates,
    so the spinner keeps moving even when no tokens are arriving (e.g. during
    initial model processing or tool execution).
    """
    def __init__(self, external_render: bool = False):
        # Braille by default; falls back to ASCII when CODEMASON_SPINNER=ascii
        # or the auto-probe finds braille unsupported (see _spinner_frames).
        self.frames = _spinner_frames()
        self.frame_idx = 0
        self.token_count = 0
        # Reasoning tokens for THIS turn. Stays 0 on servers/turns that don't
        # think, and the status line then looks exactly as it did before — the
        # breakdown is rendered only when there is something to break down.
        self.thinking_tokens = 0
        self.start_time = None
        self.active = False
        self.last_line_length = 0
        self._animation_task = None
        # external_render: the interactive REPL renders this spinner in the
        # prompt's bottom_toolbar (toolbar_text(), repainted by prompt_toolkit's
        # refresh_interval) — this object must then never write to the terminal
        # itself, or the \r-overwrites would corrupt the persistent prompt.
        self.external_render = external_render
        # Whimsical verb shown while working (Claude Code parity). Picked once
        # per turn in start() and held stable for the turn (no per-frame
        # flicker) — matches source Spinner.tsx useState(()=>sample(verbs)).
        self.verb = self._pick_verb()

    @staticmethod
    def _pick_verb() -> str:
        from .constants.spinner_verbs import get_spinner_verbs
        verbs = get_spinner_verbs()
        return random.choice(verbs) if verbs else "Thinking"

    def _display_verb(self) -> str:
        """Claude Code verb priority (Spinner.tsx:169): the in_progress todo's
        activeForm ('Running the scan…', authored by the model in TodoWrite)
        takes precedence, else its content, else the stable random whimsical
        verb. Resolved LIVE on each render so it tracks todo changes mid-turn.
        Todos are keyed by agentId ?? sessionId — the leader spinner uses the
        session id. Any failure falls back to the random verb (never crash the
        spinner over a missing todo store)."""
        # A pinned verb wins. Without this, start(verb='Compacting') was
        # silently overridden whenever a todo was in_progress — the caller asked
        # for a name and got the name of the task compaction INTERRUPTED, so a
        # reduction was reported as whatever the turn happened to be doing.
        if getattr(self, '_verb_pinned', False):
            return self.verb
        try:
            # The MAIN conversation's todos — the same key TodoWrite writes
            # under. This used to read get_session_id(), a UUID that nothing
            # ever stored against, so the loop below never ran.
            from .utils.todo_store import get_todos, todo_key
            for t in get_todos(todo_key()):
                if t.get('status') == 'in_progress':
                    return (t.get('activeForm') or t.get('content') or self.verb).strip() or self.verb
        except Exception:
            pass
        return self.verb

    def start(self, verb: Optional[str] = None):
        """Start the spinner (resets token count + picks a fresh verb).

        `verb` pins the label instead of picking a whimsical one, for work that
        is not a turn and whose name is the useful part — compaction being the
        one that matters, since it is the longest wait in the session and
        nothing else on screen says what is happening.
        """
        import time
        self.start_time = time.time()
        self.token_count = 0
        self.thinking_tokens = 0
        self.frame_idx = 0
        self.active = True
        # new turn → new verb, stable until next start()
        self.verb = verb or self._pick_verb()
        self._verb_pinned = verb is not None
        self._verb_saved = None   # a new turn carries no displaced label
        self._render()
        self._start_animation()

    def pin_verb(self, verb: Optional[str] = None):
        """Re-pin the label MID-TURN without touching the counters, for work
        that interrupts a turn and then hands it back (a compaction). None
        unpins and restores the label the turn went in with.

        start() cannot be used for this: it resets the elapsed clock, the token
        count and the thinking count, so a reduction in the middle of a turn
        would throw away the turn's own progress display.

        The elapsed clock stays the TURN's, deliberately -- it is the one
        number here that keeps describing the turn rather than the
        interruption, and resetting it would make the turn's time jump
        backwards the moment the label is handed back."""
        if verb is not None:
            if getattr(self, '_verb_saved', None) is None:
                self._verb_saved = self.verb
            self.verb = verb
            self._verb_pinned = True
        else:
            if getattr(self, '_verb_saved', None) is not None:
                self.verb = self._verb_saved
            self._verb_saved = None
            self._verb_pinned = False
        self._render()

    def resume(self):
        """Resume spinner without resetting token count (for Agent progress)"""
        if not self.active:
            self.active = True
            if self.start_time is None:
                import time
                self.start_time = time.time()
            self._render()
            self._start_animation()

    def _start_animation(self):
        """Launch background asyncio task for spinner animation"""
        if self.external_render:
            return  # the toolbar repaints on refresh_interval; no task needed
        if self._animation_task and not self._animation_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._animation_task = loop.create_task(self._animate())
        except RuntimeError:
            pass

    async def _animate(self):
        """Background animation loop - updates frame every 100ms"""
        try:
            while self.active:
                await asyncio.sleep(0.1)
                if self.active:
                    self.frame_idx += 1
                    self._render()
        except asyncio.CancelledError:
            pass

    def update(self, token_count: int):
        """Update token count and re-render"""
        if not self.active:
            return
        self.token_count = token_count
        self._render()

    def update_thinking(self, thinking_tokens: int):
        """Update the reasoning-token count and re-render.

        Only ever called while the server is emitting reasoning, so a turn with
        no thinking never touches this and the counter stays hidden."""
        if not self.active:
            return
        self.thinking_tokens = thinking_tokens
        self._render()

    def _thinking_segment(self) -> str:
        """' · ✻ 234 thinking', or '' when this turn did no thinking.

        The thinking tokens are a SUBSET of token_count (the server bills them in
        completion_tokens), so this is a breakdown, not an extra total."""
        if self.thinking_tokens <= 0:
            return ''
        return (f" {sym('dot')} {sym('think')} "
                f"{format_tokens_compact(self.thinking_tokens)} thinking")

    def _wait_segment(self) -> str:
        """' — fetching example.com', or '' when no tool has said what it is
        waiting on. Sits next to the elapsed time, where the question is asked."""
        from .utils import tokens as _tk
        n = _tk.wait_note()
        return f" {sym('dash')} {n}" if n else ''

    # Prefill big enough that the wait is the prompt rather than the model, and
    # so worth saying out loud without -dbg. At the cold rate measured for this
    # deployment (1,335 tok/s) this is roughly 15 seconds before the first token
    # -- about where an unexplained spinner starts reading as a hang.
    PREFILL_NOTEWORTHY_TOKENS = 20_000

    def _prefill_segment(self) -> str:
        """' · ↑ 73.1k to prefill' while the server is still reading the prompt
        and nothing has come back yet.

        Before the first output token there is no way to tell a warm request
        (answer in ~2s) from one whose cached prefix broke (73k re-read, 55s) —
        both show a spinner and a clock. This says which, and how much is left
        to read, so a long wait is legible instead of looking like a freeze. It
        disappears the moment real output starts. '~' marks our own estimate;
        no mark means vLLM counted it."""
        # Diagnostic, not everyday furniture: only under -dbg.
        #
        # It earned its place by settling an argument — 297 seconds of
        # "Calculating…" turned out to be "↑ 996 to prefill, 54.3k cached ·
        # ↓ 0 tokens", i.e. neither prefill nor decode, which is what finally
        # pointed at a tool waiting on itself. That question is answered now,
        # and a number nobody is currently asking about is just noise on the
        # line. -dbg brings it back, along with turns.log.
        if self.token_count > 0:
            return ''
        from .utils import tokens as _tk
        p = _tk.pending_prefill()
        if p <= 0:
            return ''
        # Under -dbg: always, as a diagnostic. Otherwise only when the wait is
        # the PROMPT and not the model, which is the one case where silence is
        # indistinguishable from a freeze.
        #
        # The clean-line argument above still holds for ordinary turns, and they
        # stay clean: a warm request re-reads a few thousand tokens and never
        # crosses the bar. What it did not cover is a reduction clearing an
        # early tool result, which invalidates every cached token after it: in
        # the Rust port on 2026-10-03 17:55 that was 242,795 tokens at 0.0%
        # cache hit and 182 seconds before the first token, spent showing the
        # turn's own label and no reason at all.
        from .utils.log import debug_enabled
        if not debug_enabled() and p < self.PREFILL_NOTEWORTHY_TOKENS:
            return ''
        cached = _tk.pending_cached()
        mark = '' if _tk.prefill_is_measured() else '~'
        hit = f", {format_tokens_compact(cached)} cached" if cached > 0 else ''
        return (f" {sym('dot')} {sym('up')} {mark}"
                f"{format_tokens_compact(p)} to prefill{hit}")

    def toolbar_text(self) -> str:
        """Status line for the interactive prompt's bottom_toolbar (the
        external_render counterpart of _render). Frame is derived from elapsed
        time, so no animation task is needed."""
        import time
        if not self.active or self.start_time is None:
            return ''
        elapsed = time.time() - self.start_time
        frame = self.frames[int(elapsed * 10) % len(self.frames)]
        return (f"{frame} {self._display_verb()}{sym('ellipsis')} ({elapsed:.1f}s {sym('dot')} "
                f"{sym('down')} {format_tokens_compact(self.token_count)} tokens"
                f"{self._thinking_segment()}{self._prefill_segment()})"
                f"{self._wait_segment()}")

    def _render(self):
        """Render spinner to terminal"""
        import time
        if self.external_render:
            return
        if not self.active or self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        frame = self.frames[self.frame_idx % len(self.frames)]

        # Build status line
        # Claude Code form: "✽ Cogitating… (16s · ↓ 759 tokens …)" — the ↓ marks
        # tokens streaming IN to the client (the response being received). plain
        # mode degrades ↓ to 'v' via sym().
        status = (f"{frame} {self._display_verb()}{sym('ellipsis')} ({elapsed:.1f}s "
                  f"{sym('dot')} {sym('down')} {format_tokens_compact(self.token_count)} tokens"
                  f"{self._thinking_segment()}{self._prefill_segment()})"
                f"{self._wait_segment()}")

        # Clip to the terminal, and measure in COLUMNS rather than characters.
        #
        # The repaint is \r, blanks, \r, which only reaches the line the cursor
        # is on. Let the status run past the window and the terminal wraps it;
        # \r then lands at the start of the LAST visual row and the rows above
        # it are never cleared, so every tick leaves another copy behind.
        # Reported 2026-09-13 from a narrowed window: eight stale 'Thinking…'
        # lines stacked up between two tool calls.
        #
        # Two separate mistakes, both fixed here. Clipping keeps the line on one
        # row so \r can always reach all of it. And the blank run is sized by
        # display_width, not len(): Hangul and the arrow/star glyphs in this
        # line occupy two columns each, so counting characters wiped about half
        # of what was actually drawn.
        _cols = terminal_columns()
        status = clip_to_width(status, _cols)
        clear = ' ' * min(self.last_line_length, _cols)
        sys.stdout.write(f"\r{clear}\r")
        sys.stdout.write(colorize(status, Colors.CYAN))
        sys.stdout.flush()

        self.last_line_length = display_width(status)

    def stop(self):
        """Stop spinner and clear the line"""
        if not self.active:
            return
        self.active = False

        # Cancel animation task
        if self._animation_task and not self._animation_task.done():
            self._animation_task.cancel()
        self._animation_task = None

        if self.external_render:
            return  # nothing on the terminal to clear

        # Clear the spinner line (columns, capped at the window — see repaint)
        clear = ' ' * min(self.last_line_length, terminal_columns())
        sys.stdout.write(f"\r{clear}\r")
        sys.stdout.flush()

    def get_final_stats(self) -> tuple:
        """Get final elapsed time and token count"""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            return elapsed, self.token_count
        return 0, 0


def _flush_stdin() -> None:
    """Flush any buffered stdin input (e.g. ESC sequences typed during processing)"""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (termios.error, ValueError, OSError):
        pass


def _set_terminal_echo(enabled: bool) -> None:
    """Enable or disable terminal echo"""
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        if enabled:
            attrs[3] |= termios.ECHO
        else:
            attrs[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    except (termios.error, ValueError, OSError):
        pass


def _start_escape_monitor_windows(abort_event: asyncio.Event) -> Optional[tuple]:
    """Windows ESC monitor. POSIX termios/tty don't exist here and
    loop.add_reader(stdin) is unsupported on the Proactor event loop, so the
    add_reader path is a no-op on Windows (that was the "ESC does nothing on
    Windows" bug). Instead poll msvcrt for a keypress in a daemon thread and set
    the abort Event from the loop thread via call_soon_threadsafe.

    Only meaningful on a real console; a redirected / non-TTY stdin has no
    keypress to read, so we no-op there — mirroring the POSIX branch, which
    likewise degrades to no interrupt when stdin isn't a TTY.
    """
    try:
        import msvcrt  # Windows-only
    except ImportError:
        return None
    try:
        if not sys.stdin.isatty():
            return None
    except (ValueError, OSError):
        return None

    import threading

    loop = asyncio.get_running_loop()
    stop = threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                # Arrow/function keys arrive as a lead char ('\x00' or '\xe0')
                # THEN a code char — they are NOT ESC-prefixed on Windows, so a
                # bare ESC ('\x1b') here is a genuine Esc press (no CSI/SS3
                # disambiguation needed, unlike the POSIX terminal path).
                if ch in ('\x00', '\xe0'):
                    if msvcrt.kbhit():
                        msvcrt.getwch()  # swallow the function-key code
                    continue
                if ch == '\x1b':
                    loop.call_soon_threadsafe(abort_event.set)
                    return
                # Any other key typed during processing is swallowed, exactly as
                # the POSIX monitor swallows stdin while in cbreak.
            else:
                stop.wait(0.03)

    t = threading.Thread(target=_poll, name='esc-monitor', daemon=True)
    t.start()
    return ('windows', stop, t)


def _start_escape_monitor(abort_event: asyncio.Event) -> Optional[tuple]:
    """Start monitoring stdin for ESC key.

    POSIX: event loop's add_reader on stdin in cbreak mode. Windows: msvcrt
    keypress polling in a daemon thread (add_reader/termios are POSIX-only) —
    see _start_escape_monitor_windows.

    Returns a cleanup handle (opaque tuple) for _stop_escape_monitor, or None on
    failure. Uses cbreak mode so output processing (OPOST) and signals (ISIG)
    stay active.
    """
    if os.name == 'nt':
        return _start_escape_monitor_windows(abort_event)
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (termios.error, ValueError, OSError):
        return None

    loop = asyncio.get_running_loop()

    def _on_stdin_readable():
        try:
            ch = os.read(fd, 1)
            if ch != b'\x1b':
                return
            # Arrow/function keys are ESC-PREFIXED SEQUENCES (up = ESC [ A),
            # so "first byte is ESC" alone would interrupt on any arrow press
            # during processing. Disambiguate the way terminal apps do: a real
            # ESC keypress has no follow-up bytes — peek briefly; if a CSI/SS3
            # tail ('[' or 'O') follows, it's a key sequence, not an interrupt.
            import select as _select
            r, _, _ = _select.select([fd], [], [], 0.05)
            if r:
                tail = os.read(fd, 32)
                if tail[:1] in (b'[', b'O'):
                    return  # swallow the sequence; not an interrupt
            abort_event.set()
            loop.remove_reader(fd)
        except OSError:
            loop.remove_reader(fd)

    # If add_reader fails (e.g. fd not selectable), restore the terminal before
    # returning — otherwise stdin is left in cbreak with no cleanup handle.
    try:
        tty.setcbreak(fd)
        loop.add_reader(fd, _on_stdin_readable)
    except Exception:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (termios.error, ValueError, OSError):
            pass
        return None
    return (fd, old_settings)


def _stop_escape_monitor(monitor_state: Optional[tuple]) -> None:
    """Stop escape monitor and restore terminal settings."""
    if not monitor_state:
        return
    # Windows msvcrt monitor: signal the poll thread to exit and join briefly.
    if monitor_state[0] == 'windows':
        _, stop, thread = monitor_state
        stop.set()
        try:
            thread.join(timeout=0.2)
        except RuntimeError:
            pass
        return
    fd, old_settings = monitor_state
    try:
        asyncio.get_running_loop().remove_reader(fd)
    except (RuntimeError, ValueError):
        pass
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (termios.error, ValueError, OSError):
        pass


# === Tool Display ===

def display_tool_use(api_tool_name: str, display_name: str, tool_input: Dict[str, Any], verbose: bool = False) -> None:
    """Print the tool-use header, then the tool's own rendering of its input.

    Dispatcher only — AssistantToolUseMessage.renderToolUseMessage:
    each tool owns its display via Tool.render_tool_use_message (tools/<Tool>/UI.tsx in
    the original). No per-tool switch here; a new tool renders correctly the day it
    lands, and no tool can spill a raw input dict on the terminal.
    """
    print(f"[{display_name}]")
    try:
        body = get_tool_by_name(api_tool_name).render_tool_use_message(tool_input, verbose=verbose)
    except ValueError:
        # Unknown tool (alias not in the registry) — never dump the dict.
        body = summarize_tool_input(tool_input)
    except Exception as e:
        # The original logs the render error and shows nothing, rather than
        # taking the REPL down over a display bug.
        debug_log(f"Error rendering tool use message for {api_tool_name}: {e}")
        body = ''
    # Sanitize before printing. The body is rendered from tool INPUT, which the
    # model writes, and this is drawn immediately above the Allow? prompt — the
    # one control standing between the model and execution. An unfiltered
    # ESC[2K\r in a command string erases the line the user just read and draws
    # a harmless-looking one over it, so the command approved is not the command
    # shown. sanitize_terminal_text was already written for untrusted text and
    # applied only to tool RESULTS; input is no less model-controlled.
    for line in sanitize_terminal_text(body or '').split('\n'):
        if line:
            safe_print(f"  {line}")


# Width of the argument echoed on a sub-agent's progress line.
_AGENT_PROGRESS_COLS = 80


# === Permission Handling ===

# 'a' (always) answers at the permission prompt land here — session-scoped,
# like Claude Code's "Yes, don't ask again for this session". Keys are per
# tool; Bash is keyed by the command's FIRST WORD ('ls', 'find', ...), and
# compound commands (;, &&, ||, |, substitution, newlines, redirection) never
# match — a blanket 'ls' allow must not wave through 'ls && rm -rf' NOR
# 'echo x > ~/.bashrc' (a redirect turns an allowed read-ish command into an
# arbitrary file write), so '>' and '<' force a fresh permission ask too.
_SESSION_TOOL_ALLOWS: set = set()
import re as _re  # (also imported below for the REPL; needed at this point)
_BASH_COMPOUND_RE = _re.compile(r'[;|&`\n<>]|\$\(')


# Fire-and-forget tasks must be kept referenced. asyncio holds only a WEAK ref to
# a bare create_task() result, so a task awaiting on input (the double-Esc message
# selector, the Ctrl+O pager) could be garbage-collected mid-await — its `finally:
# resume_prompt()` never runs and the input loop deadlocks in _park(). Track them
# in a strong set and drop each on completion.
_BACKGROUND_TASKS: set = set()


def _spawn_tracked(coro) -> "asyncio.Task":
    """create_task() that keeps a strong reference until the task completes."""
    task = asyncio.get_running_loop().create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


def _permission_key(tool_name: str, input_data: Any) -> Optional[tuple]:
    """Key for a session 'always' grant, or None to keep asking.

    A None here is the strong answer: it means no grant can ever cover this
    call. Only reached when NOT in dangerously-skip-permissions mode.
    """
    data = input_data if isinstance(input_data, dict) else {}

    # Command-executing tools: keyed on the first word, never on the tool alone.
    # PowerShell is included because keying it on the name blanket-allowed every
    # later command of any content.
    if tool_name in ('Bash', 'PowerShell'):
        cmd = (data.get('command') or '').strip()
        if not cmd or _BASH_COMPOUND_RE.search(cmd):
            return None   # compound/empty: always ask
        head = cmd.split()[0]
        # An interpreter turns one grant into unattended arbitrary code:
        # 'a' on `python build.py` would otherwise auto-approve
        # `python -c "<anything>"`, which contains no character the compound
        # guard looks for. DANGEROUS_BASH_PATTERNS was written for exactly this
        # and had no consumer; its module docstring claimed there was no
        # allow-rule engine to protect, but _SESSION_TOOL_ALLOWS is one.
        from .utils.permissions.dangerous_patterns import DANGEROUS_BASH_PATTERNS
        if head in DANGEROUS_BASH_PATTERNS or os.path.basename(head) in DANGEROUS_BASH_PATTERNS:
            return None
        return (tool_name, head)

    # File-mutating tools: keyed on the resolved path. The user approves a
    # specific file; keying on the tool name alone silently extended that to
    # every later path, including ~/.bashrc, .git/hooks/* and the config dir
    # (whose contents this program executes as shell — see _is_config_path).
    if tool_name in ('Write', 'Edit', 'NotebookEdit'):
        raw = data.get('file_path') or data.get('notebook_path') or ''
        if not raw:
            return None
        resolved = raw if os.path.isabs(raw) else os.path.join(get_cwd(), raw)
        return (tool_name, os.path.normpath(resolved))

    return (tool_name,)


def _is_config_path(path: str) -> bool:
    """True for anything under ~/.codemason.

    Files there are executed, not just read: settings.json's Stop hooks and
    goal-gates.txt both run through a shell with no permission check of their
    own, and mcp.json names a process spawned at startup. A write there is a
    code-execution request wearing a file-write prompt, so it is never covered
    by a session grant and is always shown for its own decision.
    """
    try:
        from .utils.config import CONFIG_DIR
        target = os.path.normpath(path if os.path.isabs(path) else os.path.join(get_cwd(), path))
        return os.path.commonpath([target, os.path.normpath(str(CONFIG_DIR))]) == os.path.normpath(str(CONFIG_DIR))
    except (ValueError, OSError, ImportError):
        return False   # different drive on Windows, or no config dir: not ours


async def _ask_permission_local_or_remote(router, tool, input_data, remote_ok: bool,
                                          prompt_text: str) -> str:
    """Put the question on the terminal and (if configured) on the phone at the
    same time; the first answer wins.

    Racing rather than assigning ownership is the point. Claude Code had to
    decide which attached client fields a prompt and shipped bugs doing it
    (2.1.208, 2.1.211); here neither side owns the decision, so nothing has to be
    handed over. A phone that is closed, offline or absent changes nothing — the
    terminal prompt is untouched and still the only path that has to work.

    A remote timeout is not an answer: the phone giving up drops us back to
    waiting on the terminal, exactly as if it had never been asked.
    """
    local = asyncio.ensure_future(router.request_permission(prompt_text))
    if not remote_ok:
        response = (await local).strip().lower()
        print(f"{prompt_text}{response}")
        return response

    from .remote import permissions as _remote_perm
    remote = asyncio.ensure_future(
        _remote_perm.ask(tool, input_data, cwd=os.getcwd()))
    try:
        done, _pending = await asyncio.wait(
            {local, remote}, return_when=asyncio.FIRST_COMPLETED)

        if local in done:
            response = (local.result() or '').strip().lower()
            print(f"{prompt_text}{response}")
            return response

        answer = remote.result()
        if answer:
            print(f"{prompt_text}{answer}  " + _("(remote)"))
            return answer

        # Phone timed out. Keep waiting on the terminal — never convert silence
        # into a decision.
        response = (await local).strip().lower()
        print(f"{prompt_text}{response}")
        return response
    finally:
        for task in (local, remote):
            if not task.done():
                task.cancel()


async def has_permissions_to_use_tool(
    tool: Type[Tool],
    input_data: Dict[str, Any],
    context: ToolUseContext,
    assistant_message: Any
) -> Dict[str, Any]:
    """
    Check if we have permission to use a tool.

    dangerously_skip_permissions allows EVERYTHING, with no exception: in that
    mode no permission prompt may ever appear (user instruction). The previous
    always_ask() carve-out meant ExitPlanMode still blocked an unattended run on
    a prompt nobody was there to answer, which is the opposite of what the flag
    is for. The only question the user should still see is the model's own
    AskUserQuestion, which is a deliberate tool call, not a permission gate.

    Everything below this early return is therefore non-dsp only, by
    construction — which is where the hardening belongs.
    """
    if context.options.get('dangerously_skip_permissions', False):
        return {'result': True}

    # Read-only tools are auto-approved — unless the tool itself asks to be
    # asked. The base check_permissions returns allow for a read-only tool, so
    # every existing one is unaffected; the exception is a tool whose "read"
    # reaches OUT of this machine.
    #
    # WebFetch and WebSearch change nothing locally, but the URL is
    # model-controlled and Read has no path restriction, so reading a private
    # key and posting it to an arbitrary host was two calls, neither of which
    # prompted. Under dangerously-skip-permissions nothing reaches this line at
    # all, which is the intended behaviour: skipping permissions means skipping
    # them.
    if tool.is_read_only():
        _pre = await tool.check_permissions(input_data, context)
        if getattr(_pre, 'behavior', 'allow') == 'allow':
            return {'result': True}
        if getattr(_pre, 'behavior', '') == 'deny':
            return {'result': False, 'message': getattr(_pre, 'message', '')}

    # Session 'always' grants ('a' at a previous ask). A write into the config
    # dir is never covered by one: those files are executed, so each is its own
    # decision no matter what was granted earlier.
    _pkey = _permission_key(tool.name, input_data)
    _cfg_target = ''
    if isinstance(input_data, dict) and tool.name in ('Write', 'Edit', 'NotebookEdit'):
        _cfg_target = input_data.get('file_path') or input_data.get('notebook_path') or ''
    if _cfg_target and _is_config_path(_cfg_target):
        _pkey = None
    elif _pkey is not None and _pkey in _SESSION_TOOL_ALLOWS:
        return {'result': True}

    # Display tool info
    tool_name = tool.user_facing_name(input_data)
    display_tool_use(tool.name, tool_name, input_data)

    router = getattr(context, 'input_router', None)
    from .remote import permissions as _remote_perm
    _remote_ok = _remote_perm.is_available()
    _PROMPT = '  ' + _('Allow? (y/n, a=always): ')

    while True:
        if router is not None:
            # Interactive mode: the persistent PromptSession owns stdin, so a
            # bare input() would fight it for bytes. Route the ask through the
            # session — its prompt message becomes the question and a single
            # y/n keypress answers it (other keys are swallowed; Esc/Ctrl+C
            # resolve to 'n'). Echo the answer so the ask stays in scrollback.
            # With a phone configured this races a remote prompt; see helper.
            response = await _ask_permission_local_or_remote(
                router, tool, input_data, _remote_ok, _PROMPT)
        else:
            # Headless (-p): there is no terminal to answer with. stdin is
            # normally closed, so input() takes EOF and the tool is denied —
            # which is exactly why sub-agent runs resort to
            # --dangerously-skip-permissions and end up bypassing everything.
            # Ask the phone first when one is configured; fall through to the
            # old behaviour when it is not, so nothing regresses without it.
            response = None
            if _remote_ok and not sys.stdin.isatty():
                response = await _remote_perm.ask(tool, input_data, cwd=os.getcwd())
                if response:
                    print(f"{_PROMPT}{response}  " + _("(remote)"))
            if response is None:
                _set_terminal_echo(True)
                _flush_stdin()
                try:
                    response = input(_PROMPT).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    # Ctrl+D / Ctrl+C at the permission prompt = decline, not a
                    # crash that aborts the whole query with an error.
                    return {'result': False, 'message': REJECT_MESSAGE}
                finally:
                    _set_terminal_echo(False)
        if response in ('y', 'yes'):
            return {'result': True}
        elif response in ('a', 'always'):
            if _pkey is not None:
                _SESSION_TOOL_ALLOWS.add(_pkey)
                _what = f"'{_pkey[1]}' Bash commands" if len(_pkey) > 1 else f"{_pkey[0]} calls"
                print(colorize(f"  (always allowing {_what} this session)", Colors.DIM))
            else:
                # Compound Bash command — no safe prefix to generalize.
                print(colorize("  (compound command — allowed once, will ask again)", Colors.DIM))
            return {'result': True}
        elif response in ('n', 'no'):
            return {'result': False, 'message': REJECT_MESSAGE}
        else:
            # Re-prompt on anything else instead of silently looping forever.
            print('  ' + _("Please answer 'y', 'n' or 'a'."))


# === REPL Loop ===

import re as _re
# The keep-going Stop hook asks the model to end a completed task with the exact
# line `TASK_COMPLETE` (a control sentinel, see ~/.codemason/hooks/keep_going.py).
# It belongs in history (the hook reads it) but must not be shown to the user.
# Strip a trailing standalone TASK_COMPLETE line from DISPLAY text only.
_SENTINEL_RE = _re.compile(r'\n?[ \t]*TASK_COMPLETE[ \t]*\Z')


def _strip_completion_sentinel(text: str) -> str:
    return _SENTINEL_RE.sub('', text).rstrip()


# REPL built-ins, slash-invoked (/help, /clear, ...).
_BUILTIN_COMMANDS = ('exit', 'quit', 'clear', 'compact', 'help', 'keepgoing', 'dsp',
                     'effort', 'reasoning', 'think', 'goal',
                     'resume', 'expand', 'export', 'language', 'lang', 'ab',
                     # Claude Code names this /remote-control (/rc); keep both so
                     # muscle memory from there works, with /remote as the short
                     # form that matches our other one-word commands.
                     'remote', 'remote-control', 'rc')


def _export_conversation_md(messages: List[Any]) -> str:
    """Render the in-memory conversation to Markdown (for /export).

    Items are UserMessage/AssistantMessage wrappers (`.type` + `.message.content`)
    — NOT dicts and with no to_dict — so read them by attribute, the same way the
    display loop does. Content blocks are TextBlock/ToolUseBlock/ToolResultBlock
    objects (attributes) or dicts; handle both."""
    def _at(o, name, default=None):
        if o is None:
            return default
        if isinstance(o, dict):
            return o.get(name, default)
        return getattr(o, name, default)

    def _blocks(content):
        if isinstance(content, str):
            return [{'type': 'text', 'text': content}]
        out = []
        for c in (content or []):
            out.append({
                'type': _at(c, 'type'),
                'text': _at(c, 'text', '') or '',
                'name': _at(c, 'name'),
                'input': _at(c, 'input', {}) or {},
                'content': _at(c, 'content', ''),
                'tool_use_id': _at(c, 'tool_use_id'),
            })
        return out

    lines = ['# CodeMason conversation export', '']
    for m in messages:
        role = _at(m, 'type') or _at(m, 'role') or '?'
        inner = _at(m, 'message')
        content = _at(inner, 'content', '') if inner is not None else _at(m, 'content', '')
        blocks = _blocks(content)
        only_results = bool(blocks) and all(b['type'] == 'tool_result' for b in blocks)
        if role == 'user' and not only_results:
            lines += ['## 🧑 User', '']
        elif role == 'assistant':
            lines += ['## 🤖 Assistant', '']
        for b in blocks:
            t = b['type']
            if t == 'text' and b['text'].strip():
                lines += [b['text'].rstrip(), '']
            elif t == 'tool_use':
                _inp = b['input']
                _arg = (_inp.get('command') if isinstance(_inp, dict) else None) \
                    or json.dumps(_inp, ensure_ascii=False, default=str)
                lines += [f"**⏺ {b['name'] or 'tool'}**", '', '```', str(_arg), '```', '']
            elif t == 'tool_result':
                res = b['content']
                if isinstance(res, list):
                    res = '\n'.join(_at(x, 'text', str(x)) if not isinstance(x, str) else x for x in res)
                res = str(res)
                if len(res) > 4000:
                    res = res[:4000] + f"\n… (truncated, {len(res)} chars total)"
                lines += ['  ⎿', '```', res.rstrip(), '```', '']
    return '\n'.join(lines).rstrip() + '\n'


# --- Full tool-output viewer (Ctrl+O / /expand) -----------------------------
# The display shows tool results truncated (5 lines) and very large outputs
# are persisted to disk with only a 2KB preview in context. This ring buffer
# keeps the FULL result text (and the persisted file path when one exists) so
# Ctrl+O can open the real thing in a pager without touching the scrollback.

_RECENT_TOOL_OUTPUTS: List[Dict[str, Any]] = []   # oldest → newest
_MAX_RECENT_TOOL_OUTPUTS = 20
# /expand prints inline into the scrollback — keep it small and capped so a
# huge persisted output can't flood the terminal (Ctrl+O's pager is uncapped).
# Per-entry cap is measured in SCREEN PAGES (user: "2~3 pages worth is fine").
_EXPAND_DEFAULT_COUNT = 3
_EXPAND_PAGES_PER_ENTRY = 3
_INLINE_EXPAND_BYTE_GUARD = 200 * 1024   # backstop for pathological single lines
_PERSISTED_PATH_RE = _re.compile(r'Full output saved to: (\S+)')


def _record_tool_output(tool_name: str, tool_input: Any, text: str) -> None:
    if not text:
        return
    m = _PERSISTED_PATH_RE.search(text)
    _RECENT_TOOL_OUTPUTS.append({
        'tool': tool_name or 'Tool',
        'input': tool_input,
        'text': text,
        'path': m.group(1) if m else None,
        'time': datetime.now().strftime('%H:%M:%S'),
    })
    del _RECENT_TOOL_OUTPUTS[:-_MAX_RECENT_TOOL_OUTPUTS]


def _tool_output_row(entry: Dict[str, Any]) -> str:
    """One picker row: time, tool, input preview, size."""
    inp = entry.get('input')
    if isinstance(inp, dict):
        preview = str(inp.get('command') or inp.get('file_path')
                      or inp.get('pattern') or inp)
    else:
        preview = str(inp)
    preview = ' '.join(preview.split())
    if len(preview) > 60:
        preview = preview[:60] + '…'
    size = len(entry.get('text') or '')
    return f"{entry['time']}  [{entry['tool']}] {preview}  ({size:,}B)"


def _entry_full_text(entry: Dict[str, Any]) -> str:
    """The untruncated output: the persisted original file when one exists
    (the in-context text is only a 2KB preview then), else the full result."""
    path = entry.get('path')
    if path and os.path.isfile(path):
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                return f.read()
        except OSError:
            pass
    return entry['text']


async def _run_output_pager(entries: List[Dict[str, Any]]) -> None:
    """Show tool outputs in full in `less -R` (scroll/search with '/', 'q'
    returns). One entry (via /expand) or the whole recent history (Ctrl+O) —
    multiple entries are concatenated newest-first with header rules, so
    scrolling down walks back in time. Falls back to plain printing when
    less is missing or in --plain mode (dumb consoles can't run a pager)."""
    import tempfile

    fd, tmp = tempfile.mkstemp(prefix='cm-output-', suffix='.txt')
    with os.fdopen(fd, 'w', encoding='utf-8', errors='replace') as f:
        # Key hint where it's needed — first line of the document.
        f.write("[q quit · / search · n/N match · Space/b page · g/G top/bottom]\n\n")
        for i, entry in enumerate(entries):
            if i:
                f.write('\n')
            f.write(f"{'=' * 8} {_tool_output_row(entry)} {'=' * 8}\n\n")
            f.write(_entry_full_text(entry))
            f.write('\n')
    path, cleanup = tmp, tmp
    try:
        if plain_mode():
            with open(path, encoding='utf-8', errors='replace') as f:
                print(f.read())
            return
        # `less` by whatever route it is available: on PATH, else the one
        # git-bash brought, else a copy beside the executable, else the
        # vendored tree (empty today). See utils/pager for the full order. Windows has no system less, and before this the same keystroke
        # paged when codemason was started from Git Bash and dumped every stored
        # tool output into the scrollback when it was started from PowerShell.
        from .utils.pager import resolve_pager
        less = resolve_pager()
        try:
            if less is None:
                raise FileNotFoundError
            proc = await asyncio.create_subprocess_exec(less, '-R', path)
            await proc.wait()
        except FileNotFoundError:
            with open(path, encoding='utf-8', errors='replace') as f:
                print(f.read())
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass


def _keep_going_enabled() -> bool:
    """Effective keep-going state — same precedence as the Stop hook script:
    CODEMASON_KEEP_GOING env (session override) > config.json keep_going > True."""
    env = os.environ.get('CODEMASON_KEEP_GOING')
    if env is not None:
        return env.strip().lower() not in ('0', 'false', 'no', 'off', '')
    return getattr(get_config(), 'keep_going', True)


def _observe_restored_turns(restored: List[Any]) -> None:
    """Replay a resumed transcript's typed turns as UI-language evidence.

    Only what the human wrote counts: tool results arrive as user-role messages
    too, and a compaction summary is model prose (usually English) that would
    argue for the wrong language. Text blocks of genuine user turns, nothing
    else. Best-effort — a transcript oddity must not break /resume.
    """
    from .utils.i18n import observe_user_text
    for msg in restored:
        try:
            if not isinstance(msg, dict) or msg.get('type') != 'user':
                continue
            if msg.get('isMeta') or msg.get('isCompactSummary'):
                continue
            content = (msg.get('message') or {}).get('content')
            if isinstance(content, str):
                observe_user_text(content)
            elif isinstance(content, list):
                if any(isinstance(b, dict) and b.get('type') == 'tool_result'
                       for b in content):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        observe_user_text(block.get('text') or '')
        except Exception:
            continue


def _print_resume_banner(messages: List[Any]) -> None:
    """Short orientation after a resume: what session this is and where it
    left off (subset of the original's full conversation re-render)."""
    from .utils.state import get_session_id
    from .utils.session_storage import extract_first_prompt

    n = len(messages)
    first = extract_first_prompt([m for m in messages if isinstance(m, dict)])
    print(f"Resumed session {get_session_id()} ({n} messages)")
    print(f"  First prompt: {first}")
    # Last assistant text = where the conversation left off.
    for m in reversed(messages):
        if (isinstance(m, dict) and m.get('type') == 'assistant'):
            from .utils.messages import get_text_from_message
            tail = _strip_completion_sentinel(get_text_from_message(m)).strip()
            if tail:
                if len(tail) > 500:
                    tail = tail[:500] + '…'
                print(colorize("  Last response:", Colors.DIM))
                for line in tail.split('\n')[:8]:
                    print(colorize(f"    {line}", Colors.DIM))
                break
    print()


async def repl_loop(
    tools: List[Type[Tool]],
    initial_prompt: str = '',
    dangerously_skip_permissions: bool = False,
    verbose: bool = False,
    debug: bool = False,
    initial_messages: Optional[List[Any]] = None,
):
    """Main REPL loop"""
    messages = list(initial_messages) if initial_messages else []
    system_prompt = await get_system_prompt(tools, get_config().model_name)

    # cwd and date are NOT seeded here: they are per-turn values and the system
    # prompt is the cached prefix (see process_prompt). The session's starting
    # directory is still stated once in <env>; the live one rides the turn.
    context = {}

    # Create tool use context
    from .state.app_state import get_app_state, set_app_state
    abort_controller = asyncio.Event()
    tool_use_context = ToolUseContext(
        abort_controller=abort_controller,
        options={
            'dangerously_skip_permissions': dangerously_skip_permissions,
            'tools': tools,
            'verbose': verbose,
            'commands': [],
        },
        message_id=None,
        read_file_state={},
        get_app_state=get_app_state,
        set_app_state=set_app_state,
    )

    from .utils.tokens import get_context_size, resolve_model
    from .services.compaction import get_auto_compact_threshold
    # Reconcile the configured model with what the server actually serves BEFORE
    # the first request, so a stale/renamed id (e.g. old 'ChatGPTN') self-heals
    # to the served model and is written back to config.json — instead of failing
    # late, in-stream, with "model does not exist".
    resolve_model()
    ctx_size = get_context_size()

    # Silence the kernel echo before the banner: prompt_toolkit does not own the
    # terminal yet, so anything typed during startup would be painted straight
    # across these lines and then queued into the first prompt. Restored on exit.
    from .utils.tty_compat import suppress_echo, restore_echo
    suppress_echo()
    atexit.register(restore_echo)

    print("-" * 63)
    _bid = build_id()
    print(f"{PRODUCT_NAME} v{VERSION}" + (f" ({_bid})" if _bid else ""))
    print(f"Working directory: {get_cwd()}")
    print(f"Context: {ctx_size:,} tokens (auto-compact at {get_auto_compact_threshold():,})")
    print(f"Tools available: {len(tools)}")
    # Surface the active model and reasoning effort so both are visible at a
    # glance (and changeable live with /effort). Empty effort = server default.
    _bcfg = get_config()
    _beff = (_bcfg.effective_reasoning_effort() or 'default')
    print(f"Model: {_bcfg.model_name or '(server default)'} | Reasoning effort: {_beff} (change: /effort)")
    # Sampling actually in effect. Printed only when the family branch diverges
    # from what config.json says, because that divergence is deliberate and must
    # not be silent — a value quietly different from the file on disk is the
    # exact failure that cost a day.
    _eff = _bcfg.effective_sampling()
    _manual = str(getattr(_bcfg, 'sampling_profile', '') or '').strip().lower() == 'manual'
    _parts = [f"temperature {_num(_eff['temperature'])}", f"top_p {_num(_eff['top_p'])}"]
    if _eff.get('top_k') is not None:
        _parts.append(f"top_k {_eff['top_k']}")
    from .utils.formatting import colorize as _c2, Colors as _C2
    # The VALUES are the point: they stop a silent divergence between the
    # sampling in effect and what config.json says. The paragraph that used to
    # follow them — which profile to set, which value the file holds, that the
    # family default is "recommended" — was scaffolding from the investigation
    # into the collapse and repetition symptoms. Those are fixed, so it explained
    # a problem that no longer happens, once per launch, in 156 columns against a
    # 63-column frame: 93 of them outside the rule it was printed under, which is
    # where a banner stops being read at all. How to change them belongs in
    # /help. (Ported from the Rust client, which shortened this in dfbd8f1.)
    #
    # The one thing kept is the disagreement itself, and only when there is one,
    # because that is the part no other line on screen can tell you.
    _note = ('' if (_manual or abs(float(_bcfg.temperature) - float(_eff['temperature'])) < 1e-9)
             else ' (config.json: %s)' % _num(_bcfg.temperature))
    print(_c2('Sampling: %s%s' % (', '.join(_parts), _note), _C2.DIM))

    # The served model's OWN tokenizer decides every token number in this
    # process: the auto-compact threshold, the context-window check, the usage
    # readout. Without it those come from a character heuristic — estimates,
    # not counts — and compaction can fire late enough to overrun the window.
    #
    # This used to be silent. The process fell back and kept reporting the
    # estimates as if exact, so the only way to notice was to ask. Say it, in
    # red, once per run.
    from .utils.tokens import has_model_tokenizer as _has_tok
    if not _has_tok():
        from .utils.formatting import colorize as _c, Colors as _C
        _redb = _C.BOLD + _C.RED
        _m = _bcfg.model_name or '(server default)'
        print()
        print(_c('!' * 63, _redb))
        print(_c('  WARNING: the model tokenizer was NOT found.', _redb))
        print(_c(f'  Token counts are ESTIMATES, not counts (model: {_m}).', _C.RED))
        print(_c('  Auto-compact and context-window checks may be wrong as a result.', _C.RED))
        print(_c('  Fix: ship tokenizer/<model>/tokenizer.json with the package', _C.RED))
        print(_c('        (re-run setup.sh / setup.bat from the repo).', _C.RED))
        print(_c('!' * 63, _redb))
        print()

    from .utils.goal import get_goal as _bgoal
    _bg = _bgoal()
    if _bg:
        print(f"Goal: {_bg}  (change: /goal)")
    print("Type /exit to quit, /clear to clear history, /help for commands")
    # keep-going status line: surface whether the model will be nudged to work
    # past natural stops, so the behavior is never a surprise. Toggle live with
    # `keepgoing on|off`, or launch with --keep-going / --no-keep-going.
    #
    # keep-going is ON by default and self-installing: when effectively on, make
    # sure the Stop hook is materialized on THIS machine (fresh checkout / new
    # box -> installed on first run). OFF leaves any existing hook in place; it
    # no-ops itself, so there's nothing to uninstall.
    from .utils.keep_going_installer import ensure_installed as _kg_ensure
    if _keep_going_enabled():
        _kg_actions = _kg_ensure()
        print(f"keep-going: on{'  (installed Stop hook)' if _kg_actions else ''} "
              f"(toggle: /keepgoing on|off)")
    elif has_hook_for_event('Stop'):
        print("keep-going: off (toggle: /keepgoing on|off)")
    print("-" * 63)

    if messages:
        _print_resume_banner(messages)

    if plain_mode():
        # --plain/--ascii (dumb/VT100/Hercules consoles): those terminals
        # can't do the cursor-repositioning escapes the persistent-prompt UI
        # needs (patch_stdout, bottom_toolbar), so they keep the sequential
        # prompt→turn→prompt loop unchanged.
        #
        # No input router here, and therefore no remote prompting — deliberately,
        # not as a gap left to fix. This mode exists for someone sitting at a
        # physical dumb console; a permission prompt or a question there is
        # answered by the person in front of it. Wiring the phone in would also
        # mean racing a blocking input() on a thread that cannot be cancelled,
        # which buys nothing for the one setup that has no phone in the picture.
        #
        # The channel is off wholesale here (remote.notify.off_reason), not just
        # unable to answer: half-off was worse than either, sending alerts and a
        # spinner to a phone whose taps nothing was listening for. Said out loud
        # at startup, because credentials that are configured and silently unused
        # look exactly like credentials that are broken.
        _warn_remote_off_in_plain()
        await _plain_repl(messages, system_prompt, context, tool_use_context,
                          verbose, debug, initial_prompt)
    else:
        await _interactive_repl(messages, system_prompt, context,
                                tool_use_context, verbose, debug,
                                initial_prompt)


_REWIND_MAX_TARGETS = 20  # cap the inline selector height (CC MAX_VISIBLE_MESSAGES)


def _enumerate_rewind_targets(messages: List[Any]) -> List[tuple]:
    """Visible user turns the conversation can rewind to (oldest→newest).

    Mirrors extract_first_prompt's skip rules — meta, compact-summary, and
    control-XML payloads are not user-selectable prompts, and a user turn that
    carries only a tool_result (no text) is not a rewind point. Returns
    (index_in_messages, preview_text, full_text) per selectable turn, keeping
    at most the most recent _REWIND_MAX_TARGETS so the inline picker stays a
    reasonable height on a long session.
    """
    import re as _re
    from .utils.session_storage import _SKIP_FIRST_PROMPT_PATTERN
    targets: List[tuple] = []
    for i, entry in enumerate(messages):
        if not isinstance(entry, dict):
            continue
        if entry.get('type') != 'user' or entry.get('isMeta') or entry.get('isCompactSummary'):
            continue
        content = (entry.get('message') or {}).get('content')
        texts: List[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text' and block.get('text'):
                    texts.append(block['text'])
        full: Optional[str] = None
        for text in texts:
            if not text:
                continue
            m = _re.match(r'<bash-input>(.*?)</bash-input>', text, _re.DOTALL)
            if m:
                full = f"! {m.group(1).strip()}"
                break
            if _SKIP_FIRST_PROMPT_PATTERN.match(text):
                continue
            full = text
            break
        if full is None:
            continue
        preview = full.replace('\n', ' ').strip()
        if len(preview) > 100:
            preview = preview[:100].strip() + '…'
        targets.append((i, preview, full))
    return targets[-_REWIND_MAX_TARGETS:]


async def _run_message_selector(router: 'ReplInputRouter') -> None:
    """Double-Esc-on-empty → pick a previous user message and rewind the
    conversation to it (CC's 'Restore conversation').

    Truncates the live message list in place to before the chosen turn — the
    same list object the turn loop holds, so the next query() sends only the
    rewound history — and drops that turn's text back into the input box for
    editing. Code/file restore is intentionally out of scope: codemason keeps
    no per-message file checkpoints.
    """
    messages = getattr(router, 'messages', None)
    if not messages:
        return
    targets = _enumerate_rewind_targets(messages)
    if not targets:
        return
    await router.pause_prompt()
    try:
        from .utils.arrow_picker import arrow_select
        options = [preview for (_i, preview, _f) in targets]
        picked = await arrow_select(
            'Rewind conversation to a previous message:', options)
        if not picked:
            return
        idx, preview, full = targets[picked[0]]
        # In-place truncation (mirrors /resume, /compact): drop the chosen user
        # turn and everything after it. Cutting at a user boundary never leaves
        # a dangling assistant tool_use without its tool_result.
        del messages[idx:]
        tuc = getattr(router, 'tool_use_context', None)
        rfs = getattr(tuc, 'read_file_state', None) if tuc is not None else None
        if isinstance(rfs, dict):
            rfs.clear()
        buf = getattr(router.session, 'default_buffer', None) if router.session else None
        if buf is not None:
            buf.text = full
            buf.cursor_position = len(full)
        print(f"↩ Rewound conversation to: {preview}")
    finally:
        router.resume_prompt()


# Segments in the input-line context gauge. Narrower than the 20-cell warning bar
# in get_token_warning: this one shares a line with what you are typing, and the
# percentage next to it is the number being read anyway — the bar is a glance, not
# a measurement.
_CTX_BAR_WIDTH = 8


def _make_prompt_kb(router: Optional['ReplInputRouter'] = None):
    """Key bindings + rprompt for the REPL prompt, shared by both input loops.

    Ctrl+Enter (this terminal sends c-j) inserts a newline; plain Enter
    submits. (No Alt+Enter binding on purpose: an ('escape','enter') sequence
    would make Esc a binding PREFIX, adding a 1s wait before every lone-Esc
    action — the user removed it for instant Esc.) Double-press Esc clears
    the current input, saving it to history first so Up-arrow recovers it —
    faithful to Claude Code (useTextInput.ts: useDoublePress).

    With a router (interactive mode), a single Esc while a turn is running
    interrupts it instead — prompt_toolkit parses key sequences properly, so
    this replaces the raw-stdin ESC monitor (and its arrow-key ambiguity)
    in that mode."""
    kb = KeyBindings()

    @kb.add('c-j')
    def _kb_insert_newline(event):
        event.current_buffer.insert_text('\n')

    @kb.add(Keys.BracketedPaste)
    def _kb_paste(event):
        # A terminal paste arrives as one BracketedPaste event with the whole
        # clipboard in event.data. If it's large/multi-line, stash the full text
        # and insert a compact [Pasted text #N +M lines] placeholder instead of
        # dumping hundreds of lines into the input box; it's expanded back to the
        # real content at submit (process_prompt) and viewable with Ctrl+O. Small
        # single-line pastes go in verbatim.
        # Normalise CR first: a clipboard from a Windows editor / RDP carries
        # CRLF, and prompt_toolkit inserts the CR verbatim rather than treating
        # it as a line break. It then shows as a literal ^M *inside* the text
        # ("API_KEY=x^MPROJECT=y"), silently welding two lines into one, and
        # that is what gets sent to the model.
        data = event.data.replace('\r\n', '\n').replace('\r', '\n')
        if router is not None and should_placeholder(data):
            pid = router.paste_store.add(data)
            ref = format_pasted_ref(pid, pasted_num_lines(data))
            event.current_buffer.insert_text(ref)
            # Record the full paste so Ctrl+O's viewer can expand it, the same
            # way a large tool output is viewable.
            _record_tool_output('Pasted text', ref, data)
        else:
            event.current_buffer.insert_text(data)

    esc_state = {'armed_at': 0.0, 'empty_armed_at': 0.0}

    # === Ctrl+X Ctrl+K: stop every background agent ===
    #
    # The deliberate gesture Esc deliberately is not — see
    # utils/kill_agents_gesture.py, which holds the two-press state machine and
    # is shared with Ctrl+C (the key most people will actually reach for). This
    # chord is the source's own, kept so muscle memory carries over.
    #
    # The gesture lives on the router when there is one, so both entry points
    # arm the same window: what is armed is the gesture, not the key.
    from .utils.kill_agents_gesture import KillAgentsGesture
    kill_gesture = getattr(router, 'kill_agents', None) or KillAgentsGesture()

    # Not while a permission ask is up. That mode deliberately swallows every
    # key but y/n/a/Esc so text meant for the next message cannot leak into the
    # answer, and a chord with something destructive behind it is the last thing
    # that should be an exception to it. The Rust engine ignores the chord there
    # for the same reason — it never reaches the chord state from the
    # permission branch.
    from prompt_toolkit.filters import Condition as _CondK

    @kb.add('c-x', 'c-k', filter=_CondK(
        lambda: not (router is not None and router.permission_pending)))
    def _kb_kill_agents(event):
        outcome, note = kill_gesture.press('Ctrl+X Ctrl+K')
        if outcome == 'idle':
            # Never silent: a keypress that does nothing reads as broken.
            print('(Ctrl+X Ctrl+K: no background agents running)')
        elif note:
            print(colorize('  ' + note,
                           Colors.DIM if outcome == 'armed' else Colors.CYAN))
        event.app.invalidate()

    # === Context gauge (right of the input line) ===
    #
    # How full the context window is, visible WHILE YOU TYPE. The per-response
    # footer reports each answer as it lands; this reports the state you are
    # about to add to, at the moment you are deciding how much to add — which is
    # when a long paste or a fifth "read the whole file" is still cancellable.
    #
    # The limit is resolved ONCE, here at setup, and never from inside rprompt():
    # get_auto_compact_threshold() -> get_context_size() can fire an HTTP probe at
    # the model server (5s timeout, twice, and a FAILED probe is deliberately not
    # cached), while rprompt() runs on every repaint — refresh_interval=0.1. From
    # the render path that would freeze the input line for seconds every tenth of
    # a second whenever the server is slow or down. At setup it costs nothing: the
    # startup banner already probed and cached the size a moment earlier.
    #
    # Measured against the AUTO-COMPACT THRESHOLD, matching the turn footer
    # (formatting.format_token_display) and the startup banner, so all three
    # agree. 100% here means "the history is about to be summarised away", which
    # is the number you can act on — not the raw window, which reads falsely
    # reassuring.
    _ctx_limit = 0
    if router is not None and not plain_mode():
        try:
            from .services.compaction import get_auto_compact_threshold
            _ctx_limit = get_auto_compact_threshold()
        except Exception:  # noqa: BLE001 - the input line must never fail to draw
            _ctx_limit = 0

    # token_count_with_estimation walks the whole history (and char-scans every
    # block after the last usage anchor), so it must not run per repaint. Messages
    # are appended whole and never mutated in place, so (length, identity of the
    # last one) changes exactly when the count can have changed: idle at the
    # prompt recomputes nothing, and a double-Esc rewind (which truncates the list
    # in place) invalidates on length.
    _ctx_cache: Dict[str, Any] = {'key': None, 'used': 0, 'images': ''}

    def _context_gauge():
        """'이미지 1/8 ███░░░░░ 38% · 88.4k/232.8k' — fragments, or [] when there's
        nothing to say. The image part (attached / server limit) appears only
        when the server takes images.

        Degrades by width instead of pushing the prompt around: the bar goes
        first, then the absolute counts, leaving the bare percentage on a narrow
        terminal. Hidden entirely before the first turn — the count is 0 until a
        response carries usage, and a gauge reading 0% would be a lie about a
        context that already holds the system prompt and every tool definition.
        """
        if _ctx_limit <= 0:
            return []
        messages = getattr(router, 'messages', None) or []
        if not messages:
            return []

        key = (len(messages), id(messages[-1]))
        if _ctx_cache['key'] != key:
            try:
                _ctx_cache['used'] = token_count_with_estimation(messages)
            except Exception:  # noqa: BLE001 - display must never break the prompt
                return []
            from .utils.session_status import image_text
            _ctx_cache['images'] = image_text(messages)
            _ctx_cache['key'] = key
        used = _ctx_cache['used']
        if used <= 0:
            return []

        from .utils.session_status import fill_percent
        pct = fill_percent(used, _ctx_limit)
        # Same escalation the footer warning uses, one step earlier: by the time
        # /compact is unavoidable it is too late to have shortened the question.
        style = 'ctx-crit' if pct >= 80 else 'ctx-warn' if pct >= 60 else 'ctx-ok'

        cols = 80
        from prompt_toolkit.application.current import get_app_or_none
        app = get_app_or_none()
        if app is not None:
            try:
                cols = app.output.get_size().columns
            except Exception:  # noqa: BLE001
                pass

        counts = (f"{format_tokens_compact(used)}/"
                  f"{format_tokens_compact(_ctx_limit)}")
        images = f"{_ctx_cache['images']} " if _ctx_cache['images'] else ''
        if cols >= 80:
            filled = min(_CTX_BAR_WIDTH, round(_CTX_BAR_WIDTH * pct / 100))
            bar = (sym('bar_full') * filled
                   + sym('bar_empty') * (_CTX_BAR_WIDTH - filled))
            text = f"{images}{bar} {pct}% {sym('dot')} {counts}"
        elif cols >= 60:
            text = f"{images}{pct}% {sym('dot')} {counts}"
        else:
            text = f"{pct}%"
        return [(f'class:{style}', f' {text} ')]

    def rprompt():
        now = time.monotonic()
        # The bulk-stop confirmation outranks the Esc hints: it is armed for 3s
        # against their 0.7, and it is the one with something destructive behind
        # it.
        _kill_hint = kill_gesture.hint(now)
        if _kill_hint:
            return _kill_hint
        if esc_state['armed_at'] and now - esc_state['armed_at'] < 0.7:
            return ' (Esc again to clear) '
        if esc_state['empty_armed_at'] and now - esc_state['empty_armed_at'] < 0.7:
            return ' (Esc again to rewind) '
        return _context_gauge()

    if router is not None:
        from prompt_toolkit.filters import Condition as _Cond

        # Unified Enter handler (idle + mid-turn). We take over accept so we can:
        #   * EXPAND [Pasted text #N] placeholders to the real content and echo
        #     the expanded (middle-elided) text to scrollback — the input box kept
        #     the compact placeholder, but the sent/echoed message is the full one;
        #   * while a turn runs, NOT commit the line to scrollback (which would
        #     tangle with live output) but park it in router.pending (shown in the
        #     rprompt) and queue it, echoed at the turn boundary.
        # Idle Enter with an empty buffer falls through to the default (no-op).
        _submit_enter = _Cond(
            lambda: not router.permission_pending and not router.paused
            and bool(getattr(router.session, 'default_buffer', None)
                     and router.session.default_buffer.text.strip()))

        @kb.add('enter', filter=_submit_enter)
        def _kb_submit(event):
            buf = event.current_buffer
            raw = buf.text
            buf.append_to_history()           # keep Up-arrow recall
            buf.reset()                        # we render the echo ourselves; no PT accept line
            # repair_surrogates after the expansion, not before: the Windows
            # console can split one emoji across two paste events, and only the
            # expanded text puts its two halves back next to each other.
            # Also normalise CR here, not only in the paste handler: a large
            # paste is stashed verbatim and only spliced back in at submit, so
            # its CRs would otherwise reach the model unfiltered.
            full = repair_surrogates(
                expand_pasted_refs(raw, router.paste_store)
                .replace('\r\n', '\n').replace('\r', '\n')).strip()
            if not full:
                event.app.invalidate()
                return
            router.conversation_started = True   # first input → show the input rule
            if full.lower() in ('/exit', '/quit'):
                print(colorize("  (exiting…)", Colors.DIM))
                router.interrupt()
                router.queue.put_nowait((full, True))
            elif router.processing:
                # mid-turn: INJECT into the running turn at its next model step
                # (steering), instead of parking it for a whole new turn once the
                # current one finishes. Still shown in the rprompt (pending) until
                # the turn's next gate drains it and echoes it to scrollback.
                router.pending.append(full)
                router.submit_steering(full)
            else:
                # idle: echo the expanded prompt now, queue without a re-echo.
                # A blank line first separates it from the previous turn's output
                # (the '⎯ ↓ … tokens' line would otherwise glue to '> …').
                print()
                print(_user_echo_line(full))
                router.queue.put_nowait((full, False))
            event.app.invalidate()

        # Empty/whitespace buffer while idle: swallow Enter so it is a TRUE no-op.
        # prompt_toolkit's default Enter is accept-line, which returns from
        # prompt_async() and makes _input_loop reprint '> ' on every blank Enter.
        # Bind it explicitly to do nothing (the "no-op" the submit filter intends).
        _idle_empty_enter = _Cond(
            lambda: not router.permission_pending and not router.paused
            and not (getattr(router.session, 'default_buffer', None)
                     and router.session.default_buffer.text.strip()))

        @kb.add('enter', filter=_idle_empty_enter)
        def _kb_enter_noop(event):
            event.current_buffer.reset()   # drop any whitespace-only content
            event.app.invalidate()

        # Esc during a running turn fires IMMEDIATELY: eager skips any
        # sequence-prefix wait, and the filter keeps idle-time Esc
        # (double-press clear) on the normal path.

        @kb.add('escape', eager=_Cond(
            lambda: router.processing or router.permission_pending))
        def _kb_escape_interrupt(event):
            router.interrupt()

    @kb.add('escape')
    def _kb_escape(event):
        if router is not None and router.interrupt():
            return  # Esc during a running turn = interrupt, not input-clear
        buf = event.current_buffer
        now = time.monotonic()
        if not buf.text:
            # Empty input: double-Esc opens the conversation-rewind message
            # selector (CC doublePressEscFromEmpty → onShowMessageSelector).
            # First Esc arms + hints via rprompt; a second Esc within the window
            # fires. A lone Esc never rewinds — matches CC's double-press gate.
            esc_state['armed_at'] = 0.0
            if router is None:
                return
            if esc_state['empty_armed_at'] and now - esc_state['empty_armed_at'] < 0.7:
                esc_state['empty_armed_at'] = 0.0
                _spawn_tracked(_run_message_selector(router))
            else:
                esc_state['empty_armed_at'] = now  # first press → arm + show hint
            event.app.invalidate()
            return
        # Non-empty input: a prior empty-arm no longer applies.
        esc_state['empty_armed_at'] = 0.0
        if esc_state['armed_at'] and now - esc_state['armed_at'] < 0.7:
            buf.append_to_history()   # recoverable via Up-arrow
            buf.reset()               # clear the input
            esc_state['armed_at'] = 0.0
        else:
            esc_state['armed_at'] = now  # first press → arm + show hint
        event.app.invalidate()

    if router is not None:
        # Permission-ask mode: answer with a SINGLE keypress and accept ONLY
        # y/n (+Esc=deny via the binding above). Everything else — including
        # Enter and ordinary typing — is swallowed, so text the user was
        # composing for the NEXT message can't leak into the ask. (Literal-key
        # bindings beat '<any>', so y/n/Esc still work.)
        from prompt_toolkit.filters import Condition
        _perm = Condition(lambda: router.permission_pending)

        @kb.add('<any>', filter=_perm)
        @kb.add('enter', filter=_perm)
        def _perm_swallow(event):
            pass

        @kb.add('y', filter=_perm)
        @kb.add('Y', filter=_perm)
        def _perm_yes(event):
            router.deliver('y')

        @kb.add('n', filter=_perm)
        @kb.add('N', filter=_perm)
        def _perm_no(event):
            router.deliver('n')

        @kb.add('a', filter=_perm)
        @kb.add('A', filter=_perm)
        def _perm_always(event):
            router.deliver('a')

        @kb.add('c-o')
        def _kb_open_output(event):
            # Ctrl+O: ALL recent tool outputs, full, in one pager document.
            # NOT an inline scrollback dump: heavy tool use produces
            # megabytes, which blows past terminal scrollback buffers —
            # less streams lazily and scales to any size. (Claude Code's
            # ctrl+o is also its own scrollable screen, not a dump; the
            # console-native equivalent of that mode IS the pager.)
            # Never SILENT: a keypress that does nothing reads as broken.
            if router.permission_pending or router.paused:
                return
            if not _RECENT_TOOL_OUTPUTS:
                print("(Ctrl+O: no tool output to show yet)")
                return
            # Newest first — the top of the document is the latest output;
            # scrolling down walks back in time, '/' searches everything.
            entries = list(reversed(_RECENT_TOOL_OUTPUTS))

            async def _show():
                # in_terminal ERASES the prompt, lends the terminal to less,
                # and redraws the prompt afterwards — no stale '> ' per use;
                # mid-turn prints (patch_stdout proxy) queue behind it and
                # flush above the prompt once the pager closes.
                from prompt_toolkit.application import in_terminal
                async with in_terminal():
                    await _run_output_pager(entries)

            _spawn_tracked(_show())

    return kb, rprompt


def _pending_preview_lines(pending: List[str], cols: int = 80) -> List[str]:
    """Lines for the pending-message preview shown just ABOVE the '> ' input
    while a turn runs — the messages you've typed and will send next, plainly
    (no queue/count framing). Each line is prefixed with a dim '⋯ ' and the whole
    block is capped to a few lines so a big paste can't push the prompt away."""
    if not pending:
        return []
    MAX_BODY = 4
    width = max(20, cols - 4)
    body = []
    for msg in pending:
        for ln in msg.split('\n'):
            ln = ln.rstrip()
            if len(ln) > width:
                ln = ln[:width - 1] + '…'
            body.append('⋯ ' + ln)
    if len(body) > MAX_BODY:
        omitted = len(body) - MAX_BODY
        body = body[:MAX_BODY] + [f'⋯ … (+{omitted} more)']
    return body


def _user_echo_line(text: str) -> str:
    """Build the scrollback echo of a submitted prompt: '> ' marker + the text.
    When a terminal theme is active, tint both with the theme's prompt/input
    colours using ABSOLUTE 256 (bypasses the palette so they don't collapse to
    the scheme's colour — e.g. all-green on homebrew). Otherwise keep the plain
    dim '> ' + default text."""
    from .utils.formatting import colorize, fg256_hex, hex_to_ansi256, Colors, supports_color
    from .utils.terminal_theme import active_theme_colors
    body = _format_user_echo(text)
    colors = active_theme_colors()
    if colors and supports_color():
        marker = f"{fg256_hex(colors['prompt'])}{Colors.BOLD}> {Colors.RESET}" \
            if colors.get('prompt') else colorize("> ", Colors.DIM)
        if colors.get('input'):
            body = f"{fg256_hex(colors['input'])}{body}{Colors.RESET}"
        line = marker + body
    else:
        line = colorize("> ", Colors.DIM) + body
    return _shade_user_echo(line)


def _shade_user_echo(line: str) -> str:
    """Lay a dim background behind the echoed prompt.

    The transcript is one column of text, and once it is long enough the eye has
    nothing to catch on: what you asked and what came back look the same. A
    background gives the boundary. It goes behind the *echo* — the line that
    stays in scrollback — not the live input area, which is gone the moment you
    press enter.

    Per line, and only behind the characters: padding to the terminal width would
    make the block copy out with trailing spaces, and a wrapped long prompt would
    tile unevenly. The reset is re-armed after every embedded reset in the line
    (the theme colours emit their own) so the background survives them.
    """
    from .utils.config import get_config
    from .utils.formatting import hex_to_ansi256, supports_color
    if not supports_color() or not getattr(get_config(), 'input_highlight', True):
        return line
    cfg = get_config()
    theme = str(getattr(cfg, 'theme', 'dark')).lower()
    bg_hex = getattr(cfg, 'input_highlight_bg', '') or (
        '#d0d0d0' if theme == 'light' else '#444444')
    fg_hex = getattr(cfg, 'input_highlight_fg', '') or (
        '#1c1c1c' if theme == 'light' else '#ffffff')
    bg = f'\033[48;5;{hex_to_ansi256(bg_hex)}m'
    fg = f'\033[38;5;{hex_to_ansi256(fg_hex)}m'
    on = bg + fg
    out = []
    for ln in line.split('\n'):
        # A reset inside the line would drop the background for the rest of it,
        # so every reset is followed by re-arming it.
        ln = ln.replace('\033[0m', '\033[0m' + on)
        out.append(on + ln + '\033[0m')
    return '\n'.join(out)


def _format_user_echo(text: str) -> str:
    """Scrollback echo for a submitted prompt. A pasted block is shown EXPANDED
    (the model gets the full text; the user should see what they sent) but with
    the middle elided when it's long, so a 500-line paste doesn't carpet the
    screen. Short single-line prompts pass through as-is."""
    lines = text.split('\n')
    HEAD, TAIL, MAX_LINES, MAX_CHARS = 12, 6, 22, 2400
    if len(lines) <= MAX_LINES and len(text) <= MAX_CHARS:
        return text
    kept = lines[:HEAD] + [f"  … ({len(lines) - HEAD - TAIL} lines elided) …"] + lines[-TAIL:]
    out = '\n'.join(kept)
    if len(out) > MAX_CHARS * 2:      # a few enormous lines, not many lines
        out = out[:MAX_CHARS] + f"\n  … ({len(text) - MAX_CHARS} chars elided) …"
    return out


class ReplInputRouter:
    """Coordinates the interactive REPL's persistent PromptSession (the single
    owner of stdin) with the turn loop: submitted-line queue, Esc/Ctrl+C
    interrupt routing, tool-permission prompts, and prompt pausing for sub-UIs
    that need the terminal (the resume picker's Application)."""

    _PAUSE = object()   # sentinel returned by a prompt exited for a pause
    _STEER_WAKE = object()   # queue marker: a line was enqueued mid-turn for
    #                          steering. Wakes the boundary loop to reconcile any
    #                          steering line the turn ended before it could inject
    #                          (see submit_steering / the turn-boundary handler).

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.pending: List[str] = []     # mid-turn typed lines waiting their turn —
        #                                  shown in the rprompt (router window), NOT
        #                                  committed to scrollback, until processed
        self.processing = False          # a turn is currently running
        self.turn_remote_origin = False  # the running turn was asked for from a
        #                                  remote client — its answer goes back
        #                                  there regardless of how quick it was
        self.remote_lines: Dict[str, str] = {}  # line -> echo tag, for input that
        #                                  arrived from a remote client (phone)
        #                                  rather than this keyboard, carrying how
        #                                  long it had been waiting
        self.session = None              # PromptSession, set by the REPL
        self.tool_use_context = None     # for the CURRENT abort_controller
        self.messages: List[Any] = []    # live conversation list (by reference);
        #                                  double-Esc rewind truncates it in place
        # False until the first line is submitted (and again after /clear):
        # the input rule is hidden then — at the fresh-start screen it would
        # sit right under the banner's separator as an ugly double bar.
        self.conversation_started = False
        self._permission = None          # (future, prompt_text) while asking
        self._paused = False
        self._idle_evt: Optional[asyncio.Event] = None
        self._resume_evt: Optional[asyncio.Event] = None
        # Large/multi-line pastes are replaced in the input box by a
        # [Pasted text #N] placeholder; the full text lives here and is spliced
        # back in at submit time (expand_pasted_refs). Ctrl+O shows the full text.
        self.paste_store = PasteStore()
        # The two-press bulk stop for background agents, shared by Ctrl+C (in
        # hard_stop below) and the Ctrl+X Ctrl+K chord in _make_prompt_kb. It
        # lives here because the router is what both of them have in common.
        from .utils.kill_agents_gesture import KillAgentsGesture
        self.kill_agents = KillAgentsGesture()

    # --- Esc / Ctrl+C while a turn runs ---
    def interrupt(self) -> bool:
        """Interrupt the running turn (or deny a pending permission ask).
        Returns False when idle so the caller can treat the key normally."""
        if self._permission is not None and not self._permission[0].done():
            self._permission[0].set_result('n')
            return True
        if self.processing and self.tool_use_context is not None:
            self.tool_use_context.abort_controller.set()
            return True
        return False

    def hard_stop(self) -> bool:
        """Ctrl+C: STOP — abort the turn AND drop everything queued behind it.

        Esc (interrupt) deliberately leaves the queue alone: aborting the turn so
        the line you typed mid-run fires immediately is the "interject now"
        workflow, and that stays. But it means Esc can never bring the session to
        rest — the backlog just becomes the next turn, and with several lines
        queued they chain. Ctrl+C is the brake that must always land, so it
        discards the backlog and returns to an idle prompt. The exit sentinel
        (None) is preserved: a queued Ctrl+D must still quit.

        Background agents take a second press — see the gesture at the end."""
        stopped = self.interrupt()
        dropped = 0
        saw_exit_sentinel = False
        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                saw_exit_sentinel = True
                continue
            if item is self._STEER_WAKE:
                # Internal wake marker, not a user message — its payload lives in
                # the steering queue (cleared below). Don't count it.
                continue
            dropped += 1
        if saw_exit_sentinel:
            self.queue.put_nowait(None)
        # Mid-turn steering backlog rides its own process-global queue, not
        # self.queue — discard it too, so Ctrl+C truly brings the session to rest.
        from .utils.steering_queue import drain_steering_messages
        steer_dropped = len(drain_steering_messages())
        if self.pending:
            dropped = max(dropped, len(self.pending), steer_dropped)
            self.pending.clear()
        elif steer_dropped:
            dropped = max(dropped, steer_dropped)
        if dropped:
            print(colorize(
                f"  (discarded {dropped} queued message"
                f"{'s' if dropped != 1 else ''})", Colors.DIM))
        if stopped or dropped > 0:
            # This press had work to do. Background agents are deliberately NOT
            # part of it: they are unlinked from the turn's abort scope on
            # purpose, and "killing agents from the main prompt stays a
            # deliberate gesture (chat:killAgents), not a side-effect of
            # cancelling a turn."
            return True

        # Nothing left to brake — and that is where the bulk stop lives, by the
        # same rule the source's cancel handler follows ("Priority 1: if there's
        # an active task running, cancel it first... Priority 2: pop queue when
        # Claude is idle"). Each press does the most urgent thing it can and
        # only falls through when there is nothing more urgent, which is what
        # makes mashing Ctrl+C to stop a turn unable to reach the agents: the
        # presses that brake never arm.
        #
        # Ctrl+C is the gesture's second entry point (the first is the source's
        # Ctrl+X Ctrl+K chord, which nobody will find in a REPL with no
        # shortcut footer) because it is the key someone reaches for when they
        # want everything to stop, and the alternative was closing the session.
        # It stays two presses: the first says what is still running, the second
        # ends it. This is also exactly where "(Use /exit or Ctrl+D to quit)"
        # used to be the only answer to a Ctrl+C that appeared to do nothing.
        outcome, note = self.kill_agents.press('Ctrl+C')
        if note:
            print(colorize('  ' + note,
                           Colors.DIM if outcome == 'armed' else Colors.CYAN))
        return outcome in ('armed', 'killed')

    # --- tool permission prompts (has_permissions_to_use_tool) ---
    @property
    def permission_pending(self) -> bool:
        return self._permission is not None

    @property
    def permission_prompt_text(self) -> str:
        return self._permission[1] if self._permission else '> '

    @staticmethod
    def _remote_tag(age: float, source: str = '') -> str:
        """'(remote)', or '(remote · 3h ago)' when the message waited around.

        Anything that sat long enough to matter says so. A message sent overnight
        and picked up when the session starts must not read like something just
        typed — the age is the only cue that its context may be gone.

        `source` names the transport the line came in on. Anything that is not
        the phone says so instead of borrowing its label: a line the local
        `inject` command queued shows as '(inject)', because presenting it as a
        phone message makes the user doubt a history they did not write."""
        what = 'remote' if source in ('', 'telegram') else source
        if age < 60:
            return f'  ({what})'
        if age < 3600:
            return f'  ({what} · {int(age // 60)}m ago)'
        if age < 86400:
            return f'  ({what} · {int(age // 3600)}h ago)'
        return f'  ({what} · {int(age // 86400)}d ago)'

    def submit_remote(self, text: str, age: float = 0.0, source: str = '') -> None:
        """Queue a line that arrived from a remote client (phone, later a PWA).

        Goes on the same queue as a typed line and follows the same rule: parked
        while a turn is running, taken at the next boundary. That is the queueing
        behaviour the REPL already had for mid-turn typing, so remote input needs
        no interrupt path of its own — a chat message is asynchronous by nature
        and this is exactly what the queue was for.

        Echoing follows the local rules exactly rather than adding its own. The
        queue flag means "echo this when it is processed", not "already echoed" —
        printing on arrival *and* passing it printed the line twice. So a message
        arriving mid-turn parks in `pending` (visible in the rprompt, like typing
        mid-turn does) and is echoed once at its boundary; an idle one is echoed
        straight away.

        The line is tagged so its echo can be marked `(remote)`. A line you did not
        type showing up in the transcript with no explanation is worse than a
        little noise — you have to be able to see what the session was told while
        you were away, and that it came from somewhere else.
        """
        text = (text or '').strip()
        if not text:
            return
        try:
            tag = self._remote_tag(age, source)
            self.remote_lines[text] = tag
            # A line from the phone starts the conversation just as much as a
            # typed one. Only the local Enter handler used to say so, so a
            # session driven entirely from the phone kept the fresh-start screen
            # and never drew the rule between output and input.
            self.conversation_started = True
            if self.processing:
                # Mid-turn: STEER it into the running turn at its next model step,
                # exactly like typing mid-turn locally does — a phone message is
                # asynchronous by nature and should influence the turn in flight,
                # not wait for it to end. Shown in the rprompt (pending) until the
                # turn's gate drains it; the remote tag rides in remote_lines and
                # is echoed by note_steer_injected / the boundary leftover path.
                self.pending.append(text)
                self.submit_steering(text)
            else:
                print()
                print(_user_echo_line(text), flush=True)
                print(colorize(tag, Colors.DIM), flush=True)
                self.queue.put_nowait((text, False))   # already echoed above
            app = getattr(self.session, 'app', None) if self.session else None
            if app is not None:
                app.invalidate()
        except Exception:
            pass

    async def request_permission(self, prompt_text: str) -> str:
        """Ask the user a question through the persistent prompt: the prompt
        message switches to `prompt_text` and a single y/n keypress (see
        _make_prompt_kb) answers it. Text the user was composing for the NEXT
        message is stashed on entry and restored afterwards — mid-turn typing
        must never leak into the ask."""
        fut = asyncio.get_running_loop().create_future()
        self._permission = (fut, prompt_text)
        buf = getattr(self.session, 'default_buffer', None) if self.session else None
        stash_text = buf.text if buf is not None else ''
        stash_pos = buf.cursor_position if buf is not None else 0
        if buf is not None:
            buf.reset()
        app = getattr(self.session, 'app', None) if self.session else None
        if app is not None:
            app.invalidate()   # repaint so the question shows immediately
        try:
            return await fut
        finally:
            self._permission = None
            if buf is not None:
                buf.text = stash_text
                buf.cursor_position = min(stash_pos, len(stash_text))
            if app is not None:
                app.invalidate()

    def deliver(self, line: str) -> bool:
        """Route a submitted line to a pending permission ask. True if consumed."""
        if self._permission is not None and not self._permission[0].done():
            self._permission[0].set_result(line)
            return True
        return False

    # --- mid-turn steering (interject into the RUNNING turn) ---
    def submit_steering(self, text: str) -> None:
        """A line typed while a turn is running. Instead of parking it for a
        whole new turn afterwards, enqueue it for injection into the current
        turn at its next model step (query()'s mid-turn gate drains it). The
        wake marker makes the boundary loop reconcile the case where the turn
        ends before it reaches another step — then there is nothing to inject
        into and the leftover is run as the next turn."""
        from .utils.steering_queue import enqueue_steering_message
        enqueue_steering_message(text)
        self.queue.put_nowait(self._STEER_WAKE)

    def note_steer_injected(self, text: str) -> None:
        """query() just injected a mid-turn steering line into the running turn.
        Drop it from the parked-input display (rprompt) and echo it to
        scrollback, so the transcript shows what steered the turn, and when,
        right where the model received it."""
        try:
            self.pending.remove(text)
        except ValueError:
            pass
        try:
            print()
            print(_user_echo_line(text))
            # If this steered line came from the phone, mark its origin (like the
            # queued path does) so the transcript shows it wasn't typed locally.
            _rtag = self.remote_lines.pop(text, None)
            if _rtag:
                print(colorize(_rtag, Colors.DIM))
        except Exception:
            pass
        app = getattr(self.session, 'app', None) if self.session else None
        if app is not None:
            try:
                app.invalidate()
            except Exception:
                pass

    # --- prompt pausing (the resume picker runs its own Application; two
    # prompt_toolkit Applications can't run at once) ---
    async def pause_prompt(self) -> None:
        self._paused = True
        self._idle_evt = asyncio.Event()
        self._resume_evt = asyncio.Event()
        app = getattr(self.session, 'app', None) if self.session else None
        if app is not None and app.is_running:
            app.exit(result=self._PAUSE)
        try:
            await self._idle_evt.wait()   # input loop confirms it parked
        except BaseException:
            # Undo the pause if we are cancelled while parked here. All three
            # callers put this await OUTSIDE the try whose finally calls
            # resume_prompt(), so a cancellation delivered in this window (ESC
            # force-cancelling the tool generator just as AskUserQuestion
            # starts) left _paused True with nobody left to set _resume_evt:
            # the input loop parks in _park() forever and the REPL silently
            # discards every keystroke for the rest of the process. Fixed here
            # rather than at each caller so the invariant cannot be missed by
            # the next one. CancelledError is a BaseException, hence the width.
            self.resume_prompt()
            raise

    def resume_prompt(self) -> None:
        self._paused = False
        if self._resume_evt is not None:
            self._resume_evt.set()

    @property
    def paused(self) -> bool:
        return self._paused

    def park_if_paused(self):
        """Awaitable used by the input loop at the top of each prompt cycle."""
        return self._park() if self._paused else None

    async def _park(self):
        if self._idle_evt is not None:
            self._idle_evt.set()
        if self._resume_evt is not None:
            await self._resume_evt.wait()


# constants/xml.ts LOCAL_COMMAND_CAVEAT_TAG + messages.ts:566-568, verbatim.
_LOCAL_COMMAND_CAVEAT = (
    "<local-command-caveat>Caveat: The messages below were generated by the "
    "user while running local commands. DO NOT respond to these messages or "
    "otherwise consider them in your response unless the user explicitly "
    "asks you to.</local-command-caveat>"
)


async def _process_bash_command(command: str, messages: List[Any],
                                tool_use_context: ToolUseContext) -> None:
    """`! cmd` bash passthrough — port of processUserInput/processBashCommand.

    Runs the command through BashTool (user-initiated: NO permission ask,
    same persistent shell/cwd/timeout/persist pipeline as model calls),
    prints the output, and appends <bash-input> + <bash-stdout>/<bash-stderr>
    user messages to the history — prefixed once per conversation with the
    local-command caveat — WITHOUT querying the model (shouldQuery:false).
    The model sees the result as context on the NEXT real prompt."""
    import html
    import uuid as _uuid
    from .tools.bash_tool import BashTool
    from .utils import session_storage
    from .utils.tool_result_storage import process_tool_result_content

    def _esc(s: str) -> str:
        return html.escape(s or '', quote=False)   # escapeXml: & < >

    new_msgs: List[Any] = []

    # Caveat once per conversation (createSyntheticUserCaveatMessage). The
    # scan is stateless, so /clear and resumed histories behave correctly.
    def _has_caveat() -> bool:
        for m in messages:
            if isinstance(m, dict) and m.get('type') == 'user':
                c = (m.get('message') or {}).get('content')
                if isinstance(c, list) and c and isinstance(c[0], dict) \
                        and '<local-command-caveat>' in (c[0].get('text') or ''):
                    return True
        return False

    if not _has_caveat():
        _cav = create_user_message(_LOCAL_COMMAND_CAVEAT)
        _cav['isMeta'] = True
        new_msgs.append(_cav)

    new_msgs.append(create_user_message(f"<bash-input>{command}</bash-input>"))

    # ESC-interruptible: mark processing so the interactive Esc binding fires;
    # BashTool threads context.abort_controller into the shell.
    router = tool_use_context.input_router
    if router is not None:
        router.processing = True
    try:
        data = None
        async for r in BashTool.call({'command': command}, tool_use_context):
            if getattr(r, 'type', '') == 'result':
                data = getattr(r, 'data', None)
        stdout = getattr(data, 'stdout', '') or ''
        stderr = getattr(data, 'stderr', '') or ''
        interrupted = bool(getattr(data, 'interrupted', False))
    except Exception as e:
        stdout, stderr, interrupted = '', '', False
        new_msgs.append(create_user_message(
            f"<bash-stderr>Command failed: {_esc(str(e))}</bash-stderr>"))
        print(colorize(f"Command failed: {e}", Colors.RED))
    else:
        if interrupted:
            from .utils.messages import INTERRUPT_MESSAGE
            new_msgs.append(create_user_message(INTERRUPT_MESSAGE))
            print(colorize("[Interrupted]", Colors.DIM))
        else:
            # Same persist mapping as model-initiated Bash: oversized stdout
            # becomes the trusted <persisted-output> preview (NOT re-escaped —
            # escaping would break the structural tags); small stdout passes
            # through as-is like the original's mapped.content string path.
            mapped = process_tool_result_content(
                stdout, 'Bash', str(_uuid.uuid4()),
                getattr(BashTool, 'max_result_size_chars', 100_000))
            new_msgs.append(create_user_message(
                f"<bash-stdout>{mapped}</bash-stdout>"
                f"<bash-stderr>{_esc(stderr)}</bash-stderr>"))
            if mapped and not is_tool_result_marker(mapped):
                print(mapped)
            if stderr:
                print(colorize(stderr, Colors.RED))
    finally:
        if router is not None:
            router.processing = False
        if tool_use_context.abort_controller.is_set():
            tool_use_context.abort_controller = asyncio.Event()
            if router is not None:
                router.tool_use_context = tool_use_context

    messages.extend(new_msgs)
    session_storage.record_transcript(messages)


def is_tool_result_marker(text: str) -> bool:
    """'(Bash completed with no output)' — the empty-result marker; useful in
    context, noise on the terminal."""
    return text.startswith('(') and text.endswith('completed with no output)')


def _print_remote_status() -> None:
    """`/remote` — is the phone connected, and what can it do.

    The feature had no way in from the terminal at all: no mention in help, no
    sign at startup, nothing to run. If it was not working you could not tell
    whether it was off, misconfigured, or broken — and if it *was* working you
    would not know what it could do. A capability nobody can find is not one.
    """
    from .remote import inbox as _inbox
    from .remote.permissions import is_available
    from .utils.config import get_config
    from .utils.state import get_session_id

    cfg = get_config()
    print()
    # Off wholesale (--plain) reads as "no credentials" otherwise, and the fix
    # for that — go set up a bot you already set up — is the wrong advice.
    from .remote import notify as _notify
    _off = _notify.off_reason()
    if _off:
        print(colorize(_('Remote channel is off  ({reason})', reason=_off), Colors.BOLD))
        print(colorize(
            _('  This mode is for dumb consoles, so the phone channel is unused.\n'
              '  Your settings are intact; run without --plain/--ascii to turn it back on.'),
            Colors.DIM))
        print()
        return
    if not is_available():
        # Asked where it is; answer with the file, not with a description of a
        # file. Creating it here too (not only at startup) covers the session
        # that was already running when the user went looking.
        from .remote.adapters.telegram import (
            ensure_credentials_scaffold as _scaffold, credentials_path as _cred_path)
        try:
            _scaffold()
        except Exception:
            pass
        print(colorize(_('No remote connection'), Colors.BOLD))
        print(colorize(
            _('  No Telegram bot token.\n'
              '  Fill in the slots in {path}:\n'
              '    TELEGRAM_BOT_TOKEN=...   (one line per bot; @BotFather issues them)\n'
              '    TELEGRAM_CHAT_ID=...     (your own numeric id; @userinfobot tells you)\n'
              '  (Environment variables work too. Restart the session afterwards.)',
              path=str(_cred_path())),
            Colors.DIM))
        print()
        return

    sessions = _inbox.live_sessions()
    me = get_session_id()
    mine = next((s for s in sessions if s['session_id'] == me), None)
    # Who actually holds the poll lock decides whether THIS session can be
    # reached — a detail that is invisible otherwise and explains the one
    # confusing case (several sessions, only one of them listening).
    # Asked for by name rather than rebuilt from a path literal here. The lock is
    # keyed by bot token now, so a hardcoded 'poll.lock' finds nothing, reads as
    # "somebody else is polling", and tells every session it is not listening.
    holder = ''
    try:
        from .remote.adapters.telegram import poll_lock_holder
        holder = poll_lock_holder()
    except Exception:
        pass
    listening = bool(holder) and holder == str(os.getpid())

    # Which bot this instance claimed, named as the phone shows it. With several
    # instances running, "the wrong window answered" is otherwise
    # indistinguishable from "the message never arrived".
    _bot = ''
    try:
        from .remote.adapters.telegram import describe_claimed_bot
        _bot = describe_claimed_bot()
    except Exception:
        pass

    print(colorize(_('Remote connected  (Telegram)'), Colors.BOLD))
    if _bot:
        print(_('  Bot          : {name}   (rename: /rc <name>)', name=_bot))
    print(_('  This session : {label}{state}',
            label=(mine.get('label') if mine else _('(unregistered)')),
            state=(_('  <- listening') if listening
                   else _('  (another session is listening)'))))
    if len(sessions) > 1:
        others = ', '.join(s.get('label', '?') for s in sessions
                           if s['session_id'] != me)
        print(_('  Others       : {names}', names=others))
    print()
    print(colorize(_('What you can do from the phone'), Colors.BOLD))
    print(_('  Send a message   queued for this session (mid-turn ones wait their turn)'))
    print(_('  Approve tools    risky tools raise buttons; first answer wins'))
    print(_('  Answer questions choices arrive as buttons (free text stays local)'))
    print(_('  /start           lists the running sessions'))
    print(_('  Reply            replying to an alert routes to that session'))
    print()
    print(colorize(_('Settings'), Colors.BOLD))
    _after = cfg.remote_notify_after
    print(_('  Notify after     only turns longer than {secs:g}s{every}   CODEMASON_NOTIFY_AFTER',
            secs=_after, every=(_(' (0 = every turn)') if _after == 0 else '')))
    print(_('  Approval wait    re-asks every {mins:g} min   CODEMASON_ASK_TIMEOUT',
            mins=cfg.remote_ask_timeout / 60))
    print(colorize(_('  A turn started from the phone always gets its answer.'),
                   Colors.DIM))
    print()


def _is_skill_command(name: str) -> bool:
    """True if `name` is a model-invocable skill — so `/name` is the skill
    shorthand, not an unknown command. Never raises."""
    try:
        from .utils.skills import get_skill
        return get_skill(name) is not None
    except Exception:  # noqa: BLE001
        return False


async def _dispatch_builtin(user_input: str, messages: List[Any],
                            tool_use_context: ToolUseContext,
                            remote: bool = False) -> Optional[str]:
    """Slash-command routing + built-in dispatch, shared by both REPL loops.
    Returns 'exit' (quit the REPL), 'handled' (command consumed), or None
    (ordinary prompt — send `user_input` to the model unchanged).

    Routing (handlePromptSubmit.ts): built-ins are invoked ONLY with a leading
    slash (/help, /clear, /exit, ...); bare command words are ordinary prompts
    for the model. (Upstream special-cases bare exit/quit — deliberately NOT
    ported: the user wants no bare-word shortcuts. Ctrl+D still quits.) Only
    the FIRST token counts, so mid-sentence '/help' and slash-leading paths
    ('/tmp/x.py please read this') go to the model untouched. An unknown single-word
    /command is rejected instead of being sent to the model (which
    hallucinated a plausible help page for /help). '/?' aliases /help; bare
    '/' is someone poking for the command list."""
    from .utils import session_storage

    # Local-operator affordances. `!` runs a shell command with NO permission
    # ask (it is user-initiated by definition) and `/dsp` changes the permission
    # mode itself and persists it. Neither is a decision the remote channel gets
    # to make: remote/permissions.py deliberately declines to offer even an
    # "always allow" button there, reasoning that a phone answer is a reflex tap
    # — and a remote `/dsp on` would hand out exactly the blanket permission
    # that module refused, for this session and every future one. Remote input
    # is still delivered to the model as an ordinary prompt; it just cannot
    # drive these two directly.
    if not remote:
        # Also detect it here rather than trusting every call site to pass it:
        # a remote line reaches the queue path as well as the steering path, and
        # its tag stays in remote_lines. Missing one call site would silently
        # reopen this, so the check lives with the rule it enforces.
        _rt = getattr(tool_use_context, 'input_router', None)
        if _rt is not None and user_input in getattr(_rt, 'remote_lines', {}):
            remote = True
    if remote and (user_input.startswith('!')
                   or user_input.split(maxsplit=1)[:1] in (['/dsp'], ['/DSP'])):
        print('  Remote input cannot run ! commands or change permission mode. '
              'Send it as a normal message instead.')
        return 'handled'

    # `! cmd` = bash passthrough (processUserInput mode === 'bash').
    if user_input.startswith('!'):
        _cmd = user_input[1:].strip()
        if not _cmd:
            print("Usage: ! <command>   (runs in the session shell; the model "
                  "sees the output on your next prompt)")
            return 'handled'
        await _process_bash_command(_cmd, messages, tool_use_context)
        return 'handled'

    if user_input == '/?':
        user_input = '/help'
    elif user_input == '/':
        print_help(tool_use_context.options.get('tools'))
        return 'handled'

    _first = user_input.split(maxsplit=1)[0]
    if not _re.fullmatch(r'/[A-Za-z][A-Za-z0-9_-]*', _first):
        return None
    if _first[1:].lower() not in _BUILTIN_COMMANDS:
        # Not a built-in: if it names a skill, it's the `/skill-name` shorthand —
        # send it to the model unchanged so it invokes the Skill tool (per the
        # system prompt). Only genuinely unknown /commands are rejected.
        if _is_skill_command(_first[1:]):
            return None
        print(f"Unknown command: {_first} (see /help)")
        return 'handled'
    # Drop the leading '/', not the first character. `_first` comes from
    # split(), so leading whitespace still reaches here: "  /help" passed the
    # check above and then sliced the space off instead of the slash, leaving
    # cmd = "/help", which matches no branch — the command silently did nothing.
    user_input = user_input.lstrip()[1:]
    cmd = user_input.lower().split()[0]

    if cmd in ('exit', 'quit'):
        return 'exit'

    if cmd == 'clear':
        messages.clear()
        # The repeated-call guard counts within a conversation; a cleared one
        # starts over, so a call blocked before must be allowed again.
        from .query import reset_tool_repeat_history
        reset_tool_repeat_history()
        # /clear starts a NEW session (regenerateSessionId +
        # resetSessionFilePointer): the old transcript stays resumable,
        # the next message opens a fresh file.
        session_storage.switch_session()
        # Record a clear boundary so a later `--continue` (-c) doesn't resume the
        # session we just cleared — the empty new session isn't on disk yet, so
        # without this -c would silently reload the pre-clear conversation.
        session_storage.record_clear_boundary()
        if tool_use_context.input_router is not None:
            # Back to the fresh-start look: no input rule until the next input.
            tool_use_context.input_router.conversation_started = False
        print("Conversation cleared.")
        return 'handled'

    if cmd == 'resume':
        from .utils.resume_picker import pick_session
        # The picker is its own prompt_toolkit Application — in interactive
        # mode the persistent prompt must be parked first (one Application
        # per terminal at a time).
        router = tool_use_context.input_router
        if router is not None:
            await router.pause_prompt()
        try:
            try:
                picked = await pick_session()
            except Exception as _e:
                print(f"Resume picker failed: {_e}")
                return 'handled'
            if not picked:
                return 'handled'
            try:
                _restored = session_storage.load_conversation_for_resume(
                    picked['session_id'])
                session_storage.adopt_resumed_session(picked['session_id'])
            except (FileNotFoundError, ValueError) as _e:
                print(f"Could not resume: {_e}")
                return 'handled'
            messages.clear()
            messages.extend(_restored)
            # A resume is a NEW process, so the UI-language evidence count starts
            # empty while the answer is sitting right there in the transcript.
            # Replay it before the banner prints, or the first screen of a
            # resumed Korean session comes back in English.
            _observe_restored_turns(_restored)
            _rfs = getattr(tool_use_context, 'read_file_state', None)
            if isinstance(_rfs, dict):
                _rfs.clear()   # restored history isn't this process's reads
            _print_resume_banner(messages)
            return 'handled'
        finally:
            if router is not None:
                router.resume_prompt()

    if cmd == 'ab':
        from .services import ab as _ab
        _parts = user_input.split(maxsplit=1)
        _want = _parts[1].strip() if len(_parts) > 1 else ''
        if not _want:
            _served = await _ab.query()
            if _served is None:
                _line = 'server: no /ab endpoint'
            else:
                _scale, _consistent = _served
                _state = 'on (abliterated)' if _scale > 0 else 'off (restored)'
                _line = f'server: {_state}' + ('' if _consistent else ' — WORKERS DISAGREE')
            print(
                f"/ab mode: {_ab.mode()}\n{_line}\n\n"
                "off   leave the server alone\n"
                "on    pin the refusal edit on — for measurement\n"
                "auto  on for the turn's first request, off as soon as it answers (default)\n\n"
                "The edit also costs the model its ability to say something does not exist\n"
                "(4/60 fabrications off, 18-22/60 on), so auto keeps it off for every tool\n"
                "result, link and diff — only the deciding turn gets it."
            )
            return 'handled'
        _mode = _ab.parse_mode(_want)
        if _mode is None:
            print(f"Unknown mode: {_want}   (off | on | auto)")
            return 'handled'
        _err = await _ab.set_mode(_mode)
        print(f"/ab mode: {_mode}" + (f" — server not updated: {_err}" if _err else ""))
        return 'handled'

    if cmd in ('language', 'lang'):
        # Saying it out loud outranks everything the resolver can infer, and it
        # is saved: the usual reason to reach for this command is a machine that
        # has no config.json yet, and having to retype it every session would
        # make it useless there.
        from .utils.i18n import LANGUAGES, normalize, set_language, active
        _parts = user_input.split(maxsplit=1)
        _want = _parts[1].strip() if len(_parts) > 1 else ''
        _names = ', '.join(f'{name} ({code})' for code, (name, _a) in LANGUAGES.items())
        if not _want:
            _now = active()
            _label = LANGUAGES.get(_now, ('English',))[0] if _now else 'English'
            print(f"UI language: {_label}\nAvailable: {_names}\n"
                  f"Usage: /language <name>   (Korean, ko, ko_KR, 한국어 — all work)")
            return 'handled'
        _code = normalize(_want)
        if _code is None:
            print(f"Unknown language: {_want}\nAvailable: {_names}")
            return 'handled'
        set_language(_code)
        try:
            from .utils.config import get_config, save_config
            _cfg = get_config()
            setattr(_cfg, 'ui_language', _code)
            from .utils.config import saved_label as _sl
            _saved = '' if save_config(_cfg) else f"  ({_sl(False)})"
        except Exception as _e:
            _saved = f"  (this session only — could not save: {_e})"
        print(f"UI language: {LANGUAGES[_code][0]}{_saved}")
        return 'handled'

    if cmd == 'compact':
        # Manual compaction. Optional text after `/compact` is passed as
        # custom instructions.
        _parts = user_input.split(maxsplit=1)
        _instr = _parts[1] if len(_parts) > 1 else None
        if not messages:
            print("Nothing to compact.")
            return 'handled'
        _spin = StreamingSpinner()
        try:
            from .services.compaction import compact_conversation
            _spin.start(verb='Compacting')
            _res = await compact_conversation(
                messages, tool_use_context,
                custom_instructions=_instr, is_auto_compact=False,
                on_token_update=_spin.update,
                on_thinking_update=_spin.update_thinking,
            )
            _spin.stop()
            messages.clear()
            messages.extend(_res.messages)
            _rfs = getattr(tool_use_context, 'read_file_state', None)
            if isinstance(_rfs, dict):
                _rfs.clear()   # summary lacks full file contents -> force re-read
            print("  ⎿  Compacted — conversation summarized.\n")
        except Exception as _e:
            _spin.stop()
            print(f"Compaction failed: {_e}")
        return 'handled'

    if cmd == 'expand':
        # Print the LAST FEW tool outputs in full, inline into the chat
        # scrollback (user's pick: small and right there in the flow —
        # Ctrl+O's pager is the scalable view for the whole history).
        # Bounded: a few entries, each capped, so it can't flood the
        # terminal the way an unbounded dump would. NOT an arrow-key
        # picker: a second prompt_toolkit Application fought the
        # persistent prompt when /expand arrived through the mid-turn
        # queue (typed-ahead keys landed in the picker, Enter silently
        # selected, renders piled up).
        if not _RECENT_TOOL_OUTPUTS:
            print("No tool output to expand yet.")
            return 'handled'
        _parts = user_input.split()
        _n = _EXPAND_DEFAULT_COUNT
        if len(_parts) > 1:
            try:
                _n = int(_parts[1])
            except ValueError:
                print("Usage: /expand [N]   (print the last N tool outputs; "
                      f"default {_EXPAND_DEFAULT_COUNT})")
                return 'handled'
            _n = max(1, _n)
        entries = _RECENT_TOOL_OUTPUTS[-_n:]   # chronological, newest last
        import shutil as _shutil
        _rows = _shutil.get_terminal_size((80, 24)).lines
        _max_lines = max(20, _rows * _EXPAND_PAGES_PER_ENTRY)
        print()
        for entry in entries:
            print(colorize(f"--- {_tool_output_row(entry)}", Colors.CYAN))
            text = _entry_full_text(entry)
            if len(text) > _INLINE_EXPAND_BYTE_GUARD:
                text = text[:_INLINE_EXPAND_BYTE_GUARD]
            _tlines = text.split('\n')
            if len(_tlines) > _max_lines:
                _cut = len(_tlines) - _max_lines
                text = '\n'.join(_tlines[:_max_lines])
                text += colorize(
                    f"\n... ({_cut} more lines - Ctrl+O shows the full output)",
                    Colors.DIM)
            print(text)
            print()
        return 'handled'

    if cmd == 'export':
        # /export [path]  — write the conversation to a Markdown file.
        # Default: ./codemason-export-<sessionid>.md in the current directory.
        if not messages:
            print("Nothing to export yet.")
            return 'handled'
        _parts = user_input.split(maxsplit=1)
        if len(_parts) > 1 and _parts[1].strip():
            _path = os.path.expanduser(_parts[1].strip())
        else:
            try:
                from .utils.state import get_session_id as _gsid
                _sid = _gsid()[:8]
            except Exception:
                _sid = 'session'
            _path = os.path.join(os.getcwd(), f"codemason-export-{_sid}.md")
        try:
            _md = _export_conversation_md(messages)
            with open(_path, 'w', encoding='utf-8') as _f:
                _f.write(_md)
            print(f"  ⎿  Exported {len(messages)} messages → {_path}")
        except OSError as _e:
            print(f"Export failed: {_e}")
        return 'handled'

    if cmd == 'help':
        print_help(tool_use_context.options.get('tools'))
        return 'handled'

    if cmd in ('remote', 'remote-control', 'rc'):
        # No argument keeps the old behaviour and prints status. on|off switches
        # the whole channel: one gate (off_reason) stops alerts, the spinner,
        # approvals, questions and inbound at once, so there is no half-off state
        # to end up in.
        _parts = user_input.split()
        _arg = _parts[1].lower() if len(_parts) > 1 else ''
        if _arg in ('on', 'off'):
            from .utils.config import get_config as _gc, save_config as _sc
            from .remote import notify as _notify
            if _arg == 'on':
                # Refuse without credentials. "Enabled" while nothing can be
                # delivered is the one state that lies to you — unconfigured and
                # switched off are different things and should read differently.
                if not any(a.is_configured() for a in _notify.all_adapters()):
                    from .remote.adapters.telegram import (
                        ensure_credentials_scaffold as _scaffold,
                        credentials_path as _cred_path)
                    try:
                        _scaffold()
                    except Exception:
                        pass
                    print(_('Cannot turn the remote channel on — no credentials.'))
                    print(_('  Fill in the slots in {path} and try again:',
                            path=str(_cred_path())))
                    print('    TELEGRAM_BOT_TOKEN=...')
                    print('    TELEGRAM_CHAT_ID=...')
                    return 'handled'
                import os as _os
                # If --no-remote switched this run off, lift that first.
                if _os.getenv('CODEMASON_REMOTE', '').strip().lower() in ('0', 'off', 'false', 'no'):
                    _os.environ['CODEMASON_REMOTE'] = '1'
            from .utils.config import saved_label as _sl
            _cfg = _gc(); _cfg.remote_enabled = (_arg == 'on'); _saved = _sl(_sc())
            # Outbound is decided per call, so the gate covers it. Inbound is a
            # poller thread already in its loop, which no setting reaches — it has
            # to be detached and reattached by hand.
            from .remote import input as _rinput
            print(f'  {_rinput.set_enabled(_arg == "on")}')
            _why = _notify.off_reason()
            if _arg == 'on' and _why:
                # Turned on in the config, still off for another reason (--plain).
                print(_('Remote channel on ({saved}) — but still off right now because of {reason}.', saved=_saved,
                        reason=_why))
            elif _arg == 'on':
                print(_('Remote channel on ({saved}). Alerts, approvals and questions go to the phone.', saved=_saved))
            else:
                print(_('Remote channel off ({saved}). Alerts, spinner, approvals, questions and '
                        'inbound messages all stop.', saved=_saved))
            return 'handled'
        if _arg == 'clean':
            # Explicit removal, ignoring the retention window. Automatic sweeping
            # is a backstop for crashed runs and keeps things for a month, on the
            # view that deleting the user's own words on a timer is not
            # housekeeping. Saying so out loud is a different matter.
            from .remote import inbox as _inbox
            _n = _inbox.sweep(force=True)
            print(_('Removed {count} leftover file(s).', count=_n))
            print(colorize(
                _('  Live sessions and anything still queued for them were kept.'),
                Colors.DIM))
            return 'handled'
        if _arg:
            # Anything else is a NAME for this session's bot, the way /rc <name>
            # names a remote session elsewhere. The name is only a label —
            # nothing is wired to it, the token is — so this renames the bot in
            # the phone's chat list and changes nothing about how it works.
            _name = user_input.split(maxsplit=1)[1].strip()
            from .remote.adapters import telegram as _tg
            if not _tg.claimed_token():
                print(_('No bot is claimed for this session — remote is off.'))
                return 'handled'
            _renamed, _why = _tg.set_bot_name(_name)
            if _renamed:
                print(_('Bot renamed to {name}. Its chat in Telegram is now this '
                        'session.', name=_name))
            else:
                # Say what Telegram actually said. The first version guessed
                # ("0-64 characters, some titles are rejected") and the real
                # answer was a rename cooldown measured in hours — so the guess
                # sent you looking for a bad character that was never there.
                print(_('Could not rename the bot: {why}', why=_why))
                print(colorize(_('  The session is unaffected — the name is only a '
                                 'label in the chat list.'), Colors.DIM))
            return 'handled'
        _print_remote_status()
        return 'handled'

    if cmd == 'dsp':
        # Approvals, switched mid-session. A start-up flag is the wrong shape:
        # the risky task shows up in the middle of a conversation, and what you
        # want then is a switch, not a restart. The permission gate re-reads
        # context.options on every tool call, so flipping it here takes effect on
        # the very next one; it is saved too, so the choice outlives the session.
        _parts = user_input.split()
        _arg = _parts[1].lower() if len(_parts) > 1 else 'status'
        if _arg in ('on', 'off'):
            _val = (_arg == 'on')
            tool_use_context.options['dangerously_skip_permissions'] = _val
            from .utils.config import (get_config as _gc, save_config as _sc,
                                       clear_runtime_override as _clear)
            # Deliberate, and the message says it was saved — so drop any shadow a
            # startup flag left on this field, or the save would write the flag's
            # value straight back over the choice just made.
            _clear('dangerously_skip_permissions')
            from .utils.config import saved_label as _sl
            _cfg = _gc(); _cfg.dangerously_skip_permissions = _val; _saved = _sl(_sc())
            print(_('Skip approvals {state} ({saved}). ', state=('on' if _val else 'off'), saved=_saved)
                  + (_('Tools run without asking.') if _val
                     else _('Every tool call asks for approval.'))
                  + ' ' + _('Plan approval (ExitPlanMode) and questions (AskUserQuestion) '
                            'still ask either way.'))
        elif _arg == 'status':
            _eff = tool_use_context.options.get('dangerously_skip_permissions', False)
            print(_('Skip approvals is {state}.', state=('on' if _eff else 'off'))
                  + (_('  Tools are auto-approved.') if _eff
                     else _('  Every tool call asks.')))
        else:
            print("Usage: /dsp on | off | status")
        return 'handled'

    if cmd == 'goal':
        # Persistent objective, injected into the system prompt every turn (see
        # process_prompt) so the agent keeps it in view across turns. Optional
        # completion gates (shell commands that must pass) make "done" verifiable.
        from .utils import goal as _goalmod
        _rest = user_input[len('goal'):].strip()
        _sub = _rest.lower()

        # --- gate management: /goal gate add|list|rm|clear ---
        if _sub == 'gate' or _sub.startswith('gate '):
            _gargs = _rest[len('gate'):].strip()
            _gsub = _gargs.split(maxsplit=1)
            _gcmd = _gsub[0].lower() if _gsub else 'list'
            _gval = _gsub[1] if len(_gsub) > 1 else ''
            if _gcmd == 'add' and _gval:
                _goalmod.add_gate(_gval)
                print(_('Gate added: {cmd}', cmd=_gval))
            elif _gcmd in ('list', 'ls', ''):
                _gates = _goalmod.get_gates()
                if _gates:
                    print(_('Completion gates:'))
                    for _i, _g in enumerate(_gates, 1):
                        print(f"  {_i}. $ {_g}")
                else:
                    print(_('No gates. Add one with: /goal gate add <command>'))
            elif _gcmd in ('rm', 'remove', 'del') and _gval.strip().isdigit():
                print(_('Gate removed.') if _goalmod.remove_gate(int(_gval.strip()))
                      else _('No gate at that number.'))
            elif _gcmd == 'clear':
                _goalmod.clear_gates()
                print(_('Gates cleared.'))
            elif _gcmd == 'auto':
                from .utils.config import (get_config as _gc, save_config as _sc,
                                           clear_runtime_override as _clr)
                _av = _gval.strip().lower()
                if _av in ('on', 'off'):
                    _clr('goal_gate_autorun')
                    from .utils.config import saved_label as _sl
                    _c = _gc(); _c.goal_gate_autorun = (_av == 'on'); _saved = _sl(_sc())
                    print(_('Gate auto-run {state} ({saved}).',
                            state=('on' if _av == 'on' else 'off'), saved=_saved)
                          + (_('  Gates run when a turn ends; a failure keeps the agent working.')
                             if _av == 'on'
                             else _('  Gates only run on /goal check.')))
                else:
                    _cur = getattr(_gc(), 'goal_gate_autorun', True)
                    print(_('Gate auto-run is {state}.', state=('on' if _cur else 'off')))
            else:
                print("Usage: /goal gate add <command> | list | rm <n> | clear | auto on|off")
            return 'handled'

        # --- run the gates now: /goal check ---
        if _sub in ('check', 'verify', 'gates'):
            _gates = _goalmod.get_gates()
            if not _gates:
                print(_('No gates to check. Add one with: /goal gate add <command>'))
                return 'handled'
            print(_('Running {n} gate(s)...', n=len(_gates)))
            _results = _goalmod.run_gates(cwd=get_cwd())
            _allpass = True
            for _c, _passed, _code, _tail in _results:
                _mark = 'PASS' if _passed else f'FAIL (exit {_code})'
                print(f"  [{_mark}] $ {_c}")
                if not _passed:
                    _allpass = False
                    _t = (_tail or '').strip()
                    if _t:
                        for _ln in _t.splitlines()[-8:]:
                            print(f"      {_ln}")
            print(_('All gates pass — the goal\'s completion bar is met.') if _allpass
                  else _('Some gates fail — the goal is not complete yet.'))
            return 'handled'

        if _sub in ('', 'show', 'status'):
            _cur = _goalmod.get_goal()
            if _cur:
                print(_('Active goal: {goal}', goal=_cur))
                _gates = _goalmod.get_gates()
                for _i, _g in enumerate(_gates, 1):
                    print(f"  gate {_i}: $ {_g}")
            else:
                print(_('No active goal. Set one with: /goal <your objective>'))
        elif _sub in ('clear', 'off', 'none', 'done'):
            _goalmod.clear_goal()
            print(_('Goal cleared.'))
        else:
            _goalmod.set_goal(_rest)
            print(_('Goal set — starting now (kept until /goal clear, injected each turn): {goal}',
                    goal=_rest))
            # Kick off work toward the goal immediately: schedule the objective as
            # the next turn's prompt. The REPL loop runs it (see the _goal_kickoff
            # check after _dispatch_builtin).
            try:
                tool_use_context._goal_kickoff = _rest
            except Exception:
                pass
        return 'handled'

    if cmd in ('effort', 'reasoning', 'think'):
        # Reasoning effort, switched mid-session. The request builder re-reads
        # cfg.reasoning_effort on every call, so a change here lands on the very
        # next message; it is saved too, so the choice outlives the session.
        _parts = user_input.split()
        _arg = _parts[1].lower() if len(_parts) > 1 else 'status'
        from .utils.config import (get_config as _gc, save_config as _sc,
                                   clear_runtime_override as _clear)
        _levels = ('low', 'medium', 'high', 'max')  # max: GLM-5.3 (low/high/max)
        if _arg in _levels or _arg in ('off', 'default', 'none'):
            _val = _arg if _arg in _levels else ''
            # Drop any env/flag shadow (REASONING_EFFORT) so the save writes the
            # deliberate choice, not the shadow's value.
            _clear('reasoning_effort')
            from .utils.config import saved_label as _sl
            _cfg = _gc(); _cfg.reasoning_effort = _val; _saved = _sl(_sc())
            _shown = _val if _val else _('default (server decides)')
            print(_('Reasoning effort set to {level} ({saved}). '
                    'Takes effect on your next message.', level=_shown, saved=_saved))
        elif _arg == 'status':
            _cur = (_gc().effective_reasoning_effort() if hasattr(_gc(), 'effective_reasoning_effort') else getattr(_gc(), 'reasoning_effort', '')) or _('default (server decides)')
            print(_('Reasoning effort is {level}.', level=_cur))
        else:
            print("Usage: /effort low | medium | high | max | off | status")
        return 'handled'

    if cmd == 'keepgoing':
        _parts = user_input.split()
        _arg = _parts[1].lower() if len(_parts) > 1 else 'status'
        if _arg in ('on', 'off'):
            os.environ['CODEMASON_KEEP_GOING'] = '1' if _arg == 'on' else '0'
            # Persist so the choice sticks across sessions on this machine.
            from .utils.config import get_config as _gc, save_config as _sc
            from .utils.config import saved_label as _sl
            _cfg = _gc(); _cfg.keep_going = (_arg == 'on'); _saved = _sl(_sc())
            if _arg == 'on':
                from .utils.keep_going_installer import ensure_installed as _ei
                _acts = _ei()
                print(f"keep-going on ({_saved})."
                      + (" Installed Stop hook." if _acts else ""))
            else:
                print(f"keep-going off ({_saved}). Hook left installed but inert.")
        elif _arg == 'status':
            _eff = _keep_going_enabled()
            _has = has_hook_for_event('Stop')
            print(f"keep-going is {'on' if _eff else 'off'}"
                  + ("" if _has else " — but no Stop hook is installed, so it has no effect")
                  + ".")
        else:
            print("Usage: /keepgoing on | off | status")
        return 'handled'

    return 'handled'   # matched _BUILTIN_COMMANDS but no branch — never reached


def _drain_bg_notifications(messages: List[Any]) -> None:
    """Surface background-shell completions and inject them as
    <task-notification> user messages (enqueueShellNotification's intent).
    _block is already the complete XML block — inject verbatim."""
    from .utils import session_storage
    from .utils.message_queue_manager import drain_pending_notifications
    for _block in drain_pending_notifications():
        print(colorize(f"  {sym('connector')} Background task completed", Colors.CYAN))
        messages.append(create_user_message(_block))
        session_storage.record_transcript(messages)


def _warn_remote_off_in_plain() -> None:
    """Say that configured credentials will not be used in this mode.

    Only when something is actually configured — a warning about a feature you
    never set up is noise, and this console is the one place where a wasted line
    costs the most."""
    try:
        from .remote import notify as _notify
        reason = _notify.off_reason()
        if not reason or not any(a.is_configured() for a in _notify.all_adapters()):
            return
        # ASCII only: this is the mode for terminals that cannot draw anything
        # else, so the warning must not be the one thing that arrives as garbage.
        print(f"[!] {reason}: remote channel off "
              f"(no phone alerts, approvals or questions in this mode)")
    except Exception:
        pass


async def _plain_repl(messages, system_prompt, context, tool_use_context,
                      verbose, debug, initial_prompt):
    """Sequential prompt→turn→prompt loop for --plain/--ascii consoles.
    Input is only possible between turns; Esc-interrupt during a turn comes
    from the raw-stdin monitor inside process_prompt."""
    kb, rprompt = _make_prompt_kb()
    from .utils.completion import build_completer, build_auto_suggest
    prompt_session = PromptSession(
        history=InMemoryHistory(), key_bindings=kb, rprompt=rprompt,
        completer=build_completer(), complete_while_typing=True,
        auto_suggest=build_auto_suggest(),
    )

    if initial_prompt:
        await process_prompt(initial_prompt, messages, system_prompt, context,
                             tool_use_context, verbose, debug)
        if tool_use_context.abort_controller.is_set():
            tool_use_context.abort_controller = asyncio.Event()

    while True:
        try:
            _drain_bg_notifications(messages)

            try:
                print()
                _flush_stdin()
                # Bare prompt. Claude Code does not prefix the prompt with a
                # token count — context/usage is shown in the per-response
                # footer, not glued to every prompt line.
                user_input = (await prompt_session.prompt_async("> ")).strip()
            except UnicodeDecodeError:
                print("Input encoding error. Please try again.")
                continue

            if not user_input:
                continue

            action = await _dispatch_builtin(user_input, messages, tool_use_context)
            _kick = getattr(tool_use_context, '_goal_kickoff', None)
            if _kick:
                tool_use_context._goal_kickoff = None
                await process_prompt(_kick, messages, system_prompt, context,
                                     tool_use_context, verbose, debug)
                if tool_use_context.abort_controller.is_set():
                    tool_use_context.abort_controller = asyncio.Event()
                continue
            if action == 'exit':
                print("Goodbye!")
                break
            if action == 'handled':
                continue

            await process_prompt(user_input, messages, system_prompt, context,
                                 tool_use_context, verbose, debug)

            # Reset abort controller if ESC was pressed during processing
            if tool_use_context.abort_controller.is_set():
                tool_use_context.abort_controller = asyncio.Event()

        except KeyboardInterrupt:
            print("\n[Interrupted]")
            tool_use_context.abort_controller.set()
            tool_use_context.abort_controller = asyncio.Event()
            _flush_stdin()
            continue

        except EOFError:
            print("\nGoodbye!")
            break

        except Exception as e:
            log_error(e)
            print(f"\nError: {e}")
            if debug:
                import traceback
                traceback.print_exc()


def _repl_style_dict() -> dict:
    """prompt_toolkit style for the REPL input area. When a terminal theme is
    active, tint the '>' marker and the typed input so they don't vanish into a
    single-colour body (e.g. homebrew's all-green). Without a theme, leave the
    prompt/input on the terminal's own default colours (safe on light terminals)."""
    style = {
        'spinner': 'fg:ansicyan',
        'input-rule': 'fg:#665f5a',
        'queued': 'fg:#8a8580',   # dim: pending messages waiting above the prompt
        # Context gauge (rprompt). Dim while there is room — it sits next to what
        # you are typing all session and must not compete with it — then takes
        # colour only once the number is worth reacting to.
        'ctx-ok': 'fg:#8a8580',
        'ctx-warn': 'fg:ansiyellow',
        'ctx-crit': 'fg:ansired bold',
    }
    from .utils.terminal_theme import active_theme_colors
    colors = active_theme_colors()
    if colors.get('prompt'):
        style['prompt'] = f"fg:{colors['prompt']} bold"
    if colors.get('input'):
        # '' is prompt_toolkit's base style: colours the typed buffer text (and any
        # unclassed fragment). Named classes above still override it.
        style[''] = f"fg:{colors['input']}"

    # No background on the line being typed. The shading is for the echo that
    # stays in scrollback, not for an area that disappears the moment you press
    # enter — carrying it here only smears grey across where you are working.
    return style


async def _interactive_repl(messages, system_prompt, context, tool_use_context,
                            verbose, debug, initial_prompt):
    """Persistent-input REPL (default for real terminals).

    ONE PromptSession runs for the whole app lifetime as the single owner of
    stdin; all model/tool output is printed ABOVE the live prompt via
    patch_stdout. The user can keep typing while a turn runs — a line submitted
    mid-turn is shown parked above the prompt and INJECTED into the running turn
    at its next model step (mid-turn steering; ReplInputRouter.submit_steering /
    query()'s mid-turn gate), rather than waiting for the turn to end. A line
    typed during a turn that ends before another step (its final response made no
    tool call) is reconciled at the boundary and run as the next turn. The spinner renders in the
    prompt's bottom_toolbar (repainted by refresh_interval), Esc-interrupt is
    a key binding (see _make_prompt_kb), and tool permission prompts are
    routed through the same session (ReplInputRouter.request_permission)."""
    from prompt_toolkit.patch_stdout import patch_stdout
    from .utils.state import get_session_id
    from prompt_toolkit.styles import Style

    router = ReplInputRouter()
    router.tool_use_context = tool_use_context
    router.messages = messages   # same list object the turn loop mutates in
    #                              place — double-Esc rewind truncates it here
    tool_use_context.input_router = router

    # Remote input. A message from the phone is put on the very queue the prompt
    # already feeds, so it inherits the mid-turn queueing behaviour the REPL
    # already has — chat is asynchronous anyway, which is why this needs no Esc
    # equivalent and no second interrupt path. Silently unavailable (and the
    # session behaves exactly as before) when no transport is configured.
    from .remote.input import RemoteInput
    _remote_input = RemoteInput(
        get_session_id(), cwd=os.getcwd(),
        on_message=lambda text, age=0.0, source='': router.submit_remote(text, age, source),
    )
    # One line at startup, because there is no other way to know. Remote needs no
    # flag and no command — any session with credentials is reachable from the
    # phone — so without this the feature is either silently on or silently off
    # and you cannot tell which.
    if _remote_input.start():
        # Same shape as the banner's other status lines (`keep-going: on
        # (toggle: …)`): flush left, lowercase label, hint in parentheses. The
        # first version was indented and led with an emoji, which is two cells
        # wide — so it sat out of line with every row above it.
        # Name the bot this instance claimed. On a machine running several,
        # "remote: telegram" is the same line in every window, and which one you
        # are looking at is the whole question.
        try:
            from .remote.adapters.telegram import claimed_handle
            _h = claimed_handle()
        except Exception:
            _h = ''
        print(f"remote: telegram{f' @{_h}' if _h else ''} (details: /remote)")
        # Once, at the only moment it is actionable and cannot be missed.
        try:
            from .remote.adapters.telegram import credentials_file_warning
            _warn = credentials_file_warning()
            if _warn:
                print(colorize(f"  {sym('warn')} {_warn}", Colors.YELLOW))
        except Exception:
            pass
    else:
        # Not started, which on a fresh machine means "no credentials file" —
        # and the source tree is copied between machines while ~/.codemason is
        # not, so fresh is the normal case. Lay down the empty, commented file
        # and say where it is. Silence here is what made the feature look
        # absent rather than unconfigured.
        try:
            from .remote.adapters.telegram import ensure_credentials_scaffold
            _made = ensure_credentials_scaffold()
            if _made:
                print(colorize(
                    _('remote: off — no bot tokens yet. Wrote a starter file with '
                      '4 slots: {path}', path=str(_made)), Colors.DIM))
        except Exception:
            pass
    # A resumed conversation or an argv initial prompt means there is already
    # output above the prompt — the input rule starts visible.
    router.conversation_started = bool(messages) or bool(initial_prompt)

    # Drop whatever was typed while startup ran: echo was off so none of it was
    # visible, and delivering it now would fill the first prompt with keystrokes
    # the user never saw being entered.
    from .utils.tty_compat import flush_input
    flush_input()

    spinner = StreamingSpinner(external_render=True)
    kb, rprompt = _make_prompt_kb(router)

    def _message():
        # A rule line above the input visually separates the scrolling output
        # from the always-live input area (without it the prompt glued under
        # streaming output reads as a broken screen). Two render rules:
        #   - hidden on the DONE render (the accepted line stays in scrollback
        #     as just '> text' — otherwise every turn leaves a stale bar)
        #   - width cols-1: writing the last column arms the terminal's
        #     pending-wrap state and the redraw math drifts (bars pile up)
        # The prompt doubles as the permission prompt while a tool asks y/n.
        from prompt_toolkit.application.current import get_app_or_none
        app = get_app_or_none()
        prompt_txt = ('bold', router.permission_prompt_text) \
            if router.permission_pending else ('class:prompt', '> ')
        if app is not None and app.is_done:
            # A prompt exited for a PAUSE (resume picker, /expand) is not an
            # accepted line — render nothing, or every pause would leave a
            # stale '> ' in scrollback.
            if router.paused:
                return []
            return [prompt_txt]
        parts = []
        # Live 'Thinking…' ticker sits right below the streamed body (its
        # pre-persistent-prompt position), i.e. directly above the input box.
        # Hidden while a permission ask is pending — the turn is waiting for
        # the USER then, a ticking timer would be noise.
        if spinner.active and not router.permission_pending:
            parts.append(('class:spinner', spinner.toolbar_text() + '\n'))
        cols = 80
        if app is not None:
            try:
                cols = app.output.get_size().columns
            except Exception:
                pass
        if router.conversation_started:
            # Rule between output and input. Hidden at the fresh-start screen
            # (and after /clear) — it would double the banner's separator.
            parts.append(('class:input-rule', '─' * max(10, cols - 1) + '\n'))
        # Queued-message block (mid-turn typing) sits directly ABOVE the '> '
        # input line — NOT in scrollback, NOT in the right prompt. A long paste is
        # capped to a few lines here so it can't push the prompt off-screen.
        if router.pending:
            for _pl in _pending_preview_lines(router.pending, cols):
                parts.append(('class:queued', _pl + '\n'))
        parts.append(prompt_txt)
        return parts

    from .utils.completion import build_completer, build_auto_suggest
    session = PromptSession(
        message=_message,
        history=InMemoryHistory(),
        key_bindings=kb,
        rprompt=rprompt,
        refresh_interval=0.1,   # animates the in-message spinner / elapsed time
        style=Style.from_dict(_repl_style_dict()),
        completer=build_completer(),
        complete_while_typing=True,
        auto_suggest=build_auto_suggest(),
    )
    # Parser-level lone-ESC flush: default 0.5s means even an EAGER Esc
    # binding waits half a second before the parser decides the ESC byte
    # wasn't a sequence prefix. 0.1s restores the raw-monitor feel; real key
    # sequences (arrows) arrive as one burst, so this is safe.
    session.app.ttimeoutlen = 0.1

    router.session = session

    async def _input_loop():
        """Owns the prompt. Submitted lines go to the router: a pending
        permission ask consumes them first; otherwise they enter the message
        queue ('(queued)' while a turn is running)."""
        while True:
            park = router.park_if_paused()
            if park is not None:
                await park
                continue
            try:
                await session.prompt_async()
            except KeyboardInterrupt:
                # Ctrl+C: HARD stop — abort the turn and discard the queued
                # backlog (hard_stop), so the session actually comes to rest.
                # Esc keeps its interrupt-only behaviour on purpose: it is the
                # "send what I typed, now" key.
                if not router.hard_stop():
                    print("(Use /exit or Ctrl+D to quit)")
                continue
            except EOFError:
                await router.queue.put(None)
                return
            except UnicodeDecodeError:
                print("Input encoding error. Please try again.")
                continue
            # prompt_async() only ever returns the _PAUSE sentinel here: Enter is
            # bound to _kb_submit, which queues the typed line directly and never
            # triggers accept-line (the sole app.exit() passes _PAUSE), so a real
            # line never comes back through this return. Nothing to do — loop back
            # to the top, which re-parks if paused. (A former raw.strip()/deliver/
            # queue block here was dead: raw could never be a real line.)

    async def _run_turn(text, remote_origin: bool = False):
        router.processing = True
        router.turn_remote_origin = remote_origin
        try:
            await process_prompt(text, messages, system_prompt, context,
                                 tool_use_context, verbose, debug,
                                 spinner=spinner)
        finally:
            router.processing = False
            if tool_use_context.abort_controller.is_set():
                tool_use_context.abort_controller = asyncio.Event()

    # raw=True: our output is pre-rendered ANSI (colorize/markdown) — pass it
    # through, don't escape it.
    with patch_stdout(raw=True):
        input_task = asyncio.create_task(_input_loop())
        try:
            if initial_prompt:
                await _run_turn(initial_prompt)

            _get = None  # the in-flight queue.get(), preserved across wake-ups
            while True:
                # A background task finishing is a reason to carry on, not a
                # message that waits for the user to press a key.
                #
                # The notification was already being produced and queued; only
                # the last step was missing, and that gap is why the model would
                # not just end its turn and wait. It had no way to be woken, so
                # it reconstructed a sleep out of `sleep 600` in a shell plus
                # polling — burning a full request per cycle — and the
                # alternative was an unattended run that went quiet for as long
                # as the task took. print_runner already holds itself open for
                # exactly this; the interactive loop now does too.
                from .utils.message_queue_manager import has_pending_notifications
                # The pending get() is kept ACROSS wake-ups, never re-created
                # while it is still waiting. Dropping it would leave a coroutine
                # that still takes the next item off the queue with nobody
                # reading the result — silently eating a line the user typed.
                if _get is None:
                    _get = asyncio.ensure_future(router.queue.get())
                _woke_on_task = False
                while True:
                    # Not `_`: that name is the i18n helper imported at module
                    # level, and binding it here made every `_()` call in this
                    # function an UnboundLocalError (the remote-scaffold notice
                    # above silently never printed).
                    _done, _pending = await asyncio.wait({_get}, timeout=0.3)
                    if _done:
                        break
                    if has_pending_notifications():
                        _woke_on_task = True
                        break
                if _woke_on_task:
                    # Fold the completions into the history and let the model act
                    # on them. It asked for the work; it should not need a nudge
                    # from the user to be handed the answer.
                    _drain_bg_notifications(messages)
                    await _run_turn(_TASK_WAKE_PROMPT)
                    continue
                item = _get.result()
                _get = None
                if item is None:
                    print("\nGoodbye!")
                    break
                if item is ReplInputRouter._STEER_WAKE:
                    # A line was enqueued mid-turn for steering. If the turn's own
                    # gate already injected it, the steering queue is empty now and
                    # this wake is a no-op. Otherwise the turn ended before it
                    # reached another model step (e.g. its final response made no
                    # tool call), so there was nothing to inject into — run the
                    # leftover(s) as ordinary turns now, in order.
                    from .utils.steering_queue import drain_steering_messages
                    _exit = False
                    for _leftover in drain_steering_messages():
                        if _leftover in router.pending:
                            router.pending.remove(_leftover)
                        print()
                        print(_user_echo_line(_leftover))
                        _ltag = router.remote_lines.pop(_leftover, None)
                        if _ltag:
                            print(colorize(_ltag, Colors.DIM))
                        try:
                            _drain_bg_notifications(messages)
                            _action = await _dispatch_builtin(
                                _leftover, messages, tool_use_context,
                                remote=bool(_ltag))
                            if _action == 'exit':
                                print("Goodbye!")
                                _exit = True
                                break
                            if _action == 'handled':
                                continue
                            await _run_turn(_leftover)
                        except Exception as e:
                            log_error(e)
                            print(f"\nError: {e}")
                            if debug:
                                import traceback
                                traceback.print_exc()
                    if _exit:
                        break
                    continue
                line, was_queued = item
                # Consumed the moment the line leaves the queue, on EVERY path.
                # It used to be popped inside _run_turn, which a queued line does
                # not always reach: a slash command sent from the phone is
                # answered by _dispatch_builtin and `continue`s, so its entry
                # stayed in remote_lines for the life of the session. That grows
                # without bound in a phone-driven session, and worse, the stale
                # entry re-attributes a LATER local turn that happens to repeat
                # the same text — which forces that answer to the phone past the
                # notify threshold, for a turn nobody asked for from there.
                remote_tag = router.remote_lines.pop(line, None)
                try:
                    _drain_bg_notifications(messages)

                    # A message typed mid-turn waited in router.pending (shown in
                    # the rprompt, NOT scrollback). Now it's actually being
                    # processed — drop it from pending and echo it like a fresh
                    # prompt. `line` is already paste-expanded; _format_user_echo
                    # middle-elides a long paste so it can't carpet the screen.
                    if was_queued:
                        if line in router.pending:
                            router.pending.remove(line)
                        print()   # separate from the previous turn's output
                        print(_user_echo_line(line))
                        if remote_tag:
                            print(colorize(remote_tag, Colors.DIM))

                    action = await _dispatch_builtin(line, messages, tool_use_context)
                    _kick = getattr(tool_use_context, '_goal_kickoff', None)
                    if _kick:
                        tool_use_context._goal_kickoff = None
                        await _run_turn(_kick)
                        continue
                    if action == 'exit':
                        print("Goodbye!")
                        break
                    if action == 'handled':
                        continue

                    await _run_turn(line, remote_origin=remote_tag is not None)

                except Exception as e:
                    log_error(e)
                    print(f"\nError: {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
        finally:
            input_task.cancel()
            try:
                await input_task
            except (asyncio.CancelledError, Exception):
                pass
            # Stop advertising this session and hand the poll lock to whoever
            # wants it next. Anything still sitting in the inbox is left alone —
            # a message that arrived during shutdown is still the user's, and
            # sweep() reclaims it once this process is provably gone.
            _remote_input.stop()


def _notify_turn_end(started: float, answer: str, *, cwd: str, error: str = "",
                     tools: Optional[Dict[str, int]] = None,
                     remote_origin: bool = False, progress=None,
                     prompt: str = "", messages: Optional[List[Any]] = None) -> None:
    """REPL-side shim over remote.hooks (shared with the headless -p runner)."""
    from .remote.hooks import notify_turn_end
    from .utils.session_status import status_line
    notify_turn_end(int((time.time() - started) * 1000), answer, cwd=cwd,
                    error=error, tools=tools, remote_origin=remote_origin,
                    progress=progress, prompt=prompt,
                    status=status_line(messages) if messages is not None else '')


async def process_prompt(
    user_input: str,
    messages: List[Dict[str, Any]],
    system_prompt: List[str],
    context: Dict[str, str],
    tool_use_context: ToolUseContext,
    verbose: bool = False,
    debug: bool = False,
    spinner: Optional[StreamingSpinner] = None,
):
    """Process a single prompt and display results.

    spinner=None (plain/sequential mode): a \\r-rendering spinner is created
    here and the raw-stdin ESC monitor provides interrupt during the turn.
    spinner given (interactive mode): the shared external-render spinner is
    used (bottom_toolbar) and stdin stays with the persistent PromptSession —
    no monitor, Esc-interrupt is a key binding."""
    from .utils import session_storage
    from .services import ab as _ab

    # A user turn is starting: under `/ab auto` the model is about to decide
    # whether to accept, the one moment the refusal edit is wanted. Placed in
    # the shared entry point so every caller (initial prompt, goal kickoff,
    # REPL, remote) gets the same boundary.
    await _ab.on_user_turn()

    # The image section of the system prompt follows the server's limit; a
    # session started while the server could not be asked gets it here, on the
    # first turn the server answers (constants/prompts.py refresh_system_prompt).
    try:
        from .constants.prompts import refresh_system_prompt
        _opts = getattr(tool_use_context, 'options', None) or {}
        await refresh_system_prompt(system_prompt, _opts.get('tools'), get_config().model_name)
    except Exception:  # noqa: BLE001 - a prompt refresh must never cost the turn
        pass

    # What the user TYPED decides the UI language when nothing else has (no
    # config.json, no LANG — i.e. every machine the source was merely copied to).
    # Before the paste expansion below on purpose: a pasted English log is not
    # the user writing English, and after expansion it would be indistinguishable
    # from it.
    from .utils.i18n import observe_user_text
    observe_user_text(user_input)

    # Expand any [Pasted text #N] placeholders back to their full content before
    # the message goes to the model. On screen the user saw only the compact
    # placeholder (and can still Ctrl+O it); the model receives the whole paste.
    _router = getattr(tool_use_context, 'input_router', None)
    _pstore = getattr(_router, 'paste_store', None) if _router is not None else None
    if _pstore is not None:
        user_input = expand_pasted_refs(user_input, _pstore)
    # Every path into the model funnels through here (REPL, queued messages,
    # -p), so this is where lone surrogates from the Windows console get fixed
    # before they reach json/utf-8 encoding in the backend.
    user_input = repair_surrogates(user_input)

    # NEITHER cwd NOR date goes in `context` any more. `context` is folded into
    # the SYSTEM PROMPT (format_system_prompt_with_context), which is the cached
    # prefix: every byte of it has to be stable for the whole session or the
    # server re-prefills the entire conversation.
    #
    # The date used to be PINNED to the REPL-start value for exactly that reason
    # — accuracy traded away to protect the prefix — while cwd was refreshed
    # every turn on the reasoning that it "changes rarely". Measured on a real
    # session (2026-09-05): it changed 18 times while moving between repos, and
    # 4 of the 9 worst stalls immediately follow a change — 44% against a 10.8%
    # base rate. At ~200k tokens each stall was 138-191s.
    #
    # The trade was never necessary. Measured on this backend at 96k tokens:
    # appending to the end costs 3.4s, changing the front costs 72s. Same
    # information, 20x apart, decided purely by WHERE it sits. So both values
    # now ride on the user turn, where they are appended — the model gets the
    # CURRENT directory and the CURRENT date every turn, and the prefix never
    # moves. The date is accurate again as a side effect.
    _cwd_now = get_cwd()
    _turn_text = (
        '<turn-env>\nCurrent Working Directory: %s\n'
        'All relative paths are based on this directory. "." means %s.\n'
        "Today's date: %s\n</turn-env>"
        % (_cwd_now, _cwd_now, datetime.now().strftime('%Y-%m-%d'))
    )

    # The Agent-prompt language rule. It used to be built into the Agent tool's
    # description AND its JSON schema — i.e. into the tool block, which the
    # request carries ahead of everything else — so each swing of the detected
    # language threw the whole conversation out of the server's prefix cache.
    # Same information, appended instead of embedded.
    from .utils.language import effective_language as _eff_lang
    from .utils.language import detect_language as _detect_lang, pin_session_language as _pin_lang
    from .tools.agent_tool import _prompt_language_rule as _lang_rule
    # Also the moment the SESSION language gets pinned (first prompt with a
    # script signal wins) — sub-agent prompts are built from that pin, never from
    # the per-request value, so their prefix cannot drift.
    _pin_lang(_detect_lang(user_input))
    _language = _eff_lang()
    if _language:
        _turn_text += ('\n<agent-prompt-language>\n%s\n</agent-prompt-language>'
                       % _lang_rule(_language))

    # Persistent goal: re-read every turn so a mid-session /goal change lands on
    # the next one. It used to go into `context`, which is folded into the SYSTEM
    # PROMPT — so changing the goal mid-session silently invalidated the whole
    # cached history. Appended here it cannot.
    from .utils.goal import render_goal_context as _render_goal
    _goal = _render_goal()
    if _goal:
        _turn_text += '\n<active_goal>\n%s\n</active_goal>' % _goal

    turn_env = create_user_message(_turn_text)
    turn_env['isMeta'] = True
    # Fresh user turn: reset the per-turn goal-gate auto-run counter (see
    # query.py GOAL_GATE_RECOVERY_LIMIT).
    try:
        tool_use_context._gate_recovery = 0
    except Exception:
        pass

    # Add user message
    user_msg = create_user_message(user_input)

    # Make good on a summary taken mid-turn. Same loss as the reduction below,
    # but this one cannot be replayed here: a summary is an API round trip, not
    # a function of the history, so microcompact_messages reproduces nothing.
    # The REPL summarizes its own history once, here.
    #
    # Deliberately NOT routed through auto_compact_if_needed. Its only
    # threshold-bypassing entrance is context_proven_fatal, which escalates to
    # microcompact and no further -- and by the time a mid-turn summary has
    # happened microcompact is already exhausted, so it would free 0 and leave
    # the full history in place. Its threshold gate is no use either: the
    # mid-turn requests went out summarized, so the usage anchor describes a few
    # thousand tokens while this history is still the whole conversation.
    #
    # One summary call, which an honest estimate would have made anyway. Its
    # output is capped at MAX_OUTPUT_TOKENS_FOR_SUMMARY (20k), so it fits where
    # the ordinary turn's 65,536-token reservation does not.
    _ns = getattr(tool_use_context, '_needs_summary', None)
    if isinstance(_ns, dict) and _ns.get('raised'):
        _ns['raised'] = False
        _rspin = StreamingSpinner()
        try:
            from .services.compaction import compact_conversation as _cc
            _rspin.start(verb='Compacting')
            _rres = await _cc(
                messages, tool_use_context,
                custom_instructions=None, is_auto_compact=True,
                on_token_update=_rspin.update,
                on_thinking_update=_rspin.update_thinking,
            )
            _rspin.stop()
            messages.clear()
            messages.extend(_rres.messages)
            # Same reason /compact does it: the summary carries no file
            # contents, so every path the model still believes it has read must
            # be re-read rather than assumed fresh.
            _rfs = getattr(tool_use_context, 'read_file_state', None)
            if isinstance(_rfs, dict):
                _rfs.clear()
            print(f"  {sym('connector')} Auto compacted {sym('dash')} conversation summarized. Continuing...\n")
        except Exception as _re:
            _rspin.stop()
            # Left unreduced on purpose: the threshold path below still gets its
            # turn, and a wrong summary is worse than none.
            try:
                from .utils.log import log_error as _lerr
                _lerr('a summary taken mid-turn could not be re-applied to the REPL '
                      'history: %s -- the full history is still in place' % (_re,),
                      'compaction: summary readopt failed')
            except Exception:
                pass

    # Re-apply a reduction that was taken mid-turn. _compact_before_recursion
    # computes a trimmed history, hands it to the recursion and then loses it —
    # that list belongs to the recursion, this one is authoritative — so it
    # raises _needs_reduction and the reduction is redone here, where it sticks.
    #
    # This must run BEFORE should_auto_compact below. The mid-turn requests went
    # out trimmed, so the usage anchor describes the slim prompt while this
    # history is still the fat one: should_auto_compact would weigh the slim
    # number, decline, and send the fat history. In the Rust port on 2026-10-03
    # 18:07 that was 458,753 input + 65,536 reserved output against a 524,288
    # window — a 400, with the gauge reading 262k.
    #
    # microcompact_messages is idempotent (an already-cleared result is
    # skipped), so a history that needs nothing frees 0 and is left alone.
    _nr = getattr(tool_use_context, '_needs_reduction', None)
    if isinstance(_nr, dict) and _nr.get('raised'):
        _nr['raised'] = False
        from .services.compaction import microcompact_messages as _mc
        _cleared, _freed = _mc(messages)
        if _freed > 0:
            messages.clear()
            messages.extend(_cleared)
        try:
            from .utils.log import debug_log as _dbg
            _dbg('REDUCTION_READOPTED', {'tokens_freed': _freed})
        except Exception:
            pass

    # Auto-compact before sending the new message. Summarizes and
    # replaces the conversation history when the token count crosses the threshold.
    from .services.compaction import auto_compact_if_needed, should_auto_compact
    # Gate the spinner on the same condition the function gates itself on, or it
    # would flash on every single turn. Microcompact may still relieve it
    # without a summary call, and the spinner then shows for the moment that
    # takes — which is honest: something did run.
    from .services.compaction import oob_scope as _oob_scope
    _ac_spin = (StreamingSpinner()
                if should_auto_compact(messages, False, _oob_scope(tool_use_context))
                else None)
    if _ac_spin is not None:
        _ac_spin.start(verb='Compacting')
    # Same counter query.py threads mid-turn, so the circuit breaker
    # (MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES) covers this call site too. With
    # no tracking passed, a compaction that failed every time was retried on
    # every user message — bounded by the human, and in a remote -dsp session
    # the human is a message queue.
    _tracking = getattr(tool_use_context, '_autocompact_tracking', None) or {}
    # context_proven_fatal: normally False — this is the pre-turn check, before
    # anything has been sent, so there is no collapse to have observed yet.
    #
    # True for exactly one turn after the repetition recovery's last resort
    # microcompacted a collapsed history and could not keep the result (see
    # ToolUseContext._proven_fatal). Taken here because this is the only
    # auto-compaction call site that owns `messages` and writes the reduced set
    # back; mid-turn it can only be handed to a recursion. Consumed, not read:
    # it applies to the turn that follows the rescue and then goes quiet, so a
    # one-off collapse cannot keep forcing reductions for the rest of the
    # session.
    _pf = getattr(tool_use_context, '_proven_fatal', None)
    if isinstance(_pf, dict):
        _proven_fatal, _pf['raised'] = bool(_pf.get('raised')), False
    else:
        _proven_fatal = False
    try:
        _compact = await auto_compact_if_needed(
            messages, tool_use_context, _tracking,
            on_token_update=_ac_spin.update if _ac_spin else None,
            on_thinking_update=_ac_spin.update_thinking if _ac_spin else None,
            context_proven_fatal=_proven_fatal)
    finally:
        if _ac_spin is not None:
            _ac_spin.stop()
    if 'consecutive_failures' in _compact:
        _tracking['consecutive_failures'] = _compact['consecutive_failures']
        try:
            tool_use_context._autocompact_tracking = _tracking
        except Exception:
            pass
    if _compact.get('was_compacted'):
        messages.clear()
        messages.extend(_compact['compaction_result'].messages)
        print(f"  {sym('connector')} Auto compacted {sym('dash')} conversation summarized. Continuing...\n")
    elif _compact.get('was_microcompacted'):
        # query.py:732 already handles this shape; the turn-boundary caller did
        # not, and microcompact_messages returns a NEW list rather than mutating
        # in place, so the trimmed history was computed and then dropped. Two
        # effects compounded: the full untrimmed history was sent anyway, and
        # should_auto_compact subtracts the freed tokens it believes were
        # released, inflating the threshold every turn — so the real summarizing
        # compaction fired later than designed and the request could cross the
        # server's context window before it ever triggered.
        messages.clear()
        messages.extend(_compact['messages'])
        print('  ⎿  Microcompacted — old tool results cleared, recent context kept.\n')
    elif _compact.get('consecutive_failures', 0) > 0:
        # Compaction was attempted and reduced NOTHING. This printed nothing at
        # all before: the spinner simply vanished and the turn carried on with the
        # full history, so "Compacting… (5.7s ↓ 0 tokens)" followed by silence was
        # indistinguishable from success. Worse, after
        # MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES the breaker disables compaction for
        # the rest of the SESSION — also silently — and only a successful
        # compaction resets the counter, which by then can no longer happen. So
        # say it, and say how close the breaker is.
        from .services.compaction import MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES as _max_acf
        _cf = _compact.get('consecutive_failures', 0)
        if _cf >= _max_acf:
            print(f"  {sym('connector')} Auto compaction is now DISABLED for this "
                  f"session after {_cf} failures {sym('dash')} the context will keep "
                  "growing. Use /clear, or /compact once the model server is healthy.\n")
        else:
            print(f"  {sym('connector')} Auto compaction failed ({_cf}/{_max_acf}) "
                  f"{sym('dash')} context was NOT reduced. See error.log; the turn "
                  "continues on the full history.\n")

    # Environment first, then what the user actually said. Both are APPENDED,
    # so neither disturbs the cached prefix.
    messages.append(turn_env)
    messages.append(user_msg)
    # Persist incrementally (useLogMessages pattern): full list in, only the
    # unseen tail is appended to the session JSONL. Also picks up the compact
    # summary messages when auto-compact just replaced the history.
    session_storage.record_transcript(messages)

    print()  # Blank line before response

    # Create and start spinner (owned here only in sequential mode)
    _sequential = spinner is None
    if _sequential:
        spinner = StreamingSpinner()
    spinner.start()

    # Set up token update callbacks. on_thinking_update only fires on turns that
    # actually produced reasoning, so the spinner's thinking segment appears
    # exactly then and stays absent otherwise.
    tool_use_context.on_token_update = spinner.update
    tool_use_context.on_thinking_update = spinner.update_thinking
    # pin_verb, not start(): a compaction in the middle of a turn must not reset
    # the turn's elapsed clock or its token counts.
    tool_use_context.on_status = spinner.pin_verb

    # -rt/--realtime: stream the reply live, MARKDOWN-rendered, BLOCK by block.
    # The DSML filter (in query_model) already stripped/held tool/think markup;
    # here we group the safe text into complete markdown blocks (paragraph, or a
    # whole ``` fence) and render each with the SAME safe_print(render_markdown)
    # path the rest of the UI uses. That gives three things the raw per-char
    # write couldn't: real markdown, no dangling partial line for a prompt repaint
    # to wipe, and clean coordination with tool-use/result printing. `_rt_streamed`
    # tells the assistant branch to flush the trailing block instead of re-render.
    from .utils.realtime_render import MarkdownBlockBuffer
    _realtime = bool(getattr(get_config(), 'realtime', False))
    _rt_streamed = {'any': False}
    _rt_printed = {'any': False}   # a block of THIS message is already on screen
    _rt_md = MarkdownBlockBuffer()

    def _print_md_block(block: str) -> None:
        """Render one markdown block, keeping exactly one blank line between blocks.

        Rendered whole, a markdown document gets its inter-block spacing from rich.
        Block by block it does not: rich pads a list/paragraph with a leading blank
        line but a heading with none, so '## Shortcomings' would print flush against the
        numbered list above it. Normalise here — strip the renderer's own padding
        and own the separator ourselves."""
        # Strip ANSI/control bytes from the MODEL's text before rendering (the
        # tool-result path sanitizes too; the -rt path previously did not, so a
        # quoted ESC sequence in a streamed reply reached the terminal raw when
        # color was off and render_markdown returned the text verbatim).
        block = sanitize_terminal_text(block)
        rendered = render_markdown(block).strip('\n')
        if not rendered:
            return
        if _rt_printed['any']:
            print()
        _rt_printed['any'] = True
        safe_print(rendered)

    if _realtime:
        def _emit_delta(text: str):
            for block in _rt_md.feed(text):
                if not block.strip():
                    continue
                if not _rt_streamed['any'] and _sequential:
                    spinner.stop()  # \r-spinner would clobber the first block
                _rt_streamed['any'] = True
                _print_md_block(block)
        tool_use_context.on_content_delta = _emit_delta

    # ESC key monitor (event loop add_reader) — sequential mode only; in
    # interactive mode the persistent PromptSession owns stdin and Esc is a
    # key binding there.
    monitor_state = (
        _start_escape_monitor(tool_use_context.abort_controller)
        if _sequential else None
    )

    # Track tool_use_id -> (tool_name, tool_input) for pretty display
    pending_tool_uses: Dict[str, tuple] = {}
    _shown_tool_headers: set = set()   # -rt: headers already shown before exec
    # Remote notification state (codemason.remote). The session lives here, not on
    # the phone, so nothing about this depends on a client being attached.
    _turn_started = time.time()
    _last_answer = {'text': ''}        # last assistant prose, for the alert body
    # The '⎯ ↓ …' usage line, tracked across the whole turn:
    #   shown  — has this TURN printed one at all? A turn that ends in tool calls
    #            or is interrupted before any prose used to print nothing, so the
    #            one moment you most want the number ("what did that just cost?")
    #            was the one moment it was missing.
    #   done   — has THIS assistant message printed one? Reset per message. The
    #            line belongs to an API response, not to a text block; a reply
    #            split into several text blocks printed one after each.
    #   usage/duration_ms — the newest response's numbers, so the turn-end
    #            fallback reports that response rather than re-deriving it.
    #   responded — did any assistant message arrive at all? A turn that got no
    #            response (empty stream, a failure before the first chunk) has
    #            nothing to report, and a '0 / 232.8k (0%) · 0.0s' line would be
    #            noise standing in for an answer that never came.
    _footer: Dict[str, Any] = {'shown': False, 'done': False, 'responded': False,
                               'usage': None, 'duration_ms': 0}
    _tool_counts: Dict[str, int] = {}  # tool NAMES only — never their arguments
    # A phone has no spinner, so a remote-initiated turn keeps one message
    # current while it runs. Local turns get one too, held off until the point
    # where the turn qualifies to be reported at all: walking away mid-turn is
    # the case a live message exists for, and it costs no extra chat line since
    # the same message becomes the answer.
    _remote_origin = getattr(
        getattr(tool_use_context, 'input_router', None), 'turn_remote_origin', False)
    _progress = None
    try:
        from .remote.hooks import turn_report_delay
        _delay = turn_report_delay(_remote_origin)
        if _delay is not None:
            from .remote.progress import RemoteProgress
            _progress = RemoteProgress(cwd=os.getcwd(), delay=_delay)
            _progress.start()
    except Exception:
        _progress = None
    _had_text = False                  # text seen before a tool (for the one-blank
    #                                    rule); re-set per assistant message, read by
    #                                    both the -rt immediate and paired result paths

    def _print_token_footer(usage: Any, duration_ms: int) -> None:
        """Print the usage line for one response. Callers own the de-duplication
        (`_footer['done']`); this only records that the turn has shown one."""
        line = _render_token_footer(usage, duration_ms, spinner, messages, verbose)
        if line:
            print(line)
            _footer['shown'] = True

    try:
        async for message in query(
            messages=messages,
            system_prompt=system_prompt,
            context=context,
            can_use_tool=has_permissions_to_use_tool,
            tool_use_context=tool_use_context,
        ):
            msg_type = message.type if hasattr(message, 'type') else message.get('type')

            if msg_type == 'assistant':
                # Stop spinner when we get a response
                spinner.stop()

                # Add to message history
                messages.append(message)
                session_storage.record_transcript(messages)

                # Get token usage and duration for display
                msg_obj = message.message if hasattr(message, 'message') else message.get('message', {})
                usage = msg_obj.get('usage') if isinstance(msg_obj, dict) else getattr(msg_obj, 'usage', None)
                duration_ms = message.duration_ms if hasattr(message, 'duration_ms') else message.get('duration_ms', 0)

                # If no usage from API, time the response off the spinner instead.
                if not _reported_usage(usage):
                    elapsed, tokens = spinner.get_final_stats()
                    duration_ms = int(elapsed * 1000) if elapsed else duration_ms

                # This response's numbers, for the footer: one line per response
                # (`done` resets here), and the last one seen stays available for
                # the turn-end fallback below.
                _footer['done'] = False
                _footer['responded'] = True
                _footer['usage'] = usage
                _footer['duration_ms'] = duration_ms

                # Display response
                content = message.message.content if hasattr(message, 'message') else message.get('message', {}).get('content', [])

                # -rt: only show a header before execution when this message has a
                # SINGLE tool_use — that's the slow-single-tool case (du/build/find)
                # where the mystery-spinner hurt. With MULTIPLE concurrent tools we
                # keep the paired header+result printing (else headers clump above
                # results). So single → live header, parallel → tidy pairs.
                _n_tool_uses = sum(
                    1 for b in content
                    if (b.type if hasattr(b, 'type') else b.get('type')) == 'tool_use'
                )
                _had_text = False   # text seen BEFORE a tool in THIS message (per
                #                     message, not per turn) → one blank before the
                #                     tool header, without doubling the tool→tool gap

                for block in content:
                    block_type = block.type if hasattr(block, 'type') else block.get('type')

                    if block_type == 'text':
                        text = block.text if hasattr(block, 'text') else block.get('text', '')
                        text = _strip_completion_sentinel(text)
                        if text:
                            _had_text = True   # a tool later in THIS message needs a blank before it
                            # -rt: always DRAIN the block buffer (so a single-block
                            # message's leftover can't bleed into the next message).
                            # If blocks already streamed live, print only the trailing
                            # block (re-rendering `text` would double-print). If
                            # nothing streamed (one sub-block paragraph), render the
                            # whole cleaned text as markdown once, here.
                            if _realtime:
                                _rest = _rt_md.flush()
                                if _rt_streamed['any']:
                                    if _rest.strip():
                                        _print_md_block(_rest)
                                else:
                                    _print_md_block(text)
                                # This message is done; the next one starts its own
                                # spacing (a tool result already ends with a blank).
                                _rt_streamed['any'] = False
                                _rt_printed['any'] = False
                            else:
                                safe_print(render_markdown(text))
                            _last_answer['text'] = text   # newest prose wins
                            if _progress is not None:
                                _progress.on_text(text)
                            # The usage line is NOT printed here: a reply split
                            # into several text blocks would print one after each
                            # (and one wedged between its own paragraphs). It is
                            # emitted once the response's prose is complete —
                            # below, at the end of the block loop, or just before
                            # the first tool header if one follows.

                    elif block_type == 'tool_use':
                        # Prose in THIS response is finished — close it with the
                        # usage line now, above the tool. Leaving it to the end of
                        # the loop would drop it between a tool header and that
                        # tool's own result, splitting a pair that is only legible
                        # together. Nothing is printed for a response that ran
                        # tools without speaking: those repeat several times a
                        # turn, and the turn-end line below already reports them.
                        if _had_text and not _footer['done']:
                            _footer['done'] = True
                            _print_token_footer(usage, duration_ms)

                        api_tool_name = block.name if hasattr(block, 'name') else block.get('name', '')
                        if api_tool_name:
                            _tool_counts[api_tool_name] = _tool_counts.get(api_tool_name, 0) + 1
                            if _progress is not None:
                                _progress.on_tool(api_tool_name)
                        tool_input = block.input if hasattr(block, 'input') else block.get('input', {})
                        tool_use_id = block.id if hasattr(block, 'id') else block.get('id', '')

                        if tool_use_id:
                            pending_tool_uses[tool_use_id] = (api_tool_name, tool_input)
                        # -rt: show the tool-use header NOW, before the tool runs,
                        # so a slow tool (a du/find over a 1TB tree, a build) shows
                        # WHAT is running instead of a silent spinner. The paired
                        # print in the tool_result branch skips headers already
                        # shown here. (Default mode keeps the paired-only behaviour
                        # so concurrent read-only tools still group header+result.)
                        if _realtime and tool_use_id and _n_tool_uses == 1:
                            try:
                                _tc = get_tool_by_name(api_tool_name)
                                _dn = _tc.user_facing_name(tool_input)
                            except ValueError:
                                _dn = api_tool_name
                            # If THIS message streamed text before this tool (e.g.
                            # "let me search" then WebSearch), the header would glue to
                            # that text / its '⎯ ↓ tokens' line — headers carry no
                            # leading blank. Add one. Per-MESSAGE (not per-turn): a
                            # tool that follows a previous tool's result is already
                            # separated by that result's trailing blank, so _had_text
                            # is False there and we don't double the gap.
                            if _had_text:
                                print()
                            display_tool_use(api_tool_name, _dn, tool_input, verbose)
                            _shown_tool_headers.add(tool_use_id)

                # Prose that ran to the end of the response (no tool followed):
                # the usage line closes it here, after the LAST text block.
                if _had_text and not _footer['done']:
                    _footer['done'] = True
                    _print_token_footer(usage, duration_ms)

                # Keep the spinner ALIVE during tool execution (Claude Code
                # parity). Tools run inside query() between this assistant
                # message and the tool_result 'user' message; without this the
                # spinner is dark for the whole run, so a long tool (build,
                # background wait) looks like a total freeze — the exact symptom
                # of "typed input echoes but nothing responds". The 'user'
                # branch below stops it again before printing the result.
                _tool_names = [
                    (b.name if hasattr(b, 'name') else b.get('name'))
                    for b in content
                    if (b.type if hasattr(b, 'type') else b.get('type')) == 'tool_use'
                ]
                if _tool_names:
                    # Labelled with the tool. config.spinner_random_verb restores
                    # upstream's decorative random verb for anyone who prefers
                    # it. In a long run the "[Bash] $ ..." line above has
                    # scrolled away, and a word picked at random is then the
                    # only thing on screen. An in-progress todo still wins over
                    # both — that is what activeForm is for.
                    if getattr(get_config(), 'spinner_random_verb', False):
                        _label = None
                    elif len(_tool_names) == 1:
                        _label = _tool_names[0]
                    else:
                        _label = '%s +%d' % (_tool_names[0], len(_tool_names) - 1)
                    spinner.start(_label)

            elif msg_type == 'user':
                # Stop spinner before printing. The previous tool-result block
                # ends with spinner.start(), which parks "Thinking…" at column 0
                # with no newline (carriage-return render). Without clearing it
                # first, a following tool result (e.g. parallel/consecutive Reads)
                # prints onto that same line: "Thinking… (0.0s) ⎿ 1 import…".
                spinner.stop()

                # Tool result - add to history
                messages.append(message)
                session_storage.record_transcript(messages)

                # Show tool result (abbreviated unless verbose)
                msg_obj = message.message if hasattr(message, 'message') else message.get('message', {})
                content = msg_obj.content if hasattr(msg_obj, 'content') else msg_obj.get('content', [])
                for block in content:
                    block_type = block.type if hasattr(block, 'type') else block.get('type') if isinstance(block, dict) else None
                    if block_type == 'tool_result':
                        result = block.content if hasattr(block, 'content') else block.get('content', '') if isinstance(block, dict) else str(block)
                        is_error = block.is_error if hasattr(block, 'is_error') else block.get('is_error', False) if isinstance(block, dict) else False
                        tool_use_id = block.tool_use_id if hasattr(block, 'tool_use_id') else block.get('tool_use_id', '') if isinstance(block, dict) else ''
                        # The tool's structured Output (query.py carries it next to the
                        # model-facing content) — what render_tool_result_message renders.
                        result_data = block.get('data') if isinstance(block, dict) else getattr(block, 'data', None)

                        # Print this tool's header PAIRED with its result. (Was
                        # printed in the assistant branch, which clumped all N
                        # headers from one message before any result.) Show it for
                        # read-only / skip-mode tools; write tools in interactive
                        # mode were already shown by the permission handler — don't
                        # repeat. (--dangerously-skip-permissions = no permission
                        # prompt, so write tools weren't shown there → skip term.)
                        _tu = pending_tool_uses.get(tool_use_id)
                        if _tu:
                            _tname, _tinput = _tu
                            try:
                                _tcls = get_tool_by_name(_tname)
                                _ro = _tcls.is_read_only()
                                _dname = _tcls.user_facing_name(_tinput)
                            except ValueError:
                                _ro, _dname = True, _tname
                            _skip = tool_use_context.options.get('dangerously_skip_permissions', False)
                            # A session-'always'-allowed tool was auto-approved
                            # WITHOUT the permission prompt showing its header, and
                            # (unlike read-only/skip) the gate below skipped it too
                            # — so a Bash/Write after "always allow" printed only
                            # its result with no "[Bash] ls …" header. Show it here.
                            try:
                                _session_allowed = _permission_key(_tname, _tinput) in _SESSION_TOOL_ALLOWS
                            except Exception:
                                _session_allowed = False
                            # -rt already printed the header before execution — don't repeat it.
                            if (_ro or _skip or _session_allowed) and tool_use_id not in _shown_tool_headers:
                                # Same rule as the -rt immediate path: if text was
                                # streamed just before this tool (paired/non-rt path,
                                # e.g. parallel tools), one blank separates the header
                                # from that text. Consumed once so parallel results
                                # don't each add a blank.
                                if _had_text:
                                    print()
                                    _had_text = False
                                display_tool_use(_tname, _dname, _tinput, verbose)

                        # Handle result that might be a list of TextBlocks
                        if isinstance(result, list):
                            result = '\n'.join(
                                b.text if hasattr(b, 'text') else b.get('text', str(b)) if isinstance(b, dict) else str(b)
                                for b in result
                            )
                        elif hasattr(result, 'text'):
                            result = result.text
                        elif not isinstance(result, str):
                            result = str(result)

                        # Strip terminal-corrupting escape/control bytes before any
                        # display path. Tool output is untrusted (a package parser
                        # can dump binary AXML/DEX with NULs); echoing it raw wrecks
                        # the terminal. Applied here so every branch below — and the
                        # Ctrl+O expand record — sees clean text.
                        result = sanitize_terminal_text(result)

                        # Keep the FULL text for Ctrl+O / /expand before the
                        # display truncates it below.
                        if result and not is_error:
                            _rt = pending_tool_uses.get(tool_use_id)
                            _record_tool_output(
                                _rt[0] if _rt else '', _rt[1] if _rt else None, result)

                        # The tool's own result rendering (Tool.render_tool_result_message,
                        # the port of renderToolResultMessage) — fed the structured Output,
                        # so a tool whose model-facing payload is JSON/XML (TaskStop,
                        # TaskOutput) still prints a human line here. None = no custom
                        # rendering; fall through to the raw text below.
                        tu_info = pending_tool_uses.get(tool_use_id)
                        _rendered = None
                        if tu_info and not is_error:
                            try:
                                _rendered = get_tool_by_name(tu_info[0]).render_tool_result_message(
                                    result_data, verbose=verbose)
                            except ValueError:
                                _rendered = None
                            except Exception as _e:
                                debug_log(f"Error rendering tool result for {tu_info[0]}: {_e}")
                                _rendered = None

                        if _rendered is not None:
                            # '' = the tool renders nothing (its result is surfaced
                            # elsewhere, e.g. TodoWrite's list above).
                            if _rendered:
                                _prefix = f"  {sym('connector')} "
                                _indent = ' ' * display_width(_prefix)
                                print(_prefix, end='')
                                for i, line in enumerate(_rendered.split('\n')):
                                    safe_print(line if i == 0 else f"{_indent}{line}")
                        elif result:
                            if is_error:
                                lines = result.split('\n')
                                print(f"  {sym('connector')} Error: {lines[0][:100]}")
                            else:
                                lines = result.split('\n')
                                max_lines = 20 if verbose else 5
                                # First line carries the connector; continuation
                                # lines must indent by the connector prefix's
                                # actual display width (1 cell for '⎿', 2 for the
                                # plain-ASCII '\_'), or line 1 ends up shifted.
                                _prefix = f"  {sym('connector')} "
                                _indent = ' ' * display_width(_prefix)
                                print(_prefix, end='')
                                for i, line in enumerate(lines[:max_lines]):
                                    if i == 0:
                                        safe_print(line[:100] + ('...' if len(line) > 100 else ''))
                                    else:
                                        safe_print(f"{_indent}{line[:100]}" + ('...' if len(line) > 100 else ''))

                                if len(lines) > max_lines:
                                    # Claude Code shows how to expand right at
                                    # the cut; Ctrl+O only exists in the
                                    # interactive (persistent-prompt) mode.
                                    _hint = '' if _sequential else ' · ctrl+o to expand'
                                    print(f"{_indent}"
                                          + colorize(f"... ({len(lines) - max_lines} more lines{_hint})", Colors.DIM))

                print()  # Blank line after tool result for readability

                # Restart spinner for next LLM response
                spinner.start()

            elif msg_type == 'progress':
                # Progress message from Agent/Task
                progress_data = message.data if hasattr(message, 'data') else message.get('data', {})
                result_text = message.result_for_assistant if hasattr(message, 'result_for_assistant') else message.get('result_for_assistant', '')

                if result_text:
                    spinner.stop()

                    tool_name = progress_data.get('tool_name', '') if progress_data else ''
                    tool_input = progress_data.get('tool_input', {}) if progress_data else {}

                    if tool_name:
                        # One line per sub-agent tool call (renderToolActivity.tsx).
                        # summarize_tool_input, not the tool's renderer: this line is
                        # colourised here and must stay ANSI-free before truncation,
                        # and it must never be a raw input dict.
                        input_preview = summarize_tool_input(tool_input, max_len=_AGENT_PROGRESS_COLS)
                        _agent_label = colorize(f"Agent {sym('arrow')} [{tool_name}]", Colors.CYAN)
                        safe_print(f"  {sym('connector')} {_agent_label} {colorize(input_preview, Colors.DIM)}")
                    elif 'Initializing' in str(result_text):
                        print(f"  {sym('connector')} {colorize('Agent starting...', Colors.CYAN)}")
                    else:
                        print(f"  {sym('connector')} {result_text}")

                    # Resume spinner (keep token count while agent sub-query runs)
                    spinner.resume()

        # Stop spinner at the end
        spinner.stop()

        # A turn that printed no usage line still gets exactly one. Ending in
        # tool calls with no closing prose, or being interrupted mid-stream, is
        # not a reason to answer "what did that cost" with silence — and those
        # are the turns where the answer matters most.
        if _footer['responded'] and not _footer['shown']:
            _print_token_footer(_footer['usage'], _footer['duration_ms'])

        _notify_turn_end(_turn_started, _last_answer['text'], cwd=os.getcwd(),
                         tools=_tool_counts,
                         remote_origin=_remote_origin, progress=_progress,
                         prompt=user_input, messages=messages)

        # Show context warning after all messages processed. Counted WITH
        # estimation, like the input-line gauge and the footer: count_tokens
        # reads only the last response's usage, so everything appended since —
        # tool results, the images among them — was invisible to the warning,
        # and a turn could cross the threshold between two consecutive checks
        # without either of them saying so.
        total_tokens = token_count_with_estimation(messages)
        warning = get_token_warning(total_tokens)
        if warning:
            print(f"\n{warning}")

    except Exception as e:
        log_error(e)
        print(f"\nError during query: {e}")
        _notify_turn_end(_turn_started, '', cwd=os.getcwd(), error=str(e),
                         tools=_tool_counts,
                         remote_origin=_remote_origin, progress=_progress,
                         prompt=user_input, messages=messages)
        if debug:
            import traceback
            traceback.print_exc()
    finally:
        spinner.stop()  # Ensure spinner is always stopped (including KeyboardInterrupt)
        _stop_escape_monitor(monitor_state)
        if _sequential:
            # Interactive mode must NOT flush — stdin belongs to the persistent
            # PromptSession and a flush would eat input typed during the turn.
            _flush_stdin()


def print_help(tools: Optional[List[Any]] = None):
    """Print help information. The tool list shows the SESSION's active
    tools when given (conditional tools like Skill may be disabled), falling
    back to the registry — never a hardcoded list, which drifted before."""
    if tools is None:
        from .tools import ALL_TOOLS
        tools = ALL_TOOLS
    tool_names = '\n'.join(f"  {t.name}" for t in tools)
    print(f"""
{PRODUCT_NAME} Commands (slash-only; bare words go to the model):
  /exit, /quit  - Exit the program (Ctrl+D also works)
  /clear        - Clear conversation history (starts a new session)
  /resume       - Pick a stored session and continue it (also: --resume <id>)
  /compact      - Summarize the conversation to free context
  /remote       - /remote [on|off]  (remote channel status/toggle, alias /rc)
  /dsp          - /dsp on|off|status  (toggle skipping tool approvals)
  /effort       - /effort low|medium|high|max|off|status  (reasoning effort; also /reasoning, /think)
  /goal         - /goal <objective> | show | clear  (persistent goal injected each turn)
  /goal gate    - /goal gate add <cmd> | list | rm <n> | clear | auto on|off   (completion gates)
  /goal check   - run the completion gates and report pass/fail
  /keepgoing    - /keepgoing on|off|status
  /language     - UI language: /language (show) or /language Korean|ko|한국어
  /expand       - Print the last 3 tool outputs in full into the chat
  /expand N     - Same, but the last N outputs
  /export       - Save the conversation to a Markdown file (./codemason-export-<id>.md)
  /export PATH  - Same, to the given path
  /remote       - Phone/remote status: what's connected and what it can do
                  (also /remote-control, /rc)
  /help, /?     - Show this help message
  ! <command>   - Run a shell command directly (output shared with the model)
  Ctrl+O        - View ALL recent tool outputs in the pager (newest first)

Pager keys (Ctrl+O):
  Space   - page down
  b       - page up
  d / u   - half page down / up
  j / k   - one line down / up
  g / G   - jump to top / bottom
  /text   - search down
  ?text   - search up
  n / N   - next / previous match
  q       - quit

Available Tools:
{tool_names}
""")


# === Non-Interactive Mode ===

# (run_single_query was superseded by print_runner.run_print, which implements
# the full text/json/stream-json output contract.)


# === Setup ===

def setup(cwd: str, dangerously_skip_permissions: bool = False):
    """Setup the CLI environment"""
    set_original_cwd(cwd)
    set_cwd(cwd)

    # Reflect the -dsp flag into the config so get_system_prompt() can raise the
    # autonomous execution layer (execute system changes without confirmation).
    # Without this the flag only reaches the permission gate, not the prompt.
    #
    # As a runtime override, so it cannot be written to disk by an unrelated save
    # later in the run. This one matters more than the others: the value is
    # already resolved from --no-dsp > -dsp > config, so persisting it would fold
    # a single run's flag into the stored setting — and in the direction that
    # turns approvals off for every session after it. /dsp on|off is the way to
    # change it for real, and clears the shadow when it does.
    from .utils.config import set_runtime_override as _flag_override
    _flag_override('dangerously_skip_permissions', dangerously_skip_permissions)

    # Security check for --dangerously-skip-permissions
    if dangerously_skip_permissions:
        # os.getuid() is POSIX-only; native Windows has no such attribute. Upstream
        # spells this `process.getuid?.() === 0`, i.e. "not root" when the platform
        # has no uid concept — mirror that instead of crashing with AttributeError.
        getuid = getattr(os, 'getuid', None)
        if getuid is not None and getuid() == 0:
            print("Error: --dangerously-skip-permissions cannot be used with root/sudo privileges", file=sys.stderr)
            sys.exit(1)


# === Main Entry Point ===

# Depth head-room for query()'s recursion, and the C stack to back it.
#
# query() recurses once per tool-calling round: each batch of tool results
# re-enters it with the grown history, and the recovery paths add levels on top.
# Every level stays live — the outer frames are suspended at `async for msg in
# query(...)`, not returned — so depth tracks the number of tool rounds in a
# turn, which on a long unattended run is hundreds.
#
# CPython's default 1000 is not room for that, and raising the limit alone only
# moves the failure from a catchable RecursionError to a segfault, because the
# real constraint is the thread's C stack. Both have to move together, so main()
# hands the work to a thread created with a large one.
#
# The Rust port hit the same wall from the other side (2026-09-06, 271k tokens
# of context: "thread 'main' has overflowed its stack", process aborted with no
# crash handler and no terminal restore) and takes the same measure in main.rs.
#
# The stack is reserved address space, not committed memory: pages are faulted
# in only as it grows, so a short session pays nothing. This raises the ceiling
# rather than removing it; driving the tool loop iteratively is the real fix and
# is a restructuring of query().
_RECURSION_LIMIT = 20000
_MAIN_STACK_BYTES = 256 * 1024 * 1024


def main():
    """Main entry point"""
    # Re-enter on a thread with a stack sized for query()'s recursion (see
    # above). Guarded by a flag so the re-entry runs the real body exactly once.
    import sys as _sys
    import threading as _threading
    if not getattr(main, '_on_big_stack', False):
        main._on_big_stack = True
        _sys.setrecursionlimit(_RECURSION_LIMIT)
        try:
            _threading.stack_size(_MAIN_STACK_BYTES)
        except (ValueError, RuntimeError):
            pass  # platform refused the size; carry on with the default
        _rc = {}

        def _runner():
            try:
                main()
            except SystemExit as e:      # argparse and the early-exit paths
                _rc['code'] = e.code
            except BaseException:        # noqa: BLE001 - re-raised on the main thread
                import traceback
                traceback.print_exc()
                _rc['code'] = 1

        t = _threading.Thread(target=_runner, name='codemason-main')
        t.start()
        t.join()
        raise SystemExit(_rc.get('code', 0))

    # First of all: make a crash leave its cause on disk. stderr in an overnight
    # run is scrollback that may be gone by morning, and the run being lost does
    # not mean the reason has to be.
    from .utils.log import install_crash_logger, run_with_crash_logger  # noqa: F401
    install_crash_logger()
    globals()['run_with_crash_logger'] = run_with_crash_logger

    # Before argparse: its help text contains an em dash, which a legacy Windows
    # codepage cannot encode once stdout is redirected.
    ensure_utf8_stdio()

    # `codemason mcp ...` / `codemason skill ...` — management subcommands.
    # Intercepted before argparse, which owns the positional `prompt` (a
    # subparser would collide with it).
    if len(sys.argv) >= 2 and sys.argv[1] == 'mcp':
        from .utils.mcp_cli import handle_mcp_cli
        sys.exit(handle_mcp_cli(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == 'skill':
        from .utils.skill_cli import handle_skill_cli
        sys.exit(handle_skill_cli(sys.argv[2:]))

    # The self-wake surface: who am I, wake a session, push an alert. Same
    # interception point and the same reason — these answer and exit, they
    # never open a session.
    if len(sys.argv) >= 2 and sys.argv[1] == 'whoami':
        from .utils.wake_cli import handle_whoami_cli
        sys.exit(handle_whoami_cli(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == 'inject':
        from .utils.wake_cli import handle_inject_cli
        sys.exit(handle_inject_cli(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == 'notify':
        from .utils.wake_cli import handle_notify_cli
        sys.exit(handle_notify_cli(sys.argv[2:]))

    # Before any child spawn: give git-bash/MSYS2 a UTF-8 charset so it doesn't
    # mangle non-ASCII (e.g. Korean) paths in tool output. No-op off Windows.
    ensure_utf8_locale()

    parser = argparse.ArgumentParser(
        description=f"{PRODUCT_NAME} - AI coding assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'prompt',
        nargs='?',
        default='',
        help='Initial prompt (optional)'
    )
    parser.add_argument(
        '-C', '--cwd',
        default=os.getcwd(),
        help='Working directory (git-style -C; -c now means --continue)'
    )
    parser.add_argument(
        '-p', '--print',
        action='store_true',
        dest='print_mode',
        help='Print response and exit (non-interactive)'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--output-format',
        default='text',
        choices=['text', 'json', 'stream-json'],
        dest='output_format',
        help="Print-mode output format: text (default), json (single result "
             "object), or stream-json (NDJSON: init, each message, result). "
             "stream-json requires --verbose."
    )
    parser.add_argument(
        '--input-format',
        default='text',
        choices=['text', 'stream-json'],
        dest='input_format',
        help="Print-mode input format: text (default) or stream-json (stdin is "
             "NDJSON of user messages)."
    )
    parser.add_argument(
        '-dsp', '--dangerously-skip-permissions',
        action='store_true',
        help='Skip permission checks (-dsp is shorthand)'
    )
    parser.add_argument(
        '--allowed-tools',
        default='',
        dest='allowed_tools',
        metavar='NAMES',
        help="Offer only these tools (comma-separated, e.g. Bash,Grep,Glob,Read). "
             "Anything else is never shown to the model, so it cannot be called."
    )
    parser.add_argument(
        '--disallowed-tools',
        default='',
        dest='disallowed_tools',
        metavar='NAMES',
        help="Remove these tools even if --allowed-tools lists them."
    )
    parser.add_argument(
        '-np', '--no-plan',
        dest='no_plan',
        action='store_true',
        help="Drop the plan-mode tools (EnterPlanMode/ExitPlanMode) from this "
             "run. Their descriptions are ~1,400 tokens of the fixed prefix "
             "that is resent every turn; a session that never plans pays it "
             "for nothing."
    )
    parser.add_argument(
        '--no-remote',
        action='store_true',
        help='Turn the remote (phone) channel off for this run only — alerts, '
             'approvals and questions. Use /remote off to make it stick.'
    )
    parser.add_argument(
        '--no-dsp', '--ask',
        dest='no_dsp',
        action='store_true',
        help='Ask for tool approvals even if config enables skipping (for -p; '
             'in the REPL use /dsp off)'
    )
    parser.add_argument(
        '-r', '--resume',
        default=None,
        metavar='SESSION_ID',
        dest='resume',
        help='Resume a stored session by id and continue it (works with and '
             'without -p). In the REPL, /resume opens an interactive picker.'
    )
    parser.add_argument(
        '-c', '--continue',
        action='store_true',
        dest='continue_session',
        help='Continue the most recently modified session of this directory '
             '(works with and without -p).'
    )
    parser.add_argument(
        '--plain', '--ascii',
        action='store_true',
        dest='plain',
        help='Pure-ASCII compatibility mode: no colour, no markdown rendering, '
             'no emoji/braille/box-drawing (for dumb/old VT100/Hercules consoles)'
    )
    parser.add_argument(
        '-rt', '--realtime',
        action='store_true',
        dest='realtime',
        help='Stream the model reply to the screen token-by-token as it '
             'generates (instead of buffering the whole turn and printing at the '
             'end). DSML tool markup, <think> blocks and <|special|> tokens are '
             'held back / stripped safely; text is streamed plain (no live '
             'markdown). A long generation no longer looks frozen. On by '
             'default; pass --no-rt to buffer the whole turn instead.'
    )
    parser.add_argument(
        '--no-rt', '--no-realtime',
        dest='no_realtime',
        action='store_true',
        help='Buffer the whole turn and print it at the end, instead of '
             'streaming it live (the opposite of -rt, which is the default)'
    )
    parser.add_argument(
        '-dbg', '--debug-log',
        action='store_true',
        dest='debug_log',
        help='Full logging to /tmp/codemason-dbg-<pid>/ (turns.log: one line '
             'per request with prefill/think/output; debug.log: full trace). '
             'Append-only, safe to tail while the session runs. Also shows the '
             'live prefill readout on the spinner.'
    )
    parser.add_argument(
        '--theme',
        default=None,
        metavar='NAME',
        help="Repaint the terminal for this session via OSC (opt-in). "
             "'homebrew' = green-on-black; restored on exit. Or set CODEMASON_THEME."
    )
    parser.add_argument(
        '--keep-going',
        dest='keep_going',
        action='store_true',
        default=None,
        help="Keep the model working past natural turn-ends until the task is "
             "done (via the installed Stop hook). Overrides config for this "
             "session. Default: on."
    )
    parser.add_argument(
        '--no-keep-going',
        dest='keep_going',
        action='store_false',
        help="Disable keep-going for this session — the model stops as soon as "
             "it stops calling tools (the Stop hook becomes a no-op)."
    )
    parser.add_argument(
        '--directive',
        metavar='TEXT',
        help="Operator directive pinned to the END of the system prompt, so it "
             "outranks the built-in sections. Set once at startup and kept for "
             "the session (it is part of the cached prefix). Also readable from "
             "CODEMASON_DIRECTIVE."
    )
    parser.add_argument(
        '--directive-file',
        metavar='PATH',
        help="Read the operator directive from a file instead of the argument."
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'{PRODUCT_NAME} {VERSION}'
    )

    args = parser.parse_args()

    # 도구 제한. 도구 목록이 만들어지기 전에 걸어야 한다 — 허용되지 않은
    # 도구는 보여주고 거절하는 것이 아니라 아예 없는 것이 되어야 한다.
    #
    # -np/--no-plan 은 그 위에 얹는 설탕이다. 플랜 도구를 따로 끄는 길을
    # 새로 파지 않고 --disallowed-tools 와 같은 통로로 보낸다 — 걸러내는
    # 자리가 하나면 감사할 곳도 하나다.
    _disallowed = getattr(args, 'disallowed_tools', '')
    if getattr(args, 'no_plan', False):
        from .tools.plan_mode_tools import (
            ENTER_PLAN_MODE_TOOL_NAME, EXIT_PLAN_MODE_TOOL_NAME,
        )
        _disallowed = ','.join(
            x for x in (_disallowed,
                        f'{ENTER_PLAN_MODE_TOOL_NAME},{EXIT_PLAN_MODE_TOOL_NAME}')
            if x
        )
    if getattr(args, 'allowed_tools', '') or _disallowed:
        from .tools import set_tool_filter
        set_tool_filter(args.allowed_tools, _disallowed)

    # Operator directive -> the env the prompt assembly reads. Set before any
    # prompt is built. A flag beats an inherited CODEMASON_DIRECTIVE, so a bot
    # pool can set a default in the environment and a single run can override it.
    if getattr(args, 'directive_file', None):
        try:
            with open(args.directive_file, encoding='utf-8') as _f:
                os.environ['CODEMASON_DIRECTIVE'] = _f.read()
        except OSError as e:
            parser.error(f"--directive-file {args.directive_file}: {e}")
    if getattr(args, 'directive', None):
        os.environ['CODEMASON_DIRECTIVE'] = args.directive

    # keep-going session override -> the env the Stop hook reads. Left unset when
    # neither flag is given, so the hook falls back to config.json keep_going.
    if args.keep_going is not None:
        os.environ['CODEMASON_KEEP_GOING'] = '1' if args.keep_going else '0'

    # Pure-ASCII mode: set the env the formatting layer reads (plain_mode()), so
    # every output path — colour, markdown, spinner, symbols — degrades to ASCII.
    if args.plain:
        os.environ['CODEMASON_PLAIN'] = '1'

    # -rt/--realtime: stream the reply live (see StreamingSpinner / query loop).
    # Streaming is the default, so the flag that actually decides anything is
    # --no-rt. Resolution mirrors the permission flags: --no-rt > -rt > config,
    # and neither flag present leaves the stored setting alone. Runtime only —
    # a flag passed for one run is not written back (see config._env_shadow for
    # the same rule applied to the environment).
    # -dbg: turn full logging on and put it where it can be watched. Set
    # BEFORE anything logs — LOG_DIR and the debug gate are read at import.
    if getattr(args, 'debug_log', False):
        import pathlib
        import tempfile
        # gettempdir(), not a literal '/tmp': on native Windows that literal
        # resolves to C:\tmp at the drive root, which is not the temp directory
        # and is not always writable. Same resolution the task output files
        # already use (utils/task/disk_output).
        _d = pathlib.Path(tempfile.gettempdir()) / ('codemason-dbg-%d' % os.getpid())
        _d.mkdir(parents=True, exist_ok=True)
        os.environ['CODEMASON_DEBUG'] = '1'
        os.environ['CODEMASON_LOG_DIR'] = str(_d)
        import codemason.utils.config as _cfgmod
        _cfgmod.LOG_DIR = _d
        print('  %s debug logging -> %s' % (sym('connector'), _d))
        print('     %s/turns.log   one line per request (prefill / think / output)' % _d)
        print('     %s/debug.log   full trace' % _d)

    from .utils.config import set_runtime_override as _rt_override
    if getattr(args, 'no_realtime', False):
        _rt_override('realtime', False)
    elif getattr(args, 'realtime', False):
        _rt_override('realtime', True)

    # Optional terminal theme (OSC repaint, opt-in, restored on exit). Applied
    # after plain-mode is set so it correctly no-ops in pure-ASCII mode.
    theme = args.theme or os.getenv('CODEMASON_THEME')
    if theme:
        from .utils.terminal_theme import apply_theme
        apply_theme(theme)

    # The effective value for skipping approvals: the file is the default and the
    # -dsp flag overrides it.
    #
    # The wiring used to run one way only — the flag was written into the config
    # so the system prompt could raise its autonomous layer — so on startup the
    # flag's default (False) overwrote whatever the file said, and the permission
    # gate, which reads the value from the tool-use context, never saw it. The
    # setting was present, documented and inert. For it to mean anything without
    # typing -dsp every time, the file has to be the default.
    _dsp = (not getattr(args, 'no_dsp', False)) and (
        args.dangerously_skip_permissions
        or bool(getattr(get_config(), 'dangerously_skip_permissions', False)))

    # --no-remote applies to this run only, so it goes in the environment rather
    # than the file (off_reason checks it first). /remote off is the durable one.
    if getattr(args, 'no_remote', False):
        os.environ['CODEMASON_REMOTE'] = '0'

    # Setup
    setup(args.cwd, _dsp)

    # -c/--continue = resume the most recently modified session of this
    # directory. Resolved to a session id here so both the REPL and -p paths
    # reuse the --resume plumbing unchanged. An explicit --resume <id> wins.
    if args.continue_session and not args.resume:
        from .utils.session_storage import list_sessions, cleared_after_session
        _recent = list_sessions(limit=1)
        if not _recent:
            print("Error: --continue: no stored sessions for this directory",
                  file=sys.stderr)
            sys.exit(1)
        # Honor a /clear: if the last real session predates a deliberate clear,
        # start fresh instead of silently resuming the pre-clear conversation.
        if cleared_after_session(_recent[0]['session_id']):
            print("Starting a fresh session (the last one was cleared with /clear).")
        else:
            args.resume = _recent[0]['session_id']

    # Get tools
    async def get_tools_async():
        return await get_all_tools()

    tools = run_with_crash_logger(get_tools_async())

    # Run
    if args.print_mode:
        from .print_runner import run_print, read_prompt_from_input

        # stream-json output requires --verbose (cli/print.ts:787-790).
        if args.output_format == 'stream-json' and not args.verbose:
            print("Error: When using --print, --output-format=stream-json "
                  "requires --verbose", file=sys.stderr)
            sys.exit(1)

        prompt = read_prompt_from_input(args.prompt, args.input_format)
        if not prompt:
            print("Error: Input must be provided either through stdin or as a "
                  "prompt argument when using --print", file=sys.stderr)
            sys.exit(1)

        exit_code = run_with_crash_logger(run_print(
            prompt,
            tools,
            output_format=args.output_format,
            verbose=args.verbose,
            dangerously_skip_permissions=_dsp,
            resume_session_id=args.resume,
        ))
        sys.exit(exit_code)

    else:
        # Interactive REPL — --resume <id> restores the stored conversation and
        # continues appending to the same transcript file.
        initial_messages = None
        if args.resume:
            from .utils import session_storage
            try:
                initial_messages = session_storage.load_conversation_for_resume(args.resume)
                session_storage.adopt_resumed_session(args.resume)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: --resume {args.resume}: {e}", file=sys.stderr)
                sys.exit(1)
        try:
            run_with_crash_logger(repl_loop(
                tools=tools,
                initial_prompt=args.prompt,
                dangerously_skip_permissions=_dsp,
                verbose=args.verbose,
                debug=args.debug,
                initial_messages=initial_messages,
            ))
        except KeyboardInterrupt:
            print("\nGoodbye!")


if __name__ == '__main__':
    main()
