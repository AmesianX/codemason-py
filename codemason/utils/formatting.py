"""
Formatting utilities for CodeMason CLI output.
Extracted from cli.py to reduce file size.
"""

import os
import re
import sys
import difflib
import shutil
import unicodedata

# Unified-diff hunk header: @@ -oldStart[,oldCount] +newStart[,newCount] @@.
# Used to re-anchor the running line counters at each hunk (see format_edit_diff).
_DIFF_HUNK_RE = re.compile(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')
from typing import Any


# === ANSI Color Support ===

def plain_mode() -> bool:
    """Pure-ASCII compatibility mode.

    When on, output degrades to 7-bit ASCII so it stays legible on ancient/dumb
    terminals (Hercules monochrome, bare VT100, dumb log capture): no ANSI colour
    or escape sequences, no markdown rendering, no emoji/braille/box-drawing —
    everything is plain printable ASCII. The pretty (default) mode is unaffected.

    Toggled by CODEMASON_PLAIN / CODEMASON_ASCII env vars; the --plain/--ascii CLI
    flag sets CODEMASON_PLAIN early in main()."""
    v = os.getenv('CODEMASON_PLAIN') or os.getenv('CODEMASON_ASCII')
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on') if v else False


# Output glyphs: (pretty unicode, plain ASCII). sym() picks per plain_mode().
_SYMBOLS = {
    'connector': ('L', 'L'),     # tool-result branch connector (looks like └, 1-cell ASCII everywhere)
    'warn':      ('⚠', '!'),
    'bolt':      ('⚡', '*'),
    'bar_full':  ('█', '#'),
    'bar_empty': ('░', '-'),
    'rule':      ('⎯', '-'),
    'dash':      ('—', '-'),      # em dash
    'dot':       ('·', '-'),      # middle dot separator
    'ellipsis':  ('…', '...'),
    'arrow':     ('→', '->'),
    'up':        ('↑', '^'),
    'down':      ('↓', 'v'),
    'think':     ('✻', '*'),      # reasoning/thinking token marker (spinner)
}


def sym(name: str) -> str:
    """Return a UI glyph as pretty unicode, or plain ASCII in plain_mode()."""
    pair = _SYMBOLS.get(name)
    if not pair:
        return ''
    return pair[1] if plain_mode() else pair[0]


def _ambiguous_is_wide() -> bool:
    """In CJK locales most terminals render East-Asian 'Ambiguous'-width chars as
    2 cells. Detect that from the locale env vars.

    Returning on the FIRST *set* variable is intentional and POSIX-correct: the
    ctype precedence is LC_ALL > LC_CTYPE > LANG, so a set LC_ALL is
    authoritative even when its value is non-CJK (e.g. LC_ALL=C must NOT be
    overridden by a CJK LANG). Do not 'fall through' to later vars."""
    for var in ('LC_ALL', 'LC_CTYPE', 'LANG'):
        v = os.getenv(var, '')
        if v:
            vl = v.lower()
            return any(t in vl for t in ('ko', 'ja', 'zh', 'cjk'))
    return False


def display_width(s: str) -> int:
    """Best-effort terminal cell width of a string: wide (W/F) chars count 2,
    East-Asian ambiguous (A) count 2 in CJK locales else 1, zero-width combining
    marks count 0. Used to align continuation lines under a variable-width prefix
    (e.g. the tool-result connector, which is 1 cell as '⎿' but 2 as plain '\\_')."""
    amb = 2 if _ambiguous_is_wide() else 1
    width = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        ea = unicodedata.east_asian_width(ch)
        if ea in ('W', 'F'):
            width += 2
        elif ea == 'A':
            width += amb
        else:
            width += 1
    return width


def clip_to_width(s: str, width: int) -> str:
    """Cut a string so it occupies at most `width` terminal columns.

    Counts in columns, not characters — Hangul and the arrow/box glyphs the
    spinner uses are two cells each. ANSI escape sequences are copied through
    without being counted, and any style left open is closed with a reset so a
    cut in the middle of a coloured run cannot bleed into the rest of the line.
    """
    out = []
    used = 0
    styled = False
    it = iter(range(len(s)))
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\x1b':
            out.append(ch)
            styled = True
            i += 1
            while i < len(s):
                out.append(s[i])
                if s[i].isalpha():
                    i += 1
                    break
                i += 1
            continue
        w = display_width(ch)
        if used + w > width:
            if styled:
                out.append('\x1b[0m')
            return ''.join(out)
        out.append(ch)
        used += w
        i += 1
    return ''.join(out)


# How long a one-line tool summary may be (constants/toolLimits.ts
# TOOL_SUMMARY_MAX_LENGTH). Every tool's get_activity_description truncates to
# this, so a pasted 4KB regex cannot push the spinner off the screen.
TOOL_SUMMARY_MAX_LENGTH = 50


# The ellipsis costs what THIS process says it costs. The original reserves a
# hardcoded 1 column (`maxWidth - 1`) because its width function — the
# `string-width` package — calls U+2026 one cell wide everywhere. codemason's
# display_width is locale-aware, and '…' is East-Asian *ambiguous*: two cells
# wherever ambiguous counts wide. Reserving 1 there returned max_width + 1
# columns and broke the one promise this function makes.
def _ellipsis_width() -> int:
    return max(1, display_width('…'))


def truncate(text: str, max_width: int, single_line: bool = False) -> str:
    """Cut `text` to `max_width` columns, marking the cut (utils/truncate.ts).

    Width, not characters — the same reason clip_to_width counts columns. The
    ellipsis replaces the last column rather than being added past the limit, so
    the result never exceeds `max_width`.
    """
    if text is None:
        return ''
    result = str(text)
    reserve = _ellipsis_width()
    if single_line:
        newline = result.find('\n')
        if newline != -1:
            result = result[:newline]
            if display_width(result) + reserve > max_width:
                return clip_to_width(result, max(0, max_width - reserve)) + '…'
            return result + '…'
    if display_width(result) <= max_width:
        return result
    return clip_to_width(result, max(0, max_width - reserve)) + '…'


def terminal_columns(default: int = 80) -> int:
    """Usable columns for a one-line \r repaint.

    Read fresh at every use rather than cached: the window can be resized while
    a turn runs, which is exactly when the stale-line bug showed up. One column
    is held back because some terminals wrap as soon as the last cell is filled
    rather than when the next character arrives.
    """
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
    except Exception:
        cols = default
    return max(20, cols - 1)


def supports_color() -> bool:
    """Check if terminal supports colors"""
    if plain_mode():
        return False  # pure-ASCII: no ANSI escapes at all
    if os.getenv('NO_COLOR'):
        return False
    # FORCE_COLOR is checked BEFORE the isatty gate so it can force color on a
    # pipe/redirect (its whole purpose) — previously the non-tty return above it
    # made FORCE_COLOR a no-op off a terminal.
    if os.getenv('FORCE_COLOR'):
        return True
    if not sys.stdout.isatty():
        return False
    return True


class Colors:
    RESET = '\033[0m'
    GRAY = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    # Background colors for diff display
    BG_GREEN = '\033[48;5;22m'   # Dark green background
    BG_RED = '\033[48;5;52m'     # Dark red background


def colorize(text: str, color: str) -> str:
    """Apply color if supported"""
    if supports_color():
        return f"{color}{text}{Colors.RESET}"
    return text


def hex_to_ansi256(hexstr: str) -> int:
    """Nearest xterm-256 index for a #rrggbb colour. ABSOLUTE 256 (ESC[38;5;Nm)
    bypasses the terminal's palette remap — named ANSI colours collapse to the
    scheme's colour (e.g. everything -> green on macOS 'Homebrew'), 256 does not.
    See _build_md_renderer for the same reasoning."""
    h = hexstr.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if r == g == b:                                  # grayscale ramp (232-255)
        if r < 8:
            return 16
        if r > 248:
            return 231
        return round((r - 8) / 247 * 24) + 232
    levels = (0, 95, 135, 175, 215, 255)             # 6x6x6 colour cube
    def lvl(v):
        return min(range(6), key=lambda i: abs(levels[i] - v))
    return 16 + 36 * lvl(r) + 6 * lvl(g) + lvl(b)


def fg256_hex(hexstr: str) -> str:
    """SGR escape setting foreground to the nearest 256-colour for a hex value."""
    return f'\033[38;5;{hex_to_ansi256(hexstr)}m'


def format_number(n: int) -> str:
    """Format number with comma separators"""
    return f"{n:,}"


def safe_print(text: str) -> None:
    """Print text safely, handling encoding errors"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('utf-8', errors='replace').decode('utf-8'))


# Escape sequences + control bytes that corrupt the terminal when RAW tool output
# is echoed to the console (binary dumps, compiled AXML/DEX, progress bars, stray
# cursor/mode escapes). Applied ONLY to untrusted tool-result text before display
# — never to our own colorized output, whose ESC codes are intentional.
_RE_ANSI_CSI = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')          # CSI: colors, cursor
_RE_ANSI_OSC = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)')  # OSC: title, etc.
_RE_ANSI_OTHER = re.compile(r'\x1b[@-Z\\-_]')                    # other 2-byte escapes
_RE_CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')  # C0/C1 minus \t\n\r


def sanitize_terminal_text(text: str) -> str:
    """Strip terminal-corrupting escape sequences and control bytes from
    untrusted text, keeping \\t \\n \\r and all printable characters. Use on tool
    output before printing it to the console; a package parser dumping a binary
    XML blob (NUL, \\x03, ...) must not be able to wreck the user's terminal."""
    if not text:
        return text
    text = _RE_ANSI_CSI.sub('', text)
    text = _RE_ANSI_OSC.sub('', text)
    text = _RE_ANSI_OTHER.sub('', text)
    text = _RE_CTRL.sub('', text)
    return text


_color_system_cache = None


def _detect_color_system() -> str:
    """Pick the color depth the terminal actually advertises (truecolor only if
    COLORTERM says so; 256 for *-256color; else 16-color 'standard'). Memoized —
    the env doesn't change within a session, and render_markdown runs this per
    rendered block."""
    global _color_system_cache
    if _color_system_cache is not None:
        return _color_system_cache
    import os
    if os.environ.get('COLORTERM', '').lower() in ('truecolor', '24bit'):
        _color_system_cache = 'truecolor'
    else:
        term = os.environ.get('TERM', '')
        _color_system_cache = '256' if ('256color' in term or '256' in term) else 'standard'
    return _color_system_cache


_md_renderer_cache = None  # (MarkdownClass, Theme) built once


def _build_md_renderer():
    """Build (and cache) the custom Markdown class + theme.

    Two deliberate choices, both driven by real terminal behaviour:

    1. ABSOLUTE 256-COLOURS, not named ANSI. Named colours like 'bright_cyan'
       emit an ANSI SGR code (ESC[96m) that the terminal remaps through its OWN
       palette — so on a green-on-black scheme (e.g. macOS "Homebrew") every
       accent collapses to green and nothing stands out. Specifying colours as
       explicit 256-cube indices (color(75) -> ESC[38;5;75m) bypasses the
       palette: the colour shows as itself regardless of the terminal theme.
       (This is also why code blocks already looked right — Pygments emits
       absolute 256 colours.)

    2. MUTED but DISTINCT palette + bold carried by a warm colour, so '**bold**'
       is visibly different from body text even when bold weight isn't rendered.
       H1 is left-aligned (rich centres it by default, which reads oddly in a CLI).
    """
    global _md_renderer_cache
    if _md_renderer_cache is not None:
        return _md_renderer_cache
    from rich.markdown import Markdown, Heading
    from rich.theme import Theme

    class _LeftHeading(Heading):
        # rich centres h1; left-align it like every other level. Guard for older
        # `rich` (e.g. a system-python install) that lacks Heading.LEVEL_ALIGN —
        # without the guard the AttributeError propagates out of _build_md_renderer
        # and render_markdown silently drops ALL markdown to raw text. With it,
        # old rich still renders markdown; only h1 stays rich's default (centred).
        if hasattr(Heading, 'LEVEL_ALIGN'):
            LEVEL_ALIGN = {**Heading.LEVEL_ALIGN, 'h1': 'left'}

    class _CmMarkdown(Markdown):
        elements = {**Markdown.elements, 'heading_open': _LeftHeading}

    theme = Theme({
        # headings: muted, distinct per level, theme-independent (absolute 256)
        'markdown.h1': 'bold color(75)',    # soft sky blue
        'markdown.h2': 'bold color(214)',   # bright orange
        'markdown.h3': 'bold color(220)',   # yellow
        'markdown.h4': 'bold color(245)',   # grey
        'markdown.h5': 'bold color(245)',
        'markdown.h6': 'bold color(245)',
        'markdown.h1.border': 'none',
        # **bold** -> bold + warm colour so it pops on any background
        'markdown.strong': 'bold color(220)',   # yellow
        'markdown.emph': 'italic color(250)',
        'markdown.code': 'color(180)',          # inline code: warm tan
        'markdown.link': 'underline color(75)',
        'markdown.link_url': 'color(240)',
        'markdown.item.bullet': 'bold color(75)',
        'markdown.item.number': 'bold color(75)',
        'markdown.block_quote': 'italic color(108)',
        'markdown.hr': 'color(240)',
    })
    _md_renderer_cache = (_CmMarkdown, theme)
    return _md_renderer_cache


def render_markdown(text: str) -> str:
    """Render markdown to ANSI for the terminal.

    Terminal-substrate equivalent of the original's Ink markdown renderer
    (components/Markdown.tsx: marked tokens -> ANSI). Uses `rich` (marked->rich,
    same feature set: headings, bold/italic, syntax-highlighted code blocks,
    lists, links, blockquotes, tables). Falls back to the raw text when color
    isn't supported (piped / non-tty / plain_mode) so -p/redirected output and
    pure-ASCII mode stay plain.
    """
    if not text or not supports_color():
        return text
    try:
        import shutil
        from io import StringIO
        from rich.console import Console

        markdown_cls, md_theme = _build_md_renderer()
        width = min(shutil.get_terminal_size((100, 24)).columns, 120)
        buf = StringIO()
        # Match the terminal's ACTUAL advertised color depth (we render to a
        # buffer, so rich can't auto-detect — force_terminal disables detection).
        # COLORTERM=truecolor/24bit -> truecolor; TERM=*256color* -> 256; else 16.
        console = Console(file=buf, force_terminal=True, color_system=_detect_color_system(),
                          width=width, highlight=False, theme=md_theme, emoji=False)
        console.print(markdown_cls(text, code_theme='monokai', inline_code_theme='monokai'))
        return buf.getvalue().rstrip('\n')
    except Exception:
        return text


# === Diff Formatting ===

def format_edit_diff(file_path: str, old_string: str, new_string: str) -> str:
    """Format an edit operation as a colored diff display, Claude Code style.

    Returns a formatted string with:
    - Summary line: "Updated file.py with N additions and M removals"
    - Colored diff with line numbers
    """
    filename = os.path.basename(file_path)
    old_lines = old_string.split('\n')
    new_lines = new_string.split('\n')

    # Count additions and removals
    num_additions = 0
    num_removals = 0

    diff_lines = list(difflib.unified_diff(old_lines, new_lines, n=3, lineterm=''))

    for line in diff_lines:
        if line.startswith('+') and not line.startswith('+++'):
            num_additions += 1
        elif line.startswith('-') and not line.startswith('---'):
            num_removals += 1

    # Build summary
    parts = []
    if num_additions > 0:
        parts.append(f"{num_additions} addition{'s' if num_additions != 1 else ''}")
    if num_removals > 0:
        parts.append(f"{num_removals} removal{'s' if num_removals != 1 else ''}")
    summary = ' and '.join(parts) if parts else 'no changes'

    result = [f"  {sym('connector')} Updated {colorize(filename, Colors.BOLD)} with {summary}"]

    # Find start line from the file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            file_lines = f.readlines()
        file_content = ''.join(file_lines)
        new_pos = file_content.find(new_string)
        if new_pos >= 0:
            start_line = file_content[:new_pos].count('\n') + 1
        else:
            start_line = 1
    except Exception:
        start_line = 1

    has_color = supports_color()
    diff_output = []

    hunk_lines = [l for l in diff_lines if not l.startswith('---') and not l.startswith('+++')]

    old_line = start_line
    new_line = start_line

    max_line_num = start_line + len(old_lines) + len(new_lines)
    line_num_width = len(str(max_line_num))

    for line in hunk_lines:
        if line.startswith('@@'):
            # Re-anchor the counters to THIS hunk's start offsets. Without this a
            # multi-hunk diff (old/new_string sharing >6 unchanged interior lines)
            # kept counting sequentially across the omitted gap, so every line
            # number in the 2nd+ hunk was too low by the gap size.
            m = _DIFF_HUNK_RE.match(line)
            if m:
                old_line = start_line + int(m.group(1)) - 1
                new_line = start_line + int(m.group(2)) - 1
            continue
        elif line.startswith('-'):
            content = line[1:]
            ln = str(old_line).rjust(line_num_width)
            if has_color:
                diff_output.append(f"     {Colors.DIM}{ln}{Colors.RESET} {Colors.BG_RED}{Colors.RED}-{content}{Colors.RESET}")
            else:
                diff_output.append(f"     {ln} -{content}")
            old_line += 1
        elif line.startswith('+'):
            content = line[1:]
            ln = str(new_line).rjust(line_num_width)
            if has_color:
                diff_output.append(f"     {Colors.DIM}{ln}{Colors.RESET} {Colors.BG_GREEN}{Colors.GREEN}+{content}{Colors.RESET}")
            else:
                diff_output.append(f"     {ln} +{content}")
            new_line += 1
        elif line.startswith(' '):
            content = line[1:]
            ln = str(new_line).rjust(line_num_width)
            if has_color:
                diff_output.append(f"     {Colors.DIM}{ln}{Colors.RESET}  {content}")
            else:
                diff_output.append(f"     {ln}  {content}")
            old_line += 1
            new_line += 1

    if diff_output:
        result.append('')
        result.extend(diff_output)

    return '\n'.join(result)


def format_edit_diff_preview(file_path: str, old_string: str, new_string: str) -> str:
    """Format diff preview BEFORE edit is applied (for permission request).

    Unlike format_edit_diff(), this reads from the ORIGINAL file to find line numbers.
    """
    filename = os.path.basename(file_path)
    old_lines = old_string.split('\n')
    new_lines = new_string.split('\n')

    diff_lines = list(difflib.unified_diff(old_lines, new_lines, n=3, lineterm=''))

    num_additions = sum(1 for l in diff_lines if l.startswith('+') and not l.startswith('+++'))
    num_removals = sum(1 for l in diff_lines if l.startswith('-') and not l.startswith('---'))

    parts = []
    if num_additions > 0:
        parts.append(f"{num_additions} addition{'s' if num_additions != 1 else ''}")
    if num_removals > 0:
        parts.append(f"{num_removals} removal{'s' if num_removals != 1 else ''}")
    summary = ' and '.join(parts) if parts else 'no changes'

    result = [f"  {colorize(filename, Colors.DIM)} {sym('dash')} {summary}"]

    start_line = 1
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            file_content = f.read()
        pos = file_content.find(old_string)
        if pos >= 0:
            start_line = file_content[:pos].count('\n') + 1
    except Exception:
        pass

    has_color = supports_color()
    diff_output = []
    hunk_lines = [l for l in diff_lines if not l.startswith('---') and not l.startswith('+++')]

    old_line = start_line
    new_line = start_line
    max_line_num = start_line + len(old_lines) + len(new_lines)
    line_num_width = len(str(max_line_num))

    for line in hunk_lines:
        if line.startswith('@@'):
            # Re-anchor the counters to THIS hunk's start offsets. Without this a
            # multi-hunk diff (old/new_string sharing >6 unchanged interior lines)
            # kept counting sequentially across the omitted gap, so every line
            # number in the 2nd+ hunk was too low by the gap size.
            m = _DIFF_HUNK_RE.match(line)
            if m:
                old_line = start_line + int(m.group(1)) - 1
                new_line = start_line + int(m.group(2)) - 1
            continue
        elif line.startswith('-'):
            content = line[1:]
            ln = str(old_line).rjust(line_num_width)
            if has_color:
                diff_output.append(f"     {Colors.DIM}{ln}{Colors.RESET} {Colors.BG_RED}{Colors.RED}-{content}{Colors.RESET}")
            else:
                diff_output.append(f"     {ln} -{content}")
            old_line += 1
        elif line.startswith('+'):
            content = line[1:]
            ln = str(new_line).rjust(line_num_width)
            if has_color:
                diff_output.append(f"     {Colors.DIM}{ln}{Colors.RESET} {Colors.BG_GREEN}{Colors.GREEN}+{content}{Colors.RESET}")
            else:
                diff_output.append(f"     {ln} +{content}")
            new_line += 1
        elif line.startswith(' '):
            content = line[1:]
            ln = str(new_line).rjust(line_num_width)
            if has_color:
                diff_output.append(f"     {Colors.DIM}{ln}{Colors.RESET}  {content}")
            else:
                diff_output.append(f"     {ln}  {content}")
            old_line += 1
            new_line += 1

    if diff_output:
        result.append('')
        result.extend(diff_output)

    return '\n'.join(result)


# === Tool Display ===

# Every tool renders its own tool-use line (Tool.render_tool_use_message, the port
# of per-tool renderToolUseMessage in tools/<Tool>/UI.tsx). This is only
# the fallback for a tool that defines none: show the argument that identifies the
# call, else a compact key=value line — never a raw Python dict repr, which spills
# escaped newlines and quotes across the terminal.
_PRIMARY_ARG_KEYS = ('command', 'query', 'url', 'pattern', 'file_path',
                     'notebook_path', 'path', 'task_id', 'shell_id', 'skill')


def summarize_tool_input(tool_input: Any, max_len: int = 160) -> str:
    if isinstance(tool_input, dict):
        text = ''
        for key in _PRIMARY_ARG_KEYS:
            if tool_input.get(key):
                text = str(tool_input[key])
                break
        else:
            text = ' '.join(f"{k}={v}" for k, v in tool_input.items()
                            if v not in (None, '', [], {}))
    else:
        text = str(tool_input or '')
    text = ' '.join(text.split())
    return (text[:max_len] + '…') if len(text) > max_len else text


# === Token Display ===

def format_tokens_compact(n: int) -> str:
    """Compact token count: 2600 -> '2.6k', 10000 -> '10k', 262144 -> '262.1k',
    999 -> '999'. Used for all token displays (spinner + footer), matching
    Claude Code's '2.6k tokens' form rather than '10,000 tokens'."""
    if n < 1000:
        return str(n)
    s = f"{n / 1000:.1f}k"
    return (s[:-3] + 'k') if s.endswith('.0k') else s


def format_token_display(usage, duration_ms: int, show_details: bool = False, fallback_tokens: int = 0) -> str:
    """Format token usage for display"""
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_create = 0

    if usage is not None:
        input_tokens = getattr(usage, 'input_tokens', 0) if hasattr(usage, 'input_tokens') else usage.get('input_tokens', 0) if isinstance(usage, dict) else 0
        output_tokens = getattr(usage, 'output_tokens', 0) if hasattr(usage, 'output_tokens') else usage.get('output_tokens', 0) if isinstance(usage, dict) else 0
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) if hasattr(usage, 'cache_read_input_tokens') else usage.get('cache_read_input_tokens', 0) if isinstance(usage, dict) else 0
        cache_create = getattr(usage, 'cache_creation_input_tokens', 0) if hasattr(usage, 'cache_creation_input_tokens') else usage.get('cache_creation_input_tokens', 0) if isinstance(usage, dict) else 0

    total_tokens = input_tokens + output_tokens + cache_read + cache_create

    if total_tokens == 0 and fallback_tokens > 0:
        total_tokens = fallback_tokens

    duration_sec = duration_ms / 1000

    if show_details:
        parts = [f"{sym('down')}{format_tokens_compact(input_tokens)}"]
        if cache_read > 0:
            parts.append(f"(+{format_tokens_compact(cache_read)} cached)")
        parts.append(f"{sym('arrow')} {sym('up')}{format_tokens_compact(output_tokens)}")
        parts.append(f"[{duration_sec:.1f}s]")
        return ' '.join(parts)
    else:
        # Default footer: how full the context window is, compact k form
        # (e.g. "13.3k / 262.1k (5%)"). The lump "N tokens" (input+output) was
        # misleading — a trivial turn looked huge because of the fixed
        # system+tools input. Context fill is the meaningful number.
        #
        # Measured against the AUTO-COMPACT THRESHOLD, not the raw window. That
        # is the number you can act on — at 100% the history is summarised away —
        # and it is the same limit the startup banner announces, so the two
        # agree. Against the raw window the figure reads falsely reassuring: the
        # threshold sits a summary-reserve plus buffer below it, and on a server
        # whose usable window is shorter than its advertised one the gap is far
        # wider than that.
        from .tokens import get_context_size
        try:
            from ..services.compaction import get_auto_compact_threshold
            limit = get_auto_compact_threshold()
        except Exception:  # noqa: BLE001 - display must never break a turn
            limit = get_context_size()
        pct = f" ({round(total_tokens / limit * 100)}%)" if limit > 0 else ''
        return (f"{sym('rule')} {sym('down')} {format_tokens_compact(total_tokens)} / "
                f"{format_tokens_compact(limit)}{pct} {sym('dot')} {duration_sec:.1f}s")
