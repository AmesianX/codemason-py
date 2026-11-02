"""
Grep Tool for CodeMason
Based on src/tools/GrepTool/ (GrepTool.ts, prompt.ts) from Claude Code.

The tool is "built on ripgrep". This port stays faithful by shelling out
to `rg` (ripgrep) via subprocess, reproducing the exact argument construction of
the original (--hidden, VCS-directory exclusions, --max-columns 500, -U
--multiline-dotall, -i, -l/-c, -n, -A/-B/-C context, -e for dash patterns,
--type, --glob splitting). When `rg` is not on PATH, it falls back to a pure
Python regex implementation that honors the same parameters.

Schema, parameter names (including -A/-B/-C/-n/-i), Field descriptions and the
description/prompt text are ported verbatim from the source. Substrate (cwd,
subprocess, re) is codemason's local stdlib-based one; no Anthropic SDK.
"""

import os
import re
import subprocess
import threading
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import Field

from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
    ValidationResultFailure,
)
from ..utils.state import get_cwd
from ..utils.formatting import Colors, colorize


GREP_TOOL_NAME = 'Grep'

# Names of the tools referenced in the description. The source interpolates the
# Agent and Bash tool names; these are their values.
_AGENT_TOOL_NAME = 'Agent'   # registered tool is 'Agent'; 'Task' is a legacy alias
_BASH_TOOL_NAME = 'Bash'

# Description ported VERBATIM from prompt.ts getDescription().
DESCRIPTION = f"""A powerful search tool built on ripgrep

  Usage:
  - ALWAYS use {GREP_TOOL_NAME} for search tasks. NEVER invoke `grep` or `rg` as a {_BASH_TOOL_NAME} command. The {GREP_TOOL_NAME} tool has been optimized for correct permissions and access.
  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
  - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
  - Use {_AGENT_TOOL_NAME} tool for open-ended searches requiring multiple rounds
  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{{\\}}` to find `interface{{}}` in Go code)
  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{{[\\s\\S]*?field`, use `multiline: true`
"""


# === Input Schema (ported VERBATIM from GrepTool.ts inputSchema) ===

class GrepInput(ToolInputSchema):
    pattern: str = Field(
        ...,
        description="The regular expression pattern to search for in file contents",
    )
    path: Optional[str] = Field(
        default=None,
        description="File or directory to search in (rg PATH). Defaults to current working directory.",
    )
    glob: Optional[str] = Field(
        default=None,
        description='Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob',
    )
    output_mode: Optional[Literal['content', 'files_with_matches', 'count']] = Field(
        default=None,
        description='Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), "files_with_matches" shows file paths (supports head_limit), "count" shows match counts (supports head_limit). Defaults to "files_with_matches".',
    )
    context_before: Optional[int] = Field(
        default=None,
        alias='-B',
        description='Number of lines to show before each match (rg -B). Requires output_mode: "content", ignored otherwise.',
    )
    context_after: Optional[int] = Field(
        default=None,
        alias='-A',
        description='Number of lines to show after each match (rg -A). Requires output_mode: "content", ignored otherwise.',
    )
    context_c: Optional[int] = Field(
        default=None,
        alias='-C',
        description='Alias for context.',
    )
    context: Optional[int] = Field(
        default=None,
        description='Number of lines to show before and after each match (rg -C). Requires output_mode: "content", ignored otherwise.',
    )
    line_numbers: Optional[bool] = Field(
        default=None,
        alias='-n',
        description='Show line numbers in output (rg -n). Requires output_mode: "content", ignored otherwise. Defaults to true.',
    )
    case_insensitive: Optional[bool] = Field(
        default=None,
        alias='-i',
        description='Case insensitive search (rg -i)',
    )
    type: Optional[str] = Field(
        default=None,
        description='File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types.',
    )
    head_limit: Optional[int] = Field(
        default=None,
        description='Limit output to first N lines/entries, equivalent to "| head -N". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 250 when unspecified. Pass 0 for unlimited (use sparingly — large result sets waste context).',
    )
    offset: Optional[int] = Field(
        default=None,
        description='Skip first N lines/entries before applying head_limit, equivalent to "| tail -n +N | head -N". Works across all output modes. Defaults to 0.',
    )
    multiline: Optional[bool] = Field(
        default=None,
        description='Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.',
    )

    class Config:
        extra = 'forbid'
        populate_by_name = True


# === Constants (ported from GrepTool.ts) ===

# Version control system directories to exclude from searches.
VCS_DIRECTORIES_TO_EXCLUDE = ['.git', '.svn', '.hg', '.bzr', '.jj', '.sl']

# Default cap on grep results when head_limit is unspecified.
DEFAULT_HEAD_LIMIT = 250

# Max line length passed to ripgrep (--max-columns) to prevent base64/minified
# content from cluttering output.
MAX_COLUMNS = 500


def apply_head_limit(
    items: List[Any],
    limit: Optional[int],
    offset: int = 0,
) -> Tuple[List[Any], Optional[int]]:
    """Port of applyHeadLimit. Returns (items, applied_limit)."""
    # Explicit 0 = unlimited escape hatch.
    if limit == 0:
        return items[offset:], None
    effective_limit = limit if limit is not None else DEFAULT_HEAD_LIMIT
    sliced = items[offset:offset + effective_limit]
    # Only report applied_limit when truncation actually occurred.
    was_truncated = (len(items) - offset) > effective_limit
    return sliced, (effective_limit if was_truncated else None)


def format_limit_info(applied_limit: Optional[int], applied_offset: Optional[int]) -> str:
    """Port of formatLimitInfo."""
    parts: List[str] = []
    if applied_limit is not None:
        parts.append(f"limit: {applied_limit}")
    if applied_offset:
        parts.append(f"offset: {applied_offset}")
    return ', '.join(parts)


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + 's'


def _to_relative_path(file_path: str) -> str:
    """toRelativePath (path.ts:96-98): relativize under cwd, but keep paths
    OUTSIDE cwd absolute rather than emitting '../foo' (matches the original and
    glob_tool's relativizer — grep previously diverged from both)."""
    try:
        rel = os.path.relpath(file_path, get_cwd())
    except ValueError:
        return file_path
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return file_path
    return rel


def _path_content_colon(line: str) -> int:
    """Index of the `path:content` separator colon in a ripgrep content-mode
    line, skipping a leading Windows drive prefix (`C:\\...`) so drive-letter
    absolute paths are not split at the drive colon. On POSIX paths (leading
    `/`) the drive test never fires, so this is just the first colon."""
    start = 2 if len(line) >= 3 and line[1] == ":" and line[0].isalpha() and line[2] in ("\\", "/") else 0
    return line.find(":", start)


def _expand_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(get_cwd(), expanded)
    return expanded


def _split_glob(glob: str) -> List[str]:
    """Port of the glob-splitting logic in call(): split on whitespace, then on
    commas (unless the pattern contains braces)."""
    glob_patterns: List[str] = []
    for raw_pattern in re.split(r'\s+', glob):
        if not raw_pattern:
            continue
        if '{' in raw_pattern and '}' in raw_pattern:
            glob_patterns.append(raw_pattern)
        else:
            glob_patterns.extend(p for p in raw_pattern.split(',') if p)
    return [g for g in glob_patterns if g]


def _build_rg_args(
    pattern: str,
    glob: Optional[str],
    type_: Optional[str],
    output_mode: str,
    context_before: Optional[int],
    context_after: Optional[int],
    context_c: Optional[int],
    context: Optional[int],
    show_line_numbers: bool,
    case_insensitive: bool,
    multiline: bool,
) -> List[str]:
    """Faithful port of the ripgrep arg construction in call()."""
    args: List[str] = ['--hidden']

    # Exclude VCS directories to avoid noise from version control metadata.
    for d in VCS_DIRECTORIES_TO_EXCLUDE:
        args += ['--glob', f'!{d}']

    # Limit line length to prevent base64/minified content from cluttering output.
    args += ['--max-columns', str(MAX_COLUMNS)]

    # Only apply multiline flags when explicitly requested.
    if multiline:
        args += ['-U', '--multiline-dotall']

    if case_insensitive:
        args.append('-i')

    # Output mode flags.
    if output_mode == 'files_with_matches':
        args.append('-l')
    elif output_mode == 'count':
        args.append('-c')

    if show_line_numbers and output_mode == 'content':
        args.append('-n')

    # Context flags (-C/context takes precedence over before/after).
    if output_mode == 'content':
        if context is not None:
            args += ['-C', str(context)]
        elif context_c is not None:
            args += ['-C', str(context_c)]
        else:
            if context_before is not None:
                args += ['-B', str(context_before)]
            if context_after is not None:
                args += ['-A', str(context_after)]

    # If pattern starts with dash, use -e so rg does not treat it as an option.
    if pattern.startswith('-'):
        args += ['-e', pattern]
    else:
        args.append(pattern)

    if type_:
        args += ['--type', type_]

    if glob:
        for g in _split_glob(glob):
            args += ['--glob', g]

    return args


_RIPGREP_TIMEOUT_SEC = 30   # bound the search (Glob uses the same); original execFile has a timeout


# Hard cap on lines held from a single ripgrep run. A broad pattern over a huge
# tree could otherwise buffer millions of match lines into RAM before the caller
# ever applies its head-limit. Far above any realistic result set, so it only
# ever trips on pathological searches.
_RIPGREP_MAX_LINES = 200_000


def _merge_rg_passes(primary: List[str], secondary: List[str],
                     output_mode: str) -> List[str]:
    """Merge a second ripgrep pass into the first, per output mode.

    ripgrep can only be told ONE encoding for a whole run, so covering a tree
    that mixes UTF-8 and a legacy encoding means running it twice. The two
    result sets overlap: an ASCII pattern matches the same line in both passes,
    because ASCII decodes identically either way. So this is not concatenation.

    - content / files_with_matches: identical lines are the same finding; keep
      first-seen order so the primary pass still reads first.
    - count: the LARGER of the two counts, not their sum. Summing was wrong and
      measured wrong — it reported every file at exactly double. The premise was
      inverted: the two passes do not find different matches in a file, they
      find THE SAME lines. ASCII is byte-identical under both encodings, so an
      ASCII pattern matches the same lines in both passes (measured: 2 and 2,
      summed to 4 for a file with 2). For a pattern that is not ASCII only the
      correctly-decoded pass finds anything, and the other reports 0 (measured:
      1 and 0). Either way one pass's result contains the other's, so the larger
      is the count and the sum is the double-counting.

      max under-reports only for a file holding real matches under BOTH
      interpretations at once — lines valid only as UTF-8 and lines valid only
      as CP949 in one file. ASCII lines decode identically so they cannot
      differ, which leaves only deliberately malformed input; under-reporting
      there is a far better failure than doubling every ordinary count.
    """
    if not secondary:
        return primary
    if output_mode != 'count':
        seen = set()
        out = []
        for line in list(primary) + list(secondary):
            if line not in seen:
                seen.add(line)
                out.append(line)
        return out
    order: List[str] = []
    totals = {}
    for line in list(primary) + list(secondary):
        colon = line.rfind(':')
        if colon <= 0:
            continue
        name, count = line[:colon], line[colon + 1:]
        try:
            n = int(count.strip())
        except ValueError:
            continue
        if name not in totals:
            order.append(name)
            totals[name] = 0
        totals[name] = max(totals[name], n)
    return [f'{name}:{totals[name]}' for name in order]


def _run_ripgrep_both_encodings(rg_path: str, args: List[str], cwd: str,
                                output_mode: str) -> List[str]:
    """Run ripgrep twice — once as it comes, once in the configured legacy
    encoding — and merge.

    ripgrep's automatic detection only reads a BOM, so a CP949 file is searched
    as if it were UTF-8 and Hangul inside it matches nothing at all. Passing
    --encoding is not a fix by itself: measured, `-E euc-kr` finds that file and
    MISSES the UTF-8 one, trading one blind spot for another. Both passes
    together are what covers a mixed tree.
    """
    primary = _run_ripgrep(rg_path, args, cwd)
    from ..utils.config import get_config
    legacy = (getattr(get_config(), 'grep_legacy_encoding', '') or '').strip()
    if not legacy:
        return primary
    try:
        # A failure of the extra pass must never lose the results of the one
        # that worked — this is an addition to the search, not a precondition.
        secondary = _run_ripgrep(rg_path, list(args) + ['--encoding', legacy], cwd)
    except Exception:  # noqa: BLE001
        secondary = []
    return _merge_rg_passes(primary, secondary, output_mode)


def _run_ripgrep(rg_path: str, args: List[str], cwd: str) -> List[str]:
    """Invoke ripgrep and return its stdout lines (excluding trailing empty).

    Streams stdout with a hard line cap (see _RIPGREP_MAX_LINES) so output can't
    grow unbounded in memory, and a watchdog timer kills rg at the deadline even
    if it is blocked producing nothing — preserving the hang-safety the old
    subprocess.run(timeout=...) gave while bounding memory too."""
    try:
        proc = subprocess.Popen(
            [rg_path] + args + [cwd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # ripgrep emits UTF-8. Without encoding= this decodes with the
            # process locale, which is cp949 on Korean Windows, so every Hangul
            # match line came back as mojibake — and errors='replace' masked the
            # failure instead of surfacing it. glob_tool.py:220 already pins
            # utf-8 for the same reason; 'replace' is kept here (not glob's
            # 'strict') because a match line may legitimately contain a
            # non-UTF-8 byte from the searched file, and losing the whole search
            # over one bad byte is worse than one replacement character.
            encoding='utf-8',
            errors='replace',
        )
    except OSError as e:
        raise RuntimeError(f'failed to launch ripgrep: {e}')

    timed_out = {'v': False}

    def _kill_on_timeout():
        timed_out['v'] = True
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(_RIPGREP_TIMEOUT_SEC, _kill_on_timeout)
    timer.start()

    lines: List[str] = []
    stderr = ''
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line[:-1] if line.endswith('\n') else line)
            if len(lines) >= _RIPGREP_MAX_LINES:
                proc.kill()
                break
        if proc.stderr is not None:
            stderr = proc.stderr.read()
    finally:
        timer.cancel()
        proc.wait()

    if timed_out['v']:
        raise RuntimeError(
            f'ripgrep timed out after {_RIPGREP_TIMEOUT_SEC}s — search incomplete; '
            'narrow the pattern or path.'
        )
    # rg exit codes: 0 = matches, 1 = no matches, 2 = error. A deliberate kill
    # after hitting the line cap yields a negative rc; treat a capped result as
    # a successful (truncated) search rather than an error.
    if proc.returncode not in (0, 1) and len(lines) < _RIPGREP_MAX_LINES:
        raise RuntimeError((stderr or '').strip() or f'ripgrep exited {proc.returncode}')
    return lines


# === Python fallback (used when `rg` is not on PATH) ===

_FALLBACK_SKIP_DIRS = set(VCS_DIRECTORIES_TO_EXCLUDE) | {'node_modules', '__pycache__'}

# ripgrep's built-in type definitions are extensive; reproduce the most common
# ones referenced by the description (js, py, rust, go, java, etc.).
_FALLBACK_TYPES: Dict[str, List[str]] = {
    'py': ['*.py', '*.pyi', '*.pyw'],
    'js': ['*.js', '*.jsx', '*.mjs', '*.cjs', '*.vue'],
    'ts': ['*.ts', '*.tsx', '*.cts', '*.mts'],
    'rust': ['*.rs'],
    'go': ['*.go'],
    'java': ['*.java', '*.jsp'],
    'c': ['*.c', '*.h'],
    'cpp': ['*.cpp', '*.cc', '*.cxx', '*.hpp', '*.hh', '*.hxx', '*.C', '*.H'],
    'cs': ['*.cs'],
    'rb': ['*.rb'],
    'php': ['*.php', '*.php3', '*.php4', '*.php5', '*.phtml'],
    'html': ['*.html', '*.htm', '*.xhtml'],
    'css': ['*.css', '*.scss', '*.sass', '*.less'],
    'json': ['*.json', '*.jsonl'],
    'yaml': ['*.yaml', '*.yml'],
    'md': ['*.md', '*.markdown', '*.mkd'],
    'sh': ['*.sh', '*.bash', '*.zsh'],
    'sql': ['*.sql'],
    'toml': ['*.toml'],
    'xml': ['*.xml'],
}


def _fallback_iter_files(root: str, globs: List[str], type_: Optional[str]) -> List[str]:
    import fnmatch

    type_globs = _FALLBACK_TYPES.get(type_, []) if type_ else []
    # rg matches a glob containing '/' (e.g. 'src/*.js') against the path relative
    # to the search root, not just the basename — matching a slash-glob on the
    # bare filename (as before) never succeeded, so the fallback silently found
    # nothing for such globs. Split accordingly.
    name_globs = [g for g in globs if '/' not in g]
    path_globs = [g for g in globs if '/' in g]

    def matches(relpath: str, name: str) -> bool:
        if globs and not (
            any(fnmatch.fnmatch(name, g) for g in name_globs)
            or any(fnmatch.fnmatch(relpath, g) for g in path_globs)
        ):
            return False
        if type_globs and not any(fnmatch.fnmatch(name, g) for g in type_globs):
            return False
        return True

    if os.path.isfile(root):
        return [root]

    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _FALLBACK_SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            if matches(rel, fn):
                files.append(full)
    return files


def _fallback_search(
    root: str,
    pattern: str,
    globs: List[str],
    type_: Optional[str],
    output_mode: str,
    context_before: int,
    context_after: int,
    show_line_numbers: bool,
    case_insensitive: bool,
    multiline: bool,
) -> List[str]:
    """Pure-Python honoring of the same params. Returns rg-equivalent lines:
    - files_with_matches: absolute file paths
    - count: 'path:count'
    - content: 'path:n:line' (or 'path:line') with context lines, '-' separated
    """
    flags = 0
    if case_insensitive:
        flags |= re.IGNORECASE
    if multiline:
        # ripgrep's --multiline-dotall makes `.` match newlines WITHOUT changing
        # `^`/`$` anchoring, so use only re.DOTALL (not re.MULTILINE).
        flags |= re.DOTALL
    regex = re.compile(pattern, flags)

    files = _fallback_iter_files(root, globs, type_)
    out_lines: List[str] = []

    for fp in files:
        # Use the shared detector rather than assuming UTF-8. Unlike the
        # ripgrep path above — one stream carrying lines from many files, where
        # no single encoding is right — here we read ONE known file, so its
        # encoding is knowable.
        #
        # What this actually fixes, measured: a UTF-16 file was INVISIBLE to
        # this search. Decoded as UTF-8 its ASCII text keeps the NUL padding
        # bytes, the binary probe below sees \x00 and skips the file entirely,
        # so a pattern plainly present in it returned no match at all. With the
        # BOM honoured it decodes and matches normally.
        #
        # It does NOT fix legacy CJK: the detector answers latin-1 for anything
        # it cannot prove is UTF-8, which preserves bytes (what Edit needs to
        # round-trip safely) but is not the text, so Hangul in a CP949 file
        # still does not match a Hangul pattern. Doing that needs real charset
        # detection plus a CJK codec, which Rust has no equivalent for without a
        # new dependency — fixing it here alone would put the two languages out
        # of step. Read-only either way: nothing is written back.
        try:
            from ..utils.file import detect_file_encoding
            enc = detect_file_encoding(str(fp))
            with open(fp, 'r', encoding=enc, errors='replace') as f:
                text = f.read()
        except (OSError, UnicodeError, LookupError):
            continue
        # Skip likely-binary content.
        if '\x00' in text[:8192]:
            continue

        lines = text.splitlines()

        if multiline:
            matched = bool(regex.search(text))
            match_line_idxs = set()
            if matched:
                for m in regex.finditer(text):
                    start_line = text.count('\n', 0, m.start())
                    match_line_idxs.add(start_line)
        else:
            match_line_idxs = {i for i, ln in enumerate(lines) if regex.search(ln)}
            matched = bool(match_line_idxs)

        if not matched:
            continue

        if output_mode == 'files_with_matches':
            out_lines.append(fp)
        elif output_mode == 'count':
            out_lines.append(f"{fp}:{len(match_line_idxs)}")
        else:  # content
            emitted: List[str] = []
            shown = set()
            for idx in sorted(match_line_idxs):
                lo = max(0, idx - context_before)
                hi = min(len(lines), idx + context_after + 1)
                for j in range(lo, hi):
                    if j in shown:
                        continue
                    shown.add(j)
                    is_match = j in match_line_idxs
                    sep = ':' if is_match else '-'
                    # Truncate long lines to MAX_COLUMNS to mirror rg --max-columns 500.
                    line_text = lines[j][:MAX_COLUMNS]
                    if show_line_numbers:
                        emitted.append(f"{fp}{sep}{j + 1}{sep}{line_text}")
                    else:
                        emitted.append(f"{fp}{sep}{line_text}")
            out_lines.extend(emitted)

    return out_lines


class GrepTool(Tool):
    """A powerful search tool built on ripgrep."""

    name = GREP_TOOL_NAME
    input_schema = GrepInput
    # GrepTool.maxResultSizeChars (lower than the 50k default → persists sooner).
    max_result_size_chars = 20_000
    DESCRIPTION = DESCRIPTION

    @classmethod
    async def description(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    def is_read_only(cls) -> bool:
        return True

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def needs_permissions(cls) -> bool:
        return True

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return 'Search'

    @classmethod
    async def validate_input(
        cls,
        input_data: Dict[str, Any],
        context: Any,
    ) -> ValidationResult:
        # If path is provided, validate that it exists (port of validateInput).
        path = input_data.get('path')
        if path:
            absolute_path = _expand_path(path)
            # SECURITY: skip fs ops for UNC paths.
            if absolute_path.startswith('\\\\') or absolute_path.startswith('//'):
                return ValidationResultSuccess()
            if not os.path.exists(absolute_path):
                return ValidationResultFailure(
                    result=False,
                    message=f"Path does not exist: {path}. Note: paths are relative to "
                    f"the current working directory {get_cwd()}.",
                )
        return ValidationResultSuccess()

    @classmethod
    def get_tool_use_summary(cls, input_data):
        """The pattern, truncated for display (UI.tsx getToolUseSummary)."""
        from ..utils.formatting import TOOL_SUMMARY_MAX_LENGTH, truncate
        if not isinstance(input_data, dict):
            return None
        value = input_data.get('pattern')
        if not value:
            return None
        return truncate(str(value), TOOL_SUMMARY_MAX_LENGTH)

    @classmethod
    def get_activity_description(cls, input_data):
        """"Searching for <pattern>" (getActivityDescription)."""
        summary = cls.get_tool_use_summary(input_data)
        return f'Searching for {summary}' if summary else 'Searching'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """pattern (in path) — GrepTool/UI.tsx renders the args, not a sentence."""
        if not isinstance(input_data, dict):
            return ''
        pattern = input_data.get('pattern', '')
        if not pattern:
            return ''
        path = input_data.get('path')
        summary = f"{pattern} in {path}" if path else str(pattern)
        return colorize(summary, Colors.DIM)

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """SearchResultSummary (GrepTool/UI.tsx): a count per output_mode, and the
        matched content only in verbose (Ctrl+O expands it otherwise)."""
        if not isinstance(data, dict) or data.get('error') is not None:
            return None
        mode = data.get('mode', 'files_with_matches')
        if mode == 'content':
            n = data.get('numLines', 0)
            line = f"Found {n} line{'' if n == 1 else 's'}"
        elif mode == 'count':
            n = data.get('numMatches', 0)
            files = data.get('numFiles', 0)
            line = (f"Found {n} match{'' if n == 1 else 'es'} across "
                    f"{files} file{'' if files == 1 else 's'}")
        else:
            n = data.get('numFiles', 0)
            line = f"Found {n} file{'' if n == 1 else 's'}"
        if not verbose:
            return line
        content = data.get('content') or '\n'.join(data.get('filenames') or [])
        return '\n'.join([line, *content.split('\n')]) if content else line

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        pattern = input_data.get('pattern', '')
        path = input_data.get('path')

        glob = input_data.get('glob')
        type_ = input_data.get('type')
        output_mode = input_data.get('output_mode') or 'files_with_matches'

        context_before = input_data.get('-B', input_data.get('context_before'))
        context_after = input_data.get('-A', input_data.get('context_after'))
        context_c = input_data.get('-C', input_data.get('context_c'))
        context_val = input_data.get('context')

        show_line_numbers = input_data.get('-n', input_data.get('line_numbers'))
        if show_line_numbers is None:
            show_line_numbers = True
        case_insensitive = bool(input_data.get('-i', input_data.get('case_insensitive')) or False)

        head_limit = input_data.get('head_limit')
        offset = input_data.get('offset') or 0
        multiline = bool(input_data.get('multiline') or False)

        absolute_path = _expand_path(path) if path else get_cwd()

        try:
            from ..utils.ripgrep import resolve_ripgrep
            rg_path = resolve_ripgrep()
            if rg_path:
                args = _build_rg_args(
                    pattern=pattern,
                    glob=glob,
                    type_=type_,
                    output_mode=output_mode,
                    context_before=context_before,
                    context_after=context_after,
                    context_c=context_c,
                    context=context_val,
                    show_line_numbers=show_line_numbers,
                    case_insensitive=case_insensitive,
                    multiline=multiline,
                )
                results = _run_ripgrep_both_encodings(
                    rg_path, args, absolute_path, output_mode)
            else:
                # Resolve effective context for fallback (mirrors precedence).
                if output_mode == 'content':
                    if context_val is not None:
                        cb = ca = context_val
                    elif context_c is not None:
                        cb = ca = context_c
                    else:
                        cb = context_before or 0
                        ca = context_after or 0
                else:
                    cb = ca = 0
                results = _fallback_search(
                    root=absolute_path,
                    pattern=pattern,
                    globs=_split_glob(glob) if glob else [],
                    type_=type_,
                    output_mode=output_mode,
                    context_before=cb,
                    context_after=ca,
                    show_line_numbers=show_line_numbers,
                    case_insensitive=case_insensitive,
                    multiline=multiline,
                )
        except Exception as e:
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=f"Error searching: {str(e)}",
                data={'error': str(e)},
            )
            return

        if output_mode == 'content':
            limited, applied_limit = apply_head_limit(results, head_limit, offset)
            final_lines = []
            for line in limited:
                colon = _path_content_colon(line)
                if colon > 0:
                    final_lines.append(_to_relative_path(line[:colon]) + line[colon:])
                else:
                    final_lines.append(line)
            content = '\n'.join(final_lines)
            applied_offset = offset if offset > 0 else None
            limit_info = format_limit_info(applied_limit, applied_offset)
            result_content = content or 'No matches found'
            final_content = (
                f"{result_content}\n\n[Showing results with pagination = {limit_info}]"
                if limit_info else result_content
            )
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=final_content,
                data={
                    'mode': 'content',
                    'numFiles': 0,
                    'filenames': [],
                    'content': content,
                    'numLines': len(final_lines),
                    'appliedLimit': applied_limit,
                    'appliedOffset': applied_offset,
                },
            )
            return

        if output_mode == 'count':
            limited, applied_limit = apply_head_limit(results, head_limit, offset)
            final_count_lines = []
            for line in limited:
                colon = line.rfind(':')
                if colon > 0:
                    final_count_lines.append(_to_relative_path(line[:colon]) + line[colon:])
                else:
                    final_count_lines.append(line)

            total_matches = 0
            file_count = 0
            for line in final_count_lines:
                colon = line.rfind(':')
                if colon > 0:
                    try:
                        total_matches += int(line[colon + 1:])
                        file_count += 1
                    except ValueError:
                        pass

            applied_offset = offset if offset > 0 else None
            limit_info = format_limit_info(applied_limit, applied_offset)
            raw_content = '\n'.join(final_count_lines) or 'No matches found'
            summary = (
                f"\n\nFound {total_matches} total "
                f"{'occurrence' if total_matches == 1 else 'occurrences'} across "
                f"{file_count} {'file' if file_count == 1 else 'files'}."
                f"{f' with pagination = {limit_info}' if limit_info else ''}"
            )
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=raw_content + summary,
                data={
                    'mode': 'count',
                    'numFiles': file_count,
                    'filenames': [],
                    'content': '\n'.join(final_count_lines),
                    'numMatches': total_matches,
                    'appliedLimit': applied_limit,
                    'appliedOffset': applied_offset,
                },
            )
            return

        # files_with_matches mode (default). Sort by mtime desc, filename tiebreak.
        def _mtime(p: str) -> float:
            try:
                return os.path.getmtime(p)
            except OSError:
                return 0.0

        sorted_matches = sorted(results, key=lambda p: (-_mtime(p), p))
        final_matches, applied_limit = apply_head_limit(sorted_matches, head_limit, offset)
        relative_matches = [_to_relative_path(p) for p in final_matches]
        num_files = len(relative_matches)

        applied_offset = offset if offset > 0 else None
        limit_info = format_limit_info(applied_limit, applied_offset)

        if num_files == 0:
            result = 'No files found'
        else:
            result = (
                f"Found {num_files} {_plural(num_files, 'file')}"
                f"{f' {limit_info}' if limit_info else ''}\n"
                + '\n'.join(relative_matches)
            )

        yield ToolResultSuccess(
            type='result',
            result_for_assistant=result,
            data={
                'mode': 'files_with_matches',
                'filenames': relative_matches,
                'numFiles': num_files,
                'appliedLimit': applied_limit,
                'appliedOffset': applied_offset,
            },
        )
