"""
Glob Tool for CodeMason
Based on src/tools/GlobTool/ (GlobTool.ts, prompt.ts) from Claude Code.

Schema (pattern, path), DESCRIPTION and behavior are ported verbatim:
- glob pattern matching (e.g. "**/*.js", "src/**/*.ts")
- optional `path` search root (defaults to cwd), validated as an existing dir
- results sorted by modification time (oldest first, per ripgrep
  --sort=modified), capped at `limit` keeping the first (oldest) files
- paths relativized under cwd to save tokens
- result-block formatting: "No files found" when empty, otherwise the
  filenames joined by newlines plus a truncation note when capped.

The original delegates matching to a ripgrep-backed glob() util; here it is
reimplemented on an os.walk-based matcher (which, like ripgrep --hidden,
does NOT skip dot-entries) + os.path.getmtime. is_read_only.
"""

import os
import re
import subprocess
import threading
import sys
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

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


GLOB_TOOL_NAME = 'Glob'

DESCRIPTION = """- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead"""

# GlobTool.ts: const limit = globLimits?.maxResults ?? 100
DEFAULT_LIMIT = 100

# FILE_NOT_FOUND_CWD_NOTE equivalent (utils/file.ts), used in validateInput.
FILE_NOT_FOUND_CWD_NOTE = (
    'Note that the file path should be either an absolute path or a path '
    'relative to the current working directory:'
)


def _expand_path(path: str) -> str:
    """expandPath: resolve ~ and make relative paths absolute against cwd."""
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(get_cwd(), expanded)
    return os.path.normpath(expanded)


def _to_relative_path(path: str) -> str:
    """toRelativePath: relativize paths under cwd to save tokens."""
    cwd = get_cwd()
    try:
        rel = os.path.relpath(path, cwd)
    except ValueError:
        return path
    # Only relativize when the path is actually under cwd (no '..' escape),
    # matching the original which leaves outside paths absolute.
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return path
    return rel


def _get_path(input_data: Dict[str, Any]) -> str:
    """getPath({ path }): path ? expandPath(path) : getCwd()."""
    path = input_data.get('path')
    return _expand_path(path) if path else get_cwd()


def _glob_to_regex(pattern: str) -> str:
    """
    Translate a glob pattern into a regex string (no anchors), matched against a
    '/'-joined relative path. Supports ripgrep-style '**' (spans any number of
    directory segments, including zero), '*'/'?'/'[...]' within a single segment,
    and '{a,b,c}' brace alternation (ripgrep's globset supports it; the original
    delegates to `ripGrep --glob`). Braces may nest and their alternatives may
    contain glob metacharacters (e.g. '*.{ts,tsx}', '{src,test}/**/*.js').
    """
    pattern = pattern.replace(os.sep, '/')
    if os.altsep:
        pattern = pattern.replace(os.altsep, '/')

    i = 0
    n = len(pattern)
    out: List[str] = []
    while i < n:
        c = pattern[i]
        if c == '*':
            # Check for '**'.
            if i + 1 < n and pattern[i + 1] == '*':
                # Consume the '**'.
                i += 2
                # '**/' (or '**' at end) matches zero or more path segments.
                if i < n and pattern[i] == '/':
                    i += 1
                    # Matches any sequence ending in a separator, or nothing.
                    out.append('(?:.*/)?')
                else:
                    out.append('.*')
            else:
                # Single '*': any run of non-separator characters.
                out.append('[^/]*')
                i += 1
        elif c == '?':
            out.append('[^/]')
            i += 1
        elif c == '[':
            # Character class: pass through to fnmatch-translated form.
            j = i + 1
            if j < n and pattern[j] in ('!', '^'):
                j += 1
            if j < n and pattern[j] == ']':
                j += 1
            while j < n and pattern[j] != ']':
                j += 1
            if j >= n:
                # Unterminated class: treat '[' literally.
                out.append(re.escape('['))
                i += 1
            else:
                stuff = pattern[i + 1:j]
                if stuff.startswith('!'):
                    stuff = '^' + stuff[1:]
                out.append('[' + stuff + ']')
                i = j + 1
        elif c == '{':
            # Brace alternation: find the matching '}' (depth-aware), split on
            # top-level commas, translate each alternative as a sub-glob.
            depth = 1
            j = i + 1
            start = j
            alts: List[str] = []
            while j < n and depth > 0:
                cj = pattern[j]
                if cj == '{':
                    depth += 1
                elif cj == '}':
                    depth -= 1
                    if depth == 0:
                        break
                elif cj == ',' and depth == 1:
                    alts.append(pattern[start:j])
                    start = j + 1
                j += 1
            if depth != 0:
                # Unterminated '{': treat literally.
                out.append(re.escape('{'))
                i += 1
            else:
                alts.append(pattern[start:j])  # final alternative
                out.append('(?:' + '|'.join(_glob_to_regex(a) for a in alts) + ')')
                i = j + 1  # past the closing '}'
        else:
            out.append(re.escape(c))
            i += 1

    return ''.join(out)


def _translate_glob(pattern: str) -> 're.Pattern[str]':
    """
    Compile a glob pattern to an anchored regex matched against a '/'-joined
    relative path.

    Examples (relative to search root):
      '**/*.js'      matches 'a.js', '.config/x.js', 'a/b/c.js'
      'src/*.ts'     matches 'src/x.ts' but not 'src/a/x.ts'
      '**/*.{ts,tsx}' matches 'App.tsx', 'src/x.ts'
    """
    return re.compile('(?s:' + _glob_to_regex(pattern) + ')\\Z')


_rg_warned = False


def _warn_rg_missing_once() -> None:
    """Warn once that ripgrep is missing so Glob fell back to the slow os.walk."""
    global _rg_warned
    if not _rg_warned:
        _rg_warned = True
        print("[codemason] WARNING: ripgrep (rg) not found — Glob is using the slow "
              "os.walk fallback, which can stall on large trees (e.g. .venv) and make "
              "the spinner look frozen. Install it for the fast canonical path: "
              "sudo apt install ripgrep", file=sys.stderr)


# Bound the retained path list from one rg --files run. rg emits oldest-first
# (--sort=modified) and Glob keeps the OLDEST (offset:offset+limit), so a capped
# tail is exactly what Glob would drop anyway — while a pathological tree can no
# longer buffer an unbounded path list into RAM.
_RG_FILES_MAX = 1_000_000


def _rg_files_streamed(rg: str, rel_pattern: str, walk_root: str) -> List[str]:
    """Stream `rg --files --glob`, capping the retained paths (see _RG_FILES_MAX).
    A watchdog kills rg at the 30s deadline even if it blocks. rg emits UTF-8;
    strict decoding means a non-UTF-8 filename raises (→ os.walk fallback in the
    caller), exactly as the old subprocess.run(encoding='utf-8') did."""
    proc = subprocess.Popen(
        [rg, '--files', '--glob', rel_pattern, '--no-ignore', '--hidden', '--sort=modified'],
        cwd=walk_root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding='utf-8', errors='strict',
    )
    timed_out = {'v': False}

    def _kill():
        timed_out['v'] = True
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(30.0, _kill)
    timer.start()
    out: List[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            p = line[:-1] if line.endswith('\n') else line
            if p:
                out.append(os.path.abspath(os.path.join(walk_root, p)))
                if len(out) >= _RG_FILES_MAX:
                    proc.kill()
                    break
    finally:
        timer.cancel()
        proc.wait()
    if timed_out['v']:
        raise RuntimeError('ripgrep --files timed out')
    return out


def _glob_files(pattern: str, search_dir: str) -> List[str]:
    """
    Walk search_dir and return absolute file paths matching the glob pattern.

    Mirrors the ripgrep-backed util (ripgrep invoked with --hidden by default):
    '**' spans any directory depth (recursive), hidden files/dirs ARE included
    and traversed, and only files (not directories) are returned.
    """
    # An absolute pattern is split into a static base dir + relative pattern so
    # the walk roots at the right place (mirrors extractGlobBaseDirectory).
    if os.path.isabs(pattern):
        norm = pattern.replace(os.sep, '/')
        if os.altsep:
            norm = norm.replace(os.altsep, '/')
        match = re.search(r'[*?\[{]', norm)
        if match is None:
            base = os.path.dirname(pattern)
            rel_pattern = os.path.basename(pattern)
        else:
            static_prefix = norm[:match.start()]
            last_sep = static_prefix.rfind('/')
            if last_sep == -1:
                base = '/'
                rel_pattern = pattern
            else:
                base = static_prefix[:last_sep] or '/'
                rel_pattern = pattern[last_sep + 1:]
        walk_root = base
    else:
        walk_root = search_dir
        rel_pattern = pattern

    # ── PRIMARY: ripgrep — faithful to Claude Code (utils/glob.ts):
    #    rg --files --glob <pat> --no-ignore --hidden --sort=modified
    #    Native speed handles huge trees (.venv ~20k files) without stalling, and
    #    being a subprocess it RELEASES the GIL so the spinner keeps ticking
    #    (no false "frozen" look). --no-ignore --hidden = same semantics as the
    #    reference (gitignore not respected, hidden dirs traversed).
    from ..utils.ripgrep import resolve_ripgrep
    rg = resolve_ripgrep()
    if rg:
        try:
            return _rg_files_streamed(rg, rel_pattern, walk_root)
        except Exception:
            pass  # rg present but errored (incl. non-UTF-8 path) → fall to os.walk
    else:
        _warn_rg_missing_once()

    # ── FALLBACK: pure-Python os.walk. SLOWER and runs synchronously (blocks the
    #    event loop → the spinner timer can appear frozen, and very large trees
    #    like .venv can stall for seconds). Install ripgrep to use the fast path.
    #    Same --hidden semantics: hidden dirs are traversed (no exclusions).
    regex = _translate_glob(rel_pattern)
    # ripgrep matches a slash-less glob (e.g. '*.js') against the BASENAME at any
    # depth; the anchored full-path regex alone would only match top-level files,
    # so the fallback silently missed nested matches the rg path found. Mirror rg:
    # match the basename for slash-less patterns, the relative path otherwise.
    _basename_only = ('/' not in rel_pattern) and (os.sep not in rel_pattern)
    files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(walk_root):
        for fname in filenames:
            abs = os.path.join(dirpath, fname)
            if _basename_only:
                if regex.match(fname):
                    files.append(os.path.abspath(abs))
                continue
            rel = os.path.relpath(abs, walk_root)
            rel = rel.replace(os.sep, '/')
            if os.altsep:
                rel = rel.replace(os.altsep, '/')
            if regex.match(rel):
                files.append(os.path.abspath(abs))
    return files


class GlobInput(ToolInputSchema):
    pattern: str = Field(
        ...,
        description="The glob pattern to match files against",
    )
    path: Optional[str] = Field(
        default=None,
        description="The directory to search in. If not specified, the current "
        "working directory will be used. IMPORTANT: Omit this field to use the "
        "default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it "
        "for the default behavior. Must be a valid directory path if provided.",
    )


class GlobTool(Tool):
    """Fast file pattern matching by glob pattern."""

    name = GLOB_TOOL_NAME
    input_schema = GlobInput
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
        return cls.name

    @classmethod
    async def validate_input(
        cls,
        input_data: Dict[str, Any],
        context: Any,
    ) -> ValidationResult:
        """validateInput: if path is provided, it must exist and be a directory."""
        path = input_data.get('path')
        if path:
            absolute_path = _expand_path(path)

            # SECURITY: skip filesystem ops for UNC paths to avoid NTLM leaks.
            if absolute_path.startswith('\\\\') or absolute_path.startswith('//'):
                return ValidationResultSuccess()

            if not os.path.exists(absolute_path):
                message = (
                    f"Directory does not exist: {path}. {FILE_NOT_FOUND_CWD_NOTE} "
                    f"{get_cwd()}."
                )
                return ValidationResultFailure(
                    result=False,
                    message=message,
                    meta={'errorCode': 1},
                )

            if not os.path.isdir(absolute_path):
                return ValidationResultFailure(
                    result=False,
                    message=f"Path is not a directory: {path}",
                    meta={'errorCode': 2},
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
        """"Finding <pattern>" (getActivityDescription)."""
        summary = cls.get_tool_use_summary(input_data)
        return f'Finding {summary}' if summary else 'Finding files'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        if not isinstance(input_data, dict):
            return ''
        pattern = input_data.get('pattern', '')
        if not pattern:
            return ''
        path = input_data.get('path', '')
        return colorize(f"{pattern} in {path}" if path else str(pattern), Colors.DIM)

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """Found N files (GlobTool/UI.tsx SearchResultSummary) — the file list is
        the model's business; the user gets the count and Ctrl+O."""
        if not isinstance(data, dict) or data.get('error') is not None:
            return None
        n = data.get('numFiles', 0)
        line = f"Found {n} file{'' if n == 1 else 's'}"
        if data.get('truncated'):
            line += colorize(' (truncated)', Colors.DIM)
        if not verbose:
            return line
        return '\n'.join([line, *(data.get('filenames') or [])])

    @classmethod
    def _map_to_result_block(cls, filenames: List[str], truncated: bool) -> str:
        """mapToolResultToToolResultBlockParam."""
        if len(filenames) == 0:
            return 'No files found'
        lines = list(filenames)
        if truncated:
            lines.append(
                '(Results are truncated. Consider using a more specific path or pattern.)'
            )
        return '\n'.join(lines)

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        """
        call: glob the pattern under getPath, sort by modification time (oldest
        first, mirroring ripgrep --sort=modified), cap at the limit keeping the
        first (oldest) files, relativize paths, and emit the formatted result.
        """
        start = time.monotonic()
        pattern = input_data.get('pattern', '')

        # globLimits?.maxResults ?? 100
        limit = DEFAULT_LIMIT
        glob_limits = getattr(context, 'glob_limits', None)
        if glob_limits is not None:
            max_results = getattr(glob_limits, 'max_results', None)
            if max_results is None and isinstance(glob_limits, dict):
                max_results = glob_limits.get('max_results') or glob_limits.get('maxResults')
            if max_results:
                limit = max_results

        search_dir = _get_path(input_data)

        try:
            all_files = _glob_files(pattern, search_dir)

            # Sort by modification time, oldest first. ripgrep's
            # --sort=modified emits OLDEST->NEWEST, and the original slices
            # [offset:offset+limit] keeping the FIRST (oldest) files.
            with_mtime: List[Tuple[str, float]] = []
            for f in all_files:
                try:
                    mtime = os.path.getmtime(f)
                except OSError:
                    mtime = 0.0
                with_mtime.append((f, mtime))
            with_mtime.sort(key=lambda x: x[1])

            sorted_files = [f for f, _ in with_mtime]

            # Apply the cap; truncated when there were more than `limit` matches.
            offset = 0
            truncated = len(sorted_files) > offset + limit
            capped = sorted_files[offset:offset + limit]

            # Relativize paths under cwd to save tokens (same as GrepTool).
            filenames = [_to_relative_path(f) for f in capped]

            duration_ms = int((time.monotonic() - start) * 1000)
            result_for_assistant = cls._map_to_result_block(filenames, truncated)

            yield ToolResultSuccess(
                type='result',
                result_for_assistant=result_for_assistant,
                data={
                    'filenames': filenames,
                    'durationMs': duration_ms,
                    'numFiles': len(filenames),
                    'truncated': truncated,
                },
            )

        except Exception as e:
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=f"Error searching files: {str(e)}",
                data={'error': str(e)},
            )
