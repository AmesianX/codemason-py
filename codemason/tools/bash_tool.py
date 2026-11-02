"""
Bash Tool for CodeMason

Based on Claude Code src/tools/BashTool/.

This module pairs the tool surface (input schema, description,
command-security model) with CodeMason's LOCAL command-execution substrate
(`PersistentShell.exec` — an async subprocess with timeout / abort / SIGKILL /
background-pipe handling, plus cwd-reset). The original executes commands via
its own shell infrastructure (which we don't have), so the execution mechanism
below is intentionally CodeMason-specific and is preserved verbatim.

Ported (replacing the older CodeMason versions):
  - prompt.ts  getSimplePrompt()  -> DESCRIPTION (imported from
    constants.tool_prompts.BASH_DESCRIPTION; not duplicated here)
  - toolName.ts                   -> name = 'Bash'
  - BashTool.tsx schema           -> command / timeout / run_in_background /
                                     description
  - bashSecurity.ts + commands.ts -> robust command splitting on control
                                     operators (splitCommandWithOperators /
                                     splitCommand_DEPRECATED /
                                     filterControlOperators)
  - pathValidation.ts             -> wrapper/env-prefix stripping
                                     (stripWrappersFromArgv) + cd/path
                                     containment (validatePath against the
                                     session working directory)

Sandbox sections (shouldUseSandbox.ts, getSimpleSandboxSection,
dangerouslyDisableSandbox, AST/tree-sitter parsing) are intentionally omitted —
CodeMason has no sandbox substrate.
"""

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from pydantic import Field

from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
    ValidationResultFailure,
    PermissionResult,
)
from ..utils.state import get_cwd, get_original_cwd, is_in_directory, set_cwd as _set_global_cwd
from ..utils.log import log_event
from ..utils.platform import is_windows, reap_children_with_this_process
from ..utils.windows_shell import (
    shell_argv as _win_shell_argv,
    spawn_group_kwargs as _spawn_group_kwargs,
    kill_process_tree as _kill_process_tree,
    posix_path_to_windows as _posix_to_win,
    windows_path_to_posix as _win_to_posix,
    decode_console_output as _decode_output,
)
from ..constants.tool_prompts import BASH_DESCRIPTION


# === Timeouts (utils/timeouts.ts: getDefaultBashTimeoutMs / getMaxBashTimeoutMs) ===

DEFAULT_TIMEOUT_MS = 120_000  # 2 minutes
MAX_TIMEOUT_MS = 600_000      # 10 minutes

# Display cap for the echoed command (the full command still runs).
_MAX_DISPLAY_COMMAND_CHARS = 400


# === Input Schema (BashTool.tsx fullInputSchema) ===
#
# `dangerouslyDisableSandbox` and the internal `_simulatedSedEdit` field are
# omitted: CodeMason has no sandbox / sed-preview substrate. `run_in_background`
# is accepted on the schema.

_DESCRIPTION_FIELD = (
    'Clear, concise description of what this command does in active voice. '
    'Never use words like "complex" or "risk" in the description - just '
    'describe what it does.\n'
    '\n'
    'For simple commands (git, npm, standard CLI tools), keep it brief '
    '(5-10 words):\n'
    '- ls → "List files in current directory"\n'
    '- git status → "Show working tree status"\n'
    '- npm install → "Install package dependencies"\n'
    '\n'
    'For commands that are harder to parse at a glance (piped commands, '
    'obscure flags, etc.), add enough context to clarify what it does:\n'
    '- find . -name "*.tmp" -exec rm {} \\; → "Find and delete all '
    '.tmp files recursively"\n'
    '- git reset --hard origin/main → "Discard all local changes and '
    'match remote main"\n'
    '- curl -s url | jq \'.data[]\' → "Fetch JSON from URL and extract '
    'data array elements"'
)


class BashInput(ToolInputSchema):
    """Input schema for Bash tool."""

    command: str = Field(..., description="The command to execute")
    timeout: Optional[int] = Field(
        default=None,
        description=f"Optional timeout in milliseconds (max {MAX_TIMEOUT_MS})",
    )
    run_in_background: Optional[bool] = Field(
        default=None,
        description=(
            "Set to true to run this command in the background. "
            "Use Read to read the output later."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description=_DESCRIPTION_FIELD,
    )


# === Output Type ===

@dataclass
class BashOutput:
    stdout: str
    stdout_lines: int
    stderr: str
    stderr_lines: int
    interrupted: bool


# === Command Security Model ===
#
# Ported from bashSecurity.ts / utils/bash/commands.ts / pathValidation.ts.
#
# The original splits the command into subcommands on shell control operators
# (using the shell-quote AST), strips redirections, then validates each
# subcommand's resolved paths against the session's allowed working
# directories. It also strips env-var prefixes (VAR=val) and known wrapper
# commands (env / timeout / nice / nohup / time / stdbuf) before extracting the
# command root, and resolves the *basename* of the first token so an absolute
# path (e.g. /usr/bin/curl) is treated identically to its bare name.
#
# We do not have shell-quote / tree-sitter here, so the splitter below is a
# faithful re-implementation of the SAME approach (control-operator splitting
# with quote / subshell / heredoc awareness) rather than a 1:1 port of the
# shell-quote pipeline. This deliberately closes the bypasses present in the
# older CodeMason splitter (which split only on `&&`/`;`, leaving pipes,
# subshells, env-prefixes, and absolute paths unchecked).


# Control operators that separate one subcommand from the next
# (commands.ts COMMAND_LIST_SEPARATORS, minus AST sugar like ';;').
# Ordered longest-first so the scanner matches multi-char operators greedily.
_CONTROL_OPERATORS = ['&&', '||', ';;', ';', '|', '&', '(', ')', '\n']

# Wrapper commands whose first non-flag operand is the *real* command
# (pathValidation.ts stripWrappersFromArgv). We strip them before reading the
# command root so e.g. `env FOO=1 curl ...` or `timeout 5 curl ...` is detected.
_WRAPPER_COMMANDS = {'time', 'nohup', 'timeout', 'nice', 'stdbuf', 'env'}


def _join_line_continuations(command: str) -> str:
    r"""Join backslash-newline line continuations (commands.ts:106-139).

    SECURITY: a shell joins continuation tokens with NO space, so `tr\<NL>aceroute`
    executes as the single word `traceroute`. Splitting on the raw newline would
    see two tokens (`tr` and `aceroute`) and mis-validate. Only join on an ODD
    number of backslashes before the newline (an even count pairs up as escapes
    and the newline stays a separator)."""
    def repl(m):
        run = m.group(0)
        backslashes = len(run) - 1  # minus the newline
        if backslashes % 2 == 1:
            return '\\' * (backslashes - 1)  # consume the escaping backslash + newline
        return run  # even: keep as-is (separator)
    return re.sub(r'\\+\n', repl, command)


def _split_on_operators(command: str) -> List[str]:
    """Split a command string into subcommands on shell control operators.

    Reproduces the approach (splitCommandWithOperators +
    filterControlOperators): tokenize on control operators while respecting
    single/double quotes and heredoc bodies, and treat subshell parens as
    separators so `(curl ...)` and `a && b | c` all surface every subcommand.
    """
    # Join odd-backslash line continuations BEFORE any newline tokenization
    # (security-critical — see _join_line_continuations).
    command = _join_line_continuations(command)
    # Strip heredoc bodies (<<EOF ... EOF). Their contents are data, not
    # commands; the original extracts them before parsing. We keep the command
    # *before* the heredoc operator and drop the body.
    command = _strip_heredoc_bodies(command)

    subcommands: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(command)

    def flush():
        text = ''.join(current).strip()
        if text:
            subcommands.append(text)
        current.clear()

    while i < n:
        ch = command[i]

        # Quote handling (a backslash-escaped quote is a literal char).
        # NOT inside single quotes: bash treats a backslash there as an ordinary
        # character, so it cannot escape the closing quote. Running this branch
        # first made the validator read `echo 'a\'; rm -rf ~/x #'` as one
        # `echo` subcommand — it thought \' was an escaped quote and stayed in
        # the string — while bash closes at the second quote and runs the `rm`.
        # tokens[0] was then `echo`, which is not a path command, so the
        # dangerous-removal and containment checks never ran on the `rm` at all.
        if ch == '\\' and i + 1 < n and not in_single:
            current.append(ch)
            current.append(command[i + 1])
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue

        if in_single or in_double:
            current.append(ch)
            i += 1
            continue

        # Comment: everything to end-of-line is ignored (matches bash).
        if ch == '#' and (not current or current[-1].isspace()):
            while i < n and command[i] != '\n':
                i += 1
            continue

        matched = False
        for op in _CONTROL_OPERATORS:
            if command.startswith(op, i):
                flush()
                i += len(op)
                matched = True
                break
        if matched:
            continue

        current.append(ch)
        i += 1

    flush()
    return subcommands


def _strip_heredoc_bodies(command: str) -> str:
    """Remove heredoc bodies so their contents aren't mistaken for commands."""
    # Match `<<` or `<<-` followed by an optionally-quoted delimiter.
    pattern = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
    out = command
    while True:
        m = pattern.search(out)
        if not m:
            break
        delim = m.group(2)
        # Find the terminating delimiter line after the operator.
        rest_start = m.end()
        end_re = re.compile(r"\n[ \t]*" + re.escape(delim) + r"[ \t]*(\n|$)")
        end_m = end_re.search(out, rest_start)
        if end_m:
            out = out[: m.start()] + ' ' + out[end_m.end():]
        else:
            # Unterminated heredoc: drop everything from the operator on.
            out = out[: m.start()]
            break
    return out


def _strip_redirections(tokens: List[str]) -> List[str]:
    """Drop redirection operators and their targets (commands.ts).

    Mirrors splitCommand_DEPRECATED's redirection stripping: tokens like
    `>`, `>>`, `2>`, `2>&1`, `<` and their following target are removed so they
    don't masquerade as the command root.
    """
    result: List[str] = []
    i = 0
    redirect_re = re.compile(r'^\d*(>>|>&|>|<<<|<<|<)')
    while i < len(tokens):
        tok = tokens[i]
        m = redirect_re.match(tok)
        if m:
            # If the operator is fused with its target (e.g. `>file`,
            # `2>&1`), the whole token is consumed; otherwise also consume
            # the following target token.
            if tok == m.group(0):
                i += 2  # operator + separate target
            else:
                i += 1  # fused operator+target
            continue
        result.append(tok)
        i += 1
    return result


def _strip_env_and_wrappers(tokens: List[str]) -> List[str]:
    """Strip leading `VAR=val` assignments and wrapper commands.

    Port of pathValidation.ts stripWrappersFromArgv + the env-var prefix
    handling: `FOO=1 BAR=2 timeout 5 curl x` -> `curl x`.
    """
    assign_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')
    changed = True
    while changed and tokens:
        changed = False

        # Strip leading VAR=val assignments.
        while tokens and assign_re.match(tokens[0]) and not tokens[0].startswith('-'):
            tokens = tokens[1:]
            changed = True

        if not tokens:
            break

        root = _basename(tokens[0])
        if root in _WRAPPER_COMMANDS:
            # Drop the wrapper, then skip its leading flags / flag-values
            # heuristically (covers `timeout 5`, `nice -n 10`, `env -i`, etc.).
            rest = tokens[1:]
            j = 0
            while j < len(rest):
                a = rest[j]
                if a == '--':
                    j += 1
                    break
                if a.startswith('-'):
                    j += 1
                    continue
                # A bare numeric/duration operand belongs to the wrapper
                # (timeout's duration, nice's adjustment); skip one.
                if re.fullmatch(r'\d+(?:\.\d+)?[smhd]?', a):
                    j += 1
                    continue
                break
            tokens = rest[j:]
            changed = True

    return tokens


def _basename(token: str) -> str:
    """Resolve a command token to its bare name (closes the abs-path bypass)."""
    token = token.strip().strip('\'"')
    if not token:
        return ''
    return os.path.basename(token)


def _tokenize(subcommand: str) -> List[str]:
    """Split a single subcommand into whitespace-separated tokens.

    Quote-aware so `'a b'` stays one token. Good enough to extract the command
    root and cd target; we are not executing this, only validating it.
    """
    tokens: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(subcommand)
    while i < n:
        ch = subcommand[i]
        # Same rule as _split_on_operators: a backslash inside single quotes is
        # a literal character in bash and cannot escape the closing quote.
        if ch == '\\' and i + 1 < n and not in_single:
            current.append(subcommand[i + 1])
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if ch.isspace() and not in_single and not in_double:
            if current:
                tokens.append(''.join(current))
                current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    if current:
        tokens.append(''.join(current))
    return tokens


def get_command_subcommands(command: str) -> List[List[str]]:
    """Return each subcommand as a cleaned token list (root-extractable)."""
    result: List[List[str]] = []
    for sub in _split_on_operators(command):
        tokens = _tokenize(sub)
        tokens = _strip_redirections(tokens)
        tokens = _strip_env_and_wrappers(tokens)
        if tokens:
            result.append(tokens)
    return result


# === Path-constraint security (pathValidation.ts checkPathConstraints) ===
#
# Faithful port of the substrate-INDEPENDENT core: process-substitution
# detection, output-redirection target validation, dangerous-removal blocking,
# and per-command path containment against the session's allowed working
# directory (get_original_cwd()). NOT ported (substrate codemason lacks): the
# permission-rule engine, the upstream model-based command-injection classifier
# (it calls a model we do not serve), and the
# tree-sitter / shell-quote AST — we use the quote/heredoc-aware splitter above.
# These checks run in check_permissions (the original's location), so they are
# bypassed under --dangerously-skip-permissions exactly like the original.

# pathValidation.ts SUPPORTED_PATH_COMMANDS — commands whose path args we validate.
_WRITE_PATH_COMMANDS = {'rm', 'rmdir', 'mv', 'cp', 'mkdir', 'touch', 'sed'}
_READ_PATH_COMMANDS = {
    'cd', 'ls', 'find', 'cat', 'head', 'tail', 'sort', 'uniq', 'wc', 'cut',
    'paste', 'column', 'tr', 'file', 'stat', 'diff', 'awk', 'strings',
    'hexdump', 'od', 'base64', 'nl', 'grep', 'rg', 'git', 'jq',
    'sha256sum', 'sha1sum', 'md5sum',
}
_SUPPORTED_PATH_COMMANDS = _WRITE_PATH_COMMANDS | _READ_PATH_COMMANDS

# Process substitution / dangerous redirection regex (checkPathConstraints:1028).
_PROC_SUB_RE = re.compile(r'>>\s*>\s*\(|>\s*>\s*\(|<\s*\(')

# Glob metacharacters (pathValidation.ts GLOB_PATTERN_REGEX, simplified).
_GLOB_RE = re.compile(r'[*?\[\]{}]')


def _expand_tilde(path: str) -> str:
    """expandTilde: only bare ~ and ~/ resolve to $HOME (other variants are
    rejected by _validate_path as unexpanded)."""
    if path == '~' or path.startswith('~/'):
        return os.path.expanduser('~') + path[1:]
    return path


def is_dangerous_removal_path(resolved_path: str) -> bool:
    """isDangerousRemovalPath (pathValidation.ts): true for paths whose removal
    would be catastrophic — '/', a direct child of '/', a Windows drive root
    (``C:\\``) or direct child of one (``C:\\Windows``), '*'/'/*' globs, or the
    home directory.

    The Windows-drive branches matter because the PowerShell tool (Windows-only)
    routes its removal checks through here; without them `Remove-Item C:\\Windows`
    downgraded from deny to ask. Applied unconditionally, matching the upstream
    source — a resolved POSIX path is absolute ('/...') so it never matches the
    drive regexes."""
    fwd = re.sub(r'/+', '/', resolved_path.replace('\\', '/'))
    if fwd == '*' or fwd.endswith('/*'):
        return True
    normalized = fwd if fwd == '/' else fwd.rstrip('/')
    if normalized == '/' or not normalized:
        return True
    # Windows drive root: C:  or  C:/   (WINDOWS_DRIVE_ROOT_REGEX)
    if re.match(r'^[A-Za-z]:/?$', normalized):
        return True
    home = re.sub(r'/+', '/', os.path.expanduser('~').replace('\\', '/')).rstrip('/')
    if normalized == home:
        return True
    # Direct children of root: /usr, /tmp, /etc (but not /usr/local).
    if os.path.dirname(normalized) == '/':
        return True
    # Direct child of a Windows drive root: C:/Windows, C:/Program Files
    # (WINDOWS_DRIVE_CHILD_REGEX). No trailing '/…' so C:/a/b is not matched.
    if re.match(r'^[A-Za-z]:/[^/]+$', normalized):
        return True
    return False


def _filter_out_flags(args: List[str]) -> List[str]:
    """filterOutFlags: positional args, honoring the POSIX `--` end-of-options
    delimiter (after `--`, everything is positional even if it starts with -)."""
    result: List[str] = []
    after_dd = False
    for arg in args:
        if after_dd:
            result.append(arg)
        elif arg == '--':
            after_dd = True
        elif not arg.startswith('-'):
            result.append(arg)
    return result


def _extract_paths(command: str, args: List[str]) -> List[str]:
    """PATH_EXTRACTORS[command]: pull the path operands for a path command."""
    if command == 'cd':
        return [os.path.expanduser('~')] if not args else [' '.join(args)]
    if command == 'ls' or command == 'find':
        paths = _filter_out_flags(args)
        return paths if paths else ['.']
    if command == 'tr':
        has_delete = any(a == '-d' or a == '--delete' or (a.startswith('-') and 'd' in a) for a in args)
        non_flags = _filter_out_flags(args)
        return non_flags[1 if has_delete else 2:]  # skip SET1 / SET1+SET2
    # Default (rm/rmdir/mv/cp/mkdir/touch/cat/head/... and pattern commands):
    # filter flags. For grep/rg/sed/git/jq this slightly over-includes the
    # pattern/subcommand token, which only ever resolves inside cwd (allowed),
    # so it never causes a false block.
    return _filter_out_flags(args)


def _validate_path(path: str, op_type: str) -> Optional[str]:
    """validatePath: reject tilde-variants / shell-expansion / UNC, block globs
    in write ops, and require containment in the allowed working directory.
    Returns a deny/ask reason string, or None if allowed."""
    clean = _expand_tilde(path.strip().strip('\'"'))
    if not clean:
        return None
    # UNC network paths.
    if clean.startswith('\\\\') or clean.startswith('//') and len(clean) > 2 and clean[2] != '/':
        return 'UNC network paths require manual approval'
    # Unexpanded tilde variants (~user, ~+, ~-, ~N).
    if clean.startswith('~'):
        return 'Tilde expansion variants (~user, ~+, ~-) in paths require manual approval'
    # Shell expansion syntax (TOCTOU): $VAR, ${}, $(), %VAR%, =cmd.
    if '$' in clean or '%' in clean or clean.startswith('='):
        return 'Shell expansion syntax in paths requires manual approval'
    # Globs: blocked outright for write/create (write tools use paths literally).
    if _GLOB_RE.search(clean):
        if op_type in ('write', 'create'):
            return 'Glob patterns are not allowed in write operations. Please specify an exact file path.'
        # Read globs: validate the base directory they would expand in.
        clean = os.path.dirname(clean) or '.'
    full = clean if os.path.isabs(clean) else os.path.join(get_cwd(), clean)
    full = os.path.normpath(full)
    if not is_in_directory(full, get_original_cwd()):
        return (
            f"'{full}' is outside the allowed working directory for this "
            f"session ({get_original_cwd()})."
        )
    return None


def _extract_output_redirections(command: str):
    """extractOutputRedirections (commands.ts): find `>`/`>>`/`&>` targets.
    Returns (redirections, has_dangerous) where has_dangerous flags a target
    carrying shell-expansion syntax that can't be validated statically."""
    redirections = []
    has_dangerous = False
    # Match an optional leading FD, the operator, optional space, then a target
    # token. Skip pure FD-dup forms (2>&1) — the (&\d) target isn't a file.
    for m in re.finditer(r'(?:&|\d+)?(>>|&>|>)\s*([^\s;|&()<>]+)', command):
        op_raw, target = m.group(1), m.group(2)
        if target.startswith('&'):  # FD duplication (>&1), not a file target
            continue
        if '$' in target or '`' in target or '%' in target:
            has_dangerous = True
            continue
        operator = '>>' if op_raw == '>>' else '>'
        redirections.append((target, operator))
    return redirections, has_dangerous


def _validate_output_redirections(redirections, compound_has_cd: bool) -> Optional[PermissionResult]:
    """validateOutputRedirections: cd+redirect compound → ask; each non-/dev/null
    target must be inside the allowed dir. Returns a PermissionResult to short-
    circuit, or None to pass through."""
    if compound_has_cd and redirections:
        return PermissionResult(
            behavior='ask',
            message="Commands that change directories and write via output "
                    "redirection require explicit approval (the final working "
                    "directory can't be determined safely).",
        )
    for target, _op in redirections:
        if target == '/dev/null':
            continue
        reason = _validate_path(target, 'create')
        if reason:
            return PermissionResult(
                behavior='ask',
                message=f"Output redirection to '{target}' was blocked: {reason}",
            )
    return None


def check_path_constraints(command: str) -> PermissionResult:
    """checkPathConstraints port → a PermissionResult.

    'deny'/'ask' when the command would write outside the session directory,
    remove a critical path, redirect to a disallowed target, or use process
    substitution; 'allow' (passthrough) otherwise. The caller decides what
    passthrough means (Bash maps it to its default 'ask' prompt)."""
    joined = _join_line_continuations(command)

    # Process substitution can execute arbitrary commands writing to files that
    # never appear as redirect targets — always require manual approval.
    if _PROC_SUB_RE.search(joined):
        return PermissionResult(
            behavior='ask',
            message='Process substitution (>(...) or <(...)) can execute '
                    'arbitrary commands and requires manual approval.',
        )

    redirections, has_dangerous = _extract_output_redirections(joined)
    if has_dangerous:
        return PermissionResult(
            behavior='ask',
            message='Shell expansion syntax in redirection paths requires manual approval.',
        )

    subcommands = get_command_subcommands(command)
    compound_has_cd = any(tokens and _basename(tokens[0]) == 'cd' for tokens in subcommands)

    redir = _validate_output_redirections(redirections, compound_has_cd)
    if redir is not None:
        return redir

    for tokens in subcommands:
        if not tokens:
            continue
        base = _basename(tokens[0])
        if base not in _SUPPORTED_PATH_COMMANDS:
            continue
        args = tokens[1:]

        # Dangerous removal (rm/rmdir) — flagged even though it would also be a
        # normal prompt, so the user sees WHY (checkDangerousRemovalPaths).
        if base in ('rm', 'rmdir'):
            for p in _extract_paths(base, args):
                clean = _expand_tilde(p.strip().strip('\'"'))
                full = clean if os.path.isabs(clean) else os.path.join(get_cwd(), clean)
                if is_dangerous_removal_path(os.path.normpath(full)):
                    return PermissionResult(
                        behavior='ask',
                        message=f"Dangerous {base} operation detected: '{full}'. "
                                "This would remove a critical directory and requires "
                                "explicit approval.",
                    )

        op_type = 'write' if base in _WRITE_PATH_COMMANDS else 'read'
        for p in _extract_paths(base, args):
            reason = _validate_path(p, op_type)
            if reason:
                return PermissionResult(
                    behavior='ask',
                    message=f"{base}: {reason}",
                )

    return PermissionResult(behavior='allow', message='All path commands validated')


# === Output Truncation (utils/shell/outputLimits.ts + BashTool/utils.ts) ===
#
# Port of formatOutput: CHARACTER-based, HEAD-ONLY truncation. The cap is
# BASH_MAX_OUTPUT_DEFAULT (30000), overridable via the BASH_MAX_OUTPUT_LENGTH
# env var and clamped to BASH_MAX_OUTPUT_UPPER_LIMIT (150000). When content
# exceeds the cap we keep the leading slice and append a count of the newlines
# discarded from the tail.

BASH_MAX_OUTPUT_UPPER_LIMIT = 150_000
BASH_MAX_OUTPUT_DEFAULT = 30_000


def get_max_output_length() -> int:
    """Resolve the output cap (outputLimits.ts getMaxOutputLength).

    Reads BASH_MAX_OUTPUT_LENGTH if set to a valid non-negative integer,
    falling back to BASH_MAX_OUTPUT_DEFAULT, then clamps to the upper limit.
    """
    raw = os.environ.get('BASH_MAX_OUTPUT_LENGTH')
    if raw is None:
        value = BASH_MAX_OUTPUT_DEFAULT
    else:
        try:
            value = int(raw.strip())
            if value < 0:
                value = BASH_MAX_OUTPUT_DEFAULT
        except (ValueError, AttributeError):
            value = BASH_MAX_OUTPUT_DEFAULT
    return min(value, BASH_MAX_OUTPUT_UPPER_LIMIT)


def format_output(content: str) -> tuple:
    """Return (total_lines, content) with NO head truncation.

    The original truncates the inline preview but PERSISTS the full output to
    disk (BashTool.tsx maybePersistLargeToolResult) and exposes its path. This
    port persists via the query() tool-result layer (process_tool_result_content,
    keyed on BashTool.max_result_size_chars) — but that layer only sees what the
    tool returns, so pre-truncating here dropped the tail irrecoverably. Passing
    the full content through lets that layer persist + preview it, so large
    output is recoverable (the model can Read the <persisted-output> path)
    instead of silently lost.
    """
    total_lines = content.count('\n') + 1
    return total_lines, content


def _interpret_command_result(command: str, exit_code: int) -> Dict[str, Any]:
    """Port of BashTool/commandSemantics.ts interpretCommandResult.

    Many commands use a non-zero exit to convey INFORMATION, not failure —
    marking them as errors misleads the model (e.g. grep exit 1 = no match, not
    a crash). Base command = first word of the LAST pipeline/list segment (the
    one that sets $?); heuristic, per the original ("may get it super wrong").
    """
    segments = re.split(r'\|\||&&|[|;&]', command)
    last = segments[-1].strip() if segments else command
    base = (last.split() or [''])[0].rsplit('/', 1)[-1]

    if base in ('grep', 'rg'):
        return {'is_error': exit_code >= 2, 'message': 'No matches found' if exit_code == 1 else None}
    if base == 'find':
        return {'is_error': exit_code >= 2, 'message': 'Some directories were inaccessible' if exit_code == 1 else None}
    if base == 'diff':
        return {'is_error': exit_code >= 2, 'message': 'Files differ' if exit_code == 1 else None}
    if base in ('test', '['):
        return {'is_error': exit_code >= 2, 'message': 'Condition is false' if exit_code == 1 else None}
    return {
        'is_error': exit_code != 0,
        'message': f'Command failed with exit code {exit_code}' if exit_code != 0 else None,
    }


# =====================================================================
# LOCAL SUBSTRATE (PRESERVED) — async subprocess execution
# =====================================================================
#
# The block below is CodeMason's own command-execution mechanism. It is NOT
# part of the port and is kept verbatim: timeout / abort / SIGKILL /
# background-pipe handling and the cwd-reset logic that the upstream tool gets
# from its own shell infrastructure.

class PersistentShell:
    """Singleton persistent shell for command execution (local substrate)."""

    _instance: Optional['PersistentShell'] = None

    def __init__(self):
        self._cwd = get_cwd()

    @classmethod
    def get_instance(cls) -> 'PersistentShell':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    async def exec(
        self,
        command: str,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        abort_event: Optional[asyncio.Event] = None,
        description: Optional[str] = None,
        tool_use_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a command and return the result.

        Reads stdout/stderr INCREMENTALLY into buffers (not communicate()) so a
        timeout still has every byte produced so far and can hand the live process
        to the background (auto-background) instead of killing the work. A watcher
        closes the pipe transports after the main process exits so the drains reach
        EOF even when a `cmd &` grandchild holds a pipe open.
        """
        timeout_sec = timeout_ms / 1000

        # cwd PERSISTENCE: the original keeps one shell alive so `cd` sticks; this
        # port spawns a fresh subprocess per command, so we append a `pwd -P`
        # sentinel (written to a temp file, off stdout/stderr) and adopt the
        # resulting directory. Without this, `cd subdir` is silently lost on the
        # next call even though the tool prompt promises "persists between commands".
        import shlex as _shlex
        import tempfile as _tempfile
        _pwd_fd, _pwd_file = _tempfile.mkstemp(prefix='cm-pwd-')
        os.close(_pwd_fd)
        # The sentinel runs inside bash. On Windows the temp path is a Windows
        # path (C:\...\cm-pwd-x); bash's redirect target must be its POSIX form
        # (/c/.../cm-pwd-x), else the redirect writes to the wrong place.
        _pwd_redirect = _win_to_posix(_pwd_file) if is_windows() else _pwd_file
        wrapped = (
            f'{command}\n'
            '__CM_EC=$?\n'
            f'command pwd -P > {_shlex.quote(_pwd_redirect)} 2>/dev/null\n'
            'exit $__CM_EC\n'
        )

        try:
            # Faithful equivalent of the original's stdin handling
            # (bashProvider.ts: detached shell + `< /dev/null` redirect via
            # shouldAddStdinRedirect). codemason spawns a fresh subprocess
            # per command, so setting the spawn's stdin to /dev/null covers
            # every command in a pipe at once — no `rearrangePipeCommand`
            # special-case needed. Interactive commands get EOF and fail
            # fast instead of inheriting the user's TTY and hanging.
            #
            # POSIX: create_subprocess_shell runs `wrapped` in /bin/sh.
            # Windows: create_subprocess_shell would route through cmd.exe, which
            # can't parse the POSIX body ($?, pwd -P, /dev/null). Route through
            # git-bash's bash.exe (`bash -c wrapped`) so the same body runs
            # unchanged. spawn_group_kwargs() swaps start_new_session (POSIX) for
            # CREATE_NEW_PROCESS_GROUP (Windows).
            _common = dict(
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                # PYTHONUNBUFFERED: stdout is a pipe, not a tty, so a python
                # child block-buffers; when this command times out and moves
                # to the background, its output file would show nothing while
                # it visibly works. Dropped by mistake in the TaskOutput revert.
                env={**os.environ, 'TERM': 'dumb', 'PYTHONUNBUFFERED': '1'},
                **_spawn_group_kwargs(),  # original: detached: true
            )
            # Everything spawned from here on dies with this process, however this
            # process dies -- taskkill /F runs no atexit handler, and this shell was
            # outliving the session. Before the spawn on purpose: a job member's
            # children join the job at creation, so there is no window for a
            # grandchild to start outside it. No-op off Windows.
            reap_children_with_this_process()

            _argv = _win_shell_argv(wrapped)
            if _argv is not None:
                process = await asyncio.create_subprocess_exec(*_argv, **_common)
            else:
                process = await asyncio.create_subprocess_shell(wrapped, **_common)


            interrupted = False

            async def _force_close_pipes_after_exit():
                """After process exits, close pipe transports to unblock communicate().

                Python's asyncio Process.wait()/communicate() won't return until
                ALL pipe transports are disconnected. Background processes (e.g.
                'cmd &') inherit pipes and keep them open. We detect process exit
                via returncode polling, then close pipe transports from our side.
                """
                while process.returncode is None:
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.5)  # let remaining data flush
                for fd in [1, 2]:
                    try:
                        transport = process._transport.get_pipe_transport(fd)
                        if transport and not transport.is_closing():
                            transport.close()
                    except (AttributeError, OSError):
                        pass

            async def _abort_monitor():
                nonlocal interrupted
                if not abort_event:
                    return
                await abort_event.wait()
                interrupted = True
                # ESC cancel = "stop now, discard the result"
                # interruptBehavior 'cancel'): SIGKILL the whole group so a child
                # that traps or ignores SIGTERM can't stall teardown — a trapped
                # SIGTERM leaves the main process alive, its pipes open, and the
                # drains below blocking, which is exactly the sluggish-cancel feel.
                # No sig -> killpg(SIGKILL) on POSIX, taskkill /F /T on Windows
                # (kill_process_tree). Mirrors treeKill(pid, 'SIGKILL').
                _kill_process_tree(process)

            eof_task = asyncio.create_task(_force_close_pipes_after_exit())
            abort_task = asyncio.create_task(_abort_monitor()) if abort_event else None

            out_buf, err_buf = bytearray(), bytearray()

            # Cap what we hold in RAM per stream. Without this, a command that
            # streams gigabytes (`yes`, `cat huge`) grows the buffer unbounded
            # before the caller ever sees the result → OOM. We keep reading past
            # the cap (so the pipe never blocks and the process can finish) but
            # discard the overflow, appending a one-time truncation marker.
            _MAX_CAPTURE = 20 * 1024 * 1024  # 20 MiB per stream

            async def _drain(reader, buf):
                if reader is None:
                    return
                capped = False
                while True:
                    chunk = await reader.read(4096)
                    if not chunk:
                        break
                    if len(buf) < _MAX_CAPTURE:
                        buf.extend(chunk[:_MAX_CAPTURE - len(buf)])
                        if len(buf) >= _MAX_CAPTURE and not capped:
                            capped = True
                            buf.extend(b'\n... [output truncated: exceeded 20 MiB] ...\n')
                    # else: drain and discard to keep the pipe from blocking.

            d_out = asyncio.create_task(_drain(process.stdout, out_buf))
            d_err = asyncio.create_task(_drain(process.stderr, err_buf))

            try:
                done, pending = await asyncio.wait({d_out, d_err}, timeout=timeout_sec)

                if pending:
                    # TIMEOUT — hand the still-running process to the background so
                    # its work (and all output so far) is preserved, instead of
                    # killing it. Fail-safe: any handoff error falls back to SIGKILL
                    # and the original timed-out result.
                    try:
                        for _d in pending:
                            _d.cancel()
                        from ..tasks.local_shell_task import adopt_timed_out_process
                        combined = bytes(out_buf) + (b'\n' if out_buf and err_buf else b'') + bytes(err_buf)
                        task_id, out_path = adopt_timed_out_process(
                            process, combined, description=description, tool_use_id=tool_use_id,
                        )
                        return {
                            'stdout': (f'Command timed out after {timeout_sec}s and was moved to '
                                       f'the background.\nTask ID: {task_id}\n'
                                       f'Output is being written to: {out_path}\n'
                                       f'Use Read to view it; you will be notified on completion.'),
                            'stderr': '',
                            'code': 0,
                            'interrupted': False,
                            'backgrounded': True,
                        }
                    except Exception:
                        _kill_process_tree(process)  # no sig: SIGKILL on POSIX
                        try:
                            await asyncio.wait_for(asyncio.gather(d_out, d_err, return_exceptions=True), timeout=3.0)
                        except (asyncio.TimeoutError, Exception):
                            pass
                        try:
                            await asyncio.wait_for(process.wait(), timeout=3.0)  # reap — no zombie
                        except (asyncio.TimeoutError, Exception):
                            pass
                        return {
                            'stdout': _decode_output(out_buf),
                            'stderr': _decode_output(err_buf)
                            + f'\nCommand timed out after {timeout_sec}s',
                            'code': -1,
                            'interrupted': True,
                        }

                # Normal completion (or abort): both drains reached EOF.
                returncode = await process.wait()

                # Adopt the post-command cwd so `cd` persists to the next command.
                try:
                    with open(_pwd_file, 'r', encoding='utf-8', errors='replace') as _pf:
                        _new_cwd = _pf.read().strip()
                    # `pwd -P` under bash prints a POSIX path. On Windows convert
                    # it back (/c/foo -> C:\foo) so os.path.isdir and the next
                    # spawn's cwd= (a Windows path) both work.
                    if _new_cwd and is_windows():
                        _new_cwd = _posix_to_win(_new_cwd)
                    if _new_cwd and os.path.isdir(_new_cwd):
                        self._cwd = _new_cwd
                        # Original setCwd(): the shell's cwd IS the session cwd, so
                        # file tools (which resolve relative paths via get_cwd)
                        # follow `cd` too.
                        _set_global_cwd(_new_cwd)
                except OSError:
                    pass

                return {
                    'stdout': _decode_output(out_buf),
                    'stderr': _decode_output(err_buf),
                    'code': returncode,
                    'interrupted': interrupted,
                }

            finally:
                # On handoff the background worker owns the process now; still cancel
                # OUR watcher/drains so they don't race its pipe reads.
                for task in [eof_task, abort_task, d_out, d_err]:
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass
                try:
                    os.unlink(_pwd_file)
                except OSError:
                    pass

        except Exception as e:
            try:
                os.unlink(_pwd_file)
            except OSError:
                pass
            return {
                'stdout': '',
                'stderr': str(e),
                'code': -1,
                'interrupted': False,
            }

    def set_cwd(self, path: str) -> None:
        """Set the current working directory."""
        self._cwd = path

    @property
    def cwd(self) -> str:
        return self._cwd


# =====================================================================
# END LOCAL SUBSTRATE
# =====================================================================


# === Bash Tool (BashTool.tsx) ===

class BashTool(Tool):
    """Executes a given bash command and returns its output."""

    name = 'Bash'
    input_schema = BashInput
    # BashTool.maxResultSizeChars (lower than the 50k default → persists sooner).
    max_result_size_chars = 30_000

    # getSimplePrompt() resolved for the external build, ported verbatim in
    # constants/tool_prompts.py and reused here (not duplicated).
    DESCRIPTION = BASH_DESCRIPTION

    @classmethod
    async def description(cls, command: str = '', description: str = '', **kwargs) -> str:
        """Per-call description (BashTool.tsx passes `description || command`)."""
        return description or command or 'Execute bash command'

    @classmethod
    async def prompt(cls) -> str:
        return cls.DESCRIPTION

    @classmethod
    def is_read_only(cls) -> bool:
        return False

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def needs_permissions(cls) -> bool:
        return True

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return 'Bash'

    @classmethod
    async def validate_input(
        cls,
        input_data: Dict[str, Any],
        context: Any = None,
    ) -> ValidationResult:
        """Validate the command's SHAPE only (BashTool.tsx validateInput, which
        ports just the blocked-sleep check — the security model lives in
        check_permissions, not here). We keep the empty-command guard; the
        path/redirection/removal security has moved to check_permissions so it
        runs in the permission layer and is bypassed under
        --dangerously-skip-permissions exactly like the original."""
        command = input_data.get('command', '') or ''
        if not command.strip():
            return ValidationResultFailure(
                result=False,
                message='Command cannot be empty',
            )
        return ValidationResultSuccess()

    @classmethod
    async def check_permissions(
        cls,
        input_data: Dict[str, Any],
        context: Any = None,
    ) -> PermissionResult:
        """BashTool checkPermissions (bashToolHasPermission → checkPathConstraints
        core). Runs the path-constraint security model; a 'deny' hard-blocks, an
        'ask' prompts with the security reason. A clean command still 'ask's —
        codemason has no allowlist/classifier to auto-approve it, so Bash always
        prompts when permissions aren't skipped (the path checks add the WHY)."""
        command = input_data.get('command', '') or ''
        result = check_path_constraints(command)
        # Bash's check_path_constraints only ever returns 'ask' or 'allow'/
        # passthrough (never 'deny' — that's the PowerShell variant), but forward
        # a 'deny' too if one is ever introduced, rather than silently downgrading
        # it to 'ask'. Everything non-blocking still falls through to Bash's
        # default prompt (no auto-approve here).
        if result.behavior in ('deny', 'ask'):
            return result
        return PermissionResult(behavior='ask')

    @classmethod
    def get_activity_description(cls, input_data):
        """"Running <description or command>" (BashTool getActivityDescription).

        The model's own one-line description wins over the command text — it is
        what the command is FOR, which is the useful thing to read while waiting.
        """
        from ..utils.formatting import TOOL_SUMMARY_MAX_LENGTH, truncate
        if not isinstance(input_data, dict) or not input_data.get('command'):
            return 'Running command'
        description = input_data.get('description')
        desc = description or truncate(str(input_data['command']), TOOL_SUMMARY_MAX_LENGTH)
        return f'Running {desc}'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """Render command for display (BashTool.tsx renderToolUseMessage)."""
        command = input_data.get('command', '') if isinstance(input_data, dict) else ''

        # Clean up HEREDOC patterns for display.
        if "$(cat <<'EOF'" in command:
            match = re.match(
                r"^(.*?)\"?\$\(cat <<'EOF'\n([\s\S]*?)\n\s*EOF\n\s*\)\"(.*)$",
                command,
            )
            if match:
                prefix = match.group(1)
                content = match.group(2)
                suffix = match.group(3) or ''
                command = f'{prefix.strip()} "{content.strip()}"{suffix.strip()}'

        command = command.strip()
        if len(command) > _MAX_DISPLAY_COMMAND_CHARS:
            # Show the END as well as the beginning. call() runs the whole
            # string, and this rendering is the entire basis for the user's
            # consent — a trailing '...' after 400 plausible characters hid
            # whatever came next, which is exactly where an appended
            # `; curl ... | sh` would sit. The heredoc rewrite above runs before
            # this, so it can shorten the visible prefix and push more past the
            # cut. Keeping the tail visible closes that.
            _tail_chars = 120
            _head = command[:_MAX_DISPLAY_COMMAND_CHARS - _tail_chars]
            _tail = command[-_tail_chars:]
            _hidden = len(command) - len(_head) - len(_tail)
            command = f'{_head}\n… ({_hidden} chars hidden) …\n{_tail}'
        # Shell prompt on the first line only; continuation lines are indented under
        # it (a heredoc body or a wrapped pipeline is not a second command).
        lines = [line for line in command.split('\n') if line.strip()]
        return '\n'.join(f"$ {line}" if i == 0 else f"  {line}"
                         for i, line in enumerate(lines))

    @classmethod
    def render_result_for_assistant(cls, result: BashOutput) -> str:
        """Render result for the assistant (BashTool.tsx mapToolResult...)."""
        error_message = result.stderr.strip()

        if result.interrupted:
            if error_message:
                error_message += '\n'
            error_message += '<error>Command was aborted before completion</error>'

        stdout = result.stdout.strip()
        has_both = stdout and error_message

        return f"{stdout}{chr(10) if has_both else ''}{error_message.strip()}"

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        """Execute the bash command (local substrate)."""
        command = input_data.get('command', '')
        timeout = input_data.get('timeout') or DEFAULT_TIMEOUT_MS
        # Clamp to the documented maximum (timeouts.ts getMaxBashTimeoutMs).
        if timeout > MAX_TIMEOUT_MS:
            timeout = MAX_TIMEOUT_MS

        shell = PersistentShell.get_instance()

        # run_in_background: spawn detached, stream output to a disk file, and
        # return immediately with the task id + output path (LocalShellTask
        # mechanism). The model reads the output later via the Read tool.
        if input_data.get('run_in_background'):
            from ..tasks.local_shell_task import spawn_shell_task
            # LocalShellTask spawnShellTask labels the task with the tool's
            # `description` input field (falling back to the command) — that
            # label is what the completion notification's <summary> echoes. The
            # tool_use id is threaded through so the <task-notification> carries
            # the optional <tool-use-id> line the original emits.
            description = input_data.get('description') or command
            tool_use_id = getattr(context, 'tool_use_id', None) or getattr(context, 'message_id', None)
            task_id, output_path = spawn_shell_task(
                command, shell.cwd, description=description, tool_use_id=tool_use_id
            )
            info = (
                f"Command running in background with ID: {task_id}. "
                f"Output is being written to: {output_path}\n"
                f"Use Read to read the output later."
            )
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=info,
                data={'backgroundTaskId': task_id, 'output_path': output_path},
            )
            return

        # Execute command with abort signal from context.
        abort_event = None
        if hasattr(context, 'abort_controller'):
            abort_event = context.abort_controller
        # Say that the wait is bounded, and by what. A foreground command runs
        # up to `timeout` before it is moved to the background — worth knowing
        # before deciding to interrupt. Set and cleared HERE, the one call site,
        # rather than inside exec(), which has six separate returns.
        from ..utils.tokens import set_wait_note
        set_wait_note('running, moves to background at %ds' % int(timeout / 1000))
        try:
            result = await shell.exec(
                command, timeout, abort_event=abort_event,
                description=input_data.get('description') or command,
                tool_use_id=getattr(context, 'tool_use_id', None) or getattr(context, 'message_id', None),
            )
        finally:
            set_wait_note(None)

        stdout = (result.get('stdout', '') or '').strip()
        stderr = (result.get('stderr', '') or '').strip()

        _code = result.get('code', 0)
        _interp = _interpret_command_result(command, _code)
        if _interp['is_error']:
            stderr += f"\n{_interp['message']}"
        elif _code != 0 and _interp.get('message') and not stdout and not stderr:
            # Non-failure non-zero (grep no-match, `[ ]` false, diff differ) with
            # no other output — surface the note so an empty result isn't read as
            # a silent success, but do NOT flag it as an error.
            stderr += _interp['message']

        # Check if shell cwd is outside original directory; reset if so.
        if not is_in_directory(shell.cwd, get_original_cwd()):
            shell.set_cwd(get_original_cwd())
            # Reset the PROCESS-wide cwd too, not just this shell instance.
            # PersistentShell.exec adopts the post-command directory into both
            # (self._cwd and _set_global_cwd), but set_cwd writes only the
            # instance attribute — so after `cd /tmp/x` the containment reset
            # fixed the shell while get_cwd() still returned /tmp/x. Every file
            # tool resolves relative paths through get_cwd(), so a later
            # Write("setup.sh") was shown to the user as "setup.sh" and landed
            # in /tmp/x/setup.sh, outside the project and outside what the
            # prompt displayed. The PowerShell tool already does both.
            _set_global_cwd(get_original_cwd())
            stderr = f"{stderr.strip()}\nShell cwd was reset to {get_original_cwd()}"
            log_event('bash_tool_reset_to_original_dir', {})

        # Format output.
        stdout_lines, stdout_content = format_output(stdout)
        stderr_lines, stderr_content = format_output(stderr)

        data = BashOutput(
            stdout=stdout_content,
            stdout_lines=stdout_lines,
            stderr=stderr_content,
            stderr_lines=stderr_lines,
            interrupted=result.get('interrupted', False),
        )

        yield ToolResultSuccess(
            type='result',
            result_for_assistant=cls.render_result_for_assistant(data),
            data=data,
        )
