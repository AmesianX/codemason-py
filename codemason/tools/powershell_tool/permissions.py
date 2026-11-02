"""PowerShell tool permission engine (adaptation of powershellPermissions.ts).

The final security decision: integrates the security validators, git-safety,
read-only allowlist, and path validation into one allow/ask/deny result.

ADAPTED, not 1:1: the source's flow is built around Claude Code's permission-RULE
engine (exact/prefix/per-subcommand deny/ask/allow rules, rule suggestions).
CodeMason has NO rule engine (state/app_state.py) — its Bash tool likewise runs
a pure security-decision flow. So all rule-matching is dropped and only the
security decisions remain, kept in the source's collect-then-reduce shape
(deny > ask > allow > passthrough) so an earlier ask can never mask a later deny.
"""

import os
import re

from ...utils.state import get_cwd
from ...utils.powershell.parser import (
    ParsedPowerShellCommand,
    PS_TOKENIZER_DASH_CHARS,
    get_file_redirections,
)
from ...utils.shell.read_only_command_validation import contains_vulnerable_unc_path
from ..base import PermissionResult
from .git_safety import is_dot_git_path_ps, is_git_internal_path_ps
from .mode_validation import is_symlink_creating_command
from .path_validation import (
    check_path_constraints,
    dangerous_removal_deny,
    is_dangerous_removal_raw_path,
)
from .powershell_security import powershell_command_is_safe
from .read_only_validation import (
    is_cwd_changing_cmdlet,
    is_read_only_command,
    resolve_to_canonical,
)

_PS_ASSIGN_PREFIX_RE = re.compile(r"^\$[\w:]+\s*(?:[+\-*/%]|\?\?)?\s*=\s*")

# Cmdlets that can place a file at a caller-specified path (git-internal guard).
_GIT_SAFETY_WRITE_CMDLETS = frozenset(
    {"new-item", "set-content", "add-content", "out-file", "copy-item", "move-item",
     "rename-item", "expand-archive", "invoke-webrequest", "invoke-restmethod",
     "tee-object", "export-csv", "export-clixml"}
)

# External archive extractors (opaque contents → git TOCTOU).
_GIT_SAFETY_ARCHIVE_EXTRACTORS = frozenset(
    {"tar", "tar.exe", "bsdtar", "bsdtar.exe", "unzip", "unzip.exe", "7z", "7z.exe",
     "7za", "7za.exe", "gzip", "gzip.exe", "gunzip", "gunzip.exe", "expand-archive"}
)

_NON_FS_PROVIDER_RE = re.compile(
    r"^(?:[\w.]+\\)?(env|hklm|hkcu|function|alias|variable|cert|wsman|registry)::?", re.IGNORECASE
)


def is_current_directory_bare_git_repo() -> bool:
    """True if cwd looks like a bare git repo (HEAD/objects/refs without a valid
    .git/HEAD) — git would run hooks from cwd (isCurrentDirectoryBareGitRepo)."""
    cwd = get_cwd()
    git_path = os.path.join(cwd, ".git")
    try:
        if os.path.isfile(git_path):
            return False  # worktree/submodule
        if os.path.isdir(git_path):
            head = os.path.join(git_path, "HEAD")
            if os.path.isfile(head):
                return False  # normal repo
    except OSError:
        pass
    try:
        if os.path.isfile(os.path.join(cwd, "HEAD")):
            return True
    except OSError:
        pass
    try:
        if os.path.isdir(os.path.join(cwd, "objects")):
            return True
    except OSError:
        pass
    try:
        if os.path.isdir(os.path.join(cwd, "refs")):
            return True
    except OSError:
        pass
    return False


def _ask(message: str) -> PermissionResult:
    return PermissionResult(behavior="ask", message=message)


def _extract_provider_value(arg: str) -> str:
    s = arg
    if len(s) > 0 and s[0] in PS_TOKENIZER_DASH_CHARS:
        colon_idx = s.find(":", 1)
        if colon_idx > 0:
            s = s[colon_idx + 1:]
    return s.replace("`", "")


def _provider_or_unc_decision(arg: str):
    value = _extract_provider_value(arg)
    if _NON_FS_PROVIDER_RE.match(value):
        return _ask(f"Command argument '{arg}' uses a non-filesystem provider path and requires approval")
    if contains_vulnerable_unc_path(value):
        return _ask(f"Command argument '{arg}' contains a UNC path that could trigger network requests")
    return None


def _all_command_elements(parsed: ParsedPowerShellCommand):
    """Every CommandAst element (statements + nested), with its statement."""
    out = []
    for statement in parsed.statements:
        for cmd in statement.commands:
            if cmd.element_type == "CommandAst":
                out.append((cmd, statement))
        if statement.nested_commands:
            for cmd in statement.nested_commands:
                out.append((cmd, statement))
    return out


def powershell_tool_has_permission(command: str, parsed: ParsedPowerShellCommand) -> PermissionResult:
    """Decide allow/ask/deny for a PowerShell command (security-only adaptation).

    Collect-then-reduce (deny > ask > allow > else prompt). No rule matching —
    CodeMason has no permission-rule engine.
    """
    command = command.strip()
    if not command:
        return PermissionResult(behavior="allow", message="Empty command is safe")

    # Parse-failed: best-effort dangerous-removal hard-deny, else ask.
    if not parsed.valid:
        backtick_stripped = re.sub(r"`[\r\n]+\s*", "", command).replace("`", "")
        for fragment in re.split(r"[;|\n\r{}()&]+", backtick_stripped):
            frag = fragment.strip()
            if not frag:
                continue
            normalized = frag
            while True:
                m = _PS_ASSIGN_PREFIX_RE.match(normalized)
                if not m:
                    break
                normalized = normalized[m.end():]
            normalized = re.sub(r"^[&.]\s+", "", normalized)
            tokens = normalized.split()
            first_tok = re.sub(r"^['\"]|['\"]\Z", "", tokens[0]) if tokens else ""
            if resolve_to_canonical(first_tok) == "remove-item":
                for arg in tokens[1:]:
                    if arg and arg[0] in PS_TOKENIZER_DASH_CHARS:
                        continue
                    if is_dangerous_removal_raw_path(arg):
                        return dangerous_removal_deny(arg)
        return _ask("Command contains malformed syntax that cannot be parsed and requires approval")

    decisions = []

    # Raw-command UNC.
    if contains_vulnerable_unc_path(command):
        decisions.append(_ask("Command contains a UNC path that could trigger network requests"))

    # Security validators (subexpr / script blocks / encoded / cradles / etc.).
    safety = powershell_command_is_safe(command, parsed)
    if safety.behavior != "passthrough":
        decisions.append(_ask(safety.message or "This command contains patterns that could pose security risks"))

    # using / #Requires — load external code, invisible to the block walk.
    if parsed.has_using_statements:
        decisions.append(_ask("Command contains a `using` statement that may load external code"))
    if parsed.has_script_requirements:
        decisions.append(_ask("Command contains a `#Requires` directive that may trigger module loading"))

    elements = _all_command_elements(parsed)

    # Provider / UNC arg scan (first match).
    for cmd, _stmt in elements:
        hit = None
        for arg in cmd.args:
            hit = _provider_or_unc_decision(arg)
            if hit:
                break
        if hit:
            decisions.append(hit)
            break

    # Git-safety guards.
    has_cd_subcommand = len(elements) > 1 and any(is_cwd_changing_cmdlet(c.name) for c, _ in elements)
    has_symlink_create = len(elements) > 1 and any(is_symlink_creating_command(c) for c, _ in elements)
    has_git_subcommand = any(resolve_to_canonical(c.name) == "git" for c, _ in elements)

    if has_cd_subcommand and has_git_subcommand:
        decisions.append(_ask("Compound commands with cd/Set-Location and git require approval to prevent bare repository attacks"))

    if has_git_subcommand and is_current_directory_bare_git_repo():
        decisions.append(_ask("Git command in a directory with bare-repository indicators. Git may execute hooks from cwd."))

    if has_git_subcommand:
        def _writes_git_internal(cmd, statement) -> bool:
            for r in cmd.redirections or []:
                if is_git_internal_path_ps(r.target):
                    return True
            if resolve_to_canonical(cmd.name) not in _GIT_SAFETY_WRITE_CMDLETS:
                return False
            for a in cmd.args:
                for part in a.split(","):
                    if is_git_internal_path_ps(part):
                        return True
            if statement is not None:
                for c in statement.commands:
                    if c.element_type == "CommandAst":
                        continue
                    if is_git_internal_path_ps(c.text):
                        return True
            return False

        writes = any(_writes_git_internal(c, s) for c, s in elements)
        redir_writes = any(is_git_internal_path_ps(r.target) for r in get_file_redirections(parsed))
        if writes or redir_writes:
            decisions.append(_ask("Command writes to a git-internal path (HEAD/objects/refs/hooks/.git) and runs git — could plant a malicious hook."))
        if any(c.name.lower() in _GIT_SAFETY_ARCHIVE_EXTRACTORS for c, _ in elements):
            decisions.append(_ask("Compound command extracts an archive and runs git — archive contents may plant bare-repository indicators."))

    # .git/ writes are dangerous even without a git subcommand.
    def _writes_dot_git(cmd) -> bool:
        for r in cmd.redirections or []:
            if is_dot_git_path_ps(r.target):
                return True
        if resolve_to_canonical(cmd.name) not in _GIT_SAFETY_WRITE_CMDLETS:
            return False
        return any(is_dot_git_path_ps(part) for a in cmd.args for part in a.split(","))

    if any(_writes_dot_git(c) for c, _ in elements) or any(is_dot_git_path_ps(r.target) for r in get_file_redirections(parsed)):
        decisions.append(_ask("Command writes to .git/ — hooks or config planted there execute on the next git operation."))

    # Path constraints (deny-capable). check_path_constraints returns allow|ask|deny.
    path_result = check_path_constraints(parsed, has_cd_subcommand)
    if path_result.behavior != "allow":
        decisions.append(path_result)

    # Read-only allowlist (allow).
    if is_read_only_command(command, parsed):
        decisions.append(PermissionResult(behavior="allow", message="Command is read-only and safe to execute"))

    # File redirections (write to arbitrary paths).
    if get_file_redirections(parsed):
        decisions.append(_ask("Command contains file redirections that could write to arbitrary paths"))

    # REDUCE: deny > ask > allow. Symlink-create disables any read-only allow
    # (a just-created link can redirect a later read).
    for d in decisions:
        if d.behavior == "deny":
            return d
    for d in decisions:
        if d.behavior == "ask":
            return d
    if not has_symlink_create:
        for d in decisions:
            if d.behavior == "allow":
                return d

    # Nothing decided → prompt (CodeMason's default for non-read-only).
    return _ask("This command requires approval")
