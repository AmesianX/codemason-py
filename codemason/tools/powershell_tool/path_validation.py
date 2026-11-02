"""PowerShell path validation (adaptation of pathValidation.ts).

Extracts filesystem paths from a parsed PowerShell command (via CMDLET_PATH_CONFIG
+ the AST) and validates they stay within the session working directory.

ADAPTED, not 1:1: the source ties into Claude Code's full permission-rule /
filesystem system (rule matching, internal editable/readable paths, sandbox
allowlist, rule suggestions). CodeMason simplified that in its Bash tool to
`_validate_path` (containment in the session dir) + `is_dangerous_removal_path`
+ a 3-state PermissionResult. This module mirrors that: it ports the PowerShell-
specific path EXTRACTION faithfully, then routes the actual check through
CodeMason's existing validators — same simplification the Bash tool made.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...utils.state import get_cwd
from ..base import PermissionResult
from ..bash_tool import _validate_path, is_dangerous_removal_path
from ...utils.powershell.parser import (
    ParsedCommandElement,
    ParsedPowerShellCommand,
    ParsedStatement,
    is_powershell_parameter,
)
from .common_parameters import COMMON_SWITCHES, COMMON_VALUE_PARAMS
from .read_only_validation import resolve_to_canonical

# PowerShell wildcards are only * ? [ ] (braces are literal — no brace expansion).
_GLOB_PATTERN_RE = re.compile(r"[*?[\]]")

# Only these AST element types are statically verifiable as a path arg.
_SAFE_PATH_ELEMENT_TYPES = frozenset({"StringConstant", "Parameter"})


@dataclass
class CmdletPathConfig:
    """Per-cmdlet parameter configuration (CmdletPathConfig)."""

    operation_type: str  # 'read' | 'write' | 'create'
    path_params: List[str] = field(default_factory=list)
    known_switches: List[str] = field(default_factory=list)
    known_value_params: List[str] = field(default_factory=list)
    leaf_only_path_params: Optional[List[str]] = None
    positional_skip: Optional[int] = None
    optional_write: bool = False
    tolerate_unknown_params: bool = False


CMDLET_PATH_CONFIG: Dict[str, CmdletPathConfig] = {
    'set-content': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-passthru', '-force', '-whatif', '-confirm', '-usetransaction', '-nonewline', '-asbytestream'], known_value_params=['-value', '-filter', '-include', '-exclude', '-credential', '-encoding', '-stream']),
    'add-content': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-passthru', '-force', '-whatif', '-confirm', '-usetransaction', '-nonewline', '-asbytestream'], known_value_params=['-value', '-filter', '-include', '-exclude', '-credential', '-encoding', '-stream']),
    'remove-item': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-recurse', '-force', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-filter', '-include', '-exclude', '-credential', '-stream']),
    'clear-content': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-filter', '-include', '-exclude', '-credential', '-stream']),
    'out-file': CmdletPathConfig(operation_type='write', path_params=['-filepath', '-path', '-literalpath', '-pspath', '-lp'], known_switches=['-append', '-force', '-noclobber', '-nonewline', '-whatif', '-confirm'], known_value_params=['-inputobject', '-encoding', '-width']),
    'tee-object': CmdletPathConfig(operation_type='write', path_params=['-filepath', '-path', '-literalpath', '-pspath', '-lp'], known_switches=['-append'], known_value_params=['-inputobject', '-variable', '-encoding']),
    'export-csv': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-append', '-force', '-noclobber', '-notypeinformation', '-includetypeinformation', '-useculture', '-noheader', '-whatif', '-confirm'], known_value_params=['-inputobject', '-delimiter', '-encoding', '-quotefields', '-usequotes']),
    'export-clixml': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-noclobber', '-whatif', '-confirm'], known_value_params=['-inputobject', '-depth', '-encoding']),
    'new-item': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-itemtype', '-value', '-credential', '-type'], leaf_only_path_params=['-name']),
    'copy-item': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp', '-destination'], known_switches=['-container', '-force', '-passthru', '-recurse', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-filter', '-include', '-exclude', '-credential', '-fromsession', '-tosession']),
    'move-item': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp', '-destination'], known_switches=['-force', '-passthru', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-filter', '-include', '-exclude', '-credential']),
    'rename-item': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-passthru', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-newname', '-credential', '-filter', '-include', '-exclude']),
    'set-item': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-passthru', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-value', '-credential', '-filter', '-include', '-exclude']),
    'get-content': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-usetransaction', '-wait', '-raw', '-asbytestream'], known_value_params=['-readcount', '-totalcount', '-tail', '-first', '-head', '-last', '-filter', '-include', '-exclude', '-credential', '-delimiter', '-encoding', '-stream']),
    'get-childitem': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-recurse', '-force', '-name', '-usetransaction', '-followsymlink', '-directory', '-file', '-hidden', '-readonly', '-system'], known_value_params=['-filter', '-include', '-exclude', '-depth', '-attributes', '-credential']),
    'get-item': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-usetransaction'], known_value_params=['-filter', '-include', '-exclude', '-credential', '-stream']),
    'get-itemproperty': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-usetransaction'], known_value_params=['-name', '-filter', '-include', '-exclude', '-credential']),
    'get-itempropertyvalue': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-usetransaction'], known_value_params=['-name', '-filter', '-include', '-exclude', '-credential']),
    'get-filehash': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=[], known_value_params=['-algorithm', '-inputstream']),
    'get-acl': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-audit', '-allcentralaccesspolicies', '-usetransaction'], known_value_params=['-inputobject', '-filter', '-include', '-exclude']),
    'format-hex': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-raw'], known_value_params=['-inputobject', '-encoding', '-count', '-offset']),
    'test-path': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-isvalid', '-usetransaction'], known_value_params=['-filter', '-include', '-exclude', '-pathtype', '-credential', '-olderthan', '-newerthan']),
    'resolve-path': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-relative', '-usetransaction', '-force'], known_value_params=['-credential', '-relativebasepath']),
    'convert-path': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-usetransaction'], known_value_params=[]),
    'select-string': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-simplematch', '-casesensitive', '-quiet', '-list', '-notmatch', '-allmatches', '-noemphasis', '-raw'], known_value_params=['-inputobject', '-pattern', '-include', '-exclude', '-encoding', '-context', '-culture']),
    'set-location': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-passthru', '-usetransaction'], known_value_params=['-stackname']),
    'push-location': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-passthru', '-usetransaction'], known_value_params=['-stackname']),
    'pop-location': CmdletPathConfig(operation_type='read', path_params=[], known_switches=['-passthru', '-usetransaction'], known_value_params=['-stackname']),
    'select-xml': CmdletPathConfig(operation_type='read', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=[], known_value_params=['-xml', '-content', '-xpath', '-namespace']),
    'get-winevent': CmdletPathConfig(operation_type='read', path_params=['-path'], known_switches=['-force', '-oldest'], known_value_params=['-listlog', '-logname', '-listprovider', '-providername', '-maxevents', '-computername', '-credential', '-filterxpath', '-filterxml', '-filterhashtable']),
    'invoke-webrequest': CmdletPathConfig(operation_type='write', path_params=['-outfile', '-infile'], known_switches=['-allowinsecureredirect', '-allowunencryptedauthentication', '-disablekeepalive', '-nobodyprogress', '-passthru', '-preservefileauthorizationmetadata', '-resume', '-skipcertificatecheck', '-skipheadervalidation', '-skiphttperrorcheck', '-usebasicparsing', '-usedefaultcredentials'], known_value_params=['-uri', '-method', '-body', '-contenttype', '-headers', '-maximumredirection', '-maximumretrycount', '-proxy', '-proxycredential', '-retryintervalsec', '-sessionvariable', '-timeoutsec', '-token', '-transferencoding', '-useragent', '-websession', '-credential', '-authentication', '-certificate', '-certificatethumbprint', '-form', '-httpversion'], positional_skip=1, optional_write=True),
    'invoke-restmethod': CmdletPathConfig(operation_type='write', path_params=['-outfile', '-infile'], known_switches=['-allowinsecureredirect', '-allowunencryptedauthentication', '-disablekeepalive', '-followrellink', '-nobodyprogress', '-passthru', '-preservefileauthorizationmetadata', '-resume', '-skipcertificatecheck', '-skipheadervalidation', '-skiphttperrorcheck', '-usebasicparsing', '-usedefaultcredentials'], known_value_params=['-uri', '-method', '-body', '-contenttype', '-headers', '-maximumfollowrellink', '-maximumredirection', '-maximumretrycount', '-proxy', '-proxycredential', '-responseheaderstvariable', '-retryintervalsec', '-sessionvariable', '-statuscodevariable', '-timeoutsec', '-token', '-transferencoding', '-useragent', '-websession', '-credential', '-authentication', '-certificate', '-certificatethumbprint', '-form', '-httpversion'], positional_skip=1, optional_write=True),
    'expand-archive': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp', '-destinationpath'], known_switches=['-force', '-passthru', '-whatif', '-confirm'], known_value_params=[]),
    'compress-archive': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp', '-destinationpath'], known_switches=['-force', '-update', '-passthru', '-whatif', '-confirm'], known_value_params=['-compressionlevel']),
    'set-itemproperty': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-passthru', '-force', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-name', '-value', '-type', '-filter', '-include', '-exclude', '-credential', '-inputobject']),
    'new-itemproperty': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-name', '-value', '-propertytype', '-type', '-filter', '-include', '-exclude', '-credential']),
    'remove-itemproperty': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-name', '-filter', '-include', '-exclude', '-credential']),
    'clear-item': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-force', '-whatif', '-confirm', '-usetransaction'], known_value_params=['-filter', '-include', '-exclude', '-credential']),
    'export-alias': CmdletPathConfig(operation_type='write', path_params=['-path', '-literalpath', '-pspath', '-lp'], known_switches=['-append', '-force', '-noclobber', '-passthru', '-whatif', '-confirm'], known_value_params=['-name', '-description', '-scope', '-as']),
    # Native read exes (findstr/tree/file) that take POSITIONAL path/pattern args.
    # They are in the read-only allowlist (so is_read_only_command -> True), but
    # without a path config NO containment check ran — so e.g.
    # `findstr /S password C:\Users\*` or `tree /F C:\Users` silently READ/enumerated
    # files OUTSIDE the session directory with no prompt, while the cmdlet
    # equivalents (Select-String / Get-ChildItem / Get-Content) were correctly
    # contained. Register them as 'read' so their positionals get extracted: a
    # flag ('/S', '-b') or pattern resolves relative to cwd (inside -> allowed),
    # while an absolute/outside path forces the same ask the cmdlets get.
    # git's path operands, under session containment. The flag vocabulary is
    # vetted before this runs: is_read_only_command puts every flag through
    # validate_flags against GIT_READ_ONLY_COMMANDS, and a flag outside that
    # list yields no allow at all. Without an entry here,
    # extract_paths_from_command returned early with no paths, so
    # check_path_constraints answered allow and the read-only allow stood —
    # `git diff --no-index <absolute path outside the session> NUL` printed a
    # file from anywhere on disk with no prompt (measured on Windows).
    #
    # positional_skip drops the subcommand (grep/diff/log), which is not a
    # path. tolerate_unknown_params keeps ordinary git quiet: without it every
    # flag lands in the unknown branch and `git status --short` becomes a
    # prompt. Flag VALUES are still checked — an unconsumed value falls
    # through to the positional branch and is containment-checked there.
    'git': CmdletPathConfig(operation_type='read', positional_skip=1, tolerate_unknown_params=True),
    'findstr': CmdletPathConfig(operation_type='read'),
    'tree': CmdletPathConfig(operation_type='read'),
    'file': CmdletPathConfig(operation_type='read'),
}


# =============================================================================
# Helpers.
# =============================================================================
def _expand_tilde(file_path: str) -> str:
    """~, ~/, ~\\ → homedir + rest (expandTilde; PS form includes ~\\)."""
    if file_path == "~" or file_path.startswith("~/") or file_path.startswith("~\\"):
        return os.path.expanduser("~") + file_path[1:]
    return file_path


def _matches_param(param_lower: str, param_list: List[str]) -> bool:
    """Abbreviation-aware: exact, or a list entry starts with the input
    (`-pa` → `-path`); input must be > 1 char (matchesParam)."""
    for p in param_list:
        if p == param_lower or (len(param_lower) > 1 and p.startswith(param_lower)):
            return True
    return False


def _has_complex_colon_value(raw_value: str) -> bool:
    """A colon-bound value that isn't a plain literal (hasComplexColonValue)."""
    return (
        "," in raw_value
        or raw_value.startswith("(")
        or raw_value.startswith("[")
        or "`" in raw_value
        or "@(" in raw_value
        or raw_value.startswith("@{")
        or "$" in raw_value
    )


def is_dangerous_removal_raw_path(file_path: str) -> bool:
    """Dangerous-removal check on the RAW path (pre-realpath), backslash-normalised
    (isDangerousRemovalRawPath). Reuses CodeMason's is_dangerous_removal_path."""
    # \Z (absolute end), not $ — Python '$' also matches before a trailing '\n',
    # so `foo'\n` would lose the quote; JS `.replace(/^['"]|['"]$/g,...)` doesn't.
    expanded = _expand_tilde(re.sub(r"^['\"]|['\"]\Z", "", file_path)).replace("\\", "/")
    return is_dangerous_removal_path(expanded)


def dangerous_removal_deny(path: str) -> PermissionResult:
    """Hard-deny a removal targeting a protected system path (dangerousRemovalDeny)."""
    return PermissionResult(
        behavior="deny",
        message=f"Remove-Item on system path '{path}' is blocked. This path is protected from removal.",
    )


# =============================================================================
# Path extraction from a command element.
# =============================================================================
@dataclass
class _ExtractResult:
    paths: List[str]
    operation_type: str
    has_unvalidatable_path_arg: bool
    optional_write: bool


def extract_paths_from_command(cmd: ParsedCommandElement) -> _ExtractResult:
    """Extract filesystem paths from a command element via CMDLET_PATH_CONFIG +
    the AST (extractPathsFromCommand). Sets has_unvalidatable_path_arg when a path
    arg has a non-static element type so the caller forces an ask."""
    canonical = resolve_to_canonical(cmd.name)
    config = CMDLET_PATH_CONFIG.get(canonical)
    if not config:
        return _ExtractResult([], "read", False, False)

    switch_params = list(config.known_switches) + list(COMMON_SWITCHES)
    value_params = list(config.known_value_params) + list(COMMON_VALUE_PARAMS)

    paths: List[str] = []
    args = cmd.args
    element_types = cmd.element_types
    has_unvalidatable = False
    positionals_seen = 0
    positional_skip = config.positional_skip or 0

    def check_arg_element_type(arg_idx: int) -> bool:
        if not element_types:
            return False
        et = element_types[arg_idx + 1] if arg_idx + 1 < len(element_types) else None
        return bool(et and et not in _SAFE_PATH_ELEMENT_TYPES)

    i = 0
    while i < len(args):
        arg = args[i]
        if not arg:
            i += 1
            continue

        arg_element_type = element_types[i + 1] if element_types and i + 1 < len(element_types) else None
        if is_powershell_parameter(arg, arg_element_type):
            normalized = "-" + arg[1:]
            colon_idx = normalized.find(":", 1)
            param_name = normalized[:colon_idx] if colon_idx > 0 else normalized
            param_lower = param_name.lower()

            if _matches_param(param_lower, config.path_params):
                value = None
                if colon_idx > 0:
                    raw_value = arg[colon_idx + 1:]
                    if _has_complex_colon_value(raw_value):
                        has_unvalidatable = True
                    else:
                        value = raw_value
                else:
                    next_val = args[i + 1] if i + 1 < len(args) else None
                    next_type = element_types[i + 2] if element_types and i + 2 < len(element_types) else None
                    if next_val and not is_powershell_parameter(next_val, next_type):
                        value = next_val
                        if check_arg_element_type(i + 1):
                            has_unvalidatable = True
                        i += 1
                if value:
                    paths.append(value)

            elif config.leaf_only_path_params and _matches_param(param_lower, config.leaf_only_path_params):
                value = None
                if colon_idx > 0:
                    raw_value = arg[colon_idx + 1:]
                    if _has_complex_colon_value(raw_value):
                        has_unvalidatable = True
                    else:
                        value = raw_value
                else:
                    next_val = args[i + 1] if i + 1 < len(args) else None
                    next_type = element_types[i + 2] if element_types and i + 2 < len(element_types) else None
                    if next_val and not is_powershell_parameter(next_val, next_type):
                        value = next_val
                        if check_arg_element_type(i + 1):
                            has_unvalidatable = True
                        i += 1
                if value is not None:
                    if "/" in value or "\\" in value or value == "." or value == "..":
                        has_unvalidatable = True  # non-leaf can't resolve correctly
                    else:
                        paths.append(value)

            elif _matches_param(param_lower, switch_params):
                pass  # switch — takes no value, don't consume next arg

            elif _matches_param(param_lower, value_params):
                if colon_idx > 0:
                    raw_value = arg[colon_idx + 1:]
                    if _has_complex_colon_value(raw_value):
                        has_unvalidatable = True
                else:
                    next_arg = args[i + 1] if i + 1 < len(args) else None
                    next_arg_type = element_types[i + 2] if element_types and i + 2 < len(element_types) else None
                    if next_arg and not is_powershell_parameter(next_arg, next_arg_type):
                        if check_arg_element_type(i + 1):
                            has_unvalidatable = True
                        i += 1

            else:
                # Unknown parameter — flag the whole command unvalidatable,
                # unless this command's flags are vetted elsewhere.
                if not config.tolerate_unknown_params:
                    has_unvalidatable = True
                if colon_idx > 0:
                    raw_value = arg[colon_idx + 1:]
                    if not _has_complex_colon_value(raw_value):
                        paths.append(raw_value)
            i += 1
            continue

        # Positional argument.
        if positionals_seen < positional_skip:
            positionals_seen += 1
            i += 1
            continue
        positionals_seen += 1
        if check_arg_element_type(i):
            has_unvalidatable = True
        paths.append(arg)
        i += 1

    return _ExtractResult(paths, config.operation_type, has_unvalidatable, config.optional_write)


# =============================================================================
# Main check (adapted to CodeMason's simplified permission model).
# =============================================================================
def _check_statement(statement: ParsedStatement, compound_command_has_cd: bool) -> PermissionResult:
    first_ask: Optional[PermissionResult] = None

    if compound_command_has_cd:
        first_ask = PermissionResult(
            behavior="ask",
            message=(
                "Compound command changes working directory (Set-Location/Push-Location/"
                "Pop-Location/New-PSDrive) — relative paths cannot be validated against "
                "the original cwd and require manual approval"
            ),
        )

    has_expression_pipeline_source = False

    for cmd in statement.commands:
        if cmd.element_type != "CommandAst":
            has_expression_pipeline_source = True
            continue

        res = extract_paths_from_command(cmd)
        canonical = resolve_to_canonical(cmd.name)

        if has_expression_pipeline_source and first_ask is None:
            first_ask = PermissionResult(
                behavior="ask",
                message=f"{canonical} receives its path from a pipeline expression source "
                        "that cannot be statically validated and requires manual approval",
            )

        if res.has_unvalidatable_path_arg and first_ask is None:
            first_ask = PermissionResult(
                behavior="ask",
                message=f"{canonical} uses a parameter or complex path expression "
                        "(array literal, subexpression, unknown parameter, etc.) that "
                        "cannot be statically validated and requires manual approval",
            )

        if (
            res.operation_type != "read"
            and not res.optional_write
            and len(res.paths) == 0
            and CMDLET_PATH_CONFIG.get(canonical)
        ):
            if first_ask is None:
                first_ask = PermissionResult(
                    behavior="ask",
                    message=f"{canonical} is a write operation but no target path could "
                            "be determined; requires manual approval",
                )
            continue

        is_removal = canonical == "remove-item"

        for file_path in res.paths:
            # Hard-deny removal of dangerous system paths (raw, then resolved).
            if is_removal and is_dangerous_removal_raw_path(file_path):
                return dangerous_removal_deny(file_path)
            if is_removal:
                clean = _expand_tilde(file_path.strip().strip("'\""))
                full = os.path.normpath(clean if os.path.isabs(clean) else os.path.join(get_cwd(), clean))
                if is_dangerous_removal_path(full):
                    return dangerous_removal_deny(full)

            # Containment check via CodeMason's simplified validator.
            reason = _validate_path(file_path, res.operation_type)
            if reason and first_ask is None:
                first_ask = PermissionResult(behavior="ask", message=f"{canonical}: {reason}")

    return first_ask or PermissionResult(behavior="allow", message="All path constraints validated successfully")


def check_path_constraints(
    parsed: ParsedPowerShellCommand,
    compound_command_has_cd: bool = False,
) -> PermissionResult:
    """Validate paths across all statements (checkPathConstraints).

    Two-pass: a deny (dangerous removal) wins immediately; otherwise the first
    ask is returned; else 'allow'. On a failed parse, 'allow' (passthrough) —
    the caller's default prompt still applies.
    """
    if not parsed.valid:
        return PermissionResult(behavior="allow", message="Cannot validate paths for unparsed command")

    first_ask: Optional[PermissionResult] = None
    for statement in parsed.statements:
        result = _check_statement(statement, compound_command_has_cd)
        if result.behavior == "deny":
            return result
        if result.behavior == "ask" and first_ask is None:
            first_ask = result

    return first_ask or PermissionResult(behavior="allow", message="All path constraints validated successfully")
