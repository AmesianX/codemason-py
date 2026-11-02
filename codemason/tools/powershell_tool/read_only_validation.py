"""PowerShell read-only command validation (adapted from readOnlyValidation.ts).

Decides whether a parsed PowerShell command is read-only (safe to auto-allow),
using a cmdlet allowlist plus the shared external-command validation for
git/gh/docker/dotnet. Cmdlets are matched case-insensitively (lowercase keys).
"""

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ...utils.platform import is_windows
from ...utils.powershell.parser import (
    COMMON_ALIASES,
    ParsedCommandElement,
    ParsedPowerShellCommand,
    ParsedStatement,
    derive_security_flags,
    get_pipeline_segments,
    is_null_redirection_target,
    is_powershell_parameter,
)
from ...utils.shell.read_only_command_validation import (
    DOCKER_READ_ONLY_COMMANDS,
    EXTERNAL_READONLY_COMMANDS,
    GH_READ_ONLY_COMMANDS,
    GIT_READ_ONLY_COMMANDS,
    ExternalCommandConfig,
    validate_flags,
)
from .common_parameters import COMMON_PARAMETERS

# dotnet read-only top-level flags (DOTNET_READ_ONLY_FLAGS).
_DOTNET_READ_ONLY_FLAGS = frozenset({"--version", "--info", "--list-runtimes", "--list-sdks"})

# Metachar class flagging a runtime-evaluated arg: $ ( @ { [ (matches source /[$(@{[]/).
_METACHAR_RE = re.compile(r"[$(@{\[]")


@dataclass
class CommandConfig:
    """Allowlist config for a read-only PowerShell cmdlet / native exe."""

    safe_flags: List[str] = field(default_factory=list)
    # When True, all flags are allowed (entire flag surface is read-only).
    allow_all_flags: bool = False
    # Regex constraint on the original command string.
    regex: Optional["re.Pattern"] = None
    # Returns True if the command is dangerous (command, element).
    additional_command_is_dangerous_callback: Optional[
        Callable[[str, Optional[ParsedCommandElement]], bool]
    ] = None


def _arg_leaks_value(_cmd: str, element: Optional[ParsedCommandElement] = None) -> bool:
    """True if a cmdlet arg leaks a value (prints/coerces a non-literal). Port of
    argLeaksValue — the callback shared by Write-Output/Host, Start-Sleep, and the
    Format-*/Select-Object/etc. pipeline-tail cmdlets."""
    arg_types = (element.element_types or [])[1:] if element else []
    args = element.args if element else []
    children = element.children if element else None
    for i in range(len(arg_types)):
        if arg_types[i] != "StringConstant" and arg_types[i] != "Parameter":
            if not _METACHAR_RE.search(args[i] if i < len(args) else ""):
                continue
            return True
        if arg_types[i] == "Parameter":
            param_children = children[i] if children and i < len(children) else None
            if param_children:
                if any(c.type != "StringConstant" for c in param_children):
                    return True
            else:
                arg = args[i] if i < len(args) else ""
                colon_idx = arg.find(":")
                if colon_idx > 0 and _METACHAR_RE.search(arg[colon_idx + 1:]):
                    return True
    return False


def _ipconfig_cb(_cmd: str, element=None) -> bool:
    """ipconfig: reject any positional (non-/ non-- arg) — `ipconfig set ...`
    writes network config on macOS."""
    return any(not a.startswith("/") and not a.startswith("-") for a in (element.args if element else []))


def _hostname_cb(_cmd: str, element=None) -> bool:
    """hostname: reject any positional — `hostname NAME` sets the hostname."""
    return any(not a.startswith("-") for a in (element.args if element else []))


def _route_cb(_cmd: str, element=None) -> bool:
    """route: the first non-flag positional is the verb; only `print` is read-only."""
    if not element:
        return True
    verb = next((a for a in element.args if not a.startswith("-")), None)
    return (verb.lower() if verb else None) != "print"


CMDLET_ALLOWLIST: Dict[str, CommandConfig] = {
    'get-childitem': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Filter', '-Include', '-Exclude', '-Recurse', '-Depth', '-Name', '-Force', '-Attributes', '-Directory', '-File', '-Hidden', '-ReadOnly', '-System']),
    'get-content': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-TotalCount', '-Head', '-Tail', '-Raw', '-Encoding', '-Delimiter', '-ReadCount']),
    'get-item': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Force', '-Stream']),
    'get-itemproperty': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Name']),
    'test-path': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-PathType', '-Filter', '-Include', '-Exclude', '-IsValid', '-NewerThan', '-OlderThan']),
    'resolve-path': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Relative']),
    'get-filehash': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Algorithm', '-InputStream']),
    'get-acl': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Audit', '-Filter', '-Include', '-Exclude']),
    'set-location': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-PassThru', '-StackName']),
    'push-location': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-PassThru', '-StackName']),
    'pop-location': CommandConfig(safe_flags=['-PassThru', '-StackName']),
    'select-string': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Pattern', '-InputObject', '-SimpleMatch', '-CaseSensitive', '-Quiet', '-List', '-NotMatch', '-AllMatches', '-Encoding', '-Context', '-Raw', '-NoEmphasis']),
    'convertto-json': CommandConfig(safe_flags=['-InputObject', '-Depth', '-Compress', '-EnumsAsStrings', '-AsArray']),
    'convertfrom-json': CommandConfig(safe_flags=['-InputObject', '-Depth', '-AsHashtable', '-NoEnumerate']),
    'convertto-csv': CommandConfig(safe_flags=['-InputObject', '-Delimiter', '-NoTypeInformation', '-NoHeader', '-UseQuotes']),
    'convertfrom-csv': CommandConfig(safe_flags=['-InputObject', '-Delimiter', '-Header', '-UseCulture']),
    'convertto-xml': CommandConfig(safe_flags=['-InputObject', '-Depth', '-As', '-NoTypeInformation']),
    'convertto-html': CommandConfig(safe_flags=['-InputObject', '-Property', '-Head', '-Title', '-Body', '-Pre', '-Post', '-As', '-Fragment']),
    'format-hex': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-InputObject', '-Encoding', '-Count', '-Offset']),
    'get-member': CommandConfig(safe_flags=['-InputObject', '-MemberType', '-Name', '-Static', '-View', '-Force']),
    'get-unique': CommandConfig(safe_flags=['-InputObject', '-AsString', '-CaseInsensitive', '-OnType']),
    'compare-object': CommandConfig(safe_flags=['-ReferenceObject', '-DifferenceObject', '-Property', '-SyncWindow', '-CaseSensitive', '-Culture', '-ExcludeDifferent', '-IncludeEqual', '-PassThru']),
    'join-string': CommandConfig(safe_flags=['-InputObject', '-Property', '-Separator', '-OutputPrefix', '-OutputSuffix', '-SingleQuote', '-DoubleQuote', '-FormatString']),
    'get-random': CommandConfig(safe_flags=['-InputObject', '-Minimum', '-Maximum', '-Count', '-SetSeed', '-Shuffle']),
    'convert-path': CommandConfig(safe_flags=['-Path', '-LiteralPath']),
    'join-path': CommandConfig(safe_flags=['-Path', '-ChildPath', '-AdditionalChildPath']),
    'split-path': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Qualifier', '-NoQualifier', '-Parent', '-Leaf', '-LeafBase', '-Extension', '-IsAbsolute']),
    'get-hotfix': CommandConfig(safe_flags=['-Id', '-Description']),
    'get-itempropertyvalue': CommandConfig(safe_flags=['-Path', '-LiteralPath', '-Name']),
    'get-psprovider': CommandConfig(safe_flags=['-PSProvider']),
    'get-process': CommandConfig(safe_flags=['-Name', '-Id', '-Module', '-FileVersionInfo', '-IncludeUserName']),
    'get-service': CommandConfig(safe_flags=['-Name', '-DisplayName', '-DependentServices', '-RequiredServices', '-Include', '-Exclude']),
    'get-computerinfo': CommandConfig(allow_all_flags=True),
    'get-host': CommandConfig(allow_all_flags=True),
    'get-date': CommandConfig(safe_flags=['-Date', '-Format', '-UFormat', '-DisplayHint', '-AsUTC']),
    'get-location': CommandConfig(safe_flags=['-PSProvider', '-PSDrive', '-Stack', '-StackName']),
    'get-psdrive': CommandConfig(safe_flags=['-Name', '-PSProvider', '-Scope']),
    'get-module': CommandConfig(safe_flags=['-Name', '-ListAvailable', '-All', '-FullyQualifiedName', '-PSEdition']),
    'get-alias': CommandConfig(safe_flags=['-Name', '-Definition', '-Scope', '-Exclude']),
    'get-history': CommandConfig(safe_flags=['-Id', '-Count']),
    'get-culture': CommandConfig(allow_all_flags=True),
    'get-uiculture': CommandConfig(allow_all_flags=True),
    'get-timezone': CommandConfig(safe_flags=['-Name', '-Id', '-ListAvailable']),
    'get-uptime': CommandConfig(allow_all_flags=True),
    'write-output': CommandConfig(safe_flags=['-InputObject', '-NoEnumerate'], additional_command_is_dangerous_callback=_arg_leaks_value),
    'write-host': CommandConfig(safe_flags=['-Object', '-NoNewline', '-Separator', '-ForegroundColor', '-BackgroundColor'], additional_command_is_dangerous_callback=_arg_leaks_value),
    'start-sleep': CommandConfig(safe_flags=['-Seconds', '-Milliseconds', '-Duration'], additional_command_is_dangerous_callback=_arg_leaks_value),
    'format-table': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'format-list': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'format-wide': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'format-custom': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'measure-object': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'select-object': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'sort-object': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'group-object': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'where-object': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'out-string': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'out-host': CommandConfig(allow_all_flags=True, additional_command_is_dangerous_callback=_arg_leaks_value),
    'get-netadapter': CommandConfig(safe_flags=['-Name', '-InterfaceDescription', '-InterfaceIndex', '-Physical']),
    'get-netipaddress': CommandConfig(safe_flags=['-InterfaceIndex', '-InterfaceAlias', '-AddressFamily', '-Type']),
    'get-netipconfiguration': CommandConfig(safe_flags=['-InterfaceIndex', '-InterfaceAlias', '-Detailed', '-All']),
    'get-netroute': CommandConfig(safe_flags=['-InterfaceIndex', '-InterfaceAlias', '-AddressFamily', '-DestinationPrefix']),
    'get-dnsclientcache': CommandConfig(safe_flags=['-Entry', '-Name', '-Type', '-Status', '-Section', '-Data']),
    'get-dnsclient': CommandConfig(safe_flags=['-InterfaceIndex', '-InterfaceAlias']),
    'get-eventlog': CommandConfig(safe_flags=['-LogName', '-Newest', '-After', '-Before', '-EntryType', '-Index', '-InstanceId', '-Message', '-Source', '-UserName', '-AsBaseObject', '-List']),
    'get-winevent': CommandConfig(safe_flags=['-LogName', '-ListLog', '-ListProvider', '-ProviderName', '-Path', '-MaxEvents', '-FilterXPath', '-Force', '-Oldest']),
    'get-cimclass': CommandConfig(safe_flags=['-ClassName', '-Namespace', '-MethodName', '-PropertyName', '-QualifierName']),
    'git': CommandConfig(safe_flags=[]),
    'gh': CommandConfig(safe_flags=[]),
    'docker': CommandConfig(safe_flags=[]),
    'ipconfig': CommandConfig(safe_flags=['/all', '/displaydns', '/allcompartments'], additional_command_is_dangerous_callback=_ipconfig_cb),
    'netstat': CommandConfig(safe_flags=['-a', '-b', '-e', '-f', '-n', '-o', '-p', '-q', '-r', '-s', '-t', '-x', '-y']),
    'systeminfo': CommandConfig(safe_flags=['/FO', '/NH']),
    'tasklist': CommandConfig(safe_flags=['/M', '/SVC', '/V', '/FI', '/FO', '/NH']),
    'where.exe': CommandConfig(allow_all_flags=True),
    'hostname': CommandConfig(safe_flags=['-a', '-d', '-f', '-i', '-I', '-s', '-y', '-A'], additional_command_is_dangerous_callback=_hostname_cb),
    'whoami': CommandConfig(safe_flags=['/user', '/groups', '/claims', '/priv', '/logonid', '/all', '/fo', '/nh']),
    'ver': CommandConfig(allow_all_flags=True),
    'arp': CommandConfig(safe_flags=['-a', '-g', '-v', '-N']),
    'route': CommandConfig(safe_flags=['print', 'PRINT', '-4', '-6'], additional_command_is_dangerous_callback=_route_cb),
    'getmac': CommandConfig(safe_flags=['/FO', '/NH', '/V']),
    'file': CommandConfig(safe_flags=['-b', '--brief', '-i', '--mime', '-L', '--dereference', '--mime-type', '--mime-encoding', '-z', '--uncompress', '-p', '--preserve-date', '-k', '--keep-going', '-r', '--raw', '-v', '--version', '-0', '--print0', '-s', '--special-files', '-l', '-F', '--separator', '-e', '-P', '-N', '--no-pad', '-E', '--extension']),
    'tree': CommandConfig(safe_flags=['/F', '/A', '/Q', '/L']),
    'findstr': CommandConfig(safe_flags=['/B', '/E', '/L', '/R', '/S', '/I', '/X', '/V', '/N', '/M', '/O', '/P', '/C', '/G', '/D', '/A']),
    'dotnet': CommandConfig(safe_flags=[]),
}


# =============================================================================
# Cmdlet-name classification / lookup.
# =============================================================================
SAFE_OUTPUT_CMDLETS = frozenset({"out-null"})

_PIPELINE_TAIL_CMDLETS = frozenset(
    {"format-table", "format-list", "format-wide", "format-custom", "measure-object",
     "select-object", "sort-object", "group-object", "where-object", "out-string", "out-host"}
)

# External .exe names allowed past the nameType='application' gate. Each MUST have
# a matching CMDLET_ALLOWLIST entry for flag validation.
_SAFE_EXTERNAL_EXES = frozenset({"where.exe"})

# Windows PATHEXT extensions PowerShell resolves via PATH (.ps1 excluded).
_WINDOWS_PATHEXT = re.compile(r"\.(exe|cmd|bat|com)$")


def resolve_to_canonical(name: str) -> str:
    """Canonical cmdlet name via COMMON_ALIASES; strips .exe/.cmd/.bat/.com on
    bare (path-free) names so git.exe → git. (resolveToCanonical)"""
    lower = name.lower()
    if "\\" not in lower and "/" not in lower:
        lower = _WINDOWS_PATHEXT.sub("", lower)
    alias = COMMON_ALIASES.get(lower)
    if alias:
        return alias.lower()
    return lower


def is_cwd_changing_cmdlet(name: str) -> bool:
    """True if the cmdlet alters the path-resolution namespace for later
    statements (Set/Push/Pop-Location, New-PSDrive; ndr/mount Windows-only)."""
    canonical = resolve_to_canonical(name)
    return (
        canonical == "set-location"
        or canonical == "push-location"
        or canonical == "pop-location"
        or canonical == "new-psdrive"
        or (is_windows() and (canonical == "ndr" or canonical == "mount"))
    )


def is_safe_output_command(name: str) -> bool:
    return resolve_to_canonical(name) in SAFE_OUTPUT_CMDLETS


def _lookup_allowlist(name: str) -> Optional[CommandConfig]:
    lower = name.lower()
    direct = CMDLET_ALLOWLIST.get(lower)
    if direct:
        return direct
    canonical = resolve_to_canonical(lower)
    if canonical != lower:
        return CMDLET_ALLOWLIST.get(canonical)
    return None


# =============================================================================
# Main read-only check.
# =============================================================================
def is_read_only_command(command: str, parsed: Optional[ParsedPowerShellCommand] = None) -> bool:
    """True if the parsed PowerShell command is read-only (isReadOnlyCommand)."""
    trimmed_command = command.strip()
    if not trimmed_command:
        return False
    if not parsed:
        return False
    if not parsed.valid:
        return False

    security = derive_security_flags(parsed)
    if (
        security.has_script_blocks
        or security.has_sub_expressions
        or security.has_expandable_strings
        or security.has_splatting
        or security.has_member_invocations
        or security.has_assignments
        or security.has_stop_parsing
    ):
        return False

    segments = get_pipeline_segments(parsed)
    if len(segments) == 0:
        return False

    # Block compound commands containing a cwd-changing cmdlet (stale-cwd path
    # resolution — finding #27).
    total_commands = sum(len(seg.commands) for seg in segments)
    if total_commands > 1:
        has_cd = any(is_cwd_changing_cmdlet(cmd.name) for seg in segments for cmd in seg.commands)
        if has_cd:
            return False

    for pipeline in segments:
        if not pipeline or len(pipeline.commands) == 0:
            return False
        # Reject file redirections (not `> $null`).
        if pipeline.redirections:
            if any(not r.is_merging and not is_null_redirection_target(r.target) for r in pipeline.redirections):
                return False
        first_cmd = pipeline.commands[0]
        if not first_cmd:
            return False
        if not is_allowlisted_command(first_cmd, command):
            return False
        for i in range(1, len(pipeline.commands)):
            cmd = pipeline.commands[i]
            if not cmd or cmd.name_type == "application":
                return False
            if is_safe_output_command(cmd.name) and len(cmd.args) == 0:
                continue
            if not is_allowlisted_command(cmd, command):
                return False
        if pipeline.nested_commands and len(pipeline.nested_commands) > 0:
            return False

    return True


def is_allowlisted_command(cmd: ParsedCommandElement, original_command: str) -> bool:
    """True if a single command element is allowlisted and passes flag/arg
    validation (isAllowlistedCommand)."""
    # nameType='application' → raw name has path chars; never auto-allow except
    # explicit safe .exe matched on the raw first token of cmd.text.
    if cmd.name_type == "application":
        raw_first_token = (cmd.text.split(None, 1)[0].lower() if cmd.text.split(None, 1) else "")
        if raw_first_token not in _SAFE_EXTERNAL_EXES:
            return False

    config = _lookup_allowlist(cmd.name)
    if not config:
        return False

    if config.regex and not config.regex.search(original_command):
        return False

    if config.additional_command_is_dangerous_callback and config.additional_command_is_dangerous_callback(original_command, cmd):
        return False

    # SECURITY: whitelist arg elementTypes — only StringConstant/Parameter are
    # statically verifiable; everything else evaluates at runtime (leak vectors).
    if not cmd.element_types:
        return False
    for i in range(1, len(cmd.element_types)):
        t = cmd.element_types[i]
        if t != "StringConstant" and t != "Parameter":
            if not _METACHAR_RE.search(cmd.args[i - 1] if i - 1 < len(cmd.args) else ""):
                continue
            return False
        if t == "Parameter":
            param_children = cmd.children[i - 1] if cmd.children and i - 1 < len(cmd.children) else None
            if param_children:
                if any(c.type != "StringConstant" for c in param_children):
                    return False
            else:
                arg = cmd.args[i - 1] if i - 1 < len(cmd.args) else ""
                colon_idx = arg.find(":")
                if colon_idx > 0 and _METACHAR_RE.search(arg[colon_idx + 1:]):
                    return False

    canonical = resolve_to_canonical(cmd.name)

    if canonical in ("git", "gh", "docker", "dotnet"):
        return _is_external_command_safe(canonical, cmd.args)

    is_cmdlet = "-" in canonical

    if config.allow_all_flags:
        return True
    if not config.safe_flags:
        # No safeFlags + no allowAllFlags → reject any flags; positional args OK.
        def _is_flag(pair):
            i, arg = pair
            if is_cmdlet:
                et = cmd.element_types[i + 1] if cmd.element_types and i + 1 < len(cmd.element_types) else None
                return is_powershell_parameter(arg, et)
            return arg.startswith("-") or (is_windows() and arg.startswith("/"))
        has_flags = any(_is_flag((i, a)) for i, a in enumerate(cmd.args))
        return not has_flags

    for i in range(len(cmd.args)):
        arg = cmd.args[i]
        if is_cmdlet:
            et = cmd.element_types[i + 1] if cmd.element_types and i + 1 < len(cmd.element_types) else None
            is_flag = is_powershell_parameter(arg, et)
        else:
            is_flag = arg.startswith("-") or (is_windows() and arg.startswith("/"))
        if is_flag:
            param_name = ("-" + arg[1:]) if is_cmdlet else arg
            colon_index = param_name.find(":")
            if colon_index > 0:
                param_name = param_name[:colon_index]
            param_lower = param_name.lower()
            if is_cmdlet and param_lower in COMMON_PARAMETERS:
                continue
            if not any(flag.lower() == param_lower for flag in config.safe_flags):
                return False

    return True


# =============================================================================
# External command validation (git / gh / docker / dotnet).
# =============================================================================
def _is_external_command_safe(command: str, args: List[str]) -> bool:
    if command == "git":
        return _is_git_safe(args)
    if command == "gh":
        return _is_gh_safe(args)
    if command == "docker":
        return _is_docker_safe(args)
    if command == "dotnet":
        return _is_dotnet_safe(args)
    return False


_DANGEROUS_GIT_GLOBAL_FLAGS = frozenset(
    {"-c", "-C", "--exec-path", "--config-env", "--git-dir", "--work-tree", "--attr-source"}
)
_GIT_GLOBAL_FLAGS_WITH_VALUES = frozenset(
    {"-c", "-C", "--exec-path", "--config-env", "--git-dir", "--work-tree",
     "--namespace", "--super-prefix", "--shallow-file"}
)
_DANGEROUS_GIT_SHORT_FLAGS_ATTACHED = ["-c", "-C"]


def _is_git_safe(args: List[str]) -> bool:
    if len(args) == 0:
        return True
    # Blanket $ rejection (variable-expansion parser differential).
    for arg in args:
        if "$" in arg:
            return False
    # Skip/reject global flags before the subcommand.
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if not arg or not arg.startswith("-"):
            break
        for short_flag in _DANGEROUS_GIT_SHORT_FLAGS_ATTACHED:
            if (
                len(arg) > len(short_flag)
                and arg.startswith(short_flag)
                and (short_flag == "-C" or arg[len(short_flag)] != "-")
            ):
                return False
        has_inline_value = "=" in arg
        flag_name = arg.split("=")[0] if has_inline_value else arg
        if flag_name in _DANGEROUS_GIT_GLOBAL_FLAGS:
            return False
        if not has_inline_value and flag_name in _GIT_GLOBAL_FLAGS_WITH_VALUES:
            idx += 2
        else:
            idx += 1

    if idx >= len(args):
        return True

    first = (args[idx] or "").lower()
    second = (args[idx + 1] or "").lower() if idx + 1 < len(args) else ""
    two_word_key = f"git {first} {second}"
    one_word_key = f"git {first}"

    config: Optional[ExternalCommandConfig] = GIT_READ_ONLY_COMMANDS.get(two_word_key)
    subcommand_tokens = 2
    if not config:
        config = GIT_READ_ONLY_COMMANDS.get(one_word_key)
        subcommand_tokens = 1
    if not config:
        return False

    flag_args = args[idx + subcommand_tokens:]

    # git ls-remote URL/exfil rejection.
    if first == "ls-remote":
        for arg in flag_args:
            if not arg.startswith("-"):
                if "://" in arg or "@" in arg or ":" in arg or "$" in arg:
                    return False

    if config.additional_command_is_dangerous_callback and config.additional_command_is_dangerous_callback("", flag_args):
        return False
    return validate_flags(flag_args, 0, config, command_name="git")


def _is_gh_safe(args: List[str]) -> bool:
    # gh is network-dependent; ant-only.
    if os.environ.get("USER_TYPE") != "ant":
        return False
    if len(args) == 0:
        return True
    config: Optional[ExternalCommandConfig] = None
    subcommand_tokens = 0
    if len(args) >= 2:
        two_word_key = f"gh {(args[0] or '').lower()} {(args[1] or '').lower()}"
        config = GH_READ_ONLY_COMMANDS.get(two_word_key)
        subcommand_tokens = 2
    if not config and len(args) >= 1:
        one_word_key = f"gh {(args[0] or '').lower()}"
        config = GH_READ_ONLY_COMMANDS.get(one_word_key)
        subcommand_tokens = 1
    if not config:
        return False
    flag_args = args[subcommand_tokens:]
    for arg in flag_args:
        if "$" in arg:
            return False
    if config.additional_command_is_dangerous_callback and config.additional_command_is_dangerous_callback("", flag_args):
        return False
    return validate_flags(flag_args, 0, config)


def _is_docker_safe(args: List[str]) -> bool:
    if len(args) == 0:
        return True
    for arg in args:
        if "$" in arg:
            return False
    one_word_key = f"docker {(args[0] or '').lower()}"
    if one_word_key in EXTERNAL_READONLY_COMMANDS:
        return True
    config: Optional[ExternalCommandConfig] = DOCKER_READ_ONLY_COMMANDS.get(one_word_key)
    if not config:
        return False
    flag_args = args[1:]
    if config.additional_command_is_dangerous_callback and config.additional_command_is_dangerous_callback("", flag_args):
        return False
    return validate_flags(flag_args, 0, config)


def _is_dotnet_safe(args: List[str]) -> bool:
    if len(args) == 0:
        return False
    for arg in args:
        if arg.lower() not in _DOTNET_READ_ONLY_FLAGS:
            return False
    return True
