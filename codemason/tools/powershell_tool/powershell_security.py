"""PowerShell-specific security analysis (adapted from powershellSecurity.ts).

Detects dangerous patterns — code injection, download cradles, privilege
escalation, dynamic command names, COM objects, etc. All checks are AST-based:
if parsing failed (valid=False), no individual check matches and
powershell_command_is_safe returns 'ask'.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from ...utils.powershell.dangerous_cmdlets import (
    DANGEROUS_SCRIPT_BLOCK_CMDLETS,
    FILEPATH_EXECUTION_CMDLETS,
    MODULE_LOADING_CMDLETS,
    WMI_CIM_CMDLETS,
)
from ...utils.powershell.parser import (
    COMMON_ALIASES,
    ParsedCommandElement,
    ParsedPowerShellCommand,
    command_has_arg_abbreviation,
    derive_security_flags,
    get_all_commands,
    get_variables_by_scope,
    has_command_named,
)
from .clm_types import is_clm_allowed_type


@dataclass
class PowerShellSecurityResult:
    behavior: str  # 'passthrough' | 'ask' | 'allow'
    message: Optional[str] = None


_PASSTHROUGH = PowerShellSecurityResult(behavior="passthrough")


def _ask(message: str) -> PowerShellSecurityResult:
    return PowerShellSecurityResult(behavior="ask", message=message)


_POWERSHELL_EXECUTABLES = frozenset({"pwsh", "pwsh.exe", "powershell", "powershell.exe"})


def is_powershell_executable(name: str) -> bool:
    """True if `name` (bare or a full path) resolves to a PowerShell executable."""
    lower = name.lower()
    if lower in _POWERSHELL_EXECUTABLES:
        return True
    last_sep = max(lower.rfind("/"), lower.rfind("\\"))
    if last_sep >= 0:
        return lower[last_sep + 1:] in _POWERSHELL_EXECUTABLES
    return False


# Alt parameter-prefix chars PowerShell treats as `-` (plus 5.1's `/`).
_PS_ALT_PARAM_PREFIXES = frozenset({"/", "–", "—", "―"})


def _ps_exe_has_param_abbreviation(cmd: ParsedCommandElement, full_param: str, min_prefix: str) -> bool:
    """command_has_arg_abbreviation that also normalises `/`/en-dash/em-dash/bar → `-`."""
    if command_has_arg_abbreviation(cmd, full_param, min_prefix):
        return True
    normalized_args = [
        ("-" + a[1:]) if (len(a) > 0 and a[0] in _PS_ALT_PARAM_PREFIXES) else a
        for a in cmd.args
    ]
    synthetic = ParsedCommandElement(
        name=cmd.name, name_type=cmd.name_type, element_type=cmd.element_type,
        args=normalized_args, text=cmd.text, element_types=cmd.element_types,
        children=cmd.children, redirections=cmd.redirections,
    )
    return command_has_arg_abbreviation(synthetic, full_param, min_prefix)


def check_invoke_expression(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    if has_command_named(parsed, "Invoke-Expression"):
        return _ask("Command uses Invoke-Expression which can execute arbitrary code")
    return _PASSTHROUGH


def check_dynamic_command_name(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    """Command name in a non-StringConstant element position is dynamic → ask."""
    for cmd in get_all_commands(parsed):
        if cmd.element_type != "CommandAst":
            continue
        name_element_type = cmd.element_types[0] if cmd.element_types else None
        if name_element_type is not None and name_element_type != "StringConstant":
            return _ask("Command name is a dynamic expression which cannot be statically validated")
    return _PASSTHROUGH


def check_encoded_command(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        if is_powershell_executable(cmd.name):
            if _ps_exe_has_param_abbreviation(cmd, "-encodedcommand", "-e"):
                return _ask("Command uses encoded parameters which obscure intent")
    return _PASSTHROUGH


def check_pwsh_command_or_file(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        if is_powershell_executable(cmd.name):
            return _ask("Command spawns a nested PowerShell process which cannot be validated")
    return _PASSTHROUGH


_DOWNLOADER_NAMES = frozenset(
    {"invoke-webrequest", "iwr", "invoke-restmethod", "irm", "new-object", "start-bitstransfer"}
)


def _is_downloader(name: str) -> bool:
    return name.lower() in _DOWNLOADER_NAMES


def _is_iex(name: str) -> bool:
    lower = name.lower()
    return lower == "invoke-expression" or lower == "iex"


def check_download_cradles(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    # Per-statement piped cradle (IWR ... | IEX)
    for statement in parsed.statements:
        cmds = statement.commands
        if len(cmds) < 2:
            continue
        if any(_is_downloader(c.name) for c in cmds) and any(_is_iex(c.name) for c in cmds):
            return _ask("Command downloads and executes remote code")
    # Cross-statement split cradle
    all_cmds = get_all_commands(parsed)
    if any(_is_downloader(c.name) for c in all_cmds) and any(_is_iex(c.name) for c in all_cmds):
        return _ask("Command downloads and executes remote code")
    return _PASSTHROUGH


def check_download_utilities(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        lower = cmd.name.lower()
        if lower == "start-bitstransfer":
            return _ask("Command downloads files via BITS transfer")
        if lower in ("certutil", "certutil.exe"):
            if any(a.lower() in ("-urlcache", "/urlcache") for a in cmd.args):
                return _ask("Command uses certutil to download from a URL")
        if lower in ("bitsadmin", "bitsadmin.exe"):
            if any(a.lower() == "/transfer" for a in cmd.args):
                return _ask("Command downloads files via BITS transfer")
    return _PASSTHROUGH


def check_add_type(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    if has_command_named(parsed, "Add-Type"):
        return _ask("Command compiles and loads .NET code")
    return _PASSTHROUGH


def check_com_object(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        if cmd.name.lower() != "new-object":
            continue
        if _ps_exe_has_param_abbreviation(cmd, "-comobject", "-com"):
            return _ask("Command instantiates a COM object which may have execution capabilities")
        # Extract -TypeName (named / colon-bound / positional-0) → CLM check.
        type_name: Optional[str] = None
        for i, a in enumerate(cmd.args):
            lower = a.lower()
            if lower.startswith("-t") and ":" in a:
                colon_idx = a.index(":")
                param_part = lower[:colon_idx]
                if "-typename".startswith(param_part):
                    type_name = a[colon_idx + 1:]
                    break
            if lower.startswith("-t") and "-typename".startswith(lower) and i + 1 < len(cmd.args):
                type_name = cmd.args[i + 1]
                break
        if type_name is None:
            value_params = {"-argumentlist", "-comobject", "-property"}
            switch_params = {"-strict"}
            i = 0
            while i < len(cmd.args):
                a = cmd.args[i]
                if a.startswith("-"):
                    lower = a.lower()
                    if lower.startswith("-t") and "-typename".startswith(lower):
                        i += 2
                        continue
                    if ":" in lower:
                        i += 1
                        continue
                    if lower in switch_params:
                        i += 1
                        continue
                    if lower in value_params:
                        i += 2
                        continue
                    i += 1
                    continue
                type_name = a
                break
        if type_name is not None and not is_clm_allowed_type(type_name):
            return _ask(
                f"New-Object instantiates .NET type '{type_name}' outside the ConstrainedLanguage allowlist"
            )
    return _PASSTHROUGH


def check_dangerous_file_path_execution(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        lower = cmd.name.lower()
        resolved = COMMON_ALIASES.get(lower, lower).lower()
        if resolved not in FILEPATH_EXECUTION_CMDLETS:
            continue
        if _ps_exe_has_param_abbreviation(cmd, "-filepath", "-f") or _ps_exe_has_param_abbreviation(cmd, "-literalpath", "-l"):
            return _ask(f"{cmd.name} -FilePath executes an arbitrary script file")
        for i, arg in enumerate(cmd.args):
            arg_type = cmd.element_types[i + 1] if cmd.element_types and i + 1 < len(cmd.element_types) else None
            if arg_type == "StringConstant" and arg and not arg.startswith("-"):
                return _ask(f"{cmd.name} with positional string argument binds to -FilePath and executes a script file")
    return _PASSTHROUGH


def check_for_each_member_name(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        lower = cmd.name.lower()
        resolved = COMMON_ALIASES.get(lower, lower).lower()
        if resolved != "foreach-object":
            continue
        if _ps_exe_has_param_abbreviation(cmd, "-membername", "-m"):
            return _ask("ForEach-Object -MemberName invokes methods by string name which cannot be validated")
        for i, arg in enumerate(cmd.args):
            arg_type = cmd.element_types[i + 1] if cmd.element_types and i + 1 < len(cmd.element_types) else None
            if arg_type == "StringConstant" and arg and not arg.startswith("-"):
                return _ask("ForEach-Object with positional string argument binds to -MemberName and invokes methods by name")
    return _PASSTHROUGH


_VERB_COLON_RE = re.compile(r"^[-–—―/]v[a-z]*:", re.IGNORECASE)
_VERB_RUNAS_RE = re.compile(r"^[-–—―/]v[a-z]*:['\"` ]*runas['\"` ]*$", re.IGNORECASE)


def check_start_process(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        lower = cmd.name.lower()
        if lower not in ("start-process", "saps", "start"):
            continue
        # Vector 1: -Verb RunAs (space syntax)
        if _ps_exe_has_param_abbreviation(cmd, "-Verb", "-v") and any(a.lower() == "runas" for a in cmd.args):
            return _ask("Command requests elevated privileges")
        # Colon syntax (a) structural via children
        if cmd.children:
            for i, raw_arg in enumerate(cmd.args):
                arg_clean = raw_arg.replace("`", "")
                if not _VERB_COLON_RE.match(arg_clean):
                    continue
                kids = cmd.children[i] if i < len(cmd.children) else None
                if not kids:
                    continue
                for child in kids:
                    if re.sub(r"['\"`\s]", "", child.text).lower() == "runas":
                        return _ask("Command requests elevated privileges")
        # Colon syntax (b) regex fallback
        for a in cmd.args:
            clean = a.replace("`", "")
            if _VERB_RUNAS_RE.match(clean):
                return _ask("Command requests elevated privileges")
        # Vector 2: target is a PowerShell executable
        for arg in cmd.args:
            stripped = re.sub(r"^['\"]|['\"]\Z", "", arg)
            if is_powershell_executable(stripped):
                return _ask("Start-Process launches a nested PowerShell process which cannot be validated")
    return _PASSTHROUGH


_SAFE_SCRIPT_BLOCK_CMDLETS = frozenset(
    {"where-object", "sort-object", "select-object", "group-object",
     "format-table", "format-list", "format-wide", "format-custom"}
)


def check_script_block_injection(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    if not derive_security_flags(parsed).has_script_blocks:
        return _PASSTHROUGH
    for cmd in get_all_commands(parsed):
        if cmd.name.lower() in DANGEROUS_SCRIPT_BLOCK_CMDLETS:
            return _ask("Command contains script block with dangerous cmdlet that may execute arbitrary code")

    def _safe(cmd: ParsedCommandElement) -> bool:
        lower = cmd.name.lower()
        if lower in _SAFE_SCRIPT_BLOCK_CMDLETS:
            return True
        alias = COMMON_ALIASES.get(lower)
        if alias and alias.lower() in _SAFE_SCRIPT_BLOCK_CMDLETS:
            return True
        return False

    if all(_safe(cmd) for cmd in get_all_commands(parsed)):
        return _PASSTHROUGH
    return _ask("Command contains script block that may execute arbitrary code")


def check_sub_expressions(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    if derive_security_flags(parsed).has_sub_expressions:
        return _ask("Command contains subexpressions $()")
    return _PASSTHROUGH


def check_expandable_strings(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    if derive_security_flags(parsed).has_expandable_strings:
        return _ask("Command contains expandable strings with embedded expressions")
    return _PASSTHROUGH


def check_splatting(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    if derive_security_flags(parsed).has_splatting:
        return _ask("Command uses splatting (@variable)")
    return _PASSTHROUGH


def check_stop_parsing(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    if derive_security_flags(parsed).has_stop_parsing:
        return _ask("Command uses stop-parsing token (--%)")
    return _PASSTHROUGH


def check_member_invocations(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    if derive_security_flags(parsed).has_member_invocations:
        return _ask("Command invokes .NET methods")
    return _PASSTHROUGH


def check_type_literals(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for t in parsed.type_literals or []:
        if not is_clm_allowed_type(t):
            return _ask(f"Command uses .NET type [{t}] outside the ConstrainedLanguage allowlist")
    return _PASSTHROUGH


def check_invoke_item(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        lower = cmd.name.lower()
        if lower in ("invoke-item", "ii"):
            return _ask(
                "Invoke-Item opens files with the default handler (ShellExecute). "
                "On executable files this runs arbitrary code."
            )
    return _PASSTHROUGH


_SCHEDULED_TASK_CMDLETS = frozenset(
    {"register-scheduledtask", "new-scheduledtask", "new-scheduledtaskaction", "set-scheduledtask"}
)


def check_scheduled_task(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        lower = cmd.name.lower()
        if lower in _SCHEDULED_TASK_CMDLETS:
            return _ask(f"{cmd.name} creates or modifies a scheduled task (persistence primitive)")
        if lower in ("schtasks", "schtasks.exe"):
            if any(a.lower() in ("/create", "/change", "-create", "-change") for a in cmd.args):
                return _ask("schtasks with create/change modifies scheduled tasks (persistence primitive)")
    return _PASSTHROUGH


_ENV_WRITE_CMDLETS = frozenset(
    {"set-item", "si", "new-item", "ni", "remove-item", "ri", "del", "rm", "rd", "rmdir",
     "erase", "clear-item", "cli", "set-content", "add-content", "ac"}
)


def check_env_var_manipulation(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    env_vars = get_variables_by_scope(parsed, "env")
    if not env_vars:
        return _PASSTHROUGH
    for cmd in get_all_commands(parsed):
        if cmd.name.lower() in _ENV_WRITE_CMDLETS:
            return _ask("Command modifies environment variables")
    if derive_security_flags(parsed).has_assignments and env_vars:
        return _ask("Command modifies environment variables")
    return _PASSTHROUGH


def check_module_loading(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        if cmd.name.lower() in MODULE_LOADING_CMDLETS:
            return _ask(
                "Command loads, installs, or downloads a PowerShell module or script, "
                "which can execute arbitrary code"
            )
    return _PASSTHROUGH


_RUNTIME_STATE_CMDLETS = frozenset(
    {"set-alias", "sal", "new-alias", "nal", "set-variable", "sv", "new-variable", "nv"}
)


def check_runtime_state_manipulation(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        raw = cmd.name.lower()
        lower = raw[raw.rfind("\\") + 1:] if "\\" in raw else raw
        if lower in _RUNTIME_STATE_CMDLETS:
            return _ask(
                "Command creates or modifies an alias or variable that can affect future command resolution"
            )
    return _PASSTHROUGH


# Single source of truth in dangerous_cmdlets (was duplicated verbatim here as
# _WMI_SPAWN_CMDLETS — a drift hazard: a fix to one copy missed the other).
_WMI_SPAWN_CMDLETS = WMI_CIM_CMDLETS


def check_wmi_process_spawn(parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    for cmd in get_all_commands(parsed):
        if cmd.name.lower() in _WMI_SPAWN_CMDLETS:
            return _ask(f"{cmd.name} can spawn arbitrary processes via WMI/CIM (Win32_Process Create)")
    return _PASSTHROUGH


# Order-sensitive: mirrors the source's validators array exactly.
_VALIDATORS = [
    check_invoke_expression,
    check_dynamic_command_name,
    check_encoded_command,
    check_pwsh_command_or_file,
    check_download_cradles,
    check_download_utilities,
    check_add_type,
    check_com_object,
    check_dangerous_file_path_execution,
    check_invoke_item,
    check_scheduled_task,
    check_for_each_member_name,
    check_start_process,
    check_script_block_injection,
    check_sub_expressions,
    check_expandable_strings,
    check_splatting,
    check_stop_parsing,
    check_member_invocations,
    check_type_literals,
    check_env_var_manipulation,
    check_module_loading,
    check_runtime_state_manipulation,
    check_wmi_process_spawn,
]


def powershell_command_is_safe(command: str, parsed: ParsedPowerShellCommand) -> PowerShellSecurityResult:
    """Main entry: validate a PowerShell command against dangerous patterns.

    AST-based; a failed parse (valid=False) returns 'ask' as a safe default.
    Returns the first validator that says 'ask', else 'passthrough'.
    """
    if not parsed.valid:
        return _ask("Could not parse command for security analysis")
    for validator in _VALIDATORS:
        result = validator(parsed)
        if result.behavior == "ask":
            return result
    return _PASSTHROUGH
