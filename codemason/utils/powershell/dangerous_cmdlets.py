"""PowerShell cmdlets that execute arbitrary code (adapted from dangerousCmdlets.ts).

Consumed by the permission-engine validators (powershell_security) and the UI
suggestion gate (static_prefix). Single source so the lists don't drift.
"""

from typing import FrozenSet, List, Set

from ..permissions.dangerous_patterns import CROSS_PLATFORM_CODE_EXEC
from .parser import COMMON_ALIASES

# Cmdlets that accept a -FilePath (or positional path) and execute the file.
FILEPATH_EXECUTION_CMDLETS: FrozenSet[str] = frozenset(
    {"invoke-command", "start-job", "start-threadjob", "register-scheduledjob"}
)

# Cmdlets where a scriptblock argument executes arbitrary code (not just filtering).
DANGEROUS_SCRIPT_BLOCK_CMDLETS: FrozenSet[str] = frozenset(
    {
        "invoke-command", "invoke-expression", "start-job", "start-threadjob",
        "register-scheduledjob", "register-engineevent", "register-objectevent",
        "register-wmievent", "new-pssession", "enter-pssession",
    }
)

# Cmdlets that load and execute module/script code (.psm1 runs on import).
MODULE_LOADING_CMDLETS: FrozenSet[str] = frozenset(
    {"import-module", "ipmo", "install-module", "save-module", "update-module",
     "install-script", "save-script"}
)

# Shells and process spawners.
_SHELLS_AND_SPAWNERS = [
    "pwsh", "powershell", "cmd", "bash", "wsl", "sh",
    "start-process", "start", "add-type", "new-object",
]

# Network cmdlets — wildcard rules enable exfil/download without prompt.
NETWORK_CMDLETS: FrozenSet[str] = frozenset({"invoke-webrequest", "invoke-restmethod"})

# Alias/variable mutation cmdlets (rebind command resolution / poison defaults).
ALIAS_HIJACK_CMDLETS: FrozenSet[str] = frozenset(
    {"set-alias", "sal", "new-alias", "nal", "set-variable", "sv", "new-variable", "nv"}
)

# WMI/CIM process spawn — Start-Process equivalent that bypasses checkStartProcess
# (Invoke-WmiMethod -Class Win32_Process -Name Create). Source finding #34.
WMI_CIM_CMDLETS: FrozenSet[str] = frozenset({"invoke-wmimethod", "iwmi", "invoke-cimmethod"})

# Cmdlets in CMDLET_ALLOWLIST with additionalCommandIsDangerousCallback — a
# `Cmdlet:*` wildcard would bypass the callback forever, so never suggest it.
ARG_GATED_CMDLETS: FrozenSet[str] = frozenset(
    {
        "select-object", "sort-object", "group-object", "where-object", "measure-object",
        "write-output", "write-host", "start-sleep", "format-table", "format-list",
        "format-wide", "format-custom", "out-string", "out-host",
        # Native executables with callback-gated args (ipconfig /flushdns rejected,
        # /all allowed).
        "ipconfig", "hostname", "route",
    }
)


def _aliases_of(targets: Set[str]) -> List[str]:
    """COMMON_ALIASES keys whose canonical target is in `targets` (lowercased)."""
    return [alias for alias, target in COMMON_ALIASES.items() if target.lower() in targets]


def _build_never_suggest() -> FrozenSet[str]:
    core: Set[str] = set()
    core.update(_SHELLS_AND_SPAWNERS)
    core.update(FILEPATH_EXECUTION_CMDLETS)
    core.update(DANGEROUS_SCRIPT_BLOCK_CMDLETS)
    core.update(MODULE_LOADING_CMDLETS)
    core.update(NETWORK_CMDLETS)
    core.update(ALIAS_HIJACK_CMDLETS)
    core.update(WMI_CIM_CMDLETS)
    core.update(ARG_GATED_CMDLETS)
    core.add("foreach-object")
    # Interpreters/runners — bare `node:*` would auto-allow -e/-p code. Multi-word
    # entries ('npm run') are filtered: NEVER_SUGGEST is a single-name lookup.
    core.update(p for p in CROSS_PLATFORM_CODE_EXEC if " " not in p)
    result = set(core)
    result.update(_aliases_of(core))
    return frozenset(result)


# Commands to never suggest as a wildcard prefix in the permission dialog.
# NOT WIRED: codemason has not ported the permission-rule/wildcard-suggestion UI,
# so NEVER_SUGGEST (and the *_CMDLETS sets that only feed it — NETWORK_CMDLETS,
# ALIAS_HIJACK_CMDLETS, ARG_GATED_CMDLETS, _SHELLS_AND_SPAWNERS) has no live
# consumer today. Retained as reference for when that UI is ported. The sets that
# ARE live (FILEPATH_EXECUTION / MODULE_LOADING / DANGEROUS_SCRIPT_BLOCK /
# WMI_CIM) are imported directly by powershell_security.
NEVER_SUGGEST: FrozenSet[str] = _build_never_suggest()
