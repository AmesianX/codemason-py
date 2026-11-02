"""PowerShell common parameters (commonParameters.ts).

Available on all cmdlets via [CmdletBinding()] (about_CommonParameters + Get-Command).
Merged by path validation (per-cmdlet known-param sets) and read-only validation
(safe-flags check). Stored lowercase with leading dash — callers lowercase input.
"""

COMMON_SWITCHES = ["-verbose", "-debug"]

COMMON_VALUE_PARAMS = [
    "-erroraction",
    "-warningaction",
    "-informationaction",
    "-progressaction",
    "-errorvariable",
    "-warningvariable",
    "-informationvariable",
    "-outvariable",
    "-outbuffer",
    "-pipelinevariable",
]

COMMON_PARAMETERS = frozenset(COMMON_SWITCHES + COMMON_VALUE_PARAMS)
