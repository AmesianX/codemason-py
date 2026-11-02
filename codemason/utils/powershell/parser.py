"""PowerShell command parser (adapted from src/utils/powershell/parser.ts).

Key architecture (faithful to the source): we do NOT reimplement a PowerShell
parser. PARSE_SCRIPT_BODY is a PowerShell script that runs the platform's own
`[System.Management.Automation.Language.Parser]::ParseInput` and emits a compact
JSON AST; this module spawns PowerShell to run it, then transforms that JSON into
typed structures and provides analysis accessors the security/permission layers
consume.

The command is passed as a UTF-8 Base64 `$EncodedCommand` variable (no here-string
injection), and the whole parse script is handed to PowerShell via
`-EncodedCommand` (UTF-16LE Base64) — no temp files, no escaping issues.
"""

import base64
import os
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..powershell_detect import get_cached_powershell_path

# =============================================================================
# The PowerShell parse script. Inlined (not a data file) so it always ships with
# the package. Body is byte-identical to the source's PARSE_SCRIPT_BODY.
# =============================================================================
PARSE_SCRIPT_BODY = r"""
if (-not $EncodedCommand) {
    Write-Output '{"valid":false,"errors":[{"message":"No command provided","errorId":"NoInput"}],"statements":[],"variables":[],"hasStopParsing":false,"originalCommand":""}'
    exit 0
}

$Command = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($EncodedCommand))

$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $Command,
    [ref]$tokens,
    [ref]$parseErrors
)

$allVariables = [System.Collections.ArrayList]::new()

function Get-RawCommandElements {
    param([System.Management.Automation.Language.CommandAst]$CmdAst)
    $elems = [System.Collections.ArrayList]::new()
    foreach ($ce in $CmdAst.CommandElements) {
        $ceData = @{ type = $ce.GetType().Name; text = $ce.Extent.Text }
        if ($ce.PSObject.Properties['Value'] -and $null -ne $ce.Value -and $ce.Value -is [string]) {
            $ceData.value = $ce.Value
        }
        if ($ce -is [System.Management.Automation.Language.CommandExpressionAst]) {
            $ceData.expressionType = $ce.Expression.GetType().Name
        }
        $a=$ce.Argument;if($a){$ceData.children=@(@{type=$a.GetType().Name;text=$a.Extent.Text})}
        [void]$elems.Add($ceData)
    }
    return $elems
}

function Get-RawRedirections {
    param($Redirections)
    $result = [System.Collections.ArrayList]::new()
    foreach ($redir in $Redirections) {
        $redirData = @{ type = $redir.GetType().Name }
        if ($redir -is [System.Management.Automation.Language.FileRedirectionAst]) {
            $redirData.append = [bool]$redir.Append
            $redirData.fromStream = $redir.FromStream.ToString()
            $redirData.locationText = $redir.Location.Extent.Text
        }
        [void]$result.Add($redirData)
    }
    return $result
}

function Get-SecurityPatterns($A) {
    $p = @{}
    foreach ($n in $A.FindAll({ param($x)
        $x -is [System.Management.Automation.Language.MemberExpressionAst] -or
        $x -is [System.Management.Automation.Language.SubExpressionAst] -or
        $x -is [System.Management.Automation.Language.ArrayExpressionAst] -or
        $x -is [System.Management.Automation.Language.ExpandableStringExpressionAst] -or
        $x -is [System.Management.Automation.Language.ScriptBlockExpressionAst] -or
        $x -is [System.Management.Automation.Language.ParenExpressionAst]
    }, $true)) { switch ($n.GetType().Name) {
        'InvokeMemberExpressionAst' { $p.hasMemberInvocations = $true }
        'MemberExpressionAst' { $p.hasMemberInvocations = $true }
        'SubExpressionAst' { $p.hasSubExpressions = $true }
        'ArrayExpressionAst' { $p.hasSubExpressions = $true }
        'ParenExpressionAst' { $p.hasSubExpressions = $true }
        'ExpandableStringExpressionAst' { $p.hasExpandableStrings = $true }
        'ScriptBlockExpressionAst' { $p.hasScriptBlocks = $true }
    }}
    if ($p.Count -gt 0) { return $p }
    return $null
}

$varExprs = $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.VariableExpressionAst] }, $true)
foreach ($v in $varExprs) {
    [void]$allVariables.Add(@{
        path = $v.VariablePath.ToString()
        isSplatted = [bool]$v.Splatted
    })
}

$typeLiterals = [System.Collections.ArrayList]::new()
foreach ($t in $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.TypeExpressionAst] -or
    $n -is [System.Management.Automation.Language.TypeConstraintAst]
}, $true)) { [void]$typeLiterals.Add($t.TypeName.FullName) }

$hasStopParsing = $false
$tk = [System.Management.Automation.Language.TokenKind]
foreach ($tok in $tokens) {
    if ($tok.Kind -eq $tk::MinusMinus) { $hasStopParsing = $true; break }
    if ($tok.Kind -eq $tk::Generic -and ($tok.Text -replace '[\u2013\u2014\u2015]','-') -eq '--%') {
        $hasStopParsing = $true; break
    }
}

$statements = [System.Collections.ArrayList]::new()

function Process-BlockStatements {
    param($Block)
    if (-not $Block) { return }

    foreach ($stmt in $Block.Statements) {
        $statement = @{
            type = $stmt.GetType().Name
            text = $stmt.Extent.Text
        }

        if ($stmt -is [System.Management.Automation.Language.PipelineAst]) {
            $elements = [System.Collections.ArrayList]::new()
            foreach ($element in $stmt.PipelineElements) {
                $elemData = @{
                    type = $element.GetType().Name
                    text = $element.Extent.Text
                }

                if ($element -is [System.Management.Automation.Language.CommandAst]) {
                    $elemData.commandElements = @(Get-RawCommandElements -CmdAst $element)
                    $elemData.redirections = @(Get-RawRedirections -Redirections $element.Redirections)
                } elseif ($element -is [System.Management.Automation.Language.CommandExpressionAst]) {
                    $elemData.expressionType = $element.Expression.GetType().Name
                    $elemData.redirections = @(Get-RawRedirections -Redirections $element.Redirections)
                }

                [void]$elements.Add($elemData)
            }
            $statement.elements = @($elements)

            $allNestedCmds = $stmt.FindAll(
                { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
                $true
            )
            $nestedCmds = [System.Collections.ArrayList]::new()
            foreach ($cmd in $allNestedCmds) {
                if ($cmd.Parent -eq $stmt) { continue }
                $nested = @{
                    type = $cmd.GetType().Name
                    text = $cmd.Extent.Text
                    commandElements = @(Get-RawCommandElements -CmdAst $cmd)
                    redirections = @(Get-RawRedirections -Redirections $cmd.Redirections)
                }
                [void]$nestedCmds.Add($nested)
            }
            if ($nestedCmds.Count -gt 0) {
                $statement.nestedCommands = @($nestedCmds)
            }
            $r = $stmt.FindAll({param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst]}, $true)
            if ($r.Count -gt 0) {
                $rr = @(Get-RawRedirections -Redirections $r)
                $statement.redirections = if ($statement.redirections) { @($statement.redirections) + $rr } else { $rr }
            }
        } else {
            $nestedCmdAsts = $stmt.FindAll(
                { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
                $true
            )
            $nested = [System.Collections.ArrayList]::new()
            foreach ($cmd in $nestedCmdAsts) {
                [void]$nested.Add(@{
                    type = 'CommandAst'
                    text = $cmd.Extent.Text
                    commandElements = @(Get-RawCommandElements -CmdAst $cmd)
                    redirections = @(Get-RawRedirections -Redirections $cmd.Redirections)
                })
            }
            if ($nested.Count -gt 0) {
                $statement.nestedCommands = @($nested)
            }
            $r = $stmt.FindAll({param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst]}, $true)
            if ($r.Count -gt 0) { $statement.redirections = @(Get-RawRedirections -Redirections $r) }
        }

        $sp = Get-SecurityPatterns $stmt
        if ($sp) { $statement.securityPatterns = $sp }

        [void]$statements.Add($statement)
    }

    if ($Block.Traps) {
        foreach ($trap in $Block.Traps) {
            $statement = @{
                type = 'TrapStatementAst'
                text = $trap.Extent.Text
            }
            $nestedCmdAsts = $trap.FindAll(
                { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
                $true
            )
            $nestedCmds = [System.Collections.ArrayList]::new()
            foreach ($cmd in $nestedCmdAsts) {
                $nested = @{
                    type = $cmd.GetType().Name
                    text = $cmd.Extent.Text
                    commandElements = @(Get-RawCommandElements -CmdAst $cmd)
                    redirections = @(Get-RawRedirections -Redirections $cmd.Redirections)
                }
                [void]$nestedCmds.Add($nested)
            }
            if ($nestedCmds.Count -gt 0) {
                $statement.nestedCommands = @($nestedCmds)
            }
            $r = $trap.FindAll({param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst]}, $true)
            if ($r.Count -gt 0) { $statement.redirections = @(Get-RawRedirections -Redirections $r) }
            $sp = Get-SecurityPatterns $trap
            if ($sp) { $statement.securityPatterns = $sp }
            [void]$statements.Add($statement)
        }
    }
}

Process-BlockStatements -Block $ast.BeginBlock
Process-BlockStatements -Block $ast.ProcessBlock
Process-BlockStatements -Block $ast.EndBlock
Process-BlockStatements -Block $ast.CleanBlock
Process-BlockStatements -Block $ast.DynamicParamBlock

if ($ast.ParamBlock) {
  $pb = $ast.ParamBlock
  $pn = [System.Collections.ArrayList]::new()
  foreach ($c in $pb.FindAll({param($n) $n -is [System.Management.Automation.Language.CommandAst]}, $true)) {
    [void]$pn.Add(@{type='CommandAst';text=$c.Extent.Text;commandElements=@(Get-RawCommandElements -CmdAst $c);redirections=@(Get-RawRedirections -Redirections $c.Redirections)})
  }
  $pr = $pb.FindAll({param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst]}, $true)
  $ps = Get-SecurityPatterns $pb
  if ($pn.Count -gt 0 -or $pr.Count -gt 0 -or $ps) {
    $st = @{type='ParamBlockAst';text=$pb.Extent.Text}
    if ($pn.Count -gt 0) { $st.nestedCommands = @($pn) }
    if ($pr.Count -gt 0) { $st.redirections = @(Get-RawRedirections -Redirections $pr) }
    if ($ps) { $st.securityPatterns = $ps }
    [void]$statements.Add($st)
  }
}

$hasUsingStatements = $ast.UsingStatements -and $ast.UsingStatements.Count -gt 0
$hasScriptRequirements = $ast.ScriptRequirements -ne $null

$output = @{
    valid = ($parseErrors.Count -eq 0)
    errors = @($parseErrors | ForEach-Object {
        @{
            message = $_.Message
            errorId = $_.ErrorId
        }
    })
    statements = @($statements)
    variables = @($allVariables)
    hasStopParsing = $hasStopParsing
    originalCommand = $Command
    typeLiterals = @($typeLiterals)
    hasUsingStatements = [bool]$hasUsingStatements
    hasScriptRequirements = [bool]$hasScriptRequirements
}

$output | ConvertTo-Json -Depth 10 -Compress
"""


# =============================================================================
# Command-length budget (derived from PARSE_SCRIPT_BODY length; UTF-8 BYTES).
# Windows CreateProcess caps the command line at 32,767 chars. The encoding chain
# (command → UTF-8 b64 → wrapper → script → UTF-16LE → b64 → argv) inflates ~8/3×,
# so the safe UTF-8 byte budget is computed from the script length and MUST be
# compared against len(command.encode('utf-8')), never len(command) — a CJK char
# is 1 code point but 3 UTF-8 bytes (source Finding #36).
# =============================================================================
_WINDOWS_ARGV_CAP = 32_767
_FIXED_ARGV_OVERHEAD = 200
_ENCODED_CMD_WRAPPER = len("$EncodedCommand = ''\n")
_SAFETY_MARGIN = 100
_SCRIPT_CHARS_BUDGET = ((_WINDOWS_ARGV_CAP - _FIXED_ARGV_OVERHEAD) * 3) / 8
_CMD_B64_BUDGET = _SCRIPT_CHARS_BUDGET - len(PARSE_SCRIPT_BODY) - _ENCODED_CMD_WRAPPER
WINDOWS_MAX_COMMAND_LENGTH = max(0, int((_CMD_B64_BUDGET * 3) / 4) - _SAFETY_MARGIN)

# Pre-existing Unix value (argv limits are ~128KB/arg; far above this). The
# Windows derivation must NOT be applied on Unix (would regress mid-size commands
# that parse fine and reach the deny loop). Platform-gated. UTF-8 BYTES.
_UNIX_MAX_COMMAND_LENGTH = 4_500
MAX_COMMAND_LENGTH = (
    WINDOWS_MAX_COMMAND_LENGTH if sys.platform == "win32" else _UNIX_MAX_COMMAND_LENGTH
)

# Warm-spawn is ~450ms; 5s covers loaded hosts. Override via env for tests.
_DEFAULT_PARSE_TIMEOUT_MS = 5_000


def _get_parse_timeout_ms() -> int:
    env = os.environ.get("CODEMASON_PWSH_PARSE_TIMEOUT_MS")
    if env:
        try:
            v = int(env)
            if v > 0:
                return v
        except ValueError:
            pass
    return _DEFAULT_PARSE_TIMEOUT_MS


# =============================================================================
# Typed output structures (ParsedPowerShellCommand and friends).
# =============================================================================
@dataclass
class CommandElementChild:
    type: str  # CommandElementType
    text: str


@dataclass
class ParsedRedirection:
    operator: str  # '>' | '>>' | '2>' | '2>>' | '*>' | '*>>' | '2>&1'
    target: str
    is_merging: bool


@dataclass
class ParsedCommandElement:
    name: str
    name_type: str  # 'cmdlet' | 'application' | 'unknown'
    element_type: str  # PipelineElementType
    args: List[str] = field(default_factory=list)
    text: str = ""
    element_types: Optional[List[str]] = None  # CommandElementType[]
    children: Optional[List[Optional[List[CommandElementChild]]]] = None
    redirections: Optional[List[ParsedRedirection]] = None


@dataclass
class ParsedStatement:
    statement_type: str  # StatementType
    commands: List[ParsedCommandElement] = field(default_factory=list)
    redirections: List[ParsedRedirection] = field(default_factory=list)
    text: str = ""
    nested_commands: Optional[List[ParsedCommandElement]] = None
    security_patterns: Optional[Dict[str, bool]] = None


@dataclass
class ParsedVariable:
    path: str
    is_splatted: bool


@dataclass
class ParseError:
    message: str
    error_id: str


@dataclass
class ParsedPowerShellCommand:
    valid: bool
    errors: List[ParseError] = field(default_factory=list)
    statements: List[ParsedStatement] = field(default_factory=list)
    variables: List[ParsedVariable] = field(default_factory=list)
    has_stop_parsing: bool = False
    original_command: str = ""
    type_literals: Optional[List[str]] = None
    has_using_statements: Optional[bool] = None
    has_script_requirements: Optional[bool] = None


def _make_invalid_result(command: str, message: str, error_id: str) -> ParsedPowerShellCommand:
    return ParsedPowerShellCommand(
        valid=False,
        errors=[ParseError(message=message, error_id=error_id)],
        statements=[],
        variables=[],
        has_stop_parsing=False,
        original_command=command,
    )


def _ensure_array(value: Any) -> List[Any]:
    """PowerShell 5.1's ConvertTo-Json unwraps single-element arrays into plain
    objects; normalise to a list (ensureArray)."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# =============================================================================
# Encoding + script assembly.
# =============================================================================
def _to_utf16le_base64(text: str) -> str:
    """Base64 of UTF-16LE (required by PowerShell -EncodedCommand)."""
    return base64.b64encode(text.encode("utf-16-le")).decode("ascii")


def build_parse_script(command: str) -> str:
    """Prefix the UTF-8 Base64 $EncodedCommand var onto PARSE_SCRIPT_BODY."""
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return f"$EncodedCommand = '{encoded}'\n{PARSE_SCRIPT_BODY}"


# =============================================================================
# Raw .NET AST type name → our union mappers.
# =============================================================================
_STATEMENT_TYPES = {
    "PipelineAst", "PipelineChainAst", "AssignmentStatementAst", "IfStatementAst",
    "ForStatementAst", "ForEachStatementAst", "WhileStatementAst", "DoWhileStatementAst",
    "DoUntilStatementAst", "SwitchStatementAst", "TryStatementAst", "TrapStatementAst",
    "FunctionDefinitionAst", "DataStatementAst",
}


def map_statement_type(raw_type: str) -> str:
    return raw_type if raw_type in _STATEMENT_TYPES else "UnknownStatementAst"


def map_element_type(raw_type: str, expression_type: Optional[str] = None) -> str:
    """Map raw AST node type to CommandElementType."""
    if raw_type == "ScriptBlockExpressionAst":
        return "ScriptBlock"
    if raw_type in ("SubExpressionAst", "ArrayExpressionAst"):
        # ArrayExpressionAst (@()) is a sibling of SubExpressionAst; both run
        # arbitrary pipelines with side effects. Map both → SubExpression.
        return "SubExpression"
    if raw_type == "ExpandableStringExpressionAst":
        return "ExpandableString"
    if raw_type in ("InvokeMemberExpressionAst", "MemberExpressionAst"):
        return "MemberInvocation"
    if raw_type == "VariableExpressionAst":
        return "Variable"
    if raw_type in ("StringConstantExpressionAst", "ConstantExpressionAst"):
        # ConstantExpressionAst covers numeric literals — inert values, treat as
        # StringConstant so numeric args don't false-positive as 'Other'.
        return "StringConstant"
    if raw_type == "CommandParameterAst":
        return "Parameter"
    if raw_type == "ParenExpressionAst":
        return "SubExpression"
    if raw_type == "CommandExpressionAst":
        # Delegate to the wrapped expression type.
        if expression_type:
            return map_element_type(expression_type)
        return "Other"
    return "Other"


import re as _re

_CMDLET_RE = _re.compile(r"^[A-Za-z]+-[A-Za-z][A-Za-z0-9_]*$")
_PATHISH_RE = _re.compile(r"[.\\/]")
_DRIVE_RE = _re.compile(r"^[A-Za-z]:")
# Any non-ASCII char. The TS source range matches astral chars in JS via their
# UTF-16 surrogate halves; Python `re` matches code points, so a BMP-only range
# would miss U+FFFF+ (e.g. a math-bold cmdlet spoof). Match all non-ASCII so the
# .NET OrdinalIgnoreCase cmdlet-spoof gate cannot be slipped.
_NONASCII_RE = _re.compile(r"[^\x00-\x7f]")


def classify_command_name(name: str) -> str:
    """Classify a command name as cmdlet | application | unknown."""
    if _CMDLET_RE.match(name):
        return "cmdlet"
    if _PATHISH_RE.search(name):
        return "application"
    return "unknown"


def strip_module_prefix(name: str) -> str:
    """Strip a module prefix (Module\\Cmdlet → Cmdlet), but never a real path."""
    idx = name.rfind("\\")
    if idx < 0:
        return name
    if _DRIVE_RE.match(name) or name.startswith("\\\\") or name.startswith(".\\") or name.startswith("..\\"):
        return name
    return name[idx + 1:]


# =============================================================================
# Transforms: raw JSON dict → typed structures.
# =============================================================================
def transform_redirection(raw: Dict[str, Any]) -> ParsedRedirection:
    if raw.get("type") == "MergingRedirectionAst":
        return ParsedRedirection(operator="2>&1", target="", is_merging=True)
    append = bool(raw.get("append") or False)
    from_stream = raw.get("fromStream") or "Output"
    if append:
        operator = {"Error": "2>>", "All": "*>>"}.get(from_stream, ">>")
    else:
        operator = {"Error": "2>", "All": "*>"}.get(from_stream, ">")
    return ParsedRedirection(operator=operator, target=raw.get("locationText") or "", is_merging=False)


def transform_command_ast(raw: Dict[str, Any]) -> ParsedCommandElement:
    cmd_elements = _ensure_array(raw.get("commandElements"))
    name = ""
    args: List[str] = []
    element_types: List[str] = []
    children: List[Optional[List[CommandElementChild]]] = []
    has_children = False
    name_type = "unknown"

    if cmd_elements:
        first = cmd_elements[0]
        is_first_string_literal = first.get("type") in (
            "StringConstantExpressionAst", "ExpandableStringExpressionAst"
        )
        first_value = first.get("value")
        raw_name_unstripped = (
            first_value
            if is_first_string_literal and isinstance(first_value, str)
            else first.get("text", "")
        )
        # Strip surrounding quotes so downstream readers see the bare name.
        raw_name = _re.sub(r"^['\"]|['\"]\Z", "", raw_name_unstripped)
        # Non-ASCII in cmdlet position is suspicious (.NET OrdinalIgnoreCase folds
        # e.g. U+017F→S, which Python .lower() misses) → force 'application' to
        # gate auto-allow. Source Finding #31.
        if _NONASCII_RE.search(raw_name):
            name_type = "application"
        else:
            name_type = classify_command_name(raw_name)
        name = strip_module_prefix(raw_name)
        element_types.append(map_element_type(first.get("type", ""), first.get("expressionType")))

        for ce in cmd_elements[1:]:
            is_string_literal = ce.get("type") in (
                "StringConstantExpressionAst", "ExpandableStringExpressionAst"
            )
            ce_value = ce.get("value")
            args.append(ce_value if is_string_literal and ce_value is not None else ce.get("text", ""))
            element_types.append(map_element_type(ce.get("type", ""), ce.get("expressionType")))
            raw_children = _ensure_array(ce.get("children"))
            if raw_children:
                has_children = True
                children.append(
                    [CommandElementChild(type=map_element_type(c.get("type", "")), text=c.get("text", ""))
                     for c in raw_children]
                )
            else:
                children.append(None)

    result = ParsedCommandElement(
        name=name,
        name_type=name_type,
        element_type="CommandAst",
        args=args,
        text=raw.get("text", ""),
        element_types=element_types,
        children=children if has_children else None,
    )
    raw_redirs = _ensure_array(raw.get("redirections"))
    if raw_redirs:
        result.redirections = [transform_redirection(r) for r in raw_redirs]
    return result


def transform_expression_element(raw: Dict[str, Any]) -> ParsedCommandElement:
    element_type = "ParenExpressionAst" if raw.get("type") == "ParenExpressionAst" else "CommandExpressionAst"
    return ParsedCommandElement(
        name=raw.get("text", ""),
        name_type="unknown",
        element_type=element_type,
        args=[],
        text=raw.get("text", ""),
        element_types=[map_element_type(raw.get("type", ""), raw.get("expressionType"))],
    )


def transform_statement(raw: Dict[str, Any]) -> ParsedStatement:
    statement_type = map_statement_type(raw.get("type", ""))
    commands: List[ParsedCommandElement] = []
    redirections: List[ParsedRedirection] = []

    if raw.get("elements") is not None:
        for elem in _ensure_array(raw.get("elements")):
            if elem.get("type") == "CommandAst":
                commands.append(transform_command_ast(elem))
            else:
                commands.append(transform_expression_element(elem))
            for redir in _ensure_array(elem.get("redirections")):
                redirections.append(transform_redirection(redir))
        # Merge the statement-level deep-FindAll redirections, deduped by
        # (operator, target) — they re-discover direct-element ones.
        seen = {(r.operator, r.target) for r in redirections}
        for redir in _ensure_array(raw.get("redirections")):
            r = transform_redirection(redir)
            key = (r.operator, r.target)
            if key not in seen:
                seen.add(key)
                redirections.append(r)
    else:
        commands.append(
            ParsedCommandElement(
                name=raw.get("text", ""),
                name_type="unknown",
                element_type="CommandExpressionAst",
                args=[],
                text=raw.get("text", ""),
            )
        )
        for redir in _ensure_array(raw.get("redirections")):
            redirections.append(transform_redirection(redir))

    nested_commands = None
    raw_nested = _ensure_array(raw.get("nestedCommands"))
    if raw_nested:
        nested_commands = [transform_command_ast(n) for n in raw_nested]

    result = ParsedStatement(
        statement_type=statement_type,
        commands=commands,
        redirections=redirections,
        text=raw.get("text", ""),
        nested_commands=nested_commands,
    )
    sp = raw.get("securityPatterns")
    if sp:
        result.security_patterns = sp
    return result


def _transform_raw_output(raw: Dict[str, Any]) -> ParsedPowerShellCommand:
    result = ParsedPowerShellCommand(
        valid=bool(raw.get("valid")),
        errors=[ParseError(message=e.get("message", ""), error_id=e.get("errorId", ""))
                for e in _ensure_array(raw.get("errors"))],
        statements=[transform_statement(s) for s in _ensure_array(raw.get("statements"))],
        variables=[ParsedVariable(path=v.get("path", ""), is_splatted=bool(v.get("isSplatted")))
                   for v in _ensure_array(raw.get("variables"))],
        has_stop_parsing=bool(raw.get("hasStopParsing")),
        original_command=raw.get("originalCommand", ""),
    )
    tl = _ensure_array(raw.get("typeLiterals"))
    if tl:
        result.type_literals = tl
    if raw.get("hasUsingStatements"):
        result.has_using_statements = True
    if raw.get("hasScriptRequirements"):
        result.has_script_requirements = True
    return result


# =============================================================================
# Parse invocation (spawn PowerShell, one retry on timeout, LRU-memoized).
# =============================================================================
_TRANSIENT_ERROR_IDS = frozenset(
    {"PwshSpawnError", "PwshError", "PwshTimeout", "EmptyOutput", "InvalidJson"}
)

_parse_cache: "OrderedDict[str, ParsedPowerShellCommand]" = OrderedDict()
_PARSE_CACHE_MAX = 256


def _no_window_flag() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def _parse_powershell_command_impl(command: str) -> ParsedPowerShellCommand:
    # UTF-8 BYTE budget (not code points) — see MAX_COMMAND_LENGTH note.
    command_bytes = len(command.encode("utf-8"))
    if command_bytes > MAX_COMMAND_LENGTH:
        return _make_invalid_result(
            command,
            f"Command too long for parsing ({command_bytes} bytes). "
            f"Maximum supported length is {MAX_COMMAND_LENGTH} bytes.",
            "CommandTooLong",
        )

    pwsh_path = get_cached_powershell_path()
    if not pwsh_path:
        return _make_invalid_result(command, "PowerShell is not available", "NoPowerShell")

    script = build_parse_script(command)
    encoded_script = _to_utf16le_base64(script)
    args = [pwsh_path, "-NoProfile", "-NonInteractive", "-NoLogo", "-EncodedCommand", encoded_script]
    timeout_s = _get_parse_timeout_ms() / 1000

    stdout = ""
    stderr = ""
    code: Optional[int] = None
    timed_out = False
    for _attempt in range(2):
        timed_out = False
        try:
            proc = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                encoding="utf-8",
                errors="replace",
                creationflags=_no_window_flag(),
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except (OSError, subprocess.SubprocessError) as e:
            return _make_invalid_result(command, f"Failed to spawn PowerShell: {e}", "PwshSpawnError")
        if not timed_out:
            break

    if timed_out:
        return _make_invalid_result(
            command, f"pwsh timed out after {_get_parse_timeout_ms()}ms (2 attempts)", "PwshTimeout"
        )
    if code != 0:
        return _make_invalid_result(command, f"pwsh exited with code {code}: {stderr}", "PwshError")

    trimmed = stdout.strip()
    if not trimmed:
        return _make_invalid_result(command, "No output from PowerShell parser", "EmptyOutput")

    try:
        import json
        raw = json.loads(trimmed)
        return _transform_raw_output(raw)
    except (ValueError, TypeError):
        return _make_invalid_result(command, "Invalid JSON from PowerShell parser", "InvalidJson")


def parse_powershell_command(command: str) -> ParsedPowerShellCommand:
    """Parse a PowerShell command via the native AST parser (LRU-memoized).

    Transient failures (spawn/timeout/JSON) are evicted so later calls retry;
    deterministic results (valid parses, CommandTooLong) stay cached.
    """
    cached = _parse_cache.get(command)
    if cached is not None:
        _parse_cache.move_to_end(command)
        return cached
    result = _parse_powershell_command_impl(command)
    if not (not result.valid and result.errors and result.errors[0].error_id in _TRANSIENT_ERROR_IDS):
        _parse_cache[command] = result
        _parse_cache.move_to_end(command)
        while len(_parse_cache) > _PARSE_CACHE_MAX:
            _parse_cache.popitem(last=False)
    return result


def reset_parse_cache() -> None:
    """Clear the parse cache (tests only)."""
    _parse_cache.clear()


# =============================================================================
# Analysis accessors — derived from the parsed AST (consumed by security /
# permission layers). Port of the parser.ts accessor section.
# =============================================================================

# Common PowerShell aliases → canonical cmdlet (COMMON_ALIASES). Resolution is
# single-hop. SECURITY: aliases that collide with native .exe on PS Core 6+
# (sc/sort/curl/wget) are deliberately OMITTED — resolving them would auto-allow
# the wrong program.
COMMON_ALIASES: Dict[str, str] = {
    # Directory listing
    "ls": "Get-ChildItem", "dir": "Get-ChildItem", "gci": "Get-ChildItem",
    # Content
    "cat": "Get-Content", "type": "Get-Content", "gc": "Get-Content",
    # Navigation
    "cd": "Set-Location", "sl": "Set-Location", "chdir": "Set-Location",
    "pushd": "Push-Location", "popd": "Pop-Location", "pwd": "Get-Location", "gl": "Get-Location",
    # Items
    "gi": "Get-Item", "gp": "Get-ItemProperty", "ni": "New-Item",
    "mkdir": "New-Item", "md": "New-Item",
    "ri": "Remove-Item", "del": "Remove-Item", "rd": "Remove-Item", "rmdir": "Remove-Item",
    "rm": "Remove-Item", "erase": "Remove-Item",
    "mi": "Move-Item", "mv": "Move-Item", "move": "Move-Item",
    "ci": "Copy-Item", "cp": "Copy-Item", "copy": "Copy-Item", "cpi": "Copy-Item",
    "si": "Set-Item", "rni": "Rename-Item", "ren": "Rename-Item",
    # Process
    "ps": "Get-Process", "gps": "Get-Process", "kill": "Stop-Process", "spps": "Stop-Process",
    "start": "Start-Process", "saps": "Start-Process", "sajb": "Start-Job", "ipmo": "Import-Module",
    # Output
    "echo": "Write-Output", "write": "Write-Output", "sleep": "Start-Sleep",
    # Help
    "help": "Get-Help", "man": "Get-Help", "gcm": "Get-Command",
    # Service
    "gsv": "Get-Service",
    # Variables
    "gv": "Get-Variable", "sv": "Set-Variable",
    # History
    "h": "Get-History", "history": "Get-History",
    # Invoke
    "iex": "Invoke-Expression", "iwr": "Invoke-WebRequest", "irm": "Invoke-RestMethod",
    "icm": "Invoke-Command", "ii": "Invoke-Item",
    # PSSession — remote code execution surface
    "nsn": "New-PSSession", "etsn": "Enter-PSSession", "exsn": "Exit-PSSession",
    "gsn": "Get-PSSession", "rsn": "Remove-PSSession",
    # Misc
    "cls": "Clear-Host", "clear": "Clear-Host", "select": "Select-Object",
    "where": "Where-Object", "foreach": "ForEach-Object", "%": "ForEach-Object", "?": "Where-Object",
    "measure": "Measure-Object", "ft": "Format-Table", "fl": "Format-List", "fw": "Format-Wide",
    "oh": "Out-Host", "ogv": "Out-GridView",
    "ac": "Add-Content", "clc": "Clear-Content", "tee": "Tee-Object", "epcsv": "Export-Csv",
    "sp": "Set-ItemProperty", "rp": "Remove-ItemProperty", "cli": "Clear-Item", "epal": "Export-Alias",
    # Text search
    "sls": "Select-String",
}

_DIRECTORY_CHANGE_CMDLETS = frozenset({"set-location", "push-location", "pop-location"})
_DIRECTORY_CHANGE_ALIASES = frozenset({"cd", "sl", "chdir", "pushd", "popd"})


def get_all_command_names(parsed: ParsedPowerShellCommand) -> List[str]:
    """Lowercased names across all statements + nested commands."""
    names: List[str] = []
    for statement in parsed.statements:
        for cmd in statement.commands:
            names.append(cmd.name.lower())
        if statement.nested_commands:
            for cmd in statement.nested_commands:
                names.append(cmd.name.lower())
    return names


def get_all_commands(parsed: ParsedPowerShellCommand) -> List[ParsedCommandElement]:
    """Flat list of every command (statements + nested)."""
    commands: List[ParsedCommandElement] = []
    for statement in parsed.statements:
        commands.extend(statement.commands)
        if statement.nested_commands:
            commands.extend(statement.nested_commands)
    return commands


def get_all_redirections(parsed: ParsedPowerShellCommand) -> List[ParsedRedirection]:
    """All redirections across statements + nested-command redirections."""
    redirections: List[ParsedRedirection] = []
    for statement in parsed.statements:
        redirections.extend(statement.redirections)
        if statement.nested_commands:
            for cmd in statement.nested_commands:
                if cmd.redirections:
                    redirections.extend(cmd.redirections)
    return redirections


def get_variables_by_scope(parsed: ParsedPowerShellCommand, scope: str) -> List[ParsedVariable]:
    """Variables whose path starts with `<scope>:` (e.g. 'env')."""
    prefix = scope.lower() + ":"
    return [v for v in parsed.variables if v.path.lower().startswith(prefix)]


def has_command_named(parsed: ParsedPowerShellCommand, name: str) -> bool:
    """Case-insensitive command presence, resolving aliases both directions."""
    lower_name = name.lower()
    canonical_from_alias = COMMON_ALIASES.get(lower_name)
    canonical_from_alias = canonical_from_alias.lower() if canonical_from_alias else None
    for cmd_name in get_all_command_names(parsed):
        if cmd_name == lower_name:
            return True
        canonical = COMMON_ALIASES.get(cmd_name)
        canonical = canonical.lower() if canonical else None
        if canonical == lower_name:
            return True
        if canonical_from_alias and cmd_name == canonical_from_alias:
            return True
        if canonical and canonical_from_alias and canonical == canonical_from_alias:
            return True
    return False


def has_directory_change(parsed: ParsedPowerShellCommand) -> bool:
    """True if any Set-/Push-/Pop-Location (or cd/sl/chdir/pushd/popd) present."""
    for cmd_name in get_all_command_names(parsed):
        if cmd_name in _DIRECTORY_CHANGE_CMDLETS or cmd_name in _DIRECTORY_CHANGE_ALIASES:
            return True
    return False


def is_single_command(parsed: ParsedPowerShellCommand) -> bool:
    """True if one statement, one command, no nested commands."""
    if len(parsed.statements) != 1:
        return False
    stmt = parsed.statements[0]
    return len(stmt.commands) == 1 and not stmt.nested_commands


def command_has_arg(command: ParsedCommandElement, arg: str) -> bool:
    """Case-insensitive exact arg/flag presence."""
    lower = arg.lower()
    return any(a.lower() == lower for a in command.args)


# Tokenizer-level dash chars PowerShell accepts as parameter prefixes
# (SpecialCharacters.IsDash): ASCII hyphen, en-dash, em-dash, horizontal bar.
PS_TOKENIZER_DASH_CHARS = frozenset({"-", "–", "—", "―"})


def is_powershell_parameter(arg: str, element_type: Optional[str] = None) -> bool:
    """AST element type is authoritative ('Parameter'); else fall back to a
    leading-dash-char check."""
    if element_type is not None:
        return element_type == "Parameter"
    return len(arg) > 0 and arg[0] in PS_TOKENIZER_DASH_CHARS


def command_has_arg_abbreviation(command: ParsedCommandElement, full_param: str, min_prefix: str) -> bool:
    """True if any arg is an unambiguous abbreviation of full_param (≥ min_prefix)."""
    lower_full = full_param.lower()
    lower_min = min_prefix.lower()
    for a in command.args:
        colon = a.find(":", 1)
        param_part = a[:colon] if colon > 0 else a
        lower = param_part.replace("`", "").lower()
        if lower.startswith(lower_min) and lower_full.startswith(lower) and len(lower) <= len(lower_full):
            return True
    return False


def get_pipeline_segments(parsed: ParsedPowerShellCommand) -> List[ParsedStatement]:
    """Statements, each a pipeline segment for per-segment permission checks."""
    return parsed.statements


def is_null_redirection_target(target: str) -> bool:
    """True for `$null`/`${null}` — discards output, not a filesystem write."""
    t = target.strip().lower()
    return t == "$null" or t == "${null}"


def get_file_redirections(parsed: ParsedPowerShellCommand) -> List[ParsedRedirection]:
    """File-writing redirections (excludes merging and $null sinks)."""
    return [r for r in get_all_redirections(parsed) if not r.is_merging and not is_null_redirection_target(r.target)]


@dataclass
class SecurityFlags:
    has_sub_expressions: bool = False
    has_script_blocks: bool = False
    has_splatting: bool = False
    has_expandable_strings: bool = False
    has_member_invocations: bool = False
    has_assignments: bool = False
    has_stop_parsing: bool = False


def derive_security_flags(parsed: ParsedPowerShellCommand) -> SecurityFlags:
    """Security-relevant flags from the parsed AST (deriveSecurityFlags)."""
    flags = SecurityFlags(has_stop_parsing=parsed.has_stop_parsing)

    def check_elements(cmd: ParsedCommandElement) -> None:
        if not cmd.element_types:
            return
        for et in cmd.element_types:
            if et == "ScriptBlock":
                flags.has_script_blocks = True
            elif et == "SubExpression":
                flags.has_sub_expressions = True
            elif et == "ExpandableString":
                flags.has_expandable_strings = True
            elif et == "MemberInvocation":
                flags.has_member_invocations = True

    for stmt in parsed.statements:
        if stmt.statement_type == "AssignmentStatementAst":
            flags.has_assignments = True
        for cmd in stmt.commands:
            check_elements(cmd)
        if stmt.nested_commands:
            for cmd in stmt.nested_commands:
                check_elements(cmd)
        sp = stmt.security_patterns
        if sp:
            if sp.get("hasMemberInvocations"):
                flags.has_member_invocations = True
            if sp.get("hasSubExpressions"):
                flags.has_sub_expressions = True
            if sp.get("hasExpandableStrings"):
                flags.has_expandable_strings = True
            if sp.get("hasScriptBlocks"):
                flags.has_script_blocks = True

    for v in parsed.variables:
        if v.is_splatted:
            flags.has_splatting = True
            break

    return flags
