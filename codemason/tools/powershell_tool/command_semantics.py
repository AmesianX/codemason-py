"""Exit-code semantics for external executables in PowerShell (commandSemantics.ts).

PowerShell-native cmdlets signal failure via terminating errors ($?), not exit
codes, so they need no special handling. But EXTERNAL exes set $LASTEXITCODE and
many use non-zero to convey info, not failure (grep 1=no-match, robocopy 0-7=ok).
Without this, PowerShell tool would report those as errors.
"""

import re
from typing import Callable, Dict, Tuple

# CommandSemantic: (exit_code, stdout, stderr) -> (is_error, message|None)
CommandSemantic = Callable[[int, str, str], Tuple[bool, "str | None"]]


def _default_semantic(exit_code: int, _stdout: str, _stderr: str):
    return (exit_code != 0, f"Command failed with exit code {exit_code}" if exit_code != 0 else None)


def _grep_semantic(exit_code: int, _stdout: str, _stderr: str):
    # 0 = matches, 1 = no matches, 2+ = error
    return (exit_code >= 2, "No matches found" if exit_code == 1 else None)


def _robocopy_semantic(exit_code: int, _stdout: str, _stderr: str):
    # Bitfield: 0-7 success, 8+ failure. bit0 = files copied.
    if exit_code == 0:
        msg = "No files copied (already in sync)"
    elif 1 <= exit_code < 8:
        msg = "Files copied successfully" if (exit_code & 1) else "Robocopy completed (no errors)"
    else:
        msg = None
    return (exit_code >= 8, msg)


# Keys: lowercase command names WITHOUT .exe. diff/fc/find/test omitted (ambiguous
# native-vs-cmdlet aliasing); native cmdlets (select-string etc.) exit 0.
_COMMAND_SEMANTICS: Dict[str, CommandSemantic] = {
    "grep": _grep_semantic,
    "rg": _grep_semantic,
    "findstr": _grep_semantic,
    "robocopy": _robocopy_semantic,
}

_CALL_OP_RE = re.compile(r"^[&.]\s+")
_QUOTE_RE = re.compile(r"^[\"']|[\"']$")
_EXE_RE = re.compile(r"\.exe$")


def _extract_base_command(segment: str) -> str:
    """Command name from a pipeline segment: strip call op / quotes / path / .exe."""
    stripped = _CALL_OP_RE.sub("", segment.strip())
    first_token = stripped.split()[0] if stripped.split() else ""
    unquoted = _QUOTE_RE.sub("", first_token)
    basename = re.split(r"[\\/]", unquoted)[-1] or unquoted
    return _EXE_RE.sub("", basename.lower())


def _heuristically_extract_base_command(command: str) -> str:
    """Base command of the LAST pipeline segment (it sets the exit code). Heuristic
    split on ; and | — do NOT use for security."""
    segments = [s for s in re.split(r"[;|]", command) if s.strip()]
    last = segments[-1] if segments else command
    return _extract_base_command(last)


def interpret_command_result(command: str, exit_code: int, stdout: str, stderr: str):
    """Interpret a command result → (is_error, message|None) per semantic rules."""
    base = _heuristically_extract_base_command(command)
    semantic = _COMMAND_SEMANTICS.get(base, _default_semantic)
    return semantic(exit_code, stdout, stderr)
