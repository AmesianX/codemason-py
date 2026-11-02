"""Destructive-command warnings for the permission dialog (destructiveCommandWarning.ts).

Purely informational — returns a warning string, does NOT affect permission logic
or auto-approval.
"""

import re
from typing import List, Optional, Tuple

# (pattern, warning). Order matters (first match wins).
_DESTRUCTIVE_PATTERNS: List[Tuple["re.Pattern", str]] = [
    # Remove-Item -Recurse + -Force (either order), anchored to statement start so
    # `git rm --force` doesn't match.
    (re.compile(r"(?:^|[|;&\n({])\s*(Remove-Item|rm|del|rd|rmdir|ri)\b[^|;&\n}]*-Recurse\b[^|;&\n}]*-Force\b", re.I),
     "Note: may recursively force-remove files"),
    (re.compile(r"(?:^|[|;&\n({])\s*(Remove-Item|rm|del|rd|rmdir|ri)\b[^|;&\n}]*-Force\b[^|;&\n}]*-Recurse\b", re.I),
     "Note: may recursively force-remove files"),
    (re.compile(r"(?:^|[|;&\n({])\s*(Remove-Item|rm|del|rd|rmdir|ri)\b[^|;&\n}]*-Recurse\b", re.I),
     "Note: may recursively remove files"),
    (re.compile(r"(?:^|[|;&\n({])\s*(Remove-Item|rm|del|rd|rmdir|ri)\b[^|;&\n}]*-Force\b", re.I),
     "Note: may force-remove files"),
    # Clear-Content on broad paths
    (re.compile(r"\bClear-Content\b[^|;&\n]*\*", re.I), "Note: may clear content of multiple files"),
    # Disk operations
    (re.compile(r"\bFormat-Volume\b", re.I), "Note: may format a disk volume"),
    (re.compile(r"\bClear-Disk\b", re.I), "Note: may clear a disk"),
    # Git destructive operations
    (re.compile(r"\bgit\s+reset\s+--hard\b", re.I), "Note: may discard uncommitted changes"),
    (re.compile(r"\bgit\s+push\b[^|;&\n]*\s+(--force|--force-with-lease|-f)\b", re.I),
     "Note: may overwrite remote history"),
    (re.compile(r"\bgit\s+clean\b(?![^|;&\n]*(?:-[a-zA-Z]*n|--dry-run))[^|;&\n]*-[a-zA-Z]*f", re.I),
     "Note: may permanently delete untracked files"),
    (re.compile(r"\bgit\s+stash\s+(drop|clear)\b", re.I), "Note: may permanently remove stashed changes"),
    # Database operations
    (re.compile(r"\b(DROP|TRUNCATE)\s+(TABLE|DATABASE|SCHEMA)\b", re.I),
     "Note: may drop or truncate database objects"),
    # System operations
    (re.compile(r"\bStop-Computer\b", re.I), "Note: will shut down the computer"),
    (re.compile(r"\bRestart-Computer\b", re.I), "Note: will restart the computer"),
    (re.compile(r"\bClear-RecycleBin\b", re.I), "Note: permanently deletes recycled files"),
]


def get_destructive_command_warning(command: str) -> Optional[str]:
    """Warning string for a known destructive pattern, or None."""
    for pattern, warning in _DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            return warning
    return None
