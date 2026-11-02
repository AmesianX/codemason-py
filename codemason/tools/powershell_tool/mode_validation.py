"""PowerShell mode validation (modeValidation.ts) — adapted.

The source's checkPermissionMode auto-allows filesystem-modifying cmdlets in
acceptEdits mode. CodeMason has NO acceptEdits mode (its permission model is
simplified — see state/app_state.py; the Bash tool has no mode-validation port
either), so checkPermissionMode would always passthrough and is intentionally
omitted. is_symlink_creating_command is a standalone security helper worth
keeping — links redirect path resolution at runtime, a TOCTOU vector.
"""

import re

from ...utils.powershell.parser import PS_TOKENIZER_DASH_CHARS
from .read_only_validation import resolve_to_canonical

# New-Item -ItemType values that create filesystem links (redirect path resolution).
_LINK_ITEM_TYPES = frozenset({"symboliclink", "junction", "hardlink"})
_QUOTE_RE = re.compile(r"^['\"]|['\"]\Z")


def _is_item_type_param_abbrev(p: str) -> bool:
    """Unambiguous abbrev of New-Item -ItemType / -Type (min `-it` / `-ty`)."""
    return (len(p) >= 3 and "-itemtype".startswith(p)) or (len(p) >= 3 and "-type".startswith(p))


def is_symlink_creating_command(cmd) -> bool:
    """True if `cmd` is New-Item creating a SymbolicLink/Junction/HardLink.

    `cmd` needs `.name` and `.args`. Handles PS abbreviation (-it.. / -ty..),
    unicode-dash / `/` prefixes, colon-bound values, backtick escapes, quotes.
    """
    canonical = resolve_to_canonical(cmd.name)
    if canonical != "new-item":
        return False
    args = cmd.args
    for i in range(len(args)):
        raw = args[i] or ""
        if len(raw) == 0:
            continue
        normalized = ("-" + raw[1:]) if (raw[0] in PS_TOKENIZER_DASH_CHARS or raw[0] == "/") else raw
        lower = normalized.lower()
        colon_idx = lower.find(":", 1)
        param_raw = lower[:colon_idx] if colon_idx > 0 else lower
        param = param_raw.replace("`", "")
        if not _is_item_type_param_abbrev(param):
            continue
        if colon_idx > 0:
            raw_val = lower[colon_idx + 1:]
        else:
            raw_val = (args[i + 1].lower() if i + 1 < len(args) and args[i + 1] else "")
        val = _QUOTE_RE.sub("", raw_val.replace("`", ""))
        if val in _LINK_ITEM_TYPES:
            return True
    return False
