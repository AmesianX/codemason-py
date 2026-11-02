"""Dangerous shell-tool allow-rule prefixes (adapted from dangerousPatterns.ts).

An allow rule like `Bash(python:*)` / `PowerShell(node:*)` would let the model
run arbitrary code via that interpreter, bypassing an auto-mode classifier.

STATUS — NOT WIRED: in the upstream these lists feed the
isDangerous{Bash,PowerShell}Permission predicates that strip such rules at
auto-mode entry, but codemason has NOT ported the permission-rule engine or
auto/acceptEdits modes (see state/app_state.py). There is therefore no
allow-rule engine here for them to protect, so this module is currently
reference data only — retained verbatim so that if the rule engine is ever
ported, the predicate can be wired to it. Do NOT read this as active protection:
until it's wired, adding such an allow-rule would not be neutralized.
"""

import os
from typing import List

# Cross-platform code-execution entry points present on both Unix and Windows.
# Shared so the two lists don't drift on interpreter additions.
CROSS_PLATFORM_CODE_EXEC: List[str] = [
    # Interpreters
    "python", "python3", "python2", "node", "deno", "tsx", "ruby", "perl", "php", "lua",
    # Package runners
    "npx", "bunx", "npm run", "yarn run", "pnpm run", "bun run",
    # Shells reachable from both (Git Bash / WSL on Windows, native on Unix)
    "bash", "sh",
    # Remote arbitrary-command wrapper (native OpenSSH on Win10+)
    "ssh",
]


def _dangerous_bash_patterns() -> List[str]:
    base = list(CROSS_PLATFORM_CODE_EXEC) + ["zsh", "fish", "eval", "exec", "env", "xargs", "sudo"]
    # Internal (USER_TYPE=ant) additions — empirical over-allowlist risk,
    # not a universal "unsafe" judgment. External users get the base list.
    if os.environ.get("USER_TYPE") == "ant":
        base += ["fa run", "coo", "gh", "gh api", "curl", "wget", "git", "kubectl", "aws", "gcloud", "gsutil"]
    return base


DANGEROUS_BASH_PATTERNS: List[str] = _dangerous_bash_patterns()
