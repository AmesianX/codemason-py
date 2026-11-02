"""Pager binary resolution, on the shape utils/ripgrep.py uses.

    system `less` on PATH  →  git-bash's own less  →  the vendored binary  →  None

Ctrl+O writes every recent tool output to a temp file and hands it to a pager.
That assumed `less`, which is fine on Linux and macOS and false on Windows:
`less` is not a Windows program, and whether it resolves depends entirely on
which shell started codemason. Measured on a stock machine — from Git Bash
`less` is found (it ships in Git's usr/bin), from PowerShell or cmd.exe it is
not on PATH at all. The same keystroke therefore either paged or dumped every
stored tool output straight into the scrollback, and the dump still carried the
document's first line telling the reader to press `q` — with no pager running to
receive it.

The fix is to USE the less that is already there rather than ship a second copy
of it. On Windows the Bash tool requires git-bash, and Git for Windows puts
less.exe in its usr/bin, so any machine that can run the Bash tool at all
already has this pager — the step below just looks where we already found bash.
Falling back to `more.exe` was the other option and is worse: `more` has no `/`
search and no `g/G`, so the key hints would stop being true and the hint string
would need a Windows variant instead of staying byte-identical.

The vendor step is kept for a platform where neither holds. No binary is
vendored today, so it is a miss that costs nothing.
"""

import os
import shutil
from functools import lru_cache
from typing import Optional

from .ripgrep import _vendor_subdir


def _derive_less_path(bash: str) -> Optional[str]:
    """`<git>/usr/bin/less[.exe]` for a `<git>/bin/bash.exe`, if it exists.

    Split out from the lookup so the derivation is testable off Windows: the
    Windows-only half is FINDING bash, not climbing from it."""
    git_root = os.path.dirname(os.path.dirname(bash))
    if not git_root or git_root == bash:
        return None
    exe = 'less.exe' if os.name == 'nt' else 'less'
    cand = os.path.join(git_root, 'usr', 'bin', exe)
    return cand if os.path.isfile(cand) else None


def _less_from_git_bash() -> Optional[str]:
    """The pager git-bash already brought, if git-bash is where we expect it.

    find_git_bash_path resolves three ways (CODEMASON_GIT_BASH_PATH, the known
    install paths, a path derived from git.exe) and all three end at
    `...\\bin\\bash.exe`, so the two-level climb holds for each. The override is
    the one that can point anywhere, which is why this is taken only when the
    derived file actually exists and otherwise falls through."""
    try:
        from .windows_shell import find_git_bash_path
        bash = find_git_bash_path()
    except Exception:
        return None
    return _derive_less_path(bash) if bash else None


@lru_cache(maxsize=1)
def resolve_pager() -> Optional[str]:
    """Path to a `less` executable, or None if none is available.

    1. a system `less` on PATH wins, returned as the bare name so the OS
       resolves it — the same PATH-hijacking posture as ripgrep resolution.
       Someone who installed their own less gets theirs;
    2. else the less git-bash already installed (Windows; see module docstring);
    3. else the bundled vendored binary for this platform;
    4. else None -> the caller prints the document instead of paging it."""
    if shutil.which('less'):
        return 'less'

    from_git = _less_from_git_bash()
    if from_git:
        return from_git

    sub = _vendor_subdir()
    if sub:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # codemason/
        exe = 'less.exe' if sub.endswith('-win32') else 'less'
        cand = os.path.join(pkg_dir, 'vendor', 'less', sub, exe)
        if os.path.isfile(cand):
            return cand

    return None
