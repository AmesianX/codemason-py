"""ripgrep binary resolution.

The Glob/Grep tools shell out to `rg`. The original ships a per-platform binary
under vendor/ripgrep/<arch-os>/rg[.exe] and resolves it as:

    system `rg` on PATH  →  else the bundled vendored binary  →  else None

Resolving system-first (by the bare name 'rg', not an absolute path) matches the
original's PATH-hijacking guard; the vendored binary is the fallback so a machine
without ripgrep installed (e.g. a stock Windows desktop) still gets the fast path
instead of collapsing to the slow os.walk fallback.
"""

import os
import platform
import shutil
from functools import lru_cache
from typing import Optional


def _vendor_subdir() -> Optional[str]:
    """Map the current platform to a vendor/ripgrep/<arch-os> folder name.
    Returns None on an unmapped platform (=> no vendored fallback)."""
    machine = platform.machine().lower()
    if machine in ('x86_64', 'amd64', 'x64'):
        arch = 'x64'
    elif machine in ('arm64', 'aarch64'):
        arch = 'arm64'
    else:
        return None

    system = platform.system()
    if system == 'Windows':
        return f'{arch}-win32'
    if system == 'Linux':
        return f'{arch}-linux'
    if system == 'Darwin':
        return f'{arch}-darwin'
    return None


@lru_cache(maxsize=1)
def resolve_ripgrep() -> Optional[str]:
    """Path to an rg executable, or None if none is available.

    1. system `rg` on PATH (returned as the bare name 'rg' so the OS resolves it
       safely, matching Claude Code's PATH-hijacking guard);
    2. else the bundled vendored binary for this platform;
    3. else None -> caller falls back to its stdlib path (os.walk / re)."""
    if shutil.which('rg'):
        return 'rg'

    sub = _vendor_subdir()
    if sub:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # codemason/
        exe = 'rg.exe' if sub.endswith('-win32') else 'rg'
        cand = os.path.join(pkg_dir, 'vendor', 'ripgrep', sub, exe)
        if os.path.isfile(cand):
            return cand

    return None
