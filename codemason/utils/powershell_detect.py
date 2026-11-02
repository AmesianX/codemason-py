"""PowerShell / cmd detection for CodeMason (Tier 2, native Windows shells).

Based on Claude Code src/utils/shell/powershellDetection.ts. Finds a PowerShell
executable (prefers pwsh 7+, falls back to Windows PowerShell 5.1) and infers its
edition so the tool prompt can give version-correct syntax guidance.

Edition matters: PowerShell 7 (`pwsh`) supports `&&`/`||`/`?:`/`??`; Windows
PowerShell 5.1 (`powershell`) does not (using `&&` there is a parse error) and
has the stderr-sets-$? quirk plus UTF-16 default output encoding.
"""

import os
import shutil
from typing import Optional

# 'core' (pwsh 7+) | 'desktop' (Windows PowerShell 5.1)
PowerShellEdition = str

_cached_ps_path: Optional[str] = None
_ps_path_computed = False


def _probe_file(p: str) -> Optional[str]:
    """Return p if it is an existing regular file, else None (probePath)."""
    try:
        return p if os.path.isfile(p) else None
    except OSError:
        return None


def _is_in_cwd(path: str) -> bool:
    """True if `path` lives in (or resolves into) the current working directory.

    Used to reject a planted pwsh/powershell binary. Checks BOTH the lexical
    location (catches a symlink placed in CWD that points elsewhere) and the
    realpath (catches a junction/link that resolves back into CWD)."""
    try:
        cwd_abs = os.path.abspath(os.getcwd())
        cwd_real = os.path.realpath(cwd_abs)
    except OSError:
        return False
    for base in (cwd_abs, cwd_real):
        for cand in (os.path.abspath(path), os.path.realpath(path)):
            if cand == base or cand.startswith(base + os.sep):
                return True
    return False


def _find_powershell_windows() -> Optional[str]:
    """Windows PowerShell resolution, hardened against a CWD hijack.

    CPython's shutil.which() PREPENDS os.curdir to the PATH search on win32, so a
    pwsh.exe/powershell.exe (or a PATHEXT variant like pwsh.cmd) planted in the
    repository the agent operates in would be resolved FIRST and then spawned with
    -EncodedCommand on every PowerShell command validation — arbitrary code exec
    before any read-only check runs. So: prefer the absolute, well-known install
    locations (unhijackable), and reject any PATH hit that lives inside CWD.
    Mirrors the absolute-path hardening already in utils/windows_shell.py."""
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    for abs_path in (
        os.path.join(program_files, "PowerShell", "7", "pwsh.exe"),
        os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
    ):
        hit = _probe_file(abs_path)
        if hit:
            return hit
    # Fall back to PATH, but never accept a binary resolved out of CWD.
    for name in ("pwsh", "powershell"):
        hit = shutil.which(name)
        if hit and not _is_in_cwd(hit):
            return hit
    return None


def find_powershell() -> Optional[str]:
    """Locate PowerShell: pwsh (7+) preferred, then powershell (5.1) (findPowerShell).

    On Windows, resolution goes through the CWD-hijack-hardened path above. On
    Linux a snap launcher (/snap/...) can hang in subprocesses while snapd
    initialises confinement, so if PATH resolves there — directly or via a
    symlink chain — probe the reliable apt/rpm binary instead. (shutil.which does
    NOT prepend curdir off win32, so POSIX PATH lookup is not CWD-hijackable.)
    """
    import sys

    if sys.platform.startswith("win"):
        return _find_powershell_windows()

    pwsh = shutil.which("pwsh")
    if pwsh:
        if sys.platform.startswith("linux"):
            try:
                resolved = os.path.realpath(pwsh)
            except OSError:
                resolved = pwsh
            if pwsh.startswith("/snap/") or resolved.startswith("/snap/"):
                direct = _probe_file("/opt/microsoft/powershell/7/pwsh") or _probe_file("/usr/bin/pwsh")
                if direct:
                    try:
                        direct_resolved = os.path.realpath(direct)
                    except OSError:
                        direct_resolved = direct
                    if not direct.startswith("/snap/") and not direct_resolved.startswith("/snap/"):
                        return direct
        return pwsh

    powershell = shutil.which("powershell")
    if powershell:
        return powershell

    return None


def get_cached_powershell_path() -> Optional[str]:
    """Memoized find_powershell() (getCachedPowerShellPath)."""
    global _cached_ps_path, _ps_path_computed
    if not _ps_path_computed:
        _ps_path_computed = True
        _cached_ps_path = find_powershell()
    return _cached_ps_path


def get_powershell_edition() -> Optional[PowerShellEdition]:
    """Infer edition from the binary name without spawning (getPowerShellEdition).

    pwsh/pwsh.exe → 'core' (7+); powershell/powershell.exe → 'desktop' (5.1).
    None if no PowerShell found.
    """
    p = get_cached_powershell_path()
    if not p:
        return None
    base = p.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if base.endswith(".exe"):
        base = base[:-4]
    return "core" if base == "pwsh" else "desktop"


def reset_powershell_cache() -> None:
    """Clear the memoized path (tests only)."""
    global _cached_ps_path, _ps_path_computed
    _cached_ps_path = None
    _ps_path_computed = False
