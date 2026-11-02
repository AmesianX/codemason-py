"""Windows git-bash substrate for CodeMason (Tier 1).

Based on Claude Code src/utils/windowsPaths.ts. The discovery logic follows the
UPDATED v2.1.208 form (`nve`/`CHn`), which hardened three things over the
earlier form:

  1. CODEMASON_GIT_BASH_PATH is checked with a filesystem stat, not by
     shelling out to `dir` (no shell-parse surface).
  2. `where.exe` is invoked by absolute %SYSTEMROOT%\\System32\\where.exe, so a
     malicious where.exe on the cwd/PATH can't hijack discovery.
  3. `C:\\Program Files\\Git\\bin\\bash.exe` (and the x86 path) are checked
     directly, before the git-derived fallback.

Tier 1 = "there is a POSIX shell on this host": genuine Linux/macOS/WSL2, or
native Windows WITH git-bash/MSYS2. The command bodies our substrate builds are
POSIX sh (`pwd -P`, `$?`, `/dev/null`), so on Windows we route them through
bash.exe unchanged rather than cmd.exe. Native cmd/PowerShell is Tier 2.

Everything here is import-safe on all platforms (no os-specific top-level
imports); POSIX branches never touch Windows code paths and vice-versa.
"""

import ntpath
import os
import subprocess
from typing import List, Optional

from .platform import is_windows

# where.exe timeout (windowsPaths.ts Xwm = 5000ms).
_WHERE_TIMEOUT_S = 5.0

# Executable extensions worth trusting from a where.exe hit (windowsPaths.ts Jwm).
_EXE_EXTS = frozenset({".com", ".exe", ".bat", ".cmd"})

# Known git-bash install locations (nve). 64-bit before 32-bit.
_KNOWN_BASH_PATHS = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
)

_git_bash_cache: Optional[str] = None
_git_bash_computed = False


def _has_exe_ext(path: str) -> bool:
    """Path ends in a Windows executable extension (Qwm in windowsPaths.ts)."""
    base = path.lower().replace("\\", "/").rsplit("/", 1)[-1].rstrip(". ")
    dot = base.rfind(".")
    return dot > 0 and base[dot:] in _EXE_EXTS


def _is_in_dir(candidate: str, directory: str) -> bool:
    """True if `candidate` resolves inside `directory` (ISn cwd-hijack filter)."""
    try:
        cand = os.path.normcase(os.path.realpath(candidate))
        base = os.path.normcase(os.path.realpath(directory))
    except OSError:
        return False
    return cand == base or cand.startswith(base + os.sep)


def _find_git_via_where() -> Optional[str]:
    """Locate git.exe with absolute System32\\where.exe, skipping cwd hits (CHn).

    Filters out any hit inside the current directory (anti-hijack) and requires a
    real executable extension, mirroring the v2.1.208 discovery.
    """
    where_exe = _system32_exe("where.exe")
    try:
        out = subprocess.run(
            [where_exe, "git"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_WHERE_TIMEOUT_S,
            encoding="utf-8",
            errors="replace",  # where.exe emits OEM codepage; never let decode raise
            env=os.environ,
            creationflags=_no_window_flag(),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    cwd = os.getcwd()
    for line in (out.stdout or "").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if not os.path.exists(candidate):
            continue
        if _is_in_dir(candidate, cwd):  # SECURITY: never a cwd-planted git
            continue
        if not _has_exe_ext(candidate):
            continue
        return candidate
    return None


def _no_window_flag() -> int:
    """CREATE_NO_WINDOW on Windows (windowsHide: true), else 0."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if is_windows() else 0


def find_git_bash_path() -> Optional[str]:
    """Absolute path to git-bash's bash.exe, or None if not found (nve).

    Order: CODEMASON_GIT_BASH_PATH → known install paths → git-derived path.
    Memoized. Unlike the source we return None instead of process.exit() when
    the override points nowhere — the caller decides how to surface it (see
    git_bash_or_raise); a library must not kill the host process.
    """
    global _git_bash_cache, _git_bash_computed
    if _git_bash_computed:
        return _git_bash_cache
    _git_bash_computed = True

    env_path = os.environ.get("CODEMASON_GIT_BASH_PATH")
    if env_path:
        _git_bash_cache = env_path if os.path.exists(env_path) else None
        return _git_bash_cache

    for known in _KNOWN_BASH_PATHS:
        if os.path.exists(known):
            _git_bash_cache = known
            return known

    git_path = _find_git_via_where()
    if git_path:
        # <git>/cmd/git.exe or <git>/bin/git.exe → <git>/bin/bash.exe
        derived = os.path.join(os.path.dirname(os.path.dirname(git_path)), "bin", "bash.exe")
        if os.path.exists(derived):
            _git_bash_cache = derived
            return derived

    _git_bash_cache = None
    return None


_GIT_BASH_MISSING_MSG = (
    "CodeMason on native Windows requires git-bash (https://git-scm.com/downloads/win) "
    "for the Bash tool. If it is installed but not discovered, point "
    "CODEMASON_GIT_BASH_PATH at your bash.exe, e.g.\n"
    r"  set CODEMASON_GIT_BASH_PATH=C:\Program Files\Git\bin\bash.exe"
)


def git_bash_or_raise() -> str:
    """Return bash.exe path or raise a clear, actionable error (Windows only)."""
    path = find_git_bash_path()
    if path is None:
        raise RuntimeError(_GIT_BASH_MISSING_MSG)
    return path


# === POSIX <-> Windows path conversion (windowsPaths.ts) ===

def windows_path_to_posix(win_path: str) -> str:
    r"""C:\Users\foo -> /c/Users/foo ; \\server\share -> //server/share."""
    if win_path.startswith("\\\\"):
        return win_path.replace("\\", "/")
    if len(win_path) >= 3 and win_path[1] == ":" and win_path[2] in "/\\" and win_path[0].isalpha():
        return "/" + win_path[0].lower() + win_path[2:].replace("\\", "/")
    return win_path.replace("\\", "/")


def posix_path_to_windows(posix_path: str) -> str:
    r"""/c/Users/foo -> C:\Users\foo ; //server/share -> \\server\share (MSYS/cygdrive)."""
    if posix_path.startswith("//"):
        return posix_path.replace("/", "\\")
    if posix_path.startswith("/cygdrive/") and len(posix_path) >= 11 and posix_path[10].isalpha():
        drive = posix_path[10].upper()
        rest = posix_path[len("/cygdrive/" + posix_path[10]):]
        return drive + ":" + (rest or "\\").replace("/", "\\")
    if len(posix_path) >= 2 and posix_path[0] == "/" and posix_path[1].isalpha() and (
        len(posix_path) == 2 or posix_path[2] == "/"
    ):
        drive = posix_path[1].upper()
        rest = posix_path[2:]
        return drive + ":" + (rest or "\\").replace("/", "\\")
    return posix_path.replace("/", "\\")


# === binShell resolution: cmd / powershell / pwsh / bash → (exe, args) ===
# Port of the source's binShell resolver (e1i). Maps a shell token (bare name or
# absolute path) to the executable and its leading args, so Tier-1 (bash) and
# Tier-2 (cmd/PowerShell) share one shell-selection seam.

# PowerShell invocation args. The interactive form is -NoProfile -Command; the
# actual tool run uses EncodedCommand (see powershell_tool). fUc in source.
_PWSH_ARGS = ["-NoProfile", "-Command"]


class BinShell:
    """Resolved shell: `exe` (absolute or bare) + `args` (leading args before the
    command). Mirrors the source's {exe, args} object."""

    __slots__ = ("exe", "args")

    def __init__(self, exe: str, args: List[str]):
        self.exe = exe
        self.args = args

    def __repr__(self) -> str:
        return f"BinShell(exe={self.exe!r}, args={self.args!r})"


def _system_root() -> str:
    return os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or r"C:\Windows"


def _default_cmd() -> "BinShell":
    return BinShell(ntpath.join(_system_root(), "System32", "cmd.exe"), ["/d", "/s", "/c"])


def resolve_bin_shell(spec=None) -> "BinShell":
    r"""Resolve a shell spec to a BinShell (e1i port). Windows-oriented.

    spec: None → cmd.exe (default); a BinShell → validated passthrough; or a
    token/absolute path — 'bash'/'sh'(.exe) (must be absolute → args ['-c']),
    'pwsh'(.exe) → pwsh.exe -NoProfile -Command, 'powershell'(.exe) → 5.1 path,
    'cmd'(.exe) → cmd.exe /d /s /c.
    """
    # ntpath (== source's path/win32): correct on any host, not just Windows.
    if spec is None:
        return _default_cmd()
    if isinstance(spec, BinShell):
        if not ntpath.isabs(spec.exe):
            raise ValueError(f"binShell.exe must be an absolute path (got {spec.exe!r})")
        if not isinstance(spec.args, list):
            raise ValueError(f"binShell.args must be a list (got {spec.args!r})")
        return spec

    base = ntpath.basename(spec)
    lower = base.lower()
    is_abs = ntpath.isabs(spec)
    if not is_abs and spec != base:
        raise ValueError(f"binShell string must be a bare token or an absolute path (got {spec!r})")

    if lower in ("bash", "bash.exe", "sh", "sh.exe"):
        if not is_abs:
            raise ValueError(
                f"binShell bash path must be absolute (got {spec!r}); pass the resolved Git Bash install path"
            )
        return BinShell(spec, ["-c"])
    if lower in ("pwsh", "pwsh.exe"):
        # A bare token resolved to relative "pwsh.exe" → PATH/CWD lookup, the same
        # hijack class hardened elsewhere (a planted pwsh.exe in CWD would run).
        # Prefer the absolute standard install path when it exists; fall back to
        # the PATH name only for a non-standard install (preserves old behavior).
        if is_abs:
            return BinShell(spec, list(_PWSH_ARGS))
        _pf = ntpath.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"), "PowerShell", "7", "pwsh.exe"
        )
        return BinShell(_pf if os.path.isfile(_pf) else "pwsh.exe", list(_PWSH_ARGS))
    if lower in ("powershell", "powershell.exe"):
        default_ps = ntpath.join(
            _system_root(), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
        )
        return BinShell(spec if is_abs else default_ps, list(_PWSH_ARGS))
    if lower in ("cmd", "cmd.exe"):
        return _default_cmd() if not is_abs else BinShell(spec, ["/d", "/s", "/c"])
    raise ValueError(
        f"unrecognised binShell {spec!r}: expected 'cmd' | 'powershell' | 'pwsh' "
        "or an absolute path to bash.exe/sh.exe/pwsh.exe/powershell.exe"
    )


# === Spawn helpers shared by both substrate sites ===

# CREATE_NEW_PROCESS_GROUP: Windows analogue of start_new_session (own group so
# the whole tree can be signalled/killed).
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
# CREATE_NO_WINDOW: don't allocate a console for the child. Our shell children are
# fully redirected (stdin/stdout/stderr → pipes/DEVNULL), so they never need a
# console; without it a console-subsystem child (bash.exe, or a hook's python.exe)
# flashes a window on every tool use when the host has no console — print mode
# (`-p`) piped, headless, or spawned by another process (upstream #61051).
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def shell_argv(script: str) -> Optional[List[str]]:
    """argv to run a POSIX-sh `script` via create_subprocess_exec, or None.

    On Windows returns ['<bash.exe>', '-c', script] so the POSIX command body
    runs under git-bash instead of cmd.exe. On POSIX returns None — callers use
    create_subprocess_shell(script) unchanged.
    """
    if is_windows():
        return [git_bash_or_raise(), "-c", script]
    return None


def spawn_group_kwargs() -> dict:
    """Platform kwargs giving a fully-redirected child its own process group.

    POSIX: start_new_session=True. Windows: CREATE_NEW_PROCESS_GROUP (own group
    for tree-kill) | CREATE_NO_WINDOW (no console flash — the child is piped).
    """
    if is_windows():
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW}
    return {"start_new_session": True}


def no_window_kwargs() -> dict:
    """Windows-only kwargs to suppress the child's console; empty on POSIX.

    For spawns that must NOT change their POSIX process-group behaviour (hooks),
    but should still avoid the console flash on Windows (upstream #61051).
    """
    if is_windows():
        return {"creationflags": _CREATE_NO_WINDOW}
    return {}


def _system32_exe(name: str) -> str:
    """Absolute path to a Windows system executable under System32.

    CreateProcess searches the current directory before System32, so invoking a
    system tool (taskkill.exe, where.exe, ...) by bare name lets a malicious
    <name>.exe planted in a checked-out repo's directory hijack the call. Always
    resolve system binaries by absolute path. Mirrors the where.exe hardening.
    """
    system_root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or r"C:\Windows"
    return os.path.join(system_root, "System32", name)


def ensure_utf8_locale() -> None:
    """Default LC_CTYPE=C.UTF-8 for child processes on native Windows.

    git-bash/MSYS2 and the Unix coreutils it ships default to the C locale when
    no locale is set — the usual case when CodeMason is launched from cmd.exe or
    PowerShell. In the C locale those tools treat non-ASCII path bytes as opaque
    and mangle them in output: a Korean directory prints as ``$'\\215'``-escaped
    bytes, so ``ls``/``rm`` errors on a Korean path come back unreadable. Setting
    LC_CTYPE=C.UTF-8 makes the charset UTF-8 while every other locale category
    stays C (English messages, ASCII collation), so parsed tool output is
    unchanged and Unicode paths round-trip.

    Only LC_CTYPE, only when the user set no locale of their own (LANG/LC_ALL/
    LC_CTYPE all unset), and only on native Windows. POSIX inherits the real
    system locale untouched, so this is a no-op there.
    """
    if not is_windows():
        return
    if not any(os.environ.get(k) for k in ("LANG", "LC_ALL", "LC_CTYPE")):
        os.environ["LC_CTYPE"] = "C.UTF-8"


def decode_console_output(buf) -> str:
    """Decode captured child stdout/stderr to text.

    git-bash builtins and Unix tools emit UTF-8, but native Windows console
    programs (taskkill, tasklist, ipconfig, ...) emit the OEM codepage
    (cp949/cp932 on CJK Windows) — decoding those bytes as UTF-8 produces mojibake
    (``????``). Try strict UTF-8 first (covers the common case AND all POSIX
    output), then on Windows fall back to the OEM codepage before finally
    replacing undecodable bytes. On POSIX the fallback is skipped, so the result
    is byte-for-byte identical to a plain ``utf-8``/``replace`` decode.
    """
    if not isinstance(buf, (bytes, bytearray)):
        return buf
    data = bytes(buf)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        if is_windows():
            # Run by run, not the whole buffer. A Windows program and cmd.exe
            # itself write into ONE stream, the first often in UTF-8 and the
            # second always in the console codepage, so decoding everything as
            # the OEM page mangles the UTF-8 half exactly as decoding everything
            # as UTF-8 mangled the legacy half. Native Windows had this
            # backwards until 2026-10-11: this branch returned, so the
            # run-by-run path below was reached only OFF Windows -- under WSL,
            # where the OEM page cannot be asked for -- and here pktmon's UTF-8
            # Korean came back as mojibake while cmd.exe's CP949 came through.
            #
            # The OEM page is knowable here, so it is used in preference to the
            # configured grep_legacy_encoding: it is what the console actually
            # wrote, and it needs no user setting to be right.
            try:
                ''.encode('oem')  # LookupError if the Windows-only codec is absent
            except LookupError:
                pass
            else:
                return _decode_runs(data, 'oem')
        return _decode_mixed_console_output(data)


def _decode_mixed_console_output(data: bytes) -> str:
    """Decode a console stream that is UTF-8 in places and a legacy CJK
    codepage in others, run by run.

    This is what a Windows program invoked from WSL produces, and the two halves
    arrive interleaved in ONE command's output. Reported 2026-09-07 running
    pktmon through cmd.exe from Ubuntu-24.04: pktmon writes UTF-8 and its Korean
    came through intact, while cmd.exe's own messages are in the console
    codepage (949 on a Korean install) and arrived as
    "\ufffd\ufffd\ufffd\ufffd \ufffd\ufffd\u03b8\ufffd".

    The old code could not do better anywhere but on Windows itself: the OEM
    branch above is gated on is_windows(), and codemason under WSL runs on
    LINUX, so every non-UTF-8 byte became U+FFFD.

    Guessing is still refused. A run that is not valid UTF-8 is decoded with the
    configured legacy encoding ONLY if the result is plausible text — no control
    characters, nothing replaced. Otherwise the bytes were binary, not
    misencoded text, and they keep the lossy form they had, so `cat` of a JPEG
    still looks like a JPEG rather than like hangul.
    """
    try:
        from .config import get_config
        label = (get_config().grep_legacy_encoding or '').strip()
    except Exception:
        label = ''
    if not label:
        return data.decode('utf-8', errors='replace')
    try:
        ''.encode(label)  # LookupError if the codec is unknown
    except LookupError:
        return data.decode('utf-8', errors='replace')
    return _decode_runs(data, label)


def _decode_runs(data: bytes, label: str) -> str:
    """The run-by-run decode itself, over a legacy encoding the caller has
    already settled on: the OEM codepage on Windows, the configured one
    everywhere else. `label` must name a codec that exists."""
    out = []
    rest = data
    while rest:
        try:
            out.append(rest.decode('utf-8'))
            break
        except UnicodeDecodeError as e:
            out.append(rest[:e.start].decode('utf-8'))
            # The invalid run reaches to the next byte whose tail parses;
            # decoding the whole run at once is what lets a multi-byte legacy
            # character survive.
            end = e.start + 1
            while end < len(rest):
                try:
                    rest[end:].decode('utf-8')
                    break
                except UnicodeDecodeError:
                    end += 1
            run = rest[e.start:end]
            try:
                decoded = run.decode(label)
                plausible = bool(decoded) and not any(
                    ch == '\ufffd' or (ch.isprintable() is False and ch not in '\n\r\t')
                    for ch in decoded)
            except UnicodeDecodeError:
                decoded, plausible = '', False
            out.append(decoded if plausible else run.decode('utf-8', errors='replace'))
            rest = rest[end:]
    return ''.join(out)


def kill_process_tree(proc, sig=None) -> None:
    """Kill `proc` and its descendants (POSIX killpg equivalent).

    POSIX: os.killpg(getpgid(pid), sig). Windows: taskkill /F /T (tree kill), then
    proc.kill() as a fallback. Swallows the races the callers already tolerate
    (already-exited / no-permission).
    """
    if is_windows():
        try:
            subprocess.run(
                [_system32_exe("taskkill.exe"), "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                creationflags=_no_window_flag(),
            )
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        return

    import signal as _signal

    if sig is None:
        sig = _signal.SIGKILL
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        # Broad fallback: killpg was denied/failed and proc.kill() may raise
        # PermissionError etc. The original inline kill sites swallowed everything
        # here, and killing is best-effort — never let cleanup crash the caller.
        try:
            proc.kill()
        except Exception:
            pass


def reset_cache() -> None:
    """Clear memoized git-bash discovery (tests only)."""
    global _git_bash_cache, _git_bash_computed
    _git_bash_cache = None
    _git_bash_computed = False
