"""Platform detection for CodeMason.

Based on Claude Code src/utils/platform.ts (getPlatform / getWslVersion).

Tier-1 Windows support ("git-bash / MSYS2 / WSL2"): this module only classifies
the host. WSL2 is a real Linux kernel, so `sys.platform == 'linux'` and the shell
substrate already works there unchanged — we detect it purely to label the
platform. Native Windows is where the substrate diverges (see windows_shell.py).
"""

import os
import sys
from typing import Optional

# 'macos' | 'windows' | 'wsl' | 'linux' | 'unknown' (Platform in platform.ts)
_platform_cache: Optional[str] = None
_wsl_version_cache: Optional[str] = None
_wsl_version_computed = False


def _read_proc_version() -> str:
    """Read /proc/version (empty string if unavailable). WSL markers live here."""
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def get_platform() -> str:
    """Classify the host (memoized). Mirrors getPlatform() in platform.ts.

    Returns one of 'macos', 'windows', 'wsl', 'linux', 'unknown'. WSL is Linux
    with a 'microsoft'/'wsl' marker in /proc/version.
    """
    global _platform_cache
    if _platform_cache is not None:
        return _platform_cache

    p = sys.platform
    if p == "darwin":
        _platform_cache = "macos"
    elif p == "win32":
        _platform_cache = "windows"
    elif p.startswith("linux"):
        pv = _read_proc_version().lower()
        _platform_cache = "wsl" if ("microsoft" in pv or "wsl" in pv) else "linux"
    else:
        _platform_cache = "unknown"
    return _platform_cache


def is_windows() -> bool:
    """True only on native Windows (not WSL). WSL is 'wsl', handled as POSIX."""
    return get_platform() == "windows"


def get_wsl_version() -> Optional[str]:
    """WSL major version ('1'/'2'/...) or None (getWslVersion() in platform.ts)."""
    global _wsl_version_cache, _wsl_version_computed
    if _wsl_version_computed:
        return _wsl_version_cache
    _wsl_version_computed = True

    if not sys.platform.startswith("linux"):
        _wsl_version_cache = None
        return None

    pv = _read_proc_version()
    import re

    m = re.search(r"WSL(\d+)", pv, re.IGNORECASE)
    if m:
        _wsl_version_cache = m.group(1)
    elif "microsoft" in pv.lower():
        # Original WSL1 format ("4.4.0-19041-Microsoft") carries no explicit version.
        _wsl_version_cache = "1"
    else:
        _wsl_version_cache = None
    return _wsl_version_cache


def reset_cache() -> None:
    """Clear memoized results (tests only)."""
    global _platform_cache, _wsl_version_cache, _wsl_version_computed
    _platform_cache = None
    _wsl_version_cache = None
    _wsl_version_computed = False
def chmod_posix(path, mode: int) -> None:
    """chmod, where a mode means something.

    The 0600/0700 calls in this program exist so that request bodies, tool
    results and transcripts are not readable by other users of a shared POSIX
    host. Windows does not express that as a mode. NTFS carries ACLs, and the
    CRT folds a whole mode onto one bit -- the read-only attribute -- so the
    call there either does nothing at all (any mode with a write bit, which is
    all of ours) or marks the file read-only and breaks the next write to it.
    Neither outcome is protection.

    What keeps these paths private on Windows is the ACL they inherit from the
    user profile directory, and that is where it stops: THIS PROGRAM DOES NOT
    SET ACLs ON WINDOWS, and no later change should start. An editor that
    rewrites DACLs under the user's profile is the kind of thing that leaves a
    machine with directories its owner can no longer open, needs an
    Administrator to undo, breaks backup and sync tools that walk the tree, and
    turns every support question into an afternoon of icacls. A coding
    assistant's logs do not warrant that; the file mode they ask for on POSIX
    is a one-line convenience there, not a security boundary we owe Windows an
    equivalent of.

    So on Windows this does nothing, deliberately, rather than pretending to
    protect anything.
    """
    if is_windows():
        return
    os.chmod(path, mode)


def open_append_fd(path, mode: int = 0o600) -> int:
    """A descriptor whose every write lands at the END of the file, whatever
    else is appending to it at the same moment.

    POSIX gets O_APPEND, where the kernel makes the seek-and-write one atomic
    step. The Windows CRT has no such thing: _O_APPEND there is implemented as
    "seek to end, then write", so two appenders can resolve the end to the same
    offset and one lands on top of the other. Measured on Windows 11 Pro
    10.0.26300, 16 processes appending 200 lines each to one file: 248 of 3200
    lines lost and 70 more torn in half. That is not a corner case here --
    every codemason process on a machine appends to the same turns.log, and
    concurrent sub-agents append to one session transcript.

    Win32 does have an atomic append, but only for a handle opened with
    FILE_APPEND_DATA and WITHOUT FILE_WRITE_DATA: such a handle ignores the
    file pointer entirely and every write goes to the end, which is what makes
    it indivisible. The CRT descriptor wrapped around it may still seek; the OS
    does not care. The same 16x200 run through this path: 3200 lines, none
    lost, none torn.

    FILE_SHARE_DELETE is in the share mode because rotation renames the log out
    from under its appenders (utils/log._rotate_if_needed); without it that
    rename fails for as long as any process holds the log open.

    `mode` applies on POSIX only. Windows has no mode bits to set at open time,
    so a caller that needs them must set them separately -- but these files are
    created under a 0700 directory, which is what keeps them unreadable there.
    """
    if not is_windows():
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, mode)

    import ctypes
    import msvcrt
    from ctypes import wintypes

    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    # restype matters: left as the default c_int, a 64-bit HANDLE comes back
    # truncated and every later use of it is a different object or garbage.
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                wintypes.HANDLE)
    FILE_APPEND_DATA = 0x0004
    SHARE_READ_WRITE_DELETE = 0x0001 | 0x0002 | 0x0004
    OPEN_ALWAYS = 4
    FILE_ATTRIBUTE_NORMAL = 0x0080
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    handle = k32.CreateFileW(str(path), FILE_APPEND_DATA, SHARE_READ_WRITE_DELETE,
                             None, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    if not handle or handle == INVALID_HANDLE_VALUE:
        win_err = ctypes.get_last_error()
        raise OSError(0, ctypes.FormatError(win_err), str(path), win_err)
    try:
        # The descriptor owns the handle from here: closing it closes the handle.
        return msvcrt.open_osfhandle(handle, os.O_APPEND)
    except Exception:
        k32.CloseHandle(handle)
        raise
# The process-wide job every shell we spawn is put into. Created on first use
# and then held for the life of the process: it is the HANDLE being open that
# keeps the job alive, and the handle closing is what kills the members.
_KILL_JOB = None
_KILL_JOB_TRIED = False


def _kill_on_close_job():
    """This process's job object, or None if one could not be made.

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: when the last handle to the job closes,
    Windows terminates everything still in it. We never close the handle, so
    the close that matters is the one the kernel does when this process ends --
    by return, by exception, by TerminateProcess, by taskkill /F, by the machine
    losing power. That is the guarantee atexit cannot give.
    """
    global _KILL_JOB, _KILL_JOB_TRIED
    if _KILL_JOB is not None or _KILL_JOB_TRIED:
        return _KILL_JOB
    _KILL_JOB_TRIED = True

    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [('ReadOperationCount', ctypes.c_ulonglong),
                    ('WriteOperationCount', ctypes.c_ulonglong),
                    ('OtherOperationCount', ctypes.c_ulonglong),
                    ('ReadTransferCount', ctypes.c_ulonglong),
                    ('WriteTransferCount', ctypes.c_ulonglong),
                    ('OtherTransferCount', ctypes.c_ulonglong)]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
                    ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
                    ('LimitFlags', wintypes.DWORD),
                    ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t),
                    ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t),
                    ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BASIC_LIMIT),
                    ('IoInfo', IO_COUNTERS),
                    ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t),
                    ('PeakJobMemoryUsed', ctypes.c_size_t)]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9

    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = ()
    # argtypes matter here: GetCurrentProcess returns the pseudo-handle -1, and
    # undeclared ctypes narrows it to c_int, which overflows on 64-bit.
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

    job = k32.CreateJobObjectW(None, None)   # unnamed: ours alone, not shared
    if not job:
        return None

    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(job)
        return None

    # Put OURSELVES in it. This is the whole race fix: Windows enrols every
    # process a job member creates in that same job, at creation, so a shell we
    # spawn from here -- and anything that shell spawns -- is a member before it
    # runs its first instruction. Adopting the child after spawn cannot do that:
    # `bash -c "python ..."` had already started python inside the window, and
    # that grandchild survived a taskkill /F of the session (measured).
    #
    # Failure is not fatal: without this we still adopt each child explicitly,
    # which covers the child itself if not its immediate offspring.
    k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())

    _KILL_JOB = job
    return job


def reap_children_with_this_process() -> bool:
    """Make everything this process spawns from now on die when it does.

    Call it BEFORE spawning a shell. It creates the job (once) and puts this
    process in it, which is what makes the guarantee airtight: a job member's
    children join the same job at creation, so there is no window in which a
    grandchild can be started outside it. Adopting a child after spawn leaves
    exactly that window, and it is not theoretical -- `bash -c "python ..."`
    reached python inside it, and that python outlived a taskkill /F.

    Only shells go through here. Something the user is meant to keep -- a
    browser, an editor -- must not be started by a job member without
    CREATE_BREAKAWAY_FROM_JOB, or it would close when codemason exits.

    No-op off Windows, where start_new_session plus the cleanup registry and
    killpg already cover the tidy paths.
    """
    if not is_windows():
        return False
    return _kill_on_close_job() is not None


def reap_with_this_process(pid: int) -> bool:
    """Make `pid` die when this process does, however this process dies.

    Sub-agents are tasks inside this process, so they already go when it goes.
    A shell is the only real child, and it was outliving us: measured on
    Windows 11 Pro 10.0.26300, `taskkill /F` on codemason left the background
    shell running, because TerminateProcess runs no atexit handler and no
    cleanup registry. POSIX has the same hole with SIGKILL; Windows users reach
    for taskkill /F far more often.

    Windows answers this with a job object, which is the closest thing it has to
    a process group: a child assigned to one cannot escape it, its own children
    are in it too, and the kernel terminates the whole job when our last handle
    to it closes. Nested jobs have been allowed since Windows 8, so this still
    works when something else (a terminal, a CI runner) has already put us in a
    job of its own.

    Call it right after spawning. A child can in principle fork a grandchild in
    the microseconds before the assignment lands; that grandchild is not in the
    job, which is the one gap, and it is the same gap a POSIX setpgid race has.

    Only for processes that MUST NOT outlive the session -- the shells. Never
    for something the user is meant to keep, like a browser or an editor:
    assigning those would close them when codemason exits.

    No-op off Windows, where the cleanup registry plus killpg already covers
    the tidy paths. Returns whether the child was adopted.
    """
    if not is_windows():
        return False
    job = _kill_on_close_job()
    if not job:
        return False

    import ctypes
    from ctypes import wintypes

    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001

    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

    handle = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
    if not handle:
        return False          # already gone, or not ours to adopt
    try:
        return bool(k32.AssignProcessToJobObject(job, handle))
    finally:
        k32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """Is this pid running? Portable.

    POSIX spells the test `kill(pid, 0)` — signal 0 checks permission to signal
    without sending one. On Windows there are no signals, and Python maps
    os.kill onto TerminateProcess with the signal as the exit code: `os.kill(pid,
    0)` does not ask whether the process is alive, it **kills it**. Used for a
    stale-lock check, that turns "is the holder still there?" into "there, now it
    isn't" — and the holder is another codemason session.

    So Windows goes through OpenProcess and asks for the exit code instead.
    """
    if pid <= 0:
        return False
    if not is_windows():
        import os
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True                 # exists, owned by someone else
        except (ValueError, OSError):
            return False
    import ctypes
    from ctypes import wintypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False                    # gone, or not ours to look at
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(h)
