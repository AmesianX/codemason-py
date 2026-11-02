"""
File Utilities for CodeMason
Equivalent to src/utils/file.ts
"""

import codecs
import os
import glob as globlib
from pathlib import Path
from typing import Optional, Tuple, List
from functools import lru_cache

from .state import get_cwd
from .log import log_error
from .platform import chmod_posix


# === Types ===

LineEndingType = str  # 'CRLF' | 'LF'


# === Glob ===

def glob(
    file_pattern: str,
    cwd: str,
    limit: int = 1000,
    offset: int = 0,
) -> Tuple[List[str], bool]:
    """
    Find files matching a glob pattern.

    Args:
        file_pattern: Glob pattern to match
        cwd: Working directory
        limit: Maximum number of files to return
        offset: Number of files to skip

    Returns:
        Tuple of (matching files, whether results were truncated)
    """
    try:
        pattern = os.path.join(cwd, file_pattern)
        all_paths = globlib.glob(pattern, recursive=True)

        # Filter out directories
        file_paths = [p for p in all_paths if os.path.isfile(p)]

        # Sort by modification time
        file_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        truncated = len(file_paths) > offset + limit
        result = file_paths[offset:offset + limit]

        return result, truncated

    except Exception as e:
        log_error(f"Error globbing pattern {file_pattern}: {e}")
        return [], False


# === File Reading ===

def read_file_safe(filepath: str) -> Optional[str]:
    """Read file content safely, returning None on error"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        log_error(f"Error reading file {filepath}: {e}")
        return None


def read_text_content(
    file_path: str,
    offset: int = 0,
    max_lines: Optional[int] = None
) -> Tuple[str, int, int]:
    """
    Read text content from a file with optional line offset and limit.

    Args:
        file_path: Path to the file
        offset: Line offset (0-indexed)
        max_lines: Maximum number of lines to read

    Returns:
        Tuple of (content, line_count, total_lines)
    """
    encoding = detect_file_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
    except Exception as e:
        log_error(f"Error reading file {file_path}: {e}")
        return '', 0, 0

    lines = content.split('\n')
    total_lines = len(lines)

    # Apply offset and limit
    if max_lines is not None and len(lines) - offset > max_lines:
        to_return = lines[offset:offset + max_lines]
    else:
        to_return = lines[offset:]

    return '\n'.join(to_return), len(to_return), total_lines


def atomic_write_bytes(file_path: str, data: bytes) -> None:
    """Write bytes atomically: a temp file in the same directory + os.replace,
    with fsync and preserved mode. Prevents a truncated/corrupt file if the
    process dies mid-write (the original's writeFileSyncAndFlush). Follows a
    symlink to its target so we don't clobber the link with a plain file."""
    import tempfile

    real = os.path.realpath(file_path)
    directory = os.path.dirname(real) or '.'
    try:
        mode = os.stat(real).st_mode
    except OSError:
        mode = None

    fd, tmp = tempfile.mkstemp(dir=directory, prefix='.cm-tmp-')
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            try:
                chmod_posix(tmp, mode)
            except OSError:
                pass
        os.replace(tmp, real)   # atomic within the same filesystem
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text_content(
    file_path: str,
    content: str,
    encoding: str = 'utf-8',
    endings: LineEndingType = 'LF'
) -> None:
    """
    Write text content to a file.

    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding
        endings: Line ending type ('LF' or 'CRLF')
    """
    to_write = content
    if endings == 'CRLF':
        to_write = content.replace('\n', '\r\n')

    # newline='' disables the text layer's own line-ending translation, which
    # this function must own: without it, '\n' becomes os.linesep on write, so
    # on Windows asking for LF produced CRLF, and asking for CRLF produced
    # '\r\r\n' — the replace above ran first and the text layer then translated
    # the '\n' inside each '\r\n' again. Invisible on Linux, where os.linesep is
    # already '\n'. Measured on Windows: LF -> b'one\r\ntwo\r\n',
    # CRLF -> b'one\r\r\ntwo\r\r\n'.
    #
    # atomic_write_bytes above is the better primitive and is what the Edit tool
    # uses; this one is kept for its signature and currently has no callers, so
    # the minimal correctness fix is the right scope.
    with open(file_path, 'w', encoding=encoding, newline='') as f:
        f.write(to_write)


# === Encoding Detection ===


# Legacy encodings we are willing to name, most likely first. Korean first
# because that is the environment this runs in; the rest are the other
# single/double-byte families a source file on a CJK machine turns up in.
#
# PYTHON CODEC NAMES, chosen to match what the Rust port's WHATWG labels
# actually resolve to — not the WHATWG spellings. This bit, and it bit silently:
# Python's 'euc-kr' is strict KS X 1001, so it CANNOT decode the extended Hangul
# that CP949 adds, which is most of what a Korean Windows editor writes. A real
# CP949 file therefore fell past it to 'gbk', which round-trips those bytes
# perfectly well and reads them as Chinese — '똠방각하 코드' came back as
# '宑规阿窍 内靛'. Rust never had this because encoding_rs's "euc-kr" label IS
# CP949/UHC.
#
# That failure also corrects a claim in verified_legacy_encoding below: the
# round-trip proves the bytes SURVIVE, which is what Edit needs, but it does not
# prove the text is right. Two encodings can both round-trip the same bytes and
# only one of them means anything. Order is what supplies correctness, so the
# encoding this machine actually writes has to come first.
_LEGACY_CANDIDATES = ('cp949', 'cp932', 'gbk', 'big5')


def is_legacy_label(label: str) -> bool:
    """Is this one of the legacy labels the detector can return?"""
    return label in _LEGACY_CANDIDATES


def verified_legacy_encoding_exact(data: bytes):
    """Same as verified_legacy_encoding, but the input must round-trip WHOLE.

    The trimming variant exists for a 4096-byte HEADER, where a multi-byte
    sequence really can be cut by the sample boundary. Applied to a whole file it
    is unsound, and that was a real hole: a label accepted on `data[:-2]` was
    then used to encode all of `data`, so Edit could rewrite the final bytes of a
    file it had never proved it could reproduce. Measured — a big5 file ending in
    one of Python's duplicate mappings had its last two bytes silently changed by
    an edit to an unrelated ASCII line.
    """
    return _verified_legacy(data, max_trim=0)


def verified_legacy_encoding(data: bytes):
    """The candidate that reproduces `data` EXACTLY, or None.

    A verification, not a guess, and that distinction is the whole safety
    argument. Edit reads a file, edits text, and writes it back through the same
    codec — so naming an encoding that cannot reproduce the original bytes is
    how a CP949 source lost every Hangul character to U+FFFD once already. Here
    a candidate is accepted only when decode-then-encode returns the input byte
    for byte, which makes the write-back lossless by construction rather than by
    hoping the detector guessed right.

    What it does NOT prove is that the text means anything: several CJK
    encodings will round-trip the same bytes while reading them as different
    languages. Byte-safety comes from this check; correctness comes from trying
    the likeliest encoding first.

    `data` may be a header sample, so a multi-byte sequence can be cut at the
    end; one or two trailing bytes are dropped before giving up, since a
    truncated tail says nothing about the encoding of everything before it.
    """
    return _verified_legacy(data, max_trim=2)


def _verified_legacy(data: bytes, max_trim: int):
    for label in _LEGACY_CANDIDATES:
        for trim in range(max_trim + 1):
            if trim >= len(data):
                break
            sample = data[:len(data) - trim] if trim else data
            if not sample:
                break
            try:
                text = sample.decode(label)
                if text.encode(label) == sample:
                    return label
            except (UnicodeDecodeError, UnicodeEncodeError, LookupError):
                continue
    return None


@lru_cache(maxsize=1000)
def detect_file_encoding(file_path: str) -> str:
    """
    Detect file encoding by reading BOM or analyzing content.

    Returns:
        Encoding string (e.g., 'utf-8', 'utf-16-le')
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4096)

        # Check BOM. utf-16-be is detected too: without it a UTF-16BE file falls
        # through to the latin-1 branch, which round-trips its bytes but reads as
        # mojibake, and Read (file_read_tool) already handles \xfe\xff — so the
        # two disagreed about the same file.
        if header[:2] == b'\xff\xfe':
            return 'utf-16-le'
        if header[:2] == b'\xfe\xff':
            return 'utf-16-be'
        if header[:3] == b'\xef\xbb\xbf':
            return 'utf-8-sig'

        # Try UTF-8 on the 4096-byte sample. An INCREMENTAL decoder is required:
        # a plain header.decode('utf-8') raises when a multi-byte character
        # straddles the 4096 boundary, so a perfectly valid UTF-8 file with any
        # CJK/emoji content near that offset was misdetected as latin-1 — which
        # then made Read return mojibake and made Write refuse to encode Hangul
        # into it. final=False buffers a trailing partial sequence instead of
        # failing on it; a genuinely invalid byte inside the sample still raises.
        try:
            codecs.getincrementaldecoder('utf-8')().decode(header, False)
            return 'utf-8'
        except UnicodeDecodeError:
            # A legacy CJK file, proven by round-trip rather than sniffed.
            # Without this a CP949 source was filed as latin-1: bytes were
            # preserved (so Edit was safe) but the text was not the text, so
            # searching it for Hangul matched nothing.
            legacy = verified_legacy_encoding(header)
            return legacy if legacy else 'latin-1'

    except Exception as e:
        log_error(f"Error detecting encoding for {file_path}: {e}")
        return 'utf-8'


@lru_cache(maxsize=1000)
def detect_line_endings(file_path: str) -> LineEndingType:
    """
    Detect line ending type of a file.

    Returns:
        'CRLF' or 'LF'
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read(4096)

        crlf_count = content.count(b'\r\n')
        lf_count = content.count(b'\n') - crlf_count

        return 'CRLF' if crlf_count > lf_count else 'LF'

    except Exception as e:
        log_error(f"Error detecting line endings for {file_path}: {e}")
        return 'LF'


# === Path Utilities ===

# (A second, unused is_in_directory lived here with the same str.startswith
# sibling-prefix bug fixed in utils/state.py. It was dead code — removed. Use
# utils.state.is_in_directory, the single segment-containment implementation.)


def normalize_file_path(file_path: str) -> str:
    """
    Normalize a file path, fixing common issues.

    - Fixes paths missing leading slash (e.g., 'home/user' -> '/home/user')
    - Resolves relative paths
    """
    # Fix paths that look like absolute paths but are missing leading slash
    normalized_path = file_path
    if not os.path.isabs(file_path):
        for prefix in ['home/', 'usr/', 'tmp/', 'var/', 'etc/']:
            if file_path.startswith(prefix):
                normalized_path = '/' + file_path
                break

    # Resolve to absolute path
    if os.path.isabs(normalized_path):
        return normalized_path
    return os.path.abspath(os.path.join(get_cwd(), normalized_path))


def get_absolute_path(path: Optional[str]) -> Optional[str]:
    """Get absolute path from a path string"""
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(get_cwd(), path))


def get_absolute_and_relative_paths(path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Get both absolute and relative paths"""
    absolute_path = get_absolute_path(path)
    relative_path = os.path.relpath(absolute_path, get_cwd()) if absolute_path else None
    return absolute_path, relative_path


def get_display_path(file_path: Optional[str]) -> str:
    """The shortest honest way to name a path on screen (getDisplayPath).

    Three steps, in the source's order: cwd-relative when the file is under the
    working directory, `~`-prefixed when it is under the home directory, and the
    absolute path otherwise. The `..` check is what keeps a sibling directory
    from being displayed as a relative path that is longer and harder to read
    than the absolute one.
    """
    if not file_path:
        return ''
    _, relative_path = get_absolute_and_relative_paths(file_path)
    if relative_path and not relative_path.startswith('..'):
        return relative_path
    home = os.path.expanduser('~')
    if home and file_path.startswith(home + os.sep):
        return '~' + file_path[len(home):]
    return file_path


# === File Discovery ===

def find_file_in_subdirectories(file_path: str) -> Optional[str]:
    """
    Find a file in subdirectories when the exact path doesn't exist.

    For example, if ai_providers/file.py does not exist,
    search for pattern to find deep-eye/ai_providers/file.py
    """
    try:
        # Don't search if path is absolute or already exists
        if os.path.isabs(file_path) or os.path.exists(file_path):
            return None

        # Search for the file in subdirectories
        pattern = f"**/{file_path}"
        matches = globlib.glob(pattern, recursive=True)

        # Filter out node_modules and .git
        matches = [
            m for m in matches
            if 'node_modules' not in m and '.git' not in m
        ]

        return matches[0] if matches else None

    except Exception as e:
        log_error(f"Error finding file in subdirectories for {file_path}: {e}")
        return None


def find_similar_file(file_path: str) -> Optional[str]:
    """
    Find files with the same name but different extensions.

    Args:
        file_path: Path to the file that doesn't exist

    Returns:
        Filename of similar file, or None
    """
    try:
        dir_path = os.path.dirname(file_path)
        file_basename = os.path.splitext(os.path.basename(file_path))[0]

        if not os.path.exists(dir_path):
            return None

        files = os.listdir(dir_path)

        # Find files with same base name but different extension
        similar_files = [
            f for f in files
            if os.path.splitext(f)[0] == file_basename and
               os.path.join(dir_path, f) != file_path
        ]

        return similar_files[0] if similar_files else None

    except Exception as e:
        log_error(f"Error finding similar file for {file_path}: {e}")
        return None


# === Line Numbers ===

def add_line_numbers(content: str, start_line: int = 1) -> str:
    """
    Add cat -n style line numbers to content.

    Args:
        content: The content to add line numbers to
        start_line: Starting line number (1-indexed)

    Returns:
        Content with line numbers
    """
    if not content:
        return ''

    lines = content.split('\n')
    numbered_lines = []

    for i, line in enumerate(lines):
        line_num = i + start_line
        num_str = str(line_num)

        # Handle large numbers differently
        if len(num_str) >= 6:
            numbered_lines.append(f"{num_str}\t{line}")
        else:
            # Regular numbers get padding to 6 characters
            padded = num_str.rjust(6)
            numbered_lines.append(f"{padded}\t{line}")

    return '\n'.join(numbered_lines)


# === Directory Utilities ===

def is_dir_empty(dir_path: str) -> bool:
    """Check if a directory is empty"""
    try:
        with os.scandir(dir_path) as it:
            return next(it, None) is None
    except Exception as e:
        log_error(f"Error checking directory {dir_path}: {e}")
        return False


# === Inter-process locks for shared files ===================================
#
# Several codemason processes share one config dir, and they are not all the
# same program: a Python session, a Rust session, two or three of each, on
# different models and different endpoints. Every file in that tree is therefore
# either owned by exactly one writer (a session transcript, an inbox) or it is a
# SHARED DOCUMENT that any of them may rewrite — config.json, settings.json,
# mcp.json. A shared document is read, changed and written back, and that
# sequence is not atomic however carefully the final write is done: two
# processes that read the same bytes and then both write produce whatever the
# later rename happened to carry. Measured on this box, with the two builds
# saving one setting each at the same moment: 150 of 150 trials lost one of the
# two edits.
#
# So shared documents get one primitive, and these are the rules it follows.
#
# flock, not a lock file whose EXISTENCE is the lock. The kernel releases an
# flock when the descriptor closes — on return, on exception, on sys.exit, on
# SIGKILL, on a segfault, on a power cut — so there is no stale state to
# reclaim and nothing to clean up. A pid file or an O_EXCL create has to be
# removed by the holder, and the one run that does not remove it (kill -9,
# a full disk, a reboot) wedges every run after it. The lock file here carries
# no content and is never deleted: an empty file left behind is not residue,
# it is the lock we take next time.
#
# A SIDECAR, never the document itself. The document is replaced by rename
# (atomic_write_bytes), which swaps the inode. A lock held on the old inode is
# invisible to a process that opens the path afterwards and gets the new one,
# so both would write while both believed they held the lock — the lock would
# silently do nothing. The sidecar is created once and never renamed, never
# unlinked, so every process locks the same inode forever. For the same reason
# this code must never unlink a lock file: unlinking a flocked path lets the
# next process create a fresh inode and lock that, which is how you get two
# holders.
#
# LOCK_NB in a deadline loop, never a blocking LOCK_EX. A blocking acquire is
# uninterruptible, and flock survives SIGSTOP, so one wedged or stopped process
# would park every other session indefinitely with nothing on screen.
#
# A timeout DEGRADES, it does not fail. Not acquiring leaves exactly today's
# behaviour — a possible lost update, never a corrupt file, because the write
# is still an atomic rename. Refusing to save would turn a rare race into a
# visible failure of a command the user just typed.
#
# Held for the read-modify-write only: a few milliseconds, on a cold path (a
# settings change, a first-run backfill). Never across network I/O, never
# across a prompt, never on the streaming path.

_LOCK_POLL_SECONDS = 0.02
_LOCK_TIMEOUT_SECONDS = 5.0

try:
    import fcntl as _fcntl
    _msvcrt = None
except ImportError:                 # Windows
    _fcntl = None
    try:
        import msvcrt as _msvcrt
    except ImportError:             # neither: no advisory locking on this platform
        _msvcrt = None


class FileLock:
    """Exclusive inter-process lock on `lock_path`, for a short critical section.

    Used as a context manager; the value it yields is whether the lock is
    actually held, so a caller can record a degraded write:

        with FileLock(p) as held:
            ...read, change, atomic_write...

    See the rules above. `wait=False` asks for the lock without waiting at all
    (for work that another process doing the same work makes unnecessary, like
    log rotation).
    """

    def __init__(self, lock_path, timeout: float = _LOCK_TIMEOUT_SECONDS,
                 wait: bool = True):
        self._path = str(lock_path)
        self._timeout = timeout
        self._wait = wait
        self._fd: Optional[int] = None
        self.held = False

    def __enter__(self) -> bool:
        import time
        try:
            os.makedirs(os.path.dirname(self._path) or '.', exist_ok=True)
            # O_CREAT without O_EXCL: the file's existence is not the lock, so a
            # file left by any earlier run is simply the lock we take now.
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            return False            # unwritable config dir: proceed unlocked
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                if _fcntl is not None:
                    _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                elif _msvcrt is not None:
                    _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
                else:
                    break           # no advisory locking here; carry on unlocked
                self._fd = fd
                self.held = True
                return True
            except InterruptedError:
                continue            # a signal landed mid-call; says nothing about the lock
            except OSError:
                if not self._wait or time.monotonic() >= deadline:
                    break
                time.sleep(_LOCK_POLL_SECONDS)
        os.close(fd)                # holding an unlocked descriptor buys nothing
        return False

    def __exit__(self, *_exc) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        self.held = False
        try:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            elif _msvcrt is not None:
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
        except OSError:
            pass                    # closing the descriptor releases it anyway
        finally:
            os.close(fd)


def update_json_document(path, lock_path, mutate, *, trailing_newline: bool = False,
                         indent: int = 2) -> bool:
    """Read a shared JSON document, apply `mutate`, write it back atomically.

    The whole read-modify-write runs under `lock_path`, so a second process
    cannot read the same bytes and overwrite the change with its own view. This
    is the one way the shared documents in the config dir are written; see the
    rules above.

    `mutate` is called with the parsed document — None when the file does not
    exist yet — and returns what to write, or None to write nothing. Returns
    whether the file was written.

    A file that EXISTS but cannot be read is never overwritten, and `mutate` is
    not called at all: the caller gets False. Missing and unreadable are not the
    same answer, and treating them alike is how a one-character typo in a hand-
    edited settings.json would be "fixed" by replacing the user's hooks and
    permissions with whatever this process happened to want to add. config.json
    has refused to do that since it learned the lesson (see config._load_failed);
    this is that rule for every shared document.
    """
    import json

    with FileLock(lock_path):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                current = json.load(fh)
        except FileNotFoundError:
            current = None
        except (OSError, ValueError, UnicodeDecodeError):
            return False
        updated = mutate(current)
        if updated is None:
            return False
        text = json.dumps(updated, indent=indent, ensure_ascii=False)
        if trailing_newline:
            text += '\n'
        atomic_write_bytes(str(path), text.encode('utf-8'))
        return True
