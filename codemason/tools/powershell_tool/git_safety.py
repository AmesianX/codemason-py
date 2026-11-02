"""Git sandbox-escape guards (gitSafety.ts).

Git can be weaponized two ways: (1) bare-repo attack — cwd with HEAD/objects/
refs/ but no .git makes git treat cwd as a bare repo and run hooks from cwd;
(2) git-internal write + git — a compound creates HEAD/objects/refs/hooks/ then
runs git, executing the freshly-planted hooks. These detect args that resolve to
git-internal paths in cwd.
"""

import os
import posixpath
import re

from ...utils.state import get_cwd
from ...utils.powershell.parser import PS_TOKENIZER_DASH_CHARS

_GIT_INTERNAL_PREFIXES = ("head", "objects", "refs", "hooks")
_GIT_SHORTNAME_RE = re.compile(r"^git~\d+($|/)")
_PROVIDER_RE = re.compile(r"^(?:[A-Za-z0-9_.]+\\){0,3}FileSystem::", re.IGNORECASE)
_DRIVE_REL_RE = re.compile(r"^[A-Za-z]:(?![/\\])")
_DRIVE_ABS_RE = re.compile(r"^[a-z]:")


def _resolve_cwd_reentry(normalized: str) -> str:
    """`../<cwd-basename>/x` re-enters cwd via parent → strip to cwd-relative."""
    if not normalized.startswith("../"):
        return normalized
    cwd_base = os.path.basename(get_cwd()).lower()
    if not cwd_base:
        return normalized
    prefix = "../" + cwd_base + "/"
    s = normalized
    while s.startswith(prefix):
        s = s[len(prefix):]
    if s == "../" + cwd_base:
        return "."
    return s


def _normalize_git_path_arg(arg: str) -> str:
    """Normalize PS arg text → canonical path for git-internal matching."""
    s = arg
    # Parameter prefix (dash chars / PS 5.1 `/`) with colon-bound value.
    if len(s) > 0 and (s[0] in PS_TOKENIZER_DASH_CHARS or s[0] == "/"):
        c = s.find(":", 1)
        if c > 0:
            s = s[c + 1:]
    s = re.sub(r"^['\"]|['\"]\Z", "", s)
    s = s.replace("`", "")
    s = _PROVIDER_RE.sub("", s)          # FileSystem::path
    s = _DRIVE_REL_RE.sub("", s)         # C:foo (drive-relative) → foo
    s = s.replace("\\", "/")
    # Win32 per-component: iteratively strip trailing spaces then dots (keeping ./..)
    parts = []
    for c in s.split("/"):
        if c == "":
            parts.append(c)
            continue
        prev = None
        while c != prev:
            prev = c
            c = c.rstrip(" ")
            if c == "." or c == "..":
                break
            c = c.rstrip(".")
        parts.append(c or ".")
    s = "/".join(parts)
    s = posixpath.normpath(s)
    if s.startswith("./"):
        s = s[2:]
    return s.lower()


def _resolve_escaping_path_to_cwd_relative(n: str):
    """Resolve an escaping (../, absolute, drive) path against cwd; return the
    cwd-relative remainder if it lands back inside cwd, else None."""
    cwd = get_cwd()
    abs_path = os.path.abspath(os.path.join(cwd, n))
    sep = os.sep
    cwd_with_sep = cwd if cwd.endswith(sep) else cwd + sep
    abs_lower = abs_path.lower()
    cwd_lower = cwd.lower()
    cwd_with_sep_lower = cwd_with_sep.lower()
    if abs_lower == cwd_lower:
        return "."
    if not abs_lower.startswith(cwd_with_sep_lower):
        return None
    return abs_path[len(cwd_with_sep):].replace("\\", "/").lower()


def _matches_git_internal_prefix(n: str) -> bool:
    if n == "head" or n == ".git":
        return True
    if n.startswith(".git/") or _GIT_SHORTNAME_RE.match(n):
        return True
    for p in _GIT_INTERNAL_PREFIXES:
        if p == "head":
            continue
        if n == p or n.startswith(p + "/"):
            return True
    return False


def _matches_dot_git_prefix(n: str) -> bool:
    if n == ".git" or n.startswith(".git/"):
        return True
    return bool(_GIT_SHORTNAME_RE.match(n))


def is_git_internal_path_ps(arg: str) -> bool:
    """True if arg resolves to a git-internal path in cwd (bare-repo hooks/refs/
    or standard .git/*)."""
    n = _resolve_cwd_reentry(_normalize_git_path_arg(arg))
    if _matches_git_internal_prefix(n):
        return True
    if n.startswith("../") or n.startswith("/") or _DRIVE_ABS_RE.match(n):
        rel = _resolve_escaping_path_to_cwd_relative(n)
        if rel is not None and _matches_git_internal_prefix(rel):
            return True
    return False


def is_dot_git_path_ps(arg: str) -> bool:
    """True if arg resolves to a path inside .git/ (standard-repo metadata). Unlike
    is_git_internal_path_ps, does NOT match bare-repo root-level hooks/refs/."""
    n = _resolve_cwd_reentry(_normalize_git_path_arg(arg))
    if _matches_dot_git_prefix(n):
        return True
    if n.startswith("../") or n.startswith("/") or _DRIVE_ABS_RE.match(n):
        rel = _resolve_escaping_path_to_cwd_relative(n)
        if rel is not None and _matches_dot_git_prefix(rel):
            return True
    return False
