"""
State management for CodeMason
Equivalent to src/utils/state.ts
"""

import os
import uuid
from pathlib import Path
from typing import Optional

# Global state
_cwd: Optional[str] = None
_original_cwd: Optional[str] = None
_session_id: Optional[str] = None


def get_session_id() -> str:
    """Get the current session id (bootstrap/state.ts getSessionId). Generated
    lazily on first access; stable for the process lifetime unless switched by
    resume ('clear' / --resume both go through set_session_id)."""
    global _session_id
    if _session_id is None:
        _session_id = str(uuid.uuid4())
    return _session_id


def set_session_id(session_id: str) -> None:
    """Switch the active session id (switchActiveSession subset). Callers that
    change the id must also reset the session-storage file pointer — use
    session_storage.switch_session() rather than calling this directly."""
    global _session_id
    _session_id = session_id


def get_cwd() -> str:
    """Get current working directory"""
    global _cwd
    if _cwd is None:
        _cwd = os.getcwd()
    return _cwd


def set_cwd(path: str) -> None:
    """Set current working directory"""
    global _cwd
    _cwd = path


def get_original_cwd() -> str:
    """Get original working directory (from startup)"""
    global _original_cwd
    if _original_cwd is None:
        _original_cwd = os.getcwd()
    return _original_cwd


def set_original_cwd(path: str) -> None:
    """Set original working directory"""
    global _original_cwd
    _original_cwd = path


def reset_state() -> None:
    """Reset all state to defaults"""
    global _cwd, _original_cwd, _session_id
    _cwd = None
    _original_cwd = None
    _session_id = None


def is_in_directory(path: str, base_dir: str) -> bool:
    """Check if path is within base directory (inclusive of base itself).

    Uses path-segment containment via Path.is_relative_to, NOT string
    startswith — the latter wrongly matches sibling prefixes (e.g. treating
    '/home/user/work-evil' as inside '/home/user/work'), which is the exact
    containment bypass pathInAllowedWorkingPath guards against.
    """
    try:
        path_resolved = Path(path).resolve()
        base_resolved = Path(base_dir).resolve()
        if path_resolved == base_resolved:
            return True
        return path_resolved.is_relative_to(base_resolved)
    except AttributeError:
        # Python < 3.9 fallback: compare resolved parts.
        try:
            pr = Path(path).resolve().parts
            br = Path(base_dir).resolve().parts
            return len(pr) >= len(br) and pr[: len(br)] == br
        except Exception:
            return False
    except Exception:
        return False
