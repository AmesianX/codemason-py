"""This install's prefix-cache session id (Rust parity: utils/cache_session.rs).

Sent only inside the encrypted request body (``cache_session``); the gateway
turns it into this session's cache salt with a server-only secret and removes
it. Everything after our public system prompt is cached under that salt, so no
other session can reuse -- or time -- it.

One random id per install, kept in ``<config dir>/cache_session`` (0600):
stable across runs, so /resume still hits this user's own cached conversation.
"""
import os
import re
import uuid

_VALID = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_cached = None


def _path() -> str:
    from .config import CONFIG_DIR
    return os.path.join(str(CONFIG_DIR), "cache_session")


def session_id() -> str:
    global _cached
    if _cached:
        return _cached
    path = _path()
    try:
        with open(path, encoding="utf-8") as f:
            existing = f.read().strip()
        if _VALID.match(existing):
            _cached = existing
            return existing
    except OSError:
        pass
    fresh = uuid.uuid4().hex
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(fresh)
    except OSError:
        pass
    _cached = fresh
    return fresh
