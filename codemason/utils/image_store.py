"""
Images of a session, on disk next to its transcript:

    <CONFIG_DIR>/projects/<sanitized-cwd>/<sessionId>/images/
        <sha256>.<ext>        an image as it was sent to the model
        orig-<sha256>.<ext>   the original it was made from (debugging)

The transcript does not carry the base64 payload. When an entry is written, an
image block's base64 source is stored as a file and replaced by a reference

    {"type": "codemason_file", "path": "<sha256>.<ext>", "sha256": ..., "media_type": ...}

and the reverse happens when a session is resumed. Only the copy written to disk
changes; the in-memory messages keep their base64. A reference whose file is
gone comes back as a text note instead of an image, so a resumed session never
sends a broken image part. Deleting a session in the resume picker deletes this
folder with it (resume_picker.py).
"""

import base64
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_EXT = {'image/png': 'png', 'image/jpeg': 'jpg', 'image/gif': 'gif', 'image/webp': 'webp'}
# What dehydrate writes as a reference: a bare file name, nothing that walks.
_REF_NAME = re.compile(r'^[0-9a-f]{64}\.[a-z0-9]{1,5}$')


def images_dir(session_id: Optional[str] = None) -> Path:
    from .session_storage import get_project_dir
    from .state import get_original_cwd, get_session_id
    return get_project_dir(get_original_cwd()) / (session_id or get_session_id()) / 'images'


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # A writer-unique temp name: two writers of the same image (parallel
    # subagents) must not share one, or the second rename finds it gone.
    # Whoever renames last wins with identical bytes.
    tmp = path.with_suffix(f'{path.suffix}.{uuid.uuid4().hex}.tmp')
    tmp.write_bytes(data)
    try:
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        if not path.exists():
            raise


def _in_own_store(path: Path) -> bool:
    """True for a file codemason itself stored by content hash (a phone
    attachment, a session image): never edited in place, so a hard link to it is
    as good as a copy."""
    try:
        from .config import CONFIG_DIR
        from ..remote.adapters.telegram import attachments_dir
        p = Path(path).resolve()
        return any(p.is_relative_to(Path(root).resolve())
                   for root in (CONFIG_DIR, attachments_dir()))
    except (OSError, ValueError, ImportError):
        return False


def save_original(data: bytes, ext: str, session_id: Optional[str] = None,
                  source: Optional[Path] = None) -> Path:
    """Keep the original file an image was made from; returns its path.

    `source` is the file `data` was read from. When that is one of codemason's
    own stores (a photo from the phone, an image Read back from this store), the
    copy is a hard link: same bytes, no second copy on disk."""
    sha = hashlib.sha256(data).hexdigest()
    path = images_dir(session_id) / f'orig-{sha}.{ext.lower().lstrip(".") or "bin"}'
    if path.exists():
        return path
    if source is not None and _in_own_store(source):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, path)
            return path
        except OSError:
            pass                    # another filesystem, or no links: copy
    try:
        _write_once(path, data)
    except OSError:
        pass
    return path


def _dehydrate_source(source: dict, session_id: Optional[str]) -> dict:
    data = source.get('data') or ''
    media_type = source.get('media_type') or 'image/png'
    try:
        raw = base64.b64decode(data)
    except Exception:
        return source
    sha = hashlib.sha256(raw).hexdigest()
    name = f'{sha}.{_EXT.get(media_type, "bin")}'
    try:
        _write_once(images_dir(session_id) / name, raw)
    except OSError:
        return source  # keep the payload inline rather than lose the image
    return {'type': 'codemason_file', 'path': name, 'sha256': sha, 'media_type': media_type}


# stored_path answers, by image. Evicted images are labelled on every request
# and each answer costs a full base64 decode and SHA-256, so it is worked out
# once. A failure is remembered too: the label must not gain a path on a later
# request, which would move the request prefix.
_STORED: Dict[Tuple, Optional[Path]] = {}
_STORED_MAX = 4096


def stored_path(source: Any, session_id: Optional[str] = None) -> Optional[Path]:
    """Where a base64 image source lives in the session store, writing it there
    first if it is not yet (dehydrate normally has). None when it cannot be
    stored or is not base64."""
    if not isinstance(source, dict) or source.get('type') != 'base64':
        return None
    data = source.get('data') or ''
    # hash() of a str is computed once per string object and kept, and the
    # conversation holds the same object request after request.
    key = (str(images_dir(session_id)), len(data), hash(data), data[:64], data[-64:])
    if key in _STORED:
        return _STORED[key]
    ref = _dehydrate_source(source, session_id)
    path = (images_dir(session_id) / ref['path']
            if ref.get('type') == 'codemason_file' else None)
    if len(_STORED) >= _STORED_MAX:
        _STORED.clear()
    _STORED[key] = path
    return path


def dehydrate(obj: Any, session_id: Optional[str] = None) -> Any:
    """Copy of a JSON-able message with base64 image sources moved to files."""
    if isinstance(obj, list):
        return [dehydrate(x, session_id) for x in obj]
    if not isinstance(obj, dict):
        return obj
    if obj.get('type') == 'image' and isinstance(obj.get('source'), dict) \
            and obj['source'].get('type') == 'base64':
        out = dict(obj)
        out['source'] = _dehydrate_source(obj['source'], session_id)
        return out
    return {k: dehydrate(v, session_id) for k, v in obj.items()}


def hydrate(obj: Any, session_id: Optional[str] = None) -> Any:
    """Reverse of dehydrate. A missing file becomes a text note."""
    if isinstance(obj, list):
        return [hydrate(x, session_id) for x in obj]
    if not isinstance(obj, dict):
        return obj
    src = obj.get('source')
    if obj.get('type') == 'image' and isinstance(src, dict) and src.get('type') == 'codemason_file':
        name = str(src.get('path', ''))
        if not _REF_NAME.match(name):
            # Not something dehydrate wrote: never follow it out of the store.
            return {'type': 'text', 'text': f'[Image reference not valid: {name[:80]}]'}
        path = images_dir(session_id) / name
        try:
            raw = path.read_bytes()
        except OSError:
            return {'type': 'text', 'text': f'[Image file missing: {path}]'}
        out = dict(obj)
        out['source'] = {'type': 'base64', 'media_type': src.get('media_type') or 'image/png',
                         'data': base64.b64encode(raw).decode('ascii')}
        return out
    return {k: hydrate(v, session_id) for k, v in obj.items()}
