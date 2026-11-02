"""
Vision capability — how many images the configured server accepts per request.

The limit belongs to the server (vLLM --limit-mm-per-prompt), so ask the server
instead of trusting a config value: send one request carrying more tiny images
than any server allows. vLLM checks the count while parsing the request, before
it decodes a single image, and answers

    400 "At most N image(s) may be provided in one prompt."

N == 0 means the server runs text-only (--language-model-only), as does "is not
a multimodal model". The result is cached for the process. An answer that says
neither (connection error, 401, 5xx, a 400 for some other reason) is NOT cached
as a limit: the next call asks again once the server is up (same policy as
services/backend.py), but not sooner than _RETRY_AFTER seconds — every request
and every image Read asks, and while the server is down each would otherwise
send a probe of its own.

Env CODEMASON_VISION_MAX_IMAGES overrides the probe: 0 turns vision off, N trusts
N. (An env var, not a config.json field: the config file holds measured user
values and is not extended for this.) Only the vLLM backend calls this.
"""

import json
import os
import re
import time
from typing import Optional

# 28x28 white PNG: one 28-pixel cell, the smallest size every processor takes
# (Qwen-style processors reject an image under their 28-pixel factor, which
# would read as "text-only"). Never decoded when the probe works as intended
# (the count check rejects the request first); if a server accepts this many,
# each costs only the minimum vision tokens.
_TINY_PNG = (
    'data:image/png;base64,'
    'iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAKElEQVR42mP8//8/A7UBEwMNwKiho4aOGjpq'
    '6Kiho4aOGjpq6OAzFADRYgM1sivIJAAAAABJRU5ErkJggg=='
)
_PROBE_COUNT = 65
_LIMIT_RE = re.compile(r'At most (\d+) image', re.IGNORECASE)
# vLLM's words for a model with no image input at all (multimodal/registry.py).
_TEXT_ONLY_RE = re.compile(r'is not a multimodal model', re.IGNORECASE)
# Seconds before an inconclusive probe is repeated.
_RETRY_AFTER = 30.0

_limit: Optional[int] = None  # cached probe result
_last_inconclusive = 0.0      # monotonic time of the last probe that could not tell


def _probe(cfg) -> Optional[int]:
    """Ask the server. None = could not tell (do not cache).

    Sent the way every model request is (services/_openai_common.py): bearer
    key when set, and the body sealed when model_aes_key is set — a gateway
    that refuses plaintext would otherwise answer 400 and read as "no images".
    Error bodies come back in the clear (deploy/encrypt_gateway.py)."""
    import httpx
    headers = {'Content-Type': 'application/json'}
    if cfg.model_api_key:
        headers['Authorization'] = f'Bearer {cfg.model_api_key}'
    content = [{'type': 'text', 'text': '.'}]
    content += [{'type': 'image_url', 'image_url': {'url': _TINY_PNG}}] * _PROBE_COUNT
    body = {
        'model': cfg.model_name,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': 1,
    }
    kwargs = dict(json=body, headers=headers)
    try:
        if cfg.model_aes_key:
            from ..utils.body_crypto import encrypt, load_key
            kwargs = dict(
                content=encrypt(load_key(cfg.model_aes_key), json.dumps(body)).encode('ascii'),
                headers={**headers, 'Content-Type': 'application/octet-stream',
                         'X-Body-Encrypted': 'aesgcm'})
        resp = httpx.post(f"{cfg.model_base_url}/v1/chat/completions",
                          timeout=30.0, **kwargs)
    except Exception:
        return None
    if resp.status_code == 200:
        return _PROBE_COUNT  # accepts at least this many
    text = resp.text or ''
    m = _LIMIT_RE.search(text)
    if m:
        return int(m.group(1))
    if resp.status_code in (400, 422) and _TEXT_ONLY_RE.search(text):
        return 0
    return None   # 401/5xx/another 400 — cannot tell yet


def image_limit(cfg=None) -> int:
    """Images the server accepts in one request; 0 = no vision."""
    global _limit
    if cfg is None:
        from ..utils.config import get_config
        cfg = get_config()
    override = os.environ.get('CODEMASON_VISION_MAX_IMAGES', '').strip()
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            pass
    if _limit is not None:
        return _limit
    global _last_inconclusive
    if _last_inconclusive and time.monotonic() - _last_inconclusive < _RETRY_AFTER:
        return 0
    found = _probe(cfg)
    if found is None:
        _last_inconclusive = time.monotonic()
        return 0
    _limit = found
    return _limit


_last_enabled = 0  # what enabled_limit() last returned


def enabled_limit() -> int:
    """image_limit() when the active backend is vLLM, else 0: what a system
    prompt states and what a request may carry."""
    global _last_enabled
    from .backend import get_backend
    n = image_limit() if get_backend() == 'vllm' else 0
    _last_enabled = n
    return n


def known_limit() -> int:
    """What enabled_limit() last returned (0 before it ever ran), without asking
    anyone. For display code that runs inside a repaint, where a probe would
    freeze the prompt while the server is down."""
    return _last_enabled
