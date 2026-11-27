"""The abliteration edit, asked for per request.

Python parity for services/ab.rs -- same modes, same lifecycle, same wire
format.

The served checkpoint has its refusal behaviour intact; the edit that removes
it is a rank-24 adapter the server adds at ``o_proj``, and a request gets it by
carrying ``{"vllm_xargs": {"ab_scale": 1}}`` (see the lane's AB-SWITCH.md). It
applies to that request's tokens and nobody else's. The reason it is a
per-request choice and not a checkpoint choice: removing refusal also removes
the model's ability to say a thing does not exist, because declining a request
and denying a premise run through the same direction in the weights. Measured
on the served pair -- 4/60 fabrications with it off, 18-22/60 with it on. So
the model that judges a tool result, a link or a diff should be the restored
one.

Which leaves exactly one REQUEST that needs the edit: the first one of a user
turn, the one that decides whether to accept. Hence ``auto``::

    user turn starts       -> arm (no request)
    turn's first request   -> ab_scale 1
    every request after    -> ab_scale 0 (retries, tool loop, sub-agents)
    next user turn         -> arm again

Zero trust (2026-11-27). This module used to drive a SERVER-WIDE switch,
``POST /ab {"scale":0|1}``: on before the turn's first request, off after it.
That switch was the engine's default for every request that did not carry its
own scale, i.e. every other client's -- one user's ``/ab on`` changed
everyone's model, and measured on the live lane it did. A client cannot be
trusted to leave a shared knob alone (this code is MIT; anyone can send
anything), so the server no longer has one (``POST /ab`` answers 410) and this
client never writes server state. It only decides what to put on ITS OWN
requests. That also retires the failure the old design was built around -- the
edit left on for hours because nothing turned it off (2026-10-01): there is no
"off" to forget any more, because there is no state left behind.

``on`` and ``off`` exist to pin the choice for A/B measurement, not for daily
use. Only the conversation loop's requests carry the key; compaction and helper
calls carry none and get the server default (the original weights).
"""

import os
import threading

from ..utils.config import get_config

OFF = 'off'
ON = 'on'
AUTO = 'auto'
MODES = (OFF, ON, AUTO)

_ALIASES = {'0': OFF, 'false': OFF, '1': ON, 'true': ON}

_lock = threading.Lock()
# ``auto`` by default: the lane serves the restored checkpoint, so without the
# edit every request that needs it gets refused. A server without the plugin
# (or an llama.cpp backend) ignores the key, so this default costs nothing
# where the feature does not exist. Override with CODEMASON_AB=off|on|auto.
_mode = _ALIASES.get((os.environ.get('CODEMASON_AB') or '').strip().lower(),
                     (os.environ.get('CODEMASON_AB') or '').strip().lower())
if _mode not in MODES:
    _mode = AUTO
# Set when a user turn starts, spent by that turn's first request.
_armed = False


def parse_mode(text):
    """'off' | 'on' | 'auto' (plus 0/1/true/false), or None if unrecognised."""
    key = (text or '').strip().lower()
    if key in MODES:
        return key
    return _ALIASES.get(key)


def mode():
    with _lock:
        return _mode


def set_mode(new_mode):
    """Set the mode. Local only: nothing is sent anywhere, now or later -- the
    choice rides on each request (``request_scale``).

    A pending arm is dropped: a mode typed mid-turn is the user's latest word,
    and a stale arm from the mode before it must not fire on their behalf.
    """
    global _mode, _armed
    if new_mode not in MODES:
        raise ValueError(f'unknown /ab mode: {new_mode!r}')
    with _lock:
        _mode = new_mode
        _armed = False


async def on_user_turn():
    """A user turn is starting: under ``auto``, the turn's FIRST request gets
    the edit. Costs nothing -- it only sets a latch ``request_scale`` spends."""
    global _armed
    with _lock:
        if _mode == AUTO:
            _armed = True


def request_scale():
    """The ``ab_scale`` for the conversation request about to be sent.

    ``on`` 1.0, ``off`` 0.0, ``auto`` 1.0 for the turn's first request and 0.0
    for every one after it. Always an explicit value, never "whatever the
    server defaults to": the default is not ours, so the request says what it
    wants.
    """
    global _armed
    with _lock:
        if _mode == ON:
            return 1.0
        if _mode == OFF:
            return 0.0
        was, _armed = _armed, False
        return 1.0 if was else 0.0


def _endpoint():
    cfg = get_config()
    base = (cfg.model_base_url or '').strip().rstrip('/')
    if not base:
        return None
    return base + '/ab', (cfg.model_api_key or '')


def _effective(body):
    """The scale the server will actually apply, read out of an ``/ab`` body.

    Returns ``(scale, consistent)`` or None.

    The lane's switch went per-token (``ab_switch.py``): the value that reaches
    the forward pass is ``_default_scale``, which ``fill_token_scales`` writes
    into the step's vector for every request that did not send its own
    ``ab_scale``, and which the server mirrors into each worker's
    ``details[].default_scale``. The top-level ``scale`` is ``get_scale()`` --
    the legacy scalar ``_scale``, which ``_token_scale()`` stops allocating the
    moment the per-token buffer is reserved, so it reads 0.0 for the life of the
    process.

    Reading that field printed ``server: off (restored)`` over a server holding
    the edit ON (measured 2026-11-21: ``POST {"scale":1}`` answered
    ``{"scale":1.0,"mode":"abliterated"}``, the GET right after it answered
    ``{"scale":0.0,"mode":"original","details":[{"default_scale":1.0,..}]}``).
    It is the mirror image of the 2026-11-20 bug: the display said ``on`` over a
    server at 0.0, the fix was to believe the server rather than the client's cache,
    and the field we started believing had quietly stopped meaning anything.

    ``consistent`` has the same hole -- the server derives it from those
    always-zero scalars, so ranks holding different defaults still look agreed.
    Compare the per-worker defaults instead.

    A body without ``per_token`` is a server from before that change (or another
    server entirely), and there the top-level pair is the right answer.
    """
    if not isinstance(body, dict):
        return None
    if body.get('per_token'):
        details = body.get('details')
        if isinstance(details, list):
            defaults = [float(d['default_scale']) for d in details
                        if isinstance(d, dict) and d.get('default_scale') is not None]
            if defaults:
                head = defaults[0]
                return head, all(abs(s - head) <= 1e-6 for s in defaults)
    if body.get('scale') is None:
        return None
    return float(body['scale']), bool(body.get('consistent', True))


async def query():
    """The server's DEFAULT -- what a request with no ab_scale gets -- as
    (scale, consistent), or None. For ``/ab`` with no argument. Read-only
    (``GET /ab``): the default is fixed at the server's startup and nothing here
    can change it.
    """
    target = _endpoint()
    if target is None:
        return None
    url, key = target
    try:
        import httpx
    except ImportError:
        return None
    headers = {'Authorization': f'Bearer {key}'} if key else {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
        return _effective(resp.json())
    except Exception:
        return None
