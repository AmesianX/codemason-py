"""The server's abliteration switch, driven from the turn loop.

Python parity for services/ab.rs — same modes, same lifecycle, same wire
format.

The served checkpoint has its refusal behaviour intact; the edit that removes
it is a rank-24 adapter the server can add to ``o_proj`` per request
(``POST /ab {"scale":0|1}``, see the lane's AB-SWITCH.md). The reason it is a
switch and not a checkpoint choice: removing refusal also removes the model's
ability to say a thing does not exist, because declining a request and denying
a premise run through the same direction in the weights. Measured on the
served pair — 4/60 fabrications with it off, 18-22/60 with it on. So the model
that judges a tool result, a link or a diff should be the restored one.

Which leaves exactly one REQUEST that needs the edit: the first one of a user
turn, the one that decides whether to accept. Hence ``auto`` — on for that
request, off the moment it answers::

    user turn starts       -> arm (no request)
    turn's first request   -> on
    it answers, anything   -> off
    every request after    -> stays off
    next user turn         -> arm again

The off is paired with the on, inside one request, and that pairing is the
whole point. It used to hang off the first TOOL CALL instead, on the assumption
that a turn which accepts goes on to call one. Turns that never call a tool — a
plain text answer, an empty turn, an abort, a generation the degenerate guard
cuts — left the switch ON, and nothing else ever turned it off: not the end of
the turn, not the next turn (which pushes the same 1.0 and is skipped as
already-there), not process exit.

Measured 2026-10-01 on this install. The last tool call of the session was
16:34:51; everything after it was question-and-answer, so the edit stayed on
from there. At 16:40:20 the reasoning channel collapsed into a single repeated
character, and all seven repetition retries went out on the edited weights —
the off was unreachable in precisely the failure mode that needed it, because a
collapsed turn cannot call a tool. The turn ended on REPETITION_GIVEUP_NOTE,
the next user message did the same, and ``GET /ab`` still read
``{"scale":1.0,"mode":"abliterated"}`` an hour later: the exact "switch left on
for hours" this module was written to avoid. Compare the same day before the
switch shipped — 13 collapses between 10:35 and 10:40, 13 escapes.

``on`` and ``off`` exist to pin the state for A/B measurement, not for daily
use.
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
# switch every request that needs the edit gets refused. A server with no /ab
# route (or an llama.cpp backend) simply fails the push and is left alone, so
# this default costs nothing where the feature does not exist. Override with
# CODEMASON_AB=off|on|auto for a run.
_mode = _ALIASES.get((os.environ.get('CODEMASON_AB') or '').strip().lower(),
                     (os.environ.get('CODEMASON_AB') or '').strip().lower())
if _mode not in MODES:
    _mode = AUTO
# Last scale we pushed, so the common case (already there) costs no request.
_pushed = None
# Set when a user turn starts, spent by that turn's first request. A latch
# rather than a push: arming costs no request, and nothing can be left on by
# arming alone, because only begin_request() turns the switch on and it hands
# the caller the obligation to turn it back off.
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


def _endpoint():
    cfg = get_config()
    base = (cfg.model_base_url or '').strip().rstrip('/')
    if not base:
        return None
    return base + '/ab', (cfg.model_api_key or '')


async def _push(scale):
    """Push a scale, skipping the request when the server already has it.

    Returns None on success or a message on failure. A failure is reported to
    the caller but never blocks the turn: a server without the plugin (or an
    llama.cpp backend) has no /ab, and that is a normal configuration.
    """
    global _pushed
    with _lock:
        if _pushed == scale:
            return None
    target = _endpoint()
    if target is None:
        return 'model_base_url is not set'
    url, key = target
    try:
        import httpx
    except ImportError:
        return 'httpx is not installed'
    headers = {'Authorization': f'Bearer {key}'} if key else {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json={'scale': scale}, headers=headers)
        if resp.status_code >= 400:
            return f'{resp.status_code}: {resp.text.strip()}'
    except Exception as exc:  # network, DNS, refused — all non-fatal here
        return repr(exc)
    with _lock:
        _pushed = scale
    return None


async def set_mode(new_mode):
    """Set the mode and bring the server in line with it right away."""
    global _mode
    with _lock:
        _mode = new_mode
    # Leaving auto or turning it off must not strand the server on.
    return await _push(1.0 if new_mode == ON else 0.0)


async def on_user_turn():
    """A user turn is starting: arm the edit for that turn's FIRST request.

    Nothing is pushed here. The push belongs to the request, so that the off can
    be paired with it — see begin_request() / end_request().
    """
    global _armed
    if mode() == AUTO:
        with _lock:
            _armed = True


async def begin_request():
    """About to send a request: turn the edit on iff it is the turn's first.

    Returns whether it was turned on. The caller MUST hand that back to
    end_request(), which is what makes "on" impossible to leak: there is no path
    that turns it on and then waits for some later event to turn it off.
    """
    global _armed
    if mode() != AUTO:
        return False
    with _lock:
        armed, _armed = _armed, False
    if not armed:
        return False
    await _push(1.0)
    return True


async def end_request(turned_on):
    """That request has answered — off, whatever it answered with.

    A tool call, a sentence, an empty turn, an abort and a generation the
    degenerate guard cut all reach here. That is the point: the previous version
    turned off only on a tool call, which the collapsed turns of 2026-10-01
    never reached (see the module docstring).
    """
    if turned_on:
        await _push(0.0)


async def query():
    """What the server currently holds: (scale, consistent), or None.

    Used by ``/ab`` with no argument, so the reported state is the server's
    rather than our idea of it.
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
        body = resp.json()
        return float(body['scale']), bool(body.get('consistent', True))
    except Exception:
        return None
