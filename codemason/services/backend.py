"""
Backend gate — routes model queries to the right service implementation.

    llama.cpp server  ->  services/llamacpp.py   (original, unchanged)
    vLLM server       ->  services/vllm.py        (continuous_usage_stats etc.)

Selection order:
  1. Explicit: config.backend / env CODEMASON_BACKEND = 'vllm' | 'llamacpp'.
     Set this when you configure the endpoint — you already know whether the URL
     is a vLLM or a llama-server.
  2. Fallback auto-detect (config.backend == 'auto', the default): probe the
     server once. /slots is llama.cpp-only; vLLM 404s it. The result is cached;
     a connection error is NOT cached (assume vLLM for that call, retry later).

Callers import query_model from HERE.
"""

import asyncio
import atexit
import contextlib
import os
import random
import time

from ..utils.i18n import _
from typing import Optional

_detected: Optional[str] = None  # cached auto-detect result ('vllm' | 'llamacpp')


def _normalize(value: str) -> str:
    v = (value or '').strip().lower()
    if v in ('llamacpp', 'llama.cpp', 'llama-server', 'llama_cpp', 'llama'):
        return 'llamacpp'
    if v == 'vllm':
        return 'vllm'
    return 'auto'


def _auto_detect(cfg) -> str:
    """Probe the configured server: /slots (200 + list) => llama.cpp, else vLLM."""
    global _detected
    if _detected is not None:
        return _detected

    import httpx
    headers = {}
    if cfg.model_api_key:
        headers['Authorization'] = f'Bearer {cfg.model_api_key}'
    try:
        resp = httpx.get(f"{cfg.model_base_url}/slots", headers=headers, timeout=5.0)
        _detected = 'llamacpp' if (resp.status_code == 200 and isinstance(resp.json(), list)) else 'vllm'
        return _detected
    except Exception:
        # Server unreachable — assume vLLM for this call but don't cache the
        # guess, so the next call re-probes once the server is up.
        return 'vllm'


def get_backend() -> str:
    """Resolve the active backend: 'vllm' or 'llamacpp'."""
    from ..utils.config import get_config
    cfg = get_config()
    choice = _normalize(getattr(cfg, 'backend', 'auto'))
    if choice in ('vllm', 'llamacpp'):
        return choice
    return _auto_detect(cfg)


async def query_model(*args, **kwargs):
    """Neutral model-query entry point. Every call site imports this; it
    dispatches to the backend-specific shim (query_llamacpp / query_vllm) based
    on the detected/configured server."""
    if get_backend() == 'llamacpp':
        from .llamacpp import query_llamacpp as _impl
    else:
        from .vllm import query_vllm as _impl
    return await _impl(*args, **kwargs)


# --- Shared HTTP client + connect retry --------------------------------------
# Both service backends stream chat completions over httpx. They used to open a
# NEW httpx.AsyncClient per request, so a fast agentic loop opened a fresh
# TCP+TLS handshake for every turn — and the more handshakes, the higher the
# chance one transiently fails (SYN loss / TLS reset / momentary path blip),
# surfacing as ConnectError('') and killing the turn (no retry). These helpers
# fix both halves: a persistent keep-alive client so a rapid loop REUSES one
# connection (few handshakes), and a connect-phase retry so a lone blip is
# absorbed instead of fatal. Env-tunable: CODEMASON_MODEL_CONNECT_RETRIES /
# CODEMASON_MODEL_RETRY_BACKOFF_BASE / CODEMASON_MODEL_RETRY_BACKOFF_CAP
# (the legacy LLAMACPP_* names for these still work).

_shared_client = None  # httpx.AsyncClient, lazily created, reused for the session


def _http_retry_budget_secs() -> float:
    """Total time to keep retrying a transient failure, in seconds.

    The count-based schedule alone gave up after a few seconds, which is right
    for a person watching a dead server and wrong for the way this is mostly
    run: started and left for hours. A server restart, a gateway coming back, a
    reconnecting VPN — all take minutes, not seconds, and losing hours of
    unattended work to a blip that healed 30 seconds later is the worst possible
    trade.

    The deadline is the last of utils.escalation's retry checkpoints — two hours.
    That is not a guess at how long outages last, it is the cheaper way to be
    wrong: calling a temporary outage permanent throws away everything the run
    has done, while calling a permanent one temporary idles a machine nobody is
    watching. The early attempts are still fast (half a second, then one, then
    two), so a blip that clears immediately costs almost nothing.
    Env: CODEMASON_MODEL_RETRY_BUDGET. 0 restores count-only behaviour.
    """
    from ..utils.escalation import retry_deadline_secs
    default = retry_deadline_secs()
    v = os.getenv('CODEMASON_MODEL_RETRY_BUDGET') or os.getenv('LLAMACPP_RETRY_BUDGET')
    try:
        f = float(v) if v is not None else default
    except (TypeError, ValueError):
        return default
    return f if f >= 0 else default


def _is_retryable_status(code: int) -> bool:
    """Statuses worth trying again: the server is there but not ready.

    Deliberately NOT 500 — a deterministic application error would just burn the
    whole budget re-asking a question that has already been answered. These four
    are what a restarting backend, a gateway in front of it, or a rate limiter
    actually emit while recovering.
    """
    return code in (429, 502, 503, 504)


def _retry_delay_secs(attempt: int, base: float, cap: float) -> float:
    """Backoff for `attempt` (0-based), before jitter."""
    return min(cap, base * (2 ** attempt))


def _http_retry_cfg():
    def _raw(new, old):
        v = os.getenv(new)
        return v if v is not None else os.getenv(old)

    def _i(new, old, default):
        v = _raw(new, old)
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def _f(new, old, default):
        v = _raw(new, old)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return (_i('CODEMASON_MODEL_CONNECT_RETRIES', 'LLAMACPP_CONNECT_RETRIES', 3),
            _f('CODEMASON_MODEL_RETRY_BACKOFF_BASE', 'LLAMACPP_RETRY_BACKOFF_BASE', 0.5),
            _f('CODEMASON_MODEL_RETRY_BACKOFF_CAP', 'LLAMACPP_RETRY_BACKOFF_CAP', 30.0))


def get_shared_client():
    """Session-lived httpx.AsyncClient with a keep-alive pool. Reused across all
    LLM calls so rapid turns share connections instead of re-handshaking each."""
    global _shared_client
    import httpx
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=32,
                max_connections=64,
                # Must stay BELOW the proxy's keep-alive timeout, with margin.
                # This was 90.0 and the nginx in front of our own server is the
                # nginx default, also 90s — a dead heat. When the server's timer
                # won, the pool handed out a socket the server had already
                # closed and the next request died on a reset BEFORE it was
                # written, so it never reached the server: the access log showed
                # 88/88 requests 200 and not one 499 or 5xx, while the client
                # raised a bare transport error. Verified 2026-10-14 against our
                # own gateway: idle 30s reuses fine, idle 95s and 120s both come
                # back RemoteDisconnected. nginx sends no `Keep-Alive: timeout=`
                # header, so this cannot be negotiated — it has to be a number
                # we pick, and it has to be the smaller one. Expiring first is
                # what makes the race unreachable; the exact value only sets how
                # much slack absorbs clock skew and a slow write.
                keepalive_expiry=60.0,
            ),
        )
    return _shared_client


async def aclose_shared_client() -> None:
    """Close the shared client's keep-alive pool. Safe to call from an app
    shutdown path that still has a running loop; also invoked best-effort at
    interpreter exit so the process doesn't leave the pool open (unclosed
    client/transport warning)."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        try:
            await _shared_client.aclose()
        except Exception:
            pass


def _atexit_close_shared_client() -> None:
    if _shared_client is None or _shared_client.is_closed:
        return
    try:
        asyncio.run(aclose_shared_client())
    except Exception:
        pass  # loop already torn down at exit — best-effort only


atexit.register(_atexit_close_shared_client)


@contextlib.asynccontextmanager
async def shared_client_cm():
    """`async with` handle over the shared client that does NOT close it on exit
    (so the keep-alive pool survives across turns). Drop-in for the old
    `async with httpx.AsyncClient(...) as client:` line — body indentation stays."""
    yield get_shared_client()


class RetryAborted(RuntimeError):
    """The run was stopped while the connect-retry ladder was waiting.

    Separate from the budget-exhausted RuntimeError because the two mean
    opposite things: that one is "the server never came back", this one is "the
    person said stop". Callers that only catch Exception still behave — query.py
    checks the abort event right after the request returns and prints the normal
    interrupt line — but anything that wants to tell them apart now can.
    """


async def _wait_or_abort(seconds: float, signal) -> bool:
    """Wait `seconds`, returning True as soon as `signal` is set instead.

    The plain asyncio.sleep() this replaces was the whole reason a lost
    connection could not be stopped: Esc and Ctrl+C only SET the turn's abort
    event (cli.RequestRouter.interrupt), and nothing in the retry ladder ever
    looked at it. The stream BODY has been racing against the same event since
    the prefill fix (utils.generators.abortable), but the connect phase in front
    of it was never covered — so an outage parked the session in a two-hour
    ladder whose only exit was killing the process.
    """
    if signal is None:
        await asyncio.sleep(seconds)
        return False
    if signal.is_set():
        return True
    if seconds <= 0:
        return False
    try:
        await asyncio.wait_for(signal.wait(), timeout=seconds)
    except (asyncio.TimeoutError, TimeoutError):
        return False
    return True


@contextlib.asynccontextmanager
async def retrying_stream(client, method, url, timeout, stream_kwargs,
                          signal=None):
    """Open ``client.stream()`` retrying ONLY the connect / response-header phase
    on transient network errors (ConnectError, timeouts, resets). Once the
    response is yielded, streaming errors are NOT retried — content may already
    have been emitted, so a mid-stream failure propagates to the caller. This is
    what lets a fast loop survive a momentary connect blip instead of dying on the
    first ConnectError. Drop-in for the old `async with client.stream(...)` line.

    `signal` is the turn's abort event (Esc / Ctrl+C). It is checked after every
    failed attempt and waited on INSTEAD of sleeping, so a stop lands within the
    event loop's own latency rather than at the end of the current backoff."""
    import httpx
    retryable = (
        httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout,
        httpx.ReadError, httpx.ReadTimeout, httpx.WriteError,
        httpx.RemoteProtocolError,
    )
    retries, base, cap = _http_retry_cfg()
    min_attempts = max(1, retries)
    budget = _http_retry_budget_secs()
    started = time.monotonic()
    cm = None
    response = None
    attempt = 0
    announced = False
    reported = 0
    last_err = ''
    while True:
        cm = client.stream(method, url, timeout=timeout, **stream_kwargs)
        try:
            response = await cm.__aenter__()
            # A response whose STATUS says "not ready" is as transient as a
            # refused connection: a restarting backend, or the gateway in front
            # of it, answers 502/503 for exactly as long as it takes to come
            # back. The body has not been read yet, so retrying is still clean.
            if not _is_retryable_status(response.status_code):
                if announced:
                    from ..utils.log import log_error
                    log_error(
                        f'model server recovered after '
                        f'{int(time.monotonic() - started)}s / {attempt} attempts',
                        'backend: retry recovered')
                break
            last_err = f'HTTP {response.status_code}'
            await cm.__aexit__(None, None, None)
            cm = None
        except retryable as e:
            # __aenter__ failed before any body was read → safe to retry cleanly.
            last_err = type(e).__name__

        attempt += 1
        elapsed = time.monotonic() - started
        # Stopped by the user. Nothing below is worth doing: no checkpoint line,
        # no backoff, no reconnect. Logged like the other two outcomes so a
        # post-mortem can tell "someone stopped it" from "it gave up" and from
        # "it is still going".
        if signal is not None and signal.is_set():
            from ..utils.log import log_error
            log_error(
                f'model server retry stopped by the user ({last_err}) after '
                f'{int(elapsed)}s / {attempt} attempts',
                'backend: retry aborted')
            raise RetryAborted(
                f'stopped while retrying the model server after '
                f'{int(elapsed)}s: {last_err}')
        # Keep going while either the count floor or the time budget still
        # allows it — the count covers a blip on a fast link, the budget covers
        # an outage measured in minutes.
        if attempt >= min_attempts and elapsed >= budget:
            # The retry wait is announced with print(), i.e. to the TERMINAL
            # only. Scrollback is gone by morning and a post-mortem then has no
            # trace of a multi-minute wait at all — it looks like the process
            # simply stopped (no socket, every thread parked, nothing logged).
            # The outcome therefore goes to the error log too.
            from ..utils.log import log_error
            log_error(
                f'model server never recovered ({last_err}) — gave up after '
                f'{int(elapsed)}s / {attempt} attempts',
                'backend: retry budget exhausted')
            raise RuntimeError(f'model server not responding after '
                               f'{int(elapsed)}s: {last_err}')
        delay = _retry_delay_secs(attempt - 1, base, cap)
        wait = delay + random.uniform(0, delay)
        # Never sleep past the budget once the count floor is met.
        if attempt >= min_attempts:
            wait = min(wait, max(0.0, budget - elapsed))
        # Say it once at the start, then again at each checkpoint. A silent
        # multi-hour wait is indistinguishable from a hang, and the checkpoints
        # widen (5, 10, 15, 20, 30, 60, 90 min) so the reporting is dense while
        # someone might still be watching and sparse long after they have gone.
        if budget > 0:
            from ..utils.escalation import (
                minutes_for_message, retry_checkpoints_passed, retry_minutes_left,
            )
            crossed = retry_checkpoints_passed(elapsed)
            from ..utils.log import log_error
            if not announced:
                announced = True
                reported = crossed
                print('  \u23bf  ' + _(
                    'Model server not responding ({err}) — retrying for up to '
                    '{mins} min. Press Esc or Ctrl+C to stop.'
                ).format(err=last_err[:80],
                         mins=minutes_for_message(budget)))
                # Also on disk: see the note at the budget-exhausted raise.
                log_error(
                    f'model server not responding ({last_err}) — retrying for up '
                    f'to {int(budget)}s',
                    'backend: retry started')
            elif crossed > reported:
                reported = crossed
                print('  \u23bf  ' + _(
                    'Still no response — retrying ({mins} min left). '
                    'Press Esc or Ctrl+C to stop.'
                ).format(mins=retry_minutes_left(elapsed)))
                log_error(
                    f'still no response from the model server ({last_err}) — '
                    f'{int(elapsed)}s elapsed, {attempt} attempts',
                    'backend: retry continuing')
        if await _wait_or_abort(wait, signal):
            from ..utils.log import log_error
            elapsed = time.monotonic() - started
            log_error(
                f'model server retry stopped by the user ({last_err}) after '
                f'{int(elapsed)}s / {attempt} attempts',
                'backend: retry aborted')
            raise RetryAborted(
                f'stopped while retrying the model server after '
                f'{int(elapsed)}s: {last_err}')
    try:
        yield response
    finally:
        await cm.__aexit__(None, None, None)
