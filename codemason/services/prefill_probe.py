"""What the server is actually about to re-read, asked of the server.

Before the first output token arrives there is nothing on screen but a spinner
and a clock, and a warm request (answer in ~2s) looks exactly like one whose
cached prefix broke. On this deployment the difference is 55 seconds for a 73k
conversation and 175 for a 233k one, so "is it prefilling or is it stuck" is
the question every long wait raises and none of them answer.

vLLM already knows. Two counters move the moment a request is SCHEDULED,
seconds before any token comes back:

    vllm:prefix_cache_queries_total   tokens this prompt asked about
    vllm:prefix_cache_hits_total      tokens it could serve from cache

Their difference is the work that has to be computed — the server's own
number, not a guess from our side of the wire. Measured 2026-09-07 on a
deliberately cold 59,384-token prompt: queries +59,384, hits +0, scheduled
4.7s in, answered at 44.8s. Exactly the shape a stalled turn has.

Caveat, and the reason this only ever informs a display: the counters are
global. With another request in flight the delta belongs to whichever was
scheduled, so the number can be wrong when the server is shared. It is a
diagnostic, never an input to behaviour.
"""
import asyncio
import threading

from ..utils import tokens as _tokens
from ..utils.config import get_config
from ..utils.log import debug_log, log_error

_PROBING = threading.Lock()
PREFILL_TOKENS_PER_SEC = 1335.0  # measured on this deployment, 2026-09-07


def _metrics_target():
    cfg = get_config()
    base = (cfg.model_base_url or '').strip().rstrip('/')
    if not base:
        return None
    return base + '/metrics', (cfg.model_api_key or '')


async def _read_counters(url: str, key: str):
    """(queried, hits) cumulative, or None when the server has no such
    counters (llama.cpp) or is unreachable."""
    try:
        import httpx
    except ImportError:
        return None
    headers = {'Authorization': 'Bearer ' + key} if key else {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            body = (await c.get(url, headers=headers)).text
    except Exception:
        return None
    q = h = None
    for line in body.splitlines():
        if line.startswith('vllm:prefix_cache_queries_total'):
            q = line.rsplit(' ', 1)[-1]
        elif line.startswith('vllm:prefix_cache_hits_total'):
            h = line.rsplit(' ', 1)[-1]
    try:
        return int(float(q)), int(float(h))
    except (TypeError, ValueError):
        return None


async def _probe():
    target = _metrics_target()
    if not target:
        return
    url, key = target
    base = await _read_counters(url, key)
    if base is None:
        return  # no counters: the client-side estimate stands
    q0, h0 = base
    # Scheduling lands within a few seconds; past that the request is queued
    # behind something else and the delta would not be ours.
    for _ in range(20):
        await asyncio.sleep(0.7)
        if _tokens.output_started():
            return
        cur = await _read_counters(url, key)
        if cur is None:
            return
        q, h = cur
        if q > q0:
            queried, cached = q - q0, h - h0
            compute = queried - cached
            _tokens.set_pending_prefill_measured(compute, cached)
            hit = (100.0 * cached / queried) if queried else 100.0
            # A cold or badly-degraded prefix is the single most expensive
            # thing that happens without saying so. Record it where an
            # unattended run leaves a trace: at ~1,335 tok/s a 73k re-read is
            # 55 seconds of silence that otherwise looks like a hang, and
            # afterwards nothing in the transcript says it happened.
            if compute > 20000 and hit < 50.0:
                log_error(
                    'cold prefill: %d of %d prompt tokens must be recomputed '
                    '(%.1f%% served from cache) — about %.0fs before the first token'
                    % (compute, queried, hit, compute / PREFILL_TOKENS_PER_SEC),
                    'prefill_probe: vllm /metrics',
                )
            debug_log('PREFILL', {'queried': queried, 'cached': cached,
                                  'compute': compute, 'hit_percent': hit})
            return


def probe_next_request() -> None:
    """Watch for THIS request being scheduled and publish how much of it the
    server must compute. Returns immediately."""
    if not _PROBING.acquire(blocking=False):
        return  # one at a time: concurrent probes read each other's deltas

    async def run():
        try:
            await _probe()
        finally:
            _PROBING.release()

    try:
        asyncio.get_running_loop().create_task(run())
    except RuntimeError:
        _PROBING.release()
