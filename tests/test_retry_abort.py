#!/usr/bin/env python3
"""Esc / Ctrl+C must end the connect-retry ladder, not wait it out.

When the model server goes away, services.backend.retrying_stream keeps trying
for the last checkpoint on the escalation ladder — two hours. That part is
deliberate: an unattended run should ride out a restart rather than throw away
everything it has done. What was not deliberate is that the ladder could not be
stopped. Esc and Ctrl+C only SET the turn's abort event (cli.RequestRouter.
interrupt); the stream BODY has raced that event since the prefill fix
(utils.generators.abortable), but the connect phase in front of it was never
handed the event at all. So a dropped connection parked the session for two
hours and the only way out was killing the process.

These cases pin the way out: the wait is raced against the event, the ladder
stops at once, and a caller that passes no event behaves exactly as before.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == 'win32':
    # Our own case names carry em dashes and Korean, and the console
    # encoder here is cp949: the first such line would raise
    # UnicodeEncodeError inside the reporter and stop the run mid-suite,
    # blaming whatever case it was printing.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    async def aclose(self):
        pass


class _FakeStream:
    """One client.stream() call: fails for the first `fail_times` attempts."""

    def __init__(self, owner):
        self._owner = owner

    async def __aenter__(self):
        import httpx
        self._owner.attempts += 1
        if self._owner.attempts <= self._owner.fail_times:
            raise httpx.ConnectError('')
        return _FakeResponse(200)

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.attempts = 0

    def stream(self, method, url, timeout=None, **kwargs):
        return _FakeStream(self)


async def main():
    from codemason.services import backend
    from codemason.services.backend import (
        RetryAborted, _http_retry_cfg, _retry_delay_secs, _wait_or_abort,
        retrying_stream,
    )

    # --- the helper the whole fix rests on ---------------------------------
    started = time.monotonic()
    aborted = await _wait_or_abort(0.05, None)
    check('with no event, the wait is an ordinary sleep',
          not aborted and time.monotonic() - started >= 0.04)

    evt = asyncio.Event()
    evt.set()
    started = time.monotonic()
    aborted = await _wait_or_abort(30.0, evt)
    check('an event already set does not wait at all',
          aborted and time.monotonic() - started < 0.1)

    evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, evt.set)
    started = time.monotonic()
    aborted = await _wait_or_abort(30.0, evt)
    elapsed = time.monotonic() - started
    check('an event set DURING the wait cuts it short',
          aborted and elapsed < 0.5, f'took {elapsed:.2f}s')

    # --- the ladder itself --------------------------------------------------
    # A server that never comes back, so the loop spends all its time in the
    # backoff — exactly where it used to be deaf.
    retries, base, cap = _http_retry_cfg()
    first_backoff = _retry_delay_secs(0, base, cap)

    client = _FakeClient(fail_times=10 ** 6)
    signal = asyncio.Event()
    loop.call_later(0.05, signal.set)
    started = time.monotonic()
    raised = None
    try:
        async with retrying_stream(client, 'POST', 'http://x/', 1.0, {},
                                   signal=signal):
            pass
    except BaseException as e:   # noqa: BLE001 - the type IS the assertion
        raised = e
    elapsed = time.monotonic() - started

    check('stopping the run ends the ladder with RetryAborted',
          isinstance(raised, RetryAborted), f'got {raised!r}')
    # The count floor alone runs past the first backoff before the time budget
    # even applies, so coming back inside it means the stop was ACTED ON rather
    # than waited out. Derived from the constants so this keeps meaning
    # something if the backoff changes.
    check('...immediately, not at the end of the current backoff',
          elapsed < max(0.5, first_backoff),
          f'took {elapsed:.2f}s, first backoff is {first_backoff:.2f}s')
    check('...and it is an Exception, so the caller\'s handler still catches it',
          isinstance(raised, Exception))

    # --- no regression for the callers that pass nothing --------------------
    client = _FakeClient(fail_times=1)
    async with retrying_stream(client, 'POST', 'http://x/', 1.0, {}) as resp:
        check('with no event a blip is still absorbed',
              resp.status_code == 200 and client.attempts == 2,
              f'attempts={client.attempts}')

    # --- giving up is still a different outcome from being stopped ----------
    os.environ['CODEMASON_MODEL_RETRY_BUDGET'] = '0'
    try:
        client = _FakeClient(fail_times=10 ** 6)
        raised = None
        try:
            async with retrying_stream(client, 'POST', 'http://x/', 1.0, {},
                                       signal=asyncio.Event()):
                pass
        except BaseException as e:   # noqa: BLE001
            raised = e
        check('a server that never came back still raises the plain RuntimeError',
              isinstance(raised, RuntimeError) and not isinstance(raised, RetryAborted),
              f'got {raised!r}')
    finally:
        os.environ.pop('CODEMASON_MODEL_RETRY_BUDGET', None)

    # --- the wiring: the turn's event actually reaches the ladder -----------
    src = Path(backend.__file__).with_name('_openai_common.py').read_text(encoding='utf-8')
    check('_openai_common hands the turn\'s abort event to the ladder',
          'signal=signal,' in src.split('retrying_stream(')[1].split(') as response:')[0])

    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
