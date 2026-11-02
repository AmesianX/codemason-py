#!/usr/bin/env python3
"""Closing an abortable stream must not leave its pull unread.

`abortable` races each pull against the abort signal, which means a pull can be
in flight at any suspension point — including the race itself. A consumer that
closes this generator throws GeneratorExit at whichever await is current, and
when that is the race, the pending pull used to survive the finally untouched:
the event loop kept the task until it finished with whatever the half-torn-down
stream raised, and nobody ever read the result, so asyncio printed "Task
exception was never retrieved" and a full httpx traceback.

Which is what stopping a sub-agent looked like. TaskStop did exactly the right
thing — status killed, partial result delivered, notification enqueued — and
then dumped a stack trace on the terminal, so a clean stop read as a crash.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == 'win32':
    # Our own case names carry em dashes and Korean, and the console
    # encoder here is cp949: the first such line would raise
    # UnicodeEncodeError inside the reporter and stop the run mid-suite,
    # blaming whatever case it was printing. Pinned inline, importing
    # nothing -- several of these files must set CODEMASON_CONFIG_DIR
    # before anything under codemason/ is imported.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


from codemason.utils.generators import abortable


class StreamBroke(Exception):
    """Stands in for httpx.ReadError on a connection torn down mid-read."""


async def run_case(close_while_pending: bool):
    """Returns (unretrieved, cleanup_ran).

    `unretrieved` is what asyncio would have printed: an exception left on a
    task nobody read.
    """
    unretrieved = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, ctx: unretrieved.append(ctx.get('message', ''))
    )

    cleanup_ran = []
    started = asyncio.Event()

    async def source():
        try:
            yield 'first'
            started.set()
            # Parked here, exactly like a streaming HTTP read. When the
            # connection is torn down this is what surfaces.
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise StreamBroke('connection reset') from None
        finally:
            cleanup_ran.append(True)

    signal = asyncio.Event()
    wrapped = abortable(source(), signal)
    got = await wrapped.__anext__()
    if close_while_pending:
        # Let the next pull actually start, so the close lands on the race and
        # not between iterations.
        pull = asyncio.ensure_future(wrapped.__anext__())
        await started.wait()
        await asyncio.sleep(0.05)
        pull.cancel()
        try:
            await pull
        except BaseException:
            pass
    await wrapped.aclose()

    # Give the loop every chance to report an unretrieved task, and force the
    # collection that would surface one.
    import gc
    for _ in range(10):
        await asyncio.sleep(0.02)
    gc.collect()
    for _ in range(10):
        await asyncio.sleep(0.02)

    loop.set_exception_handler(previous)
    return got, unretrieved, cleanup_ran


async def main():
    got, unretrieved, cleanup_ran = await run_case(close_while_pending=True)
    check('the stream still delivers before it is closed', got == 'first', got)
    check('closing it mid-pull leaves nothing unretrieved',
          unretrieved == [], unretrieved)
    check('...and still runs the source\'s own cleanup',
          cleanup_ran == [True], cleanup_ran)

    got, unretrieved, cleanup_ran = await run_case(close_while_pending=False)
    check('closing it between pulls is clean too', unretrieved == [], unretrieved)
    check('...with the cleanup run', cleanup_ran == [True], cleanup_ran)

    # And the ordinary paths are untouched.
    async def three():
        for item in ('a', 'b', 'c'):
            yield item

    signal = asyncio.Event()
    check('a stream that ends on its own yields everything',
          [x async for x in abortable(three(), signal)] == ['a', 'b', 'c'])

    async def two_then_park():
        yield 'a'
        yield 'b'
        await asyncio.sleep(3600)

    signal = asyncio.Event()
    seen = []
    async for item in abortable(two_then_park(), signal, grace_seconds=0.05):
        seen.append(item)
        if len(seen) == 2:
            signal.set()
    check('an abort stops it after the item that came with the abort',
          seen == ['a', 'b'], seen)

    check('no signal degrades to a plain passthrough',
          [x async for x in abortable(three(), None)] == ['a', 'b', 'c'])

    # And the symptom itself: nothing on stderr. The in-process checks above
    # cannot see this — asyncio and the asyncgen finalizer write straight to
    # sys.stderr, sometimes at collection time, after any redirect has been
    # restored. A child process is the only honest way to assert "a stop prints
    # nothing", which is what the user actually saw go wrong.
    import subprocess
    child = subprocess.run(
        [sys.executable, '-c', _CHILD],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    check('closing it mid-pull prints nothing on stderr',
          child.stderr.strip() == '', child.stderr.strip()[:400])
    check('...and the child still got its item and ran the cleanup',
          child.stdout.strip() == 'first cleanup', child.stdout.strip())

    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


# Run in a child so its stderr is this process's to inspect. Deliberately a
# separate, minimal reproduction rather than an import of the helpers above: the
# point is what reaches a terminal, and anything this file does to the event
# loop would change that.
_CHILD = '''
import asyncio, gc, sys
sys.path.insert(0, ".")
from codemason.utils.generators import abortable

async def main():
    cleaned = []
    started = asyncio.Event()

    async def source():
        try:
            yield "first"
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise RuntimeError("connection reset") from None
        finally:
            cleaned.append(True)

    wrapped = abortable(source(), asyncio.Event())
    got = await wrapped.__anext__()
    pull = asyncio.ensure_future(wrapped.__anext__())
    await started.wait()
    await asyncio.sleep(0.05)
    pull.cancel()
    try:
        await pull
    except BaseException:
        pass
    await wrapped.aclose()
    for _ in range(10):
        await asyncio.sleep(0.02)
    gc.collect()
    for _ in range(10):
        await asyncio.sleep(0.02)
    print(got, "cleanup" if cleaned else "no-cleanup")

asyncio.run(main())
'''


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
