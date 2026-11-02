"""
Generator utilities for CodeMason
Equivalent to src/utils/generators.ts
"""

import asyncio
from typing import Any, AsyncGenerator, List, Optional, TypeVar, Set
from dataclasses import dataclass


T = TypeVar('T')


# Grace window (seconds) a tool gets to finish cooperatively after abort fires,
# before its in-flight pull is force-cancelled. Cooperative tools (bash/powershell
# watch abort_controller and SIGKILL their subprocess, then return an "interrupted"
# result); the window lets that clean teardown run instead of injecting
# CancelledError mid-await — which could cancel the tool's own kill-monitor task
# before it fires and orphan the subprocess. Non-cooperative tools (a bare httpx
# fetch) never return, so they hit the force-cancel after the window.
_ABORT_GRACE_SECONDS = 0.5


async def abortable(
    gen: AsyncGenerator[T, None],
    signal: Optional['asyncio.Event'],
    grace_seconds: float = _ABORT_GRACE_SECONDS,
) -> AsyncGenerator[T, None]:
    """Yield from ``gen``, but stop when ``signal`` (an asyncio.Event) is set —
    force-cancelling whatever await the generator is stuck on if it doesn't finish
    cooperatively within ``grace_seconds``.

    WHY: the abort_controller is otherwise only polled BETWEEN tools (query.py).
    A tool sitting in one long await — an httpx fetch, a blocking to_thread SERP,
    a sleep loop — therefore runs to completion before the interrupt is noticed,
    so ESC "does nothing" mid-tool (the WebFetch-doesn't-stop bug). Racing each
    pull against ``signal.wait()`` catches the abort immediately; force-cancelling
    the pending pull raises CancelledError inside the generator's current await
    (e.g. httpx.aiter_bytes), tearing the connection down at once; for a to_thread
    await the coroutine stops waiting immediately while the orphan thread drains
    harmlessly. Applied once at the single tool.call() dispatch point, so EVERY
    tool becomes abort-aware uniformly — no per-tool polling to add or forget.

    The grace window before force-cancel is what keeps cooperative tools clean:
    bash/powershell already kill their subprocess on abort and return; cancelling
    them mid-await could pre-empt that teardown and orphan the process. So we wait
    briefly for a self-terminated result and only force the issue if the tool is
    genuinely wedged. aclose() in the finally still runs the generator's own
    cleanup. signal=None degrades to a plain passthrough, so headless / test
    callers with no abort controller are unaffected.
    """
    if signal is None:
        async for item in gen:
            yield item
        return

    ait = gen.__aiter__()
    abort_wait = asyncio.ensure_future(signal.wait())
    # The pull in flight, visible to the finally. Every path through the loop
    # below retrieves it — but a consumer that aclose()s THIS generator throws
    # GeneratorExit at whichever await is current, and that can be the race
    # below, with `nxt` still pending. The finally then ran without touching it:
    # the event loop kept the task until it finished with whatever the
    # half-torn-down stream raised, and nobody ever read the result, so asyncio
    # printed "Task exception was never retrieved" and a full httpx traceback.
    #
    # Which is what stopping a sub-agent looked like. TaskStop did exactly the
    # right thing — status killed, partial result delivered, notification
    # enqueued — and then dumped a stack trace on the terminal, so a clean stop
    # read as a crash.
    nxt = None
    try:
        while True:
            nxt = asyncio.ensure_future(ait.__anext__())
            done, _ = await asyncio.wait(
                {nxt, abort_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if nxt in done:
                try:
                    item = nxt.result()
                except StopAsyncIteration:
                    return
                yield item
                if signal.is_set():
                    return  # deliver the item that completed alongside abort, then stop
                continue

            # Abort fired with a pull in flight. Give the tool grace_seconds to
            # finish cooperatively (kill its subprocess and return), then force it.
            g_done, _ = await asyncio.wait({nxt}, timeout=grace_seconds)
            if nxt in g_done:
                try:
                    item = nxt.result()
                except StopAsyncIteration:
                    return
                except BaseException:
                    return
                yield item  # the tool's own "interrupted" result
                return
            nxt.cancel()
            try:
                await nxt
            except BaseException:
                pass
            return
    finally:
        abort_wait.cancel()
        # Cancel the pull BEFORE aclose(), so the generator is not mid-__anext__
        # when it is closed, and retrieve whatever it ends up carrying — a
        # cancellation, or the stream error the teardown produced. Both are
        # expected here; what is not acceptable is leaving either unread.
        if nxt is not None and not nxt.done():
            nxt.cancel()
        if nxt is not None:
            try:
                await nxt
            except BaseException:
                pass
        try:
            await gen.aclose()
        except BaseException:
            # Tearing down a stream that was force-cancelled is not a failure —
            # httpx maps the broken read to its own exception on the way out, and
            # anything raised here escapes to the event loop, where prompt_toolkit
            # prints "Unhandled exception" and stops the REPL for a keypress. The
            # abort already did what it was asked to do.
            pass


@dataclass
class QueuedGenerator:
    """Queued generator state. `error` carries an exception raised by the
    generator's __anext__ as a VALUE rather than letting the wrapping task
    complete-with-exception — so a sibling failure never leaves an unretrieved
    task ("Task exception was never retrieved") and propagation order matches
    the TS Promise.race reject."""
    done: bool
    value: Any
    generator: AsyncGenerator
    error: Optional[BaseException] = None


async def all(
    generators: List[AsyncGenerator[T, None]],
    concurrency_cap: int = float('inf'),
) -> AsyncGenerator[T, None]:
    """
    Run all generators concurrently up to a concurrency cap, yielding values as they come in.
    Equivalent to src/utils/generators.ts all()

    The TS original relies on the JS GC to reclaim the promises it abandons on
    early-exit / rejection. Python tasks and async generators are NOT reclaimed
    that way, so this port adds an explicit finally that cancels in-flight tasks
    and aclose()s every generator it started — the legitimate substrate
    equivalent of the GC the original leans on (without it, aborting mid-tool or
    a sibling tool raising would leak orphan tasks + open tool generators that
    still hold subprocess/file state).
    """
    async def get_next(gen: AsyncGenerator) -> QueuedGenerator:
        try:
            value = await gen.__anext__()
            return QueuedGenerator(done=False, value=value, generator=gen)
        except StopAsyncIteration:
            return QueuedGenerator(done=True, value=None, generator=gen)
        except Exception as e:  # NOT BaseException: CancelledError must propagate
            return QueuedGenerator(done=True, value=None, generator=gen, error=e)

    waiting = list(generators)
    pending: Set[asyncio.Task] = set()
    # Generators we've started and not yet seen exhaust — the set we must close.
    started: Set[AsyncGenerator] = set()

    def _start(gen: AsyncGenerator) -> None:
        started.add(gen)
        pending.add(asyncio.ensure_future(get_next(gen)))

    try:
        # Start initial batch up to concurrency cap
        while len(pending) < concurrency_cap and waiting:
            _start(waiting.pop(0))

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                result: QueuedGenerator = task.result()  # never raises (errors carried)

                if result.error is not None:
                    # A tool generator raised — propagate (finally cleans up the
                    # rest). Matches the TS race rejecting.
                    raise result.error

                if not result.done:
                    # Yield the value (unconditionally — a generator may
                    # legitimately yield None, and the serial path forwards it;
                    # dropping None here diverged the two dispatch paths), then
                    # continue with the same generator.
                    yield result.value
                    _start(result.generator)
                else:
                    started.discard(result.generator)  # exhausted, no close needed
                    if waiting:
                        _start(waiting.pop(0))
    finally:
        # Cancel any in-flight pulls first so no generator is mid-__anext__,
        # then close every generator we started (and any never-started ones).
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for gen in [*started, *waiting]:
            try:
                await gen.aclose()
            except Exception:
                pass


