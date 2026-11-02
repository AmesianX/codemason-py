"""Cleanup registry — adapted from utils/cleanupRegistry.ts.

The source keeps one module-level `Set` of cleanup callbacks. `registerCleanup`
adds one and returns the function that removes it again; `runCleanupFunctions`
awaits every registered callback during graceful shutdown. The module is kept
separate from the shutdown path in the source to avoid a circular dependency,
and for the same reason here.

This is the mechanism that makes a task's `unregister_cleanup` handle mean
something: a sub-agent registers "kill me" at spawn, so a process that dies
mid-run still tears the agent down, and the handle is what cancels that
registration once the agent has finished on its own. Without the pair, one of
the two lifecycle bugs is unavoidable — either the agent outlives the process
that owns it, or its cleanup fires after it already completed.

Divergence from the source: its callbacks are async because the registered work
awaits an output-file flush. codemason's agent cleanup is a synchronous state
mutation plus an abort-flag set, so the callbacks are plain callables and the
runner is synchronous — which is also what `atexit` can actually drive. An async
callback is still accepted and driven to completion by `run_cleanup_functions`.
"""

import asyncio
import atexit
import inspect
import threading
from typing import Callable, List

# The registered callbacks (`wr8`). A list, not a set: callables are compared by
# identity here and a list keeps registration order, which makes the teardown
# order reproducible in tests.
_cleanups: List[Callable] = []
_lock = threading.Lock()


def register_cleanup(fn: Callable) -> Callable[[], None]:
    """Register `fn` to run during shutdown; returns its unregister fn."""
    with _lock:
        _cleanups.append(fn)

    def _unregister() -> None:
        with _lock:
            try:
                _cleanups.remove(fn)
            except ValueError:
                # Already unregistered (or already run) — idempotent by design:
                # every terminal transition calls the handle, and so does task
                # removal, so a double call must not raise.
                pass

    return _unregister


def cleanup_function_count() -> int:
    """How many cleanups are registered. No source counterpart — the Set is
    module-private there; this exists so a test can assert the registration was
    released rather than inferring it."""
    with _lock:
        return len(_cleanups)


def run_cleanup_functions() -> None:
    """Run every registered cleanup, draining the registry (runCleanupFunctions).

    Best-effort: a callback that raises must not stop the others, because this
    runs on the way out and each registration belongs to a different subsystem.
    """
    with _lock:
        pending = list(_cleanups)
        _cleanups.clear()
    for fn in pending:
        try:
            result = fn()
            if inspect.isawaitable(result):
                # An async callback registered from a loop that is already gone:
                # drive it to completion on a throwaway loop rather than leaving
                # a never-awaited coroutine behind.
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(_await(result))
                else:
                    asyncio.ensure_future(result)
        except Exception:
            pass


async def _await(awaitable) -> None:
    await awaitable


atexit.register(run_cleanup_functions)
