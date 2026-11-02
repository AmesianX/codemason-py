"""Abort scopes for sub-agents: a parent's stop reaches its children, a child's
does not reach the parent or its siblings.

A sub-agent used to be handed the parent's own asyncio.Event. That was harmless
while one ran at a time — there was nothing else to hit. Running three at once it
is wrong twice:

  - Stopping ONE is impossible. TaskStop on an agent would set the shared event,
    which every sibling and the parent read; `subagent_gate.acquire` returns None
    on it and reports 'Aborted while waiting for a sub-agent slot', so a stop
    aimed at one agent takes down a sibling that was only queued.
  - A child can stop its parent. Anything inside a sub-agent that sets the event
    to end its own turn ends the whole session.

The source gives each agent its own controller (`abortController: P3()` on the
task state) and links it to the outer one the way the bundled SDK does: abort now
if the outer already aborted, otherwise forward the outer's abort inward. One
direction only.

This reads that linkage instead of forwarding it, which needs no persistent task
and no listener: `is_set` is true when this scope was stopped OR any ancestor
was, and `set` touches only this scope. A standing watcher task would be a thing
to cancel on every exit path, which is the leak this subsystem is trying to
avoid; `wait` instead races the two for as long as someone is actually waiting
and cancels both when it returns.
"""
from typing import Any, Optional


class AbortScope:
    """An abort flag that also reports its parent's. `set()` is local."""

    __slots__ = ('_event', '_parent')

    def __init__(self, parent: Optional[Any] = None):
        import asyncio
        self._event = asyncio.Event()
        # Any object with is_set(): an asyncio.Event or another AbortScope.
        self._parent = parent

    def set(self) -> None:
        """Stop this scope and its descendants. The parent is left alone."""
        self._event.set()

    def detach(self) -> None:
        """Stop reading the parent. This scope's own `set()` still stops it.

        For an agent that moves to the background mid-run. The source does the
        same thing by swapping which controller the run uses: a foreground agent
        runs on the turn's controller (so ESC ends it with the turn), and the
        moment it is backgrounded the continuation is re-launched with
        `task.abortController` — a controller `registerAgentForeground` built
        with `createAbortController()`, linked to nothing. Its sibling
        `registerAsyncAgent` is called with no parent at all, under the comment
        "Don't link to parent's abort controller -- background agents should
        survive when the user presses ESC to cancel the main thread. They are
        killed explicitly via chat:killAgents."

        We cannot swap the object: the task state and the agent's ToolUseContext
        are already holding this one, and TaskStop sets it by that reference.
        Dropping the link has the same effect and keeps the stop path intact.
        """
        self._parent = None

    def is_set(self) -> bool:
        return self._event.is_set() or bool(self._parent is not None and self._parent.is_set())

    async def wait(self) -> bool:
        """Wait until this scope is stopped, or any ancestor is.

        It must wake on the ancestor too. `utils.generators.make_abortable` is
        the single tool-dispatch point that makes every tool interruptible, and
        it races each pull against exactly this coroutine. Waiting only on the
        local event put a sub-agent's long tool — a WebFetch, a bash read —
        right back in the state that helper was written to fix: the user's stop
        set the parent, `is_set()` turned true, and nothing woke, so the tool
        ran to completion.

        The waiters are created per call and cancelled on the way out, so
        nothing survives this frame. `self._parent.wait()` recurses, which is
        what carries a grandparent's stop down.
        """
        import asyncio
        if self.is_set():
            return True
        waiters = [asyncio.ensure_future(self._event.wait())]
        if self._parent is not None:
            waiters.append(asyncio.ensure_future(self._parent.wait()))
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waiters:
                w.cancel()
        return True


def child_of(parent: Optional[Any]) -> AbortScope:
    """A scope under `parent` (which may be None, an Event, or an AbortScope)."""
    return AbortScope(parent)
