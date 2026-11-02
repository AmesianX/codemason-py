"""Transport adapters.

An adapter turns an `Event` into something a human actually sees. Swapping the
adapter is the only difference between "notify me on Telegram" and "stream it to
my own PWA", which is why the choice does not have to be made once and for all.

Phase 1 needs `send` only. `ask` and `listen` are declared now so the Telegram
adapter's inline keyboards and the future WebSocket adapter answer to the same
shape when the permission round-trip lands — that round-trip is the risky part of
this project (Claude Code shipped bugs in it twice: 2.1.208, 2.1.211), so the
seam it will need is worth fixing early even while unimplemented.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from ..events import Event, Sensitivity


@runtime_checkable
class Adapter(Protocol):
    """A place notifications can go."""

    name: str

    @property
    def max_sensitivity(self) -> Sensitivity:
        """Highest classification this transport is cleared to carry. `notify`
        redacts against this before the adapter ever sees the payload."""
        ...

    def is_configured(self) -> bool:
        """False when credentials are missing — callers degrade quietly rather
        than crashing a turn over a notification."""
        ...

    def send(self, event: Event) -> Optional[str]:
        """Deliver one event. Returns a transport message id when it has one.
        Must never raise: a failed notification is not worth a failed turn."""
        ...

    # --- Phase 3 (permission round-trip) — declared, not yet implemented ------

    def ask(self, event: Event, options: List[str], timeout: float,
            cancel: Optional['Event'] = None) -> Optional[str]:
        """Present `options` and block until one is chosen, cancelled, or timed out.

        `cancel` (a threading.Event) means the question stopped being live —
        normally because the local terminal answered it first. Adapters must
        check it between polls and return promptly: this runs on a thread that
        cannot be killed, and on transports where polling is exclusive it may be
        holding a lock other sessions need.

        `timeout` bounds the wait for the case where nothing else ever will —
        a headless run has no terminal to fall back to. It is not a display
        deadline: while the question is still live the buttons must stay usable,
        because the turn is still blocked on them.
        """
        raise NotImplementedError

    def listen(self):
        """Yield inbound user messages for injection into the session queue."""
        raise NotImplementedError
