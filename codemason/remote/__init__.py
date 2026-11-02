"""Remote session layer — notifications now, full remote control later.

Claude Code implements the same capability across ~14k lines (`bridge/`,
`remote/`, `server/`), but most of that is infrastructure we do not have and do
not need: registering an "environment" with `api.anthropic.com`, minting session
ingress URLs, OAuth federation, trusted-device attestation, cloud poll/capacity
management. Those exist because their desktop sits behind NAT and the phone can
only reach it through their relay.

We are not on that system. The phone is already inside the network (WireGuard
full-tunnel), so there is no relay to build and no federation to speak. What we
keep from their design is the one judgement that matters: the wire carries
*semantic events*, not terminal bytes. Their bridge runs sessions with
`--input-format stream-json --output-format stream-json`, and codemason already
emits exactly that shape from `print_runner`. So a remote client renders the
conversation natively instead of emulating a TTY.

Layering, so the transport choice stays reversible:

    events.py    what happened (backend-agnostic)
    adapters/    how it reaches a human (telegram now, web push / PWA later)
    notify.py    the API everything else calls

Only the adapter differs between Telegram and a future PWA; the session runner,
protocol and permission round-trip planned above this layer are shared. Picking
Telegram first is therefore not a fork in the road.
"""

from .events import Event, EventKind, Sensitivity

__all__ = ['Event', 'EventKind', 'Sensitivity', 'notify', 'is_configured']


# `notify` is resolved lazily (PEP 562). Importing it here would load
# codemason.remote.notify as a side effect of importing the package, so
# `python -m codemason.remote.notify` would then execute an already-imported
# module and warn about it — and shell callers use exactly that entry point.
# Events are cheap and dependency-free, so they stay eager.
def __getattr__(name):
    if name in ('notify', 'is_configured', 'register', 'adapters'):
        # import_module, NOT `from . import notify` — the latter goes through
        # _handle_fromlist, which asks this very __getattr__ for the submodule
        # and recurses until the stack dies.
        import importlib
        return getattr(importlib.import_module('.notify', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
