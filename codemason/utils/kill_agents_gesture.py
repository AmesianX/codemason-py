"""Stopping every background agent at once — the two-press gesture.

A background sub-agent is unlinked from the turn's abort scope on purpose, so
Esc cancels the turn and leaves it running. The source pairs that with a
deliberate way to end them in bulk, and says why it is deliberate: "killing
agents from the main prompt stays a deliberate gesture (chat:killAgents), not a
side-effect of cancelling a turn."

Two presses, inside the source's own 3s window, because one keystroke must not
be able to throw away three agents' work. The first press arms and says so; the
second does it. A press with nothing running says nothing is running, rather
than looking broken.

Two entry points share one gesture here, which is why this is a small object
rather than a closure in the key binding:

  * Ctrl+C — our brake. Esc is "interject now", Ctrl+C is the one key that
    brings the session to rest (it drops the queued backlog too), so it is also
    where someone reaches when they want everything to stop. Its hint only
    appears when agents are actually running, which is what makes the gesture
    discoverable at the moment it matters.
  * Ctrl+X Ctrl+K — the source's own chord, kept so muscle memory from Claude
    Code works here.

The window is shared between them: what is armed is the gesture, not the key.
"""
from typing import Callable, Optional, Tuple

# KILL_AGENTS_CONFIRM_WINDOW_MS, in seconds.
CONFIRM_WINDOW_S = 3.0


class KillAgentsGesture:
    """The arm-then-fire state for the bulk stop. One per REPL."""

    __slots__ = ('_armed_at', '_window', '_key')

    def __init__(self, window: float = CONFIRM_WINDOW_S):
        self._armed_at = 0.0
        self._window = window
        # Which key armed it, so the hint names the key in the user's hand
        # rather than whichever one this module happens to prefer.
        self._key = 'Ctrl+C'

    # --- the state the prompt paints ---------------------------------------
    def armed(self, now: Optional[float] = None) -> bool:
        if not self._armed_at:
            return False
        import time
        now = time.monotonic() if now is None else now
        return (now - self._armed_at) <= self._window

    def hint(self, now: Optional[float] = None) -> Optional[str]:
        """What to show while armed, or None. Painted by the rprompt."""
        if not self.armed(now):
            return None
        return f' ({self._key} again to stop background agents) '

    # --- the press ---------------------------------------------------------
    def press(
        self,
        key: str = 'Ctrl+C',
        now: Optional[float] = None,
        get_app_state_fn: Optional[Callable] = None,
        set_app_state_fn: Optional[Callable] = None,
    ) -> Tuple[str, Optional[str]]:
        """One press. Returns (outcome, message-for-the-user).

        outcome is 'idle' (nothing to stop), 'armed' (press again) or 'killed'.
        The message is what to print; the model is told separately, by
        kill_all_agents_and_notify, and only on the press that kills.

        The task table is read on every press rather than captured, because a
        gesture must act on what is running NOW — the source reads its store
        here for the same reason ("Reads tasks from the store directly to avoid
        stale closures").
        """
        import time
        from ..state.app_state import get_app_state, set_app_state
        from ..tasks.local_agent_task import (
            is_local_agent_task,
            kill_all_agents_and_notify,
        )
        _get = get_app_state_fn or get_app_state
        _set = set_app_state_fn or set_app_state
        now = time.monotonic() if now is None else now

        running = [
            t for t in _get().tasks.values()
            if is_local_agent_task(t) and t.status == 'running'
        ]
        if not running:
            self._armed_at = 0.0
            return 'idle', None

        if self.armed(now):
            self._armed_at = 0.0
            summary = kill_all_agents_and_notify(
                set_app_state_fn=_set, get_app_state_fn=_get
            )
            return 'killed', summary

        self._armed_at = now
        self._key = key
        n = len(running)
        return 'armed', (
            f'{n} background agent{"s" if n != 1 else ""} still running'
            f' — press {key} again to stop {"them" if n != 1 else "it"}'
        )
