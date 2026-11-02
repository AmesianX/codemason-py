"""AppState store — equivalent of state/AppStateStore.ts.

The original `AppState` is a large DeepImmutable record driving the Ink UI
(spinner, expandedView, teammates, speculation, …). Those fields are bound to
React rendering — substrate — and are not ported. The one slice the task
framework actually keys off is `tasks: { [taskId]: TaskState }`, accessed via
`getAppState()` and updated through `setAppState(prev => next)`.

We model exactly that slice on a real `AppState` object held in a single
module-global store, with `get_app_state()` / `set_app_state(updater)` mirroring
the original's get/set semantics (the updater receives the previous state and
returns the next). New slices can be added as `AppState` fields later without
changing this contract — which is why the registry lives on an AppState object
rather than being smeared into the LocalShellTask producer.

This is the same module-singleton idiom codemason already uses in
`utils/state.py`; it is the Python substrate equivalent of Ink's React store.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from ..tasks.types import TaskState


@dataclass
class ToolPermissionContext:
    """Ported slice of the source's toolPermissionContext (AppStateStore.ts).

    Only the fields plan mode actually keys off are modeled: `mode` and
    `pre_plan_mode`. The source's toolPermissionContext also carries the
    permission-rule collections (alwaysAllow/Deny/Ask) and auto-mode state —
    codemason has no permission-rule engine and no auto/acceptEdits modes (whole
    absent subsystems with no equivalent), so those are not modeled. `mode` is
    therefore just 'default' | 'plan'; `pre_plan_mode` stashes the mode in effect
    when plan mode was entered so ExitPlanMode can restore it.
    """

    mode: str = "default"
    pre_plan_mode: Optional[str] = None


@dataclass
class AppState:
    """The ported slice of the AppState.

    `tasks` is the task registry — the single source of truth keyed by task id.
    `tool_permission_context` holds the permission mode (plan vs default). Kept as
    fields on a dedicated object (not bare module globals) so additional state
    slices can be added here as the port grows, exactly as the original AppState
    carries many slices.
    """

    tasks: Dict[str, "TaskState"] = field(default_factory=dict)
    tool_permission_context: ToolPermissionContext = field(
        default_factory=ToolPermissionContext
    )
    # Todo lists keyed by agentId ?? sessionId (source AppState.todos). The
    # TodoWrite tool reads/writes this slice; see utils/todo_store.py.
    todos: Dict[str, list] = field(default_factory=dict)
    # name -> agentId for the sub-agents the model gave a name to
    # (AppState.agentNameRegistry). This is what makes SendMessage({to: name})
    # work: without it the model can only address an agent by the 16-hex id the
    # spawn happened to mint, which it has to copy out of a tool result and
    # carry by hand. Written only by a BACKGROUND spawn — a foreground agent is
    # blocking its parent, so there is nobody left to send it anything.
    agent_name_registry: Dict[str, str] = field(default_factory=dict)


# Single maintained store instance (the AppStateStore singleton).
_app_state: AppState = AppState()


def get_app_state() -> AppState:
    """Read the current AppState (getAppState)."""
    return _app_state


def set_app_state(updater: Callable[[AppState], AppState]) -> None:
    """Apply an immutable update (setAppState).

    `updater` receives the previous AppState and returns the next one. Mirrors
    the original `setAppState(prev => next)`; framework helpers build the next
    state with `dataclasses.replace(prev, tasks={...})`. If the updater returns
    the same object (a no-op), the store is left untouched.
    """
    global _app_state
    _app_state = updater(_app_state)


def reset_app_state() -> None:
    """Reset the store (used by tests / `clear`)."""
    global _app_state
    _app_state = AppState()
