"""Todo store for CodeMason.

Faithful accessor over `AppState.todos` (the source keeps todo lists on
`appState.todos`, keyed by `agentId ?? sessionId`, read/written by TodoWriteTool
via getAppState/setAppState). codemason now has a real AppState, so todos live
there — NOT in a separate module-global dict. These helpers are the thin
get/set layer over that slice (same role as the task framework's
update_task_state over AppState.tasks).
"""

from dataclasses import replace
from typing import List

from ..state.app_state import AppState, get_app_state, set_app_state


def get_todos(key: str) -> List[dict]:
    """Read the todo list for a key (appState.todos[key] ?? [])."""
    return get_app_state().todos.get(key, [])


def set_todos(key: str, todos: List[dict]) -> None:
    """Write the todo list for a key (setAppState -> {...prev.todos, [key]: ...})."""
    def _apply(prev: AppState) -> AppState:
        return replace(prev, todos={**prev.todos, key: todos})

    set_app_state(_apply)


def todo_key(agent_id=None) -> str:
    """The key a conversation's todos live under: the sub-agent's id, or the
    literal "session" for the main one.

    This exists because the two sides disagreed. TodoWrite stored under
    `agent_id or 'session'` while the spinner read `get_session_id()` — a UUID —
    so the lookup never matched and the in_progress todo's activeForm, which is
    the whole reason that field exists, was dead code. Both comments said
    "agentId ?? sessionId", which is why the mismatch survived review. One
    function now, so there is nothing left to disagree."""
    return agent_id if agent_id else 'session'
