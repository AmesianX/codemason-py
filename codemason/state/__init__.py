"""Application state for CodeMason.

The source keeps runtime state in a single React/Ink store
(`state/AppStateStore.ts` → `AppState`). The full AppState (spinner, expanded
view, teammates, speculation, …) is UI state bound to the Ink rendering tree —
that is substrate and is NOT ported. What IS ported is the `tasks` slice: the
single registry every task mechanism keys off (`appState.tasks[taskId]`), read
via `getAppState()` and mutated via `setAppState(prev => next)`.

`app_state` here is the faithful codemason equivalent of that store: a single
maintained AppState object (so future slices can be added as fields) plus
module-global `get_app_state()` / `set_app_state()` accessors — the same
module-singleton idiom codemason already uses for `utils/state.py` (get_cwd).
"""
