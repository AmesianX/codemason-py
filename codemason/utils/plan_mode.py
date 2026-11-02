"""Plan mode state for CodeMason.

Based on the permission-mode 'plan' concept from Claude Code. The original keeps
this on `AppState.toolPermissionContext.mode`; codemason now has a real AppState
(state/app_state.py), so plan mode lives there too — NOT a separate bool flag.

- enter_plan_mode(): prepareContextForPlanMode (stash current mode as
  pre_plan_mode) + applyPermissionUpdate({setMode: 'plan'}). Net effect:
  {mode: 'plan', pre_plan_mode: <previous mode>}.
- exit_plan_mode(): restore mode = pre_plan_mode ?? 'default', clear
  pre_plan_mode (ExitPlanModeV2Tool.ts call()).

Only the setMode behavior of applyPermissionUpdate is ported — its full
permission-rule update machinery (addRules etc.) backs a rule engine codemason
doesn't have. While mode == 'plan', write tools are blocked in the permission
check (query.py) and the model explores + plans.
"""

from dataclasses import replace

from ..state.app_state import AppState, get_app_state, set_app_state


def is_plan_mode_active() -> bool:
    """True when the current permission mode is 'plan'."""
    return get_app_state().tool_permission_context.mode == "plan"


def enter_plan_mode() -> None:
    """Enter plan mode (EnterPlanModeTool.call()).

    Stashes the current mode as pre_plan_mode (prepareContextForPlanMode) then
    sets mode='plan' (applyPermissionUpdate setMode). No-op shape if already in
    plan mode preserves the original pre_plan_mode.
    """
    def _apply(prev: AppState) -> AppState:
        ctx = prev.tool_permission_context
        if ctx.mode == "plan":
            return prev
        return replace(
            prev,
            tool_permission_context=replace(ctx, pre_plan_mode=ctx.mode, mode="plan"),
        )

    set_app_state(_apply)


def exit_plan_mode() -> None:
    """Exit plan mode, restoring the prior mode (ExitPlanModeV2Tool.call()).

    Restore target is pre_plan_mode, falling back to 'default'; pre_plan_mode is
    cleared. No-op when not in plan mode.
    """
    def _apply(prev: AppState) -> AppState:
        ctx = prev.tool_permission_context
        if ctx.mode != "plan":
            return prev
        restore_mode = ctx.pre_plan_mode or "default"
        return replace(
            prev,
            tool_permission_context=replace(ctx, mode=restore_mode, pre_plan_mode=None),
        )

    set_app_state(_apply)
