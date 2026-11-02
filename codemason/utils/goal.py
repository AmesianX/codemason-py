"""
Persistent session goal, with optional completion gates.

A goal is a free-form user objective that stays active across turns: once set
with ``/goal <text>``, it is injected into the system prompt on every turn (as an
``<active_goal>`` context block) so the agent keeps the overarching objective in
view instead of drifting after the first sub-task.

A goal can also carry **completion gates** — deterministic shell commands that
must all pass (exit 0) before the goal is genuinely done (e.g. the test suite,
the linter). The gates are listed to the agent inside the goal block as the
completion bar, and ``/goal check`` runs them and reports pass/fail. Gates make
"done" verifiable instead of a matter of opinion.

State persists as ``goal.txt`` + ``goal-gates.txt`` **inside the project
directory** (``<CONFIG_DIR>/projects/<sanitized-cwd>/``), so it survives a
restart until explicitly cleared.

The project directory, and not the config dir, because a goal is a property of
the work — and its gates are literally that project's test command. One file per
home cannot answer a question two instances answer differently: with a `cm2` in
one repo and a `codemason` in another, a config-dir goal.txt meant whoever typed
``/goal`` last silently retargeted every other session, and ``/goal clear``
deleted a goal the other window was still working to. That is the same reasoning
— and the same key — as session_storage's per-directory settings and the
transcripts that already live there. (No migration: the old config-dir files
were never released with anything that wrote them on this box.)
"""

from __future__ import annotations

import subprocess
from typing import List, Tuple

from .session_storage import get_project_dir, project_lock
from .state import get_original_cwd


def _goal_file():
    return get_project_dir(get_original_cwd()) / "goal.txt"


def _gates_file():
    return get_project_dir(get_original_cwd()) / "goal-gates.txt"

_GATE_TIMEOUT_S = 300          # per-gate hard timeout
_GATE_OUTPUT_TAIL = 2000       # chars of a failed gate's output fed back


# ── goal text ───────────────────────────────────────────────────────────────
def get_goal() -> str:
    """The active goal text, or '' if none is set. Never raises."""
    try:
        path = _goal_file()
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


def set_goal(text: str) -> None:
    """Set (and persist) the active goal."""
    text = (text or "").strip()
    if not text:
        clear_goal()
        return
    try:
        from .file import atomic_write_bytes
        path = _goal_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: another session in this directory, or the turn that is about to
        # render the goal into its system prompt, must never read half of it.
        # No lock needed — this replaces the whole document rather than editing
        # one line of it, so there is nothing to merge.
        atomic_write_bytes(str(path), text.encode("utf-8"))
    except OSError:
        pass


def clear_goal() -> None:
    """Remove the active goal and its gates."""
    for f in (_goal_file(), _gates_file()):
        try:
            f.unlink()
        except OSError:
            pass


# ── completion gates ──────────────────────────────────────────────────────────
def get_gates() -> List[str]:
    """The gate commands, in order. Never raises."""
    try:
        path = _gates_file()
        if path.is_file():
            return [ln for ln in (l.strip() for l in
                    path.read_text(encoding="utf-8").splitlines()) if ln]
    except OSError:
        pass
    return []


def _save_gates(gates: List[str]) -> None:
    """Replace the gate list. Call it holding the gates lock (see _edit_gates)
    when the new list was derived from the old one."""
    try:
        from .file import atomic_write_bytes
        path = _gates_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        if gates:
            atomic_write_bytes(str(path), ("\n".join(gates) + "\n").encode("utf-8"))
        else:
            try:
                path.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _edit_gates(mutate):
    """Read the gates, change them, write them back as one step.

    Adding and removing are read-modify-write: two sessions in this directory
    doing it at once would otherwise each write a list that never saw the
    other's edit, and one gate — a test command the user means to be a release
    bar — would vanish with nothing said.
    """
    from .file import FileLock
    with FileLock(project_lock("goal-gates.txt")):
        gates = get_gates()
        updated = mutate(gates)
        if updated is None:
            return False
        _save_gates(updated)
        return True


def add_gate(command: str) -> None:
    command = (command or "").strip()
    if not command:
        return

    def append(gates: List[str]) -> List[str]:
        gates.append(command)
        return gates

    _edit_gates(append)


def remove_gate(index: int) -> bool:
    """Remove the 1-based gate at ``index``. Returns True if it existed."""
    hit = []

    def drop(gates: List[str]):
        if 1 <= index <= len(gates):
            hit.append(gates.pop(index - 1))
            return gates
        return None

    _edit_gates(drop)
    return bool(hit)


def clear_gates() -> None:
    _save_gates([])


def run_gate(command: str, cwd: str = None) -> Tuple[bool, int, str]:
    """Run one gate command through the shell. Returns (passed, exit_code, tail)."""
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=_GATE_TIMEOUT_S, cwd=cwd or None,
        )
        combined = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return proc.returncode == 0, proc.returncode, combined[-_GATE_OUTPUT_TAIL:]
    except subprocess.TimeoutExpired:
        return False, -1, f"[gate timed out after {_GATE_TIMEOUT_S}s]"
    except Exception as exc:  # noqa: BLE001
        return False, -1, f"[gate could not run: {type(exc).__name__}: {exc}]"


def run_gates(cwd: str = None) -> List[Tuple[str, bool, int, str]]:
    """Run every gate. Returns [(command, passed, exit_code, output_tail), ...]."""
    return [(g, *run_gate(g, cwd=cwd)) for g in get_gates()]


# ── injection ─────────────────────────────────────────────────────────────────
def render_goal_context() -> str:
    """The text injected as the <active_goal> block each turn, or '' if no goal.

    Includes the completion gates as the explicit "done" bar so the agent knows
    not to declare the goal complete until they pass."""
    goal = get_goal()
    if not goal:
        return ""
    gates = get_gates()
    if not gates:
        return goal
    lines = [goal, "",
             "This goal is complete ONLY when ALL of these gate commands pass "
             "(exit 0). Run them to verify; do not declare the goal done while "
             "any gate fails:"]
    for i, g in enumerate(gates, 1):
        lines.append(f"  {i}. $ {g}")
    return "\n".join(lines)
