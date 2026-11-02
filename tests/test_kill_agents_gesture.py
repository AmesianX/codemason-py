#!/usr/bin/env python3
"""Stopping every background agent takes two presses.

Esc cancels the turn and leaves background agents running on purpose, so there
has to be a way to end them — and it must not be one keystroke. The source's
own rule: "killing agents from the main prompt stays a deliberate gesture
(chat:killAgents), not a side-effect of cancelling a turn", two presses inside
a 3s window.

Two keys reach it here: Ctrl+C, which is already the brake that brings the
session to rest, and the source's Ctrl+X Ctrl+K chord. They share one window —
what is armed is the gesture, not the key — and the hint names whichever key
armed it, because that is the key in the user's hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == 'win32':
    # Our own case names carry em dashes and Korean, and the console
    # encoder here is cp949: the first such line would raise
    # UnicodeEncodeError inside the reporter and stop the run mid-suite,
    # blaming whatever case it was printing. Pinned inline, importing
    # nothing -- several of these files must set CODEMASON_CONFIG_DIR
    # before anything under codemason/ is imported.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


from codemason.state.app_state import get_app_state, reset_app_state, set_app_state
from codemason.tasks.local_agent_task import register_async_agent
from codemason.utils import message_queue_manager as mq
from codemason.utils.abort import child_of
from codemason.utils.kill_agents_gesture import CONFIRM_WINDOW_S, KillAgentsGesture


def spawn(n):
    """n running background agents, each on its own unlinked scope."""
    reset_app_state()
    mq.reset_pending_notifications()
    scopes = {}
    for i in range(n):
        aid = 'a' + str(i) * 16
        scopes[aid] = child_of(None)
        register_async_agent(
            agent_id=aid, description=f'agent {i}', prompt='p',
            selected_agent=None, abort_controller=scopes[aid],
        )
    return scopes


def main():
    # ---- nothing running ---------------------------------------------------
    reset_app_state()
    mq.reset_pending_notifications()
    g = KillAgentsGesture()
    outcome, note = g.press()
    check('with nothing running the press is idle', outcome == 'idle', outcome)
    check('...with nothing to say', note is None, note)
    check('...and nothing is armed', g.armed() is False)
    check('...so no hint is painted', g.hint() is None)
    check('...and the model is told nothing', mq.drain_pending_notifications() == [])

    # ---- one press arms, it does not kill ----------------------------------
    scopes = spawn(3)
    g = KillAgentsGesture()
    outcome, note = g.press('Ctrl+C')
    check('the first press arms', outcome == 'armed', outcome)
    check('...naming what is at stake and the key to repeat',
          note == '3 background agents still running — press Ctrl+C again to stop them',
          note)
    check('...it kills NOTHING', all(not s.is_set() for s in scopes.values()))
    check('...every task still runs',
          all(t.status == 'running' for t in get_app_state().tasks.values()))
    check('...the model is told nothing yet', mq.drain_pending_notifications() == [])
    check('...and the prompt shows the confirmation',
          g.hint() == ' (Ctrl+C again to stop background agents) ', g.hint())

    # ---- the second press kills -------------------------------------------
    outcome, note = g.press('Ctrl+C')
    check('the second press kills', outcome == 'killed', outcome)
    check('...every scope is stopped', all(s.is_set() for s in scopes.values()))
    check('...every task is killed',
          all(t.status == 'killed' for t in get_app_state().tasks.values()),
          [t.status for t in get_app_state().tasks.values()])
    notes = mq.drain_pending_notifications()
    check('...and the model is told once', len(notes) == 1, notes)
    check('...which is what the press reported', note == (notes[0] if notes else None))
    check('the gesture disarms after firing', g.armed() is False)
    check('...and the hint is gone', g.hint() is None)

    # ---- the window expires ------------------------------------------------
    scopes = spawn(2)
    g = KillAgentsGesture()
    g.press('Ctrl+C', now=100.0)
    check('armed inside the window', g.armed(now=100.0 + CONFIRM_WINDOW_S - 0.01) is True)
    check('...expired past it', g.armed(now=100.0 + CONFIRM_WINDOW_S + 0.01) is False)
    outcome, _ = g.press('Ctrl+C', now=100.0 + CONFIRM_WINDOW_S + 0.5)
    check('a late press arms again rather than killing', outcome == 'armed', outcome)
    check('...so nothing died', all(not s.is_set() for s in scopes.values()))
    outcome, _ = g.press('Ctrl+C', now=100.0 + CONFIRM_WINDOW_S + 0.6)
    check('and the pair that IS inside the window kills', outcome == 'killed', outcome)
    mq.drain_pending_notifications()

    # ---- the two keys share one window -------------------------------------
    scopes = spawn(1)
    g = KillAgentsGesture()
    outcome, note = g.press('Ctrl+C')
    check('one agent gets the singular wording',
          note == '1 background agent still running — press Ctrl+C again to stop it',
          note)
    outcome, summary = g.press('Ctrl+X Ctrl+K')
    check('the chord completes a gesture Ctrl+C armed', outcome == 'killed', outcome)
    check('...with the source\'s singular phrasing',
          summary == 'Background agent "agent 0" was stopped by the user.', summary)
    mq.drain_pending_notifications()

    scopes = spawn(2)
    g = KillAgentsGesture()
    _, note = g.press('Ctrl+X Ctrl+K')
    check('the hint names the chord when the chord armed it',
          g.hint() == ' (Ctrl+X Ctrl+K again to stop background agents) ', g.hint())
    check('...and says so in the line it prints too',
          'press Ctrl+X Ctrl+K again' in (note or ''), note)

    # ---- an agent that finishes between the presses -------------------------
    # The arm is not a promise: by the time the second press lands the agents
    # may have finished on their own, and the gesture must read the table again
    # rather than killing from a list it captured.
    scopes = spawn(2)
    g = KillAgentsGesture()
    g.press('Ctrl+C')
    from codemason.tasks.local_agent_task import complete_agent_task
    for aid in scopes:
        complete_agent_task(aid, {'agent_id': aid, 'text': 'done'})
    outcome, note = g.press('Ctrl+C')
    check('nothing left to kill: the armed gesture goes idle', outcome == 'idle', outcome)
    check('...and says nothing to the model',
          [n for n in mq.drain_pending_notifications() if 'stopped by the user' in n] == [])

    reset_app_state()
    mq.reset_pending_notifications()
    print(f'\n{sum(results)}/{len(results)} checks passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
