"""Which live session a wake is delivered to.

`codemason inject` exists so work that outlives a turn can start the next one.
That only helps if it reaches the session that asked for it: a report delivered
to the wrong conversation reads as delivered, so the person waiting is told
nothing AND believes they were told. Every branch here is therefore a refusal
rather than a fallback, and these cases are what stop the refusals quietly
turning back into fallbacks.

The world is passed in — pick_session takes the session list — so none of this
needs a running session, a registry on disk, or a bot.

Run: python3 tests/test_wake_target.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from codemason.utils.wake_cli import SESSION_ENV, pick_session  # noqa: E402

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


def entry(sid, pid=1):
    return {'session_id': sid, 'pid': pid, 'cwd': '/tmp', 'label': '~', 'started': 0.0}


# The ordinary case a watcher script relies on: one session, nothing said.
sid, why = pick_session([entry('aaa')])
check('a lone session needs no argument', sid == 'aaa', f'got {sid!r} / {why!r}')

# Two sessions and no choice given must refuse. Picking one would deliver into
# a conversation that never asked, which is the failure this whole file guards.
sid, why = pick_session([entry('aaa'), entry('bbb', 2)])
check('an ambiguous target is refused, not guessed', sid is None, f'got {sid!r}')
check('the refusal says how to resolve it', '--session' in why, why)
check('the refusal lists the candidates', 'aaa' in why and 'bbb' in why, why)

# The exported id wins over "the only one", so a script spawned by session A
# keeps addressing A even after B starts beside it.
sid, _ = pick_session([entry('aaa'), entry('bbb', 2)], env_id='bbb')
check('the exported id decides before the count does', sid == 'bbb', f'got {sid!r}')

# A named session that has since exited is an error, not a fallback to whatever
# else happens to be running: the watcher is reporting to a conversation that no
# longer exists and has to be told so.
sid, why = pick_session([entry('aaa')], explicit='gone')
check('a dead --session does not fall back to a live one', sid is None, f'got {sid!r}')
sid, why = pick_session([entry('aaa')], env_id='gone')
check('a stale exported id does not fall back either', sid is None, f'got {sid!r}')
check('and it names the variable that is wrong', SESSION_ENV in why, why)

# Unattended and unaddressed: the "only live session" convenience is a guess
# about someone else's conversation, and a script is exactly where nobody is
# watching it be wrong. A -p run is the sharp case — it registers no session, so
# it has no id of its own and would otherwise take whatever happened to be open.
sid, why = pick_session([entry('aaa')], interactive=False)
check('a script with no target refuses rather than taking the only session',
      sid is None, f'got {sid!r}')
check('and says what to pass instead', '--session' in why and SESSION_ENV in why, why)
sid, _ = pick_session([entry('aaa')], env_id='aaa', interactive=False)
check('but an inherited id is a target, terminal or not', sid == 'aaa', f'got {sid!r}')
sid, _ = pick_session([entry('aaa'), entry('bbb', 2)], explicit='bbb', interactive=False)
check('as is an explicit one', sid == 'bbb', f'got {sid!r}')

# Nothing running at all: say so, and point at the command that still works
# without a session, because that is the next thing the caller needs.
sid, why = pick_session([])
check('with nothing running it refuses', sid is None, f'got {sid!r}')
check('and names the session-free route', 'notify' in why, why)

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
