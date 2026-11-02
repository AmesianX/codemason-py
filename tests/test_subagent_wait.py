"""What a sub-agent does when every slot is taken.

The gate itself is pinned in test_subagent_slots. This is its CALLER: the branch
in AgentTool.call that runs when try_acquire comes back empty, which is every
time two sub-agents overlap on one endpoint (the default is ONE slot).

That branch was unreachable. It yielded `ToolResult(type='progress', ...)`, and
ToolResult is Union[ToolResultSuccess, ToolResultProgress] — a type alias, so
calling it raised 'Cannot instantiate typing.Union'. The tool died at that line:
no wait, no queue, nothing in error.log, and the model was told the sub-agent
had produced no output. Measured against the real app: a run whose slot was held
for 20s finished in 9s having run no sub-agent. With the fix it takes 22s and
comes back with the sub-agent's answer.

Nothing in either suite noticed, because every test drove the gate directly and
never the tool. These cases drive the tool.

  constructible   the waiting branch yields a result that can be built at all
  visible         the text rides on result_for_assistant, the FIELD that query
                  and cli read — buried in `data` it reached nobody
  waits           the branch is entered only when the pool is actually full

flock is per open file description, so one process holding a slot blocks itself:
no subprocess needed, and the tool is driven only as far as the gate, never into
the model call behind it.

Run directly (exit 0 = all pass).
"""
import asyncio
import os
import shutil
import sys
import pathlib
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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
# Before importing codemason: the config module resolves its paths once.
_CFG = tempfile.mkdtemp(prefix='cm-wait-')
os.environ['CODEMASON_CONFIG_DIR'] = _CFG
# One slot, so holding it fills the pool and the waiting branch is reachable.
# The default is per line (1 where sub-agents are serial, 3 where they are not),
# and at 3 one held slot left two free: the tool never waited, it ran the query
# and failed on an unauthenticated request instead. The version is stamped so the
# migration that carries a serial-line count forward leaves the pin alone.
import json as _json
from codemason.utils.config import CONFIG_VERSION as _CV
pathlib.Path(_CFG, 'config.json').write_text(
    _json.dumps({'config_version': _CV, 'max_concurrent_subagents': 1}), encoding='utf-8')

from codemason.tools.agent_tool import AgentTool                 # noqa: E402
from codemason.tools.base import ToolResult, ToolResultSuccess   # noqa: E402
from codemason.utils import subagent_gate                        # noqa: E402

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


WAITING = 'Waiting for another sub-agent to finish'


class _Ctx:
    """The parent context, as much of it as the gate branch reads."""
    abort_controller = None
    on_token_update = None
    options: dict = {}


async def _progress_until_wait(limit=6, timeout=20):
    """Drive AgentTool.call only as far as the gate, collecting progress yields.

    Closed as soon as the waiting line appears, so the model call behind the
    gate is never reached — the point here is the branch, not a sub-agent, and
    the suite stays offline.

    Returns (yields, failure). Both ways this branch has broken are failures to
    REPORT, not to raise: constructing the bare alias raised TypeError out of
    the generator, and burying the text left the tool waiting on a slot nobody
    would see it waiting for. So an exception is captured and the drive is
    capped — a case that crashes or hangs tells a reader nothing.
    """
    seen = []
    failure = None
    gen = AgentTool.call({'prompt': 'x', 'subagent_type': 'general-purpose'}, _Ctx())

    async def _drive():
        nonlocal failure
        for _ in range(limit):
            try:
                r = await gen.__anext__()
            except StopAsyncIteration:
                return
            except Exception as e:                       # noqa: BLE001
                failure = f'{type(e).__name__}: {e}'
                return
            seen.append(r)
            if WAITING in (getattr(r, 'result_for_assistant', '') or ''):
                return

    try:
        await asyncio.wait_for(_drive(), timeout)
    except asyncio.TimeoutError:
        failure = failure or (
            f'no waiting line within {timeout}s — the tool went past the gate '
            'or announced the wait somewhere nobody reads')
    finally:
        try:
            await gen.aclose()
        except Exception:                                # noqa: BLE001
            pass
    return seen, failure


# ── 1. The alias cannot be called, which is why the branch must not call it ──
try:
    ToolResult(type='progress', data={})
    _constructible = True
    _why = ''
except Exception as e:
    _constructible = False
    _why = f'{type(e).__name__}: {e}'
check('ToolResult is a type alias, not something a tool can construct',
      not _constructible, 'it constructed, so this case no longer proves anything')
check('  (and the error it raises is the one that was swallowed)',
      'Union' in _why, _why)

# ── 2. With the pool full, the tool announces the wait ──────────────────────
held = subagent_gate.try_acquire()
check('a slot was free to take for the test', held is not None)

seen, failure = asyncio.run(_progress_until_wait())
check('driving the tool into the full-pool branch raises nothing and ends',
      failure is None, failure or '')
waiting = [r for r in seen
           if WAITING in (getattr(r, 'result_for_assistant', '') or '')]
check('the tool yields the waiting line when every slot is taken',
      len(waiting) == 1,
      f'{len(waiting)} of {len(seen)}: '
      f'{[getattr(r, "result_for_assistant", None) for r in seen]}')

if waiting:
    w = waiting[0]
    check('it is a result that could be built (not the bare alias)',
          isinstance(w, ToolResultSuccess), type(w).__name__)
    check("it is typed 'progress', so the turn is not ended by it",
          getattr(w, 'type', None) == 'progress', getattr(w, 'type', None))
    check('the text rides on result_for_assistant, where query and cli read it',
          WAITING in (w.result_for_assistant or ''), repr(w.result_for_assistant))
    check('max_concurrent_subagents is named, so the line says what to change',
          'max_concurrent_subagents' in (w.result_for_assistant or ''))

# ── 3. The branch is entered on the gate's answer, nothing else ────────────
#
# Asserted on the gate rather than by driving the tool again: with a slot free
# the tool walks straight past this branch into the model call, and a test that
# reaches the network to prove a negative is a test that fails on a train. The
# branch's condition is `_slot is None`, so this is the whole of it.
if held is not None:
    held.release()
free = subagent_gate.try_acquire()
check('a released slot is free again, so the branch is not entered',
      free is not None)
if free is not None:
    free.release()

shutil.rmtree(_CFG, ignore_errors=True)
sys.exit(0 if all(results) else 1)
