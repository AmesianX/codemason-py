#!/usr/bin/env python3
"""A parent's stop reaches its sub-agents; a sub-agent's does not reach anyone else.

While one sub-agent ran at a time, handing it the parent's own Event cost nothing.
Running three it decides whether stopping one is possible at all: the gate's wait
returns None on that event and reports 'Aborted while waiting for a sub-agent
slot', so a stop aimed at one agent used to cancel a sibling that was only queued.
"""
import asyncio
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


from codemason.utils.abort import AbortScope, child_of


async def main():
    parent = asyncio.Event()
    a = child_of(parent)
    b = child_of(parent)

    check('a fresh scope is not stopped', not a.is_set() and not b.is_set())

    a.set()
    check('stopping one agent does not stop its sibling', not b.is_set())
    check('...and does not stop the parent', not parent.is_set())
    check('...while that agent reads as stopped', a.is_set())

    parent.set()
    check("a parent's stop reaches every child", b.is_set())

    late = child_of(parent)
    check('a child spawned under a stopped parent starts stopped', late.is_set())

    # Nesting: a forked skill inside a sub-agent is a scope under a scope.
    g = child_of(b)
    check('the link is transitive through a nested scope', g.is_set())

    standalone = child_of(None)
    check('no parent is allowed (and reads as running)', not standalone.is_set())
    standalone.set()
    check('...and can still be stopped', standalone.is_set())

    # The gate reads is_set() only; wait() is for symmetry with asyncio.Event.
    own = child_of(asyncio.Event())
    own.set()
    check('wait() returns once this scope is stopped',
          await asyncio.wait_for(own.wait(), timeout=1) is True)

    # wait() has to wake on an ancestor, not just on this scope. make_abortable
    # races every tool pull against exactly this coroutine, so a scope that only
    # watches its own event leaves a sub-agent's long tool running after the
    # user's stop — the bug that helper exists to prevent.
    top = asyncio.Event()
    mid = child_of(top)
    deep = child_of(mid)
    waiting = asyncio.ensure_future(deep.wait())
    await asyncio.sleep(0)
    check('a grandchild is not woken before anything stops', not waiting.done())
    top.set()
    check('a grandparent\'s stop wakes it',
          await asyncio.wait_for(waiting, timeout=1) is True)

    done = asyncio.Event()
    own = child_of(done)
    waiting = asyncio.ensure_future(own.wait())
    await asyncio.sleep(0)
    own.set()
    check('its own stop wakes it too',
          await asyncio.wait_for(waiting, timeout=1) is True)
    check('...without touching the parent', not done.is_set())

    pre = asyncio.Event()
    pre.set()
    check('a scope under an already-stopped parent returns at once',
          await asyncio.wait_for(child_of(pre).wait(), timeout=1) is True)

    shared = asyncio.Event()
    one, two = child_of(shared), child_of(shared)
    watching_two = asyncio.ensure_future(two.wait())
    await asyncio.sleep(0)
    one.set()
    await asyncio.sleep(0.02)
    check('a sibling\'s stop does not wake its sibling', not watching_two.done())
    watching_two.cancel()

    # What the sub-agent tools actually pass.
    from codemason.tools import agent_tool, skill_tool
    src = Path(agent_tool.__file__).read_text(encoding='utf-8')
    check('AgentTool gives the sub-agent its own scope',
          'abort_controller=agent_abort' in src and 'child_of(' in src)
    check('...and waits for a slot on that scope, not the parent',
          'subagent_gate.acquire(agent_abort)' in src)
    src = Path(skill_tool.__file__).read_text(encoding='utf-8')
    check('a forked skill does the same',
          'abort_controller=skill_abort' in src
          and 'subagent_gate.acquire(skill_abort)' in src)

    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
