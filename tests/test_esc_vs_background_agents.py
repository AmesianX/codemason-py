#!/usr/bin/env python3
"""ESC cancels the turn. It does not kill a background agent.

That split is the source's, stated twice in its own words:

  * registerAsyncAgent is called with no parent controller — "Don't link to
    parent's abort controller -- background agents should survive when the user
    presses ESC to cancel the main thread. They are killed explicitly via
    chat:killAgents."
  * and the Ctrl+C handler: "killing agents from the main prompt stays a
    deliberate gesture (chat:killAgents), not a side-effect of cancelling a
    turn."

A foreground agent is the other case and must keep dying with the turn: it runs
inside the tool call the parent is blocked on, so cancelling the turn and
leaving it alive would orphan a run nobody is waiting for. The source does that
by simply not overriding the controller (the agent runs on the turn's); we give
each agent its own scope under the parent's, which propagates the same way while
also letting TaskStop hit one agent without its siblings.

So three abort scopes, three behaviours:
  foreground            -> child of the turn's scope. ESC ends it.
  background from start  -> unlinked. ESC leaves it alone.
  foreground, then moved -> detaches on the way out. ESC leaves it alone.

And the bulk stop that ESC is not: kill_all_agents_and_notify, one aggregate
message rather than one per agent.
"""
import asyncio
import contextlib
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


import codemason.query as query_mod
from codemason.state.app_state import get_app_state, reset_app_state, set_app_state
from codemason.tasks.local_agent_task import (
    background_agent_task,
    is_local_agent_task,
    kill_all_agents_and_notify,
    register_async_agent,
)
from codemason.types.query import ToolUseContext
from codemason.utils import message_queue_manager as mq
from codemason.utils import subagent_gate
from codemason.utils.abort import child_of
from codemason.utils.messages import create_assistant_message, create_user_message


def assistant(text, tools=(), base=0):
    m = create_assistant_message(text or 'placeholder')
    m['message']['usage'] = {'input_tokens': 500, 'output_tokens': 20,
                             'cache_read_input_tokens': 0,
                             'cache_creation_input_tokens': 0}
    m['message']['content'] = list(m['message']['content']) + [
        {'type': 'tool_use', 'id': f'tu_{base + i}', 'name': n, 'input': {}}
        for i, n in enumerate(tools)
    ]
    return m


def tool_result(tool_use_id, text='ok'):
    return create_user_message(
        [{'type': 'tool_result', 'tool_use_id': tool_use_id, 'content': text}]
    )


def is_resumed(messages):
    return any(
        (m.get('type') if isinstance(m, dict) else getattr(m, 'type', None)) == 'assistant'
        for m in messages
    )


class RecordingSlot:
    def __init__(self, log):
        self.log = log

    def release(self):
        self.log.append('released')


def make_ctx(parent):
    """The cli's shape: state hooks present and None, one turn-wide abort scope."""
    return ToolUseContext(
        abort_controller=parent,
        options={'dangerously_skip_permissions': False},
        set_app_state=None,
        get_app_state=None,
        tool_use_id='toolu_turn',
    )


def abort_watching_stub(seen, hold=5.0):
    """Steps once, then waits — but notices if its own scope is stopped.

    `seen` collects what the agent's abort scope read at each step, which is the
    thing under test: a background agent must never see the turn's ESC.
    """
    def _query(messages, system_prompt, context, can_use_tool, tool_use_context, **kw):
        async def _gen():
            scope = tool_use_context.abort_controller
            if is_resumed(messages):
                await asyncio.sleep(0.15)
                seen.append(('resumed', scope.is_set()))
                yield assistant('resumed and finished', base=9)
                return
            yield assistant('first step', tools=('Read',))
            yield tool_result('tu_0')
            seen.append(('step', scope.is_set()))
            for _ in range(int(hold / 0.02)):
                await asyncio.sleep(0.02)
                if scope.is_set():
                    seen.append(('stopped', True))
                    return
            seen.append(('never stopped', False))
            yield assistant('finished on its own')
        return _gen()
    return _query


async def wait_for_agent(progress_at_least=1, timeout=6.0):
    """The newest local_agent task, once it has produced a step."""
    agent_id = None
    for _ in range(int(timeout / 0.02)):
        await asyncio.sleep(0.02)
        for tid, t in get_app_state().tasks.items():
            if is_local_agent_task(t):
                agent_id = tid
        if agent_id is not None:
            t = get_app_state().tasks.get(agent_id)
            if t is not None and t.progress is not None \
                    and t.progress.tool_use_count >= progress_at_least:
                return agent_id
    return agent_id


async def main():
    original_query = query_mod.query
    original_try = subagent_gate.try_acquire
    try:
        # ---- the scope primitive -------------------------------------------
        parent = asyncio.Event()
        scope = child_of(parent)
        check('a linked scope reads its parent', (parent.set(), scope.is_set())[1])
        scope.detach()
        check('...and stops once detached', scope.is_set() is False)
        scope.set()
        check('...while its own stop still lands', scope.is_set() is True)

        # detach() has to reach `wait` too, not just `is_set` — make_abortable
        # races every tool pull against wait(), so a scope that reports unset
        # but still wakes on the parent would end the tool anyway.
        parent2 = asyncio.Event()
        scope2 = child_of(parent2)
        scope2.detach()
        waiter = asyncio.ensure_future(scope2.wait())
        parent2.set()
        await asyncio.sleep(0.05)
        check('a detached scope does not wake on the parent', waiter.done() is False)
        scope2.set()
        await asyncio.wait_for(waiter, timeout=1.0)
        check('...and still wakes on its own stop', waiter.done() is True)

        # ---- foreground: ESC must still end it -----------------------------
        reset_app_state()
        mq.reset_pending_notifications()
        release_log = []
        subagent_gate.try_acquire = lambda: RecordingSlot(release_log)
        seen = []
        query_mod.query = abort_watching_stub(seen)
        from codemason.tools.agent_tool import AgentTool

        turn_abort = asyncio.Event()
        out = []

        async def _drive_fg():
            async for r in AgentTool.call(
                {'prompt': 'survey the lane', 'description': 'foreground survey'},
                make_ctx(turn_abort),
            ):
                out.append(r)

        driver = asyncio.ensure_future(_drive_fg())
        fg_id = await wait_for_agent()
        check('a foreground agent registers', fg_id is not None)
        turn_abort.set()                      # ESC
        await asyncio.sleep(0.3)
        check('ESC reaches a FOREGROUND agent',
              ('stopped', True) in seen, seen)
        driver.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await driver

        # ---- background from start: ESC must not touch it -------------------
        reset_app_state()
        mq.reset_pending_notifications()
        release_log = []
        subagent_gate.try_acquire = lambda: RecordingSlot(release_log)
        seen = []
        query_mod.query = abort_watching_stub(seen, hold=1.2)
        turn_abort = asyncio.Event()
        out = []
        async for r in AgentTool.call(
            {'prompt': 'survey the lane', 'description': 'detached survey',
             'run_in_background': True},
            make_ctx(turn_abort),
        ):
            out.append(r)
        check('run_in_background answers at once',
              out and out[-1].data.get('status') == 'async_launched',
              out[-1].data if out else None)
        bg_id = out[-1].data['agentId']
        task = get_app_state().tasks.get(bg_id)
        check('...and its scope is not linked to the turn',
              task is not None and task.abort_controller is not None
              and task.abort_controller.is_set() is False)
        await asyncio.sleep(0.1)
        turn_abort.set()                      # ESC on the main thread
        check('the turn is stopped but the agent is not',
              turn_abort.is_set() is True and task.abort_controller.is_set() is False)
        for _ in range(200):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(bg_id)
            if t is not None and t.status != 'running':
                break
        t = get_app_state().tasks.get(bg_id)
        check('ESC does NOT reach a BACKGROUND agent — it runs to completion',
              t is not None and t.status == 'completed'
              and ('never stopped', False) in seen, (t.status if t else None, seen))
        mq.drain_pending_notifications()

        # ...and the explicit stop still ends it, which is the other half of
        # the contract: unlinked is not unkillable.
        reset_app_state()
        mq.reset_pending_notifications()
        seen = []
        query_mod.query = abort_watching_stub(seen, hold=5.0)
        turn_abort = asyncio.Event()
        out = []
        async for r in AgentTool.call(
            {'prompt': 'survey the lane', 'description': 'stoppable survey',
             'run_in_background': True},
            make_ctx(turn_abort),
        ):
            out.append(r)
        bg_id = out[-1].data['agentId']
        await wait_for_agent()
        from codemason.tasks.local_agent_task import kill_async_agent
        kill_async_agent(bg_id, set_app_state)
        await asyncio.sleep(0.3)
        check('an explicit stop still ends a background agent',
              ('stopped', True) in seen, seen)
        mq.drain_pending_notifications()

        # ---- moved to the background mid-run: ESC must not follow it --------
        reset_app_state()
        mq.reset_pending_notifications()
        release_log = []
        subagent_gate.try_acquire = lambda: RecordingSlot(release_log)
        seen = []
        query_mod.query = abort_watching_stub(seen, hold=5.0)
        turn_abort = asyncio.Event()
        out = []

        async def _drive_moved():
            async for r in AgentTool.call(
                {'prompt': 'survey the lane', 'description': 'moved survey'},
                make_ctx(turn_abort),
            ):
                out.append(r)

        driver = asyncio.ensure_future(_drive_moved())
        mv_id = await wait_for_agent()
        check('the agent to be moved registers', mv_id is not None)
        check('background_agent_task moves it',
              background_agent_task(mv_id, get_app_state, set_app_state) is True)
        await asyncio.wait_for(driver, timeout=10)
        check('...and the parent is handed async_launched',
              out and out[-1].data.get('status') == 'async_launched',
              out[-1].data if out else None)
        turn_abort.set()                      # ESC right after the hand-over
        for _ in range(300):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(mv_id)
            if t is not None and t.status != 'running':
                break
        t = get_app_state().tasks.get(mv_id)
        check('ESC after the hand-over does not kill the moved agent',
              t is not None and t.status == 'completed'
              and t.result['text'] == 'resumed and finished',
              (t.status if t else None, t.result if t else None))
        check('...and the resumed run never saw a stop',
              ('resumed', False) in seen, seen)
        mq.drain_pending_notifications()

        # ---- the deliberate gesture ESC is not -----------------------------
        reset_app_state()
        mq.reset_pending_notifications()
        check('nothing running: the gesture is a no-op',
              kill_all_agents_and_notify() is None)
        check('...and says nothing to the model',
              mq.drain_pending_notifications() == [])

        scopes = {}
        for i, desc in enumerate(('read the lane', 'count the blocks', 'write it up')):
            aid = 'a' + str(i) * 16
            scopes[aid] = child_of(None)
            register_async_agent(
                agent_id=aid, description=desc, prompt='p', selected_agent=None,
                abort_controller=scopes[aid], tool_use_id=f'toolu_{i}',
            )
        summary = kill_all_agents_and_notify()
        check('three agents: every scope is stopped',
              all(s.is_set() for s in scopes.values()))
        check('...every task is marked killed',
              all(get_app_state().tasks[a].status == 'killed' for a in scopes),
              {a: get_app_state().tasks[a].status for a in scopes})
        notes = mq.drain_pending_notifications()
        check('...and the model is told ONCE, not three times',
              len(notes) == 1, notes)
        check('...naming all three',
              notes and all(f'"{d}"' in notes[0]
                            for d in ('read the lane', 'count the blocks', 'write it up')),
              notes)
        check('...counting them', notes and notes[0].startswith('3 background agents'), notes)
        check('...and the gesture returns what it said', summary == (notes[0] if notes else None))

        reset_app_state()
        mq.reset_pending_notifications()
        one = child_of(None)
        register_async_agent(agent_id='b' + '0' * 16, description='the only one',
                            prompt='p', selected_agent=None, abort_controller=one)
        kill_all_agents_and_notify()
        notes = mq.drain_pending_notifications()
        check('one agent gets the singular phrasing',
              notes and notes[0] == 'Background agent "the only one" was stopped by the user.',
              notes)

    finally:
        query_mod.query = original_query
        subagent_gate.try_acquire = original_try

    print(f'\n{sum(results)}/{len(results)} checks passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
