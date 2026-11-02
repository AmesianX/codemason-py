#!/usr/bin/env python3
"""A foreground sub-agent can be moved to the background mid-run.

Two things fire the switch: the auto-background timer the spawn arms
(getAutoBackgroundMs, off unless asked for) and an explicit
background_agent_task. Either way the parent stops waiting and is told where to
collect the result, while the run itself carries on detached.

What has to hold at that moment:
  * the parent is answered with async_launched, not an empty result;
  * the task survives the foreground teardown, because its result is collected
    later;
  * the run RESUMES — the conversation so far is handed over, minus any tool
    call that never came back, which would be rejected outright if sent again;
  * the counts the parent reads carry on rather than rewinding to zero;
  * the gate slot travels with the agent, because a backgrounded agent is still
    occupying the pool.
"""
import asyncio
import os
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
    register_agent_foreground,
    unregister_agent_foreground,
)
from codemason.tools.agent_tool_utils import filter_incomplete_tool_calls
from codemason.types.query import ToolUseContext
from codemason.utils import message_queue_manager as mq
from codemason.utils import subagent_gate
from codemason.utils.messages import create_assistant_message, create_user_message


def assistant(text, inp=500, out=20, tools=(), base=0):
    m = create_assistant_message(text or 'placeholder')
    if not text:
        m['message']['content'] = []
    m['message']['usage'] = {'input_tokens': inp, 'output_tokens': out,
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
    """A resumed run is handed a conversation that already has assistant turns."""
    return any(
        (m.get('type') if isinstance(m, dict) else getattr(m, 'type', None)) == 'assistant'
        for m in messages
    )


def two_phase_stub(hold=10.0):
    """Yields one complete step, then parks. A resume answers and finishes.

    The park is where the backgrounding lands, which is the realistic case: the
    switch fires while the agent is in the middle of a step, not between them.
    """
    def _query(messages, system_prompt, context, can_use_tool, tool_use_context, **kw):
        async def _gen():
            if is_resumed(messages):
                yield assistant('resumed and finished', inp=900, out=30,
                                tools=('Grep',), base=9)
                yield tool_result('tu_9')
                return
            yield assistant('foreground step', tools=('Read',))
            yield tool_result('tu_0')
            await asyncio.sleep(hold)
            yield assistant('the foreground never gets here')
        return _gen()
    return _query


def orphan_stub(hold=10.0):
    """Parks with a tool call still in flight — no tool_result for it."""
    def _query(messages, system_prompt, context, can_use_tool, tool_use_context, **kw):
        async def _gen():
            if is_resumed(messages):
                yield assistant('resumed after the orphan was dropped')
                return
            yield assistant('step one', tools=('Read',))
            yield tool_result('tu_0')
            yield assistant('reaching for a tool', tools=('Bash',), base=5)
            await asyncio.sleep(hold)
        return _gen()
    return _query


class RecordingSlot:
    """A gate slot that remembers whether it was released."""

    def __init__(self, log):
        self.log = log

    def release(self):
        self.log.append('released')


def make_ctx(wire_state=True):
    """The tool's context.

    `wire_state=False` is the shape the REAL cli builds: the attributes EXIST
    and are None. Passing them explicitly everywhere is what let a
    `getattr(context, 'get_app_state', get_app_state)` through review — getattr
    finds the None, the default never applies, and calling it killed the tool
    the moment the auto-background timer fired. Every case here that can run
    without them does.
    """
    return ToolUseContext(
        abort_controller=asyncio.Event(),
        options={'dangerously_skip_permissions': False},
        set_app_state=set_app_state if wire_state else None,
        get_app_state=get_app_state if wire_state else None,
        tool_use_id='toolu_fg',
    )


async def drive_until_backgrounded(stub, release_log, wire_state=True):
    """Run a foreground agent, flip it to the background, collect its results."""
    reset_app_state()
    mq.reset_pending_notifications()
    query_mod.query = stub
    from codemason.tools.agent_tool import AgentTool

    out = []

    async def _drive():
        async for r in AgentTool.call(
            {'prompt': 'survey the lane', 'description': 'survey the lane'},
            make_ctx(wire_state),
        ):
            out.append(r)

    driver = asyncio.ensure_future(_drive())

    # Wait until the agent has registered AND produced its first step, so the
    # hand-over has something to carry.
    agent_id = None
    for _ in range(300):
        await asyncio.sleep(0.02)
        for tid, t in get_app_state().tasks.items():
            if is_local_agent_task(t):
                agent_id = tid
        if agent_id is not None:
            t = get_app_state().tasks.get(agent_id)
            if t is not None and t.progress is not None and t.progress.tool_use_count >= 1:
                break
    assert agent_id is not None, 'the agent never registered'

    moved = background_agent_task(agent_id, get_app_state, set_app_state)
    await asyncio.wait_for(driver, timeout=10)
    return agent_id, out, moved


async def main():
    original_query = query_mod.query
    original_try = subagent_gate.try_acquire
    try:
        release_log = []
        subagent_gate.try_acquire = lambda: RecordingSlot(release_log)

        # ---- the hand-over ------------------------------------------------
        agent_id, out, moved = await drive_until_backgrounded(two_phase_stub(), release_log)
        check('background_agent_task reports the move', moved is True)
        data = out[-1].data if out else {}
        check('the parent is answered with async_launched',
              data.get('status') == 'async_launched' and data.get('isAsync') is True, data)
        check('...naming the agent it can now address', data.get('agentId') == agent_id)
        check('...and where to collect the output',
              str(data.get('outputFile', '')).endswith(f'{agent_id}.output'),
              data.get('outputFile'))
        check('the task survives the foreground teardown',
              agent_id in get_app_state().tasks)
        check('...marked backgrounded', get_app_state().tasks[agent_id].is_backgrounded is True)
        check('the slot travels with the agent, not released by the parent',
              release_log == [], release_log)

        # ---- and the detached run finishes --------------------------------
        for _ in range(300):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(agent_id)
            if t is not None and t.status != 'running':
                break
        t = get_app_state().tasks.get(agent_id)
        check('the detached continuation completes',
              t is not None and t.status == 'completed', t.status if t else None)
        check('...and it RESUMED rather than restarting',
              t.result['text'] == 'resumed and finished', t.result if t else None)
        check('...counting the whole run, foreground steps included',
              t.progress.tool_use_count == 2, t.progress.tool_use_count if t else None)
        # latest input (900, the resumed turn) + every turn's output summed
        # across BOTH runs (20 + 30). The output half is what proves the
        # tracker was seeded rather than started fresh.
        check('...carrying the latest context size, not the first',
              t.progress.token_count == 950, t.progress.token_count if t else None)
        check('the slot is released once, by whoever finished the run',
              release_log == ['released'], release_log)

        notes = mq.drain_pending_notifications()
        check('it announces itself — nothing was awaiting it', len(notes) == 1, len(notes))
        note = notes[0] if notes else ''
        check('...as completed, carrying the answer',
              '<status>completed</status>' in note
              and '<result>resumed and finished</result>' in note, note)
        check('...stamped with the tool use that launched it',
              '<tool-use-id>toolu_fg</tool-use-id>' in note)

        # ---- the hand-over with the context the REAL cli builds ------------
        # Attributes present and None, which is what the cli passes. A
        # `getattr(context, 'get_app_state', get_app_state)` reads that None and
        # calls it, so this case is the one that catches the wrong idiom:
        # "'NoneType' object is not callable", every time the timer fired.
        release_log = []
        subagent_gate.try_acquire = lambda: RecordingSlot(release_log)
        agent_id, out, moved = await drive_until_backgrounded(
            two_phase_stub(), release_log, wire_state=False)
        check('the hand-over works on an unwired context too',
              moved is True and out
              and out[-1].data.get('status') == 'async_launched',
              out[-1].data if out else None)
        for _ in range(300):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(agent_id)
            if t is not None and t.status != 'running':
                break
        t = get_app_state().tasks.get(agent_id)
        check('...and the detached run still finishes',
              t is not None and t.status == 'completed'
              and t.result['text'] == 'resumed and finished',
              (t.status, t.result) if t else None)
        mq.drain_pending_notifications()

        # ---- an in-flight tool call is not handed over --------------------
        release_log = []
        subagent_gate.try_acquire = lambda: RecordingSlot(release_log)
        agent_id, out, _ = await drive_until_backgrounded(orphan_stub(), release_log)
        check('a hand-over mid-tool still answers async_launched',
              out and out[-1].data.get('status') == 'async_launched',
              out[-1].data if out else None)
        for _ in range(300):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(agent_id)
            if t is not None and t.status != 'running':
                break
        t = get_app_state().tasks.get(agent_id)
        check('...and the resumed run is accepted, the orphan dropped',
              t is not None and t.status == 'completed'
              and t.result['text'] == 'resumed after the orphan was dropped',
              (t.status, t.result) if t else None)
        mq.drain_pending_notifications()

        # ---- the timer -----------------------------------------------------
        reset_app_state()
        handle = register_agent_foreground(
            agent_id='a' + '0' * 16, description='timed', prompt='p',
            selected_agent=None, auto_background_delay_ms=30,
        )
        check('a spawn with a delay arms a cancellable timer',
              handle.cancel_auto_background is not None)
        check('...and its signal starts unset', handle.background_signal.is_set() is False)
        await asyncio.sleep(0.12)
        check('the timer fires the background signal',
              handle.background_signal.is_set() is True)
        check('...and flips the task, which is what the loop re-checks',
              get_app_state().tasks[handle.task_id].is_backgrounded is True)
        check('...so the foreground teardown leaves it alone',
              (unregister_agent_foreground(handle.task_id, set_app_state)
               or handle.task_id in get_app_state().tasks))

        reset_app_state()
        handle = register_agent_foreground(
            agent_id='a' + '1' * 16, description='cancelled', prompt='p',
            selected_agent=None, auto_background_delay_ms=30,
        )
        handle.cancel_auto_background()
        await asyncio.sleep(0.12)
        check('a cancelled timer never fires', handle.background_signal.is_set() is False)
        check('...leaving the task in the foreground',
              get_app_state().tasks[handle.task_id].is_backgrounded is False)

        reset_app_state()
        handle = register_agent_foreground(
            agent_id='a' + '2' * 16, description='untimed', prompt='p',
            selected_agent=None, auto_background_delay_ms=0,
        )
        check('no delay means no timer at all', handle.cancel_auto_background is None)
        await asyncio.sleep(0.05)
        check('...and nothing backgrounds it on its own',
              handle.background_signal.is_set() is False)

        # ---- closing the tool must not abandon the agent --------------------
        # The pull now lives in its own task rather than in an `async for`, so
        # closing the tool no longer sends GeneratorExit straight down into the
        # query generator. It still has to get there: an abandoned pull would
        # keep the sub-agent running — burning a slot and a lane — for a tool
        # call nobody is listening to. (It does, and in ~0.1ms; this pins it so
        # a later change to the loop cannot quietly take it away.)
        reset_app_state()
        mq.reset_pending_notifications()
        release_log = []
        subagent_gate.try_acquire = lambda: RecordingSlot(release_log)
        cleanup_ran = []

        def parking_stub():
            def _query(messages, system_prompt, context, can_use_tool,
                       tool_use_context, **kw):
                async def _gen():
                    try:
                        yield assistant('first step', tools=('Read',))
                        await asyncio.sleep(30)
                    finally:
                        cleanup_ran.append(True)
                return _gen()
            return _query

        query_mod.query = parking_stub()
        from codemason.tools.agent_tool import AgentTool
        gen = AgentTool.call(
            {'prompt': 'survey the lane', 'description': 'survey the lane'},
            make_ctx(),
        )
        # Drive it until the agent is actually inside the stream loop — the
        # tool yields progress before that, and closing it there would prove
        # nothing — then walk away.
        for _ in range(10):
            r = await gen.__anext__()
            if (r.data or {}).get('tool_name'):
                break
        await gen.aclose()
        for _ in range(100):
            await asyncio.sleep(0.02)
            if cleanup_ran:
                break
        check('closing the tool reaches the sub-agent\'s own cleanup',
              cleanup_ran == [True], cleanup_ran)
        check('...and releases the slot it was holding',
              release_log == ['released'], release_log)
        check('...and leaves no live task behind',
              all(not is_local_agent_task(t) or t.status != 'running'
                  for t in get_app_state().tasks.values()),
              [(t.id, t.status) for t in get_app_state().tasks.values()])

        # ---- the filter -----------------------------------------------------
        done = [create_user_message('go'), assistant('one', tools=('Read',)),
                tool_result('tu_0'), assistant('two')]
        check('a complete conversation passes through untouched',
              filter_incomplete_tool_calls(done) == done)
        orphaned = [create_user_message('go'), assistant('one', tools=('Read',)),
                    tool_result('tu_0'), assistant('reaching', tools=('Bash',), base=5)]
        kept = filter_incomplete_tool_calls(orphaned)
        check('an assistant turn whose tool never returned is dropped',
              len(kept) == 3 and kept == orphaned[:3], len(kept))
        check('...and the rest of the conversation is kept',
              [m['type'] for m in kept] == ['user', 'assistant', 'user'],
              [m['type'] for m in kept])
        check('a tool_use with its result is kept even out of order',
              len(filter_incomplete_tool_calls(
                  [tool_result('tu_0'), assistant('one', tools=('Read',))])) == 2)
        check('an empty conversation is empty', filter_incomplete_tool_calls([]) == [])
        check('a user-only conversation is untouched',
              len(filter_incomplete_tool_calls([create_user_message('go')])) == 1)
    finally:
        query_mod.query = original_query
        subagent_gate.try_acquire = original_try
        os.environ.pop('CODEMASON_AUTO_BACKGROUND_TASKS', None)

    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
