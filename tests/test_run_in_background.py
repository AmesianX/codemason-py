#!/usr/bin/env python3
"""A backgrounded agent answers the parent at once and reports back later.

The tool call returns the moment the agent is registered, so nothing is awaiting
the run — which is why every exit from the detached driver has to announce
itself. The notification queue is the only route left back into the
conversation, and an agent that finishes without enqueueing one has done its
work in silence.
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


import codemason.query as query_mod
from codemason.state.app_state import get_app_state, reset_app_state, set_app_state
from codemason.tasks.local_agent_task import auto_background_ms, kill_async_agent
from codemason.tools.agent_tool_utils import extract_partial_result
from codemason.types.query import ToolUseContext
from codemason.utils import message_queue_manager as mq
from codemason.utils.messages import create_assistant_message


def assistant(text, inp=500, out=20, tools=()):
    m = create_assistant_message(text or 'placeholder')
    # create_assistant_message substitutes '(no content)' for an empty string,
    # so an assistant turn that is ONLY tool_use has to be built by emptying
    # the content list — which is the shape extract_partial_result has to skip.
    if not text:
        m['message']['content'] = []
    m['message']['usage'] = {'input_tokens': inp, 'output_tokens': out,
                             'cache_read_input_tokens': 0,
                             'cache_creation_input_tokens': 0}
    m['message']['content'] = list(m['message']['content']) + [
        {'type': 'tool_use', 'id': f'tu_{i}', 'name': n, 'input': {}}
        for i, n in enumerate(tools)
    ]
    return m


def stub(behaviour):
    def _query(messages, system_prompt, context, can_use_tool, tool_use_context, **kw):
        async def _gen():
            await asyncio.sleep(0.02)
            if behaviour == 'raise':
                raise RuntimeError('the lane refused')
            if behaviour == 'slow':
                yield assistant('still working', tools=('Read',))
                await asyncio.sleep(5)
                yield assistant('never reached')
                return
            yield assistant('found it in lane 3', tools=('Read', 'Grep'))
        return _gen()
    return _query


async def launch(behaviour, **extra):
    reset_app_state()
    mq.reset_pending_notifications()
    query_mod.query = stub(behaviour)
    from codemason.tools.agent_tool import AgentTool
    ctx = ToolUseContext(
        abort_controller=asyncio.Event(),
        options={'dangerously_skip_permissions': False},
        set_app_state=set_app_state, get_app_state=get_app_state,
        tool_use_id='toolu_bg',
    )
    out = []
    async for r in AgentTool.call(
        {'prompt': 'survey the lane', 'description': 'survey the lane',
         'run_in_background': True, **extra}, ctx
    ):
        out.append(r)
    return out[-1].data


async def main():
    original_query = query_mod.query
    try:
        # ---- the launch answers immediately -------------------------------
        data = await launch('ok')
        check('a background launch reports async_launched',
              data['status'] == 'async_launched' and data['isAsync'] is True, data)
        agent_id = data['agentId']
        check('...naming the agent so the parent can address it',
              agent_id and data['description'] == 'survey the lane')
        check('...and where its output will be',
              data['outputFile'].endswith(f'{agent_id}.output'), data['outputFile'])
        check('...saying whether the parent can read that file itself',
              data['canReadOutputFile'] is True)

        task = get_app_state().tasks[agent_id]
        check('the task is registered BACKGROUNDED from the start',
              task.is_backgrounded is True and task.status == 'running')
        check('...so the foreground teardown cannot remove it',
              agent_id in get_app_state().tasks)

        # ---- and the run finishes on its own ------------------------------
        for _ in range(100):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(agent_id)
            if t is not None and t.status != 'running':
                break
        t = get_app_state().tasks.get(agent_id)
        check('the detached run completes', t is not None and t.status == 'completed',
              t.status if t else None)
        check('...recording the answer', t.result['text'] == 'found it in lane 3')
        check('...and the counts', t.progress.token_count == 520 and t.progress.tool_use_count == 2,
              (t.progress.token_count, t.progress.tool_use_count))

        notes = mq.drain_pending_notifications()
        check('it announces itself — nothing was awaiting it', len(notes) == 1, len(notes))
        note = notes[0] if notes else ''
        check('...as completed', '<status>completed</status>' in note)
        check('...carrying the answer, which is the only delivery it gets',
              '<result>found it in lane 3</result>' in note, note)
        check('...and the usage the parent accounts it by',
              '<total_tokens>520</total_tokens>' in note and '<tool_uses>2</tool_uses>' in note)
        check('...stamped with the tool use that launched it',
              '<tool-use-id>toolu_bg</tool-use-id>' in note)
        check('...and marked notified so it is announced once',
              get_app_state().tasks[agent_id].notified is True)

        # ---- a failure is announced too ------------------------------------
        data = await launch('raise')
        agent_id = data['agentId']
        for _ in range(100):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(agent_id)
            if t is not None and t.status != 'running':
                break
        t = get_app_state().tasks.get(agent_id)
        check('a failing background agent is marked failed',
              t is not None and t.status == 'failed', t.status if t else None)
        check('...with its error on the task', t.error == 'the lane refused')
        notes = mq.drain_pending_notifications()
        check('...and announced rather than lost silently', len(notes) == 1)
        check('...naming the failure',
              'failed: the lane refused' in notes[0], notes[0] if notes else '')

        # ---- a stop keeps the partial result -------------------------------
        data = await launch('slow')
        agent_id = data['agentId']
        await asyncio.sleep(0.1)
        t = get_app_state().tasks.get(agent_id)
        check('a slow background agent is still running', t.status == 'running')
        check('...with progress already visible to the parent',
              t.progress is not None and t.progress.tool_use_count == 1,
              t.progress)
        kill_async_agent(agent_id, set_app_state)
        for _ in range(100):
            await asyncio.sleep(0.02)
            if mq.has_pending_notifications():
                break
        t = get_app_state().tasks.get(agent_id)
        check('stopping it marks it killed', t is not None and t.status == 'killed')
        notes = mq.drain_pending_notifications()
        check('...and it still announces itself', len(notes) == 1, len(notes))
        check('...as stopped', '<status>killed</status>' in notes[0] if notes else False)
        check('...handing back what it had reached, not nothing',
              '<result>still working</result>' in notes[0] if notes else False,
              notes[0] if notes else '')

        # ---- the shared extractor -----------------------------------------
        check('the answer is the last assistant message that HAS text',
              extract_partial_result([
                  assistant('first'), assistant('second'),
                  assistant('', tools=('Read',)),
              ]) == 'second')
        check('...and None when there is no text at all',
              extract_partial_result([assistant('', tools=('Read',))]) is None)
        check('...which is also what a tool_use-only ending needs',
              extract_partial_result([]) is None)

        # ---- a named spawn is addressable by that name ---------------------
        data = await launch('slow', name='scout')
        agent_id = data['agentId']
        check('a background launch registers the name it was given',
              get_app_state().agent_name_registry.get('scout') == agent_id,
              get_app_state().agent_name_registry)
        kill_async_agent(agent_id, set_app_state)
        for _ in range(100):
            await asyncio.sleep(0.02)
            if mq.has_pending_notifications():
                break
        mq.drain_pending_notifications()

        data = await launch('ok')
        check('an unnamed one registers nothing',
              get_app_state().agent_name_registry == {},
              get_app_state().agent_name_registry)
        for _ in range(100):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(data['agentId'])
            if t is not None and t.status != 'running':
                break
        mq.drain_pending_notifications()

        # ---- auto-background is off unless asked for -----------------------
        check('the auto-background timer is off by default', auto_background_ms() == 0)
        import os
        os.environ['CODEMASON_AUTO_BACKGROUND_TASKS'] = '1'
        try:
            check('...and 120s when switched on', auto_background_ms() == 120_000)
        finally:
            del os.environ['CODEMASON_AUTO_BACKGROUND_TASKS']
        check('...off again once unset', auto_background_ms() == 0)
    finally:
        query_mod.query = original_query

    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
