#!/usr/bin/env python3
"""A stopped sub-agent can be told one more thing instead of being replaced.

An agent that was killed, or that finished, still has everything needed to carry
on: its conversation is on disk (append_agent_transcript) and its task record
says what kind of agent it was. SendMessage used to refuse — honestly, but the
only way to continue that agent's work was to spawn a fresh one and re-describe
the whole task, throwing away the transcript sitting right there.

The three filters are the interesting part. An interrupted conversation is
malformed in three different ways, and each filter removes one; getting the
first one's rule wrong (ANY vs ALL unresolved tool calls) would silently drop a
turn that had done real work.
"""
import asyncio
import json
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
    append_agent_transcript,
    complete_agent_task,
    get_agent_transcript,
    kill_async_agent,
    register_async_agent,
)
from codemason.tools.resume_agent import load_resumable_messages, resume_agent_background
from codemason.types.query import ToolUseContext
from codemason.utils import message_queue_manager as mq
from codemason.utils import subagent_gate
from codemason.utils.agent_id import create_agent_id
from codemason.utils.messages import (
    create_assistant_message,
    create_user_message,
    filter_orphaned_thinking_only_messages,
    filter_unresolved_tool_uses,
    filter_whitespace_only_assistant_messages,
)
from codemason.utils.task.disk_output import cleanup_task_output, get_task_output_path


def assistant(text=None, tools=(), base=0, msg_id=None, thinking=None):
    m = create_assistant_message(text or 'placeholder')
    if not text:
        m['message']['content'] = []
    m['message']['usage'] = {'input_tokens': 500, 'output_tokens': 20,
                             'cache_read_input_tokens': 0,
                             'cache_creation_input_tokens': 0}
    content = list(m['message']['content'])
    if thinking is not None:
        content = [{'type': 'thinking', 'thinking': thinking}]
    content += [{'type': 'tool_use', 'id': f'tu_{base + i}', 'name': n, 'input': {}}
                for i, n in enumerate(tools)]
    m['message']['content'] = content
    if msg_id:
        m['message']['id'] = msg_id
    return m


def tool_result(tool_use_id):
    return create_user_message(
        [{'type': 'tool_result', 'tool_use_id': tool_use_id, 'content': 'ok'}])


class Slot:
    def release(self):
        pass


def stub():
    def _q(messages, system_prompt, context, can_use_tool, tool_use_context, **kw):
        async def _g():
            # Report what it was handed, so the test can assert the resume
            # really continued the conversation rather than starting over.
            yield assistant(f'resumed with {len(messages)} messages')
        return _g()
    return _q


def ctx():
    return ToolUseContext(
        abort_controller=asyncio.Event(),
        options={'dangerously_skip_permissions': False},
        set_app_state=None, get_app_state=None,   # the shape the real cli passes
        tool_use_id='toolu_resume',
    )


async def main():
    original_query = query_mod.query
    original_try = subagent_gate.try_acquire
    original_acq = subagent_gate.acquire
    try:
        # ---- the filters ---------------------------------------------------
        # ALL unresolved, not ANY: a turn that got some results back is kept.
        partly = [assistant('two calls', tools=('Read', 'Grep')), tool_result('tu_0')]
        check('a turn with SOME results back survives',
              len(filter_unresolved_tool_uses(partly)) == 2,
              len(filter_unresolved_tool_uses(partly)))
        wholly = [assistant('one call', tools=('Read',)), tool_result('tu_0'),
                  assistant('reaching', tools=('Bash',), base=5)]
        kept = filter_unresolved_tool_uses(wholly)
        check('...and the turn whose calls ALL went unanswered is dropped',
              len(kept) == 2, len(kept))
        check('a conversation with nothing unresolved is returned untouched',
              filter_unresolved_tool_uses(partly[:2]) == partly[:2])

        blank = [assistant('   '), assistant('real')]
        check('a turn that said only whitespace is dropped',
              len(filter_whitespace_only_assistant_messages(blank)) == 1)
        check('...but an EMPTY content list is kept, which means something else',
              len(filter_whitespace_only_assistant_messages([assistant(None)])) == 1)

        orphan = [assistant(thinking='hmm', msg_id='m1')]
        check('thinking with nothing to merge into is dropped',
              filter_orphaned_thinking_only_messages(orphan) == [])
        paired = [assistant(thinking='hmm', msg_id='m1'),
                  assistant('and here it is', msg_id='m1')]
        check('...but thinking that shares an id with real content is kept',
              len(filter_orphaned_thinking_only_messages(paired)) == 2,
              len(filter_orphaned_thinking_only_messages(paired)))

        # ---- the transcript round trip -------------------------------------
        reset_app_state()
        agent_id = create_agent_id()
        cleanup_task_output(agent_id)
        register_async_agent(agent_id=agent_id, description='survey the lane',
                             prompt='survey the lane', selected_agent=None,
                             set_app_state_fn=set_app_state)
        check('an agent with no transcript cannot be resumed',
              load_resumable_messages(agent_id) is None)

        for m in (assistant('reading the config', tools=('Read',)),
                  tool_result('tu_0'),
                  assistant('lane 3 is the one'),
                  # The interrupted tail: a call that never came back.
                  assistant('reaching for a tool', tools=('Bash',), base=7)):
            append_agent_transcript(agent_id, m)

        raw = get_agent_transcript(agent_id)
        check('the transcript reads back in order', [m['type'] for m in raw]
              == ['assistant', 'user', 'assistant', 'assistant'],
              [m['type'] for m in raw])
        resumable = load_resumable_messages(agent_id)
        check('...and the interrupted tail is filtered out of it',
              len(resumable) == 3, len(resumable))

        # ---- resuming a killed agent ---------------------------------------
        mq.reset_pending_notifications()
        query_mod.query = stub()
        subagent_gate.try_acquire = lambda: Slot()
        async def _acq(_abort):
            return Slot()
        subagent_gate.acquire = _acq

        kill_async_agent(agent_id, set_app_state)
        check('the agent is stopped before the resume', 
              get_app_state().tasks[agent_id].status == 'killed')

        out = await resume_agent_background(
            agent_id=agent_id, prompt='also check beta.py',
            tool_use_context=ctx(), can_use_tool=None,
            get_app_state_fn=get_app_state, set_app_state_fn=set_app_state)
        check('a stopped agent is resumed', out['resumed'] is True, out)
        check('...under the SAME id, so its name still points at it',
              out['agentId'] == agent_id)
        check('...saying how much conversation it picked up',
              '3 messages' in out['message'], out['message'])
        check('...and it is running again', 
              get_app_state().tasks[agent_id].status == 'running',
              get_app_state().tasks[agent_id].status)
        check('...keeping its description rather than inventing one',
              get_app_state().tasks[agent_id].description == 'survey the lane')

        for _ in range(300):
            await asyncio.sleep(0.02)
            t = get_app_state().tasks.get(agent_id)
            if t is not None and t.status != 'running':
                break
        t = get_app_state().tasks.get(agent_id)
        check('the resumed run completes', t is not None and t.status == 'completed',
              t.status if t else None)
        # 3 filtered + the new prompt = 4 handed to the model.
        check('...having been handed the transcript AND the new message',
              t.result['text'] == 'resumed with 4 messages', t.result)
        notes = mq.drain_pending_notifications()
        check('...and announces itself, since nothing was awaiting it',
              len(notes) == 1 and '<status>completed</status>' in notes[0],
              len(notes))

        # ---- the chain has to survive the stop -----------------------------
        # A terminal transition drops the in-memory chain, and the resume then
        # appends to the SAME file. Writing a second root there made everything
        # before the resume unreachable from the newest leaf — the reader walks
        # parentUuid back from it — so a resumed agent was handed one message
        # instead of its whole history, and a second resume lost the first
        # resume's work too. Caught end-to-end: a real run reported "(1
        # messages)" for an agent that had read two files, and the resumed agent
        # hallucinated a third file it had never seen.
        reset_app_state()
        chained = create_agent_id()
        cleanup_task_output(chained)
        register_async_agent(agent_id=chained, description='d', prompt='p',
                             selected_agent=None, set_app_state_fn=set_app_state)
        for m in (assistant('step one', tools=('Read',)), tool_result('tu_0'),
                  assistant('done one')):
            append_agent_transcript(chained, m)
        check('three turns before the stop', len(load_resumable_messages(chained)) == 3)

        kill_async_agent(chained, set_app_state)
        register_async_agent(agent_id=chained, description='d', prompt='next',
                             selected_agent=None, set_app_state_fn=set_app_state)
        append_agent_transcript(chained, assistant('step two'))

        entries = [json.loads(line) for line
                   in open(get_task_output_path(chained), encoding='utf-8')
                   if line.strip()]
        roots = [e for e in entries if not e.get('parentUuid')]
        check('the file has ONE root, not one per resume', len(roots) == 1, len(roots))
        check('...so the whole history is still reachable',
              len(load_resumable_messages(chained)) == 4,
              len(load_resumable_messages(chained)))

        kill_async_agent(chained, set_app_state)
        register_async_agent(agent_id=chained, description='d', prompt='again',
                             selected_agent=None, set_app_state_fn=set_app_state)
        append_agent_transcript(chained, assistant('step three'))
        check('...and a SECOND resume keeps the first resume\'s work',
              len(load_resumable_messages(chained)) == 5,
              len(load_resumable_messages(chained)))
        cleanup_task_output(chained)

        # ---- the refusals that must stay refusals --------------------------
        reset_app_state()
        other = create_agent_id()
        register_async_agent(agent_id=other, description='busy', prompt='p',
                             selected_agent=None, set_app_state_fn=set_app_state)
        out = await resume_agent_background(
            agent_id=other, prompt='x', tool_use_context=ctx(),
            get_app_state_fn=get_app_state, set_app_state_fn=set_app_state)
        check('a RUNNING agent is not resumed — there is nothing to resume',
              out['resumed'] is False and 'still running' in out['message'], out)

        out = await resume_agent_background(
            agent_id='a' + '0' * 16, prompt='x', tool_use_context=ctx(),
            get_app_state_fn=get_app_state, set_app_state_fn=set_app_state)
        check('an id with no task is not resumed',
              out['resumed'] is False and 'No agent task' in out['message'], out)

        cleanup_task_output(agent_id)
        cleanup_task_output(other)
    finally:
        query_mod.query = original_query
        subagent_gate.try_acquire = original_try
        subagent_gate.acquire = original_acq

    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
