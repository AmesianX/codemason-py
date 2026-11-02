#!/usr/bin/env python3
"""TaskOutput on a RUNNING sub-agent has to return what it is doing.

The source points a local_agent task's output path at the agent's own
transcript JSONL and says why in its own comment: the file is "the full session
transcript (every message, tool use, etc.), not just the subagent's answer",
which is exactly why it prefers the in-memory `result` once the run is over.

codemason had the reader and the preference but no writer, so the path stayed
an empty file for the whole life of the agent. A parent could read an agent's
token count and its tool count and never one word of its work — which is half
of the thing this subsystem exists for ("look at the sub-agent's context, stop
it if it is going wrong").
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
    _agent_transcript_chains,
    append_agent_transcript,
    complete_agent_task,
    forget_agent_transcript,
    create_activity_description_resolver,
    create_progress_tracker,
    get_progress_update,
    is_local_agent_task,
    register_async_agent,
    update_agent_progress,
    update_progress_from_message,
)
from codemason.tools.task_output_tool import TaskOutputTool
from codemason.types.query import ToolUseContext
from codemason.utils import message_queue_manager as mq
from codemason.utils.agent_id import create_agent_id
from codemason.utils.messages import create_assistant_message, create_user_message
from codemason.utils.session_storage import get_session_id
from codemason.utils.task.disk_output import cleanup_task_output, get_task_output, get_task_output_path


def assistant(text, tools=()):
    m = create_assistant_message(text or 'placeholder')
    if not text:
        m['message']['content'] = []
    m['message']['usage'] = {'input_tokens': 500, 'output_tokens': 20,
                             'cache_read_input_tokens': 0,
                             'cache_creation_input_tokens': 0}
    m['message']['content'] = list(m['message']['content']) + [
        {'type': 'tool_use', 'id': f'tu_{i}', 'name': n, 'input': {'pattern': n}}
        for i, n in enumerate(tools)
    ]
    return m


async def task_output(agent_id):
    ctx = ToolUseContext(
        abort_controller=asyncio.Event(),
        options={'dangerously_skip_permissions': False},
        set_app_state=set_app_state, get_app_state=get_app_state,
    )
    out = []
    async for r in TaskOutputTool.call({'task_id': agent_id}, ctx):
        out.append(r)
    return out[-1].data


async def main():
    reset_app_state()
    mq.reset_pending_notifications()
    agent_id = create_agent_id()
    cleanup_task_output(agent_id)
    register_async_agent(
        agent_id=agent_id, description='survey the lane', prompt='survey the lane',
        selected_agent=None, set_app_state_fn=set_app_state,
    )

    # ---- nothing written yet -------------------------------------------
    check('a freshly spawned agent has an output file, empty',
          Path(get_task_output_path(agent_id)).exists()
          and get_task_output(agent_id) == '',
          repr(get_task_output(agent_id)[:40]))

    # ---- the run writes as it goes -------------------------------------
    # Exactly what the two run loops do per message: write the line, fold the
    # message into the tally, publish the tally.
    tracker = create_progress_tracker()
    resolve = create_activity_description_resolver([])
    for msg in (assistant('reading the lane config', tools=('Grep',)),
                create_user_message('tool said: three lanes'),
                assistant('lane 3 is the one')):
        append_agent_transcript(agent_id, msg)
        update_progress_from_message(tracker, msg, resolve, [])
        update_agent_progress(agent_id, get_progress_update(tracker), set_app_state)

    body = get_task_output(agent_id)
    lines = [l for l in body.splitlines() if l.strip()]
    check('every message is one JSONL line', len(lines) == 3, len(lines))
    parsed = [json.loads(l) for l in lines]
    check('...and each line parses', all(isinstance(e, dict) for e in parsed))
    check('...in the order they happened',
          [e['type'] for e in parsed] == ['assistant', 'user', 'assistant'],
          [e['type'] for e in parsed])
    check('...carrying the work, not just a status',
          'reading the lane config' in body and 'Grep' in body,
          body[:120])
    check('...marked as a sidechain, which is what a sub-agent is',
          all(e.get('isSidechain') is True for e in parsed))
    check('...and naming the agent, so a line ties back to its task',
          all(e.get('agentId') == agent_id for e in parsed))
    check('...chained: each entry names the one before it',
          parsed[0]['parentUuid'] is None
          and parsed[1]['parentUuid'] == parsed[0]['uuid']
          and parsed[2]['parentUuid'] == parsed[1]['uuid'],
          [(e['uuid'][:8], (e['parentUuid'] or '-')[:8]) for e in parsed])
    check('...without touching the main session, whose chain is its own',
          parsed[0].get('sessionId') == get_session_id())

    # ---- which is what the parent reads mid-run ------------------------
    data = await task_output(agent_id)
    task = data.get('task') or {}
    check('TaskOutput on a running agent reports it running',
          task.get('status') == 'running', data)
    check('...and hands back its work rather than an empty string',
          'reading the lane config' in (task.get('output') or ''),
          repr((task.get('output') or '')[:80]))
    check('...with the counts it is accounted by', 'usage' in task, list(task))

    # ---- once it finishes, the clean answer wins ------------------------
    complete_agent_task(agent_id, {'agent_id': agent_id, 'text': 'lane 3 is the one',
                                   'tool_use_count': 1, 'duration': 1.0}, set_app_state)
    data = await task_output(agent_id)
    task = data.get('task') or {}
    check('a finished agent answers with its result, not the raw transcript',
          task.get('result') == 'lane 3 is the one', repr(task.get('result'))[:80])
    check('...which is why the file is only the fallback',
          'isSidechain' not in (task.get('output') or ''),
          repr((task.get('output') or '')[:60]))

    # ---- and the chain state does not accumulate -----------------------
    check('finishing drops the agent\'s chain state',
          agent_id not in _agent_transcript_chains,
          list(_agent_transcript_chains))
    forget_agent_transcript(agent_id)  # idempotent
    check('...and dropping it twice is harmless',
          agent_id not in _agent_transcript_chains)

    # ---- a progress message is not a transcript entry ------------------
    other = create_agent_id()
    cleanup_task_output(other)
    append_agent_transcript(other, {'type': 'progress', 'data': {}})
    check('progress messages are not written, here as in the source',
          get_task_output(other) == '', repr(get_task_output(other)[:40]))

    cleanup_task_output(agent_id)
    cleanup_task_output(other)
    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
