#!/usr/bin/env python3
"""The parent's two task tools reach a sub-agent, not only a shell task.

Three things have to be true for the parent to manage its agents: asking for an
agent's output has to return the agent's ANSWER rather than the raw transcript,
the counts the parent would act on — context size, tool uses — have to reach
the parent rather than a panel that does not exist here, and stopping one has
to actually stop it.
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


from codemason.state.app_state import get_app_state, reset_app_state, set_app_state
from codemason.tasks.local_agent_task import (
    AgentProgress,
    complete_agent_task,
    register_agent_foreground,
    update_agent_progress,
    update_agent_summary,
)
from codemason.tools.task_output_tool import _get_task_output_data, _map_result
from codemason.utils.abort import child_of
from codemason.utils.agent_id import create_agent_id
from codemason.utils.task.disk_output import get_task_output_path


class FakeAgent:
    agent_type = 'Explore'


def spawn(prompt='find the thing'):
    agent_id = create_agent_id()
    register_agent_foreground(
        agent_id=agent_id, description='look around', prompt=prompt,
        selected_agent=FakeAgent(), abort_controller=child_of(asyncio.Event()),
    )
    return agent_id


async def main():
    reset_app_state()
    agent_id = spawn()

    # A running agent with the transcript file holding something else entirely.
    Path(get_task_output_path(agent_id)).write_text(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"noise"}]}}\n',
        encoding='utf-8')
    update_agent_progress(agent_id, AgentProgress(tool_use_count=3, token_count=12345), set_app_state)
    update_agent_summary(agent_id, 'reading the lane config', set_app_state)

    data = _get_task_output_data(get_app_state().tasks[agent_id])
    check('a running agent reports its type', data['task_type'] == 'local_agent')
    check('...its status', data['status'] == 'running')
    check('...the brief it was given', data['prompt'] == 'find the thing')
    check('...and falls back to the file while it has no answer yet',
          'noise' in data['output'])
    check('...carrying the counts the parent would act on',
          data['usage'] == {'total_tokens': 12345, 'tool_uses': 3}, data.get('usage'))
    check('...and the summary', data['summary'] == 'reading the lane config')

    rendered = _map_result({'retrieval_status': 'success', 'task': data})
    check('the parent is actually told the context size',
          '<total_tokens>12345</total_tokens>' in rendered, rendered)
    check('...and the tool count', '<tool_uses>3</tool_uses>' in rendered)
    check('...and the summary', '<summary>reading the lane config</summary>' in rendered)

    # Once it finishes, the recorded answer wins over the transcript.
    complete_agent_task(
        agent_id,
        {'agent_id': agent_id, 'text': 'the thing is in lane 3',
         'usage': {'total_tokens': 99, 'tool_uses': 4, 'duration_ms': 10}},
        set_app_state,
    )
    data = _get_task_output_data(get_app_state().tasks[agent_id])
    check('a finished agent answers with its own text, not the transcript',
          data['result'] == 'the thing is in lane 3' and data['output'] == 'the thing is in lane 3',
          data['output'])
    check('...and reports completed', data['status'] == 'completed')
    rendered = _map_result({'retrieval_status': 'success', 'task': data})
    check('the answer is what reaches the parent',
          '<output>\nthe thing is in lane 3\n</output>' in rendered, rendered)
    check('...with no transcript noise', 'noise' not in rendered)

    # A failure carries its error.
    reset_app_state()
    agent_id = spawn()
    from codemason.tasks.local_agent_task import fail_agent_task
    fail_agent_task(agent_id, 'the lane refused', set_app_state)
    data = _get_task_output_data(get_app_state().tasks[agent_id])
    check('a failed agent reports its error', data['error'] == 'the lane refused')
    check('...and says failed', data['status'] == 'failed')
    check('the error reaches the parent',
          '<error>the lane refused</error>' in _map_result(
              {'retrieval_status': 'success', 'task': data}))

    # A shell task is untouched by any of this.
    from dataclasses import replace as dc_replace
    from codemason.task import create_task_state_base
    from codemason.tasks.local_shell_task import LocalShellTaskState
    from codemason.utils.task.framework import register_task
    reset_app_state()
    base = create_task_state_base('b12345678', 'local_bash', 'run tests', None)
    register_task(LocalShellTaskState(**{**base, 'status': 'completed'},
                                      command='pytest', kind='bash',
                                      result={'code': 2, 'interrupted': False}),
                  set_app_state)
    data = _get_task_output_data(get_app_state().tasks['b12345678'])
    check('a shell task still reports its exit code', data['exitCode'] == 2)
    check('...and grows no agent fields', 'prompt' not in data and 'usage' not in data)

    # ---- stopping one ------------------------------------------------------
    from codemason.tasks.stop_task import StopTaskError, stop_task

    reset_app_state()
    agent_id = spawn()
    scope = get_app_state().tasks[agent_id].abort_controller
    stopped = await stop_task(agent_id, get_app_state, set_app_state)
    check('TaskStop reaches a sub-agent', stopped['taskType'] == 'local_agent')
    check('...and names it by the brief it was spawned with',
          stopped['command'] == 'look around', stopped['command'])
    t = get_app_state().tasks[agent_id]
    check('...marking it killed', t.status == 'killed')
    check('...and telling the agent itself to stop', scope.is_set())
    # The source suppresses the notification for a shell task only: "Agent
    # tasks: don't suppress — the AbortError catch sends a notification
    # carrying extractPartialResult(agentMessages), which is the payload not
    # noise."
    check('a stopped agent stays announceable, so its partial result is not lost',
          t.notified is False)

    raised = None
    try:
        await stop_task(agent_id, get_app_state, set_app_state)
    except StopTaskError as exc:
        raised = exc
    check('stopping it twice reports not_running',
          raised is not None and raised.code == 'not_running')

    raised = None
    try:
        await stop_task('anowhere0000000000', get_app_state, set_app_state)
    except StopTaskError as exc:
        raised = exc
    check('an unknown id reports not_found',
          raised is not None and raised.code == 'not_found')

    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
