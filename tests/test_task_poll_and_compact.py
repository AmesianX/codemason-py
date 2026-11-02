#!/usr/bin/env python3
"""The bookkeeping pass, and what the model is told about its agents after a compaction.

The poll pass is the part that looks wrong until you read why: it emits no
attachments. The source builds the list and never fills it, because each task
type announces its own completion — generating one here would deliver the same
completion twice. What the pass actually does is advance output offsets and drop
terminal tasks whose notice has been consumed.

The compaction attachments are the opposite: without them the model comes out of
a compaction having forgotten it has agents running, and spawns them again.
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


from dataclasses import replace

from codemason.services.compaction import create_async_agent_attachments_if_needed
from codemason.state.app_state import get_app_state, reset_app_state, set_app_state
from codemason.task import create_task_state_base, now_ms
from codemason.tasks.local_agent_task import (
    AgentProgress,
    PANEL_GRACE_MS,
    complete_agent_task,
    fail_agent_task,
    kill_async_agent,
    register_agent_foreground,
    update_agent_summary,
)
from codemason.tasks.local_shell_task import LocalShellTaskState
from codemason.utils.abort import child_of
from codemason.utils.agent_id import create_agent_id
from codemason.utils.task.disk_output import get_task_output_path
from codemason.utils.task.framework import (
    apply_task_offsets_and_evictions,
    generate_task_attachments,
    get_status_text,
    poll_tasks,
    register_task,
)


class FakeAgent:
    agent_type = 'Explore'


def spawn(description='look around'):
    agent_id = create_agent_id()
    register_agent_foreground(
        agent_id=agent_id, description=description, prompt='find the thing',
        selected_agent=FakeAgent(), abort_controller=child_of(asyncio.Event()),
    )
    return agent_id


def patch(task_id, **kw):
    set_app_state(lambda prev: replace(
        prev, tasks={**prev.tasks, task_id: replace(prev.tasks[task_id], **kw)}))


def text_of(message):
    content = message['message']['content']
    return content[0]['text'] if isinstance(content, list) else content


async def main():
    # ---- the pass never announces anything ---------------------------------
    reset_app_state()
    agent_id = spawn()
    # newline='': write_text translates '\n' to '\r\n' on Windows, so the
    # file would be 12 bytes while the cases below count 11 and every offset
    # here would be one short. The offsets are byte counts, which is right --
    # it is the fixture that must not quietly grow.
    Path(get_task_output_path(agent_id)).write_text('first line\n', encoding='utf-8', newline='')
    found = generate_task_attachments(get_app_state())
    check('a running task yields no attachment — its own type announces it',
          found['attachments'] == [])
    check('...but its offset moves past what was read',
          found['updated_task_offsets'].get(agent_id) == len('first line\n'),
          found['updated_task_offsets'])
    check('...and nothing is evicted', found['evicted_task_ids'] == [])

    poll_tasks(get_app_state, set_app_state)
    check('the pass writes the offset back',
          get_app_state().tasks[agent_id].output_offset == len('first line\n'))

    Path(get_task_output_path(agent_id)).write_text('first line\nsecond\n', encoding='utf-8', newline='')
    poll_tasks(get_app_state, set_app_state)
    check('a second pass advances it again, reading only the new bytes',
          get_app_state().tasks[agent_id].output_offset == len('first line\nsecond\n'))

    poll_tasks(get_app_state, set_app_state)
    check('a pass with nothing new leaves the offset alone',
          get_app_state().tasks[agent_id].output_offset == len('first line\nsecond\n'))

    # ---- eviction respects notified and the grace period -------------------
    complete_agent_task(agent_id, {'text': 'done'}, set_app_state)
    poll_tasks(get_app_state, set_app_state)
    check('a finished but un-notified task is kept — its completion is unread',
          agent_id in get_app_state().tasks)

    patch(agent_id, notified=True)
    poll_tasks(get_app_state, set_app_state)
    check('...and still kept while its grace period runs',
          agent_id in get_app_state().tasks,
          f"evict_after={get_app_state().tasks[agent_id].evict_after} now={now_ms()}")

    patch(agent_id, evict_after=now_ms() - 1)
    poll_tasks(get_app_state, set_app_state)
    check('...then dropped once it has passed', agent_id not in get_app_state().tasks)

    # ---- the write-back re-checks the state it is writing into -------------
    reset_app_state()
    agent_id = spawn()
    Path(get_task_output_path(agent_id)).write_text('x' * 40, encoding='utf-8', newline='')
    found = generate_task_attachments(get_app_state())
    # The agent finishes in the window between the walk and the write-back.
    complete_agent_task(agent_id, {'text': 'raced'}, set_app_state)
    apply_task_offsets_and_evictions(
        set_app_state, found['updated_task_offsets'], found['evicted_task_ids'])
    t = get_app_state().tasks[agent_id]
    check('an offset for a task that finished mid-pass is dropped, not written',
          t.status == 'completed' and t.output_offset == 0,
          f'status={t.status} offset={t.output_offset}')
    check('...so the finish is not clobbered back to running', t.status == 'completed')

    reset_app_state()
    agent_id = spawn()
    complete_agent_task(agent_id, {'text': 'done'}, set_app_state)
    patch(agent_id, notified=True, evict_after=now_ms() - 1)
    found = generate_task_attachments(get_app_state())
    check('a consumed terminal task is picked for eviction',
          found['evicted_task_ids'] == [agent_id])
    # A resume replaces it in the window.
    replacement = create_task_state_base(agent_id, 'local_agent', 'resumed', None)
    patch(agent_id, status='running', notified=False, evict_after=None)
    apply_task_offsets_and_evictions(
        set_app_state, found['updated_task_offsets'], found['evicted_task_ids'])
    check('a task that came back to life mid-pass is not evicted',
          agent_id in get_app_state().tasks)

    # ---- a shell task goes through the same books --------------------------
    reset_app_state()
    base = create_task_state_base('b99999999', 'local_bash', 'run tests', None)
    register_task(LocalShellTaskState(**{**base, 'status': 'completed', 'notified': True},
                                      command='pytest', kind='bash'), set_app_state)
    poll_tasks(get_app_state, set_app_state)
    check('a consumed shell task is evicted with no grace period',
          'b99999999' not in get_app_state().tasks)

    check('the status wording is the source\'s', [
        get_status_text('completed'), get_status_text('failed'),
        get_status_text('killed'), get_status_text('running'), get_status_text('pending'),
    ] == ['completed successfully', 'failed', 'was stopped', 'is running', 'is pending'])

    # ---- what the model is told after a compaction -------------------------
    class Ctx:
        get_app_state = staticmethod(get_app_state)
        agent_id = None

    reset_app_state()
    running = spawn('survey the lane')
    update_agent_summary(running, 'reading the config', set_app_state)
    done = spawn('count the tokens')
    complete_agent_task(done, {'text': 'counted'}, set_app_state)
    broke = spawn('try the kpool')
    fail_agent_task(broke, 'kpool refused', set_app_state)
    stopped = spawn('the one the user stopped')
    kill_async_agent(stopped, set_app_state)

    msgs = create_async_agent_attachments_if_needed(Ctx())
    check('every agent the model should know about is reported', len(msgs) == 4, len(msgs))
    all_text = '\n'.join(text_of(m) for m in msgs)
    check('all of them are meta messages', all(m.get('isMeta') for m in msgs))
    check('they are system reminders', all('<system-reminder>' in text_of(m) for m in msgs))

    check('a running agent is named and warned against duplication',
          'Background agent "survey the lane"' in all_text
          and 'Do NOT spawn a duplicate.' in all_text)
    check('...with its summary as the progress line',
          'Progress: reading the config' in all_text)
    check('...and where to read partial output',
          f'You can read partial output at {get_task_output_path(running)}' in all_text)
    # The model has just lost the record of the spawn, so the reminder has to
    # tell it that the agent is still reachable — not only readable.
    check('...and that it can still send the agent a message',
          'or send it a message with SendMessage' in all_text, all_text)

    check('a finished agent is reported by id, type, status and brief',
          f'Task {done} (type: local_agent) (status: completed) (description: count the tokens)' in all_text,
          all_text)
    check('...and told where the result is',
          f'Read the output file to retrieve the result: {get_task_output_path(done)}' in all_text)
    check('a failed agent carries its error as the delta',
          'Delta: kpool refused' in all_text)
    check('a stopped agent gets the brief line, with no transcript delta',
          f'Task "the one the user stopped" ({stopped}) was stopped by the user.' in all_text)

    # ---- the three that are skipped ----------------------------------------
    reset_app_state()
    retrieved = spawn('already read')
    patch(retrieved, retrieved=True)
    check('a retrieved agent is not reported again',
          create_async_agent_attachments_if_needed(Ctx()) == [])

    reset_app_state()
    pending = spawn('not started')
    patch(pending, status='pending')
    check('a pending agent is not reported', create_async_agent_attachments_if_needed(Ctx()) == [])

    reset_app_state()
    mine = spawn('my own work')

    class SelfCtx:
        get_app_state = staticmethod(get_app_state)
        agent_id = mine

    check('an agent is not told about itself',
          create_async_agent_attachments_if_needed(SelfCtx()) == [])

    reset_app_state()
    base = create_task_state_base('b11111111', 'local_bash', 'run tests', None)
    register_task(LocalShellTaskState(**{**base, 'status': 'running'},
                                      command='pytest', kind='bash'), set_app_state)
    check('a shell task is not an agent attachment',
          create_async_agent_attachments_if_needed(Ctx()) == [])

    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
