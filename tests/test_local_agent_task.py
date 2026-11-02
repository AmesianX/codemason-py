#!/usr/bin/env python3
"""A sub-agent neither vanishes before its result is read nor outlives its owner.

Both halves have bitten this project before, and in the source they are two
different guards: task removal refuses to drop a backgrounded task, and spawning
registers a process-exit kill whose handle every terminal transition releases.
These tests pin both, plus the eviction arithmetic the parent's accounting reads
a finished agent back through.
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
from codemason.task import is_terminal_task_status
from codemason.task import create_task_state_base
from codemason.tasks.local_agent_task import (
    AgentProgress,
    MAX_RECENT_ACTIVITIES,
    ToolActivity,
    create_progress_tracker,
    get_progress_update,
    get_token_count_from_tracker,
    is_panel_agent_task,
    kill_all_running_agent_tasks,
    mark_agents_notified,
    update_agent_progress,
    update_agent_summary,
    update_progress_from_message,
    auto_background_ms,
    background_agent_task,
    complete_agent_task,
    register_agent_foreground,
    register_async_agent,
    fail_agent_task,
    LocalAgentTaskState,
    is_local_agent_task,
    kill_async_agent,
    unregister_agent_foreground,
)
from codemason.tasks.registry import get_task_by_type
from codemason.utils.abort import AbortScope, child_of
from codemason.utils.agent_id import create_agent_id, is_agent_id
from codemason.utils.cleanup_registry import cleanup_function_count, run_cleanup_functions
from codemason.utils.task.framework import PANEL_GRACE_MS, evict_terminal_task, register_task
from dataclasses import replace


class FakeAgent:
    agent_type = 'Explore'


def spawn(**kw):
    agent_id = create_agent_id()
    scope = child_of(asyncio.Event())
    handle = register_agent_foreground(
        agent_id=agent_id,
        description='look around',
        prompt='find the thing',
        selected_agent=FakeAgent(),
        tool_use_id='toolu_1',
        abort_controller=scope,
        **kw,
    )
    return agent_id, scope, handle


def task_of(agent_id):
    return get_app_state().tasks.get(agent_id)


async def main():
    # ---- spawn registers a running task ------------------------------------
    reset_app_state()
    before = cleanup_function_count()
    agent_id, scope, handle = spawn()
    t = task_of(agent_id)
    check('spawn registers the agent as a local_agent task', is_local_agent_task(t))
    check('...with status running', t is not None and t.status == 'running')
    check('...the task id IS the agent id', t is not None and t.id == agent_id == handle.task_id)
    check('...and that id is a well-formed agent id', is_agent_id(agent_id) == agent_id)
    check('...carrying the agent type from the definition', t.agent_type == 'Explore')
    check('...its own abort scope, not the parent event', t.abort_controller is scope)
    check('...foreground, with an empty message queue',
          t.is_backgrounded is False and t.pending_messages == [])
    check('...not retained (nothing is watching it)', t.retain is False)
    check('spawn registers a process-exit cleanup', cleanup_function_count() == before + 1)
    check('...and hands back the handle that cancels it', t.unregister_cleanup is not None)
    check('no auto-background timer by default', handle.cancel_auto_background is None
          and auto_background_ms() == 0)

    # ---- completion is terminal and releases the runtime handles ------------
    complete_agent_task(agent_id, {'text': 'found it', 'tool_use_count': 3}, set_app_state)
    t = task_of(agent_id)
    check('completion marks the task completed', t.status == 'completed'
          and is_terminal_task_status(t.status))
    check('...keeps the result for the parent to read', t.result['text'] == 'found it')
    check('...sets an eviction deadline ~30s out',
          t.evict_after is not None and 0 < t.evict_after - t.end_time <= PANEL_GRACE_MS)
    check('...drops the abort scope, the cleanup handle and the definition',
          t.abort_controller is None and t.unregister_cleanup is None
          and t.selected_agent is None)
    check('...and releases the process-exit cleanup', cleanup_function_count() == before)

    # ---- the terminal guard ------------------------------------------------
    fail_agent_task(agent_id, 'too late', set_app_state)
    kill_async_agent(agent_id, set_app_state)
    check('a finished task cannot be re-marked', task_of(agent_id).status == 'completed')

    # ---- removal: foreground goes, background stays ------------------------
    unregister_agent_foreground(agent_id, set_app_state)
    check('a finished foreground task is removed', task_of(agent_id) is None)

    reset_app_state()
    agent_id, scope, _ = spawn()
    moved = background_agent_task(agent_id, get_app_state, set_app_state)
    check('backgrounding a running agent reports that it moved', moved is True)
    check('...and flips the flag', task_of(agent_id).is_backgrounded is True)
    check('backgrounding it twice reports nothing to do',
          background_agent_task(agent_id, get_app_state, set_app_state) is False)
    complete_agent_task(agent_id, {'text': 'bg done'}, set_app_state)
    unregister_agent_foreground(agent_id, set_app_state)
    check('a BACKGROUNDED task survives removal — its result is collected later',
          task_of(agent_id) is not None and task_of(agent_id).status == 'completed')

    # ---- the background signal ---------------------------------------------
    reset_app_state()
    agent_id, scope, handle = spawn()
    check('the background signal is unset while the agent is in the foreground',
          not handle.background_signal.is_set())
    background_agent_task(agent_id, get_app_state, set_app_state)
    check('...and is set the moment it moves', handle.background_signal.is_set())

    # ---- the async spawn path ----------------------------------------------
    reset_app_state()
    parent = asyncio.Event()
    agent_id = create_agent_id()
    scope = child_of(parent)
    state = register_async_agent(
        agent_id=agent_id, description='async one', prompt='go',
        selected_agent=FakeAgent(), abort_controller=scope,
    )
    check('the async spawn starts already backgrounded', state.is_backgrounded is True
          and task_of(agent_id).is_backgrounded is True)
    check('...under a scope the parent can still stop', scope.is_set() is False)
    parent.set()
    check('...which the parent reaches', scope.is_set() is True)

    # ---- kill aborts the scope it is holding --------------------------------
    reset_app_state()
    agent_id, scope, _ = spawn()
    kill_async_agent(agent_id, set_app_state)
    check('killing the task aborts that agent', scope.is_set() is True)
    check('...and marks it killed', task_of(agent_id).status == 'killed')

    # ---- a sibling is spared ------------------------------------------------
    reset_app_state()
    parent = asyncio.Event()
    a_id, a_scope = create_agent_id(), None
    a_scope = child_of(parent)
    register_agent_foreground(agent_id=a_id, description='a', prompt='a',
                      selected_agent=FakeAgent(), abort_controller=a_scope)
    b_id = create_agent_id()
    b_scope = child_of(parent)
    register_agent_foreground(agent_id=b_id, description='b', prompt='b',
                      selected_agent=FakeAgent(), abort_controller=b_scope)
    kill_async_agent(a_id, set_app_state)
    check('stopping one agent spares its sibling',
          a_scope.is_set() is True and b_scope.is_set() is False
          and task_of(b_id).status == 'running')

    # ---- the registry dispatches a stop ------------------------------------
    impl = get_task_by_type('local_agent')
    check('the registry knows how to stop a local_agent', impl is not None)
    await impl.kill(b_id, set_app_state)
    check('...and stopping through it kills the agent',
          task_of(b_id).status == 'killed' and b_scope.is_set() is True)

    # ---- process exit tears a running agent down ---------------------------
    reset_app_state()
    agent_id, scope, _ = spawn()
    run_cleanup_functions()
    check('a process dying mid-run kills the agent it owns',
          task_of(agent_id).status == 'killed' and scope.is_set() is True)

    # ---- retain exempts a task from eviction -------------------------------
    reset_app_state()
    agent_id, scope, _ = spawn()
    set_app_state(lambda prev: replace(
        prev, tasks={**prev.tasks, agent_id: replace(prev.tasks[agent_id], retain=True)}))
    complete_agent_task(agent_id, {'text': 'watched'}, set_app_state)
    check('a retained task gets NO eviction deadline', task_of(agent_id).evict_after is None)

    # ---- eviction honours the deadline -------------------------------------
    def mark(**kw):
        set_app_state(lambda prev: replace(
            prev, tasks={**prev.tasks, agent_id: replace(prev.tasks[agent_id], **kw)}))

    mark(notified=True)
    evict_terminal_task(agent_id, set_app_state)
    check('a retained task is never evicted', task_of(agent_id) is not None)

    mark(evict_after=10 ** 13)  # far future
    evict_terminal_task(agent_id, set_app_state)
    check('a finished task is kept until its deadline passes', task_of(agent_id) is not None)

    mark(evict_after=1)  # 1970
    evict_terminal_task(agent_id, set_app_state)
    check('...and evicted once it has', task_of(agent_id) is None)

    reset_app_state()
    agent_id, scope, _ = spawn()
    complete_agent_task(agent_id, {'text': 'x'}, set_app_state)
    mark(evict_after=1)
    evict_terminal_task(agent_id, set_app_state)
    check('an un-notified completion is never dropped', task_of(agent_id) is not None)

    # ---- the auto-background timer -----------------------------------------
    reset_app_state()
    agent_id, scope, handle = spawn(auto_background_delay_ms=20)
    check('an armed timer hands back its canceller', handle.cancel_auto_background is not None)
    await asyncio.sleep(0.08)
    check('the timer backgrounds the agent', task_of(agent_id).is_backgrounded is True)
    check('...and sets the background signal', handle.background_signal.is_set())

    reset_app_state()
    agent_id, scope, handle = spawn(auto_background_delay_ms=20)
    handle.cancel_auto_background()
    await asyncio.sleep(0.08)
    check('a cancelled timer leaves the agent in the foreground',
          task_of(agent_id).is_backgrounded is False
          and not handle.background_signal.is_set())

    # ---- the progress tracker ----------------------------------------------
    def assistant(inp, out, tools=(), cache_read=0):
        class U:
            input_tokens = inp
            output_tokens = out
            cache_read_input_tokens = cache_read
            cache_creation_input_tokens = 0

        class Inner:
            usage = U()
            content = [{'type': 'tool_use', 'name': t, 'input': {}} for t in tools]

        class M:
            type = 'assistant'
            message = Inner()

        return M()

    tracker = create_progress_tracker()
    update_progress_from_message(tracker, assistant(100, 10, ('Read',)))
    update_progress_from_message(tracker, assistant(250, 20, ('Grep',), cache_read=50))
    check('the prompt count is the LATEST, not a sum — it already includes the turns before',
          tracker.latest_input_tokens == 300, tracker.latest_input_tokens)
    check('completions are summed', tracker.cumulative_output_tokens == 30)
    check('the context size is prompt + completions',
          get_token_count_from_tracker(tracker) == 330)
    check('tool uses are counted', tracker.tool_use_count == 2)

    update_progress_from_message(tracker, {'type': 'user', 'message': {'content': []}})
    check('a non-assistant message changes nothing',
          get_token_count_from_tracker(tracker) == 330 and tracker.tool_use_count == 2)

    for i in range(MAX_RECENT_ACTIVITIES + 3):
        update_progress_from_message(tracker, assistant(300, 0, (f'T{i}',)))
    check('the activity list is capped',
          len(tracker.recent_activities) == MAX_RECENT_ACTIVITIES)
    snap = get_progress_update(tracker)
    check('the snapshot names the most recent activity',
          snap.last_activity.tool_name == f'T{MAX_RECENT_ACTIVITIES + 2}')
    check('...and copies the list rather than aliasing it',
          snap.recent_activities is not tracker.recent_activities)

    # ---- progress and summary do not clobber each other --------------------
    reset_app_state()
    agent_id, scope, _ = spawn()
    update_agent_summary(agent_id, 'reading the config', set_app_state)
    update_agent_progress(agent_id, AgentProgress(tool_use_count=7, token_count=1234), set_app_state)
    t = task_of(agent_id)
    check('a counts update keeps a summary it did not write',
          t.progress.summary == 'reading the config')
    check('...and stores the counts',
          t.progress.tool_use_count == 7 and t.progress.token_count == 1234)
    update_agent_summary(agent_id, 'now editing', set_app_state)
    t = task_of(agent_id)
    check('a summary update keeps the counts it did not write',
          t.progress.tool_use_count == 7 and t.progress.token_count == 1234
          and t.progress.summary == 'now editing')

    complete_agent_task(agent_id, {'text': 'x'}, set_app_state)
    update_agent_progress(agent_id, AgentProgress(tool_use_count=99), set_app_state)
    update_agent_summary(agent_id, 'too late', set_app_state)
    t = task_of(agent_id)
    check('a finished agent takes no more progress',
          t.progress.tool_use_count == 7 and t.progress.summary == 'now editing')
    unregister_agent_foreground(agent_id, set_app_state)

    # ---- notified without announcing ---------------------------------------
    reset_app_state()
    agent_id, scope, _ = spawn()
    mark_agents_notified(agent_id, set_app_state)
    check('marking notified sets the flag', task_of(agent_id).notified is True)
    mark_agents_notified(agent_id, set_app_state)
    check('...and is idempotent', task_of(agent_id).notified is True)

    # ---- one interrupt reaches every agent ---------------------------------
    reset_app_state()
    parent = asyncio.Event()
    ids, scopes = [], []
    for _i in range(3):
        aid = create_agent_id()
        sc = child_of(parent)
        register_agent_foreground(agent_id=aid, description='x', prompt='x',
                                  selected_agent=FakeAgent(), abort_controller=sc)
        ids.append(aid); scopes.append(sc)
    kill_all_running_agent_tasks(dict(get_app_state().tasks), set_app_state)
    check('a bulk stop reaches every running agent',
          all(task_of(i).status == 'killed' for i in ids)
          and all(sc.is_set() for sc in scopes))
    check('...without stopping the parent', not parent.is_set())

    # ---- the main session is not a panel agent -----------------------------
    reset_app_state()
    agent_id, scope, _ = spawn()
    check('a spawned agent is a panel agent', is_panel_agent_task(task_of(agent_id)))
    set_app_state(lambda prev: replace(
        prev, tasks={**prev.tasks, agent_id: replace(prev.tasks[agent_id],
                                                     agent_type='main-session')}))
    check('the main session is not', not is_panel_agent_task(task_of(agent_id)))
    check('...though it is still a local_agent task',
          is_local_agent_task(task_of(agent_id)))
    unregister_agent_foreground(agent_id, set_app_state)

    # ---- re-registering a task carries the UI-held state forward -----------
    reset_app_state()
    agent_id, scope, _ = spawn()
    set_app_state(lambda prev: replace(prev, tasks={**prev.tasks, agent_id: replace(
        prev.tasks[agent_id], retain=True, start_time=4242,
        pending_messages=['keep me'], messages=['transcript'], disk_loaded=True)}))
    replacement = LocalAgentTaskState(
        **{**create_task_state_base(agent_id, 'local_agent', 'resumed', None),
           'status': 'running'},
        agent_id=agent_id, prompt='resumed', agent_type='Explore')
    register_task(replacement, set_app_state)
    t = task_of(agent_id)
    check('a resume keeps the hold the user placed', t.retain is True)
    check('...the original start time, so ordering is stable', t.start_time == 4242)
    check('...a queued message, which would otherwise be dropped',
          t.pending_messages == ['keep me'])
    check('...and the transcript as it stands',
          t.messages == ['transcript'] and t.disk_loaded is True)
    check('...while taking the replacement\'s own fields', t.description == 'resumed')

    # ---- what the tool actually wires up -----------------------------------
    from codemason.tools import agent_tool
    src = Path(agent_tool.__file__).read_text(encoding='utf-8')
    check('the tool registers the agent as a task', 'register_agent_foreground(' in src)
    check('...marks it completed on the happy path', 'complete_agent_task(' in src)
    check('...marks it failed on an error', 'fail_agent_task(' in src)
    check('...killed on a cancel', 'kill_async_agent(' in src)
    check('...and removes it in the finally', 'unregister_agent_foreground(' in src)
    check('...cancelling the auto-background timer first',
          src.index('cancel_auto_background()') < src.index('unregister_agent_foreground(agent_id'))
    check('...and the task id is the agent id it gave the sub-context',
          'tool_use_context.agent_id = agent_id' in src)

    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
