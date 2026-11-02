#!/usr/bin/env python3
"""The Agent tool drives the task lifecycle for real, not just in the source text.

Three runs with a stubbed query loop: one that finishes, one that raises, one that
is cancelled. Each must leave the task gone and the process-exit cleanup released
— an agent that lingers in AppState is the leak, and one whose exit cleanup is
still registered would be killed again long after it finished.
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
from codemason.types.query import ToolUseContext
from codemason.utils.cleanup_registry import cleanup_function_count
from codemason.utils.messages import create_assistant_message


# What the tool saw while the agent was running, captured from inside the loop.
seen = {}


def _assistant_with_usage(inp, out, tools):
    """An assistant message carrying usage and optional tool uses."""
    msg = create_assistant_message('working')
    msg['message']['usage'] = {
        'input_tokens': inp, 'output_tokens': out,
        'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0,
    }
    msg['message']['content'] = list(msg['message']['content']) + [
        {'type': 'tool_use', 'id': f'tu_{i}', 'name': name, 'input': {}}
        for i, name in enumerate(tools)
    ]
    return msg


def stub_query(behaviour):
    """Replace the query loop with one that does exactly one thing."""
    async def _query(messages, system_prompt, context, can_use_tool, tool_use_context, **kw):
        tasks = get_app_state().tasks
        seen['ids'] = list(tasks)
        seen['status'] = [t.status for t in tasks.values()]
        seen['cleanups'] = cleanup_function_count()
        seen['agent_id'] = tool_use_context.agent_id
        seen['abort'] = tool_use_context.abort_controller
        if behaviour == 'raise':
            raise RuntimeError('agent blew up')
        if behaviour == 'cancel':
            raise asyncio.CancelledError()
        if behaviour == 'progress':
            # Two turns: one with a tool use, one with the answer. The task's
            # progress must be readable from outside WHILE the agent runs —
            # that is the whole point of storing it.
            yield _assistant_with_usage(100, 10, ['Read'])
            seen['mid'] = get_app_state().tasks[tool_use_context.agent_id].progress
            yield _assistant_with_usage(260, 20, [])
            seen['late'] = get_app_state().tasks[tool_use_context.agent_id].progress
            return
        yield create_assistant_message('the answer')
    return _query


async def run(behaviour):
    reset_app_state()
    seen.clear()
    base = cleanup_function_count()
    original = query_mod.query
    query_mod.query = stub_query(behaviour)
    try:
        from codemason.tools.agent_tool import AgentTool
        ctx = ToolUseContext(
            abort_controller=asyncio.Event(),
            options={'dangerously_skip_permissions': False},
            set_app_state=set_app_state,
            get_app_state=get_app_state,
            tool_use_id='toolu_test',
        )
        error = None
        out = []
        try:
            async for r in AgentTool.call({'prompt': 'do it', 'description': 'test run'}, ctx):
                out.append(r)
        except BaseException as exc:          # noqa: BLE001 — the point is which one
            error = exc
        return base, out, error
    finally:
        query_mod.query = original


async def main():
    # ---- the happy path ----------------------------------------------------
    base, out, error = await run('ok')
    check('a finishing agent raises nothing', error is None, repr(error))
    check('the task was registered BEFORE the query loop ran',
          len(seen.get('ids', [])) == 1 and seen['status'] == ['running'])
    check('...with its exit cleanup already in place', seen.get('cleanups') == base + 1)
    check('...and the sub-context got that same agent id',
          seen.get('ids') == [seen.get('agent_id')])
    check('...on its own abort scope', seen.get('abort') is not None
          and seen['abort'] is not None and hasattr(seen['abort'], 'is_set'))
    check('the agent returned its text',
          any(getattr(r, 'result_for_assistant', '') == 'the answer' for r in out),
          [getattr(r, 'result_for_assistant', None) for r in out])
    check('the finished task is gone from AppState', get_app_state().tasks == {})
    check('...and its exit cleanup is released', cleanup_function_count() == base)


    # ---- the parent can watch an agent's context grow ----------------------
    base, out, error = await run('progress')
    check('a progress run raises nothing', error is None, repr(error))
    mid, late = seen.get('mid'), seen.get('late')
    check('the task carries progress while the agent is still running',
          mid is not None and mid.token_count == 110, mid)
    check('...and the tool count with it', mid is not None and mid.tool_use_count == 1)
    check('...and names the last thing it did',
          mid is not None and mid.last_activity is not None
          and mid.last_activity.tool_name == 'Read')
    check('the next turn replaces the prompt count rather than adding to it',
          late is not None and late.token_count == 260 + 30, late)
    check('...and the tool count did not move on a turn with no tool',
          late is not None and late.tool_use_count == 1)
    check('the finished task is gone', get_app_state().tasks == {})
    check('...and leaks no cleanup', cleanup_function_count() == base)

    # ---- the failure path --------------------------------------------------
    base, out, error = await run('raise')
    check('a failing agent re-raises instead of reporting success',
          isinstance(error, RuntimeError), repr(error))
    check('a failed task is removed too', get_app_state().tasks == {})
    check('...and leaks no exit cleanup', cleanup_function_count() == base)

    # ---- the cancel path ---------------------------------------------------
    base, out, error = await run('cancel')
    check('a cancelled agent propagates the cancellation',
          isinstance(error, asyncio.CancelledError), repr(error))
    check('a cancelled task is removed', get_app_state().tasks == {})
    check('...and leaks no exit cleanup', cleanup_function_count() == base)

    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
