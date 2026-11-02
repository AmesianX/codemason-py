#!/usr/bin/env python3
"""Every tool says what it is doing, in words.

`create_activity_description_resolver` has always looked for a
`get_activity_description` on each tool — and no tool defined one, so the
resolver returned None for every call and `progress.recent_activities[].
activity_description` was permanently empty. The field existed, the plumbing
existed, and there was nothing at the far end of it.

This is what a parent watching a sub-agent reads. "Reading codemason/query.py"
and "Running the test suite" are the difference between knowing what an agent is
doing and knowing only which tool it picked.

The verbs and the fallbacks are the source's, tool by tool; the arguments come
from each tool's get_tool_use_summary, truncated to TOOL_SUMMARY_MAX_LENGTH.
"""
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


from codemason.tasks.local_agent_task import (
    create_activity_description_resolver,
    create_progress_tracker,
    update_progress_from_message,
)
from codemason.tools.agent_tool import AgentTool
from codemason.tools.bash_tool import BashTool
from codemason.tools.file_edit_tool import FileEditTool
from codemason.tools.file_read_tool import FileReadTool
from codemason.tools.file_write_tool import FileWriteTool
from codemason.tools.glob_tool import GlobTool
from codemason.tools.grep_tool import GrepTool
from codemason.tools.notebook_edit_tool import NotebookEditTool
from codemason.tools.web_fetch_tool import WebFetchTool
from codemason.tools.web_search_tool import WebSearchTool
from codemason.utils.file import get_display_path
from codemason.utils.formatting import (
    TOOL_SUMMARY_MAX_LENGTH,
    display_width,
    truncate,
)
from codemason.utils.messages import create_assistant_message
from codemason.utils.task.disk_output import get_task_output_path

HERE = str(Path(__file__).resolve())


def main():
    # ---- the verbs, tool by tool, against the source ----------------------
    cases = [
        (FileReadTool, {'file_path': HERE}, f'Reading {get_display_path(HERE)}', 'Reading file'),
        (FileWriteTool, {'file_path': HERE}, f'Writing {get_display_path(HERE)}', 'Writing file'),
        (FileEditTool, {'file_path': HERE}, f'Editing {get_display_path(HERE)}', 'Editing file'),
        (NotebookEditTool, {'notebook_path': HERE},
         f'Editing notebook {get_display_path(HERE)}', 'Editing notebook'),
        (GlobTool, {'pattern': '**/*.py'}, 'Finding **/*.py', 'Finding files'),
        (GrepTool, {'pattern': 'TODO'}, 'Searching for TODO', 'Searching'),
        (WebFetchTool, {'url': 'https://example.com/x'},
         'Fetching https://example.com/x', 'Fetching web page'),
        (WebSearchTool, {'query': 'vllm prefix cache'},
         'Searching for vllm prefix cache', 'Searching the web'),
    ]
    for tool, inp, expected, fallback in cases:
        got = tool.get_activity_description(inp)
        check(f'{tool.name}: {expected!r}', got == expected, got)
        check(f'{tool.name}: falls back without its argument',
              tool.get_activity_description({}) == fallback,
              tool.get_activity_description({}))
        check(f'{tool.name}: a non-dict input is not a crash',
              tool.get_activity_description(None) == fallback,
              tool.get_activity_description(None))

    # ---- the two that do not follow the pattern ---------------------------
    bash_both = BashTool.get_activity_description(
        {'command': 'bun test', 'description': 'run the test suite'})
    check('Bash prefers the model\'s own description over the command',
          bash_both == 'Running run the test suite', bash_both)
    check('...and uses the command when there is no description',
          BashTool.get_activity_description({'command': 'bun test'}) == 'Running bun test')
    check('...falling back with no command at all',
          BashTool.get_activity_description({}) == 'Running command')
    check('Agent uses the task description as-is, no verb',
          AgentTool.get_activity_description({'description': 'survey the lane'})
          == 'survey the lane')
    check('...and "Running task" when it has none',
          AgentTool.get_activity_description({}) == 'Running task')

    # ---- truncation --------------------------------------------------------
    long_pattern = 'x' * 200
    got = GrepTool.get_tool_use_summary({'pattern': long_pattern})
    # COLUMNS, not characters. Asserting len() is what let the off-by-one
    # through: the ellipsis is East-Asian ambiguous, so it costs two cells
    # wherever ambiguous counts wide, and reserving one column for it produced
    # a 51-column summary that measured 50 characters.
    check('a pasted 200-char pattern is cut to the summary limit, in columns',
          got == truncate(long_pattern, TOOL_SUMMARY_MAX_LENGTH)
          and display_width(got) <= TOOL_SUMMARY_MAX_LENGTH,
          (display_width(got), len(got), got[-4:]))
    check('...and marked as cut, so it does not read as the whole thing',
          got.endswith('…'))
    check('a long bash command is cut too',
          BashTool.get_activity_description({'command': 'y' * 200}).endswith('…'))

    # ---- the agent-output special case ------------------------------------
    out_path = get_task_output_path('a' + '0' * 16)
    check('an agent output file shows the task id, not a temp path',
          FileReadTool.get_tool_use_summary({'file_path': out_path}) == 'a' + '0' * 16,
          FileReadTool.get_tool_use_summary({'file_path': out_path}))
    check('...while an unrelated file in that directory is still a path',
          FileReadTool.get_tool_use_summary(
              {'file_path': get_task_output_path('x') + '.txt'}) != 'x')

    # ---- which is what the agent's progress tally now carries -------------
    tools = [FileReadTool, BashTool, GrepTool]
    resolve = create_activity_description_resolver(tools)
    check('the resolver finds a tool by name', resolve('Read', {'file_path': HERE})
          == f'Reading {get_display_path(HERE)}', resolve('Read', {'file_path': HERE}))
    check('...and returns None for a tool that is not in the pool',
          resolve('Write', {'file_path': HERE}) is None)

    tracker = create_progress_tracker()
    msg = create_assistant_message('looking')
    msg['message']['usage'] = {'input_tokens': 10, 'output_tokens': 1,
                               'cache_read_input_tokens': 0,
                               'cache_creation_input_tokens': 0}
    msg['message']['content'] = list(msg['message']['content']) + [
        {'type': 'tool_use', 'id': 'tu_0', 'name': 'Read',
         'input': {'file_path': HERE}},
        {'type': 'tool_use', 'id': 'tu_1', 'name': 'Bash',
         'input': {'command': 'bun test', 'description': 'run the test suite'}},
    ]
    update_progress_from_message(tracker, msg, resolve, tools)
    descs = [a.activity_description for a in tracker.recent_activities]
    check('the tally records what each tool use was doing, not just its name',
          descs == [f'Reading {get_display_path(HERE)}', 'Running run the test suite'],
          descs)
    check('...which is the field that was permanently None before',
          all(d is not None for d in descs), descs)

    # ---- and the prose line the compaction attachment reads ---------------
    # codemason's local equivalent of startAgentSummarization: the source forks
    # the agent's transcript every 30s and asks the model for a line whose own
    # prompt examples are "Reading runAgent.ts" — which is what the resolver
    # above already produces for free.
    from codemason.state.app_state import get_app_state, reset_app_state, set_app_state
    from codemason.tasks.local_agent_task import (
        derive_agent_summary,
        get_progress_update,
        register_async_agent,
        update_agent_progress,
        update_agent_summary,
    )
    from codemason.utils.agent_id import create_agent_id

    check('nothing done yet means no summary, not an empty one',
          derive_agent_summary(create_progress_tracker()) is None)

    fresh = create_progress_tracker()
    update_progress_from_message(fresh, msg, resolve, tools)
    check('the summary is the most recent action',
          derive_agent_summary(fresh) == 'Running run the test suite',
          derive_agent_summary(fresh))

    # A tool with no description must not blank the line — the walk goes back.
    only_unknown = create_assistant_message('x')
    only_unknown['message']['usage'] = {'input_tokens': 10, 'output_tokens': 1,
                                        'cache_read_input_tokens': 0,
                                        'cache_creation_input_tokens': 0}
    only_unknown['message']['content'] = list(only_unknown['message']['content']) + [
        {'type': 'tool_use', 'id': 'tu_9', 'name': 'NoSuchTool', 'input': {}}]
    update_progress_from_message(fresh, only_unknown, resolve, tools)
    check('an undescribed tool falls back to the last line that had one',
          derive_agent_summary(fresh) == 'Running run the test suite',
          derive_agent_summary(fresh))

    # ---- which survives the counts update that follows it -----------------
    reset_app_state()
    agent_id = create_agent_id()
    register_async_agent(agent_id=agent_id, description='survey', prompt='p',
                         selected_agent=None, set_app_state_fn=set_app_state)
    update_agent_progress(agent_id, get_progress_update(fresh), set_app_state)
    update_agent_summary(agent_id, derive_agent_summary(fresh), set_app_state)
    stored = get_app_state().tasks[agent_id].progress
    check('the task carries the line AND the counts',
          stored.summary == 'Running run the test suite'
          and stored.tool_use_count == 3, (stored.summary, stored.tool_use_count))
    update_agent_progress(agent_id, get_progress_update(fresh), set_app_state)
    check('...and a later counts update does not wipe the line',
          get_app_state().tasks[agent_id].progress.summary == 'Running run the test suite',
          get_app_state().tasks[agent_id].progress.summary)
    reset_app_state()

    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
