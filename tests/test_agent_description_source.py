#!/usr/bin/env python3
"""Where the Agent tool's wire description comes from, and that only one copy exists.

The request builder resolves a tool's description as
    build_description() -> DESCRIPTION -> TOOL_DESCRIPTIONS -> simple blurb
(services/_openai_common.py). AgentTool sets DESCRIPTION, so the map is never
reached for it — and the copy that used to sit in the map had drifted: it was
missing two usage notes and worded the language rule differently. One source,
pinned here, is the only way that stays true.
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


from codemason.constants.tool_prompts import TOOL_DESCRIPTIONS
from codemason.tools.agent_tool import AgentTool
from codemason.tools.built_in_agents import build_agent_description

check('the Agent tool carries its own description', bool(AgentTool.DESCRIPTION))
check('and it is the built one', AgentTool.DESCRIPTION == build_agent_description())
check('so the fallback map holds no Agent copy to drift',
      'Agent' not in TOOL_DESCRIPTIONS, f'keys: {sorted(TOOL_DESCRIPTIONS)}')
check('the tools that DO resolve through the map are still there',
      {'Read', 'Write', 'Edit', 'Glob', 'Grep', 'Bash'} <= set(TOOL_DESCRIPTIONS))

d = AgentTool.DESCRIPTION
check('the description carries the proactive note', 'should be used proactively' in d)
check('and the one that says what "in parallel" means',
      'you MUST send a single message with multiple Agent tool use content blocks' in d)
check('and lists fetches among research', 'web fetches, etc.' in d)
check('with no unsubstituted placeholder', '{AGENT_TOOL_NAME}' not in d)
check('and no unsubstituted tool-name placeholder', '{SEND_MESSAGE_TOOL_NAME}' not in d)

# Restored as each capability landed. A bullet describing something the tool
# cannot do is a lie to the model, so these are only present once true.
for needle in [
    'run_in_background parameter',
    'do NOT sleep, poll, or proactively check on its progress',
    '**Foreground vs background**',
    'To send a message to a running agent, use SendMessage',
]:
    check(f'the description advertises: {needle[:44]}', needle in d)
check('...and still does not advertise worktree isolation, which is not ported',
      'isolation' not in d)

# The prompt-writing section the earlier port compressed into one paragraph, and
# the examples it dropped entirely. "Never delegate understanding" is the part
# that earns its tokens: without it the model writes "based on your findings,
# fix the bug" and pushes the synthesis onto an agent that cannot do it.
for needle in [
    '## Writing the prompt',
    'report in under 200 words',
    'Lookups: hand over the exact command',
    'prescribed steps become dead weight when the premise is wrong',
    '**Never delegate understanding.**',
    'Example usage:',
    '<example_agent_descriptions>',
    '"test-runner": use this agent after you are done writing code to run tests',
    'launch the greeting-responder agent',
    'if (n <= 1) return false',
]:
    check(f'the description keeps: {needle[:46]}', needle in d)
check('no doubled format braces leaked into the text', '{{' not in d)

print()
print(f'{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
