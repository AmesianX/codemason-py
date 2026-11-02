#!/usr/bin/env python3
"""A message to a running sub-agent reaches it, and reaches only it.

The queue is the whole mechanism: the agent is mid-turn and the message list it
is working from belongs to the loop running it, so a message waits on the
agent's own task until that loop hits its next tool-round boundary. Being
per-task is also what makes it safe — unlike the process-global notification
queue, there is no way for one agent to drain another's mail.
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

from codemason.state.app_state import get_app_state, reset_app_state, set_app_state
from codemason.tasks.local_agent_task import (
    append_message_to_local_agent,
    complete_agent_task,
    drain_pending_messages,
    queue_pending_message,
    register_agent_foreground,
)
from codemason.tools.send_message_tool import SendMessageTool
from codemason.utils.abort import child_of
from codemason.utils.agent_id import create_agent_id
from codemason.utils.messages import wrap_command_text


class FakeAgent:
    agent_type = 'Explore'


class Ctx:
    get_app_state = staticmethod(get_app_state)
    set_app_state = staticmethod(set_app_state)
    agent_id = None


def spawn():
    agent_id = create_agent_id()
    register_agent_foreground(
        agent_id=agent_id, description='look around', prompt='find it',
        selected_agent=FakeAgent(), abort_controller=child_of(asyncio.Event()),
    )
    return agent_id


async def send(to, message, summary=None):
    payload = {'to': to, 'message': message}
    if summary:
        payload['summary'] = summary
    validation = await SendMessageTool.validate_input(payload, Ctx())
    if not getattr(validation, 'result', True):
        return {'invalid': getattr(validation, 'message', '')}
    out = None
    async for r in SendMessageTool.call(payload, Ctx()):
        out = r
    return out.data


async def main():
    # ---- the queue ---------------------------------------------------------
    reset_app_state()
    agent_id = spawn()

    data = await send(agent_id, 'skip the vendor directory', 'narrow the search')
    check('a message to a running agent is accepted', data.get('success') is True, data)
    check('...and says it waits for the next tool round',
          'at its next tool round' in data['message'], data['message'])
    check('...landing on that agent\'s own task',
          get_app_state().tasks[agent_id].pending_messages == ['skip the vendor directory'])

    await send(agent_id, 'and the build directory')
    check('a second message queues behind the first, in order',
          get_app_state().tasks[agent_id].pending_messages
          == ['skip the vendor directory', 'and the build directory'])

    # ---- the drain ---------------------------------------------------------
    drained = drain_pending_messages(agent_id, get_app_state, set_app_state)
    check('the drain returns everything queued, in order',
          drained == ['skip the vendor directory', 'and the build directory'])
    check('...and leaves the queue empty',
          get_app_state().tasks[agent_id].pending_messages == [])
    check('draining again returns nothing',
          drain_pending_messages(agent_id, get_app_state, set_app_state) == [])

    # ---- one agent cannot read another's mail ------------------------------
    reset_app_state()
    a, b = spawn(), spawn()
    queue_pending_message(a, 'for a only', set_app_state)
    check('a message is queued on exactly one agent',
          get_app_state().tasks[a].pending_messages == ['for a only']
          and get_app_state().tasks[b].pending_messages == [])
    check('draining the other agent takes nothing',
          drain_pending_messages(b, get_app_state, set_app_state) == [])
    check('...and leaves the first agent\'s message alone',
          get_app_state().tasks[a].pending_messages == ['for a only'])

    # ---- the framing is the instruction ------------------------------------
    framed = wrap_command_text('skip the vendor directory', 'coordinator')
    check('the delivered message says who sent it',
          framed.startswith('The coordinator sent a message while you were working:'))
    check('...and when to deal with it',
          framed.endswith('Address this before completing your current task.'))

    # ---- what cannot be sent -----------------------------------------------
    reset_app_state()
    agent_id = spawn()
    complete_agent_task(agent_id, {'text': 'done'}, set_app_state)
    data = await send(agent_id, 'one more thing')
    check('a finished agent cannot be messaged', data.get('success') is False)
    check('...and is told why, not silently dropped',
          'no transcript to resume from' in data['message'], data['message'])

    data = await send('a' + '0' * 16, 'hello?')
    check('an unknown agent id fails', data.get('success') is False)
    check('...naming what to do instead',
          'launch a new agent' in data['message'].lower(), data['message'])

    data = await send('researcher', 'hello?')
    check('a name nobody was spawned under is rejected',
          'invalid' in data and 'neither a name' in data['invalid'], data)

    # ---- addressing an agent by the name it was spawned under --------------
    # The whole point of the name: the model should not have to carry a 16-hex
    # id out of a tool result and quote it back correctly.
    reset_app_state()
    named_id = spawn()
    set_app_state(lambda prev: replace(
        prev, agent_name_registry={**prev.agent_name_registry, 'researcher': named_id}))
    data = await send('researcher', 'skip the vendor directory')
    check('a registered name reaches the agent it names',
          data.get('success') is True, data)
    check('...and the message lands on that agent, not on the name',
          get_app_state().tasks[named_id].pending_messages
          == ['skip the vendor directory'],
          get_app_state().tasks[named_id].pending_messages)
    check('...while its raw id still works, since the name only adds a route',
          (await send(named_id, 'and the build dir')).get('success') is True)
    check('...both messages queued, in order',
          get_app_state().tasks[named_id].pending_messages
          == ['skip the vendor directory', 'and the build dir'],
          get_app_state().tasks[named_id].pending_messages)
    check('a name that is not registered does not fall through to some agent',
          'invalid' in (await send('reviewer', 'hello?')),
          await send('reviewer', 'hello?'))

    data = await send('', 'hello?')
    check('an empty recipient is rejected', 'invalid' in data)

    # ---- a shell task is not a messageable agent ---------------------------
    from codemason.task import create_task_state_base
    from codemason.tasks.local_shell_task import LocalShellTaskState
    from codemason.utils.task.framework import register_task
    reset_app_state()
    base = create_task_state_base('b12121212', 'local_bash', 'run tests', None)
    register_task(LocalShellTaskState(**{**base, 'status': 'running'},
                                      command='pytest', kind='bash'), set_app_state)
    data = await send('a' + '1' * 16, 'hello?')
    check('a shell task is not reachable by SendMessage', data.get('success') is False)

    # ---- the main session is not messageable -------------------------------
    reset_app_state()
    agent_id = spawn()
    set_app_state(lambda prev: replace(
        prev, tasks={**prev.tasks, agent_id: replace(prev.tasks[agent_id],
                                                     agent_type='main-session')}))
    data = await send(agent_id, 'hello?')
    check('the main session is not a messageable sub-agent',
          data.get('success') is False, data)

    # ---- the displayed transcript is a separate list -----------------------
    reset_app_state()
    agent_id = spawn()
    queue_pending_message(agent_id, 'routed to the agent', set_app_state)
    append_message_to_local_agent(agent_id, {'type': 'user', 'text': 'shown to the user'},
                                  set_app_state)
    t = get_app_state().tasks[agent_id]
    check('queuing routes to the agent without touching the display',
          t.pending_messages == ['routed to the agent'])
    check('...and appending to the display does not queue anything for the agent',
          t.messages == [{'type': 'user', 'text': 'shown to the user'}])

    # ---- the tool is the parent's, not an agent's --------------------------
    from codemason.tools import get_agent_tools
    from codemason.tools.built_in_agents import get_agent_definition, resolve_agent_tools
    pool = await get_agent_tools()
    agent_pool = [t.name for t in resolve_agent_tools(get_agent_definition('general-purpose'), pool)]
    check('a sub-agent is not given SendMessage — it would steer its siblings',
          'SendMessage' not in agent_pool)

    check('the tool is read-only, so it can run concurrently',
          SendMessageTool.is_read_only() and SendMessageTool.is_concurrency_safe())

    # ---- where the drain is wired ------------------------------------------
    # The drain itself runs inside query()'s turn loop, which these tests do not
    # drive (stubbing the loop would stub the thing under test). Its behaviour
    # is covered above; what is checked here is that it is wired at the turn
    # boundary and behind the right gate — a sub-agent drains its own queue, and
    # the main thread does not drain anything agent-shaped.
    src = Path('codemason/query.py').read_text(encoding='utf-8')
    check('query drains a sub-agent\'s queue', 'drain_pending_messages(' in src)
    check('...inside the turn loop, appending to the next turn',
          src.count('next_messages = [*next_messages, _sent]') == 1)
    check('...and yielding it, so the durable history keeps it',
          'yield _sent' in src)
    check('...framed by who sent it', "wrap_command_text(_queued, 'coordinator')" in src)
    check('...wrapped as a system reminder',
          'wrap_messages_in_system_reminder([_sent])' in src)
    check('...only when this IS a sub-agent',
          src.index('drain_pending_messages(') > src.index(
              "if not getattr(tool_use_context, 'agent_id', None):"))
    check('the notification drain is still main-thread only, unchanged',
          'drain_pending_notifications()' in src)

    reset_app_state()
    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
