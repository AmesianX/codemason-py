"""Self-installer for the keep-going Stop hook.

keep-going is OFF by default (see config.py). CodeMason auto-installs this hook
only when keep-going is effectively enabled (REPL `keepgoing on` / env / config),
so a fresh checkout materializes it on first such run. Turning keep-going OFF does
NOT uninstall it — the hook's own `_enabled()` check short-circuits to a no-op
(identical to having no Stop hook), which is cheap and trivially reversible.

Why embed the script as a string instead of shipping a data file: the whole
policy then travels inside version-controlled .py source. Move the repo (git)
to another box and the first run re-materializes the hook there — no
package-data wiring, no stale file left behind on the old machine to chase.

The hook is rewritten whenever its content drifts from HOOK_SCRIPT, so a code
update to the policy below propagates on the next run without a manual
reinstall."""

import json
from pathlib import Path
from typing import List

from .config import CONFIG_DIR, SETTINGS_FILE

HOOKS_DIR = CONFIG_DIR / 'hooks'
HOOK_FILE = HOOKS_DIR / 'keep_going.py'
# What settings.json runs. Use the resolved hook path (honors
# $CODEMASON_CONFIG_DIR) so it matches wherever the single config dir lives.
HOOK_COMMAND = f'python3 "{HOOK_FILE}"'

# The Stop hook script, embedded verbatim. Keep this in sync with the decision
# logic documented in its own docstring; ensure_installed() rewrites the on-disk
# copy whenever it differs from this string.
HOOK_SCRIPT = r'''#!/usr/bin/env python3
"""
CodeMason "Stop" hook — keep the model working until the task is actually done.

How it is wired: ~/.codemason/settings.json -> hooks.Stop runs this script every
time the model ends a turn WITHOUT calling a tool (the loop's natural stop
point). CodeMason feeds this script a JSON object on stdin and reads the result:

  - exit 0 with no output                -> let the model stop (turn ends)
  - stdout {"decision":"block","reason"} -> inject `reason` and CONTINUE the model
  - stdout {"continue": false}           -> force-stop the turn

stdin fields we use:
  - last_assistant_message : text of the model's final message this turn
  - used_tools             : True if the turn ran any tool since the last user
                             input (real agentic work vs a chat reply)
  - stop_hook_active       : True when the model is ALREADY continuing because
                             of this hook (set after a previous block). Use it
                             to avoid infinite loops.

Decision logic here = the "0 conversational gate + 1 sentinel + 2 progress-
safety" hybrid:
  0. If the turn did NO tool work since the last user input (used_tools=False),
     it's a pure conversational reply (greeting / plain Q&A) -> allow stop. This
     keeps keep-going from hijacking chat: only turns that actually ran tools are
     candidates for nudging. (Engine reports `used_tools` in the hook input.)
  1. If the model declared completion (printed TASK_COMPLETE) -> allow stop.
  2. Else, if we already nudged once (stop_hook_active) and it STILL ended with
     no tool use, treat it as done/stuck -> allow stop (loop guard). Note:
     productive tool turns reset stop_hook_active to False, so as long as the
     model keeps doing real tool work it keeps getting nudged; only TWO
     consecutive no-tool stops end it.
  3. Otherwise -> block and tell it to keep going.

There is NO hard turn cap (the hook owns termination).
The model emitting TASK_COMPLETE is the intended clean exit; otherwise the
progress-safety in step 2 stops it. You can always interrupt with Ctrl-C.

---------------------------------------------------------------------------
UPGRADE to "objective test gate" (best for a specific coding project): judge
completion by GROUND TRUTH instead of the model's word. Replace step 1 with a
real check, e.g.:

    import subprocess
    r = subprocess.run(['pytest', '-q'], cwd=d.get('cwd', '.'),
                       capture_output=True, text=True)
    if r.returncode == 0:
        sys.exit(0)                                   # tests pass -> done
    if d.get('stop_hook_active'):
        sys.exit(0)                                   # safety: still failing -> give up
    print(json.dumps({'decision': 'block',
        'reason': f'Tests still failing - keep fixing:\n{r.stdout[-1500:]}'}))
    sys.exit(0)

Keep that as a PER-PROJECT hook, not here -- this file is global and would run
the wrong test command in other projects.
---------------------------------------------------------------------------
"""
import json
import os
import sys
from pathlib import Path


def _enabled():
    """Effective keep-going state, resolved with this precedence:
       1. CODEMASON_KEEP_GOING env  (session override: CLI flag / REPL command)
       2. ~/.codemason/config.json "keep_going"  (persistent default)
       3. True  (default ON)
    OFF makes this hook a clean no-op -- identical to having no Stop hook."""
    env = os.environ.get('CODEMASON_KEEP_GOING')
    if env is not None:
        return env.strip().lower() not in ('0', 'false', 'no', 'off', '')
    try:
        cfg_dir = Path(os.environ.get('CODEMASON_CONFIG_DIR') or (Path.home() / '.codemason'))
        cfg = json.loads((cfg_dir / 'config.json').read_text())
        return bool(cfg.get('keep_going', False))
    except Exception:
        return False


# Cleanly disabled -> behave exactly like no Stop hook (let the model stop).
if not _enabled():
    sys.exit(0)

try:
    d = json.load(sys.stdin)
except Exception:
    # Can't read the hook input -> never block (fail open, let the model stop).
    sys.exit(0)

last = d.get('last_assistant_message') or ''

# 0) Pure conversational turn (no tool work since the last user input) -> stop.
#    `used_tools` defaults missing->True so an older engine that doesn't send it
#    keeps the prior always-nudge behavior rather than silently going quiet.
if not d.get('used_tools', True):
    sys.exit(0)

# 1) Model declared completion -> allow stop.
if 'TASK_COMPLETE' in last:
    sys.exit(0)

# 2) Loop guard: already nudged and it still ended with no tool use -> stop.
if d.get('stop_hook_active'):
    sys.exit(0)

# 3) Otherwise, push it to keep going.
print(json.dumps({
    'decision': 'block',
    'reason': (
        'Keep working toward the goal. Do not stop to ask for confirmation '
        'and do not summarize prematurely. Continue using tools until the '
        'task is fully done. When it is genuinely complete, end your final '
        'message with the exact line: TASK_COMPLETE'
    ),
}))
'''


def is_installed() -> bool:
    """True if both the hook file exists AND settings.json registers it."""
    if not HOOK_FILE.exists():
        return False
    return _stop_hook_registered(_read_settings())


def _read_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text()) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _stop_hook_registered(settings: dict) -> bool:
    stop = (settings.get('hooks') or {}).get('Stop') or []
    for entry in stop:
        for h in entry.get('hooks', []) if isinstance(entry, dict) else []:
            if h.get('command') == HOOK_COMMAND:
                return True
    return False


def ensure_installed() -> List[str]:
    """Materialize the hook on this machine. Idempotent: writes the script only
    when missing/drifted, registers the Stop hook only when absent. Returns a
    list of human-readable actions taken (empty = already fully installed)."""
    actions: List[str] = []

    from .config import lock_path
    from .file import atomic_write_bytes, update_json_document

    HOOKS_DIR.mkdir(parents=True, exist_ok=True)
    if not HOOK_FILE.exists() or HOOK_FILE.read_text() != HOOK_SCRIPT:
        # Atomic, because this file is EXECUTED by whoever runs the hook. A
        # plain write truncates first, so a second session starting in that
        # window — or the hook runner itself — ran a half-written Python script.
        # Identical content from every writer, so no lock is needed: the rename
        # publishes the whole file or nothing.
        atomic_write_bytes(str(HOOK_FILE), HOOK_SCRIPT.encode('utf-8'))
        actions.append(f'wrote {HOOK_FILE}')

    # The cheap pre-check keeps the common case (already installed) off the
    # lock entirely; the authoritative one is inside the callback, where it
    # cannot straddle another process's write. Two sessions that both read
    # "not registered" and then both appended left the Stop hook registered
    # TWICE, and settings.json holds the hook and permission configuration, so
    # a plain write also risked publishing a truncated version of it.
    if not _stop_hook_registered(_read_settings()):
        def register(current):
            settings = current if isinstance(current, dict) else {}
            if _stop_hook_registered(settings):
                return None                 # another process got there first
            hooks = settings.setdefault('hooks', {})
            stop = hooks.setdefault('Stop', [])
            if not isinstance(stop, list):
                stop = []
                hooks['Stop'] = stop
            stop.append({'hooks': [{'type': 'command', 'command': HOOK_COMMAND}]})
            return settings

        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if update_json_document(SETTINGS_FILE, lock_path(SETTINGS_FILE.name), register):
            actions.append(f'registered Stop hook in {SETTINGS_FILE}')
        elif SETTINGS_FILE.exists() and not _read_settings():
            # The document exists but did not parse, so it was left alone rather
            # than replaced with just our hook (see utils/file). Say so: the
            # alternative is a keep-going that silently never fires.
            actions.append(f'NOT registered: {SETTINGS_FILE} could not be read')

    return actions
