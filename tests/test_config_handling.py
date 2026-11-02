"""config.json from the user's side: fresh install, unreadable file, hand-typed
values, and a file written by an older version.

Run directly (like the other tests here): exit 0 = all pass. Every case runs in a
throwaway CODEMASON_CONFIG_DIR in a child process, because the config module
resolves its paths once at import.
"""
import json
import os
import subprocess
import sys

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
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(config_dir, code):
    env = dict(os.environ, CODEMASON_CONFIG_DIR=config_dir, PYTHONPATH=ROOT, LANG='C')
    return subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)


def case(name, files, code, check):
    d = tempfile.mkdtemp(prefix='cm-cfg-')
    for rel, body in files.items():
        with open(os.path.join(d, rel), 'w', encoding='utf-8') as f:
            f.write(body)
    r = run(d, code)
    try:
        ok = check(d, r)
    except Exception as e:  # a failed lookup is a failed case, not a crash
        ok = False
        print(f'  {name}: {type(e).__name__}: {e}')
    print(('PASS ' if ok else 'FAIL ') + name)
    if not ok:
        print('  stdout:', r.stdout.strip()[:400])
        print('  stderr:', r.stderr.strip()[:400])
    shutil.rmtree(d, ignore_errors=True)   # one config dir per case; none left in /tmp
    return ok


def load(d):
    with open(os.path.join(d, 'config.json'), encoding='utf-8') as f:
        return json.load(f)


LOAD = 'from codemason.utils.config import get_config; c = get_config()\n'
results = []

# 1. Fresh install: every setting is in the file, private to the user.
results.append(case(
    'fresh install writes every key at 0600',
    {}, LOAD,
    lambda d, r: (r.returncode == 0
                  and load(d)['config_version'] >= 2
                  and len(load(d)) >= 70
                  and (os.name != 'posix' or (os.stat(os.path.join(d, 'config.json')).st_mode & 0o777) == 0o600))))

# 2. An unreadable file is reported and never overwritten by a later save.
BROKEN = '{\n  "model_api_key": "sk-USER",\n  "language": "Korean",\n}\n'
results.append(case(
    'unreadable config is reported and left untouched by saves',
    {'config.json': BROKEN},
    LOAD + 'from codemason.utils.config import save_config\nc.theme = "light"\nprint(save_config())\n',
    lambda d, r: (open(os.path.join(d, 'config.json'), encoding='utf-8').read() == BROKEN
                  and 'Could not read' in r.stderr
                  and r.stdout.strip().splitlines()[-1] == 'False')))
results.append(case(
    'a normal save reports success',
    {}, LOAD + 'from codemason.utils.config import save_config\nc.theme = "light"\nprint(save_config())\n',
    lambda d, r: r.stdout.strip().splitlines()[-1] == 'True' and load(d)['theme'] == 'light'))

# 3. Hand-typed values mean what they say; "false" must never enable a flag.
results.append(case(
    'string spellings are read as their meaning',
    {'config.json': json.dumps({'dangerously_skip_permissions': 'false', 'realtime': 'no',
                                'max_turns': '50', 'max_tokens': '32000', 'keep_going': 'maybe',
                                'model_base_url': None})},
    LOAD + 'import json\nprint(json.dumps([c.dangerously_skip_permissions, c.realtime, c.max_turns, '
           'c.max_tokens, c.keep_going, c.model_base_url]))\n',
    lambda d, r: (json.loads(r.stdout.strip().splitlines()[-1])
                  == [False, False, 50, 32000, False, 'https://llm.systemboan.net']
                  and 'keep_going' in r.stderr and 'model_base_url' in r.stderr)))

# 4. Defaults that were defects move forward once; user choices stay.
results.append(case(
    'old defaults are migrated once, user choices kept',
    {'config.json': json.dumps({'max_tokens': 16384, 'max_turns': 200, 'reasoning_effort': '',
                                'sampling_profile': 'vendor', 'realtime': False, 'keep_going': True})},
    LOAD,
    lambda d, r: (lambda f: (f['config_version'] >= 2 and f['max_tokens'] == 65536
                             and f['max_turns'] is None and f['reasoning_effort'] == 'high'
                             and f['sampling_profile'] == 'auto'
                             and f['realtime'] is False and f['keep_going'] is True))(load(d))))
results.append(case(
    'a value the user chose is not migrated',
    {'config.json': json.dumps({'max_tokens': 32000, 'reasoning_effort': 'low'})},
    LOAD,
    lambda d, r: load(d)['max_tokens'] == 32000 and load(d)['reasoning_effort'] == 'low'))
results.append(case(
    'a file from a newer version keeps its version and values',
    {'config.json': json.dumps({'config_version': 99, 'max_tokens': 16384})},
    LOAD,
    lambda d, r: load(d)['config_version'] == 99 and load(d)['max_tokens'] == 16384))

sys.exit(0 if all(results) else 1)
