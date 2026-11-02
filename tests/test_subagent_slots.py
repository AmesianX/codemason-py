"""One sub-agent slot pool per model server.

The limit on concurrent sub-agents exists because of one vLLM's KV cache (see
utils/subagent_gate), so the pool belongs to the endpoint, not to the machine.
These cases pin both halves of that, because only one of them is the new one
and breaking either is silent:

  rations      two sub-agents on ONE endpoint still contend. This is the whole
               point of the gate; an endpoint key that accidentally varied per
               process would leave it switched off and nothing would say so.
  does not     a sub-agent on a SECOND endpoint starts anyway. The default is
  over-ration  ONE slot, so before the split a session on endpoint A blocked
               every sub-agent on endpoint B outright, for a GPU it never
               touched.
  frees        a slot is an flock, so it comes back when its holder dies,
               however it died.
  by url only  the key is the base_url and never the model name: one server
               serving two models serves them out of the same cache, so those
               two must keep sharing a slot.

Spelling is pinned in test_parity_with_rust (the Rust build must land on the
same filenames). Here it is behaviour: slots are taken in child processes, the
way two sessions really contend, since flock is per open file description.

Run directly (exit 0 = all pass). Every case runs in a throwaway
CODEMASON_CONFIG_DIR, because the config module resolves its paths once at
import.
"""
import os
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=ROOT, LANG='C')
# The children get ROOT through PYTHONPATH; this process needs it too, because
# pool_of imports CONFIG_VERSION here to stamp the config it writes. Running the
# file by path puts tests/ on sys.path, never the repo root.
sys.path.insert(0, ROOT)

# Two endpoints, and one of them spelled with a different case. Reference values
# for the tags they hash to live in test_parity_with_rust.
A = 'https://llm.systemboan.net'
B = 'http://127.0.0.1:8000/v1'

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


def env(d, url, **kw):
    return dict(ENV, CODEMASON_CONFIG_DIR=d, CODEMASON_MODEL_BASE_URL=url, **kw)


def pool_of(d, slots):
    """Pin this config dir's pool to `slots`.

    What these cases check is that the gate RATIONS — a second taker on the same
    endpoint waits, a second endpoint does not — not that the pool happens to
    hold one. The default is per line (1 where sub-agents are serial, 3 where
    they are not), so reading it from the default made the same assertions pass
    on one line and fail on the other. Written before any child starts, so every
    child in the case sees it."""
    from codemason.utils.config import CONFIG_VERSION
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    # Stamp the version: a file without one counts as 1, and the migration that
    # carries a serial-line count forward would rewrite the pin we just made.
    pathlib.Path(d, 'config.json').write_text(
        json.dumps({'config_version': CONFIG_VERSION,
                    'max_concurrent_subagents': slots}), encoding='utf-8')


# `hold` keeps the slot until it is killed; without it the child takes a slot and
# exits, which is also how it releases one.
TAKER = '''
import sys, time

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
from codemason.utils.subagent_gate import try_acquire
s = try_acquire()
print('OK' if s is not None else 'BUSY', flush=True)
if s is not None and 'hold' in sys.argv:
    time.sleep(60)
'''


def probe(d, url, **kw):
    """Whether a fresh process can take a slot for `url` right now."""
    r = subprocess.run([sys.executable, '-c', TAKER], env=env(d, url, **kw),
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip() or ('no answer: ' + r.stderr.strip()[-200:])


def hold(d, url, **kw):
    """A child holding a slot for `url`, already acquired when this returns."""
    p = subprocess.Popen([sys.executable, '-c', TAKER, 'hold'], env=env(d, url, **kw),
                         stdout=subprocess.PIPE, text=True)
    return p, p.stdout.readline().strip()


def locks(d):
    return sorted(p.name for p in pathlib.Path(d, 'locks').glob('subagent.*'))


# ── 1. One endpoint: the gate rations, as it always did ─────────────────────
#
# With the pool pinned to one slot, the second sub-agent on the same server must
# be turned away — it would evict the parent, which then re-prefills its whole
# history (~200s at 267k tokens). The pin is the point: the default differs per
# line, the rationing does not.
d = tempfile.mkdtemp(prefix='cm-slots-')
pool_of(d, 1)
holder, first = hold(d, A)
check('a sub-agent takes the one slot this pool was pinned to', first == 'OK', f'got {first!r}')
check('a second sub-agent on the SAME endpoint is turned away',
      probe(d, A) == 'BUSY')

# ── 2. A second endpoint is a second pool ───────────────────────────────────
#
# The regression this guards: keyed by the config dir alone, this was 'BUSY' —
# a sub-agent kept waiting on a cache it could not have evicted.
check('a sub-agent on a SECOND endpoint starts while the first endpoint is full',
      probe(d, B) == 'OK')

# One server written two ways is ONE pool: endpoint_tag folds case over the
# scheme and host. Before it did, this handed out a slot too many — two sessions
# rationed against one GPU as if it were two, which is the costly direction (the
# parent gets evicted and re-prefills its whole history).
check('a mixed-case spelling of the SAME endpoint shares its slot, and waits',
      probe(d, A.replace('llm', 'LLM')) == 'BUSY')

# ── 3. A slot comes back when its holder dies ───────────────────────────────
#
# flock, not a pid file: the kernel drops it with the descriptor however the
# process ended, so one kill -9 cannot wedge every later sub-agent.
holder.kill()
holder.wait(timeout=10)
check("a killed holder's slot is free at once", probe(d, A) == 'OK')

# ── 4. The pool is the server's, so the model name is not in the key ────────
#
# One server serving two models serves both from one KV cache. If the model name
# keyed the pool, two models on one GPU would each get a slot and together
# overrun it — the thing the gate exists to prevent.
holder2, first2 = hold(d, A, CODEMASON_MODEL_NAME='model-one')
check('a sub-agent takes the slot under one model name', first2 == 'OK', f'got {first2!r}')
check('a different model on the SAME server shares that slot, and waits',
      probe(d, A, CODEMASON_MODEL_NAME='model-two') == 'BUSY')
holder2.kill()
holder2.wait(timeout=10)

# One empty carrier file per endpoint ever used, and they are never deleted:
# unlinking a flocked path is how two processes come to hold one slot.
names = locks(d)
check('one slot file per endpoint, all empty',
      len(names) == 2 and all(os.path.getsize(os.path.join(d, 'locks', n)) == 0 for n in names),
      str(names))
shutil.rmtree(d, ignore_errors=True)

sys.exit(0 if all(results) else 1)
