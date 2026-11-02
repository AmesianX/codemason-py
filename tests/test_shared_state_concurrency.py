"""Several codemason processes, one config dir.

A Python session, a Rust session, two or three of each, on different models and
different endpoints — they all share ~/.codemason. These cases pin the rules the
shared documents in that tree are written under (utils/file.FileLock):

  no lost update   two processes changing DIFFERENT settings at the same instant
                   must both survive. Before the lock this failed 150/150.
  no residue       a force-killed holder leaves the lock free. The lock
                   file carries no state and is never deleted, so there is
                   nothing to reclaim and nothing to go stale.
  stable inode     the lock sidecar is never renamed or unlinked — a lock on a
                   replaced inode is invisible to the next process and would
                   silently stop excluding anything.
  degrades         a lock that cannot be taken must not fail the write: the
                   write is atomic either way.

Run directly (exit 0 = all pass). Every case runs in a throwaway
CODEMASON_CONFIG_DIR, in child processes, because the config module resolves its
paths once at import.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=ROOT, LANG='C')

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


def fresh_dir():
    d = tempfile.mkdtemp(prefix='cm-shared-')
    subprocess.run([sys.executable, '-c', 'from codemason.utils.config import get_config; get_config()'],
                   env=dict(ENV, CODEMASON_CONFIG_DIR=d), capture_output=True)
    return d


def cfg(d):
    with open(os.path.join(d, 'config.json'), encoding='utf-8') as f:
        return json.load(f)


# ── 1. Two processes, two different settings, one instant ───────────────────
#
# Each child loads the config (so both hold the same startup baseline), changes
# one setting, then spins until the gate file appears — which releases them
# together, inside the few milliseconds the other's read-merge-write takes. That
# alignment is not artificial: launching `codemason` and `cm2` from one script
# does exactly this, and so does a first-run backfill in each.
SAVER = '''
import os, sys
key, val, gate = sys.argv[1], sys.argv[2], sys.argv[3]
from codemason.utils.config import get_config, save_config
c = get_config()
setattr(c, key, True if val == 'true' else val)
while not os.path.exists(gate):
    pass
save_config()
'''

d = fresh_dir()
base = cfg(d)
lost = []
TRIALS = 60
for t in range(TRIALS):
    base['theme'], base['verbose'] = 'dark', False
    with open(os.path.join(d, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(base, f, indent=2)
    gate = os.path.join(d, 'go')
    if os.path.exists(gate):
        os.unlink(gate)
    env = dict(ENV, CODEMASON_CONFIG_DIR=d)
    ps = [subprocess.Popen([sys.executable, '-c', SAVER, k, v, gate], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
          for k, v in (('theme', 'light'), ('verbose', 'true'))]
    time.sleep(0.4)                      # both children are now spinning on the gate
    open(gate, 'w').close()
    for p in ps:
        p.wait()
    got = cfg(d)
    if got.get('theme') != 'light' or got.get('verbose') is not True:
        lost.append((t, got.get('theme'), got.get('verbose')))
check(f'two processes changing different settings at once: both edits survive ({TRIALS} trials)',
      not lost, f'{len(lost)}/{TRIALS} lost, first: {lost[:1]}')

# The write must not have replaced the lock sidecar: a lock held on a renamed
# inode excludes nobody.
LOCK = os.path.join(d, 'locks', 'config.json.lock')
check('the lock sidecar is where both builds look for it', os.path.exists(LOCK), LOCK)
before = os.stat(LOCK).st_ino
subprocess.run([sys.executable, '-c',
                'from codemason.utils.config import get_config, save_config\n'
                'c = get_config(); c.theme = "dark"; save_config()'],
               env=dict(ENV, CODEMASON_CONFIG_DIR=d), capture_output=True)
check('a save leaves the lock inode untouched (never renamed, never unlinked)',
      os.path.exists(LOCK) and os.stat(LOCK).st_ino == before)
check('the lock file stays empty — it carries no state that could go stale',
      os.stat(LOCK).st_size == 0, f'{os.stat(LOCK).st_size} bytes')
shutil.rmtree(d, ignore_errors=True)

# ── 2. A killed holder leaves nothing behind ────────────────────────────────
#
# The whole reason the lock is an flock and not a pid file or an O_EXCL create:
# the kernel drops it when the descriptor goes, however the process ended. If
# that were not true, one kill -9 would wedge every later run.
d = fresh_dir()
HOLDER = '''
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
from codemason.utils.file import FileLock
with FileLock(sys.argv[1]) as held:
    print('held' if held else 'not-held', flush=True)
    time.sleep(60)
'''
lock = os.path.join(d, 'locks', 'config.json.lock')
holder = subprocess.Popen([sys.executable, '-c', HOLDER, lock], env=dict(ENV, CODEMASON_CONFIG_DIR=d),
                          stdout=subprocess.PIPE, text=True)
check('a holder reports the lock as held', holder.stdout.readline().strip() == 'held')

TAKE = '''
import sys
from codemason.utils.file import FileLock
with FileLock(sys.argv[1], timeout=float(sys.argv[2])) as held:
    print('held' if held else 'not-held')
'''


def take(timeout='0.2'):
    r = subprocess.run([sys.executable, '-c', TAKE, lock, timeout],
                       env=dict(ENV, CODEMASON_CONFIG_DIR=d), capture_output=True, text=True)
    return r.stdout.strip()


check('a second process cannot take a held lock', take() == 'not-held')
holder.kill()
holder.wait()
check('a force-killed holder frees the lock at once — no stale state to reclaim',
      take() == 'held')
check('the lock file survives the holder, as it must (unlinking it would allow two holders)',
      os.path.exists(lock))

# ── 3. A lock that cannot be taken degrades, it does not fail ──────────────
#
# Not acquiring leaves exactly the pre-lock behaviour: a possible lost update,
# never a corrupt file, because the write is still an atomic rename. A command
# the user just typed must not report failure because another session was mid-save.
holder = subprocess.Popen([sys.executable, '-c', HOLDER, lock], env=dict(ENV, CODEMASON_CONFIG_DIR=d),
                          stdout=subprocess.PIPE, text=True)
holder.stdout.readline()
r = subprocess.run([sys.executable, '-c',
                    'from codemason.utils.config import get_config, save_config\n'
                    'c = get_config(); c.theme = "light"; print(save_config())'],
                   env=dict(ENV, CODEMASON_CONFIG_DIR=d), capture_output=True, text=True)
holder.kill()
holder.wait()
check('a save still reports success and lands while another process holds the lock',
      r.stdout.strip().splitlines()[-1:] == ['True'] and cfg(d)['theme'] == 'light',
      r.stdout.strip() + r.stderr.strip()[:200])
shutil.rmtree(d, ignore_errors=True)

sys.exit(0 if all(results) else 1)
