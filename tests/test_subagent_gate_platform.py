"""The sub-agent slot pool locks on Windows too.

utils/subagent_gate used to `import fcntl` unconditionally. The import is lazy
(the gate is imported inside the spawn path), so on Windows codemason started
and talked normally and then raised ImportError the moment a sub-agent was
actually spawned — sub-agents did not run at all, and nothing earlier said so.
The Rust build had carried the cfg(unix)/cfg(not(unix)) split from the start;
only Python was missing its half.

The platform cannot be installed here, so what is pinned is the contract the
two primitives are written against, with fcntl hidden and a stand-in msvcrt
that enforces Windows' semantics — a byte-range lock held per HANDLE, so a
second open of the same slot is refused even inside one process. Cases:

  imports      the module imports with no fcntl at all. This is the whole bug:
               it failed here, before any slot was ever asked for.
  locks        acquiring calls locking(LK_NBLCK, 1) at offset 0 — one byte of
               an empty file, which Windows allows past end of file.
  rations      a spawn that finds every slot held is refused, the property
               flock gives us on POSIX. A lock that did not contend would leave
               the gate switched off with nothing to see. The slot count is
               read from the gate, not assumed: a config migration may rewrite
               it, and assuming 1 read a free slot 2 as "does not ration".
  frees        release calls locking(LK_UNLCK, 1) and the slot comes back.
  degrades     a platform with NEITHER primitive is granted the slot rather
               than refused: there is no advisory locking to ask, and refusing
               would mean sub-agents never run instead of running unrationed.

Run directly (exit 0 = all pass). Each case runs in a child with a throwaway
CODEMASON_CONFIG_DIR, because config resolves its paths once at import, and in
a child because the stand-in has to be in place BEFORE the gate is imported.
"""
import json
import os
import pathlib
import shutil
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
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=ROOT, LANG='C')

results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f'{"PASS" if ok else "FAIL"}: {name}' + (f' — {detail}' if detail and not ok else ''))


# The stand-in. Windows' locking() takes a mandatory byte range at the current
# file position and enforces it per handle, so the file's identity (not the
# descriptor) is what two holders contend over — hence st_ino as the key.
SHIM = r'''
import os, sys, types

# Load the stdlib — and everything codemason's package __init__ drags in — while
# the real modules are still reachable, because parts of it sniff for these very
# names: subprocess decides it is on Windows by whether `import msvcrt` works,
# and then imports _winapi, which does not exist here. Shadowing first made the
# stdlib itself the thing under test. Only subagent_gate is imported after.
import codemason.utils  # noqa: F401
import asyncio, subprocess  # noqa: F401
sys.modules.pop('codemason.utils.subagent_gate', None)


class _Hide:
    """Make `import <name>` fail the way it fails on a platform without it.

    ModuleNotFoundError, not a bare ImportError: that is what CPython raises
    for a missing module, and it is the only one some stdlib probes catch.
    """
    def __init__(self, *names): self.names = set(names)
    def find_spec(self, name, path=None, target=None):
        if name in self.names:
            raise ModuleNotFoundError(f'no module named {name}', name=name)
        return None

sys.modules.pop('fcntl', None)
sys.meta_path.insert(0, _Hide('fcntl'))

CALLS = []

if %(with_msvcrt)s:
    msvcrt = types.ModuleType('msvcrt')
    msvcrt.LK_NBLCK, msvcrt.LK_UNLCK = 2, 0
    _held = {}

    def locking(fd, mode, nbytes):
        ino = os.fstat(fd).st_ino
        CALLS.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR), ino))
        if mode == msvcrt.LK_NBLCK:
            if ino in _held and _held[ino] != fd:
                raise OSError(36, 'Resource deadlock avoided')
            _held[ino] = fd
        elif mode == msvcrt.LK_UNLCK:
            _held.pop(ino, None)

    msvcrt.locking = locking
    sys.modules['msvcrt'] = msvcrt
else:
    sys.modules.pop('msvcrt', None)
    sys.meta_path.insert(0, _Hide('msvcrt'))

from codemason.utils import subagent_gate as g
assert g.fcntl is None, 'fcntl should be hidden'
'''


def run(body, with_msvcrt=True):
    d = tempfile.mkdtemp(prefix='cm-gate-')
    pathlib.Path(d, 'config.json').write_text(json.dumps({'max_concurrent_subagents': 1}))
    try:
        p = subprocess.run(
            [sys.executable, '-c', (SHIM % {'with_msvcrt': with_msvcrt}) + body],
            env=dict(ENV, CODEMASON_CONFIG_DIR=d), capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- imports -----------------------------------------------------------------
#
# Before the fix this child died on the import line itself. It is kept separate
# from the lock cases on purpose: an ImportError here is a different defect from
# a lock that does not hold, and collapsing them would hide which one broke.
rc, out = run("print('MSVCRT', g.msvcrt is not None)")
check('the gate imports with no fcntl on the platform', rc == 0 and 'MSVCRT True' in out, out)

# --- locks + rations + frees -------------------------------------------------
rc, out = run("""
# How many slots there are is read from the gate, never assumed: a config
# migration may rewrite max_concurrent_subagents out from under the throwaway
# config dir (next/parallel-subagents raises the old default 1 to 3), and a
# case that assumed 1 then found slot 2 free and read it as "does not ration".
n = g._slots()
held = [g.try_acquire() for _ in range(n)]
print('FIRST', all(h is not None for h in held), n)
print('CALL', CALLS[0][:3] if CALLS else None)
print('NEXT', g.try_acquire())              # every slot is held, so this waits
held[0].release()
print('UNLOCKED', CALLS[-1][:2])
b = g.try_acquire()
print('AGAIN', b is not None)
b.release()
for h in held[1:]:
    h.release()
""")
check('every slot is taken through msvcrt', rc == 0 and 'FIRST True' in out, out)
check('it locks one byte at offset 0 with LK_NBLCK',
      'CALL (2, 1, 0)' in out, out)
check('a spawn that finds every slot held is refused', 'NEXT None' in out, out)
check('release unlocks the same byte with LK_UNLCK', 'UNLOCKED (0, 1)' in out, out)
check('the freed slot is taken again', 'AGAIN True' in out, out)

# --- degrades ----------------------------------------------------------------
#
# Neither primitive is not the same as an acquisition failing: a platform that
# offers no advisory locking cannot be asked. Granting means unrationed, which
# is worse than rationed and far better than no sub-agents at all.
rc, out = run("""
print('BOTH NONE', g.fcntl is None and g.msvcrt is None)
s = g.try_acquire()
print('GRANTED', s is not None)
s.release()
print('RELEASED OK')
""", with_msvcrt=False)
check('with neither primitive the slot is granted, not refused',
      rc == 0 and 'BOTH NONE True' in out and 'GRANTED True' in out, out)
check('and releasing it is still a no-op that does not raise',
      rc == 0 and 'RELEASED OK' in out, out)

print(f'\n{sum(results)}/{len(results)} pass')
sys.exit(0 if all(results) else 1)
