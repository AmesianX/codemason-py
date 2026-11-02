"""The dumps that explain a turn nobody saw finish.

When a generation is cut — a stall guard fired, or the user hit Esc — the
reasoning and content produced so far are dropped from the conversation on
purpose. They go to logs/<kind>-<ts>.txt instead, because that is the only
record left of a turn that otherwise just stops.

Both dumps were dead: they imported now_iso from utils.log, which did not
define it, so every call raised ImportError into the surrounding except and
logged 'could not write the ... dump' instead of writing one. The suite stayed
green because nothing asserted a file appeared. These cases assert the file.

  written       the dump exists and holds the reasoning and the content
  0600 / 0700   a dump carries request material, like every other file in
                logs/; a bare write_text lands it 0644 under a 0755 directory
  named         <kind>-<iso with ':' and '.' as '-'>.txt, the same shape the
                Rust build writes into the same directory
  -dbg          the dump follows CODEMASON_LOG_DIR, and the directory it
                lands in is the one that got chmod'd

Run directly (exit 0 = all pass).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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
ENV = dict(os.environ, PYTHONPATH=ROOT, LANG='C')

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


def mode(path):
    return oct(os.stat(path).st_mode & 0o777)


# NTFS has no POSIX mode bits. stat() there reports 0o666 for any writable file
# and 0o444 for a read-only one, whatever mode the create asked for, so a 0600
# case cannot pass on Windows however the file was opened -- and it is not the
# thing keeping the file private there: these paths sit under the user profile
# directory and inherit its ACL, which no mode can express. The cases below
# therefore say nothing on Windows rather than reporting a defect that is the
# platform.
POSIX_MODES = os.name != 'nt'


def check_mode(name, path, expected):
    """A 0600/0700 case, skipped where modes mean nothing."""
    if not POSIX_MODES:
        print('SKIP  ' + name + ' (no POSIX mode bits on this platform)')
        return
    check(name, mode(path) == expected, mode(path))


# Written in a child process: the config module resolves its paths once at
# import, and -dbg repoints them afterwards, so only a fresh process shows
# which directory a dump really lands in.
WRITE = '''
from codemason.services._openai_common import _write_cut_dump
_write_cut_dump('stalled', 'no token for 120s', 'REASON-TEXT', 'CONTENT-TEXT', 12.5)
'''


def write_dump(env):
    r = subprocess.run([sys.executable, '-c', WRITE], env=env,
                       capture_output=True, text=True, timeout=60)
    return r


# ── 1. The dump is written, and holds what it is for ───────────────────────
d = tempfile.mkdtemp(prefix='cm-dump-')
r = write_dump(dict(ENV, CODEMASON_CONFIG_DIR=d))
logs = os.path.join(d, 'logs')
# Only the dump. logs/ also holds error.log, which the failing path writes
# itself — counting every file made this pass on the very failure it is here to
# catch: 'could not write the stalled dump: cannot import name now_iso'.
all_files = sorted(os.listdir(logs)) if os.path.isdir(logs) else []
names = [n for n in all_files if n.startswith('stalled-')]
check('a cut generation leaves a dump behind', len(names) == 1,
      f'dumps={names} all={all_files} / stderr: {r.stderr.strip()[-200:]}')

if names:
    body = open(os.path.join(logs, names[0]), encoding='utf-8').read()
    check('the dump holds the reasoning and the content it was cut from',
          'REASON-TEXT' in body and 'CONTENT-TEXT' in body, body[:200])
    check('the dump says why it was cut and how long it ran',
          'no token for 120s' in body and 'cut after 12.5s' in body, body[:200])

    # ── 2. Permissions. A dump is request material like every other log. ────
    check_mode('the dump file is 0600', os.path.join(logs, names[0]), '0o600')
    check_mode('the logs directory is 0700', logs, '0o700')

    # ── 3. The name is the shape the Rust build writes into this same dir ──
    check('named <kind>-<iso>.txt, colons and dots folded to dashes',
          bool(re.fullmatch(r'stalled-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}\.txt', names[0])),
          names[0])
shutil.rmtree(d, ignore_errors=True)

# ── 4. now_iso is the cross-build format, not isoformat() ──────────────────
#
# Rust: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f"). isoformat() drops
# the fractional part when the microsecond is 0, which would name one dump in a
# million differently — and a stall that rare is the one worth reading.
from codemason.utils.log import now_iso                     # noqa: E402

check('now_iso always writes six fractional digits',
      bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}', now_iso())), now_iso())
import datetime as _dt                                      # noqa: E402
check('now_iso keeps the fraction even at a whole second',
      _dt.datetime(2026, 1, 2, 3, 4, 5, 0).strftime('%Y-%m-%dT%H:%M:%S.%f')
      == '2026-01-02T03:04:05.000000')

# ── 5. -dbg repoints the log dir AFTER import ──────────────────────────────
#
# This is the case the env var cannot reach: -dbg reassigns config.LOG_DIR while
# parsing arguments, long after utils.log imported it. ensure_log_dir used that
# captured value, so it created and chmod'd the old directory while the dump was
# addressed to the new one — which then did not exist, so no dump at all.
REPOINT = '''
import pathlib, sys
from codemason.utils import config as cfg
cfg.LOG_DIR = pathlib.Path(sys.argv[1])          # what -dbg does
from codemason.services._openai_common import _write_cut_dump
_write_cut_dump('stalled', 'repointed', 'R', 'C', 1.0)
'''
d2 = tempfile.mkdtemp(prefix='cm-dump-cfg2-')
moved2 = os.path.join(tempfile.mkdtemp(prefix='cm-dump-dbg-'), 'dbg')
r2 = subprocess.run([sys.executable, '-c', REPOINT, moved2],
                    env=dict(ENV, CODEMASON_CONFIG_DIR=d2), capture_output=True,
                    text=True, timeout=60)
got2 = [n for n in (sorted(os.listdir(moved2)) if os.path.isdir(moved2) else [])
        if n.startswith('stalled-')]
check('a log dir repointed after import still gets the dump', len(got2) == 1,
      f'{got2} / stderr: {r2.stderr.strip()[-200:]}')
check('and THAT directory exists', os.path.isdir(moved2), 'missing')
check_mode('and THAT directory is the one chmod 0700', moved2, '0o700')
shutil.rmtree(d2, ignore_errors=True)
shutil.rmtree(os.path.dirname(moved2), ignore_errors=True)

# ── 6. CODEMASON_LOG_DIR (resolved at import) is honoured too ───────────────
d = tempfile.mkdtemp(prefix='cm-dump-cfg-')
ld = os.path.join(tempfile.mkdtemp(prefix='cm-dump-log-'), 'relocated')
r = write_dump(dict(ENV, CODEMASON_CONFIG_DIR=d, CODEMASON_LOG_DIR=ld))
moved = [n for n in (sorted(os.listdir(ld)) if os.path.isdir(ld) else [])
         if n.startswith('stalled-')]
check('the dump lands in CODEMASON_LOG_DIR', len(moved) == 1,
      f'{moved} / stderr: {r.stderr.strip()[-200:]}')
check('the relocated log dir exists', os.path.isdir(ld), 'missing')
check_mode('the relocated log dir is the one that got 0700', ld, '0o700')
_default = os.path.join(d, 'logs')
check('no dump was written to the default logs/ instead',
      not os.path.isdir(_default)
      or not [n for n in os.listdir(_default) if n.startswith('stalled-')])
shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(os.path.dirname(ld), ignore_errors=True)

# ── 7. Every log file this build creates is 0600 ───────────────────────────
#
# log_turn appended with a bare open(path, 'a'), so turns.log landed 0644 while
# the directory and its siblings were 0600 — the one writer in utils.log that
# skipped _open_log, and the one file the Rust build (append_line, mode 0600)
# and this one disagreed about. Found by running the app, not by reading it.
MAKE_LOGS = '''
from codemason.utils.log import log_turn, log_error, debug_log
log_turn({'prompt': 1, 'out': 2, 'dur_s': 0.1})
log_error('an error line', 'ctx')
'''
d3 = tempfile.mkdtemp(prefix='cm-logmode-')
subprocess.run([sys.executable, '-c', MAKE_LOGS],
               env=dict(ENV, CODEMASON_CONFIG_DIR=d3), capture_output=True, timeout=60)
l3 = os.path.join(d3, 'logs')
made = sorted(os.listdir(l3)) if os.path.isdir(l3) else []
check('log_turn and log_error both left a file', len(made) >= 2, str(made))
if POSIX_MODES:
    bad = [n for n in made if mode(os.path.join(l3, n)) != '0o600']
    check('every log file this build creates is 0600', not bad,
          str([(n, mode(os.path.join(l3, n))) for n in bad]))
else:
    print('SKIP  every log file this build creates is 0600'
          ' (no POSIX mode bits on this platform)')
shutil.rmtree(d3, ignore_errors=True)

# ── 8. Concurrent appends from several processes lose nothing ───────────────
#
# Every codemason process on this machine appends to the same turns.log. This is
# the shape observed live: four instances sharing one config dir interleaved
# their turn lines into one file. O_APPEND is what makes it safe; the case is
# here so a future bare open(path, 'w') or a read-modify-write cannot pass.
APPEND = '''
import sys
from codemason.utils.log import log_turn
tag, n = sys.argv[1], int(sys.argv[2])
for i in range(n):
    log_turn({'proc': tag, 'seq': i})
'''
d4 = tempfile.mkdtemp(prefix='cm-append-')
PROCS, PER = 4, 40
ps = [subprocess.Popen([sys.executable, '-c', APPEND, f'p{k}', str(PER)],
                       env=dict(ENV, CODEMASON_CONFIG_DIR=d4),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      for k in range(PROCS)]
for q in ps:
    q.wait(timeout=120)
turns = os.path.join(d4, 'logs', 'turns.log')
lines = open(turns, encoding='utf-8').read().splitlines() if os.path.exists(turns) else []
check(f'all {PROCS * PER} turn lines from {PROCS} processes are present',
      len(lines) == PROCS * PER, f'{len(lines)} lines')
# A torn line is the real failure mode: two writes interleaved inside one line.
torn = [l for l in lines
        if not re.fullmatch(r'\[[\d\-T:.]+\] proc=p\d seq=\d+', l)]
check('no line was torn by another process writing mid-line', not torn,
      str(torn[:3]))
check('every process got all of its lines in',
      all(sum(1 for l in lines if f'proc=p{k} ' in l) == PER for k in range(PROCS)),
      str({f'p{k}': sum(1 for l in lines if f'proc=p{k} ' in l) for k in range(PROCS)}))
shutil.rmtree(d4, ignore_errors=True)

sys.exit(0 if all(results) else 1)
