"""/ab, per request, against a stub server.

Mirrors services/ab.rs's hook_tests. Two things are guarded:

1. Zero trust (2026-11-27): the client never writes the server's switch.
   `POST /ab` set the engine-wide default, which every request without its own
   scale follows, so one client's `/ab on` changed every other client's model
   (measured on the live lane). The server now answers 410; the client must not
   send it either. The edit rides on the request: `vllm_xargs.ab_scale`.
2. `auto` gives the edit to the turn's first request only -- and NO tool call
   appears anywhere below: the old off hung off one, and turns that never call
   a tool left the server abliterated (the 2026-10-01 incident in ab.py).

Run: python3 tests/test_ab_switch.py
"""

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from codemason.services import ab  # noqa: E402

results = []


def check(name, ok, detail=None):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail!r}" if not ok else ''))


def start_stub():
    """Records every request that reaches the stub. Returns (list, port, shutdown)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def _any(self):
            n = int(self.headers.get('Content-Length', 0) or 0)
            raw = self.rfile.read(n) if n else b''
            seen.append((self.command, self.path, raw.decode(errors='replace')))
            payload = b'{}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_POST = _any
        do_PUT = _any

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return seen, server.server_port, server.shutdown


def point_at(port):
    class Cfg:
        model_base_url = f'http://127.0.0.1:{port}'
        model_api_key = ''

    ab.get_config = lambda: Cfg()
    ab._armed = False
    ab._mode = ab.OFF


def set_mode(m):
    """set_mode, whether it is a coroutine (before 11-27) or not."""
    r = ab.set_mode(m)
    if asyncio.iscoroutine(r):
        asyncio.run(r)


seen, port, shutdown = start_stub()
point_at(port)

# --- 1. changing the mode never writes the server ----------------------------
# Failed on the code before 11-27: every set_mode POSTed {"scale":..}.
for m in (ab.ON, ab.OFF, ab.AUTO, ab.ON, ab.AUTO):
    set_mode(m)
    asyncio.run(ab.on_user_turn())
    if hasattr(ab, 'request_scale'):
        ab.request_scale()
        ab.request_scale()
check('changing the mode never writes the server', seen == [], seen)

# --- 2. auto: the turn's first request only ----------------------------------
set_mode(ab.AUTO)
got = []
got.append(ab.request_scale())          # before any user turn
asyncio.run(ab.on_user_turn())
got += [ab.request_scale() for _ in range(3)]
asyncio.run(ab.on_user_turn())
got += [ab.request_scale() for _ in range(2)]
check("auto: 1 for the turn's first request, explicit 0 for every other",
      got == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], got)

# --- 3. on / off pin every request; a user turn does not arm them ------------
set_mode(ab.ON)
asyncio.run(ab.on_user_turn())
on = [ab.request_scale() for _ in range(3)]
set_mode(ab.OFF)
asyncio.run(ab.on_user_turn())
off = [ab.request_scale() for _ in range(2)]
check('on pins 1 on every request', on == [1.0, 1.0, 1.0], on)
check('off pins 0 on every request', off == [0.0, 0.0], off)

# --- 4. a mode typed mid-turn drops the pending arm --------------------------
set_mode(ab.AUTO)
asyncio.run(ab.on_user_turn())
set_mode(ab.OFF)
a = ab.request_scale()
set_mode(ab.AUTO)
b = ab.request_scale()
check('a mode typed mid-turn wins over the old arm', (a, b) == (0.0, 0.0), (a, b))

check('parse_mode takes the three modes', (
    ab.parse_mode('auto') == ab.AUTO
    and ab.parse_mode(' ON ') == ab.ON
    and ab.parse_mode('0') == ab.OFF
))
check('parse_mode rejects anything else', ab.parse_mode('maybe') is None)

# --- 5. the wire: vllm_xargs on vLLM requests only ---------------------------
from codemason.services import _openai_common as oc  # noqa: E402
check('ab_scale rides as vllm_xargs on vLLM',
      oc._wire_vllm_xargs(1.0, True) == {'ab_scale': 1}, oc._wire_vllm_xargs(1.0, True))
check('off is said explicitly, not left to the server default',
      oc._wire_vllm_xargs(0.0, True) == {'ab_scale': 0})
check('compaction/helpers (None) send no key', oc._wire_vllm_xargs(None, True) is None)
check('llama.cpp never gets it', oc._wire_vllm_xargs(1.0, False) is None)
check('never NaN on the wire', oc._wire_vllm_xargs(float('nan'), True) is None)
check('nothing above reached the server', seen == [], seen)


# --- reading the server's answer -------------------------------------------
# The per-token switch split the body into a live scalar that is always 0.0 and
# a `default_scale` that is what actually gets applied. See ab._effective.

# The live lane's own GET, pasted from 2026-11-21 while the edit was ON.
check('per-token body reports the default, not the dead scalar',
      ab._effective({
          'scale': 0.0, 'workers': [0.0, 0.0], 'consistent': True,
          'mode': 'original', 'per_token': True, 'buffer_tokens': 9236,
          'fills': [41329, 41329],
          'details': [{'scale': 0.0, 'default_scale': 1.0, 'per_token': True},
                      {'scale': 0.0, 'default_scale': 1.0, 'per_token': True}],
      }) == (1.0, True))

# The server derives `consistent` from those always-zero scalars, so it says
# true while the ranks hold different defaults -- at TP=2 that means the halves
# of the model disagree about which weights they are.
check('per-token disagreement is caught though the server says consistent',
      ab._effective({'scale': 0.0, 'consistent': True, 'per_token': True,
                     'details': [{'default_scale': 1.0},
                                 {'default_scale': 0.0}]}) == (1.0, False))

# A server from before the per-token change carries no
# `per_token`: the top-level pair is the answer there and must stay so.
check('legacy body still reads the top-level pair',
      ab._effective({'scale': 1.0, 'workers': [1.0, 1.0],
                     'consistent': False, 'mode': 'abliterated'}) == (1.0, False))
# per_token with the details missing is not something the plugin sends (it adds
# both or neither), but guessing 0.0 there would be the same mistake again.
check('per_token without details falls back rather than inventing 0.0',
      ab._effective({'scale': 1.0, 'per_token': True}) == (1.0, True))
check('a body with no scale at all is no answer',
      ab._effective({'workers': []}) is None and ab._effective(None) is None)

shutdown()
print('\n' + (f'ALL PASS ({len(results)})' if all(results)
              else f'FAILURES: {results.count(False)} / {len(results)}'))
sys.exit(0 if all(results) else 1)
