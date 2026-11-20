"""The /ab auto lifecycle, against a stub server.

Mirrors services/ab.rs's hook_tests. The failure this guards is the switch left
on after the turn that armed it, which `auto` exists to prevent and which is
invisible by inspection. Note that NO tool call appears anywhere below: the off
used to hang off one, and turns that never call a tool (a text answer, an empty
turn, an abort, a cut generation) left the server abliterated -- see the
services/ab.py docstring for the 2026-10-01 incident.

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
    """Records every scale posted to /ab. Returns (list, port, shutdown)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            seen.append(body['scale'])
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

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
    ab._pushed = None
    ab._armed = False
    ab._mode = ab.OFF


seen, port, shutdown = start_stub()
point_at(port)


async def auto_lifecycle():
    await ab.set_mode(ab.AUTO)        # entering auto parks it off
    await ab.on_user_turn()           # arms only -- must cost no request
    after_arming = list(seen)
    on = await ab.begin_request()     # 1.0
    await ab.end_request(on)          # 0.0 -- and no tool call to get here
    # Later requests of the same turn: the latch is spent, so the switch is not
    # touched. These stand in for the repetition retries that used to go out on
    # the edited weights because the off waited for a tool call they never made.
    laters = [await _pair() for _ in range(3)]
    return after_arming, on, laters


async def _pair():
    again = await ab.begin_request()
    await ab.end_request(again)
    return again


after_arming, turned_on, laters = asyncio.run(auto_lifecycle())
check('auto: arming the turn costs no request', after_arming == [0.0], after_arming)
check("auto: on for the turn's first request", turned_on is True)
check('auto: off as soon as it answers, with no tool call involved',
      seen == [0.0, 1.0, 0.0], seen)
check('auto: every later request of the turn is left alone',
      laters == [False, False, False], laters)

seen.clear()
# Fresh state: otherwise the skip-if-already-there cache from the run above
# swallows this one's first push, and the test would assert the cache rather
# than the mode's behaviour.
ab._pushed = None
ab._armed = False


async def off_lifecycle():
    await ab.set_mode(ab.OFF)
    await ab.on_user_turn()
    on = await ab.begin_request()
    await ab.end_request(on)
    return on


on_while_pinned_off = asyncio.run(off_lifecycle())
check('off: parks the server off once and never touches it again',
      seen == [0.0], seen)
check('off: the request pair is a no-op', on_while_pinned_off is False)

check('parse_mode takes the three modes', (
    ab.parse_mode('auto') == ab.AUTO
    and ab.parse_mode(' ON ') == ab.ON
    and ab.parse_mode('0') == ab.OFF
))
check('parse_mode rejects anything else', ab.parse_mode('maybe') is None)

# --- an explicit mode command is never answered from the cache --------------
# `_pushed` is a memo about state we do not own and cannot invalidate: the
# server restarting or another client pushing rots it. With a rotten memo,
# `/ab on` used to return success without sending a byte and print "on" while
# the server sat at 0.0 -- and in ON mode begin_request() returns immediately,
# so nothing ever pushed 1.0 again. Permanently stuck, under an honest-looking
# display. Plant the rotten belief and require the push anyway.
seen.clear()
ab._armed = False
ab._pushed = 1.0                      # we believe the server is already on
err = asyncio.run(ab.set_mode(ab.ON))
check('a user-typed /ab on is sent even when we think it is already on',
      seen == [1.0] and err is None, (seen, err))

# --- pinning the edit mid-turn survives that turn ending --------------------
# The off is armed by begin_request() under the mode that held when the request
# STARTED, and the mode can change while it is in flight. Undoing a pin the
# user has since asked for is how the switch ended up stuck off while the REPL
# said "on".
seen.clear()
ab._pushed = None
ab._armed = False


async def pin_mid_turn():
    await ab.set_mode(ab.AUTO)        # parks off
    await ab.on_user_turn()           # arms
    on = await ab.begin_request()     # 1.0 -- request now in flight
    await ab.set_mode(ab.ON)          # user pins it mid-turn
    await ab.end_request(on)          # must NOT stomp the pin
    return on


pinned = asyncio.run(pin_mid_turn())
check('pinning on mid-turn survives that turn ending',
      pinned is True and seen[-1] == 1.0, seen)

# --- the memo records what the server took, not what we asked for -----------
# A server that answers with a different scale (another client raced us, or it
# is not our /ab at all) must not leave us believing our own number: the next
# push would be skipped as already-there.
lying = []


class Lying(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        lying.append(body['scale'])
        payload = json.dumps({'scale': 0.0, 'workers': [0.0, 0.0],
                              'consistent': True}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


liar = HTTPServer(('127.0.0.1', 0), Lying)
threading.Thread(target=liar.serve_forever, daemon=True).start()
point_at(liar.server_port)
err = asyncio.run(ab.set_mode(ab.ON))
check('a scale the server did not take is reported, not swallowed',
      err is not None and 'not' in err, err)
check('the memo holds the server\'s value, not ours', ab._pushed == 0.0, ab._pushed)
asyncio.run(ab.set_mode(ab.ON))
check('so the next push is not skipped as already-there', lying == [1.0, 1.0], lying)
liar.shutdown()

shutdown()
print('\n' + (f'ALL PASS ({len(results)})' if all(results)
              else f'FAILURES: {results.count(False)} / {len(results)}'))
sys.exit(0 if all(results) else 1)
