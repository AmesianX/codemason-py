"""Per-family request parameters: what actually leaves the client for GLM-5.3
vs DeepSeek-V4, and what a recovery request does differently.

Runs the real streaming path against a local SSE server that keeps the POST
body, so these are assertions about the bytes on the wire, not about a helper.
"""
import asyncio, json, os, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# This test drives the real client against a fixture server, so it must not read
# the user's live ~/.codemason: a real model_aes_key there encrypts the request
# body end-to-end and the fixture sees ciphertext it cannot parse — which is how
# both of these tests came to "fail" on a clean tree while the code was fine.
# A throwaway config dir, set before codemason is imported (the config module
# resolves its paths once, at import).
import atexit  # noqa: E402
import shutil  # noqa: E402
os.environ['CODEMASON_CONFIG_DIR'] = tempfile.mkdtemp(prefix='cm-test-cfg-')
atexit.register(shutil.rmtree, os.environ['CODEMASON_CONFIG_DIR'], True)  # noqa: E402

SENT = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        SENT['body'] = json.loads(self.rfile.read(n) or b'{}')
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()

        def send(o):
            self.wfile.write(b'data: ' + json.dumps(o).encode() + b'\n\n')
            self.wfile.flush()

        send({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hi'},
                           'finish_reason': None}]})
        send({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()


srv = HTTPServer(('127.0.0.1', 0), H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

os.environ['CODEMASON_MODEL_BASE_URL'] = f'http://127.0.0.1:{port}'
os.environ['CODEMASON_BACKEND'] = 'vllm'
os.environ.pop('TEMPERATURE', None)
sys.path.insert(0, os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..')))

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

from codemason.services._openai_common import query_openai_compatible  # noqa: E402
from codemason.utils.config import get_config  # noqa: E402
from codemason.utils.messages import create_user_message  # noqa: E402


async def send_one(model, options=None):
    cfg = get_config()
    cfg.model_name = model
    return await query_openai_compatible(
        [create_user_message('hi')], ['sys'], [],
        options=options or {},
        continuous_usage_stats=True, repetition_detection=True,
        reasoning_roundtrip=True,
    )


results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'  [{detail}]' if detail and not ok else ''))
    results.append(ok)
    return ok


loop = asyncio.new_event_loop()
cfg = get_config()
cfg.sampling_profile = 'vendor'
cfg.temperature, cfg.top_k, cfg.top_p = 0.6, 40, 0.95
cfg.repetition_penalty, cfg.presence_penalty, cfg.frequency_penalty = 1.0, 0.0, 0.0
cfg.clear_thinking = True

# 1. auto (the default) ignores the file and sends the family table.
cfg.sampling_profile = 'auto'
cfg.temperature, cfg.top_k, cfg.top_p = 0.123, 7, 0.111
cfg.frequency_penalty = 0.9
loop.run_until_complete(send_one('glm-5.3-flash'))
b = SENT['body']
check('auto/GLM: vendor temperature 1.0', b.get('temperature') == 1.0, b.get('temperature'))
check('auto/GLM: top_k omitted', 'top_k' not in b, list(b))
check('auto/GLM: top_p 0.95 regardless of the file', b.get('top_p') == 0.95, b.get('top_p'))
check('auto/GLM: min_p pinned at 0', b.get('min_p') == 0.0, b.get('min_p'))
check('auto/GLM: no penalties, even one left in the file',
      not {'repetition_penalty', 'presence_penalty', 'frequency_penalty'} & set(b), list(b))

# 2. auto on any other family sends the DeepSeek recommendation, not the file.
loop.run_until_complete(send_one('DSV4-Flash'))
b = SENT['body']
check('auto/DSV4: recommended temperature 0.6', b.get('temperature') == 0.6, b.get('temperature'))
check('auto/DSV4: recommended top_k 40', b.get('top_k') == 40, b.get('top_k'))
check('auto/DSV4: the file value never reaches the wire', b.get('temperature') != 0.123)

# 3. manual hands the wheel back: the file goes out verbatim.
cfg.sampling_profile = 'manual'
cfg.temperature, cfg.top_k = 0.42, 7
loop.run_until_complete(send_one('glm-5.3-flash'))
b = SENT['body']
check('manual: file temperature wins', b.get('temperature') == 0.42, b.get('temperature'))
check('manual: file top_k is sent', b.get('top_k') == 7, b.get('top_k'))
check('manual: an enabled penalty is sent', b.get('frequency_penalty') == 0.9, list(b))

# 4. The environment outranks both profiles.
os.environ['TEMPERATURE'] = '0.77'
loop.run_until_complete(send_one('glm-5.3-flash'))
check('env: beats the file in manual', SENT['body'].get('temperature') == 0.77,
      SENT['body'].get('temperature'))
cfg.sampling_profile = 'auto'
loop.run_until_complete(send_one('glm-5.3-flash'))
check('env: beats the family table in auto', SENT['body'].get('temperature') == 0.77,
      SENT['body'].get('temperature'))
os.environ.pop('TEMPERATURE', None)
cfg.temperature, cfg.top_k, cfg.top_p, cfg.frequency_penalty = 0.6, 40, 0.95, 0.0

# 5. Ordinary turn clears prior-turn thinking; both family kwarg names go out.
loop.run_until_complete(send_one('glm-5.3-flash'))
k = SENT['body'].get('chat_template_kwargs', {})
check('ordinary turn: clear_thinking true (GLM name)', k.get('clear_thinking') is True, k)
check('ordinary turn: drop_thinking true (DeepSeek name)', k.get('drop_thinking') is True, k)

# 6. A recovery request keeps the thinking it has to resume from, and asks for
#    no new reasoning.
loop.run_until_complete(send_one('glm-5.3-flash',
                                 {'no_reasoning': True, 'keep_thinking': True}))
b = SENT['body']
k = b.get('chat_template_kwargs', {})
check('recovery: clear_thinking false', k.get('clear_thinking') is False, k)
check('recovery: drop_thinking false', k.get('drop_thinking') is False, k)
check("recovery: reasoning_effort 'low' (omitting it lands on the server's own high)",
      b.get('reasoning_effort') == 'low', b.get('reasoning_effort'))

# 7. Compaction sets no_reasoning only — it must NOT drag every earlier turn's
#    reasoning into the one request whose job is to shrink the history.
loop.run_until_complete(send_one('glm-5.3-flash', {'no_reasoning': True}))
b = SENT['body']
k = b.get('chat_template_kwargs', {})
check('compaction: thinking still cleared', k.get('clear_thinking') is True, k)
check("compaction: reasoning_effort 'low'",
      b.get('reasoning_effort') == 'low', b.get('reasoning_effort'))

srv.shutdown()
print('\n' + ('ALL PASS' if all(results) else 'FAILURES PRESENT'))
sys.exit(0 if all(results) else 1)
