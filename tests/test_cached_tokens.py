"""Exercises the real streaming path: a local SSE server emits OpenAI-style
usage and we assert what BetaUsage ends up holding.

Runs under pytest (test_cached_tokens) and as a script. It used to call
sys.exit() at module level, which made pytest's collection die with an
INTERNALERROR and took every other test in the directory down with it.
"""
import asyncio
import json
import os
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
import threading
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

CASE = {}
D = 'prompt_tokens_details'


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()

        def send(o):
            self.wfile.write(b'data: ' + json.dumps(o).encode() + b'\n\n')
            self.wfile.flush()

        send({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hi'},
                           'finish_reason': None}]})
        for u in CASE['usages']:
            send({'choices': [], 'usage': u})
        send({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()


def _check(name, u, want_in, want_cache, want_total):
    got = (u.input_tokens, u.cache_read_input_tokens,
           u.input_tokens + u.cache_read_input_tokens
           + u.cache_creation_input_tokens + u.output_tokens)
    exp = (want_in, want_cache, want_total)
    ok = got == exp
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      got in/cache/total={got} want={exp}")
    return ok


def run_cases():
    """Serve the fixtures, drive the client through them, return per-case results."""
    srv = HTTPServer(('127.0.0.1', 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    os.environ['CODEMASON_MODEL_BASE_URL'] = f'http://127.0.0.1:{port}'
    os.environ['CODEMASON_BACKEND'] = 'vllm'
    os.environ['CODEMASON_MODEL_NAME'] = 'test-model'

    from codemason.services._openai_common import query_openai_compatible
    from codemason.utils.messages import create_user_message

    async def run(usages):
        CASE['usages'] = usages
        r = await query_openai_compatible(
            [create_user_message('hi')], ['sys'], [],
            signal=None, options={'dangerously_skip_permissions': False},
            continuous_usage_stats=True, repetition_detection=False,
            reasoning_roundtrip=False)
        return r.message.usage

    results = []
    loop = asyncio.new_event_loop()
    try:
        # 1. Warm turn: almost the whole 234k prompt served from the prefix cache.
        u = loop.run_until_complete(run([{'prompt_tokens': 234000, 'completion_tokens': 290,
                                          D: {'cached_tokens': 233000}}]))
        results.append(_check('warm turn splits prompt into uncached+cached',
                              u, 1000, 233000, 234290))

        # 2. Cold turn: server sends no details block at all (llama.cpp, old vLLM).
        u = loop.run_until_complete(run([{'prompt_tokens': 234000, 'completion_tokens': 290}]))
        results.append(_check('cold turn unchanged (no details block)',
                              u, 234000, 0, 234290))

        # 3. continuous_usage_stats: details arrive on an early chunk, final chunk
        #    carries none. The cache figure must survive being overwritten.
        u = loop.run_until_complete(run([
            {'prompt_tokens': 234000, 'completion_tokens': 10, D: {'cached_tokens': 233000}},
            {'prompt_tokens': 234000, 'completion_tokens': 290}]))
        results.append(_check('cache survives a later usage chunk without details',
                              u, 1000, 233000, 234290))

        # 4. Malformed server: more cached than prompt tokens must not go negative.
        u = loop.run_until_complete(run([{'prompt_tokens': 100, 'completion_tokens': 5,
                                          D: {'cached_tokens': 999999}}]))
        results.append(_check('cached > prompt clamps instead of going negative',
                              u, 0, 100, 105))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        srv.shutdown()
    return results


def test_cached_tokens():
    assert all(run_cases())


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.path.insert(0, os.path.abspath(sys.argv[1]))
    else:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    results = run_cases()
    print('\n' + ('ALL PASS' if all(results) else 'FAILURES PRESENT'))
    sys.exit(0 if all(results) else 1)
