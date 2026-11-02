"""utils/image_store.py — image payloads live next to the transcript, not in it.

Runs under pytest and as a script."""
import base64
import json
import os
import sys
import tempfile

import atexit  # noqa: E402
import shutil  # noqa: E402
os.environ['CODEMASON_CONFIG_DIR'] = tempfile.mkdtemp(prefix='cm-test-cfg-')
atexit.register(shutil.rmtree, os.environ['CODEMASON_CONFIG_DIR'], True)  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from codemason.utils.image_store import dehydrate, hydrate, images_dir  # noqa: E402
from codemason.utils.state import set_session_id  # noqa: E402

B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC'
MSG = {'role': 'user', 'content': [
    {'type': 'text', 'text': 'look'},
    {'type': 'tool_result', 'tool_use_id': 'r1', 'content': [
        {'type': 'text', 'text': '[Image file: /x.png]'},
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': B64}}]}]}


def test_round_trip():
    set_session_id('sess-a')
    disk = dehydrate(MSG)
    s = json.dumps(disk)
    assert B64 not in s and 'codemason_file' in s, s
    files = list(images_dir('sess-a').iterdir())
    assert len(files) == 1 and files[0].read_bytes() == base64.b64decode(B64), files
    assert hydrate(disk, 'sess-a') == MSG
    assert MSG['content'][1]['content'][1]['source']['type'] == 'base64'  # in-memory untouched


def test_missing_file_becomes_text():
    set_session_id('sess-b')
    disk = dehydrate(MSG)
    for f in images_dir('sess-b').iterdir():
        f.unlink()
    back = hydrate(disk, 'sess-b')
    img = back['content'][1]['content'][1]
    assert img['type'] == 'text' and 'Image file missing' in img['text'], img


if __name__ == '__main__':
    failed = 0
    for t in (test_round_trip, test_missing_file_becomes_text):
        try:
            t()
            print('PASS', t.__name__)
        except AssertionError as e:
            failed += 1
            print('FAIL', t.__name__, str(e)[:200])
    sys.exit(1 if failed else 0)
