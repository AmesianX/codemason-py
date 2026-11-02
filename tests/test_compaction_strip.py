"""The summary request carries no images (services/compaction.py
strip_images_for_summary, port of stripImagesFromMessages).

Guards:
  - user images and tool-result images become "[Image N: saved at PATH]",
    numbered in conversation order, and PATH holds the image;
  - the caller's messages are not modified (only the request copy changes);
  - compact_conversation sends no base64 to the summarizer.

Run: python3 tests/test_compaction_strip.py
"""
import asyncio
import json
import atexit
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

os.environ['CODEMASON_CONFIG_DIR'] = tempfile.mkdtemp(prefix='cm-test-cfg-')
atexit.register(shutil.rmtree, os.environ['CODEMASON_CONFIG_DIR'], True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from codemason.services import compaction  # noqa: E402
from codemason.services.compaction import strip_images_for_summary  # noqa: E402

B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC'
IMG = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': B64}}

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))
    assert ok, f'{name}: {detail}'   # so pytest fails too, not only the script


def conv():
    return [
        {'type': 'user', 'message': {'role': 'user', 'content': [{'type': 'text', 'text': 'see'}, IMG]}},
        {'type': 'assistant', 'message': {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': 'r1', 'name': 'Read', 'input': {}}]}},
        {'type': 'user', 'message': {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'r1', 'content': [{'type': 'text', 'text': 'Read /x.png'}, IMG]}]}},
        {'type': 'user', 'message': {'role': 'user', 'content': [{'type': 'text', 'text': 'plain'}]}},
    ]


def test_strip():
    msgs = conv()
    before = json.dumps(msgs)
    out = strip_images_for_summary(msgs)
    s = json.dumps(out)
    check('no base64 left', B64 not in s, s[:200])
    first = out[0]['message']['content'][1]['text']
    second = out[2]['message']['content'][0]['content'][1]['text']
    check('user image numbered 1 with path', first.startswith('[Image 1: saved at ') and first.endswith('.png]'), first)
    check('tool image numbered 2', second.startswith('[Image 2: saved at '), second)
    path = first[len('[Image 1: saved at '):-1]
    check('path holds the image', os.path.isfile(path), path)
    check('caller messages untouched', json.dumps(msgs) == before)
    check('image-free messages passed through as-is', out[1] is msgs[1] and out[3] is msgs[3])


def test_object_messages():
    m = SimpleNamespace(type='user', message=SimpleNamespace(role='user', content=[IMG]))
    out = strip_images_for_summary([m])
    check('object-shaped message stripped', out[0].message.content[0]['type'] == 'text')
    check('object-shaped original untouched', m.message.content[0] is IMG)


def test_compact_sends_no_images():
    sent = {}

    async def fake_query_model(messages, **kw):
        sent['messages'] = messages
        text = '<analysis>x</analysis><summary>' + 'The user showed an image and asked about it. ' * 3 + '</summary>'
        return SimpleNamespace(message=SimpleNamespace(content=[{'type': 'text', 'text': text}],
                                                       stop_reason='end_turn'),
                               is_api_error_message=False)

    import codemason.services.backend as backend
    orig = backend.query_model
    backend.query_model = fake_query_model
    try:
        asyncio.run(compaction.compact_conversation(conv(), SimpleNamespace(options={})))
    finally:
        backend.query_model = orig
    s = json.dumps(sent.get('messages', []), default=str)
    check('summary request reached the model', bool(sent.get('messages')))
    check('summary request has no base64', B64 not in s)
    check('summary request keeps the image numbers', '[Image 1: saved at' in s and '[Image 2: saved at' in s)


if __name__ == '__main__':
    for _t in (test_strip, test_object_messages, test_compact_sends_no_images):
        try:
            _t()
        except AssertionError:
            pass                    # already printed; run the rest
    sys.exit(0 if all(results) else 1)
