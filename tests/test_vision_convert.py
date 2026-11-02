"""Image blocks in convert_to_chat_messages (services/_openai_common.py).

Runs under pytest and as a script. Guards four things:
  - a text-only conversation serializes exactly as before vision existed
    (vision on or off), so turning vision on changes no existing request;
  - a user image goes out as an OpenAI image_url part, and the message survives
    the cleanup pass (content is None, everything is in content_parts);
  - an image a tool returned rides inside that tool message, with no user
    message after it (the chat template would blank the loop's reasoning);
  - with vision off no base64 payload reaches the request (it used to be
    pasted into the tool text by str(block));
  - every image carries an [Image N] label, and past the server cap the oldest
    go out as text, in batches, so the request prefix stays put in between.
"""
import json
import os
import sys
import tempfile

import atexit  # noqa: E402
import shutil  # noqa: E402
os.environ['CODEMASON_CONFIG_DIR'] = tempfile.mkdtemp(prefix='cm-test-cfg-')
atexit.register(shutil.rmtree, os.environ['CODEMASON_CONFIG_DIR'], True)  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from codemason.services._openai_common import convert_to_chat_messages, images_to_evict  # noqa: E402

B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC'
IMG = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': B64}}


def U(content):
    return {'type': 'user', 'message': {'role': 'user', 'content': content}}


def A(content):
    return {'type': 'assistant', 'uuid': '', 'message': {'role': 'assistant', 'content': content}}


def dump(msgs):
    return [m.to_dict() for m in msgs]


TEXT_CONV = [
    U('hello'),
    A([{'type': 'text', 'text': 'ok'},
       {'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {'command': 'ls'}}]),
    U([{'type': 'tool_result', 'tool_use_id': 't1', 'content': [{'type': 'text', 'text': 'a.txt'}]}]),
    U([{'type': 'text', 'text': 'next'}]),
]

# What the pre-vision code produced for TEXT_CONV (system prompt omitted).
EXPECTED_TEXT = [
    {'role': 'user', 'content': 'hello'},
    {'role': 'assistant', 'content': 'ok', 'tool_calls': [
        {'id': 't1', 'type': 'function', 'function': {'name': 'Bash', 'arguments': '{"command": "ls"}'}}]},
    {'role': 'tool', 'content': 'a.txt', 'tool_call_id': 't1'},
    {'role': 'user', 'content': 'next'},
]


def test_text_only_unchanged():
    for max_images in (0, 8):
        got = dump(convert_to_chat_messages(TEXT_CONV, [], max_images=max_images))
        assert got == EXPECTED_TEXT, (max_images, json.dumps(got, ensure_ascii=False))


def test_user_image_part():
    got = dump(convert_to_chat_messages([U([{'type': 'text', 'text': 'what?'}, IMG])], [], max_images=8))
    assert len(got) == 1, got
    parts = got[0]['content']
    assert [p['type'] for p in parts] == ['text', 'text', 'image_url'], parts
    assert parts[1]['text'] == '[Image 1]', parts
    assert parts[2]['image_url']['url'] == 'data:image/png;base64,' + B64


def test_tool_result_image():
    conv = [U('read it'),
            A([{'type': 'tool_use', 'id': 'r1', 'name': 'Read', 'input': {'file_path': '/x.png'}}]),
            U([{'type': 'tool_result', 'tool_use_id': 'r1',
                'content': [{'type': 'text', 'text': 'Read /x.png'}, IMG]}])]
    got = dump(convert_to_chat_messages(conv, [], max_images=8))
    tool = got[-1]  # no user message after it: that would blank the loop's reasoning
    assert tool['role'] == 'tool' and tool['tool_call_id'] == 'r1', got
    assert [p['type'] for p in tool['content']] == ['text', 'text', 'image_url'], tool
    assert [p.get('text') for p in tool['content'][:2]] == ['Read /x.png', '[Image 1]'], tool


def test_vision_off_no_payload():
    conv = [U([{'type': 'text', 'text': 'see'}, IMG]),
            A([{'type': 'tool_use', 'id': 'r1', 'name': 'Read', 'input': {}}]),
            U([{'type': 'tool_result', 'tool_use_id': 'r1', 'content': [IMG]}])]
    out = json.dumps(dump(convert_to_chat_messages(conv, [], max_images=0)))
    assert B64 not in out, out
    assert 'image_url' not in out, out
    assert '[Image content not supported]' in out, out


def test_evict_count():
    # All seen: limit 8 -> batches of 4; at least 5 stay attached.
    table = {(0, 8): 0, (8, 8): 0, (9, 8): 4, (12, 8): 4, (13, 8): 8, (16, 8): 8,
             (17, 8): 12, (3, 0): 0, (2, 1): 1, (5, 1): 4}
    for (total, limit), want in table.items():
        got = images_to_evict(total, limit, total)
        assert got == want, (total, limit, got)
    # Unseen images go only as far as the overflow demands.
    for total, seen, want in [(10, 0, 2), (9, 8, 4), (10, 9, 4), (14, 10, 8), (14, 4, 6), (12, 2, 4)]:
        got = images_to_evict(total, 8, seen)
        assert got == want, (total, seen, got)


def test_unseen_burst_keeps_the_limit():
    # Ten images from one parallel Read burst, not answered yet: the request
    # carries the full 8, only the overflow (2) goes as text.
    calls = [{'type': 'tool_use', 'id': f'r{i}', 'name': 'Read', 'input': {}} for i in range(10)]
    results = [{'type': 'tool_result', 'tool_use_id': f'r{i}', 'content': [IMG]} for i in range(10)]
    got = dump(convert_to_chat_messages([U('read all'), A(calls), U(results)], [], max_images=8))
    assert sum(p['type'] == 'image_url' for m in got if isinstance(m.get('content'), list)
               for p in m['content']) == 8


def _image_conv(n):
    conv = []
    for i in range(n):
        conv += [U([{'type': 'text', 'text': f'look {i + 1}'}, IMG]), A([{'type': 'text', 'text': 'ok'}])]
    return conv


def test_oldest_images_go_as_text():
    got = dump(convert_to_chat_messages(_image_conv(12), [], max_images=8))
    users = [m for m in got if m['role'] == 'user']
    assert sum(p['type'] == 'image_url' for m in users for p in m['content']) == 8
    for i, m in enumerate(users, 1):
        labels = [p['text'] for p in m['content'] if p['type'] == 'text' and p['text'].startswith('[Image')]
        if i <= 4:
            assert labels[0].startswith(f'[Image {i}: no longer attached — you saw it earlier'), labels
            assert 'saved at' in labels[0], labels
            assert not any(p['type'] == 'image_url' for p in m['content']), m
        else:
            assert labels == [f'[Image {i}]'], labels


def test_prefix_stable_between_batches():
    # 9..12 images evict the same 4, so each request starts with the previous one.
    prev = None
    for n in range(9, 13):
        cur = dump(convert_to_chat_messages(_image_conv(n), [], max_images=8))
        if prev is not None:
            assert cur[:len(prev)] == prev, n
        prev = cur
    # The 13th image is the next batch: the prefix changes once, at image 5.
    nxt = dump(convert_to_chat_messages(_image_conv(13), [], max_images=8))
    assert nxt[:8] == prev[:8] and nxt[8] != prev[8]


def test_evicted_label_matches_rust():
    # Same request text as the Rust port for the same conversation.
    got = dump(convert_to_chat_messages(_image_conv(9), [], max_images=8))
    t = got[0]['content'][1]['text']
    assert t.startswith('[Image 1: no longer attached — you saw it earlier; dropped to stay within '
                        'the server limit of 8 images per request; saved at '), t
    assert t.endswith('.png (Read it to look again)]'), t


def test_unshown_image_says_so():
    # The burst of ten, answered once and then one more image: images 1-2 were
    # never attached to an answered request, 3-8 were.
    calls = [{'type': 'tool_use', 'id': f'r{i}', 'name': 'Read', 'input': {}} for i in range(10)]
    results = [{'type': 'tool_result', 'tool_use_id': f'r{i}', 'content': [IMG]} for i in range(10)]
    conv = [U('read all'), A(calls), U(results),
            A([{'type': 'tool_use', 'id': 'x', 'name': 'Read', 'input': {}}]),
            U([{'type': 'tool_result', 'tool_use_id': 'x', 'content': [IMG]}])]
    got = dump(convert_to_chat_messages(conv, [], max_images=8))
    labels = [p['text'] for m in got if isinstance(m.get('content'), list)
              for p in m['content'] if p['type'] == 'text' and p['text'].startswith('[Image')]
    assert labels[0].startswith('[Image 1: not attached') and 'you have not seen it' in labels[0], labels
    assert labels[1].startswith('[Image 2: not attached'), labels
    assert labels[2].startswith('[Image 3: no longer attached — you saw it earlier'), labels
    assert labels[3].startswith('[Image 4: no longer attached — you saw it earlier'), labels
    assert labels[4] == '[Image 5]', labels


if __name__ == '__main__':
    tests = [test_text_only_unchanged, test_user_image_part, test_tool_result_image, test_vision_off_no_payload,
             test_evict_count, test_unseen_burst_keeps_the_limit, test_oldest_images_go_as_text, test_prefix_stable_between_batches,
             test_evicted_label_matches_rust, test_unshown_image_says_so]
    failed = 0
    for t in tests:
        try:
            t()
            print('PASS', t.__name__)
        except AssertionError as e:
            failed += 1
            print('FAIL', t.__name__, str(e)[:300])
    sys.exit(1 if failed else 0)
