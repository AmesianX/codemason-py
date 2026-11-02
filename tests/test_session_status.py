"""Image count and context fill shown on the PC footer and the phone
(utils/session_status.py, services/_openai_common.py attached_image_count).

Guards:
  - the count is what the next request carries: the same eviction rule as
    convert_to_chat_messages (checked against it directly);
  - no image part when the server takes no images;
  - the denominator is every image the conversation holds, so it parts from
    the attached count when the per-request limit pushes the oldest out, and
    the line says how many: the only place the user is told;
  - the Telegram turn message carries the status line next to the duration.

Run: python3 tests/test_session_status.py
"""
import atexit
import os
import shutil
import sys
import tempfile

os.environ['CODEMASON_CONFIG_DIR'] = tempfile.mkdtemp(prefix='cm-test-cfg-')
atexit.register(shutil.rmtree, os.environ['CODEMASON_CONFIG_DIR'], True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from codemason.services._openai_common import attached_image_count, convert_to_chat_messages  # noqa: E402

B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC'
IMG = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': B64}}
results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))
    assert ok, f'{name}: {detail}'   # so pytest fails too, not only the script


def U(c):
    return {'type': 'user', 'message': {'role': 'user', 'content': c}}


def A(c):
    return {'type': 'assistant', 'uuid': '', 'message': {'role': 'assistant', 'content': c}}


def conv(n_answered, burst):
    c = []
    for i in range(n_answered):
        c += [U([{'type': 'text', 'text': f'{i}'}, IMG]), A([{'type': 'text', 'text': 'ok'}])]
    if burst:
        calls = [{'type': 'tool_use', 'id': f'r{i}', 'name': 'Read', 'input': {}} for i in range(burst)]
        c += [A(calls), U([{'type': 'tool_result', 'tool_use_id': f'r{i}', 'content': [IMG]}
                           for i in range(burst)])]
    return c


def sent(c):
    return sum(p['type'] == 'image_url' for m in convert_to_chat_messages(c, [], max_images=8)
               if m.content_parts for p in m.content_parts)


def test_count_matches_request():
    for answered, burst in [(0, 0), (3, 0), (8, 0), (9, 0), (12, 0), (13, 0), (0, 10), (8, 2), (5, 7)]:
        c = conv(answered, burst)
        got, want = attached_image_count(c, 8), sent(c)
        check(f'count = request ({answered} answered + burst {burst})', got == want, (got, want))


def test_no_vision_no_image_part():
    os.environ['CODEMASON_VISION_MAX_IMAGES'] = '0'
    from codemason.utils.session_status import image_text, status_line
    check('no image text without vision', image_text(conv(3, 0)) == '')
    check('status has no image part', 'images' not in status_line(conv(3, 0)) and '이미지' not in status_line(conv(3, 0)))
    os.environ.pop('CODEMASON_VISION_MAX_IMAGES')


def test_dropped_images_are_said():
    """Four images with one slot to spare and six with two pushed out would both
    read "4/..." against the limit. Against what the conversation holds they read
    4/4 and 4/6, and the words say what the parting means — nothing else on
    screen tells the user, who never sees the label the model is given."""
    import codemason.services.vision as V
    from codemason.utils import i18n
    from codemason.utils.session_status import image_fill, image_text
    saved = V.known_limit
    V.known_limit = lambda: 5
    i18n.set_language('en')         # the text asserted here, not the machine's
    try:
        five, six = conv(5, 0), conv(6, 0)
        check('nothing pushed out reads as it always did',
              image_text(five) == 'images 5/5', image_text(five))
        check('attached and held agree until the limit bites',
              image_fill(five) == (5, 5), image_fill(five))
        check('the line says how many left the request',
              image_text(six) == 'images 4/6 (2 dropped)', image_text(six))
        check('held still counts what the request dropped',
              image_fill(six) == (4, 6), image_fill(six))
        i18n.set_language('ko')
        check('and says it in Korean too',
              image_text(six) == '이미지 4/6 (2장 빠짐)', image_text(six))
    finally:
        V.known_limit = saved
        i18n.reset()


def test_telegram_line():
    from codemason.remote.adapters.telegram import TelegramAdapter
    from codemason.remote.events import Event, EventKind
    t = TelegramAdapter()._format(Event(kind=EventKind.TURN_DONE, title='Turn done', body='hi',
                                        meta={'duration_ms': 3200, 'status': 'Context 22% · images 6/8'}))
    check('status beside duration', t.rstrip().endswith('3.2s · Context 22% · images 6/8'), t)


if __name__ == '__main__':
    for _t in (test_count_matches_request, test_no_vision_no_image_part,
               test_dropped_images_are_said, test_telegram_line):
        try:
            _t()
        except AssertionError:
            pass                    # already printed; run the rest
    sys.exit(0 if all(results) else 1)
