"""Photos from the phone reach the session (remote/poller.py).

Guards:
  - a photo is taken at its largest size, its caption is the text, and the
    session gets the saved path so the model can Read it;
  - an image sent as a file is accepted, other files are not;
  - an album (one update per photo, one media_group_id) is one message, not
    one turn per photo, and the offset moves past the whole group;
  - a failed download is said in the text instead of dropping the photo;
  - a sticker or a join is still not input;
  - a photo downloads on the worker: button taps are answered meanwhile,
    later messages wait behind it (order kept), and the offset does not pass
    it until it is delivered;
  - a photo whose caption starts with /help is still a photo;
  - every caption of an album is kept;
  - download_image: retried, 4xx final except 429, token never in the error.
  - getFile itself is retried on a transport failure and not on a 4xx;
  - an update the offset has already passed is not delivered a second time.

Run: python3 tests/test_remote_photo.py
"""
import atexit
import hashlib
import os
import shutil
import sys
import tempfile

os.environ['CODEMASON_CONFIG_DIR'] = tempfile.mkdtemp(prefix='cm-test-cfg-')
atexit.register(shutil.rmtree, os.environ['CODEMASON_CONFIG_DIR'], True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from codemason.remote import poller as P  # noqa: E402
from codemason.remote.adapters import telegram as _T  # noqa: E402
from pathlib import Path  # noqa: E402

# ~/.codemason/remote is fixed to the home directory, not CODEMASON_CONFIG_DIR:
# point it at the temp dir so no test file lands among real attachments.
_T._REMOTE_DIR = Path(os.environ['CODEMASON_CONFIG_DIR']) / 'remote'

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))
    assert ok, f'{name}: {detail}'   # so pytest fails too, not only the script


class FakeAdapter:
    def __init__(self, fail=False):
        self.fail = fail
        self.downloads = []
        self.offset = None

    def _is_trusted_chat(self, msg, sender=None):
        return True

    def download_image(self, file_id, hint=''):
        self.downloads.append((file_id, hint))
        if self.fail:
            return '', 'HTTP 400 file is too big'
        return f'/att/{file_id}{os.path.splitext(hint)[1] or ".jpg"}', ''

    def _write_offset(self, n):
        self.offset = n

    def _read_offset(self):
        return self.offset or 0


def poller(adapter):
    p = P.Poller(adapter)
    got = []
    p.on_message = lambda text, meta: got.append(text)
    return p, got


PHOTO = [{'file_id': 'small', 'width': 90, 'height': 60},
         {'file_id': 'big', 'width': 1280, 'height': 853},
         {'file_id': 'mid', 'width': 320, 'height': 213}]


def test_single_photo():
    a = FakeAdapter()
    p, got = poller(a)
    p._on_message({'message_id': 1, 'date': 0, 'photo': PHOTO, 'caption': '이거 뭐야?'}, 10)
    check('largest size downloaded', a.downloads == [('big', 'photo.jpg')], a.downloads)
    check('caption + path delivered', got == ['이거 뭐야?\n[Photo from the phone: /att/big.jpg]'], got)


def test_image_document_and_other_files():
    a = FakeAdapter()
    p, got = poller(a)
    p._on_message({'message_id': 2, 'document': {'file_id': 'd1', 'mime_type': 'image/png',
                                                 'file_name': 'shot.png'}}, 11)
    p._on_message({'message_id': 3, 'document': {'file_id': 'd2', 'mime_type': 'application/pdf',
                                                 'file_name': 'a.pdf'}}, 12)
    check('image file accepted, keeps its extension', got == ['[Photo from the phone: /att/d1.png]'], got)
    check('non-image file not downloaded', [d for d, _ in a.downloads] == ['d1'], a.downloads)


def test_album_is_one_message():
    a = FakeAdapter()
    p, got = poller(a)
    ups = [{'update_id': 20 + i, 'message': {'message_id': 30 + i, 'media_group_id': 'g',
                                             'photo': [{'file_id': f'p{i}', 'width': 10, 'height': 10}],
                                             **({'caption': '세 장 비교해줘'} if i == 0 else {})}}
           for i in range(3)]
    ups.append({'update_id': 23, 'message': {'message_id': 33, 'text': 'next'}})
    groups = P._group_albums(ups)
    check('grouped as album + single', [len(g) for g in groups] == [3, 1], [len(g) for g in groups])
    p._dispatch_album(groups[0])
    p._drain_worker()
    check('album delivered once with all paths', len(got) == 1 and got[0].count('[Photo from the phone:') == 3
          and got[0].startswith('세 장 비교해줘\n'), got)
    check('offset past the whole album', a.offset == 23, a.offset)


def test_failed_download_is_said():
    a = FakeAdapter(fail=True)
    p, got = poller(a)
    p._on_message({'message_id': 4, 'photo': PHOTO}, 13)
    check('failure in the text', got and 'could not be downloaded: HTTP 400 file is too big' in got[0], got)


def test_album_captions_all_kept():
    a = FakeAdapter()
    p, got = poller(a)
    p._on_message({'message_id': 1, 'photo': PHOTO, 'caption': '왼쪽'}, 1,
                  extra=[{'message_id': 2, 'photo': PHOTO, 'caption': '오른쪽'}])
    check('both captions', got and got[0].startswith('왼쪽\n오른쪽\n'), got)


def test_photo_caption_is_not_a_bot_command():
    a = FakeAdapter()
    p, got = poller(a)
    p._on_bot_command = lambda text: got.append(('BOT', text))
    p._on_message({'message_id': 1, 'photo': PHOTO, 'caption': '/help 이거 뭐야'}, 1)
    check('photo with /help caption delivered as input',
          len(got) == 1 and isinstance(got[0], str) and '[Photo from the phone:' in got[0], got)


def test_photo_on_worker_keeps_buttons_and_order():
    import threading
    gate = threading.Event()

    class Slow(FakeAdapter):
        def download_image(self, file_id, hint=''):
            gate.wait(10)
            return super().download_image(file_id, hint)

    a = Slow()
    p, got = poller(a)
    taps = []
    p._on_callback = lambda cq: taps.append(cq['id'])
    p._reset_dispatch_state()
    batch = [{'update_id': 40, 'message': {'message_id': 1, 'photo': PHOTO}},
             {'update_id': 41, 'callback_query': {'id': 'tap'}},
             {'update_id': 42, 'message': {'message_id': 2, 'text': '이게 뭐야?'}}]
    for upd in batch:                   # as _run does: _is_new, then dispatch
        if p._is_new(upd):
            p._dispatch(upd)
    check('button tap answered during the download', taps == ['tap'], taps)
    check('nothing delivered yet, text waits behind the photo', got == [], got)
    check('offset not past the photo', (a.offset or 0) <= 40, a.offset)
    check('the next getUpdates hands the batch back; none of it is new',
          not any(p._is_new(u) for u in batch))
    gate.set()
    p._drain_worker()
    check('delivered in the order sent', len(got) == 2 and '[Photo from the phone:' in got[0]
          and got[1] == '이게 뭐야?', got)
    check('offset past everything once delivered', a.offset == 43, a.offset)


def test_stale_work_not_delivered():
    import threading
    gate = threading.Event()

    class Slow(FakeAdapter):
        def download_image(self, file_id, hint=''):
            gate.wait(10)
            return super().download_image(file_id, hint)

    a = Slow()
    p, got = poller(a)
    p._reset_dispatch_state()
    p._dispatch({'update_id': 50, 'message': {'message_id': 1, 'photo': PHOTO}})
    p._reset_dispatch_state()           # the loop run ended (stop / bot change)
    gate.set()
    p._drain_worker()
    check('not delivered after its run ended, offset untouched', got == [] and a.offset is None,
          (got, a.offset))


def test_download_image_retries_and_hides_token():
    import io
    import urllib.error
    from codemason.remote.adapters import telegram as T
    token = '123456:SECRET-token'
    calls = []

    def run(script):
        calls.clear()

        def fake_urlopen(url, timeout=None):
            calls.append(url)
            step = script[len(calls) - 1]
            if isinstance(step, int):
                raise urllib.error.HTTPError(url, step, 'x', {}, io.BytesIO(b''))
            if isinstance(step, Exception):
                raise step
            return io.BytesIO(step)

        saved = (T._credentials, T.urllib.request.urlopen, T.time.sleep)
        T._credentials = lambda: (token, '1')
        T.urllib.request.urlopen = fake_urlopen
        T.time.sleep = lambda s: None
        ad = T.TelegramAdapter()
        ad._post = lambda method, payload, **k: {'ok': True, 'result': {'file_path': 'photos/f 1.jpg'}}
        try:
            return ad.download_image('fid', 'photo.jpg')
        finally:
            T._credentials, T.urllib.request.urlopen, T.time.sleep = saved

    path, err = run([ConnectionResetError('reset'), b'\xff\xd8jpegbytes'])
    check('retried after a transport error', path and not err and len(calls) == 2, (path, err, calls))
    check('saved by content hash as .jpg', path.endswith('.jpg') and os.path.exists(path), path)
    check('named by a short content hash, not all 64 digits',
          len(Path(path).stem) == T.ATTACHMENT_NAME_DIGITS
          and Path(path).stem == hashlib.sha256(b'\xff\xd8jpegbytes').hexdigest()[:T.ATTACHMENT_NAME_DIGITS],
          Path(path).name)
    again, _err = run([b'\xff\xd8jpegbytes'])
    check('the same photo twice is still one file', again == path, (again, path))

    # A short name another photo owns is not reused for these bytes: the full
    # hash cannot clash, so this photo wears it instead.
    clash = Path(path).with_name(Path(path).stem + '.jpg')
    clash.write_bytes(b'different bytes entirely')
    other, _err = run([b'\xff\xd8jpegbytes'])
    check('a short name owned by other bytes falls back to the full hash',
          Path(other).stem == hashlib.sha256(b'\xff\xd8jpegbytes').hexdigest()
          and Path(other).read_bytes() == b'\xff\xd8jpegbytes', other)
    clash.unlink(); Path(other).unlink()
    path, err = run([404, b'never'])
    check('4xx is final', not path and err == 'HTTP 404' and len(calls) == 1, (err, calls))
    path, err = run([429, 429, b'\xff\xd8ok'])
    check('429 is retried', path and len(calls) == 3, (path, err, calls))
    path, err = run([ValueError(f'bad url https://api.telegram.org/file/bot{token}/x')] * 3)
    check('token never in the error', not path and token not in err and '<token>' in err, err)


def test_attachment_ext_and_prune():
    import time as _time
    from codemason.remote.adapters import telegram as T
    check('sender file name extension kept when it is an image', T._image_ext('a.PNG', 'x') == '.png')
    check('odd extension not trusted', T._image_ext('evil.sh', 'photos/f.webp') == '.webp'
          and T._image_ext('x.exe;rm', '') == '.jpg')
    folder = T.attachments_dir()
    folder.mkdir(parents=True, exist_ok=True)
    old, new = folder / 'old.jpg', folder / 'new.jpg'
    old.write_bytes(b'1')
    new.write_bytes(b'2')
    past = _time.time() - 40 * 86400
    os.utime(old, (past, past))
    removed = T.prune_attachments()
    check('only attachments past the age are pruned',
          removed == 1 and not old.exists() and new.exists(), removed)


def test_sticker_not_input():
    a = FakeAdapter()
    p, got = poller(a)
    p._on_message({'message_id': 5, 'sticker': {'file_id': 's'}}, 14)
    check('sticker ignored', got == [] and a.downloads == [], (got, a.downloads))


def test_an_update_the_offset_has_passed_is_not_new_again():
    """The fetch in flight when the offset moves answers with the update
    again — the photo must not be delivered a second time."""
    a = FakeAdapter()
    p, _got = poller(a)
    upd = lambda i: {'update_id': i}
    check('a fresh update is new', p._is_new(upd(286)))
    with p._work_lock:
        p._inflight.add(286)            # photo handed to the worker
    check('handed back while it downloads', not p._is_new(upd(286)))
    p._commit(286)                      # worker done, offset moves
    check('the offset moved past it', a.offset == 287, a.offset)
    check('the long poll that went out with offset 286 still answers with it',
          not p._is_new(upd(286)))
    check('and the next update is still new', p._is_new(upd(287)))


def test_getfile_retries_a_transport_failure_not_a_4xx():
    """A read-only call goes out again after a transport failure, and stops at
    a 4xx: Telegram answered, and the answer will not change."""
    from codemason.remote.adapters import telegram as T
    saved = T.time.sleep
    T.time.sleep = lambda s: None
    try:
        def script(answers):
            calls = []

            def call():
                calls.append(1)
                return answers[len(calls) - 1] if len(calls) <= len(answers) else None

            return T._retry_read(call), len(calls)

        dead = {'ok': False, 'error': 'error sending request: connection reset'}
        ok = {'ok': True, 'result': {'file_path': 'photos/f.jpg'}}
        got, n = script([dead, ok])
        check('a transport failure goes out again', got == ok and n == 2, (got, n))
        gone = {'ok': False, 'error': 'HTTP 400 Bad Request: file not found'}
        got, n = script([gone, ok])
        check('a 4xx is the answer, not a reason to ask again',
              got == gone and n == 1, (got, n))
        _unused, n = script([dead, dead, dead])
        check('and it does not try forever', n == T._DOWNLOAD_ATTEMPTS, n)
    finally:
        T.time.sleep = saved


if __name__ == '__main__':
    for _t in (test_single_photo, test_image_document_and_other_files, test_album_is_one_message,
               test_failed_download_is_said, test_album_captions_all_kept,
               test_photo_caption_is_not_a_bot_command, test_photo_on_worker_keeps_buttons_and_order,
               test_stale_work_not_delivered, test_download_image_retries_and_hides_token,
               test_attachment_ext_and_prune, test_sticker_not_input,
               test_an_update_the_offset_has_passed_is_not_new_again,
               test_getfile_retries_a_transport_failure_not_a_4xx):
        try:
            _t()
        except AssertionError:
            pass                    # already printed; run the rest
    sys.exit(0 if all(results) else 1)
