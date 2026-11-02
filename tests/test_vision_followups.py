"""Fixes from the vision audit (2026-10-31), one guard each.

  - the server probe: "is not a multimodal model" is 0, any other 400 is
    "cannot tell", an inconclusive probe is not repeated within 30 s, and the
    body is sealed when model_aes_key is set;
  - a system prompt built while the server could not be asked gets its
    "# Images" section on the next turn; the sub-agent prefix follows;
  - language detection ignores the photo label and file paths;
  - a resumed sub-agent gets its images back, not store references;
  - Read: at most half the limit in tiles, a size cap, the original kept only
    for an image that was attached, and as a hard link when it is ours;
  - image store: stored_path worked out once (failure too), a reference that
    walks out of the store is refused, deleting a session deletes its folder;
  - convert: an API-error or interrupt message does not mark images seen, an
    image beside tool results is sent, the image count is the request's;
  - microcompact keeps image blocks and does not count them as freed;
  - the summary strip numbers images the way convert does;
  - the context/image status survives the PUBLIC redaction; one percent rule.

Run: python3 tests/test_vision_followups.py
"""
import asyncio
import atexit
import base64
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['CODEMASON_CONFIG_DIR'] = tempfile.mkdtemp(prefix='cm-test-cfg-')
atexit.register(shutil.rmtree, os.environ['CODEMASON_CONFIG_DIR'], True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from PIL import Image  # noqa: E402

from codemason.services import vision as V  # noqa: E402
from codemason.services._openai_common import (  # noqa: E402
    attached_image_count, convert_to_chat_messages)
from codemason.utils import image_store as S  # noqa: E402
from codemason.utils.state import set_session_id  # noqa: E402

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))
    assert ok, f'{name}: {detail}'   # so pytest fails too, not only the script


def png_bytes(w, h, color=(90, 90, 90)):
    b = io.BytesIO()
    Image.new('RGB', (w, h), color).save(b, 'PNG')
    return b.getvalue()


def img_block(w=8, h=8, color=(90, 90, 90)):
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                        'data': base64.b64encode(png_bytes(w, h, color)).decode()}}


def U(c):
    return {'type': 'user', 'message': {'role': 'user', 'content': c}}


def A(c, model='glm'):
    return {'type': 'assistant', 'uuid': '', 'message': {'role': 'assistant', 'content': c,
                                                          'model': model}}


# --- probe ---------------------------------------------------------------

class _Resp:
    def __init__(self, code, text=''):
        self.status_code, self.text = code, text


def _with_probe(answer, cfg):
    """Run image_limit() with httpx.post answering `answer`; returns
    (limit, calls)."""
    import httpx
    calls = []

    def fake_post(url, **kw):
        calls.append(kw)
        if isinstance(answer, Exception):
            raise answer
        return answer

    saved = httpx.post
    httpx.post = fake_post
    try:
        return V.image_limit(cfg), calls
    finally:
        httpx.post = saved


def _fresh_probe_state():
    V._limit = None
    V._last_inconclusive = 0.0


def test_probe():
    cfg = SimpleNamespace(model_api_key='', model_aes_key='', model_name='m',
                          model_base_url='http://x')
    os.environ.pop('CODEMASON_VISION_MAX_IMAGES', None)
    _fresh_probe_state()
    n, _ = _with_probe(_Resp(400, '{"message":"At most 8 image(s) may be provided in one prompt."}'), cfg)
    check('limit read from the count error', n == 8 and V._limit == 8, n)
    _fresh_probe_state()
    n, _ = _with_probe(_Resp(400, 'Qwen/x is not a multimodal model'), cfg)
    check('text-only model is 0, cached', n == 0 and V._limit == 0, (n, V._limit))
    _fresh_probe_state()
    n, calls = _with_probe(_Resp(400, 'height:1 must be larger than factor:28'), cfg)
    check('another 400 is not a limit', n == 0 and V._limit is None, (n, V._limit))
    n, calls = _with_probe(_Resp(400, 'At most 8 image(s)'), cfg)
    check('inconclusive probe not repeated within the wait', n == 0 and calls == [], calls)
    V._last_inconclusive = time.monotonic() - V._RETRY_AFTER - 1
    n, calls = _with_probe(_Resp(400, 'At most 8 image(s)'), cfg)
    check('asked again after the wait', n == 8 and len(calls) == 1, (n, calls))
    _fresh_probe_state()
    key = base64.b64encode(os.urandom(32)).decode()
    sealed = SimpleNamespace(model_api_key='k', model_aes_key=key, model_name='m',
                             model_base_url='http://x')
    n, calls = _with_probe(_Resp(400, 'At most 4 image(s)'), sealed)
    kw = calls[0]
    check('body sealed when model_aes_key is set',
          n == 4 and 'json' not in kw and kw['headers'].get('X-Body-Encrypted') == 'aesgcm'
          and b'image_url' not in kw['content'], kw.get('headers'))
    _fresh_probe_state()


# --- system prompt refresh ---------------------------------------------------

def test_prompt_refresh():
    from codemason.constants import prompts as P
    built = []

    async def fake_build(tools=None, model=None):
        built.append(1)
        return ['intro', P.get_vision_section(V.known_limit()) or '', 'env']

    saved = (P.get_system_prompt, V.image_limit)
    from codemason.services import backend as B
    saved_backend = B.get_backend
    P.get_system_prompt = fake_build
    B.get_backend = lambda: 'vllm'
    try:
        V.image_limit = lambda cfg=None: 0               # server down at startup
        sp = ['intro', 'env']
        changed = asyncio.run(P.refresh_system_prompt(sp))
        check('nothing to add while the server cannot be asked', not changed and sp == ['intro', 'env'], sp)
        V.image_limit = lambda cfg=None: 8               # it answers
        changed = asyncio.run(P.refresh_system_prompt(sp))
        check('section added on the next turn', changed and any(
            s.startswith('# Images\n') and 'At most 8 images' in s for s in sp), sp)
        n = len(built)
        changed = asyncio.run(P.refresh_system_prompt(sp))
        check('then left alone', not changed and len(built) == n)
        P._agent_prefix_sections = None
        V._last_enabled = 0
        before = P.agent_prefix_sections('m')
        V._last_enabled = 8
        after = P.agent_prefix_sections('m')
        check('sub-agent prefix follows the limit',
              not any(s.startswith('# Images') for s in before)
              and any(s.startswith('# Images') for s in after), (len(before), len(after)))
    finally:
        P.get_system_prompt, V.image_limit = saved[0], saved[1]
        B.get_backend = saved_backend
        V._last_enabled = 0
        P._agent_prefix_sections = None


# The Rust port pins the same text (constants/prompts.rs vision_section_tests),
# as a JSON string.
PY_SECTION_8 = '"# Images\\n\\nYou can see images. They reach you from the user\'s messages and from the Read tool on an image file, each labelled [Image N] in order of arrival. A very tall or wide image is read as several tiles; each tile is one image.\\n - At most 8 images are attached to one request. Past that the oldest are replaced by a line \\"[Image K: …; saved at PATH]\\" that says whether you saw it earlier. Such an image is not in front of you: if you saw it, work from what you noted about it then; if you need to look again, or never saw it, Read the saved PATH — that attaches it as a new image and pushes an older one out.\\n - So read at most 8 images at a time, and while they are attached write what you need from each into your reply text — not only into your thinking, which is dropped when the turn ends — instead of reading everything first and describing it afterwards.\\n - If a task needs more than 8 images in view at once, say so rather than working from images that are no longer attached."'


def test_section_matches_rust_pin():
    import json
    from codemason.constants.prompts import get_vision_section
    check('section text is the one the Rust port pins',
          get_vision_section(8) == json.loads(PY_SECTION_8))


# --- language ------------------------------------------------------------

def test_language_ignores_label_and_paths():
    from codemason.utils.language import detect_language
    label = '[Photo from the phone: /home/u/.codemason/remote/attachments/' + 'ab' * 32 + '.jpg]'
    check('short Korean caption with a photo', detect_language('이 사진 설명해줘\n' + label) == 'Korean')
    check('album of three', detect_language('뭐야\n' + '\n'.join([label] * 3)) == 'Korean')
    check('Korean around a long path',
          detect_language('~/work/codemason_v2.0/src/services/openai_common.rs 이거 봐') == 'Korean')
    check('English stays silent', detect_language('look at /tmp/a.png please') is None)


# --- sub-agent resume ----------------------------------------------------------

def test_agent_transcript_hydrated():
    from codemason.tasks import local_agent_task as L
    set_session_id('sess-agent')
    blk = img_block(12, 12)
    msg = {'type': 'user', 'uuid': 'u1', 'message': {'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': 't', 'content': [blk]}]}}
    import uuid
    from codemason.utils.task.disk_output import get_task_output_path
    agent = f'agent-{uuid.uuid4().hex}'     # the task folder is shared by every run
    try:
        L.append_agent_transcript(agent, msg)
        back = L.get_agent_transcript(agent)
    finally:
        L.forget_agent_transcript(agent)
        Path(get_task_output_path(agent)).unlink(missing_ok=True)
    src = back[0]['message']['content'][0]['content'][0]['source']
    check('resumed agent sees the image again',
          src.get('type') == 'base64' and src.get('data') == blk['source']['data'], src.get('type'))


# --- Read ----------------------------------------------------------------

async def _read(path):
    from codemason.tools.file_read_tool import FileReadTool
    ctx = SimpleNamespace(read_file_state={}, options={})
    out = [r async for r in FileReadTool.call({'file_path': str(path)}, ctx)]
    return out[-1]


def test_read_limits():
    from codemason.services import backend as B
    from codemason.tools import file_read_tool as R
    set_session_id('sess-read')
    saved = (B.get_backend, V.image_limit, R.MAX_IMAGE_FILE_BYTES)
    B.get_backend = lambda: 'vllm'
    V.image_limit = lambda cfg=None: 8
    tmp = Path(tempfile.mkdtemp(prefix='cm-read-'))
    try:
        tall = tmp / 'tall.png'
        tall.write_bytes(png_bytes(1080, 30000))
        res = asyncio.run(_read(tall))
        imgs = [b for b in res.result_for_assistant if b.get('type') == 'image']
        check('one Read takes at most half the limit', len(imgs) == 4, len(imgs))
        R.MAX_IMAGE_FILE_BYTES = 10
        res = asyncio.run(_read(tall))
        check('over the size cap: said, not attached',
              isinstance(res.result_for_assistant, str) and 'larger than' in res.result_for_assistant,
              res.result_for_assistant)
        R.MAX_IMAGE_FILE_BYTES = saved[2]
        bad = tmp / 'bad.png'
        bad.write_bytes(b'not an image at all')
        before = set(S.images_dir().glob('orig-*')) if S.images_dir().exists() else set()
        res = asyncio.run(_read(bad))
        after = set(S.images_dir().glob('orig-*')) if S.images_dir().exists() else set()
        check('undecodable: said, and no original kept',
              'could not be attached' in res.result_for_assistant and after == before,
              res.result_for_assistant)
        from codemason.utils.config import CONFIG_DIR
        own = Path(CONFIG_DIR) / 'remote' / 'attachments' / 'p.png'
        own.parent.mkdir(parents=True, exist_ok=True)
        own.write_bytes(png_bytes(30, 20))
        res = asyncio.run(_read(own))
        orig = Path(res.data['original_copy'])
        check('original of our own attachment is a hard link',
              orig.exists() and orig.stat().st_ino == own.stat().st_ino)
        outside = tmp / 'q.png'
        outside.write_bytes(png_bytes(31, 20))
        res = asyncio.run(_read(outside))
        orig = Path(res.data['original_copy'])
        check('original of a user file is a copy', orig.exists() and orig.stat().st_ino != outside.stat().st_ino)
    finally:
        B.get_backend, V.image_limit, R.MAX_IMAGE_FILE_BYTES = saved
        shutil.rmtree(tmp, ignore_errors=True)


# --- image store -------------------------------------------------------------

def test_store_cache_and_refs():
    set_session_id('sess-store')
    blk = img_block(9, 9)
    calls = []
    real = S._dehydrate_source

    def counting(src, sid):
        calls.append(1)
        return real(src, sid)

    S._dehydrate_source = counting
    try:
        S._STORED.clear()
        p1 = S.stored_path(blk['source'])
        p2 = S.stored_path(blk['source'])
        check('worked out once', p1 == p2 and p1 is not None and len(calls) == 1, calls)
        S._STORED.clear()
        S._dehydrate_source = lambda src, sid: src         # cannot store
        other = img_block(10, 10)
        first = S.stored_path(other['source'])
        S._dehydrate_source = counting
        again = S.stored_path(other['source'])
        check('a failure is remembered (label does not change later)', first is None and again is None)
    finally:
        S._dehydrate_source = real
    bad = {'type': 'image', 'source': {'type': 'codemason_file', 'path': '../../../etc/passwd',
                                       'media_type': 'image/png'}}
    out = S.hydrate(bad, 'sess-store')
    check('reference out of the store refused', out['type'] == 'text' and 'not valid' in out['text'], out)


def test_delete_session_takes_its_folder():
    from codemason.utils.session_storage import delete_session
    d = Path(tempfile.mkdtemp(prefix='cm-sess-'))
    try:
        t = d / 'abc.jsonl'
        t.write_text('{}\n')
        (d / 'abc' / 'images').mkdir(parents=True)
        (d / 'abc' / 'images' / 'x.png').write_bytes(b'1')
        (d / 'other').mkdir()
        check('transcript and folder gone, neighbours kept',
              delete_session(t) and not t.exists() and not (d / 'abc').exists() and (d / 'other').exists())
        check('missing transcript reports False', not delete_session(t))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- convert -----------------------------------------------------------------

def _labels(chat):
    return [p['text'] for m in chat for p in (m.content_parts or [])
            if p['type'] == 'text' and p['text'].startswith('[Image')]


def test_synthetic_answer_not_seen():
    set_session_id('sess-conv')
    calls = [{'type': 'tool_use', 'id': f'r{i}', 'name': 'Read', 'input': {}} for i in range(10)]
    burst = [{'type': 'tool_result', 'tool_use_id': f'r{i}', 'content': [img_block(8, 8, (i, 0, 0))]}
             for i in range(10)]
    conv = [U('read all'), A(calls), U(burst),
            A([{'type': 'text', 'text': 'Error connecting to the model server'}], model='<synthetic>'),
            A([{'type': 'tool_use', 'id': 'x', 'name': 'Read', 'input': {}}]),
            U([{'type': 'tool_result', 'tool_use_id': 'x', 'content': [img_block(8, 8, (0, 9, 0))]}])]
    labels = _labels(convert_to_chat_messages(conv, [], max_images=8))
    # The tool_use message after the error is a model answer, so the burst was
    # seen there; the API error alone must not have marked anything.
    conv_err_only = conv[:4] + [U([{'type': 'text', 'text': 'again'}, img_block(8, 8, (0, 0, 9))])]
    labels2 = _labels(convert_to_chat_messages(conv_err_only, [], max_images=8))
    check('API error does not count as having seen the images',
          not any('you saw it earlier' in t for t in labels2), labels2)
    check('a real answer still does', any('you saw it earlier' in t for t in labels), labels)


def test_image_beside_results_sent():
    conv = [U('x'), A([{'type': 'tool_use', 'id': 't', 'name': 'Bash', 'input': {}}]),
            U([{'type': 'tool_result', 'tool_use_id': 't', 'content': 'ok'},
               {'type': 'text', 'text': 'and this'}, img_block()])]
    chat = convert_to_chat_messages(conv, [], max_images=8)
    last = chat[-1]
    check('image beside tool results goes out', last.role == 'user' and last.content_parts
          and any(p['type'] == 'image_url' for p in last.content_parts), last.to_dict())
    check('and is counted', attached_image_count(conv, 8) == 1)


def test_count_skips_what_convert_skips():
    blk = img_block()
    conv = [U('x'), A([{'type': 'tool_use', 'id': 't', 'name': 'Read', 'input': {}}]),
            U([{'type': 'tool_result', 'tool_use_id': 't', 'content': [blk]}]),
            U([{'type': 'tool_result', 'tool_use_id': 't', 'content': [blk]}])]   # duplicate
    check('duplicate result counted once', attached_image_count(conv, 8) == 1,
          attached_image_count(conv, 8))


# --- compaction ----------------------------------------------------------------

def test_microcompact_keeps_images():
    from codemason.services import compaction as C
    msgs = []
    n = C.MICROCOMPACT_KEEP_RECENT + 2
    for i in range(n):
        msgs.append(A([{'type': 'tool_use', 'id': f'r{i}', 'name': 'Read', 'input': {}}]))
        msgs.append(U([{'type': 'tool_result', 'tool_use_id': f'r{i}', 'content': [
            {'type': 'text', 'text': 'x' * 4000}, img_block(8, 8, (i, i, i))]}]))
    before = [p['text'] for p in _all_label_parts(msgs)]
    out, freed = C.microcompact_messages(msgs)
    after = [p['text'] for p in _all_label_parts(out)]
    first = out[1]['message']['content'][0]['content']
    check('cleared result keeps its image', isinstance(first, list)
          and first[0]['text'] == C.TIME_BASED_MC_CLEARED_MESSAGE and first[1]['type'] == 'image', first)
    check('image numbers unchanged', before == after, (before, after))
    check('only text counted as freed', 0 < freed < 2 * 4000, freed)
    again, freed2 = C.microcompact_messages(out)
    check('idempotent', freed2 == 0 and again is out)


def _all_label_parts(msgs):
    return [p for m in convert_to_chat_messages(msgs, [], max_images=64)
            for p in (m.content_parts or []) if p['type'] == 'text' and p['text'].startswith('[Image')]


def test_strip_numbers_like_convert():
    from codemason.services.compaction import strip_images_for_summary
    blk = img_block()
    conv = [U('x'), A([{'type': 'tool_use', 'id': 't', 'name': 'Read', 'input': {}}]),
            U([img_block(7, 7), {'type': 'tool_result', 'tool_use_id': 't', 'content': [blk]}]),
            U([{'type': 'tool_result', 'tool_use_id': 't', 'content': [blk]}]),       # duplicate
            U([img_block(6, 6)])]
    out = strip_images_for_summary(conv)
    first = out[2]['message']['content']
    check('result image before the one beside it',
          first[1]['content'][0]['text'].startswith('[Image 1') and first[0]['text'].startswith('[Image 2'),
          first)
    dup = out[3]['message']['content'][0]['content'][0]['text']
    check('duplicate result unnumbered', dup.startswith('[Image:') or dup == '[Image]', dup)
    check('next one is 3', out[4]['message']['content'][0]['text'].startswith('[Image 3'))


# --- status ----------------------------------------------------------------

def test_status_rules():
    from codemason.remote.events import Event, EventKind, Sensitivity
    from codemason.utils.session_status import fill_percent
    check('half up, not half even', fill_percent(125, 1000) == 13 and fill_percent(5, 1000) == 1)
    check('capped at 100', fill_percent(1500, 1000) == 100)
    ev = Event(kind=list(EventKind)[0], title='t', body='b', sensitivity=Sensitivity.SECRET,
               meta={'duration_ms': 5, 'status': 'Context 9%', 'path': '/secret'})
    red = ev.redacted(Sensitivity.PUBLIC)
    check('status survives PUBLIC, other meta does not',
          red.meta.get('status') == 'Context 9%' and 'path' not in red.meta, red.meta)


if __name__ == '__main__':
    for _t in (test_probe, test_prompt_refresh, test_section_matches_rust_pin,
               test_language_ignores_label_and_paths,
               test_agent_transcript_hydrated, test_read_limits, test_store_cache_and_refs,
               test_delete_session_takes_its_folder, test_synthetic_answer_not_seen,
               test_image_beside_results_sent, test_count_skips_what_convert_skips,
               test_microcompact_keeps_images, test_strip_numbers_like_convert, test_status_rules):
        try:
            _t()
        except AssertionError:
            pass                    # already printed; run the rest
    sys.exit(0 if all(results) else 1)
