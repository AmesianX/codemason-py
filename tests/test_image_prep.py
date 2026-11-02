"""utils/image_prep.py — the client half of the server's image geometry.

Runs under pytest and as a script. Token counts are the prompt-token deltas
measured on the live glm53 vision server; the prepared size must keep them.
"""
import io
import os
import sys
import tempfile

import atexit  # noqa: E402
import shutil  # noqa: E402
os.environ['CODEMASON_CONFIG_DIR'] = tempfile.mkdtemp(prefix='cm-test-cfg-')
atexit.register(shutil.rmtree, os.environ['CODEMASON_CONFIG_DIR'], True)  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from PIL import Image  # noqa: E402
from codemason.utils.image_prep import (  # noqa: E402
    BUDGET_PIXELS, CELL, fit_size, image_tokens, prepare_image)


def padded(w, h):
    return -(-w // CELL) * CELL * (-(-h // CELL) * CELL)

MEASURED = {(1792, 1792): 4096, (896, 896): 1024, (4000, 3000): 4070, (8000, 6000): 4070,
            (6000, 4000): 4056, (1920, 1080): 2691, (3840, 2160): 4080}


def png(w, h):
    b = io.BytesIO()
    Image.new('RGB', (w, h), (200, 200, 200)).save(b, 'PNG')
    return b.getvalue()


def test_tokens_match_server():
    for (w, h), t in MEASURED.items():
        assert image_tokens(w, h) == t, (w, h, image_tokens(w, h), t)


def test_fit_keeps_tokens():
    for w, h in ((3840, 2160), (4000, 3000), (6000, 4000), (8000, 6000)):
        fw, fh = fit_size(w, h)
        assert fw * fh <= BUDGET_PIXELS and image_tokens(fw, fh) == image_tokens(w, h), (w, h, fw, fh)
    assert fit_size(3840, 2160) == (2380, 1338)
    assert fit_size(4000, 3000) == (2053, 1540)


def test_within_budget_untouched():
    data = png(2560, 1080)  # wider than 1792 but inside the area budget
    out = prepare_image(data, 8)
    # Same bytes: a PNG without metadata chunks passes through untouched.
    assert len(out) == 1 and out[0].data == data and (out[0].width, out[0].height) == (2560, 1080)


def test_padded_canvas_decides():
    # Its own area fits (3,200,000) but the 28-aligned canvas (2016x1624) does
    # not, and the server measures the canvas: shrunk here, once, to what the
    # server would pick, so it only pads.
    out = prepare_image(png(2000, 1600), 8)
    assert len(out) == 1 and (out[0].width, out[0].height) == fit_size(2000, 1600)
    assert padded(out[0].width, out[0].height) <= BUDGET_PIXELS
    assert out[0].tokens == image_tokens(2000, 1600)


def jpeg_with_exif(w, h, exif_pad=0):
    img = Image.new('RGB', (w, h), (10, 120, 200))
    exif = Image.Exif()
    exif[0x010F] = 'CameraMaker'              # Make
    exif[0x0112] = 1                          # orientation: none
    exif[0x8825] = {2: (37.0, 33.0, 0.0)}     # GPS latitude
    if exif_pad:
        exif[0x9286] = 'x' * exif_pad         # UserComment: a big APP1
    b = io.BytesIO()
    img.save(b, 'JPEG', quality=90, exif=exif.tobytes())
    return b.getvalue()


def test_metadata_stripped_losslessly():
    data = jpeg_with_exif(640, 480)
    assert b'Exif' in data and b'CameraMaker' in data
    out = prepare_image(data, 8)[0]
    assert b'Exif' not in out.data and b'CameraMaker' not in out.data
    # Not re-encoded: the scan data is the original's, byte for byte.
    assert out.data[out.data.index(b'\xff\xda'):] == data[data.index(b'\xff\xda'):]
    b = io.BytesIO()
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text('Comment', 'secret place')
    Image.new('RGB', (50, 40)).save(b, 'PNG', pnginfo=info)
    out = prepare_image(b.getvalue(), 8)[0]
    assert b'secret place' not in out.data and Image.open(io.BytesIO(out.data)).size == (50, 40)


def test_animated_first_frame():
    b = io.BytesIO()
    frames = [Image.new('RGB', (40, 30), c) for c in ((255, 0, 0), (0, 255, 0))]
    frames[0].save(b, 'GIF', save_all=True, append_images=frames[1:])
    out = prepare_image(b.getvalue(), 8)
    assert len(out) == 1 and out[0].animated and out[0].media_type == 'image/png'
    assert not prepare_image(png(40, 30), 8)[0].animated


def test_never_enlarged():
    out = prepare_image(png(300, 200), 8)
    assert (out[0].width, out[0].height) == (300, 200)


def test_long_capture_tiles():
    out = prepare_image(png(1080, 20000), 8)
    assert len(out) == 7 and all(p.width == 1080 for p in out), [(p.width, p.height) for p in out]
    assert all(padded(p.width, p.height) <= BUDGET_PIXELS for p in out)
    # A short side that is not a multiple of 28 still gives tiles whose padded
    # canvas fits.
    odd = prepare_image(png(1000, 30000), 8)
    assert all(padded(p.width, p.height) <= BUDGET_PIXELS for p in odd), \
        [(p.width, p.height) for p in odd]
    few = prepare_image(png(1080, 20000), 3)
    assert len(few) == 3, len(few)


def test_block_tokens_from_header():
    import base64
    from codemason.utils.image_prep import image_block_tokens
    from codemason.utils.tokens import _rough_block
    blk = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                       'data': base64.b64encode(png(1920, 1080)).decode()}}
    assert image_block_tokens(blk) == 2691 and _rough_block(blk) == 2691
    assert image_block_tokens({'type': 'image', 'source': {'data': 'AAAA'}}) == 4096
    # SOF behind an EXIF bigger than the header window: still the real size.
    small = jpeg_with_exif(640, 480)
    xmp = b'http://ns.adobe.com/xap/1.0/\0' + b' ' * 60000   # APP1 XMP, 3 of them
    seg = b'\xff\xe1' + (len(xmp) + 2).to_bytes(2, 'big') + xmp
    big = small[:2] + seg * 3 + small[2:]
    assert big.index(b'\xff\xc0') > 131072 * 3 // 4             # SOF past the window
    blk = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                       'data': base64.b64encode(big).decode()}}
    assert image_block_tokens(blk) == image_tokens(640, 480)


if __name__ == '__main__':
    tests = [test_tokens_match_server, test_fit_keeps_tokens, test_within_budget_untouched,
             test_padded_canvas_decides, test_metadata_stripped_losslessly,
             test_animated_first_frame, test_never_enlarged, test_long_capture_tiles,
             test_block_tokens_from_header]
    failed = 0
    for t in tests:
        try:
            t()
            print('PASS', t.__name__)
        except AssertionError as e:
            failed += 1
            print('FAIL', t.__name__, str(e)[:200])
    sys.exit(1 if failed else 0)
