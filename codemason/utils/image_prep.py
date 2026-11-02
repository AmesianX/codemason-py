"""
Prepare an image for the model before it is sent — the client half of the
server's own geometry, so the server only pads and never resamples again.

The server (GLM-5.3-Flash processor, pad mode) gives one image a budget of
VISION_MAX_TOKENS tokens; one token covers a 28x28 pixel cell, so the budget is
an AREA: 4096 tokens = 3,211,264 px = 1792^2, in any aspect ratio. Inside the
budget the server does not resample at all — it pads right/bottom up to the next
multiple of 28. Over the budget it shrinks the image proportionally to the
largest size whose 28-aligned canvas fits.

So, per image ("within budget" is the 28-aligned canvas, the area the server
itself measures — an image whose own area fits but whose padded canvas does not
is shrunk by the server):
  - EXIF orientation is applied to the pixels first (and the tag dropped), so the
    size we reason about is the size the model sees.
  - Within budget: sent as-is. Never enlarged — the server's own rule enlarges
    only images under its minimum (16 tokens, about 112x112). The original
    pixels when no rotation was needed: a PNG or JPEG is passed through with
    only its metadata removed (EXIF with GPS and camera, XMP, text chunks),
    losslessly; anything else is re-encoded.
  - Area over budget: shrunk ONCE, with Lanczos, to exactly the content size the
    server would pick (_fit_aligned_size_within_budget + _get_pad_content_size
    ported below). The server then pads and does not resample again.
  - Extreme aspect ratio (a whole-page capture): shrinking to the area budget
    would leave the short side under MIN_READABLE_SHORT_SIDE and the text
    unreadable, so the image is cut along its long side into tiles that each fit
    the budget at readable scale — at most `slots_left` tiles.
  - An animated GIF/WebP is sent as its first frame (`animated` says so).

Each result carries its token cost (same formula as the server), which the
context bar and compaction use.
"""

import io
import math
from dataclasses import dataclass
from typing import List

# Server constants (glm53 lane: VISION_MAX_TOKENS=4096, patch 14 x merge 2).
CELL = 28                          # pixels per token side
MAX_TOKENS_PER_IMAGE = 4096
MIN_TOKENS_PER_IMAGE = 16
BUDGET_PIXELS = MAX_TOKENS_PER_IMAGE * CELL * CELL      # 3,211,264 = 1792^2
MIN_READABLE_SHORT_SIDE = 768
MAX_TILE_SHORT_SIDE = 1792


@dataclass
class PreparedImage:
    data: bytes
    media_type: str
    width: int
    height: int
    tokens: int
    animated: bool = False   # the source had more frames; this is the first


def _ceil_to(v: int, f: int) -> int:
    return math.ceil(v / f) * f


def image_tokens(width: int, height: int) -> int:
    """Tokens the server spends on an image of this size (pad mode)."""
    cw, ch = _ceil_to(width, CELL), _ceil_to(height, CELL)
    if cw * ch > BUDGET_PIXELS:
        ch, cw = _fit_canvas(height, width)
    tokens = (cw // CELL) * (ch // CELL)
    return max(MIN_TOKENS_PER_IMAGE, tokens)


def _fit_canvas(h: int, w: int):
    """Server _fit_aligned_size_within_budget for a still image (t factored out:
    t_bar * h * w <= max_pixels with t_bar = 2 and max_pixels = 2 * BUDGET)."""
    low, high = 1, h
    best = (CELL, CELL)
    while low <= high:
        ch = (low + high) // 2
        cw = max(1, math.floor(w * ch / h))
        ah, aw = _ceil_to(ch, CELL), _ceil_to(cw, CELL)
        if ah * aw <= BUDGET_PIXELS:
            best = (ah, aw)
            low = ch + 1
        else:
            high = ch - 1
    return best


def fit_size(width: int, height: int):
    """Content size the server would shrink an over-budget image to."""
    canvas_h, canvas_w = _fit_canvas(height, width)
    scale = min(1.0, canvas_h / height, canvas_w / width)
    return (max(1, min(canvas_w, math.floor(width * scale))),
            max(1, min(canvas_h, math.floor(height * scale))))


def _padded_area(width: int, height: int) -> int:
    return _ceil_to(width, CELL) * _ceil_to(height, CELL)


# JPEG segments kept when the metadata goes: APP0 JFIF/JFXX (pixel density),
# APP2 ICC_PROFILE (colours), APP14 Adobe (colour transform). Every other APPn
# and COM is metadata — APP1 alone carries EXIF (camera, GPS) and XMP.
_JPEG_KEEP_APP = {0xE0: (b'JFIF\0', b'JFXX\0'), 0xE2: (b'ICC_PROFILE\0',), 0xEE: (b'Adobe',)}
# PNG chunks that are metadata, not pixels or colour.
_PNG_DROP = {b'eXIf', b'tEXt', b'zTXt', b'iTXt', b'tIME'}


def _strip_jpeg(data: bytes) -> bytes:
    if data[:2] != b'\xff\xd8':
        raise ValueError('not a JPEG')
    out = [data[:2]]
    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ValueError('bad JPEG marker')
        while pos + 1 < len(data) and data[pos + 1] == 0xFF:
            pos += 1                                    # fill bytes
        marker = data[pos + 1]
        if marker == 0xDA:                              # start of scan: the rest is pixels
            out.append(data[pos:])
            return b''.join(out)
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            out.append(data[pos:pos + 2])
            pos += 2
            continue
        seg_end = pos + 2 + int.from_bytes(data[pos + 2:pos + 4], 'big')
        if seg_end > len(data):
            raise ValueError('truncated JPEG segment')
        body = data[pos + 4:seg_end]
        is_meta = marker == 0xFE or (0xE0 <= marker <= 0xEF and not any(
            body.startswith(sig) for sig in _JPEG_KEEP_APP.get(marker, ())))
        if not is_meta:
            out.append(data[pos:seg_end])
        pos = seg_end
    raise ValueError('JPEG without image data')


def _strip_png(data: bytes) -> bytes:
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a PNG')
    out = [data[:8]]
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], 'big')
        ctype = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data):
            raise ValueError('truncated PNG chunk')
        if ctype not in _PNG_DROP:
            out.append(data[pos:end])
        pos = end
        if ctype == b'IEND':
            return b''.join(out)
    raise ValueError('PNG without IEND')


def strip_metadata(data: bytes, fmt: str) -> bytes:
    """The same file without its metadata, pixels untouched (no re-encode).
    Raises ValueError on a file it cannot walk."""
    return _strip_jpeg(data) if fmt == 'JPEG' else _strip_png(data)


def _encode(img, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == 'JPEG':
        img.convert('RGB').save(buf, 'JPEG', quality=92, optimize=True)
    else:
        img.save(buf, 'PNG', optimize=False)
    return buf.getvalue()


def prepare_image(data: bytes, slots_left: int = 1) -> List[PreparedImage]:
    """Turn one original image into what is sent (one image, or tiles)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        return _prepare(data, img, slots_left)


def _prepare(data: bytes, img, slots_left: int) -> List[PreparedImage]:
    from PIL import Image, ImageOps

    fmt_in = img.format or 'PNG'
    orientation = img.getexif().get(0x0112, 1)
    animated = bool(getattr(img, 'is_animated', False))
    out_fmt = 'JPEG' if fmt_in == 'JPEG' else 'PNG'
    media = 'image/jpeg' if out_fmt == 'JPEG' else 'image/png'

    if orientation not in (1, None):
        img = ImageOps.exif_transpose(img)
    w, h = img.size

    def made(pic, tw, th, raw=None):
        return PreparedImage(raw if raw is not None else _encode(pic, out_fmt),
                             media, tw, th, image_tokens(tw, th), animated)

    if _padded_area(w, h) <= BUDGET_PIXELS:
        if orientation in (1, None) and fmt_in in ('PNG', 'JPEG'):
            try:
                return [made(None, w, h, strip_metadata(data, fmt_in))]
            except ValueError:
                pass                                    # cannot walk it: re-encode
        return [made(img, w, h)]

    fw, fh = fit_size(w, h)
    short, long_ = min(w, h), max(w, h)
    if min(fw, fh) >= MIN_READABLE_SHORT_SIDE or slots_left <= 1:
        return [made(img.resize((fw, fh), Image.LANCZOS), fw, fh)]

    # Tile along the long side. Keep the short side readable (capped), and if
    # that needs more tiles than slots, shrink until it fits the slots.
    s = min(short, MAX_TILE_SHORT_SIDE)
    while True:
        f = s / short
        # The server measures the padded canvas, so the tile length is what
        # fits next to the short side rounded UP to the cell.
        tile_len = (BUDGET_PIXELS // _ceil_to(s, CELL)) // CELL * CELL
        n = math.ceil(long_ * f / tile_len)
        if n <= slots_left or s <= MIN_READABLE_SHORT_SIDE // 2:
            break
        s = max(MIN_READABLE_SHORT_SIDE // 2, int(s * math.sqrt(slots_left / n)))
    n = min(n, slots_left)
    if f < 1.0:
        img = img.resize((round(w * f), round(h * f)), Image.LANCZOS)
    W, H = img.size
    tiles: List[PreparedImage] = []
    step = math.ceil((H if H >= W else W) / n)
    for i in range(n):
        if H >= W:
            box = (0, i * step, W, min(H, (i + 1) * step))
        else:
            box = (i * step, 0, min(W, (i + 1) * step), H)
        t = img.crop(box)
        tw, th = t.size
        if _padded_area(tw, th) > BUDGET_PIXELS:   # rounding guard
            tw, th = fit_size(tw, th)
            t = t.resize((tw, th), Image.LANCZOS)
        tiles.append(made(t, tw, th))
    return tiles


# === Token cost of an image block already in the conversation ===
#
# Counting runs on every turn, so read only the header: the first ~96 KB of the
# image covers a PNG IHDR and a JPEG SOF behind its EXIF (an EXIF larger than
# that — a big embedded thumbnail — falls back to the whole image). Blocks made by
# prepare_image never carry a rotation tag (rotation is baked into the pixels),
# so the header size is the size the server sees.
_HEADER_B64_CHARS = 131072          # multiple of 4
_token_cache: dict = {}


def image_block_tokens(block) -> int:
    """Server token cost of an Anthropic image block; MAX when it cannot tell."""
    import base64
    src = block.get('source') if isinstance(block, dict) else getattr(block, 'source', None)
    data = src.get('data') if isinstance(src, dict) else getattr(src, 'data', None)
    if not data:
        return MAX_TOKENS_PER_IMAGE
    key = (len(data), data[:48], data[-48:])
    hit = _token_cache.get(key)
    if hit is not None:
        return hit
    from PIL import Image
    tokens = MAX_TOKENS_PER_IMAGE
    for chars in ((_HEADER_B64_CHARS, None) if len(data) > _HEADER_B64_CHARS else (None,)):
        try:
            raw = base64.b64decode(data[:chars] if chars else data)
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
            tokens = image_tokens(w, h)
            break
        except Exception:
            continue
    if len(_token_cache) > 4096:
        _token_cache.clear()
    _token_cache[key] = tokens
    return tokens
