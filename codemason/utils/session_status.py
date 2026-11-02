"""Short status of the conversation for the screen and the phone: how full the
context is and how many images ride along. One place, so the PC footer and the
Telegram message cannot disagree."""

from typing import Any, List, Optional, Tuple

from .i18n import _


# A limit no conversation reaches, so attached_image_count counts every image the
# history still holds instead of the number one request carries.
_COUNT_ALL = 1 << 30


def image_fill(messages: List[Any]) -> Optional[Tuple[int, int]]:
    """(attached, held), or None when the server takes no images.

    `attached` is what the next request carries, `held` is every image the
    conversation still has, so the gap between them is what the server's
    per-request limit pushed out. Reads the limit already found
    (services/vision.py known_limit) and never asks the server: this runs in the
    prompt's repaint. Both counts come from attached_image_count, so neither can
    disagree with the request itself."""
    try:
        from ..services.vision import known_limit
        # > 0 only after enabled_limit() found a vLLM server that takes images.
        # Nothing is asked from here: in auto mode get_backend() can probe too.
        n = known_limit()
        if n <= 0:
            return None
        from ..services._openai_common import attached_image_count
        return attached_image_count(messages, n), attached_image_count(messages, _COUNT_ALL)
    except Exception:  # noqa: BLE001 - display must never break a turn
        return None


def image_text(messages: List[Any]) -> str:
    fill = image_fill(messages)
    if fill is None:
        return ''
    attached, held = fill
    if attached == held:
        return _('images {k}/{n}', k=attached, n=held)
    # The denominator is every image the conversation holds, so the two numbers
    # part the moment the per-request limit pushes the oldest out — and the words
    # say what parting them means, since the figure alone cannot. Nothing else on
    # screen tells the user: the model is given the rule in its system prompt
    # (constants/prompts.py) and the eviction goes only to the debug log.
    return _('images {k}/{n} ({d} dropped)', k=attached, n=held, d=held - attached)


def fill_percent(used: int, limit: int) -> int:
    """Whole percent of `limit`, half up, at most 100 — the input-line gauge and
    the phone line both, so they never show two numbers (and Python's
    round-half-even never disagrees with the Rust port by one)."""
    return min(100, int(used * 100 / limit + 0.5))


def context_percent(messages: List[Any]) -> Optional[int]:
    """Fill against the auto-compact threshold, the same figure as the footer."""
    try:
        from .tokens import token_count_with_estimation
        from ..services.compaction import get_auto_compact_threshold
        limit = get_auto_compact_threshold()
        if limit <= 0:
            return None
        return fill_percent(token_count_with_estimation(messages), limit)
    except Exception:  # noqa: BLE001
        return None


def status_line(messages: List[Any]) -> str:
    """"Context 22% · images 6/8" — either part left out when unknown."""
    parts = []
    pct = context_percent(messages)
    if pct is not None:
        parts.append(_('Context {pct}%', pct=pct))
    img = image_text(messages)
    if img:
        parts.append(img)
    return ' · '.join(parts)
