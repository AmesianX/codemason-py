"""Repair UTF-16 surrogates that leak into input text on the Windows console.

`ReadConsoleInputW` hands back UTF-16 code units, so a non-BMP character (an
emoji, most commonly) arrives as two key events carrying a high and a low
surrogate. prompt_toolkit rejoins them in `_merge_paired_surrogates`, but that
runs per `ReadConsoleInputW` batch (2048 events) with batch-local state: paste
something bigger than one batch and any pair straddling the boundary is emitted
as two lone surrogates, one at the tail of one paste event and one at the head
of the next. Those survive in the buffer and only blow up later, at the HTTP
layer -- `'utf-8' codec can't encode characters ...: surrogates not allowed`.

Repairing at the submit boundary, after `[Pasted text #N]` placeholders are
expanded, is what makes the halves adjacent again and recovers the original
character; doing it per paste event would see only one half and destroy it.
"""
import re

__all__ = ["repair_surrogates"]

# `re` matches code points, and a lone surrogate is a code point in a str, so
# these ranges do mean surrogates here (not the UTF-16 units they encode to).
_ANY = re.compile('[\ud800-\udfff]')
_PAIR = re.compile('[\ud800-\udbff][\udc00-\udfff]')


def _join(m: "re.Match[str]") -> str:
    return m.group().encode('utf-16-le', 'surrogatepass').decode('utf-16-le')


def repair_surrogates(text: str) -> str:
    """Rejoin split surrogate pairs; replace unpairable halves with U+FFFD.

    A no-op (single scan, no copy) for text without surrogates, which is all
    text off Windows. Idempotent."""
    if not _ANY.search(text):
        return text
    return _ANY.sub('�', _PAIR.sub(_join, text))
