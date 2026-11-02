"""The compaction summary is the first text block wherever it sits — a
leading thinking or tool_use block must not make it read as "no summary".
Mirrors summary_text_is_found_behind_a_leading_non_text_block in
src/services/compaction.rs."""
from types import SimpleNamespace

from codemason.services.compaction import _extract_text


def _resp(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_text_behind_a_leading_non_text_block_is_found():
    r = _resp([{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "<summary>s</summary>"}])
    assert _extract_text(r) == "<summary>s</summary>"
    r = _resp([SimpleNamespace(type="tool_use", name="Read"), SimpleNamespace(type="text", text="t")])
    assert _extract_text(r) == "t"


def test_no_text_block_is_none():
    assert _extract_text(_resp([{"type": "thinking", "thinking": "hm"}])) is None
    assert _extract_text(_resp([])) is None
    assert _extract_text(SimpleNamespace()) is None
