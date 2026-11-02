"""The client-side collapse/stall signals in _openai_common.

The 2026-09-12 wedge: a turn whose whole output was tool-call arguments never
marked progress, so the stall guard kept the 600s first-token budget for a
stream that was in fact flowing. Tool-call deltas are output and must count —
for the stall budget AND for the collapse guard, since a decoder can collapse
inside an argument string just as well as in the answer.
"""
from codemason.services._openai_common import _DegenerateRunGuard


def test_feeding_marks_progress_without_a_usage_counter():
    g = _DegenerateRunGuard(2000)
    assert g.last_output_at is None
    g.feed('{"comm')
    assert g.last_output_at is not None


def test_note_progress_marks_progress_but_never_feeds_the_window():
    g = _DegenerateRunGuard(4)
    g.note_progress()
    assert g.last_output_at is not None
    assert not g.buf, "a usage counter carries no characters"
    assert g.tripped is None


def test_a_collapsed_argument_string_trips_the_guard():
    g = _DegenerateRunGuard(2000)
    g.feed('{"content": "')
    g.feed('!' * 1500)
    assert g.tripped is None, "under the window: not yet"
    g.feed('!' * 500)
    assert g.tripped is not None, "2000 of '!' behind a quote: a stuck decoder"
    assert '2000' in g.tripped


def test_real_arguments_do_not_trip():
    g = _DegenerateRunGuard(2000)
    for _ in range(200):
        g.feed('{"file_path": "/tmp/report.md", "content": "안녕하세요 line\\n"}')
    assert g.tripped is None


def test_zero_window_disables_the_guard_but_still_marks_progress():
    g = _DegenerateRunGuard(0)
    g.feed('!' * 5000)
    assert g.tripped is None
    assert g.last_output_at is not None
