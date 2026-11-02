"""A resume must come back the size the session closed at.

A compaction rewrites the in-memory history — the turns it summarised are gone
and the summary stands in for them — but the transcript is append-only and
records nothing about it. So the chain from leaf to root still holds every one
of those turns, and replaying the whole chain undoes the compaction.

Measured on a real transcript (3,444 entries, two compactions): the session
closed at 207,490 tokens and reopened as a 458,753-token request, which the
server refused outright — the conversation could not be opened at all, and the
number shown was the one the last turn had reported, not the one being sent.
One compaction is enough to do this, and the longer the session the surer.

Run: python3 tests/test_resume_after_compaction.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from codemason.services.compaction import (  # noqa: E402
    COMPACT_CONTINUATION_PREFIX,
    get_compact_user_summary_message,
)
from codemason.utils.session_storage import _trim_to_last_compaction  # noqa: E402

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


def turn(text, kind='user'):
    return {'type': kind, 'message': {'role': kind, 'content': [{'type': 'text', 'text': text}]}}


def summary_turn(text='요약', **extra):
    e = turn(get_compact_user_summary_message(text))
    e.update(extra)
    return e


# The shape that broke: old turns, a compaction, then the turns since.
old = [turn(f'before {i}') for i in range(50)]
after = [turn(f'after {i}') for i in range(3)]
chain = old + [summary_turn()] + after

kept = _trim_to_last_compaction(chain)
check('resume starts at the compaction, not at the root', len(kept) == 4, f'kept {len(kept)}')
check('the summary itself is kept — it IS the history now',
      COMPACT_CONTINUATION_PREFIX in kept[0]['message']['content'][0]['text'])
check('nothing the compaction replaced comes back',
      not any('before' in b['text'] for e in kept for b in e['message']['content']))
check('and everything since it does',
      sum('after' in b['text'] for e in kept for b in e['message']['content']) == 3)

# Several compactions: the LAST one wins. Cutting at the first would bring back
# everything the later ones threw away.
chain2 = old + [summary_turn('1차')] + [turn('mid')] * 10 + [summary_turn('2차')] + after
kept2 = _trim_to_last_compaction(chain2)
check('with several compactions it starts at the last one', len(kept2) == 4, f'kept {len(kept2)}')
check('so the turns between two compactions stay dropped',
      not any('mid' in b['text'] for e in kept2 for b in e['message']['content']))

# Transcripts written before anything marked them carry no flag — matching the
# summary's own opening sentence is what makes those readable.
check('a transcript with no flag is still trimmed',
      len(_trim_to_last_compaction(old + [summary_turn()] + after)) == 4)
# And a flag alone is enough, whatever the text says.
flagged = turn('nothing recognisable')
flagged['isCompactSummary'] = True
check('an explicit isCompactSummary is honoured on its own',
      len(_trim_to_last_compaction(old + [flagged] + after)) == 4)

# A session that never compacted must come back whole.
plain = old + after
check('a session that never compacted is untouched',
      _trim_to_last_compaction(plain) == plain)

# The prefix is shared, not copied: a reworded summary must still be found.
check('the boundary text comes from the writer, not a copy of it',
      get_compact_user_summary_message('x').startswith(COMPACT_CONTINUATION_PREFIX))

print(f'\n{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
