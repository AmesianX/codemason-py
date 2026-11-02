#!/usr/bin/env python3
"""Raising the slot count must not multiply the sub-agent footprint.

SUBAGENT_THRESHOLD_FRACTION was measured with ONE extra tenant in the server's
KV pool, so on its own it says nothing about concurrency. Multiply it by a slot
count and the arithmetic stops being a measurement: 6 x 133,725 on top of a
445,752-token parent is 1,248,102 before Mamba checkpoints, against a pool that
holds 1,356,053 — a wipe, costing a full cold re-prefill of the parent, with
nothing reporting that it happened.

SUBAGENT_TOTAL_FRACTION is the ceiling on the combined footprint, and
get_auto_compact_threshold takes whichever of the two binds first. These cases
pin both halves: the counts that HAVE been measured keep their measured cap, and
above them the total holds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == 'win32':
    # Our own case names carry em dashes and Korean, and the console
    # encoder here is cp949: the first such line would raise
    # UnicodeEncodeError inside the reporter and stop the run mid-suite,
    # blaming whatever case it was printing. Pinned inline, importing
    # nothing -- several of these files must set CODEMASON_CONFIG_DIR
    # before anything under codemason/ is imported.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'\n      {detail}' if detail and not ok else ''))
    results.append(bool(ok))


from codemason.services.compaction import (
    SUBAGENT_THRESHOLD_FRACTION,
    SUBAGENT_TOTAL_FRACTION,
    get_auto_compact_threshold,
)
from codemason.utils.config import get_config


def cap_at(slots):
    """The sub-agent ceiling the code derives at this slot count."""
    cfg = get_config()
    previous = cfg.max_concurrent_subagents
    try:
        cfg.max_concurrent_subagents = slots
        return get_auto_compact_threshold(is_subagent=True)
    finally:
        cfg.max_concurrent_subagents = previous


def main():
    cfg = get_config()
    previous_ceiling = getattr(cfg, 'subagent_context_ceiling_tokens', 0)
    cfg.subagent_context_ceiling_tokens = 0
    try:
        parent = get_auto_compact_threshold()
        check('the parent keeps the whole window it is allowed',
              parent == 445_752, parent)

        per_agent = int(parent * SUBAGENT_THRESHOLD_FRACTION)
        check('one extra tenant costs what the September table measured',
              per_agent == 133_725, per_agent)

        # 1 and 2 are the single-tenant table's regime; 3 is what this
        # deployment runs and what the 2026-10-10 trial measured. None of them
        # may move — a change here would be a silent re-tuning of a measured
        # number.
        for slots in (1, 2, 3):
            check(f'{slots} slots keep the measured per-agent cap',
                  cap_at(slots) == per_agent, cap_at(slots))
        check('a nonsense slot count neither divides by zero nor hands out more',
              cap_at(0) == per_agent, cap_at(0))

        check('a fourth slot has to cost each agent something',
              cap_at(4) < cap_at(3), (cap_at(4), cap_at(3)))

        budget = int(parent * SUBAGENT_TOTAL_FRACTION)
        over = [(n, cap_at(n) * n) for n in range(4, 13) if cap_at(n) * n > budget + n]
        check('the combined footprint stops growing past the measured total',
              not over, over)
        # 0.90 is not a round number chosen for comfort: it is three sub-agents
        # at the per-agent cap, which is this line's actual configuration and
        # the one the 2026-10-10 trials measured (3/3 survived, 0 blocks lost,
        # parent at its full threshold). Moving either constant without a new
        # measurement should break this.
        check('the total is three sub-agents at the measured per-agent cap',
              abs(SUBAGENT_TOTAL_FRACTION - 3 * SUBAGENT_THRESHOLD_FRACTION) < 1e-9,
              (SUBAGENT_TOTAL_FRACTION, SUBAGENT_THRESHOLD_FRACTION))
        check('...which is what the measured slot count costs together',
              cap_at(3) * 3 == int(parent * SUBAGENT_TOTAL_FRACTION) + 0
              or abs(cap_at(3) * 3 - int(parent * SUBAGENT_TOTAL_FRACTION)) <= 3,
              (cap_at(3) * 3, int(parent * SUBAGENT_TOTAL_FRACTION)))

        # An operator who names a ceiling owns the number; the derivation is
        # only the default, and the slot count must not second-guess it.
        cfg.subagent_context_ceiling_tokens = 180_000
        check('an explicit operator ceiling wins over the derivation',
              cap_at(8) == 180_000, cap_at(8))
        cfg.subagent_context_ceiling_tokens = 999_999
        check('...but still cannot raise a sub-agent above the parent',
              cap_at(8) == parent, cap_at(8))
    finally:
        cfg.subagent_context_ceiling_tokens = previous_ceiling

    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
