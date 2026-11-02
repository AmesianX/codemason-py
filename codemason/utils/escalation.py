"""The house waiting patterns: how long to keep waiting, and how often to say so.

Two ladders, running in OPPOSITE directions, because what waiting means depends
on what is being waited for.

Waiting on a PERSON, the odds fall with time. Someone who has not answered in ten
minutes is probably not at the desk, and the next ten minutes are worth less than
the last. So the steps shorten — 10, 5, 3 — and at the end the program decides
for itself and carries on. Waiting longer buys less and less.

Waiting on a SERVER, the odds do the reverse. A backend still refusing after half
an hour is more likely to be in maintenance than to be gone forever, and polling
it harder changes nothing. Meanwhile the cost of giving up grows: the work that
is lost is everything done so far. So the checkpoints spread out — 5, 10, 15, 20,
30, 60, 90 minutes, ending at 2 hours — and at the end it stops rather than
inventing an answer, because there is nothing to invent.

Both ladders live here so the two cannot drift apart into unrelated numbers, and
so the reason they differ is written down once instead of being rederived every
time someone notices they are not the same.
"""

from typing import Tuple

# Waiting on a person: stage lengths in seconds. 10 minutes, then 5, then 3.
# Shortening on purpose — see the module note. At the end the caller acts on its
# own (AskUserQuestion takes the recommended option).
ASK_LADDER_SECONDS: Tuple[int, ...] = (600, 300, 180)

# Waiting on a server: ELAPSED checkpoints in seconds, not stage lengths.
#
# 5, 10, 15, 20, 30, 60, 90 minutes, ending at 2 hours. The gaps between them
# widen (5, 5, 5, 5, 10, 30, 30, 30 minutes), which is the point: report often
# while somebody might still be watching, then back off, because a long outage is
# not made shorter by asking about it more.
#
# The last entry is the deadline. Reaching it means calling the outage permanent
# — a judgement, not a fact, and the one the caller can be wrong about in the
# cheaper direction: two idle hours on a machine nobody is watching, against
# losing everything done before it.
RETRY_CHECKPOINTS_SECONDS: Tuple[int, ...] = (300, 600, 900, 1200, 1800, 3600, 5400, 7200)


def ask_total_minutes() -> int:
    """Total the ask ladder waits before the caller acts, in whole minutes."""
    return sum(ASK_LADDER_SECONDS) // 60


def ask_minutes_left_after(stage: int) -> int:
    """Minutes left after ask stage `stage` (0-based) has elapsed."""
    return sum(ASK_LADDER_SECONDS[stage + 1:]) // 60


def retry_deadline_secs() -> float:
    """How long the retry ladder keeps trying before giving up, in seconds."""
    return float(RETRY_CHECKPOINTS_SECONDS[-1])


def retry_checkpoints_passed(elapsed_secs: float) -> int:
    """How many retry checkpoints `elapsed_secs` has passed.

    The retry loop waits in one continuous stretch rather than in discrete
    stages, so it asks this after each failure to know whether it has just
    crossed a line worth reporting.
    """
    return sum(1 for cp in RETRY_CHECKPOINTS_SECONDS if elapsed_secs >= cp)


def retry_minutes_left(elapsed_secs: float) -> int:
    """Minutes remaining until the retry deadline, from `elapsed_secs`."""
    left = retry_deadline_secs() - elapsed_secs
    return 0 if left <= 0 else round(left / 60.0)


def minutes_for_message(secs: float) -> int:
    """A duration in whole minutes for a message, never rounding down to zero.

    Plain integer division prints "retrying for up to 0 min" for any budget under
    a minute, which reads as broken rather than as short. Anything non-zero is at
    least one minute; only a genuinely disabled wait is zero.
    """
    import math
    if secs <= 0:
        return 0
    return max(1, math.ceil(secs / 60.0))
