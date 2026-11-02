"""Which language the model should write in.

The rule cannot depend on ~/.codemason/config.json: that file is per machine, and
on a box where `language` is unset every language instruction would silently
vanish — which is exactly how the Agent tool ended up authoring English briefs for
a Korean user. So the language is DERIVED from the conversation (the script the
user actually typed in) and config.language only overrides it when set.

Detection is by script, not by word lists: Hangul -> Korean, kana -> Japanese, Han
without kana -> Chinese, Cyrillic -> Russian. Latin returns None, meaning "say
nothing" — the model already answers Latin-script prompts in kind, and staying
silent keeps the upstream (no language section) shape for those users.
"""

import re
from typing import Optional

# Script ranges, checked in order; kana before Han so Japanese isn't read as Chinese.
_SCRIPTS = (
    ('Korean',   ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F))),
    ('Japanese', ((0x3040, 0x309F), (0x30A0, 0x30FF))),
    ('Chinese',  ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    ('Russian',  ((0x0400, 0x04FF),)),
)

# A stray CJK identifier in an otherwise English question ("what is 中文 mode?")
# shouldn't flip the language. Require the script to carry real weight.
_MIN_SHARE = 0.10
_MIN_CHARS = 2

# Text in a message that the user did not write in any language: the label a
# photo from the phone travels as (remote/poller.py) and file paths/URLs. A
# 64-hex attachment path is ~80 Latin letters, enough to sink a short Korean
# caption under _MIN_SHARE.
_PHOTO_LABEL = re.compile(r'^\[(?:Photo from the phone|A photo was sent from the phone)\b.*\]$',
                          re.MULTILINE)


def _written_text(text: str) -> str:
    text = _PHOTO_LABEL.sub('', text)
    return ' '.join(w for w in text.split() if '/' not in w and '\\' not in w)

# Set per request from the conversation (services build the tool schemas and the
# system prompt in the same pass). Single-threaded async CLI: no locking needed.
_request_language: Optional[str] = None


def detect_language(text: str) -> Optional[str]:
    """The language `text` is written in, or None for Latin script / no signal."""
    if not text:
        return None
    letters = [c for c in _written_text(text) if c.isalpha()]
    if not letters:
        return None
    for name, ranges in _SCRIPTS:
        hits = sum(1 for c in letters
                   if any(lo <= ord(c) <= hi for lo, hi in ranges))
        if hits < _MIN_CHARS:
            continue
        # Kana (hiragana/katakana) is used ONLY by Japanese — Chinese never has
        # it — so a couple of kana chars are a definitive Japanese signal even in
        # kanji-heavy text whose kana share dips below _MIN_SHARE (which would
        # otherwise fall through and match Chinese on the shared Han chars). The
        # ordering (Japanese before Chinese) only helps if Japanese actually
        # returns here, which the share gate previously prevented.
        if name == 'Japanese' or hits / len(letters) >= _MIN_SHARE:
            return name
    return None


def set_request_language(language: Optional[str]) -> None:
    global _request_language
    _request_language = language


def effective_language() -> Optional[str]:
    """The language to instruct the model in.

    The CONVERSATION decides; config.language is only a fallback for when the text
    carries no script signal (Latin), where it still serves its upstream purpose
    ("always answer in X"). Deliberately not the other way round: a per-machine
    setting must never gate or silently change behaviour, or the same feature works
    on the box it was written on and quietly dies everywhere else."""
    from .config import get_config
    return _request_language or get_config().language


# The language pinned for this SESSION — what the sub-agent and forked-skill
# system prompts are built from.
#
# They must not use effective_language(). That value follows the last message of
# the conversation, which inside a tool loop is a tool RESULT, so it swings with
# whatever a command printed. A sub-agent's system prompt is built once per
# spawn, so a swing did not break the cache within one sub-agent run — but it
# made every spawn's prefix (tool block + agent prompt, ~20k tokens) a coin flip
# against the previous one, so consecutive sub-agents could not share it. On a
# KV pool sized for 1.62 conversations (measured 2026-09-07: 847,533 tokens)
# every needless cold prefill also evicts the parent's blocks.
#
# Pinned by the first user prompt that carries a script signal; config.language
# until then. First pin wins for the life of the process.
_session_language: Optional[str] = None


def pin_session_language(language: Optional[str]) -> None:
    global _session_language
    if language and _session_language is None:
        _session_language = language


def session_language() -> Optional[str]:
    from .config import get_config
    return _session_language or get_config().language
