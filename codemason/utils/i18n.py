"""UI strings, translated.

The source text is the key. `_("Remote connected")` returns the translation when
there is one and the string it was handed when there is not, so an empty catalog
is a fully English program and adding a language is adding a file. No code has to
change for a new language to work, which is what makes a translation the easiest
first contribution this project can offer.

Not gettext: .mo compilation drags in a build step and a msgfmt dependency, and
this repo runs on the standard library. Keeping the English source as the key
means xgettext can lift these later if the project outgrows this.

The UI language resolves in this order, first hit wins:
    1. config.ui_language   when the UI should differ from the model's language
    2. config.language      the usual case, both together
    3. the conversation      the script the user actually types in (see below)
    4. OS locale             LANG/LC_ALL, and the Windows UI language
    5. English

Hangul, kana, Han and Cyrillic identify themselves in a conversation
(utils/language.py reads them); English, German and Spanish are all Latin script
and cannot be told apart that way. That asymmetry argues for ignoring Latin
script, not for ignoring the conversation: a user typing Hangul into an English
UI has already said which language they read, and the detector for it is sitting
right there. So a self-identifying script promotes the UI (see observe_user_text),
Latin script says nothing, and an explicit setting is never overridden.

This matters most where it is least visible. The common way to install this is to
copy the source to another machine without ~/.codemason/config.json — no
config.language, and on Windows no LANG either — which is exactly the situation
where every rule above except this one produces English for a Korean user.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

# Supported languages: catalog module name -> (display name, accepted spellings).
# The aliases cover locale codes and whatever a user might reasonably write in
# config.language.
LANGUAGES: Dict[str, tuple] = {
    # English has no catalog — it is the source. It is registered anyway so that
    # "show me English" can be said out loud: without an entry, ui_language set to
    # English reads as an unknown value and falls through to config.language.
    'en': ('English', ('en', 'eng', 'english', 'en_us', 'en_gb', 'c', 'posix')),
    'ko': ('한국어', ('ko', 'kor', 'korean', '한국어', 'ko_kr')),
    'ja': ('日本語', ('ja', 'jpn', 'japanese', '日本語', 'ja_jp')),
    'zh': ('中文', ('zh', 'chi', 'zho', 'chinese', '中文', 'zh_cn', 'zh_hans')),
    'ru': ('Русский', ('ru', 'rus', 'russian', 'русский', 'ru_ru')),
    'de': ('Deutsch', ('de', 'deu', 'ger', 'german', 'deutsch', 'de_de')),
    'es': ('Español', ('es', 'spa', 'spanish', 'español', 'espanol', 'es_es')),
    'fr': ('Français', ('fr', 'fra', 'french', 'français', 'francais', 'fr_fr')),
    'pt': ('Português', ('pt', 'por', 'portuguese', 'português', 'portugues', 'pt_br')),
}

_ALIASES = {alias: code for code, (_n, aliases) in LANGUAGES.items() for alias in aliases}

_catalogs: Dict[str, Dict[str, str]] = {}
_active: Optional[str] = None
_resolved = False

# Where _active came from, so a later signal knows whether it may overrule it:
# 'config' and 'explicit' (a /language switch) are the user speaking and are
# final; 'locale' is a guess the conversation is allowed to correct.
_source: Optional[str] = None

# Conversation evidence, until it decides. Counting is by TURN, never by
# character: one paste of a 500-line English log is a single turn, and weighing
# volume would let it outvote everything the user actually typed.
_turns: Dict[str, int] = {}
_conversation_settled = False

# How many self-identifying turns settle it. Two, so a single quoted line does
# not flip the UI mid-session — unless the OS agrees, and then one is plenty:
# two independent signals pointing the same way are not going to be a mistake.
_TURNS_TO_SETTLE = 2
_TURNS_TO_SETTLE_WITH_LOCALE = 1


def normalize(name: Optional[str]) -> Optional[str]:
    """'Korean' / 'ko_KR.UTF-8' / '한국어' -> 'ko'. None for anything unknown."""
    if not name:
        return None
    key = str(name).strip().lower().replace('-', '_')
    key = key.split('.')[0].split('@')[0]          # ko_KR.UTF-8 -> ko_kr
    if key in _ALIASES:
        return _ALIASES[key]
    head = key.split('_')[0]
    return _ALIASES.get(head)


def _from_config() -> Optional[str]:
    try:
        from .config import get_config
        cfg = get_config()
        return (normalize(getattr(cfg, 'ui_language', ''))
                or normalize(getattr(cfg, 'language', '')))
    except Exception:
        return None


def _from_locale() -> Optional[str]:
    for var in ('LC_ALL', 'LC_MESSAGES', 'LANG'):
        code = normalize(os.environ.get(var))
        if code:
            return code
    return _from_os_locale()


def _from_os_locale() -> Optional[str]:
    """The OS's own idea of the UI language, where the environment is silent.

    LC_ALL/LANG are the POSIX answer and Windows simply does not set them, so on
    Windows the documented "OS locale" step used to be no step at all: a Korean
    Windows with no config.json fell straight through to English. Ask the OS
    directly there — GetUserDefaultUILanguage is the language Windows itself is
    displayed in, which is the closer question than the formatting locale.
    """
    try:
        import ctypes
        code = normalize(_lcid_to_tag(ctypes.windll.kernel32.GetUserDefaultUILanguage()))
        if code:
            return code
    except Exception:
        pass
    try:                                            # POSIX fallback path
        import locale
        return normalize((locale.getlocale()[0] or ''))
    except Exception:
        return None


def _lcid_to_tag(lcid: int) -> Optional[str]:
    """Windows LCID -> a name normalize() understands. Only the primary language
    matters here (ko-KR and ko-KP read the same catalog), so the low 10 bits are
    all we look at."""
    if not lcid:
        return None
    return {
        0x12: 'ko', 0x11: 'ja', 0x04: 'zh', 0x19: 'ru',
        0x07: 'de', 0x0A: 'es', 0x0C: 'fr', 0x16: 'pt', 0x09: 'en',
    }.get(lcid & 0x3FF)


def active() -> Optional[str]:
    """The active language code, or None when the UI is in English."""
    global _active, _resolved, _source
    if not _resolved:
        from_config = _from_config()
        _active = from_config or _from_locale()
        _source = 'config' if from_config else ('locale' if _active else None)
        _resolved = True
    return _active


def set_language(name: Optional[str]) -> Optional[str]:
    """Switch at runtime. Returns the code; an unknown name falls back to English."""
    global _active, _resolved, _source, _conversation_settled
    _active = normalize(name)
    _resolved = True
    _source = 'explicit'
    _conversation_settled = True                    # said out loud; stop guessing
    return _active


def reset() -> None:
    """Re-resolve on the next call — after a config change, and in tests."""
    global _active, _resolved, _source, _turns, _conversation_settled
    _active, _resolved, _source = None, False, None
    _turns, _conversation_settled = {}, False


def observe_user_text(text: str) -> None:
    """Take one user turn as evidence of which language to show the UI in.

    Called for each turn the user types, and once per stored turn when a session
    is resumed (a new process starts with an empty count, and the transcript
    already holds the answer). NOT called for compaction summaries: a summary is
    the model's prose, usually English, and feeding it back here would flip the
    UI to English every time a long session compacts.

    Deliberately cheap to call and permanently free once settled — the first
    thing it does is stop having anything to do.
    """
    global _conversation_settled
    if _conversation_settled:
        return
    if active() is not None and _source in ('config', 'explicit'):
        _conversation_settled = True                # the user already said so
        return

    from .language import detect_language
    code = normalize(detect_language(_typed_only(text)))
    if not code:                                    # Latin script: no signal
        return

    _turns[code] = _turns.get(code, 0) + 1
    needed = (_TURNS_TO_SETTLE_WITH_LOCALE if _from_locale() == code
              else _TURNS_TO_SETTLE)
    if _turns[code] >= needed:
        _promote(code)


def _promote(code: str) -> None:
    """Adopt the conversation's language, once, and stop looking."""
    global _active, _resolved, _source, _conversation_settled
    _active, _resolved, _source = code, True, 'conversation'
    _conversation_settled = True


_NOISE = None


def _typed_only(text: str) -> str:
    """`text` with the parts that are not the user's own writing removed.

    Code blocks, inline code, URLs and paths are Latin whatever the author
    speaks, and a [Pasted text #N] placeholder stands in for something they did
    not type at all. Stripping them keeps a Korean question wrapped around a
    stack trace readable as Korean.
    """
    global _NOISE
    if _NOISE is None:
        import re
        _NOISE = re.compile(
            r'```.*?```'                            # fenced code
            r'|`[^`]*`'                             # inline code
            r'|\[Pasted text[^\]]*\]'               # paste placeholder
            r'|<[^>\s]+>'                           # tags, <bash-input> etc.
            r'|\b[a-z]+://\S+'                      # URLs
            r'|(?:^|\s)[~./][^\s]*',                # paths
            re.DOTALL | re.IGNORECASE)
    return _NOISE.sub(' ', text or '')


def _catalog(code: str) -> Dict[str, str]:
    if code not in _catalogs:
        try:
            mod = __import__(f'codemason.locales.{code}', fromlist=['MESSAGES'])
            _catalogs[code] = getattr(mod, 'MESSAGES', {}) or {}
        except Exception:
            _catalogs[code] = {}                   # no catalog: run in English
    return _catalogs[code]


def gettext(text: str, **kwargs) -> str:
    """The translated line, or the English it was handed.

    Placeholders are named (`{count}`), never positional: a translation reorders
    the sentence, and a translator should not have to track argument order for it
    to survive. A formatting failure — a typo in someone's catalog — falls back
    to the source rather than raising, because one line of UI is never worth
    taking the program down.
    """
    code = active()
    out = _catalog(code).get(text, text) if code else text
    if kwargs:
        try:
            return out.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            try:
                return text.format(**kwargs)
            except Exception:
                return text
    return out


# The conventional short alias at call sites.
_ = gettext
