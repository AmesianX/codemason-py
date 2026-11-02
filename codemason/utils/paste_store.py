"""Pasted-text placeholder store — based on Claude Code's
history.ts.

A large or multi-line paste is replaced in the input box by a compact
``[Pasted text #N +M lines]`` placeholder; the full text is kept here and
spliced back in when the message is actually submitted to the model. On screen
the user sees only the placeholder (Ctrl+O expands the full text) — so a 500-line
paste doesn't carpet the terminal, but the model still receives the whole thing.

Image paste is intentionally out of scope here (needs a vision-capable model).
"""
import re
from typing import Dict, List, Optional, Tuple


class PasteStore:
    """Per-session id -> full pasted text. Ids count up from 1 and are stable
    for the life of the REPL so a placeholder can always be expanded."""

    # Cap the retained pastes so a very long REPL session can't grow the store
    # without bound. Past this many pastes the oldest is dropped (its placeholder
    # can no longer be Ctrl+O-expanded — acceptable at 1000+ pastes in a session).
    _MAX_PASTES = 1000

    def __init__(self) -> None:
        self._store: Dict[int, str] = {}
        self._next_id: int = 1

    def add(self, text: str) -> int:
        pid = self._next_id
        self._next_id += 1
        self._store[pid] = text
        if len(self._store) > self._MAX_PASTES:
            del self._store[min(self._store)]  # evict oldest id
        return pid

    def get(self, pid: int) -> Optional[str]:
        return self._store.get(pid)

    def __contains__(self, pid: int) -> bool:
        return pid in self._store


def pasted_num_lines(text: str) -> int:
    """Number of line breaks (``line1\\nline2\\nline3`` -> 2), matching
    history.ts's getPastedTextRefNumLines (it counts breaks, not lines)."""
    return len(re.findall(r'\r\n|\r|\n', text))


def format_pasted_ref(pid: int, num_lines: int) -> str:
    """`[Pasted text #N]` for a single line, `[Pasted text #N +M lines]` else."""
    if num_lines == 0:
        return f"[Pasted text #{pid}]"
    return f"[Pasted text #{pid} +{num_lines} lines]"


# Matches the placeholder we emit. (history.ts also parses Image / ...Truncated
# text refs; we only handle text paste here.)
_REF_RE = re.compile(r'\[Pasted text #(\d+)(?: \+\d+ lines)?\]')


def parse_references(text: str) -> List[Tuple[int, str, int]]:
    """Return (pid, matched_str, start_index) for each placeholder in `text`."""
    out: List[Tuple[int, str, int]] = []
    for m in _REF_RE.finditer(text):
        pid = int(m.group(1))
        if pid > 0:
            out.append((pid, m.group(0), m.start()))
    return out


def expand_pasted_refs(text: str, store: PasteStore) -> str:
    """Replace ``[Pasted text #N]`` placeholders with their stored full content.
    Splices at the original offsets in reverse so earlier indices stay valid, and
    so placeholder-looking strings inside pasted content are never re-expanded
    (faithful to history.ts's expandPastedTextRefs)."""
    refs = parse_references(text)
    expanded = text
    for pid, match, index in reversed(refs):
        content = store.get(pid)
        if content is None:
            continue  # unknown id (e.g. from restored history) — leave as-is
        expanded = expanded[:index] + content + expanded[index + len(match):]
    return expanded


# A paste becomes a placeholder when it spans multiple lines OR is longer than
# this many characters. Small single-line pastes stay inline — no placeholder
# noise for pasting a URL or a short snippet.
PASTE_PLACEHOLDER_MIN_CHARS = 200


def should_placeholder(text: str) -> bool:
    return pasted_num_lines(text) >= 1 or len(text) > PASTE_PLACEHOLDER_MIN_CHARS
