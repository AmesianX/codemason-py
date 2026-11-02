"""
Full-text search over this project's past session transcripts.

The transcripts already exist (session_storage writes one JSONL per session);
this module makes them recallable. claude.ai gives the model a
conversation_search tool over past chats — locally the chats are files on
disk, so the same capability is an SQLite FTS5 index next to them:

    <project_dir>/search-index.db
        msg            fts5(text, session_id, ts, role, file)
        indexed_files  (file, mtime, size, first_prompt)

Only user/assistant TEXT is indexed — not tool results, not control payloads
(<task-notification> etc.), not compact summaries (their originals are in the
same file; indexing both would double every hit). The index is synced lazily
on each search: files whose (mtime, size) changed are re-indexed, vanished
files are dropped. The CURRENT session is excluded on both the index and the
query side — it is this context, not a memory — and a concurrent session in
the same project may index it early, which is why the query side filters too.

FTS5 can be compiled out of a distro's sqlite3; that is a real boundary, so
search() degrades to a linear scan of the transcripts there (list_sessions
already full-reads them all for /resume, so the scale is accepted).
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .session_storage import (
    extract_first_prompt,
    get_project_dir,
    load_transcript_file,
)
from .state import get_original_cwd, get_session_id
from .platform import chmod_posix

_DB_NAME = 'search-index.db'

# Snippet window FTS5 builds around matches (in tokens); matched terms are
# marked with [brackets] so the model can see WHY a row matched.
_SNIPPET_TOKENS = 24

# Control payloads that render as user turns but aren't the user speaking —
# same shape session_storage skips when picking a first prompt.
_CONTROL_TEXT = re.compile(r'^\s*(?:<[a-z][\w-]*[\s>]|\[Request interrupted)')

# Per-message indexing cap. A pasted log can be huge; past 16k chars a match
# adds recall noise, not signal.
_TEXT_CAP = 16_000

_fts5_ok: Optional[bool] = None


def _restrict_index_permissions(db_path) -> None:
    """Take the index down to 0600, like the transcripts it is built from.

    sqlite3 creates the file with the process umask, normally 0644, while
    session_storage writes the transcripts themselves 0600. The index holds the
    SAME text — the fts5 table stores every message body — so the stricter of
    the two is the right mode for it.

    Defence in depth rather than a live leak: the containing project directory
    is already 0700, so another local user cannot traverse into it today. What
    this covers is the directory mode being loosened later, and the file being
    copied somewhere flatter (a backup, an archive, a bug report), where the
    mode travels with it.

    Best effort and POSIX-only. Windows ACLs are inherited from the directory
    and have no chmod equivalent; failing to tighten is not a reason to fail a
    search.
    """
    if os.name == 'nt':
        return
    # The -wal and -shm sidecars carry the same content between checkpoints, so
    # tightening only the main file would leave the data readable next to it.
    for suffix in ('', '-wal', '-shm'):
        try:
            chmod_posix(f'{db_path}{suffix}', 0o600)
        except OSError:
            pass


def _fts5_available() -> bool:
    global _fts5_ok
    if _fts5_ok is None:
        try:
            probe = sqlite3.connect(':memory:')
            probe.execute('CREATE VIRTUAL TABLE t USING fts5(x)')
            probe.close()
            _fts5_ok = True
        except sqlite3.Error:
            _fts5_ok = False
    return _fts5_ok


def _entry_texts(entry: Dict[str, Any]) -> List[str]:
    """The searchable text of one transcript entry ([] for most entries)."""
    if entry.get('isMeta') or entry.get('isCompactSummary'):
        return []
    if entry.get('type') not in ('user', 'assistant'):
        return []
    content = (entry.get('message') or {}).get('content')
    texts: List[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text' and block.get('text'):
                texts.append(block['text'])
    return [t for t in texts if t.strip() and not _CONTROL_TEXT.match(t)]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        'CREATE VIRTUAL TABLE IF NOT EXISTS msg USING fts5('
        'text, session_id UNINDEXED, ts UNINDEXED, role UNINDEXED, file UNINDEXED)'
    )
    conn.execute(
        'CREATE TABLE IF NOT EXISTS indexed_files('
        'file TEXT PRIMARY KEY, mtime REAL, size INTEGER, first_prompt TEXT)'
    )


def _sync(conn: sqlite3.Connection, project_dir: Path, exclude_stem: str) -> None:
    files = {p.name: p for p in project_dir.glob('*.jsonl') if p.stem != exclude_stem}
    known = {
        row[0]: (row[1], row[2])
        for row in conn.execute('SELECT file, mtime, size FROM indexed_files')
    }
    for name in set(known) - set(files):
        conn.execute('DELETE FROM msg WHERE file = ?', (name,))
        conn.execute('DELETE FROM indexed_files WHERE file = ?', (name,))
    for name, path in files.items():
        try:
            st = path.stat()
        except OSError:
            continue
        if known.get(name) == (st.st_mtime, st.st_size):
            continue
        try:
            messages, order = load_transcript_file(path)
        except Exception:
            continue  # oversized/unreadable — skipped, not fatal (list_sessions rule)
        entries = [messages[u] for u in order]
        sid = next((e.get('sessionId') for e in entries if e.get('sessionId')), path.stem)
        rows = [
            (text[:_TEXT_CAP], str(sid), e.get('timestamp') or '', e.get('type'), name)
            for e in entries
            for text in _entry_texts(e)
        ]
        conn.execute('DELETE FROM msg WHERE file = ?', (name,))
        conn.executemany(
            'INSERT INTO msg(text, session_id, ts, role, file) VALUES (?, ?, ?, ?, ?)', rows
        )
        conn.execute(
            'INSERT OR REPLACE INTO indexed_files(file, mtime, size, first_prompt) '
            'VALUES (?, ?, ?, ?)',
            (name, st.st_mtime, st.st_size, extract_first_prompt(entries)),
        )
    conn.commit()


def _fts_match_expr(query: str) -> str:
    """User words -> an FTS5 MATCH that cannot be a syntax error.

    Every bare token is quoted (so `config.json` or `--flag` don't parse as
    operators) and searched as a PREFIX: the unicode61 tokenizer splits on
    whitespace, so in agglutinative text a word and its particle are ONE token
    ('텔레그램으로') and an exact-word query for the bare word would miss it —
    and prefix also gives 'auth' → 'authentication' for free, with bm25 still
    ranking the exact hits first. User quotes opt back into exact word/phrase
    matching. Tokens are implicitly ANDed, which is FTS5's default."""
    parts: List[str] = []
    for token in re.findall(r'"[^"]*"|\S+', query):
        if token.startswith('"') and token.endswith('"') and len(token) > 2:
            parts.append(token)
            continue
        bare = token.rstrip('*').replace('"', '')
        if bare:
            parts.append(f'"{bare}"*')
    return ' '.join(parts)


def _group_hits(raw: List[tuple], first_prompts: Dict[str, str],
                limit: int, per_session: int = 3) -> List[Dict[str, Any]]:
    """Rank-ordered rows -> at most `limit` session groups, best-first, at most
    `per_session` snippets each (one long session must not eat the whole
    result)."""
    groups: List[Dict[str, Any]] = []
    by_sid: Dict[str, Dict[str, Any]] = {}
    for snippet, sid, ts, role, file in raw:
        group = by_sid.get(sid)
        if group is None:
            if len(groups) >= limit:
                continue
            group = {
                'session_id': sid,
                'first_prompt': first_prompts.get(file, ''),
                'hits': [],
            }
            by_sid[sid] = group
            groups.append(group)
        if len(group['hits']) < per_session:
            group['hits'].append({'role': role, 'ts': ts, 'snippet': snippet})
    return groups


def search(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Session groups matching `query`, best-first. Raises ValueError on a
    query with no searchable tokens; degrades to a scan without FTS5."""
    project_dir = get_project_dir(get_original_cwd())
    current = get_session_id()
    if not project_dir.is_dir():
        return []
    if not _fts5_available():
        return _scan_search(project_dir, current, query, limit)
    match = _fts_match_expr(query)
    if not match:
        raise ValueError('query has no searchable words')
    _db_path = project_dir / _DB_NAME
    conn = sqlite3.connect(_db_path, timeout=5)
    _restrict_index_permissions(_db_path)
    try:
        _ensure_schema(conn)
        _sync(conn, project_dir, exclude_stem=current)
        raw = conn.execute(
            f"SELECT snippet(msg, 0, '[', ']', ' … ', {_SNIPPET_TOKENS}), "
            'session_id, ts, role, file FROM msg '
            'WHERE msg MATCH ? AND session_id != ? ORDER BY rank LIMIT ?',
            (match, current, limit * 4),
        ).fetchall()
        first_prompts = {
            row[0]: row[1] or ''
            for row in conn.execute('SELECT file, first_prompt FROM indexed_files')
        }
    except sqlite3.Error:
        # A locked/corrupt index must not make recall unavailable; the
        # transcripts themselves are the source of truth.
        return _scan_search(project_dir, current, query, limit)
    finally:
        conn.close()
    return _group_hits(raw, first_prompts, limit)


def _scan_search(project_dir: Path, current: str, query: str,
                 limit: int) -> List[Dict[str, Any]]:
    """No-FTS5 path: case-insensitive AND-of-words over a linear read of every
    transcript. Newest sessions first (no rank without FTS), same group shape."""
    words = [w.strip('"').lower() for w in re.findall(r'"[^"]*"|\S+', query)]
    words = [w.rstrip('*') for w in words if w.strip('*')]
    if not words:
        raise ValueError('query has no searchable words')
    files = sorted(
        (p for p in project_dir.glob('*.jsonl') if p.stem != current),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    groups: List[Dict[str, Any]] = []
    for path in files:
        if len(groups) >= limit:
            break
        try:
            messages, order = load_transcript_file(path)
        except Exception:
            continue
        entries = [messages[u] for u in order]
        hits = []
        for e in entries:
            for text in _entry_texts(e):
                low = text.lower()
                if not all(w in low for w in words):
                    continue
                at = low.find(words[0])
                start = max(0, at - 60)
                clip = text[start:at + 120].replace('\n', ' ').strip()
                hits.append({
                    'role': e.get('type'),
                    'ts': e.get('timestamp') or '',
                    'snippet': ('…' if start else '') + clip + '…',
                })
                break
            if len(hits) >= 3:
                break
        if hits:
            sid = next((e.get('sessionId') for e in entries if e.get('sessionId')),
                       path.stem)
            groups.append({
                'session_id': str(sid),
                'first_prompt': extract_first_prompt(entries),
                'hits': hits,
            })
    return groups
