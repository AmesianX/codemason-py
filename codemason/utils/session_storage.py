"""
Session transcript persistence — faithful subset of
utils/sessionStorage.ts + utils/sessionStoragePortable.ts + types/logs.ts
from Claude Code.

Storage layout (identical structure to the original, rooted at codemason's
own config home instead of ~/.claude):

    <CONFIG_DIR>/projects/<sanitized-cwd>/<sessionId>.jsonl

One session = one JSONL file; one line per transcript entry. Entry shape is
the original's TranscriptMessage core:

    { parentUuid, isSidechain, userType, cwd, sessionId, version, gitBranch?,
      type, message, uuid, timestamp, ...passthrough (isMeta, ...) }

PORTED (the subset codemason needs now):
  - sanitize_path / projects dir / transcript path resolution
  - append-side: record_transcript (uuid-dedup + linear parentUuid chain),
    appendEntryToFile semantics (0600 file, 0700 dir, append-only)
  - read-side: load_transcript_file (tolerant JSONL parse, 50MB cap),
    leaf detection, buildConversationChain (cycle-guarded walk root←leaf),
    first-prompt extraction, visible-message counting
  - resume: load_conversation_for_resume + adopt_resumed_session
    (adoptResumedSessionFile: same id keeps appending to the same file;
    dedup seeding prevents re-writing loaded messages)
  - listing: list_sessions (LogOption subset, sortLogs order)

NOT ported (subsystems codemason doesn't have — each is an isolated
extension point, the entry format already reserves their shapes):
  - sidechain/agent transcripts (record_transcript(is_sidechain=...) is
    plumbed; getAgentTranscriptPath and AgentTool wiring are not)
  - metadata entry types (summary / custom-title / tag / pr-link / ...):
    the parser skips unknown entry types instead of crashing, so adding a
    type later is purely additive
  - compact-boundary truncation on load (readTranscriptForLoad): a resumed
    post-compaction session restores the FULL pre-compact chain; auto-compact
    re-triggers on the next turn, so this is correct, just less pre-trimmed
  - parallel tool_result orphan recovery (recoverOrphanedParallelToolResults):
    codemason's write chain is strictly linear write-order (one entry's
    parent is simply the previously recorded entry), so the DAG shapes that
    pass recovers cannot occur in files we write
  - lite/tail reads (getSessionFilesLite): list_sessions does full reads,
    which is fine at local scale; the 50MB read cap is enforced
"""

import json
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import CONFIG_DIR
from .platform import open_append_fd
from .serialization import to_jsonable
from .state import get_cwd, get_original_cwd, get_session_id, set_session_id
from .image_store import dehydrate, hydrate

# 50 MB — sessionStorage.ts MAX_TRANSCRIPT_READ_BYTES. Callers reading a raw
# transcript bail above this to avoid OOM.
MAX_TRANSCRIPT_READ_BYTES = 50 * 1024 * 1024

# sessionStoragePortable.ts MAX_SANITIZED_LENGTH.
MAX_SANITIZED_LENGTH = 200

# sessionStorage.ts:125 — non-meaningful first prompts (XML-ish control
# payloads like <task-notification>/<local-command-stdout>, interrupt notices).
_SKIP_FIRST_PROMPT_PATTERN = re.compile(
    r'^(?:\s*<[a-z][\w-]*[\s>]|\[Request interrupted by user[^\]]*\])'
)

# Entry types that participate in the message chain (isTranscriptMessage).
_TRANSCRIPT_TYPES = ('user', 'assistant', 'attachment', 'system')


def _djb2_hash(s: str) -> int:
    h = 5381
    for ch in s:
        h = ((h * 33) ^ ord(ch)) & 0xFFFFFFFF
    # JS djb2Hash returns a signed 32-bit int; mirror that before abs().
    if h >= 0x80000000:
        h -= 0x100000000
    return h


def _to_base36(n: int) -> str:
    if n == 0:
        return '0'
    digits = '0123456789abcdefghijklmnopqrstuvwxyz'
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return ''.join(reversed(out))


def sanitize_path(name: str) -> str:
    """sessionStoragePortable.ts sanitizePath: every non-alphanumeric char
    becomes '-'; over-long results are truncated with a hash suffix."""
    sanitized = re.sub(r'[^a-zA-Z0-9]', '-', name)
    if len(sanitized) <= MAX_SANITIZED_LENGTH:
        return sanitized
    hash_suffix = _to_base36(abs(_djb2_hash(name)))
    return f"{sanitized[:MAX_SANITIZED_LENGTH]}-{hash_suffix}"


def get_projects_dir() -> Path:
    return CONFIG_DIR / 'projects'


def get_project_dir(project_path: str) -> Path:
    return get_projects_dir() / sanitize_path(project_path)


def delete_session(transcript: Path) -> bool:
    """Delete a session: its transcript <id>.jsonl and its own folder <id>/
    (the image store, utils/image_store.py), which nothing else refers to.
    False when the transcript could not be removed."""
    try:
        transcript.unlink()
    except OSError:
        return False
    side = transcript.with_suffix('')
    if side.is_dir() and not side.is_symlink():
        import shutil
        shutil.rmtree(side, ignore_errors=True)
    return True


# Settings that belong to a working directory rather than to the machine.
#
# config.json is one file per home, so it cannot answer a question two instances
# on the same machine answer differently — "which bot does this window use?" is
# exactly that shape. The project directory already exists (it holds the
# transcripts), and it is keyed by the cwd, which is the thing that actually
# distinguishes the two windows. Kept as its own file so it is never confused
# with a transcript: those are *.jsonl, this is settings.json.
_PROJECT_SETTINGS = 'settings.json'


def _project_settings_path(cwd: str = '') -> Path:
    return get_project_dir(cwd or get_original_cwd()) / _PROJECT_SETTINGS


def project_lock(name: str, cwd: str = '') -> str:
    """The lock sidecar for one of this directory's shared documents.

    Project state is shared by every session in the same directory — two `cm2`
    windows in one repo, or a `cm2` and a `codemason` — so its documents need
    the same exclusion as the config dir's, keyed by the thing that
    distinguishes them: the cwd. Every lock in the system still lives in one
    directory (config.LOCKS_DIR); the project is part of the NAME, so a repo's
    lock can never be mistaken for another's.
    """
    from .config import lock_path
    return lock_path(f'{sanitize_path(cwd or get_original_cwd())}.{name}')


def read_project_settings(cwd: str = '') -> Dict[str, Any]:
    """This directory's saved settings, or {} — never raises."""
    try:
        data = json.loads(_project_settings_path(cwd).read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_project_setting(key: str, value: Any, cwd: str = '') -> bool:
    """Merge one setting into this directory's file. True when it landed.

    Read-modify-write through a temp file: another session in the same directory
    may be writing a different key, and a plain overwrite would drop it.
    """
    from .file import update_json_document

    def merge(current):
        data = current if isinstance(current, dict) else {}
        data[key] = value
        return data

    path = _project_settings_path(cwd)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The read and the write are one step. A temp file + replace alone keeps
        # a reader from seeing half a document, which is what this used to do,
        # but it does not stop the other session from having read the SAME
        # document first and then writing its own copy over this one — the key
        # it set is simply gone. Same shape, same fix, as config.json.
        return update_json_document(path, project_lock(_PROJECT_SETTINGS, cwd), merge)
    except OSError:
        return False


def get_transcript_path() -> Path:
    """Path of the CURRENT session's transcript."""
    return get_transcript_path_for_session(get_session_id())


def get_transcript_path_for_session(session_id: str) -> Path:
    return get_project_dir(get_original_cwd()) / f"{session_id}.jsonl"


def _clear_boundary_path() -> Path:
    return get_project_dir(get_original_cwd()) / '.cleared'


def record_clear_boundary() -> None:
    """Mark that the user just ran /clear in this project.

    /clear switches to a fresh, empty session, but an empty session is never
    written to disk and list_sessions() skips it — so a later `--continue`
    would resume the PRE-clear session and the clear looks ignored. Recording a
    boundary (a file whose mtime is the clear time) lets `--continue` tell that
    the last real session predates a deliberate clear and start fresh instead.
    Best-effort — a failure just falls back to the old behavior."""
    try:
        p = _clear_boundary_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('', encoding='utf-8')
        os.utime(p, None)
    except OSError:
        pass


def cleared_after_session(session_id: str) -> bool:
    """True if a /clear happened AT OR AFTER the last write to session_id — i.e.
    resuming it would undo a deliberate clear. Used by --continue."""
    try:
        b = _clear_boundary_path()
        if not b.is_file():
            return False
        s = get_transcript_path_for_session(session_id)
        if not s.is_file():
            return True
        return b.stat().st_mtime >= s.stat().st_mtime
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------

# Per-process session-file state (the original's Project instance subset):
# uuids already on disk (dedup) and the last chain participant's uuid (the
# next entry's parentUuid). Reset on switch_session; seeded on resume-adopt.
_seen_uuids: set = set()
_warned_record_failure = False
_last_chain_uuid: Optional[str] = None

_git_branch_cache: Optional[str] = None
_git_branch_resolved = False


def _get_git_branch() -> Optional[str]:
    """Current git branch for the entry's gitBranch field (cached; None
    outside a repo). Resolved once per process — matches the original, which
    snapshots it at session scope."""
    global _git_branch_cache, _git_branch_resolved
    if not _git_branch_resolved:
        _git_branch_resolved = True
        try:
            out = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=get_cwd(), capture_output=True, text=True, timeout=2,
            )
            branch = out.stdout.strip()
            _git_branch_cache = branch if out.returncode == 0 and branch else None
        except Exception:
            _git_branch_cache = None
    return _git_branch_cache


def _get_version() -> str:
    # Lazy: cli.py imports this module, so a top-level import would be circular.
    try:
        from ..cli import VERSION
        return VERSION
    except Exception:
        return '0.0.0'


def _now_iso() -> str:
    """ISO-8601 UTC with milliseconds and 'Z' — the original's timestamp form
    (new Date().toISOString())."""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def append_entry_to_file(full_path: Path, entry: Dict[str, Any]) -> None:
    """appendEntryToFile: append one JSONL line; 0600 file, 0700 parent dir
    created on demand.

    The append has to be indivisible, because a transcript has more than
    one writer: the turn itself, and every concurrent sub-agent reporting
    into the same session. On POSIX O_APPEND is that guarantee; on Windows
    it is not one, which is why this goes through open_append_fd (see
    utils/platform for the measurement). A JSONL line lost or cut in half
    here is conversation history rather than a log entry, and it would go
    silently -- the write reports success either way."""
    line = json.dumps(entry, ensure_ascii=False, default=to_jsonable) + '\n'
    full_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd = open_append_fd(str(full_path), 0o600)
    try:
        os.write(fd, line.encode('utf-8'))
    finally:
        os.close(fd)


def _msg_field(m: Any, key: str, default: Any = None) -> Any:
    if isinstance(m, dict):
        return m.get(key, default)
    return getattr(m, key, default)


def _serialize_entry(message: Any, parent_uuid: Optional[str],
                     is_sidechain: bool) -> Dict[str, Any]:
    """Build the TranscriptMessage-core JSONL entry for one in-memory message.
    Dict messages pass their extra top-level keys through (isMeta,
    isCompactSummary, ...) so restore round-trips them."""
    entry: Dict[str, Any] = {}
    if isinstance(message, dict):
        for k, v in message.items():
            if k not in ('type', 'uuid', 'message'):
                entry[k] = to_jsonable(v)
    entry.update({
        'parentUuid': parent_uuid,
        'isSidechain': is_sidechain,
        'userType': 'external',
        'cwd': get_cwd(),
        'sessionId': get_session_id(),
        'version': _get_version(),
        'type': _msg_field(message, 'type'),
        # Image payloads go to <session>/images/, the entry keeps a reference
        # (utils/image_store.py); the in-memory message is not touched.
        'message': dehydrate(to_jsonable(_msg_field(message, 'message'))),
        'uuid': str(_msg_field(message, 'uuid', '')),
        'timestamp': _now_iso(),
    })
    branch = _get_git_branch()
    if branch:
        entry['gitBranch'] = branch
    return entry


def record_transcript(messages: List[Any], is_sidechain: bool = False) -> None:
    """recordTranscript subset: called with the FULL in-memory history after
    every mutation (the useLogMessages pattern). Messages whose uuid was
    already written are skipped, so incremental calls append only the new
    tail; each new entry's parentUuid is the previously recorded entry
    (linear write-order chain).

    Persistence must never break the conversation — failures are logged and
    swallowed."""
    global _last_chain_uuid
    try:
        path = get_transcript_path()
        for message in messages:
            mtype = _msg_field(message, 'type')
            if mtype not in _TRANSCRIPT_TYPES:
                continue  # progress etc. — not persisted (post-#24099 shape)
            uuid_str = str(_msg_field(message, 'uuid', '') or '')
            if not uuid_str or uuid_str in _seen_uuids:
                continue
            entry = _serialize_entry(message, _last_chain_uuid, is_sidechain)
            append_entry_to_file(path, entry)
            _seen_uuids.add(uuid_str)
            _last_chain_uuid = uuid_str
    except Exception as e:
        from .log import log_error
        log_error(e)
        # Say it ONCE on screen too. Persistence must never break the
        # conversation, so swallowing the error is right — but writing it only
        # to a log file meant a full disk or a read-only $HOME left the session
        # looking normal while nothing was recorded, and the user found out at
        # --resume time with the transcript short or missing. Once per process:
        # a failing disk fails on every message, and a warning per message would
        # bury the conversation itself.
        global _warned_record_failure
        if not _warned_record_failure:
            _warned_record_failure = True
            print(f"  Warning: this session is no longer being recorded ({e}). "
                  f"--resume will not have it.")


def switch_session(session_id: Optional[str] = None) -> str:
    """Switch to a new (or given) session id and reset the file-pointer state
    (switchSession + resetSessionFilePointer). The new transcript file is
    created lazily by the next record_transcript."""
    global _seen_uuids, _last_chain_uuid
    import uuid as _uuid
    new_id = session_id or str(_uuid.uuid4())
    set_session_id(new_id)
    _seen_uuids = set()
    _last_chain_uuid = None
    return new_id


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------

def load_transcript_file(file_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """loadTranscriptFile subset: parse a session JSONL into
    (messages_by_uuid, write_order_uuids). Unknown entry types and malformed
    lines are skipped (metadata entry types are a future extension, not an
    error). Raises on missing/oversized files."""
    size = file_path.stat().st_size
    if size > MAX_TRANSCRIPT_READ_BYTES:
        raise ValueError(
            f"Transcript exceeds {MAX_TRANSCRIPT_READ_BYTES} bytes ({size}); refusing to load"
        )
    messages: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            if entry.get('type') in _TRANSCRIPT_TYPES and entry.get('uuid'):
                u = str(entry['uuid'])
                if u not in messages:
                    order.append(u)
                messages[u] = entry
    return messages, order


def _find_latest_leaf(messages: Dict[str, Dict[str, Any]],
                      order: List[str]) -> Optional[Dict[str, Any]]:
    """Latest leaf = newest message that no other message names as parent
    (findLatestMessage over pre-computed leafUuids). Timestamp-latest wins;
    write order breaks ties."""
    referenced = {
        str(m.get('parentUuid')) for m in messages.values() if m.get('parentUuid')
    }
    leaf: Optional[Dict[str, Any]] = None
    leaf_ts = ''
    for u in order:
        m = messages[u]
        if u in referenced:
            continue
        ts = m.get('timestamp') or ''
        if leaf is None or ts >= leaf_ts:
            leaf, leaf_ts = m, ts
    return leaf


def build_conversation_chain(messages: Dict[str, Dict[str, Any]],
                             leaf: Dict[str, Any]) -> List[Dict[str, Any]]:
    """buildConversationChain: walk parentUuid leaf→root (cycle-guarded),
    return root→leaf. Orphaned-sibling recovery is unnecessary for files we
    write (linear chain — see module docstring)."""
    chain: List[Dict[str, Any]] = []
    seen: set = set()
    current: Optional[Dict[str, Any]] = leaf
    while current is not None:
        u = str(current.get('uuid'))
        if u in seen:
            from .log import log_error
            log_error(Exception(
                f"Cycle detected in parentUuid chain at message {u}. "
                f"Returning partial transcript."))
            break
        seen.add(u)
        chain.append(current)
        parent = current.get('parentUuid')
        current = messages.get(str(parent)) if parent else None
    chain.reverse()
    return chain


def restore_messages(chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """removeExtraFields + restore: convert transcript entries back into
    in-memory message dicts. Storage-only fields are stripped; passthrough
    extras (isMeta, isCompactSummary, ...) survive the round trip."""
    restored = []
    for entry in chain:
        msg = {
            k: v for k, v in entry.items()
            if k not in ('parentUuid', 'isSidechain', 'userType', 'cwd',
                         'sessionId', 'version', 'gitBranch', 'timestamp')
        }
        if 'message' in msg:
            # Image references back to base64, from the session that wrote them.
            msg['message'] = hydrate(msg['message'], entry.get('sessionId'))
        restored.append(msg)
    return restored


def load_conversation_for_resume(session_id: str) -> List[Dict[str, Any]]:
    """Load a session's conversation chain as in-memory messages
    (loadConversationForResume subset). Raises FileNotFoundError for an
    unknown id, ValueError for an empty/unreadable transcript."""
    path = get_transcript_path_for_session(session_id)
    if not path.is_file():
        raise FileNotFoundError(f"No session {session_id} in {path.parent}")
    messages, order = load_transcript_file(path)
    if not messages:
        raise ValueError(f"No messages found in {path}")
    leaf = _find_latest_leaf(messages, order)
    if leaf is None:
        raise ValueError(f"No valid conversation chain found in {path}")
    chain = build_conversation_chain(messages, leaf)
    return restore_messages(_trim_to_last_compaction(chain))


def _is_compaction_boundary(entry: Dict[str, Any]) -> bool:
    from ..services.compaction import COMPACT_CONTINUATION_PREFIX
    if entry.get("isCompactSummary"):
        return True
    if entry.get("type") != "user":
        return False
    content = (entry.get("message") or {}).get("content")
    starts = lambda t: isinstance(t, str) and t.lstrip().startswith(COMPACT_CONTINUATION_PREFIX)
    if isinstance(content, str):
        return starts(content)
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "text" and starts(b.get("text"))
            for b in content
        )
    return False


def _trim_to_last_compaction(chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop everything a compaction already replaced.

    A compaction rewrites the in-memory history: the turns it summarised are
    gone and the summary stands in for them. The transcript is append-only and
    knows nothing about that, so the chain from leaf to root still holds every
    one of those turns. Replaying it undoes the compaction — the session comes
    back at the size it had BEFORE, which is how a conversation that closed at
    207k reopened as a 458k request the server refused. One compaction is
    enough to make a session unopenable, and the bigger the session the surer.

    So resume starts where the last compaction did. Matching on the summary's
    own opening sentence rather than on a flag means the sessions already on
    disk are readable too — they were written before anything marked them.
    ``isCompactSummary`` is honoured first, for transcripts that carry it.
    """
    for i in range(len(chain) - 1, -1, -1):
        if _is_compaction_boundary(chain[i]):
            return chain[i:]
    return chain


def adopt_resumed_session(session_id: str) -> None:
    """adoptResumedSessionFile: make session_id the active session and keep
    appending to its EXISTING file. Seeds the dedup set with every uuid
    already on disk and the chain pointer with the latest leaf, so the
    restored history is not re-written and new entries chain onto it."""
    global _seen_uuids, _last_chain_uuid
    switch_session(session_id)
    path = get_transcript_path_for_session(session_id)
    if not path.is_file():
        return
    messages, order = load_transcript_file(path)
    _seen_uuids = set(messages.keys())
    leaf = _find_latest_leaf(messages, order)
    _last_chain_uuid = str(leaf['uuid']) if leaf else None


# ---------------------------------------------------------------------------
# Listing (for /resume)
# ---------------------------------------------------------------------------

def extract_first_prompt(chain: List[Dict[str, Any]]) -> str:
    """extractFirstPrompt + getFirstMeaningfulUserMessageTextContent subset:
    first non-meta, non-compact-summary user text that isn't a control
    payload; flattened and capped for display."""
    for entry in chain:
        if entry.get('type') != 'user' or entry.get('isMeta'):
            continue
        if entry.get('isCompactSummary'):
            continue
        content = (entry.get('message') or {}).get('content')
        texts: List[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text' and block.get('text'):
                    texts.append(block['text'])
        for text in texts:
            if not text:
                continue
            # Bash-mode input renders as the user typed it (original: `! cmd`)
            # — checked before the generic XML skip.
            m = re.match(r'<bash-input>(.*?)</bash-input>', text, re.DOTALL)
            if m:
                return f"! {m.group(1).strip()}"[:200]
            if _SKIP_FIRST_PROMPT_PATTERN.match(text):
                continue
            result = text.replace('\n', ' ').strip()
            if len(result) > 200:
                result = result[:200].strip() + '…'
            return result
    return 'No prompt'


def _count_visible_messages(chain: List[Dict[str, Any]]) -> int:
    """countVisibleMessages: user messages with visible (non-tool_result,
    non-meta) content + assistant messages with text content."""
    count = 0
    for entry in chain:
        etype = entry.get('type')
        content = (entry.get('message') or {}).get('content')
        if etype == 'user':
            if entry.get('isMeta'):
                continue
            if isinstance(content, str) and content.strip():
                count += 1
            elif isinstance(content, list) and any(
                isinstance(b, dict) and b.get('type') in ('text', 'image')
                for b in content
            ):
                count += 1
        elif etype == 'assistant':
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get('type') == 'text'
                and (b.get('text') or '').strip()
                for b in content
            ):
                count += 1
    return count


def list_sessions(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """fetchLogs subset: LogOption-lite dicts for every session of the current
    project, newest-modified first (sortLogs order). Unreadable/empty files
    are skipped, not fatal."""
    project_dir = get_project_dir(get_original_cwd())
    if not project_dir.is_dir():
        return []
    files = sorted(project_dir.glob('*.jsonl'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if limit is not None:
        files = files[:limit]
    sessions: List[Dict[str, Any]] = []
    for path in files:
        try:
            messages, order = load_transcript_file(path)
            if not messages:
                continue
            leaf = _find_latest_leaf(messages, order)
            if leaf is None:
                continue
            chain = build_conversation_chain(messages, leaf)
            first = chain[0]
            sessions.append({
                'session_id': leaf.get('sessionId') or path.stem,
                'full_path': str(path),
                'first_prompt': extract_first_prompt(chain),
                'message_count': _count_visible_messages(chain),
                'created': first.get('timestamp') or '',
                'modified': (chain[-1].get('timestamp') or ''),
                'git_branch': chain[-1].get('gitBranch'),
            })
        except Exception:
            continue
    return sessions
