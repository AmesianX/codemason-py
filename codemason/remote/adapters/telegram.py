"""Telegram adapter.

Chosen first because it hands us the two things a phone actually needs — a real
lock-screen push and a native client — for zero client code, and because the bot
dials *out* to Telegram, so it works from LTE with no relay and no VPN.

What it costs: cloud chats are not end-to-end encrypted and bots cannot use
Secret Chats, so anything sent here is stored in plaintext by a third party.
Hence `max_sensitivity` is INTERNAL, never SECRET — tool results, file contents
and paths do not go out this pipe. `notify` enforces that before we are called.

Stdlib urllib on purpose: a notification must never be the reason a dependency
resolves differently, and the payload is one small POST.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

from ...utils.i18n import _
from ..events import Event, EventKind, Sensitivity
from ...utils.platform import chmod_posix

API = 'https://api.telegram.org'

# Telegram hard-caps a message at 4096 characters. Leave room for the title,
# markup and the truncation notice so a long body degrades visibly.
_MAX_TEXT = 3800

# 429 handling. Telegram states exactly how long to wait, so these are only a
# ceiling for a malformed or missing retry_after.
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_MAX_WAIT = 30.0

_ICON = {
    EventKind.TURN_DONE: '✅',
    EventKind.PERMISSION_NEEDED: '🔐',
    EventKind.QUESTION_ASKED: '❓',
    EventKind.ERROR: '🔴',
    EventKind.TASK_DONE: '📦',
    EventKind.INFO: 'ℹ️',
}

# Events that should buzz the phone. Everything else arrives silently so routine
# progress does not train you to ignore the channel.
_LOUD = {EventKind.PERMISSION_NEEDED, EventKind.QUESTION_ASKED, EventKind.ERROR}


# getUpdates is exclusive: Telegram hands each update to exactly one caller, so
# two codemason processes polling at once would steal each other's button
# presses. A lock file serialises polling; whoever holds it polls, everyone else
# falls back to the local prompt rather than racing for the same update.
#
# Both files are keyed by the BOT, not just by the home directory. Exclusivity is
# a property of the token — Telegram serialises callers of one bot and does not
# care that two of them share a machine — so a single global pair of files got
# both halves of that wrong:
#
#   poll.lock   two sessions on DIFFERENT bots blocked each other for no reason.
#               Only one could poll, and the loser had no remote input at all
#               while its own bot sat idle with nobody listening.
#   offset      far worse, because it is silent. The offset is per-bot state
#               (it acknowledges bot A's updates), so bot B writing the same file
#               made each one confirm the other's messages: updates neither
#               session ever saw, gone with no error and no gap to notice.
#
# Keying by token fixes both and leaves the supported case untouched: sessions
# that DO share a bot still share its lock and its offset, which is what makes
# one of them poll and demultiplex for the rest.
#
# What this still cannot do is serialise across home directories — a WSL2 session
# and a native Windows session hold two different lock files. Telegram's 409 is
# the authority there; see poller._report_token_conflict.
_REMOTE_DIR = Path.home() / '.codemason' / 'remote'

# Pre-token-scoping paths. Read once for the offset so an upgrade does not
# re-deliver a day of history, never written.
_LEGACY_LOCK_PATH = _REMOTE_DIR / 'poll.lock'
_LEGACY_OFFSET_PATH = _REMOTE_DIR / 'offset'

# The legacy lock is checked once per process, not on every acquire: standing by
# for the lock retries every LOCK_RETRY seconds for the life of the session, and
# the answer cannot change after the first look.
_legacy_lock_checked = False


# Bot API getFile serves files up to 20 MB.
_MAX_DOWNLOAD = 20 * 1024 * 1024
# Tries for one file download (download_image).
_DOWNLOAD_ATTEMPTS = 3
# Attachments older than this are deleted when the poller starts. A session that
# Read one keeps its own copy (utils/image_store.py), so this folder is only the
# landing place between the phone and that Read.
ATTACHMENT_MAX_AGE_DAYS = 30
# Hex digits of the content hash an attachment is named by. The path reaches the
# model inside a line of text and comes back typed into a Read, and all 64 came
# back once with a digit dropped: ENOENT, then the identical Read succeeded. A
# name is only as good as the trip it survives, and 16 digits still name the
# bytes. Truncation is a bet the full hash does not make, so download_image does
# not take it: it compares and falls back to the full name.
ATTACHMENT_NAME_DIGITS = 16
# Extensions an image from the chat is saved under; anything else gets .jpg.
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}


def _worth_retrying(err: str) -> bool:
    """True when a failed Bot API call is worth sending again.

    _post reports a transport failure as str(e) and an HTTP answer as
    "HTTP <code> <desc>", so a 4xx is Telegram itself saying the request was
    wrong — the file id is unknown, the file is gone — and repeating it changes
    nothing. A 429 is already waited out inside _post. Everything else (no
    status at all, or a 5xx) is the connection, and a read-only call can go out
    again.
    """
    return not err.startswith('HTTP 4')


def _retry_read(call) -> dict:
    """Call a read-only Bot API method until it answers, pausing between tries.

    Only for methods that are safe to repeat. _post does not retry a transport
    failure, which is right for sendMessage — a request that timed out may well
    have arrived, and the phone does not need the message twice — but getFile
    only reads, and the failure that brought this here was on that call: a photo
    from the phone came back as "[could not be downloaded: error sending
    request]" while the file GET beside it, which does retry, never got its
    turn. One lost connection should cost a pause, not the photo.
    """
    last: dict = {}
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        if attempt:
            time.sleep(0.5 * attempt)
        resp = call() or {}
        if resp.get('ok') is True:
            return resp
        err = resp.get('error') or resp.get('description')
        last = resp
        if isinstance(err, str) and not _worth_retrying(err):
            break
    return last


def _image_ext(name_hint: str, remote: str) -> str:
    """Extension to save an image under: the sender's file name, else
    Telegram's path, else .jpg — and only ever one of _IMAGE_EXTS, since it
    lands in a file name and in the text the model reads."""
    for name in (name_hint, remote):
        ext = Path(name or '').suffix.lower()
        if ext in _IMAGE_EXTS:
            return ext
    return '.jpg'


def attachments_dir() -> Path:
    return _REMOTE_DIR / 'attachments'


def prune_attachments(max_age_days: float = ATTACHMENT_MAX_AGE_DAYS) -> int:
    """Delete attachments older than max_age_days; returns how many."""
    folder = attachments_dir()
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    try:
        entries = list(folder.iterdir())
    except OSError:
        return 0
    for f in entries:
        try:
            if f.is_file() and not f.is_symlink() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _ensure_remote_dir() -> Path:
    """Create ~/.codemason/remote, owner-only, and return it.

    Centralised because three call sites created it independently with a bare
    mkdir, and whichever ran first decided the mode. The lock file asks for 0600
    explicitly, but the offset and the last-message record are written with the
    default umask, so on a shared machine they landed world-readable whenever one
    of those bare mkdirs won the race against inbox._ensure_dirs. They are not
    secrets, but the directory is the one thing standing between another local
    account and everything else that ends up in here.

    chmod is advisory on Windows (it only moves the read-only bit) and the mkdir
    is what matters there; failing to tighten the mode is not worth refusing to
    start over, hence the swallow.
    """
    _REMOTE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        chmod_posix(_REMOTE_DIR, 0o700)
    except OSError:
        pass
    return _REMOTE_DIR


def token_tag(token: str) -> str:
    """Short fingerprint of a bot token, for use in a filename or a record.

    Hashed, not truncated: a bot token is a credential, and the raw prefix would
    put part of it in a path that shows up in listings, logs and error messages.
    """
    if not token:
        return 'none'
    return hashlib.sha256(token.encode('utf-8')).hexdigest()[:12]


def _token_tag() -> str:
    """Fingerprint of the bot THIS instance is using."""
    token, _ = _credentials()
    return token_tag(token)


def _lock_path() -> Path:
    return _REMOTE_DIR / f'poll.{_token_tag()}.lock'


def _last_message_path() -> Path:
    """Newest-message record, per bot.

    Keyed by token even though the file already stores a chat id, because a chat
    id does not identify a conversation across bots. In a private chat it IS the
    user's own account id, so every bot they talk to reports the same one — while
    message ids are per-chat and count independently. Two bots therefore wrote
    interleaved ids under a chat id that matched both, and the guard that reads
    it ("is my progress message still the last line?") was answered with the
    other bot's numbering: a spinner that reposts itself for no reason, or sits
    stranded above a newer message believing it is still at the bottom.
    """
    return _REMOTE_DIR / f'last_message.{_token_tag()}'


def poll_lock_holder() -> str:
    """PID currently holding this bot's poll lock, as a string, or ''.

    Exported so callers ask by name instead of rebuilding the path. The path
    depends on the configured token, and a literal copy elsewhere silently
    stops matching the moment the naming changes — which is what it did.
    """
    try:
        return _lock_path().read_text().strip()
    except OSError:
        return ''


def _offset_path() -> Path:
    return _REMOTE_DIR / f'offset.{_token_tag()}'

# Newest ordinary message in the chat, so an in-flight progress message can tell
# whether it is still the last line. Telegram message ids are per-chat and
# ascending, so "newer" is just a larger id — but only within one chat, which is
# why the record is per-bot (_last_message_path) and the chat id is checked on
# top of that.
#
# Shared through a file rather than a variable: prompts, answers and progress can
# each come from a different codemason process talking to the same chat.
_LEGACY_LAST_PATH = _REMOTE_DIR / 'last_message'


def note_chat_message(message_id) -> None:
    """Record a message that now sits below anything posted before it.

    Progress messages are deliberately *not* recorded. They are the one kind of
    message that relocates itself, so recording them would have two sessions each
    bump below the other's spinner, forever, one delete-and-repost per edit
    interval. A spinner sitting above another session's spinner is cosmetic; a
    ping-pong between them is not.
    """
    try:
        mid = int(message_id or 0)
    except (TypeError, ValueError):
        return
    if mid <= 0:
        return
    _, chat = _credentials()
    if not chat:
        return
    if mid <= newest_chat_message():
        return
    # Written through a temp file so a reader never sees a half-written id, and
    # per-pid so two processes recording at once cannot share one. Removed again
    # if the rename does not happen: `sweep` only walks the subdirectories, so a
    # stray temp file here would have nothing to ever collect it.
    path = _last_message_path()
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        _ensure_remote_dir()
        tmp.write_text(f'{chat} {mid}')
        os.replace(str(tmp), str(path))         # atomic: readers see one or the other
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def newest_chat_message() -> int:
    """The newest recorded message id for the configured chat, or 0.

    The chat id is still checked even though the file is now per-bot: the chat a
    bot talks to can be repointed (TELEGRAM_CHAT_ID changed), and the previous
    chat's numbering means nothing in the new one.
    """
    _, chat = _credentials()
    try:
        stored_chat, _, mid = _last_message_path().read_text().strip().partition(' ')
    except OSError:
        return 0
    if not chat or stored_chat != str(chat):
        return 0                               # a different chat's numbering
    try:
        return int(mid)
    except ValueError:
        return 0


def _config_paths() -> List[Path]:
    """Where credentials may live, most specific first.

    CONFIG_DIR rather than a second ~/.codemason literal: one env var
    (CODEMASON_CONFIG_DIR) is supposed to relocate the whole tree, and a hard
    coded home here would leave config.json moved and the credentials behind.
    """
    from ...utils.config import CONFIG_DIR
    return [
        CONFIG_DIR / 'remote.env',
    ]


# Four, because four is the number of windows somebody actually opens at once
# and because a lone slot reads as "one bot is the design" — the pool is the
# whole point. Blank slots cost nothing: _as_list drops them, so a scaffold
# nobody filled in stays exactly as unconfigured as no file at all.
_SCAFFOLD_SLOTS = 4


def credentials_path() -> Path:
    """The file the user is meant to edit. Named once, so messages agree."""
    return _config_paths()[0]


def ensure_credentials_scaffold() -> Optional[Path]:
    """Write a commented, empty remote.env if none exists. Returns it, or None.

    The feature is otherwise invisible: it needs a file the program has never
    created, in a directory the user has no reason to open, holding two key
    names nobody can guess. Telling them the shape in prose (as the old "put
    these in ~/.codemason/remote.env" line did) asks them to create a file from
    a description — the same instruction, minus the file.

    So write the file. Empty, so nothing is claimed to work that does not, and
    commented, so the four steps are read where they are acted on.

    Never overwrites: an existing file is the user's, including one they emptied
    on purpose. Failure is silent — a machine with an unwritable config dir has
    a worse problem than this, and remote is optional.
    """
    paths = _config_paths()
    if any(p.exists() for p in paths):
        return None
    path = paths[0]
    body = _(
        '# codemason remote channel — drive this session from Telegram.\n'
        '#\n'
        '# 1. In Telegram, talk to @BotFather and send /newbot. It replies with\n'
        '#    a token. Paste one per slot below. Blank slots are ignored.\n'
        '# 2. Every codemason you start claims one FREE bot by itself — no flag,\n'
        '#    no name to remember. Four slots means four windows at the same\n'
        '#    time, anywhere: this machine, a VM, another desktop. Want more at\n'
        '#    once? Add more lines.\n'
        '# 3. Send /start to each bot once, so it is allowed to message you.\n'
        '# 4. TELEGRAM_CHAT_ID is your own numeric id — ask @userinfobot for it.\n'
        '#\n'
        '# This file holds your bot tokens. Anyone who can read it can read your\n'
        '# alerts and type into your sessions, so keep it mode 600.\n'
    )
    body += '\n' + 'TELEGRAM_BOT_TOKEN=\n' * _SCAFFOLD_SLOTS
    body += '\nTELEGRAM_CHAT_ID=\n'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Created 600 rather than written and chmod-ed: the token slots are
        # empty now, but the user fills them in place, and a file that was ever
        # world-readable is one umask away from staying that way.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, body.encode('utf-8'))
        finally:
            os.close(fd)
    except OSError:
        return None
    return path


def credentials_file_warning() -> str:
    """Warn if the credentials file is readable by other accounts, or ''.

    remote.env holds a bot token, and a bot token is the whole credential: with
    it another local user reads every alert this channel sends, answers your
    permission prompts, and types into your sessions. The file is created by
    hand, so its mode is whatever the editor and umask happened to produce, and
    nothing has ever looked.

    Reported, not corrected. Silently chmod-ing a file the user wrote is a
    surprise, and on a single-user box the mode is harmless; saying it once is
    enough to act on. POSIX only — st_mode carries no comparable meaning on
    Windows, where the answer lives in ACLs this cannot read.
    """
    if os.name == 'nt':
        return ''
    for path in _config_paths():
        try:
            mode = path.stat().st_mode
        except OSError:
            continue
        if mode & 0o077:
            return _('{path} is readable by other users on this machine '
                     '(mode {mode:o}). It holds your bot token — chmod 600 it.',
                     path=str(path), mode=mode & 0o777)
    return ''


def _load_env_file(path: Path) -> dict:
    """Parse KEY=VALUE lines. Repeated keys collect into a list.

    Repetition matters for TELEGRAM_BOT_TOKEN: the file is a POOL of bots, and
    the natural way to write a pool is one line each. A dict would keep only the
    last, so the shape of the file has to survive parsing.
    """
    out: dict = {}
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in out:
                if isinstance(out[k], list):
                    out[k].append(v)
                else:
                    out[k] = [out[k], v]
            else:
                out[k] = v
    except Exception:
        pass
    return out


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def bot_pool() -> List[str]:
    """Every bot token available to this machine, in the order written.

    A pool rather than a single token because Telegram's getUpdates is exclusive:
    one bot can be received by exactly one process, so N instances that cannot
    see each other's filesystem need N bots. That is not a preference, it is the
    only arrangement that works when the instances are on different machines, in
    a VM, or in a container — the bot is the one rendezvous point they all share.

    Which bot an instance uses is NOT configured. It claims a free one at startup
    and asks Telegram whether one is free, since Telegram is the only thing that
    knows. Nothing to name, nothing to pass, nothing to remember.

    Environment wins, as before, so a single-token shell override still works.
    """
    env = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
    if env:
        return [env]
    for path in _config_paths():
        if not path.exists():
            continue
        tokens = _as_list(_load_env_file(path).get('TELEGRAM_BOT_TOKEN'))
        if tokens:
            return tokens
    return []


def chat_id() -> str:
    """The chat every bot in the pool talks to. One account, so one id."""
    env = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
    if env:
        return env
    for path in _config_paths():
        if path.exists():
            found = _as_list(_load_env_file(path).get('TELEGRAM_CHAT_ID'))
            if found:
                return found[0]
    return ''


def _w(s: str) -> int:
    """Display width in terminal cells — CJK glyphs occupy two.

    Padding by len() would leave a Korean column visibly ragged against an ASCII
    one, which is the whole point of aligning it.
    """
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def _collapse(event: Event, body: str) -> bool:
    """Should this body arrive in a quote box?

    A long answer goes in a blockquote, which Telegram draws boxed and in a
    smaller face, so it takes less room than the same text as a plain message.

    Boxed, not folded. It used to be an *expandable* blockquote — three lines
    and a tap to open — and that was the worse trade: a long answer still read
    fine boxed, while the fold closed again on its own as you went back and
    forth, so every look at it cost a tap. The size is enough; the fold only got
    in the way.

    If a client ever refuses the markup, _send_text and replace() both fall back
    to unformatted text, so the content still arrives.
    """
    if event.meta.get('collapse'):
        return True
    try:
        from ...utils.config import get_config
        over = int(getattr(get_config(), 'remote_collapse_over', 1200))
    except Exception:
        over = 1200
    return bool(over > 0 and len(body) > over)


def _table_width() -> int:
    """Widest grid still worth drawing, in cells. See `remote_table_width`."""
    try:
        from ...utils.config import get_config
        w = int(getattr(get_config(), 'remote_table_width', 36))
    except Exception:
        return 36
    return w if w > 0 else 10 ** 6          # 0 or negative: always a grid


def _inline(s: str) -> str:
    """The inline Markdown rules, applied to one already-escaped cell.

    Cells are lifted out before the document-wide pass runs, so without this a
    `**total**` in a table arrives with its asterisks showing.
    """
    s = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*\n]+)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?![\w*])', r'<i>\1</i>', s)
    return s


def _unmark(s: str) -> str:
    """Drop emphasis markers that would show as literal characters in <pre>."""
    s = re.sub(r'\*\*([^*\n]+)\*\*', r'\1', s)
    s = re.sub(r'`([^`\n]+)`', r'\1', s)
    return s


def _title_column(body: List[List[str]]) -> int:
    """Which column names a row, rather than merely describes it.

    Usually the first, and that was the whole rule — but not always, and the
    exception is common: a listing keyed by category (`Kind | Name | Notes`) puts
    a handful of repeated words in column one, so every heading read "Directory",
    six times over, while the filename that actually identifies the row was
    demoted to a labelled line underneath.

    Identity shows up as distinctness: the column you would look something up by
    is the one whose values do not repeat. The leftmost such column wins, and
    column 0 is the fallback when no column is distinct enough — a table where
    nothing identifies a row has no better heading to offer.
    """
    for i in range(len(body[0]) if body else 0):
        filled = [r[i].strip() for r in body if r[i].strip()]
        if len(filled) < len(body):         # gaps: not an identifier
            continue
        if len(set(filled)) >= max(2, int(len(filled) * 0.8)):
            return i
    return 0


def _reflow_table(cells: List[List[str]]) -> str:
    """A table too wide for the screen, re-laid one row at a time.

    Alignment stops paying the moment a line is wider than the phone: the client
    either wraps it, and the columns it took to build become mush, or scrolls it
    sideways, where the numbers you wanted to compare are off-screen. A row read
    vertically survives both, and the header travels with each value instead of
    sitting a scroll away at the top.
    """
    header = [c.strip() for c in cells[0]]
    body = cells[1:]
    if not body:                            # header-only: nothing to key by
        header, body = [], cells
    key = _title_column(body)

    out: List[str] = []
    for row in body:
        first = _inline(row[key].strip()) if key < len(row) else ''
        # Bold it only when it is plain text. A cell that already produced a tag
        # keeps it: Telegram rejects the whole message on markup it cannot parse,
        # and `code` in particular does not take other entities inside it.
        title = f'<b>{first}</b>' if first and '<' not in first else first
        pairs = []
        for i in range(len(row)):
            value = row[i].strip()
            if i == key or not value:       # an empty cell is not worth a line
                continue
            name = _inline(header[i]) if i < len(header) else ''
            pairs.append((name, _inline(value)))
        # One line while one line will hold it. Column names are what a heading
        # and a screen's width are spent on, and they only earn that once the
        # values are long enough to be ambiguous without them — `backend/ —
        # Directory · backend` needs no labels to be read, and a labelled block
        # would spend three lines, two of them on the words Name and Notes.
        if title and pairs:
            joined = f'{title} — ' + ' · '.join(v for _, v in pairs)
            if len(pairs) == 1 or _w(re.sub(r'<[^>]+>', '', joined)) <= _table_width():
                out.append(joined)
                continue
        lines = [title] if title else []
        lines += [f'{n}: {v}' if n else v for n, v in pairs]
        if lines:
            out.append('\n'.join(lines))
    if not out:
        return ''
    # Spacing decided at each boundary, not once for the table. A blank line
    # earns its place next to a row that is itself several lines — it says where
    # one row ends and the next begins. Between two single-line rows it says
    # nothing, and a listing where a couple of rows happen to carry a third
    # column would have every one of its ninety `key — value` lines double
    # spaced: two screens of blank to separate lines that were never ambiguous.
    text = out[0]
    for prev, row in zip(out, out[1:]):
        text += ('\n\n' if '\n' in prev or '\n' in row else '\n') + row
    # Quoted, so it still reads as one table rather than dissolving into the
    # prose around it — the grid's other job, beyond alignment, was saying where
    # it starts and ends.
    return f'<blockquote>{text}</blockquote>'


def _md_to_html(text: str) -> str:
    """Render the model's Markdown as the small HTML subset Telegram accepts.

    The body used to be wrapped in <pre>, which Telegram draws as a code block —
    hence the box and the `</>` badge around ordinary prose, with the Markdown
    left visible inside it. Prose is not code, so it is converted instead.

    Telegram's HTML is a short whitelist (b, i, u, s, code, pre, a, blockquote)
    and it rejects the whole message if the markup does not parse. So: escape
    first, then translate on the escaped text, never emit a tag that could nest
    wrongly, and let the caller retry unformatted if it is refused anyway.
    """
    out = html.escape(text)

    # Fenced blocks first: their contents must not be touched by the inline
    # rules below, so they are lifted out and restored at the end.
    blocks: List[str] = []

    def _table(m):
        rows = [r.strip() for r in m.group(1).strip().split('\n') if r.strip()]
        # Drop the |---|---| separator: it carries no information once the
        # columns are aligned, and eats a line of a small screen.
        rows = [r for r in rows if not re.fullmatch(r'\|[\s:|-]+\|', r)]
        cells = [[c.strip() for c in r.strip('|').split('|')] for r in rows]
        if not cells:
            return m.group(1)
        ncols = max(len(r) for r in cells)
        cells = [r + [''] * (ncols - len(r)) for r in cells]

        # Emphasis markers have no meaning inside <pre>; they would just print.
        grid = [[_unmark(c) for c in r] for r in cells]
        cols = [max(_w(r[i]) for r in grid) for i in range(ncols)]
        if sum(cols) + 2 * (ncols - 1) <= _table_width():
            lines = ['  '.join(c + ' ' * (cols[i] - _w(c)) for i, c in enumerate(r)).rstrip()
                     for r in grid]
            blocks.append('<pre>' + '\n'.join(lines) + '</pre>')
        else:
            blocks.append(_reflow_table(cells))
        # The match swallowed the newline that ended the last row, and without it
        # back the next paragraph starts flush against the table.
        return f'\x00{len(blocks) - 1}\x00\n'

    def _stash(m):
        lang, code = m.group(1) or '', m.group(2)
        blocks.append(f'<pre><code>{code}</code></pre>' if not lang
                      else f'<pre><code class="language-{lang}">{code}</code></pre>')
        return f'\x00{len(blocks) - 1}\x00'

    out = re.sub(r'```(\w*)\n?(.*?)```', _stash, out, flags=re.S)

    # A fence left open by truncation. Without this the lone ``` stays literal
    # and everything after it renders as loose prose — which is what a cut-off
    # answer looked like: a stray ``` followed by an unformatted directory tree.
    if out.count('```') % 2 == 1:
        head, _, rest = out.partition('```')
        rest = rest.split('\n', 1)[1] if '\n' in rest else ''
        blocks.append(f'<pre><code>{rest}</code></pre>')
        out = head + f'\x00{len(blocks) - 1}\x00'

    # Markdown tables. Telegram has no table support at all, so a table arrives
    # as raw pipes — the single most "why is nothing rendering" thing an answer
    # can contain, and answers about a codebase are full of them.
    #
    # Two renderings, chosen by width. A table that fits the screen becomes an
    # aligned monospace block, which is as close to a table as the markup gets.
    # One that does not is re-laid a row at a time (`_reflow_table`): a grid
    # wider than the phone is not a grid, it is a wrapped or half-visible one,
    # which is worse than the pipes it replaced.
    out = re.sub(r'((?:^\|.*\|[ \t]*$\n?){2,})', _table, out, flags=re.M)

    out = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', out)
    out = re.sub(r'\*\*([^*\n]+)\*\*', r'<b>\1</b>', out)
    out = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?![\w*])', r'<i>\1</i>', out)
    # [text](url) — the url is already escaped, and quotes inside it would break
    # the attribute, so anything with a quote is left as plain text.
    out = re.sub(r'\[([^\]\n]+)\]\((https?://[^)\s"\']+)\)', r'<a href="\2">\1</a>', out)
    # Headings have no equivalent; bold reads the same on a phone.
    out = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', out, flags=re.M)

    for i, block in enumerate(blocks):
        out = out.replace(f'\x00{i}\x00', block)
    return out


def _session_id() -> str:
    try:
        from ...utils.state import get_session_id
        return get_session_id()
    except Exception:
        return ''


# The bot this process claimed, once it has one. Module-level because every
# caller in this file asks _credentials() and none of them should have to know
# how the choice was made.
_claimed_token: Optional[str] = None


def claimed_token() -> str:
    return _claimed_token or ''


def _credentials() -> tuple:
    """(token, chat_id) for THIS instance.

    Before a claim, the first bot in the pool — enough for is_configured() and
    for outbound, which is not exclusive. After a claim, the bot this instance
    owns. There is nothing to configure and nothing to name: which bot an
    instance uses is decided by asking Telegram which one is free.
    """
    pool = bot_pool()
    token = _claimed_token or (pool[0] if pool else '')
    return token, chat_id()


# How long a probe holds its poll open while deciding whether a bot is taken.
#
# Derived, not picked. Probing cannot work by asking once: Telegram answers 409
# to the poll it TERMINATES, so a single request against a busy bot returns 200
# to the prober and hands the 409 to the incumbent — the opposite of what the
# prober needs to learn. What it must do instead is hold the connection and wait
# for the incumbent to come back, because the incumbent WILL come back: it was
# just bumped, and its loop retries after RETRY_BASE seconds. When it does, our
# poll is the one waiting, so the 409 arrives here and the answer is unambiguous.
#
# So the window has to outlast one incumbent backoff plus a round trip. Anything
# shorter reports a busy bot as free; anything longer only makes startup slower.
_PROBE_SECONDS = 6


# Probe verdicts. Three outcomes, not two, because "cannot use this bot" has two
# completely different causes and a boolean forced them together — with the wrong
# answer both times. A network outage made every bot look taken, so the user was
# told to go and create more at BotFather when nothing was wrong with the pool;
# and a revoked token answered anything-but-409, so it looked FREE and got
# claimed, leaving a session whose channel silently does not work. Neither is a
# rare case: one is a dropped wifi, the other is a token deleted in BotFather.
FREE = 'free'
TAKEN = 'taken'
INVALID = 'invalid'                     # Telegram rejects the token itself
UNREACHABLE = 'unreachable'             # we could not ask


def probe_bot(token: str) -> str:
    """FREE / TAKEN / INVALID / UNREACHABLE for one bot.

    A dead process cannot make this answer wrong. Its socket may still be open,
    but it will never issue another request, and 409 is only produced by a NEW
    request arriving — measured: a process killed mid-getUpdates leaves polls
    that answer 200.
    """
    payload = urllib.parse.urlencode({
        'offset': 0,                    # peek only; the offset file is untouched
        'timeout': _PROBE_SECONDS,
        'allowed_updates': json.dumps([]),
    }).encode()
    req = urllib.request.Request(f'{API}/bot{token}/getUpdates', data=payload)
    try:
        with urllib.request.urlopen(req, timeout=_PROBE_SECONDS + 15) as r:
            r.read()
        return FREE
    except urllib.error.HTTPError as e:
        try:
            e.read()
        except Exception:
            pass
        if e.code == 409:
            return TAKEN                # an instance is alive on this bot
        if e.code in (401, 404):
            return INVALID              # revoked, mistyped, or never existed
        return UNREACHABLE              # 429, 5xx — true later, maybe, not now
    except Exception:
        return UNREACHABLE


def _is_bot_free(token: str) -> bool:
    """Back-compat shim. Prefer probe_bot, which can say WHY not."""
    return probe_bot(token) == FREE


def claim_bot(preferred: str = '') -> Optional[str]:
    """Take a free bot from the pool, or None when every one is in use.

    `preferred` is tried first — the bot this directory used last time. Reclaiming
    it costs one probe instead of scanning, and it keeps a window talking to the
    same chat across restarts, which is the difference between a stable list of
    sessions on the phone and a shuffle.
    """
    global _claimed_token, _last_claim_verdicts
    pool = bot_pool()
    _last_claim_verdicts = {}
    if not pool:
        return None
    order = ([preferred] + [t for t in pool if t != preferred]) if preferred in pool else pool
    for token in order:
        verdict = probe_bot(token)
        _last_claim_verdicts[token_tag(token)] = verdict
        if verdict == FREE:
            _claimed_token = token
            return token
    return None


# Why the last claim failed, per bot. Kept so the report can name the actual
# situation instead of assuming the pool is full — the three ways to fail need
# three different things done about them.
_last_claim_verdicts: dict = {}


def last_claim_verdicts() -> dict:
    return dict(_last_claim_verdicts)


def release_bot() -> None:
    global _claimed_token
    _claimed_token = None


_handle_cache: dict = {}


def claimed_handle() -> str:
    """@username of the claimed bot, cached. '' when unknown.

    The handle, not the name: the name is a label this instance rewrites, the
    handle is fixed at creation and is what identifies the chat unambiguously.
    """
    token, _ = _credentials()
    if not token:
        return ''
    tag = token_tag(token)
    if tag in _handle_cache:
        return _handle_cache[tag]
    try:
        req = urllib.request.Request(f'{API}/bot{token}/getMe',
                                     data=urllib.parse.urlencode({}).encode())
        with urllib.request.urlopen(req, timeout=10) as r:
            me = json.loads(r.read().decode('utf-8', 'replace'))
        handle = (me.get('result') or {}).get('username', '') if me.get('ok') else ''
    except Exception:
        handle = ''
    _handle_cache[tag] = handle
    return handle




def describe_claimed_bot() -> str:
    """'Name (@handle)' for the claimed bot, or '' — for /remote status."""
    token, _ = _credentials()
    if not token:
        return ''
    try:
        req = urllib.request.Request(f'{API}/bot{token}/getMe',
                                     data=urllib.parse.urlencode({}).encode())
        with urllib.request.urlopen(req, timeout=10) as r:
            me = json.loads(r.read().decode('utf-8', 'replace'))
        if not me.get('ok'):
            return ''
        res = me['result']
        name = res.get('first_name') or ''
        handle = res.get('username') or ''
        return f'{name} (@{handle})' if handle else name
    except Exception:
        return ''


def get_bot_name() -> str:
    """The claimed bot's current display name, or ''."""
    token, _ = _credentials()
    if not token:
        return ''
    req = urllib.request.Request(f'{API}/bot{token}/getMyName',
                                 data=urllib.parse.urlencode({}).encode())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = json.loads(r.read().decode('utf-8', 'replace'))
        return (body.get('result') or {}).get('name', '') if body.get('ok') else ''
    except Exception:
        return ''


def set_bot_name(name: str) -> tuple:
    """Rename the claimed bot. Returns (ok, reason) — reason is '' on success.

    Reads before writing, and that is not an optimisation. setMyName is not
    rate-limited per burst but per BOT, on a cooldown measured in hours: a bot
    renamed a handful of times answers 429 with a retry_after around 22 hours,
    and no amount of waiting a few seconds helps. Measured, after an earlier
    version of this code spent a bot's whole quota by renaming on every claim —
    which meant restarting a session a few times left you unable to name it at
    all, for the rest of the day.

    So the common case must cost nothing. The name lives on Telegram's side and
    survives restarts, so a session coming back up in the same directory already
    has the name it wants, and asking is free while setting is not.

    Cosmetic either way: the name is a label, the token is the wiring. A refused
    rename must never stop a session from running.
    """
    # NOT `token, _ =` — this function calls the i18n `_()`, and unpacking into
    # the same name shadows it, so the first translated string raises
    # "'str' object is not callable" from a line that looks fine.
    token, _chat = _credentials()
    if not token:
        return False, _('no bot claimed')
    wanted = (name or '')[:64]
    if get_bot_name() == wanted:
        return True, ''                 # already right; spend nothing
    data = urllib.parse.urlencode({'name': wanted}).encode()
    req = urllib.request.Request(f'{API}/bot{token}/setMyName', data=data)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = json.loads(r.read().decode('utf-8', 'replace'))
        if body.get('ok'):
            return True, ''
        return False, str(body.get('description', ''))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode('utf-8', 'replace'))
        except Exception:
            body = {}
        if e.code == 429:
            wait = (body.get('parameters') or {}).get('retry_after')
            if wait:
                hours = int(wait) / 3600.0
                return False, _('Telegram limits how often a bot may be renamed; '
                                'this one can be renamed again in about '
                                '{hours:.0f}h.', hours=max(1.0, hours))
            return False, _('renamed too often recently')
        return False, str(body.get('description', f'HTTP {e.code}'))
    except Exception as exc:
        return False, str(exc)


class TelegramAdapter:
    name = 'telegram'

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    @property
    def max_sensitivity(self) -> Sensitivity:
        # INTERNAL, not SECRET — see module docstring.
        return Sensitivity.INTERNAL

    def is_configured(self) -> bool:
        token, chat = _credentials()
        return bool(token and chat)

    # --- rendering -----------------------------------------------------------

    def _format(self, event: Event) -> str:
        icon = _ICON.get(event.kind, '•')
        parts = [f'{icon} <b>{html.escape(event.title)}</b>']

        # Skip the directory line when the title already names it — otherwise the
        # same word lands on two consecutive lines and says nothing twice.
        if event.cwd:
            from .. import inbox as _inbox
            where = _inbox.short_label(event.cwd)
            if where and where.lower() not in event.title.lower():
                parts.append(f'<code>{html.escape(where)}</code>')

        if event.body:
            body = event.body
            if len(body) > _MAX_TEXT:
                # Say so rather than silently cutting — a truncated alert must
                # not read like the whole story.
                body = body[:_MAX_TEXT] + f'\n… (+{len(event.body) - _MAX_TEXT} chars truncated)'
            parts.append('')
            rendered = _md_to_html(body)
            # Box after the conversion, never in the body: _md_to_html escapes
            # what it is given, so a tag put there arrives as visible text.
            if _collapse(event, body):
                rendered = f'<blockquote>{rendered}</blockquote>'
            parts.append(rendered)

        # The collapsed-tool-list equivalent: what ran, how often, nothing more.
        tools = event.meta.get('tools')
        if isinstance(tools, dict) and tools:
            summary = ' · '.join(
                f'{html.escape(str(name))}{"" if n <= 1 else f" ×{n}"}'
                for name, n in sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            parts.append('')
            parts.append(summary)

        tool = event.meta.get('tool')
        if tool:
            parts.append(f'tool: <code>{html.escape(str(tool))}</code>')
        dur = event.meta.get('duration_ms')
        status = event.meta.get('status')
        if isinstance(dur, (int, float)) and dur > 0:
            # Context fill and images ride on the same line as the duration.
            parts.append(f'{dur / 1000:.1f}s' + (f' · {html.escape(status)}' if status else ''))
        elif status:
            parts.append(html.escape(status))

        return '\n'.join(parts)

    # --- transport -----------------------------------------------------------

    def _post(self, method: str, payload: dict, *, http_timeout: Optional[float] = None) -> Optional[dict]:
        """`http_timeout` overrides the socket deadline for this call.

        Long-polling needs it: getUpdates deliberately holds the connection open
        for its `timeout` seconds, so a socket deadline shorter than that aborts
        the read every time and looks exactly like an API failure. That silently
        killed the permission poll — the buttons went up and collapsed again on
        the next tick.
        """
        token, _ = _credentials()
        if not token:
            return None
        data = urllib.parse.urlencode(payload).encode()

        # Telegram throttles a single chat to roughly one message a second, so a
        # handful of sessions finishing together is enough to hit it. A 429 comes
        # with the exact wait, and honouring it costs a pause; ignoring it drops
        # the alert with nothing to show for it — which is the failure this whole
        # channel exists to avoid.
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            req = urllib.request.Request(f'{API}/bot{token}/{method}', data=data)
            try:
                with urllib.request.urlopen(req, timeout=http_timeout or self.timeout) as resp:
                    return json.loads(resp.read().decode('utf-8', 'replace'))
            except urllib.error.HTTPError as e:
                # Surface the API's own reason (bad chat_id, revoked token)
                # instead of a bare status — this is the failure people hit.
                body = {}
                try:
                    body = json.loads(e.read().decode('utf-8', 'replace'))
                except Exception:
                    pass
                if e.code == 429 and attempt < _RATE_LIMIT_RETRIES:
                    wait = (body.get('parameters') or {}).get('retry_after')
                    time.sleep(min(_RATE_LIMIT_MAX_WAIT,
                                   float(wait) if wait else 2 ** attempt))
                    continue
                return {'ok': False,
                        'error': f'HTTP {e.code} {body.get("description", "")}'.strip()}
            except Exception as e:
                return {'ok': False, 'error': str(e)}
        return {'ok': False, 'error': 'rate limited'}

    def _send_text(self, chat: str, text: str, *, silent: bool,
                   plain_fallback: str = '') -> Optional[dict]:
        """Post text as HTML, falling back to unformatted on a parse rejection.

        Telegram refuses the entire message if the markup does not parse, so a
        single bad conversion would lose the content rather than render it badly.
        Losing the alert is the one outcome worth avoiding; ugly is fine.
        """
        resp = self._post('sendMessage', {
            'chat_id': chat, 'text': text, 'parse_mode': 'HTML',
            'disable_web_page_preview': 'true',
            'disable_notification': 'false' if silent is False else 'true',
        })
        if resp and resp.get('ok'):
            return resp
        if 'parse' in str((resp or {}).get('error', '')).lower():
            from ...utils.log import debug_log
            debug_log('REMOTE_HTML_REJECTED', {'error': (resp or {}).get('error')})
            return self._post('sendMessage', {
                'chat_id': chat,
                'text': plain_fallback or re.sub(r'<[^>]+>', '', text),
                'disable_web_page_preview': 'true',
                'disable_notification': 'false' if silent is False else 'true',
            })
        return resp

    def send(self, event: Event) -> Optional[str]:
        token, chat = _credentials()
        if not (token and chat):
            return None
        split = self._split_for_file(event)
        if split:
            event = split[0]
        resp = self._send_text(chat, self._format(event),
                               silent=event.kind not in _LOUD)
        if not resp or not resp.get('ok'):
            from ...utils.log import debug_log
            debug_log('REMOTE_NOTIFY_FAILED', {
                'adapter': self.name,
                'error': (resp or {}).get('error', 'no response'),
            })
            return None
        mid = (resp.get('result') or {}).get('message_id')
        note_chat_message(mid)
        if split:
            self.send_document(split[1], split[2])
        # Remember who posted it, so replying to this alert routes back here
        # instead of going through a picker.
        if mid:
            try:
                from .. import inbox as _inbox
                _inbox.claim_message(int(mid), event.session_id or _session_id())
            except Exception:
                pass
        return str(mid or '') or None

    def progress_is_last(self, message_id: int) -> bool:
        """True while nothing else has been posted below the progress message."""
        newest = newest_chat_message()
        return not newest or newest < int(message_id)

    def delete_message(self, message_id: int) -> bool:
        resp = self._post('deleteMessage', {'chat_id': _credentials()[1],
                                            'message_id': message_id})
        return bool(resp and resp.get('ok'))

    def edit_progress(self, message_id: Optional[int], body: str,
                      cwd: str = '') -> Optional[int]:
        """Create or update the single in-progress message for a turn.

        Plain text, not HTML: this shows partial output, which can end
        mid-tag-like-token — `<b` with no close, half a code fence. Telegram
        rejects the whole edit when markup does not parse, so formatting a moving
        target would make the progress vanish exactly when it is doing its job.
        The final answer is formatted; this is a status line.

        Editing alone is not enough to keep it *current*: an edit rewrites the
        message where it already sits. Anything posted meanwhile — a permission
        prompt above all, since answering one is exactly what happens mid-turn —
        lands below it and strands the progress, and with it the answer that
        eventually replaces it, in the middle of the chat. So when that has
        happened the message is deleted and reposted at the bottom. Reposting
        costs no chat line, and the alternative is the thing you are waiting for
        appearing above the thing you just tapped.
        """
        token, chat = _credentials()
        if not (token and chat):
            return None
        where = ''
        if cwd:
            from .. import inbox as _inbox
            where = f'{_inbox.short_label(cwd)}\n'
        text = f'{where}{body}'
        if message_id is not None and not self.progress_is_last(message_id):
            self.delete_message(message_id)
            message_id = None                  # repost below whatever arrived
        if message_id is None:
            resp = self._post('sendMessage', {
                'chat_id': chat, 'text': text,
                'disable_notification': 'true',   # motion, not an alert
                'disable_web_page_preview': 'true',
            })
        else:
            resp = self._post('editMessageText', {
                'chat_id': chat, 'message_id': message_id, 'text': text,
                'disable_web_page_preview': 'true',
            })
        if not resp or not resp.get('ok'):
            return message_id
        return (resp.get('result') or {}).get('message_id') or message_id

    def download_image(self, file_id: str, name_hint: str = '') -> tuple:
        """Fetch an image the chat sent. Returns (path, error) — one of them empty.

        getFile resolves the id to a path on Telegram's file server, valid for an
        hour; the bytes then come from /file/bot<token>/<path>. Saved under
        ~/.codemason/remote/attachments (owner-only, like the rest of remote/)
        by content hash, so the same photo twice is one file. The session that
        receives the message Reads it from there, and that Read is what puts a
        copy in the session's own image store. Bot API downloads stop at 20 MB.
        """
        import hashlib
        token, _chat = _credentials()
        if not token:
            return '', 'no bot token'
        resp = _retry_read(lambda: self._post('getFile', {'file_id': file_id}))
        remote = (resp.get('result') or {}).get('file_path') if resp.get('ok') else None
        if not remote:
            why = resp.get('error') or resp.get('description') or 'getFile failed'
            from ...utils.log import log_error
            log_error(f'telegram getFile failed: {why}', 'remote: download_image')
            return '', why
        # A GET is safe to repeat, and the first failure seen in use was a bare
        # transport error on a WSL2 desktop right after getFile had worked: a
        # dropped keep-alive or a route that failed once look the same from
        # here. Retry with a short pause. A 4xx is final (the file is gone or
        # the path is wrong) except 429, which says to come back later.
        data = None
        last = ''
        url = f'{API}/file/bot{token}/{urllib.parse.quote(remote)}'
        for attempt in range(_DOWNLOAD_ATTEMPTS):
            if attempt:
                time.sleep(0.5 * attempt)
            try:
                with urllib.request.urlopen(url, timeout=max(self.timeout, 60)) as r:
                    data = r.read(_MAX_DOWNLOAD + 1)
                break
            except urllib.error.HTTPError as e:
                last = f'HTTP {e.code}'
                if 400 <= e.code < 500 and e.code != 429:
                    break
            except Exception as e:
                # urllib's messages normally carry no URL, but an InvalidURL-type
                # error quotes it — and the URL holds the token.
                last = (str(e) or type(e).__name__).replace(token, '<token>')
        if data is None:
            from ...utils.log import log_error
            log_error(f'telegram photo download failed: {last}', 'remote: download_image')
            return '', last
        if len(data) > _MAX_DOWNLOAD:
            return '', f'larger than {_MAX_DOWNLOAD // (1024 * 1024)} MB'
        ext = _image_ext(name_hint, remote)
        folder = _ensure_remote_dir() / 'attachments'
        folder.mkdir(exist_ok=True)
        try:
            chmod_posix(folder, 0o700)
        except OSError:
            pass
        sha = hashlib.sha256(data).hexdigest()
        path = folder / f'{sha[:ATTACHMENT_NAME_DIGITS]}{ext}'
        if path.exists() and path.read_bytes() != data:
            # Two different photos under one short name: ~1e-13 by chance, but
            # cheap to force (a 2**32 birthday search), and the cost would be a
            # Read quietly answering with the other photo. The full name cannot
            # clash, so the rare one wears it.
            path = folder / f'{sha}{ext}'
        if not path.exists():
            tmp = path.with_suffix(f'{ext}.{os.urandom(8).hex()}.tmp')
            tmp.write_bytes(data)
            tmp.replace(path)
        return str(path), ''

    def send_document(self, content: str, filename: str, caption: str = '') -> Optional[int]:
        """Upload text as a file. Returns the message id, or None.

        multipart/form-data assembled by hand for the same reason the rest of
        this module uses urllib: a notification path must never be why a
        dependency resolves differently. The payload is one small upload.
        """
        token, chat = _credentials()
        if not (token and chat):
            return None
        boundary = '----codemason' + os.urandom(8).hex()
        # BOM. The upload was UTF-8 all along and still arrived as mojibake: an
        # extension tells a viewer nothing about encoding, so it falls back to the
        # system default (CP949 on Windows, Latin-1 in some iOS viewers). Three
        # bytes remove the guess.
        data = b'\xef\xbb\xbf' + content.encode('utf-8')
        parts = []

        def field(name, value):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())

        field('chat_id', chat)
        if caption:
            # Telegram caps a caption at 1024 chars; the file carries the rest.
            field('caption', caption[:1000])
            field('parse_mode', 'HTML')
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="document"; '
            f'filename="{filename}"\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n'.encode()
            + data + b'\r\n')
        parts.append(f'--{boundary}--\r\n'.encode())
        body = b''.join(parts)

        req = urllib.request.Request(
            f'{API}/bot{token}/sendDocument', data=body,
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        try:
            with urllib.request.urlopen(req, timeout=max(self.timeout, 60)) as r:
                resp = json.loads(r.read().decode('utf-8', 'replace'))
        except Exception as e:
            from ...utils.log import debug_log
            debug_log('REMOTE_DOCUMENT_FAILED', {'error': str(e)})
            return None
        if not resp.get('ok'):
            from ...utils.log import debug_log
            debug_log('REMOTE_DOCUMENT_FAILED', {'error': resp.get('description')})
            return None
        mid = (resp.get('result') or {}).get('message_id')
        note_chat_message(mid)
        return mid

    def _file_threshold(self) -> int:
        try:
            from ...utils.config import get_config
            return int(getattr(get_config(), 'remote_file_threshold', 3000))
        except Exception:
            return 3000

    def _split_for_file(self, event: Event):
        """Long body -> (chat stand-in, full text, filename), else None.

        Deliberately does NOT upload. The upload happens only after the chat
        message has landed, because the caller may fail and fall back to posting
        a fresh message — and an upload done up front would then run twice,
        leaving two identical attachments for one answer.
        """
        limit = self._file_threshold()
        body = event.body or ''
        if limit <= 0 or len(body) <= limit:
            return None
        import time as _t
        try:
            from ...utils.config import get_config
            ext = str(getattr(get_config(), 'remote_file_ext', '.md')) or '.md'
            excerpt = int(getattr(get_config(), 'remote_excerpt', 3000))
        except Exception:
            ext, excerpt = '.md', 3000
        if not ext.startswith('.'):
            ext = '.' + ext
        name = f"codemason-{_t.strftime('%Y%m%d-%H%M%S')}{ext}"
        head = body[:max(200, excerpt)].rstrip()
        stand_in = Event(
            kind=event.kind,
            title=event.title,
            body=head + '\n\n' + _('… the full {chars:,} characters were sent as `{name}`.',
                                    chars=len(body), name=name),
            sensitivity=event.sensitivity,
            session_id=event.session_id,
            cwd=event.cwd,
            meta={**event.meta, 'collapse': True},
        )
        return stand_in, body, name

    def replace(self, message_id: int, event: Event) -> bool:
        """Turn the progress message into the finished alert.

        From here on it is an ordinary message: the turn is over, it will never
        move again, and a later spinner should place itself below it.
        """
        token, chat = _credentials()
        if not (token and chat):
            return False
        # Long enough for a file: this message finishes as an excerpt plus a
        # pointer. An edit cannot become an attachment, so the file follows it —
        # after the edit lands, never before, or a failed edit would upload once
        # here and once again on the caller's fallback path.
        split = self._split_for_file(event)
        if split:
            event = split[0]
        resp = self._post('editMessageText', {
            'chat_id': chat, 'message_id': message_id,
            'text': self._format(event), 'parse_mode': 'HTML',
            'disable_web_page_preview': 'true',
        })
        if resp and resp.get('ok'):
            note_chat_message(message_id)
            if split:
                self.send_document(split[1], split[2])
            return True
        # Formatting failed on the finished text — keep the content, drop the
        # markup, rather than leaving a stale "working…" as the last thing said.
        resp = self._post('editMessageText', {
            'chat_id': chat, 'message_id': message_id,
            'text': re.sub(r'<[^>]+>', '', self._format(event)),
            'disable_web_page_preview': 'true',
        })
        if resp and resp.get('ok'):
            note_chat_message(message_id)
            if split:
                self.send_document(split[1], split[2])
            return True
        return False

    def react(self, message_id: int, emoji: str) -> None:
        """Put a reaction on a message the user sent.

        The acknowledgement problem: a turn can run for a minute or more before
        it has anything to say, and until then the phone shows nothing at all —
        indistinguishable from the message never arriving. A reply saying "got
        it" would fix that and double the message count on every exchange.

        A reaction costs no message. It appears on the line you just sent, which
        is exactly where you are looking. Best-effort: older Bot API versions
        reject setMessageReaction, and an unacknowledged message is a far smaller
        problem than a failed turn.
        """
        token, chat = _credentials()
        if not (token and chat and message_id):
            return
        self._post('setMessageReaction', {
            'chat_id': chat,
            'message_id': message_id,
            'reaction': json.dumps([{'type': 'emoji', 'emoji': emoji}]),
        })

    # --- session picker ------------------------------------------------------

    def pick(self, title: str, preview: str, labels: List[str]) -> Optional[int]:
        """Ask which session a message is for. Returns the picker's message id.

        One button per session, stacked one per row: session labels are directory
        names and go long, and a wrapped row of them is unreadable on a phone.
        """
        token, chat = _credentials()
        if not (token and chat):
            return None
        keyboard = [[{'text': l, 'callback_data': f'r:{i}'}] for i, l in enumerate(labels)]
        text = f'🔀 <b>{html.escape(title)}</b>'
        if preview:
            text += f'\n\n<pre>{html.escape(preview)}</pre>'
        resp = self._post('sendMessage', {
            'chat_id': chat,
            'text': text,
            'parse_mode': 'HTML',
            'reply_markup': json.dumps({'inline_keyboard': keyboard}),
        })
        if not resp or not resp.get('ok'):
            return None
        mid = (resp.get('result') or {}).get('message_id')
        note_chat_message(mid)
        return mid

    def note_routed(self, picker_id: int, session_id: str) -> None:
        """Collapse the picker once it has been answered.

        An empty session id is the Discard choice on a held-over message. It has
        to read differently from a route: saying "Routed -> <nothing>" about a
        message you just chose to throw away is the kind of wrong that makes you
        go looking for where it went.
        """
        token, chat = _credentials()
        if not (token and chat):
            return
        if not session_id:
            self._post('editMessageText', {
                'chat_id': chat,
                'message_id': picker_id,
                'text': '🗑 <b>' + _('Discarded') + '</b>',
                'parse_mode': 'HTML',
            })
            return
        try:
            from .. import inbox as _inbox
            label = next((s.get('label') or session_id[:8]
                          for s in _inbox.live_sessions()
                          if s['session_id'] == session_id), session_id[:8])
        except Exception:
            label = session_id[:8]
        self._post('editMessageText', {
            'chat_id': chat,
            'message_id': picker_id,
            'text': '🔀 <b>' + _('Routed') + f'</b> -> <code>{html.escape(str(label))}</code>',
            'parse_mode': 'HTML',
        })

    # --- permission round-trip ----------------------------------------------

    # Bound on stale-lock reclaim attempts. Reclaiming is a race — two sessions
    # can each find the same dead holder, and each unlink and retry — so the retry
    # has to be finite. It used to recurse into itself with no limit, which under
    # a persistent race is unbounded stack growth in a daemon thread; failing to
    # get the lock is harmless (this session simply has no remote input), so a
    # small bound is strictly better than the risk.
    _LOCK_RECLAIM_TRIES = 3

    def _retire_legacy_lock(self) -> None:
        """Remove the pre-token-scoping lock file once its holder is gone.

        Nothing reads it any more, so left alone it is permanent litter in a
        directory the sweep does not walk. Only removed when the PID inside it is
        dead: an older codemason may still be running and still relying on it, and
        deleting a lock somebody holds is how two pollers happen.
        """
        global _legacy_lock_checked
        if _legacy_lock_checked:
            return
        _legacy_lock_checked = True
        try:
            holder = int(_LEGACY_LOCK_PATH.read_text().strip() or 0)
        except (OSError, ValueError):
            return
        from ...utils.platform import pid_alive
        if holder and pid_alive(holder):
            return
        try:
            _LEGACY_LOCK_PATH.unlink()
        except OSError:
            pass

    def _acquire_poll_lock(self):
        """Take the inbound poll lock, or None if someone else is polling.

        On unix this is a real flock held for as long as the returned fd is
        open; the PID written into the body is a label for the standby message
        and for _owns_poll_lock, never the exclusion mechanism.

        It used to be exclusive-create plus a PID body, and that admitted two
        pollers on the same bot — the 409-conflict condition this lock exists to
        prevent. The sequence: A opens, reads a dead PID, decides to reclaim; B
        does the same; B unlinks and creates its own lock; A's unlink then
        deletes B's *live* lock and A creates on top. Both now believe they hold
        it, both call getUpdates, and Telegram starts 409ing whichever asked
        second — which the poller reads as "a live rival" and answers by giving
        the bot away. Every step of that is a separate syscall with a window
        after it, so no amount of re-reading closes it; the decision has to be
        one atomic operation. flock is that operation, and it also deletes the
        whole stale-reclaim problem: the kernel drops a dead process's flock
        whether it exited, crashed or was killed, so there is nothing to
        heuristically reclaim and no pid-reuse hazard to get wrong.
        """
        self._retire_legacy_lock()
        path = _lock_path()
        _ensure_remote_dir()
        if os.name == 'posix':
            import fcntl
            # O_CREAT, never O_EXCL: the file persists across holders and is only
            # ever a carrier for the lock. It is deliberately NOT unlinked on
            # release — unlinking a flocked path is how you get two holders (a
            # waiter locks the orphaned inode while the next arrival creates and
            # locks a fresh one), so the name stays and the body is cleared.
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
            except OSError:
                return None
            try:
                # Non-blocking: this is called from _await_lock, which does its
                # own paced retry and reports standby. Blocking here would
                # swallow the stop event.
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                return None                 # held by a live poller, or no flock
            # Ours exclusively from here. Truncate first: a shorter pid written
            # over a longer one leaves the previous holder's trailing digits and
            # the standby message then names a process that never held it.
            try:
                os.ftruncate(fd, 0)
                os.write(fd, str(os.getpid()).encode())
            except OSError:
                pass
            return fd
        # No flock here, so the PID file stays — but hardened. Windows does most
        # of the work: the lock is opened without FILE_SHARE_DELETE, so removing
        # a file a LIVE holder still has open simply fails, and the
        # A-deletes-B's-lock step above cannot happen while B is alive. What
        # remains is the narrow window where B has created its lock but not yet
        # written its pid, so re-read immediately before unlinking and only
        # reclaim a body that has not changed since it was judged dead, then
        # re-validate that the lock taken still names us.
        for _attempt in range(self._LOCK_RECLAIM_TRIES):
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(fd, str(os.getpid()).encode())
                if not self._owns_poll_lock():
                    # Somebody reclaimed and rewrote it under us. Not ours — drop
                    # the handle and stand by rather than poll alongside them.
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    return None
                return fd
            except FileExistsError:
                def _read_holder():
                    try:
                        return int(path.read_text().strip() or 0)
                    except (OSError, ValueError):
                        return 0
                holder = _read_holder()
                # os.kill(pid, 0) terminates the process on Windows, and the
                # process here is another codemason session. Portable check.
                from ...utils.platform import pid_alive
                if holder and pid_alive(holder):
                    return None             # genuinely held
                # The body must still say what it said when we decided, or
                # somebody is mid-acquire and we would be deleting a live lock.
                if _read_holder() != holder:
                    return None
                try:
                    path.unlink()
                except OSError:
                    return None
                continue                    # reclaimed; try to take it ourselves
            except OSError:
                return None
        return None

    def holds_poll_lock(self) -> bool:
        """Public form of _owns_poll_lock, for callers that must wait it out.

        permissions._yield_poll_lock asks this instead of sleeping a guessed
        interval and hoping the lock has gone by then.
        """
        return self._owns_poll_lock()

    def _owns_poll_lock(self) -> bool:
        """True when the lock file on disk still names THIS process."""
        try:
            return int(_lock_path().read_text().strip() or 0) == os.getpid()
        except (OSError, ValueError):
            return False

    def _release_poll_lock(self, fd) -> None:
        """Close our handle and remove the lock — but only if it is still ours.

        The unlink used to be unconditional, which makes releasing a lock able to
        delete somebody else's. A holder that is briefly unresponsive (a stopped
        process, a PID check that answers wrongly) gets its file reclaimed by the
        next session, and when the original then releases, it deletes the NEW
        holder's lock. Two pollers on one bot follow, which is the exact failure
        the lock exists to prevent — and the one that is invisible from inside.
        """
        if fd is None:
            return                          # nothing was held
        if os.name == 'posix':
            # Clear the body rather than unlinking: the flock lives on the inode,
            # and unlinking a flocked path lets a waiter hold the orphan while the
            # next arrival creates and locks a fresh file — two holders, which is
            # what the lock exists to prevent. Closing the fd drops the flock.
            if self._owns_poll_lock():
                try:
                    os.ftruncate(fd, 0)
                except OSError:
                    pass
            try:
                os.close(fd)
            except OSError:
                pass
            return
        try:
            os.close(fd)
        except OSError:
            pass
        if not self._owns_poll_lock():
            return
        try:
            _lock_path().unlink()
        except OSError:
            pass

    def _is_trusted_chat(self, msg: dict, sender: Optional[dict] = None) -> bool:
        """True only for the chat we are configured to talk to, from a sender we
        accept.

        A bot can be added to any chat by anyone who knows its handle, and every
        one of those chats reaches the same getUpdates stream. Without this check
        a stranger's message would be treated as your input and their button tap
        as your approval.

        The chat id alone is not enough. If the configured chat is a GROUP, the
        id check is a tautology — the bot posted the prompt there — and every
        current and future member of that group could approve tool use and
        inject input. The sender was never examined, so the whole boundary
        rested on a deployment choice (use a personal chat) that nothing
        enforced.

        Default with no configuration: accept only a sender whose id equals the
        configured chat id. In a 1:1 chat Telegram makes those the same number,
        so the intended setup keeps working untouched and a group needs the
        allowlist below stated explicitly.

        TELEGRAM_ALLOWED_USER_IDS: comma-separated numeric ids, for a group or a
        second device. Setting it replaces the default rule entirely.
        """
        _, chat = _credentials()
        if not chat or str((msg.get('chat') or {}).get('id')) != str(chat):
            return False

        if sender is None:
            sender = msg.get('from') or {}
        sender_id = str((sender or {}).get('id') or '')

        allowed = [
            s.strip() for s in (os.environ.get('TELEGRAM_ALLOWED_USER_IDS') or '').split(',')
            if s.strip()
        ]
        if allowed:
            return sender_id in allowed
        if not sender_id:
            # No sender on the update (channel post, service message). Nothing
            # to authenticate, so it is not input from the operator.
            return False
        return sender_id == str(chat)

    def _read_offset(self) -> int:
        try:
            return int(_offset_path().read_text().strip() or 0)
        except Exception:
            pass
        # First run after the offset became token-scoped. Without this the new
        # file is missing, the offset reads 0, and Telegram replays everything it
        # has held for up to 24h — an upgrade would deliver a day of stale
        # instructions. Read once; the next _write_offset lands on the new path.
        try:
            return int(_LEGACY_OFFSET_PATH.read_text().strip() or 0)
        except Exception:
            return 0

    def _write_offset(self, value: int) -> None:
        # Written through a temp file and renamed rather than written in place: a
        # plain write is not atomic, and offsets are shared by every session on
        # this bot, so a reader in another process can otherwise observe a
        # truncated number and acknowledge the wrong update. Per-pid temp name so
        # two sessions writing at once cannot share one.
        #
        # Path.replace is os.replace, which overwrites atomically on Windows too;
        # os.rename would raise there when the destination exists. No fcntl and no
        # flock anywhere on this path — it has to work identically on both.
        path = _offset_path()
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        try:
            _ensure_remote_dir()
            tmp.write_text(str(value))
            tmp.replace(path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass

    def ask(self, event: Event, options: List[str], timeout: float,
            cancel=None) -> Optional[str]:
        """Post `event` with one button per option and wait for a tap.

        Returns the chosen option, or None on cancel / timeout / not configured /
        another process already polling. None always means "nobody answered
        here" — the caller must then fall back to the local prompt rather than
        treat it as a decision. A permission prompt that nobody answers must
        never become an implicit allow *or* an implicit deny.

        The wait itself belongs to the process-wide poller: getUpdates is
        exclusive, so a second loop here would fight the one serving inbound
        messages and the two would steal each other's updates.

        The buttons stay live for the whole `timeout` and are collapsed only once
        the question is actually over. An earlier version collapsed them when
        *polling* stopped, conflating two different clocks: walk away for half an
        hour and you came back to a turn still blocked on a prompt that no longer
        had any buttons to press.
        """
        token, chat = _credentials()
        if not (token and chat):
            return None

        # Two short options (a grant: Allow/Deny) sit side by side, which is what a
        # thumb expects. Anything else goes one per row: a question's options are
        # sentences, and Telegram shrinks a shared row to fit them all, so four of
        # them become four unreadable slivers.
        side_by_side = len(options) == 2 and all(len(o) <= 12 for o in options)
        if side_by_side:
            keyboard = [[{'text': o, 'callback_data': f'p:{i}'}
                         for i, o in enumerate(options)]]
        else:
            keyboard = [[{'text': o, 'callback_data': f'p:{i}'}]
                        for i, o in enumerate(options)]
        # A question also takes an answer of your own, as the terminal picker's
        # "Other" does. The button only explains how (a button cannot carry a
        # sentence); the typed message is the answer. Never on a permission
        # prompt — a sentence there must not become a grant.
        from ..poller import OTHER_DATA
        free_text = event.kind == EventKind.QUESTION_ASKED
        if free_text:
            keyboard.append([{'text': _('✏️ Other (type your answer)'),
                              'callback_data': OTHER_DATA}])
        sent = self._post('sendMessage', {
            'chat_id': chat,
            'text': self._format(event),
            'parse_mode': 'HTML',
            'disable_web_page_preview': 'true',
            'reply_markup': json.dumps({'inline_keyboard': keyboard}),
        })
        if not sent or not sent.get('ok'):
            return None
        msg_id = (sent.get('result') or {}).get('message_id')
        if msg_id is None:
            return None
        note_chat_message(msg_id)

        from .. import inbox as _inbox
        from .. import poller as _poller
        loop = _poller.get(self)
        question = loop.add_question(msg_id, options, free_text)
        # Publish who is waiting on this prompt. The tap will land wherever the
        # poll lock happens to be, which with more than one session is usually
        # somebody else — that receiver reads this to know where to forward it.
        _inbox.claim_question(msg_id, _session_id(), len(options), free_text)
        loop.acquire()
        try:
            chosen = question.wait(timeout, cancel)
        finally:
            loop.drop_question(msg_id)
            _inbox.release_question(msg_id)
            loop.release()

        cancelled = chosen is None and cancel is not None and cancel.is_set()
        if chosen:
            outcome = f'\n\n→ <b>{html.escape(chosen)}</b>'
        elif cancelled:
            outcome = '\n\n-> <i>' + _('answered at the terminal') + '</i>'
        else:
            # Not "expired" — the turn is still blocked and a fresh prompt is on
            # its way. Saying it expired would read as "too late, go sit at the
            # terminal", which is the one outcome this whole path exists to avoid.
            outcome = '\n\n-> <i>' + _('waiting… (a fresh prompt is on its way)') + '</i>'
        self._post('editMessageText', {
            'chat_id': chat,
            'message_id': msg_id,
            'text': self._format(event) + outcome,
            'parse_mode': 'HTML',
        })
        return chosen
