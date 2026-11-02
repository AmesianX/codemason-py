"""
Token counting utilities for CodeMason
Equivalent to src/utils/tokens.ts
"""

import os
import time
from typing import Any, Dict, List, Optional, Union


from .log import debug_log


# === Token Limits ===

# Default fallback values
_DEFAULT_CONTEXT_SIZE = 131072
_DEFAULT_REQUEST_MAX_TOKENS = 16384  # max_tokens sent per request (matches server -n)

# Cached values (set once on first query to server)
_server_info: Optional[Dict] = None

# When the last failed probe finished. A failure is still not cached as an
# ANSWER — get_context_size must not be pinned to the fallback default for the
# session, which is what the note in _fetch_server_info protects — but it is
# rate-limited. The probe is a BLOCKING httpx call with a 5s timeout made from
# inside the event loop, and the token footer asks for it on every render, so a
# backend that is down froze the UI for five seconds per rendered line: no
# spinner, no ESC, no background task drain.
_last_probe_failure: float = 0.0

# Short enough that a server coming up is picked up within seconds, long enough
# that a server that is down does not cost a timeout per rendered footer.
_PROBE_RETRY_AFTER = 30.0


def _get_server_credentials() -> tuple:
    """Get base URL and API key from Config (single source of truth)."""
    from .config import get_config
    cfg = get_config()
    return cfg.model_base_url, cfg.model_api_key


# Resolve-once guard: model discovery runs a single /v1/models probe per process.
_model_resolved = False


def resolve_model() -> Optional[str]:
    """Make the model id used this run agree with what the server actually serves.

    Why: hard-coding the model id means one stale name (e.g. a renamed-away
    'ChatGPTN') makes every request fail late, in-stream, with a cryptic "model
    does not exist". The server serves a known set, so discover it instead:

      - config value present AND served        -> keep it (no change).
      - config value missing / not served      -> if the server advertises
        exactly ONE model, adopt it FOR THIS PROCESS and say so on stderr. If
        SEVERAL, leave it alone and print the available list (can't guess the
        user's intent).
      - /v1/models unreachable / errors         -> no-op; fall back to whatever
        is configured.

    config.json is never written from here. Which id a server advertises is a
    fact about the endpoint this run is pointed at, not a correction to what the
    user meant: persisting it turned "I tried the other server once" into a
    permanent edit of a file the user maintains by hand, and the original setting
    was already gone by the time they pointed it back.

    Runs at most once per process. Returns the resolved model id (or the
    unchanged configured value on any fallback path)."""
    global _model_resolved
    from .config import get_config
    cfg = get_config()
    if _model_resolved:
        return cfg.model_name
    _model_resolved = True

    import sys
    import httpx

    current = (cfg.model_name or '').strip()

    # --- 1. Reconcile the model id against what the server serves. ---
    base_url, api_key = _get_server_credentials()
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    ids = None
    try:
        resp = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=5.0)
        if resp.status_code == 200:
            ids = [m.get('id') for m in (resp.json() or {}).get('data', [])
                   if isinstance(m, dict) and m.get('id')]
    except Exception:
        ids = None  # unreachable -> keep whatever is configured

    if ids:
        if current and current in ids:
            pass  # already valid
        elif len(ids) == 1:
            cfg.model_name = ids[0]
            if current:
                sys.stderr.write(f"[codemason] Model '{current}' is not served; "
                                 f"auto-switched to '{ids[0]}'.\n")
            else:
                sys.stderr.write(f"[codemason] Model auto-detected: '{ids[0]}'.\n")
        else:
            # Several served, configured one invalid/empty: don't guess.
            sys.stderr.write(
                f"[codemason] Configured model '{current or '(none)'}' is not "
                f"served. Available: {', '.join(ids)}. Set model_name in "
                f"config.json.\n")

    # model_dir is left exactly as written. A bundled tokenizer already wins over
    # it at lookup time, so a redundant value costs nothing — but deleting it was
    # not free: model_dir is a path the user typed, the deletion was silent, and
    # "is it redundant?" was decided from model_name, which step 1 above rewrites
    # on its own. Point the config at a llama.cpp server and model_name becomes
    # the GGUF's full path; point it back and it resolves to a bundled name, at
    # which point the path the user set for the other server is gone. What it
    # bought back was one dead key in a file that already carries every default.
    #
    # Nothing here writes to config.json. The reconciliation above exists because
    # a served id can differ from the configured one, and that is a fact about
    # this run, not a correction to the user's intent: the same file pointed at
    # the original server is right again, and persisting would have quietly made
    # a temporary endpoint permanent. The value is used for this process and the
    # file keeps what was written in it.

    return cfg.model_name


def _fetch_server_info() -> Dict:
    """Detect the served context size, SERVER-AGNOSTICALLY.

    The backend may be vLLM (current) or llama.cpp, so probe both:
      - vLLM / OpenAI servers: GET /v1/models -> data[].max_model_len
      - llama.cpp:             GET /slots     -> [0].n_ctx  (per-slot)
    (/slots 404s on vLLM, so relying on it alone pinned the size to the fallback
    default on the real backend.)

    Cached ONLY on success. A failed probe must NOT be cached: an early call may
    happen before the server is guaranteed up, and a cached miss would pin
    get_context_size() to the fallback default for the whole session —
    mis-sizing the auto-compact threshold.
    """
    global _server_info, _last_probe_failure
    if _server_info is not None:
        return _server_info
    if _last_probe_failure and (time.monotonic() - _last_probe_failure) < _PROBE_RETRY_AFTER:
        return {}

    import httpx

    base_url, api_key = _get_server_credentials()
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    # vLLM / OpenAI-compatible: /v1/models advertises max_model_len.
    #
    # Match the CONFIGURED model id FIRST. data[0] is whichever model the server
    # happened to list first, which on an endpoint serving several is not the one
    # this session talks to — and every number downstream (the auto-compact
    # threshold, the output cap, the status bar) is derived from it, then cached
    # permanently on the first success. Wrong in either direction hurts: a
    # smaller window compacts a healthy session to death, a larger one lets the
    # request 400. resolve_model() already reconciles cfg.model_name against this
    # same list, so by the time anything can send a request the configured id IS
    # the served one.
    try:
        from .config import get_config as _gc
        model_name = (_gc().model_name or '').strip()
    except Exception:
        model_name = ''
    try:
        resp = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=5.0)
        if resp.status_code == 200:
            data = (resp.json() or {}).get('data') or []

            def _max_len(m):
                mlen = m.get('max_model_len') if isinstance(m, dict) else None
                return mlen if isinstance(mlen, int) and mlen > 0 else None

            if model_name:
                for m in data:
                    if isinstance(m, dict) and m.get('id') == model_name:
                        mlen = _max_len(m)
                        if mlen:
                            _server_info = {'n_ctx': mlen, 'total_slots': 1}
                            return _server_info
            # No id matched (model_name empty, an alias the server renames, or a
            # matching entry that omits max_model_len): fall back to the first
            # entry that advertises a window — the old behaviour. Some served
            # window still beats the fallback default, which is a guess about a
            # server nobody probed.
            for m in data:
                mlen = _max_len(m)
                if mlen:
                    _server_info = {'n_ctx': mlen, 'total_slots': 1}
                    return _server_info
    except Exception:
        pass

    # llama.cpp: /slots reports per-slot n_ctx.
    try:
        resp = httpx.get(f"{base_url}/slots", headers=headers, timeout=5.0)
        if resp.status_code == 200:
            slots = resp.json()
            if isinstance(slots, list) and len(slots) > 0:
                _server_info = {
                    'n_ctx': slots[0].get('n_ctx', _DEFAULT_CONTEXT_SIZE),
                    'total_slots': len(slots),
                }
                return _server_info
    except Exception:
        pass

    # All probes failed — do NOT cache; retry on the next call.
    # All probes failed. Still NOT cached as an answer: the next call after the
    # retry window probes again, so a server that comes up later is picked up.
    # Only the retry RATE is bounded.
    _last_probe_failure = time.monotonic()
    return {}


def get_context_size() -> int:
    """Per-slot context size. The SERVER is the authority; env only fills in.

    The override used to win outright, and a stale one then silently mis-sized
    everything downstream: an env of 1M against a server restarted at 512K put
    auto-compact at 983k, well past the point where the server itself rejects the
    request — so compaction could never fire in time and the session died instead
    of compacting. An override cannot make the server accept more than it was
    started with, so a value ABOVE the served size is not a preference, it is
    wrong, and the served size wins.

    Below the served size it is a real preference (deliberately running small),
    so it is honoured. When no probe succeeds, the override is all we have.
    """
    env_val = os.getenv('CODEMASON_MODEL_CONTEXT_SIZE') or os.getenv('LLAMACPP_CONTEXT_SIZE')
    served = _fetch_server_info().get('n_ctx')

    if not env_val:
        return served if served else _DEFAULT_CONTEXT_SIZE

    try:
        requested = int(env_val)
    except ValueError:
        debug_log('CONTEXT_SIZE_ENV_INVALID', {'value': env_val})
        return served if served else _DEFAULT_CONTEXT_SIZE

    if served and requested > served:
        debug_log('CONTEXT_SIZE_ENV_CLAMPED', {
            'requested': requested,
            'served': served,
            'note': 'env override exceeds what the server was started with',
        })
        return served
    return requested


def get_output_reserve() -> int:
    """Get output token reserve = config.max_tokens."""
    env_val = os.getenv('CODEMASON_MODEL_OUTPUT_RESERVE') or os.getenv('LLAMACPP_OUTPUT_RESERVE')
    if env_val:
        return int(env_val)
    from .config import get_config
    return get_config().max_tokens


def get_max_tokens() -> int:
    """Get maximum usable input tokens = context_size - output_reserve."""
    return get_context_size() - get_output_reserve()


# NOTE: there is deliberately NO module-level `MAX_TOKENS = get_max_tokens()`
# constant. It used to exist here, but computing it at import time ran a blocking
# server probe (get_context_size) + a config write on the very first `import
# codemason.utils` — stalling CLI startup by up to ~10s when the model server was
# slow/unreachable — to populate a value nothing actually read (every caller uses
# get_max_tokens()/get_context_size() live). Call get_max_tokens() when needed.


# === Model tokenizer (exact, matches the server) ===
#
# A generic vocab (the old tiktoken cl100k fallback) belongs to a different
# model, NOT the served one — it mis-counts
# DSV4 (Korean +42%, code -13%). When config.model_dir/tokenizer.json exists we
# load the model's OWN tokenizer, which matches the server's /tokenize exactly.
# Its presence also flips context handling to the upstream-faithful path (the
# non-canonical _manage_context pre-send trim switches OFF — see services/vllm).
#
# _UNSET = not yet probed; None = probed and unavailable (cache the miss so we
# don't re-stat every count). Config is fixed for a process, so caching is safe.
_UNSET = object()
_model_tokenizer: Any = _UNSET
_model_tokenizer_key: Optional[str] = None


def get_model_tokenizer():
    """Load the served model's own tokenizer (exact, matches the server's
    /tokenize) for the currently-configured model, or None if unavailable.

    Keyed by config.model_name — the id we send to the server (e.g.
    'DSV4-Flash') — so it scales to any number of models with NO hardcoded name.
    First existing candidate wins:
      1. <package>/tokenizer/<model>/tokenizer.json    (bundled in the source tree)
      2. <repo_root>/tokenizer/<model>/tokenizer.json  (bundled, repo-root layout)
         Bundled paths resolve relative to this file, so they travel with a
         source-dir copy — no global ~/.codemason, no absolute path, no
         per-machine setup. Just drop <model>/tokenizer.json under tokenizer/.
      3. <model_dir>/<model>/tokenizer.json            (explicit config.model_dir)
      4. <model_dir>/tokenizer.json                    (model_dir = an HF checkout)
    Returns None (=> CJK-aware heuristic) if none found or `tokenizers` isn't
    installed (pip install tokenizers)."""
    global _model_tokenizer, _model_tokenizer_key

    from .config import get_config
    cfg = get_config()
    model = (cfg.model_name or '').strip()
    # Recompute if the served model changed since the cache was built. Otherwise a
    # cache computed when model_name was still empty/stale (an early count before
    # resolve_model() adopts the served id) would pin to None — the
    # heuristic fallback — for the whole process, so the exact tokenizer never
    # engages even after the real id is known.
    if _model_tokenizer is not _UNSET and _model_tokenizer_key == model:
        return _model_tokenizer
    _model_tokenizer_key = model

    model_dir = (cfg.model_dir or '').strip()

    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # codemason/ package
    repo_dir = os.path.dirname(pkg_dir)                                    # repo root

    candidates = []
    if model:
        candidates.append(os.path.join(pkg_dir, 'tokenizer', model, 'tokenizer.json'))
        candidates.append(os.path.join(repo_dir, 'tokenizer', model, 'tokenizer.json'))
        if model_dir:
            candidates.append(os.path.join(model_dir, model, 'tokenizer.json'))
    if model_dir:
        candidates.append(os.path.join(model_dir, 'tokenizer.json'))

    _model_tokenizer = None
    for tok_path in candidates:
        if os.path.isfile(tok_path):
            try:
                from tokenizers import Tokenizer
                _model_tokenizer = Tokenizer.from_file(tok_path)
            except Exception:
                # lib missing or bad file -> stay on the heuristic fallback path.
                _model_tokenizer = None
            break
    return _model_tokenizer


def _bundled_tokenizer_exists(model: str) -> bool:
    """True when a tokenizer for `model` ships in the source tree (candidates 1-2
    of get_model_tokenizer). When this holds, config.model_dir is redundant — the
    bundle always wins — so it can be cleared without losing accurate counting."""
    model = (model or '').strip()
    if not model:
        return False
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_dir = os.path.dirname(pkg_dir)
    return any(os.path.isfile(os.path.join(d, 'tokenizer', model, 'tokenizer.json'))
               for d in (pkg_dir, repo_dir))


def has_model_tokenizer() -> bool:
    """True when the exact model tokenizer is available (accurate-counting mode).
    Gates OFF the non-canonical _manage_context trim in the backends."""
    return get_model_tokenizer() is not None


def _heuristic_token_count(content: str, bytes_per_token: int = 4) -> int:
    """CJK-aware character heuristic, used whenever the served model's own
    tokenizer is unavailable. Latin-ish text at len/bytes_per_token (~chars/4),
    CJK at ~1 token per char.

    This replaced a tiktoken cl100k fallback. cl100k is ANOTHER model's vocab —
    it mis-counts the served one (Korean +42%, code -13%), so it never gave a
    count worth trusting, and it reported that wrong number as if exact. It
    also hung the process: tiktoken's byte-pair merge rescans the whole parts
    list per merge (documented O(mn), "n is often very small"), and its
    pretokenizer emits an UNBOUNDED piece for a run of punctuation. A model
    that fell into repetition emitted 26,655 consecutive '!' — one piece,
    ~7e8 comparisons plus ~5.7GB of memmove, re-run every time the growing
    buffer was recounted. One core pinned for 30+ minutes, the response socket
    never drained. An estimate that is honest about being an estimate beats an
    exact-looking count from the wrong vocabulary."""
    latinish, dense, symbol = _tally_scripts(content)
    return round(latinish / bytes_per_token
                 + dense * _DENSE_TOKENS_PER_CHAR
                 + symbol * _SYMBOL_TOKENS_PER_CHAR)


def count_text_tokens(text: Optional[str]) -> int:
    """Exact token count for a string when the model's own tokenizer is
    available (matches the server); otherwise the CJK-aware heuristic. This is
    the single substrate both backends' count_tokens_precise should call."""
    if not text:
        return 0
    tok = get_model_tokenizer()
    if tok is not None:
        return len(tok.encode(text).ids)
    return _heuristic_token_count(text)


# === Token Counting Functions ===

def extract_text_content(message: Any) -> str:
    """Extract text content from a message for token estimation"""
    if isinstance(message, dict):
        msg_obj = message.get('message', {})
        content = msg_obj.get('content', '')
    elif hasattr(message, 'message'):
        msg_obj = message.message
        content = getattr(msg_obj, 'content', '')
    else:
        return ''

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get('type', '')
                if block_type == 'text':
                    parts.append(block.get('text', ''))
                elif block_type == 'tool_use':
                    import json
                    parts.append(json.dumps(block.get('input', {})))
                elif block_type == 'tool_result':
                    block_content = block.get('content', '')
                    if isinstance(block_content, str):
                        parts.append(block_content)
                    elif isinstance(block_content, list):
                        for c in block_content:
                            if isinstance(c, str):
                                parts.append(c)
                            elif isinstance(c, dict) and c.get('type') == 'text':
                                parts.append(c.get('text', ''))
            elif hasattr(block, 'type'):
                if block.type == 'text':
                    parts.append(getattr(block, 'text', ''))
                elif block.type == 'tool_use':
                    import json
                    parts.append(json.dumps(getattr(block, 'input', {})))
        return '\n'.join(parts)

    return ''


# Synthetic message texts to skip when counting tokens. MUST match the strings
# actually produced (utils/messages.py INTERRUPT_MESSAGE*); the old values
# ("Request was interrupted by user") were a stale pre-1.0 form that matched
# nothing, so interrupt messages leaked into the count as a zero-usage anchor.
SYNTHETIC_ASSISTANT_MESSAGES = {
    "[Request interrupted by user]",
    "[Request interrupted by user for tool use]",
    "No response requested.",
}


def count_tokens(messages: List[Any]) -> int:
    """
    Count tokens from the most recent assistant message's usage info.
    Falls back to estimation if no usage info available.
    """
    i = len(messages) - 1
    while i >= 0:
        message = messages[i]

        # Check if it's an assistant message with usage info
        is_assistant = False
        msg_obj = None

        if isinstance(message, dict):
            is_assistant = message.get('type') == 'assistant'
            msg_obj = message.get('message', {})
        elif hasattr(message, 'type') and message.type == 'assistant':
            is_assistant = True
            msg_obj = message.message if hasattr(message, 'message') else None

        if is_assistant and msg_obj:
            # Check for usage info
            usage = None
            if isinstance(msg_obj, dict):
                usage = msg_obj.get('usage')
                content = msg_obj.get('content', [])
            elif hasattr(msg_obj, 'usage'):
                usage = msg_obj.usage
                content = getattr(msg_obj, 'content', [])
            else:
                i -= 1
                continue

            # Skip synthetic messages
            if content and len(content) > 0:
                first_block = content[0] if isinstance(content, list) else content
                if isinstance(first_block, dict):
                    text = first_block.get('text', '')
                elif hasattr(first_block, 'text'):
                    text = first_block.text
                else:
                    text = ''
                if text in SYNTHETIC_ASSISTANT_MESSAGES:
                    i -= 1
                    continue

            # A zero total means the server reported no usage, not a free
            # response — keep walking back for a real anchor rather than
            # returning 0. (Same collapse as token_count_with_estimation; this
            # function is legacy but shares the bug, so it shares the guard.)
            if usage and _usage_total(usage) > 0:
                return _usage_total(usage)

        i -= 1

    # No usage info found - return 0 (matches TypeScript original)
    return 0


# === Rough block-based estimation (services/tokenEstimation.ts) ===

def _battr(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


# Same dict-or-object accessor, used for message/usage fields below.
_get_attr = _battr


# Unicode blocks a byte-level BPE splits into ~1+ tokens EACH. The plain chars/4
# heuristic is tuned for latin text and under-counts these ~4x, which would make
# the auto-compact estimate fire late. Counting them at ~1 token/char is the
# FLOOR of the true range, not a padded over-estimate — and under-counting is the
# dangerous direction: the threshold fires late and the server rejects the whole
# request, where firing early costs nothing.
#
# The list was CJK-only, so every non-Latin ALPHABET — Cyrillic, Greek, Hebrew,
# Arabic, Thai, the Indic blocks — was counted at chars/4. A byte-level BPE
# spends 2-3 UTF-8 bytes on each of those and has very few merges to spend them
# on, so the real cost is 1-2 tokens/char and the old count was 4-8x low. Three
# more holes in the same list: CJK punctuation (0x3000-0x303F, the 。、「」 that
# end every Japanese sentence) sat below the old 0x3040 start; the list stopped
# at 0xFFEF, so the astral ideographs — 4 bytes each and far less likely to be
# one merge than Ext A — were quartered; and emoji, which carry a variation
# selector and often a ZWJ on top of a 4-byte base, had no bucket at all.
_DENSE_RANGES = (
    (0x0370, 0x03FF),   # Greek and Coptic
    (0x0400, 0x04FF),   # Cyrillic
    (0x0500, 0x052F),   # Cyrillic Supplement
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x0900, 0x0DFF),   # Devanagari .. Sinhala (the 3-byte Indic blocks)
    (0x0E00, 0x0E7F),   # Thai
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x1F00, 0x1FFF),   # Greek Extended
    (0x2E80, 0x2FDF),   # CJK + Kangxi Radicals
    (0x3000, 0x30FF),   # CJK punctuation + Hiragana + Katakana
    (0x3130, 0x318F),   # Hangul Compatibility Jamo
    (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
    (0x3400, 0x4DBF),   # CJK Ext A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xA8E0, 0xA8FF),   # Devanagari Extended
    (0xAC00, 0xD7A3),   # Hangul Syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE30, 0xFE4F),   # CJK Compatibility Forms
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    (0xFF00, 0xFFEF),   # Halfwidth / Fullwidth Forms
    (0x20000, 0x323AF),  # CJK Ext B..H + Compatibility Ideographs Supplement
)

# Emoji and pictographs: a 4-byte base, usually a variation selector, often a
# ZWJ sequence on top. Two is the conservative floor for one rendered glyph.
_SYMBOL_RANGES = (
    (0x2600, 0x27BF),    # Misc Symbols + Dingbats
    (0xFE00, 0xFE0F),    # Variation Selectors (U+FE0F rides most emoji)
    (0x1F000, 0x1F2FF),  # Mahjong / Domino / Cards / Enclosed
    (0x1F300, 0x1FAFF),  # Pictographs, Emoticons, Transport, Ext-A
)

_DENSE_TOKENS_PER_CHAR = 1.0
_SYMBOL_TOKENS_PER_CHAR = 2.0
# Below this everything is Latin/punctuation/control, which is what chars/4 was
# measured on. Checked first because it is the overwhelming majority of input.
_LATINISH_CEILING = 0x0370


def _in_ranges(o: int, ranges) -> bool:
    for lo, hi in ranges:
        if lo <= o <= hi:
            return True
    return False


def _tally_scripts(content: str):
    """(latinish, dense, symbol) character counts."""
    latinish = dense = symbol = 0
    for ch in content:
        o = ord(ch)
        if o < _LATINISH_CEILING:
            latinish += 1
        elif _in_ranges(o, _DENSE_RANGES):
            dense += 1
        elif _in_ranges(o, _SYMBOL_RANGES):
            symbol += 1
        else:
            # Unlisted scripts (Latin Extended Additional, Armenian, …) keep the
            # chars/4 rule: still a guess, but not a 4x one.
            latinish += 1
    return latinish, dense, symbol


def rough_token_count_estimation(content: Optional[str], bytes_per_token: int = 4) -> int:
    """roughTokenCountEstimation. When the served model's OWN tokenizer is
    available this returns the EXACT count (matches the server) so compaction /
    microcompact size on real tokens — this is the single substrate; callers no
    longer hand-roll chars/4. Otherwise it falls back to the
    CJK-aware char heuristic: Latin-ish text at len/bytesPerToken (~chars/4),
    CJK characters at ~1 token each (chars/4 under-counts CJK badly and would
    make the auto-compact threshold fire late on Korean/CJK content)."""
    if not content:
        return 0
    if has_model_tokenizer():
        return count_text_tokens(content)
    return _heuristic_token_count(content, bytes_per_token)


def _rough_block(block: Any) -> int:
    """roughTokenCountEstimationForBlock — per content-block type, chars/4."""
    if isinstance(block, str):
        return rough_token_count_estimation(block)
    btype = _battr(block, 'type')
    if btype == 'text':
        return rough_token_count_estimation(_battr(block, 'text', ''))
    if btype == 'image':
        # What the server actually spends on it (header-only read, same formula
        # as the server) — NEVER fall through to a base64 char count.
        from .image_prep import image_block_tokens
        return image_block_tokens(block)
    if btype == 'document':
        # Documents keep the original's conservative flat estimate.
        return 2000
    if btype == 'tool_result':
        return _rough_content(_battr(block, 'content'))
    if btype == 'tool_use':
        import json as _json
        return rough_token_count_estimation(
            str(_battr(block, 'name', '')) + _json.dumps(_battr(block, 'input', {}) or {})
        )
    if btype == 'thinking':
        return rough_token_count_estimation(_battr(block, 'thinking', ''))
    if btype == 'redacted_thinking':
        return rough_token_count_estimation(_battr(block, 'data', ''))
    # server_tool_use / web_search_tool_result / mcp_tool_use / unknown.
    import json as _json
    try:
        return rough_token_count_estimation(_json.dumps(block, default=str))
    except Exception:
        return rough_token_count_estimation(str(block))


def _rough_content(content: Any) -> int:
    if not content:
        return 0
    if isinstance(content, str):
        return rough_token_count_estimation(content)
    if isinstance(content, list):
        return sum(_rough_block(b) for b in content)
    return _rough_block(content)


def rough_token_count_estimation_for_messages(messages: List[Any]) -> int:
    """roughTokenCountEstimationForMessages — sum of per-message content estimates."""
    total = 0
    for message in messages:
        mtype = _get_attr(message, 'type')
        if mtype in ('assistant', 'user'):
            msg_obj = _get_attr(message, 'message', {})
            content = _get_attr(msg_obj, 'content') if msg_obj is not None else None
            total += _rough_content(content)
    return total


# Synthetic assistant messages whose usage must NOT anchor the token count.
_SYNTHETIC_MODEL = '<synthetic>'


def _get_token_usage(message: Any):
    """getTokenUsage: usage only for a real (non-synthetic) assistant message."""
    if _get_attr(message, 'type') != 'assistant':
        return None
    msg_obj = _get_attr(message, 'message')
    if msg_obj is None:
        return None
    if _get_attr(msg_obj, 'model') == _SYNTHETIC_MODEL:
        return None
    content = _get_attr(msg_obj, 'content')
    if isinstance(content, list) and content:
        first = content[0]
        if _get_attr(first, 'type') == 'text' and _get_attr(first, 'text', '') in SYNTHETIC_ASSISTANT_MESSAGES:
            return None
    return _get_attr(msg_obj, 'usage')


def _get_assistant_message_id(message: Any) -> Optional[str]:
    if _get_attr(message, 'type') != 'assistant':
        return None
    msg_obj = _get_attr(message, 'message')
    if msg_obj is None or _get_attr(msg_obj, 'model') == _SYNTHETIC_MODEL:
        return None
    mid = _get_attr(msg_obj, 'id')
    return str(mid) if mid else None


def _usage_total(usage: Any) -> int:
    if isinstance(usage, dict):
        return (usage.get('input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
                + usage.get('cache_read_input_tokens', 0) + usage.get('output_tokens', 0))
    return (getattr(usage, 'input_tokens', 0) + getattr(usage, 'cache_creation_input_tokens', 0)
            + getattr(usage, 'cache_read_input_tokens', 0) + getattr(usage, 'output_tokens', 0))


# Tokens the estimate cannot see, measured once per process.
#
# The system prompt and the tool block go out with every request but live
# outside `messages`, so a walk over the history never counts them. An ANCHORED
# estimate does not need them — it starts from the server's own count of the
# previous request, which already included both — but the unanchored fallback
# is blind to roughly 30k tokens here (~25k of tool schemas plus the system
# prompt). That gap is what let a 458,753-token input sit under the compaction
# threshold and then be rejected at 524,289 > 524,288.
#
# PER SCOPE, not per process. A single value was written by whichever request
# went out last, and the main conversation and its sub-agents do not have the
# same out-of-band size: a sub-agent carries its own system prompt and a subset
# of the tools (often three or four against the main agent's full block). They
# run concurrently, so the main agent's compaction check could read a sub-agent's
# small number and under-count its own request by ~25k — the exact gap this value
# exists to close — or a sub-agent could be measured against the main block and
# compacted while its history was still short. The scope key is the tool set (see
# oob_scope_key), which is what differs between them and is available on both the
# writing and the reading side.
_OUT_OF_BAND_TOKENS: Dict[str, int] = {}


def oob_scope_key(tool_names) -> str:
    """The scope a request's out-of-band size is filed under: its tool set.

    Built in one place so the writer (_openai_common, before the request) and the
    reader (the compaction check, a turn later) cannot drift apart — they did not
    share a key at all before, which is how one could read the other's number.
    """
    return '\x1f'.join(tool_names)


def set_out_of_band_tokens(scope: str, n: int) -> None:
    """Record the system-prompt + tool-block size for estimate_input_tokens."""
    _OUT_OF_BAND_TOKENS[scope] = max(0, int(n))


def _out_of_band_for(scope: str) -> int:
    """The out-of-band size for `scope`.

    An unregistered scope falls back to the LARGEST one recorded, not to zero.
    Zero is what the old unanchored estimate already did, and it is the direction
    that ends in a server rejection: the request goes out believing it is ~25k
    smaller than it is, and once the INPUT alone is over the window the output cap
    downstream cannot save it. Over-counting only compacts a little early.
    """
    if scope in _OUT_OF_BAND_TOKENS:
        return _OUT_OF_BAND_TOKENS[scope]
    return max(_OUT_OF_BAND_TOKENS.values(), default=0)


def estimate_input_tokens(messages: List[Any], scope: str = '') -> int:
    """What the SERVER will see: the history plus the out-of-band prompt parts.

    Use this, not token_count_with_estimation, wherever the number is compared
    against the model's context window — the compaction threshold and the output
    cap both are. token_count_with_estimation answers a narrower question ("how
    big is this history"), and using it as a stand-in for the request size is
    what made compaction miss and the request overflow.

    Adds the out-of-band tokens only when the estimate is UNANCHORED. Adding
    them to an anchored count double-counts the tool block on every turn of an
    ongoing conversation, which silently shortens answers and pulls compaction
    forward.
    """
    est = token_count_with_estimation(messages)
    oob = _out_of_band_for(scope)
    if oob and not has_usage_anchor(messages):
        est += oob
    return est


def has_usage_anchor(messages: List[Any]) -> bool:
    """True when token_count_with_estimation will anchor on a REAL usage record.

    Callers need this to know what the estimate already contains. An anchored
    count starts from `_usage_total(usage)` — the server's own count of the
    request it answered — so the system prompt and the tool definitions are
    ALREADY in it, invisible though they are to a walk over `messages`. The
    unanchored fallback is a pure per-block estimate of the message text, and
    those two are missing from it entirely.

    Adding them unconditionally is therefore wrong in the common case: it
    double-counts the tool block (~25k tokens here) on every turn of an ongoing
    conversation, which does not overflow anything but silently shortens every
    answer and pulls auto-compaction earlier than it should fire.

    The condition below must stay identical to the anchor test in
    token_count_with_estimation; both go through _get_token_usage (which skips
    synthetic messages) and the >0 total rule (a zero total means the server
    reported nothing, not that the turn was free).
    """
    for i in range(len(messages) - 1, -1, -1):
        usage = _get_token_usage(messages[i])
        if usage is not None and _usage_total(usage) > 0:
            return True
    return False


def token_count_with_estimation(messages: List[Any]) -> int:
    """tokenCountWithEstimation.

    Anchors on the most recent assistant message with real (non-synthetic) API
    usage, then rewinds past sibling records split from the SAME API response
    (parallel tool calls share message.id) so interleaved tool_results between
    splits are counted, and adds a per-block char/4 estimation of everything
    after the anchor. This makes compaction fire mid-tool-loop (new tool_results
    have no usage yet but are estimated and added on top), and counts images at
    what the server spends on them (utils/image_prep.image_block_tokens) instead
    of dropping them. Falls back to a pure estimate
    when no message has usage.

    A usage record whose counts sum to ZERO does NOT anchor. A real API response
    always spends at least its input tokens, so a zero total never means "a
    response that cost nothing" — it means the server reported no usage (no final
    include_usage chunk, or an Esc-interrupted stream). Anchoring on it returned
    `0 + estimate(nothing after it) = 0` whenever such a response was the most
    recent message, collapsing the whole count to zero — and this is the number
    compaction sizes on, so a single usage-less turn silently switched
    auto-compact off. Skipping it walks back to the last response that DID report
    usage and estimates everything after that anchor (the usage-less response
    among it), which is the honest count.
    """
    i = len(messages) - 1
    while i >= 0:
        message = messages[i]
        usage = _get_token_usage(message)
        if usage is not None and _usage_total(usage) > 0:
            # Rewind to the FIRST sibling sharing this response id so interleaved
            # tool_results between parallel-tool splits are included.
            response_id = _get_assistant_message_id(message)
            if response_id:
                j = i - 1
                while j >= 0:
                    prior_id = _get_assistant_message_id(messages[j])
                    if prior_id == response_id:
                        i = j
                    elif prior_id is not None:
                        break
                    j -= 1
            return _usage_total(usage) + rough_token_count_estimation_for_messages(messages[i + 1:])
        i -= 1

    return rough_token_count_estimation_for_messages(messages)


# === Prefill estimate (what the server must re-read before it can answer) ===
#
# The cached prefix makes most requests cost only their new tail. When it
# breaks, the same request costs the WHOLE conversation — 1,335 tok/s here, so
# 73k is 55 seconds and 233k is 175 — and the screen looks identical either
# way: a spinner, and seconds. Publishing the number turns "it froze" into
# "it is re-reading 73k tokens", which is the difference between a bug report
# and a wait the user can judge.
_pending_prefill = 0
_pending_cached = 0
_prefill_measured = False   # True once the number came from the server
_output_started = False


def set_pending_prefill(tokens: int) -> None:
    """Our own estimate, from comparing this request against the previous one.
    Available immediately; superseded by the server's number when it arrives."""
    global _pending_prefill
    if _prefill_measured:
        return  # never walk a measured value back to a guess
    _pending_prefill = max(0, int(tokens))


def set_pending_prefill_measured(compute: int, cached: int) -> None:
    """vLLM's own accounting: `compute` tokens must be computed, `cached` were
    served from the prefix cache (services.prefill_probe)."""
    global _pending_prefill, _pending_cached, _prefill_measured
    _pending_prefill = max(0, int(compute))
    _pending_cached = max(0, int(cached))
    _prefill_measured = True


def reset_prefill_state() -> None:
    """Call at the start of every request: the previous turn's numbers must not
    leak into this one's display."""
    global _pending_prefill, _pending_cached, _prefill_measured, _output_started
    _pending_prefill = 0
    _pending_cached = 0
    _prefill_measured = False
    _output_started = False


def mark_output_started() -> None:
    global _output_started
    _output_started = True


def output_started() -> bool:
    return _output_started


def pending_prefill() -> int:
    return _pending_prefill


def pending_cached() -> int:
    return _pending_cached


def prefill_is_measured() -> bool:
    return _prefill_measured


# === What a tool is currently waiting on ===
#
# A tool that waits shows nothing about WHY. web_fetch gives a slow host 60
# seconds, web_search retries with backoff (51s worst case on the shipped
# defaults), and bash runs in the foreground before moving to the background —
# all of it behind an unchanging spinner whose verb is picked at RANDOM, so a
# waiting tool and a thinking model look identical. Measured 2026-09-07: a user
# watched "Calculating… (297.6s · ↓ 0 tokens)" while a tool sat on a `sleep 60`
# loop, and read it as the model being stuck.
#
# A single slot, not a stack: tools that run concurrently are the read-only
# ones that finish fast, and the last writer is the one still going.
_wait_note: Optional[str] = None


def set_wait_note(note: Optional[str]) -> None:
    """Say what this tool is waiting on; None clears it. Keep it short — it
    shares one terminal line with the elapsed time and token counts."""
    global _wait_note
    _wait_note = note


def wait_note() -> Optional[str]:
    return _wait_note


class WaitNoteGuard:
    """Clears the wait note on exit, so an early return or an exception cannot
    leave a stale "fetching ..." on screen after the tool has finished."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_wait_note(None)
        return False
