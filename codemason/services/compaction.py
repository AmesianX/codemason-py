"""
Conversation compaction for CodeMason.
Based on src/services/compact/ (autoCompact.ts + prompt.ts +
the core of compact.ts compactConversation) from Claude Code.

WHAT (thresholds, summary prompt, summary formatting, continuation message,
circuit breaker) is ported verbatim. The token-counting it triggers
on uses codemason's local token substrate (utils.tokens), and the
summarization query goes through the llama.cpp service — the original used the
Anthropic API + its own token accounting, neither of which exists locally.

This REPLACES the old stopgap commands/compact.py.
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.tokens import estimate_input_tokens
from ..utils.tokens import (
    get_context_size, rough_token_count_estimation,
    count_text_tokens,
)
from ..utils.config import get_config
from ..utils.log import debug_log, log_error


# === Threshold constants (autoCompact.ts) ===

MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000
AUTOCOMPACT_BUFFER_TOKENS = 13_000
# Auto-compact fires when the estimated prompt reaches this share of the context
# window — a percentage so the trigger scales with the model instead of a fixed
# absolute reserve. Clamped by the overflow-safe point (see
# get_auto_compact_threshold) so a large max_tokens can't let a turn overflow.
AUTOCOMPACT_FRACTION = 0.88
# Share of the PARENT's threshold a sub-agent gets when the operator has not
# set subagent_context_ceiling_tokens. Derived, not absolute, so it scales with
# the model instead of hard-coding one deployment's KV pool into the client.
#
# Blocks, not tokens, are the currency, and the client cannot see the two
# server numbers that set the exchange rate (pool size and
# VLLM_PREFIX_CACHE_RETENTION_INTERVAL), so the default has to be a fraction
# that survives an untuned server.
#
# Measured 2026-09-13 against this deployment (vLLM, 215-block pool, 4608-token
# attention pages, retention 18432, parent parked at 441,165 tokens; each trial
# prefilled a separate conversation of size S and then re-queried the parent,
# reading the prefix-cache hit rate):
#
#   S <= 270,000  -> 99.2%, nothing lost. 270,000 ran 10/10 clean; smaller
#                    sizes 9 more trials over 8 sizes from 60k up. (The 0.8%
#                    is the unfillable last partial block, not a loss.)
#   S  = 280,000  -> 98.2% in 9 of 10 trials, 96.1% in the tenth: the parent
#                    gives up one 4608-token attention block, ~8s to recompute
#   S  = 290,000  -> survived once, evicted twice (0.0% and 25.1%)
#   S >= 296,945  -> 0.0%, ~360s full re-prefill
#
# Two things about the shape. The boundary is PROBABILISTIC, not a threshold:
# the same 290,000 survived one trial and killed the parent in the next two, so
# a size is only usable when it survives repeatedly, never on one green run.
# And the failure is all-or-nothing: a Mamba hit needs the one state block
# ending on the boundary, so losing a single checkpoint zeroes the hit rather
# than trimming its tail, costing a cold re-prefill of the whole parent. Both
# are why the default sits far below the cliff, not near it.
#
# The block cost that produces this: a conversation of L tokens holds L/4608
# attention blocks plus L/retention Mamba checkpoints. Mamba state during the
# run does not accumulate — vLLM's single_type_kv_cache_manager frees each
# superseded block under mamba_cache_mode "align" — so only the checkpoints
# scale with length. At 441,165 the parent holds ~119 of the 215 blocks and
# leaves ~96, and 0.30 of the parent threshold (133,725) needs ~37 of them:
# roughly a third of the measured room, which is the margin the all-or-nothing
# failure mode deserves.
#
# That whole derivation assumed ONE sub-agent. This line runs up to
# max_concurrent_subagents of them at once, so the footprint this fraction
# describes is multiplied by the slot count — and nothing used to notice. Hence
# the total below.
SUBAGENT_THRESHOLD_FRACTION = 0.30
# Ceiling on the COMBINED footprint of every sub-agent that may be in flight,
# as a share of the parent's threshold. The per-agent fraction above is a
# measurement of what one extra tenant costs; this is a measurement of what all
# of them together may cost, and get_auto_compact_threshold takes whichever is
# smaller.
#
# Why both numbers are needed. The per-agent fraction alone is silent about
# concurrency: raise max_concurrent_subagents to 6 and the client cheerfully
# admits 6 x 133,725 = 802,350 tokens of sub-agent on top of a 445,752-token
# parent, which is 1,248,102 before Mamba checkpoints against a pool that holds
# 1,356,053 — a wipe with nothing reporting it. A total alone is no good either:
# it would hand a single sub-agent the whole budget, far past the 270,000 the
# September table shows one tenant can cost.
#
# 0.90 is three sub-agents at the per-agent cap — this line's actual
# configuration — and that is the case it was measured in. Measured 2026-10-10
# against this deployment (KV 12GB, fp8_ds_mla, 344 blocks of 4,608 tokens =
# 1,356,053, read from vLLM's own cache_config_info; the September table above
# was taken on a 215-block pool, which is why three tenants fit now where the
# old arithmetic left room for one):
#
#   parent parked at its FULL threshold (444,860-446,967 tokens over the three
#   trials), then three concurrent conversations of ~133,000 each prefilled to
#   completion, then the parent re-read alone and vLLM's prefix-cache counters
#   read across that one request:
#
#     trial 1 -> 99.4% hit, 0 blocks lost, re-read 3.6s
#     trial 2 -> 99.4% hit, 0 blocks lost, re-read 3.7s
#     trial 3 -> 99.0% hit, 0 blocks lost, re-read 5.5s
#
#   The parent held 442,368 tokens — 96 x 4,608, every attention block it has —
#   in all three. The residue is the unfillable last partial block, not a loss,
#   which is why the verdict is taken against the warm baseline and not against
#   100%. Three trials because the note above says a size is only usable when it
#   survives repeatedly; the cost it is protecting against is the 374s cold
#   prefill those same runs measured.
#
#   Block accounting: 96 attention blocks for the parent plus 29 for each
#   sub-agent is 183 of 344, a little over half the pool, before the Mamba
#   checkpoints — this measurement did not re-read
#   VLLM_PREFIX_CACHE_RETENTION_INTERVAL, so their count is not re-derived here.
#
# A smaller pool is where the total has to come down, exactly as the per-agent
# fraction does; subagent_gate's docstring says how to read the slot count.
SUBAGENT_TOTAL_FRACTION = 0.90
WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
ERROR_THRESHOLD_BUFFER_TOKENS = 20_000
MANUAL_COMPACT_BUFFER_TOKENS = 3_000
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3

# PTL (prompt-too-long) retry: if the compaction summary REQUEST itself overflows
# the context, drop the oldest rounds and retry, so the session isn't stuck
# permanently over-limit (compact.ts truncateHeadForPTLRetry / MAX_PTL_RETRIES).
MAX_PTL_RETRIES = 3
PTL_RETRY_MARKER = '[earlier conversation truncated for compaction retry]'
# Safety margin added to the parsed prompt-too-long gap so a single retry
# actually fits: covers the re-added marker, vLLM's "at least" overflow wording,
# and rough-estimate vs server-count slack. Tiny next to one API round.
_PTL_GAP_MARGIN = 1024


def _msg_first_text(m) -> str:
    mm = m.get('message') if isinstance(m, dict) else getattr(m, 'message', None)
    content = (mm.get('content') if isinstance(mm, dict) else getattr(mm, 'content', None)) if mm is not None else None
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        b = content[0]
        return b.get('text', '') if isinstance(b, dict) else getattr(b, 'text', '')
    return ''


def _looks_like_prompt_too_long(text: str) -> bool:
    s = (text or '').lower()
    return any(k in s for k in (
        'context length', 'maximum context', 'context_length_exceeded',
        'longer than the maximum', 'reduce the length', 'max_model_len',
        'exceeds the maximum', 'too long',
    ))


POST_COMPACT_MAX_FILES_TO_RESTORE = 5
POST_COMPACT_TOKEN_BUDGET = 50_000
POST_COMPACT_MAX_CHARS_PER_FILE = 20_000   # ~5k tokens


def _post_compact_file_attachments(read_file_state):
    """Re-read the most-recently-read files fresh and return
    (path, full_content, meta_user_message) tuples, bounded to 5 files /
    ~5k tokens each / 50k-token budget. Port of createPostCompactFileAttachments:
    lets the model keep recent file context after compaction instead of re-reading.
    """
    import os
    from ..utils.messages import create_user_message

    entries = []
    for path, st in (read_file_state or {}).items():
        ts = getattr(st, 'timestamp', None)
        if ts is None and isinstance(st, dict):
            ts = st.get('timestamp')
        entries.append((path, ts or 0.0))
    entries.sort(key=lambda e: e[1], reverse=True)   # most recently read first

    out = []
    used = 0
    for path, _ts in entries[:POST_COMPACT_MAX_FILES_TO_RESTORE]:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                full = f.read()
        except OSError:
            continue
        shown = full if len(full) <= POST_COMPACT_MAX_CHARS_PER_FILE else (
            full[:POST_COMPACT_MAX_CHARS_PER_FILE] + '\n... [truncated for post-compact restore] ...'
        )
        block = (f"<system-reminder>\nRestored after compaction — recently read file "
                 f"{path}:\n\n{shown}\n</system-reminder>")
        tokens = count_text_tokens(block)  # exact (model tokenizer when available), not raw chars/4
        if used and used + tokens > POST_COMPACT_TOKEN_BUDGET:
            break
        used += tokens
        m = create_user_message(block)
        m['isMeta'] = True
        out.append((path, full, m))
    return out


def create_async_agent_attachments_if_needed(tool_use_context) -> list:
    """Messages telling the model about its agents, for after a compaction
    (createAsyncAgentAttachmentsIfNeeded).

    The summary does not carry the spawn, so without this the model comes out
    of a compaction having forgotten that it has agents running — and spawns
    them again. It covers both the ones still working (so no duplicate) and the
    ones that finished but whose result nobody has read.

    Three are skipped: a task already retrieved, one that has not started, and
    the agent's own task — an agent must not be told about itself.
    """
    from ..tasks.local_agent_task import is_local_agent_task
    from ..utils.messages import render_task_status_attachment
    from ..utils.task.disk_output import get_task_output_path

    get_app_state = getattr(tool_use_context, 'get_app_state', None)
    if get_app_state is None:
        from ..state.app_state import get_app_state as get_app_state_default
        get_app_state = get_app_state_default
    own_agent_id = getattr(tool_use_context, 'agent_id', None)

    out = []
    for task in list((get_app_state().tasks or {}).values()):
        if not is_local_agent_task(task):
            continue
        if task.retrieved or task.status == 'pending' or task.agent_id == own_agent_id:
            continue
        out.extend(render_task_status_attachment({
            'type': 'task_status',
            'taskId': task.agent_id,
            'taskType': 'local_agent',
            'description': task.description,
            'status': task.status,
            # Running: the prose summary, if the summarisation path wrote one.
            # Terminal: the error, which is the one thing the status alone does
            # not say.
            'deltaSummary': (
                (task.progress.summary if task.progress else None)
                if task.status == 'running' else task.error
            ),
            'outputFilePath': str(get_task_output_path(task.agent_id)),
        }))
    return out


def _parse_ptl_token_gap(text: Optional[str]) -> Optional[int]:
    """Parse how many tokens over the limit a prompt-too-long error is
    (getPromptTooLongTokenGap). Handles the vLLM/OpenAI form
    ("maximum context length is L tokens ... for a total of at least A tokens")
    and the input+output form, returning A-L when > 0, else None so the caller
    falls back to the 20% heuristic on unparseable (e.g. Vertex/Bedrock) errors."""
    if not text:
        return None
    s = text.lower()
    limit = re.search(r'maximum context length is\s+(\d+)', s) \
        or re.search(r'(\d+)\s+maximum', s)
    if not limit:
        return None
    limit_n = int(limit.group(1))
    total = re.search(r'total of(?:\s+at least)?\s+(\d+)', s)
    if total:
        gap = int(total.group(1)) - limit_n
        return gap if gap > 0 else None
    # input + requested-output form: actual = input + output.
    inp = re.search(r'(\d+)\s+input tokens', s)
    out = re.search(r'(\d+)\s+output tokens', s)
    if inp:
        actual = int(inp.group(1)) + (int(out.group(1)) if out else 0)
        gap = actual - limit_n
        return gap if gap > 0 else None
    return None


def _group_messages_by_api_round(messages):
    """groupMessagesByApiRound: group 0 is the leading preamble (everything
    before the first assistant); each later group starts at an assistant message
    and runs until the next one, keeping tool_use/tool_result pairs together so
    dropping whole groups never orphans a tool result."""
    groups = []
    cur = []
    for m in messages:
        mtype = m.get('type') if isinstance(m, dict) else getattr(m, 'type', None)
        if mtype == 'assistant' and cur:
            groups.append(cur)
            cur = []
        cur.append(m)
    if cur:
        groups.append(cur)
    return groups


def _truncate_head_for_ptl(messages, ptl_text: Optional[str] = None):
    """Drop the OLDEST api-round groups until the prompt-too-long token gap is
    covered — surgical, not a flat 20% — and prepend a marker. Faithful port of
    compact.ts truncateHeadForPTLRetry: parse the overflow from the error and
    peel only enough context to fit, keeping >=1 group. Falls back to 20% of
    groups when the gap is unparseable."""
    from ..utils.messages import create_user_message
    from ..utils.tokens import rough_token_count_estimation_for_messages

    # Strip our own marker from a prior retry so grouping/progress isn't stalled.
    inp = messages
    if messages:
        m0 = messages[0]
        is_user = (m0.get('type') if isinstance(m0, dict) else getattr(m0, 'type', None)) == 'user'
        if is_user and _msg_first_text(m0) == PTL_RETRY_MARKER:
            inp = messages[1:]

    groups = _group_messages_by_api_round(inp)
    if len(groups) < 2:
        return None

    gap = _parse_ptl_token_gap(ptl_text)
    if gap is not None:
        # Deviation from the strict port (acc >= gap): require a margin on top.
        # (1) we re-add PTL_RETRY_MARKER below, so freeing exactly `gap` nets to
        # zero progress when the oldest group is tiny; (2) vLLM reports "a total
        # of AT LEAST N tokens", so the real overflow may exceed the parsed gap;
        # (3) the rough estimate differs from the server's real count. The margin
        # is still tiny next to one round, so this stays surgical (far from 20%).
        need = gap + _PTL_GAP_MARGIN
        acc = 0
        drop_count = 0
        for g in groups:
            acc += rough_token_count_estimation_for_messages(g)
            drop_count += 1
            if acc >= need:
                break
    else:
        drop_count = max(1, len(groups) // 5)

    # Keep at least one group so there's something to summarize.
    drop_count = min(drop_count, len(groups) - 1)
    if drop_count < 1:
        return None

    sliced = [m for g in groups[drop_count:] for m in g]
    # Dropping the preamble can leave an assistant-first sequence (the API
    # requires a user-role first message); prepend the marker to fix it.
    first_type = (sliced[0].get('type') if isinstance(sliced[0], dict)
                  else getattr(sliced[0], 'type', None)) if sliced else None
    if first_type == 'assistant':
        marker = create_user_message(PTL_RETRY_MARKER)
        marker['isMeta'] = True
        return [marker, *sliced]
    return sliced


def get_effective_context_window_size() -> int:
    """getEffectiveContextWindowSize: context window minus reserved output.

    Reserve enough for the model's ACTUAL response. codemason's per-turn requests
    ask for cfg.max_tokens of output (65536 by default), so the effective window
    must keep at least that much free — otherwise a large input slips under the
    compaction threshold while still overflowing the model
    (input + max_tokens > context, the 458753+65536 > 524288 error). The
    original port reserved only MAX_OUTPUT_TOKENS_FOR_SUMMARY, which is correct
    only while max_tokens <= that; keep it as the floor so compaction always has
    room to write its summary too."""
    reserved = max(get_config().max_tokens, MAX_OUTPUT_TOKENS_FOR_SUMMARY)
    return get_context_size() - reserved


# The smallest prompt at which this deployment's decoder has actually collapsed
# this session, or None until one is seen.
#
# Reported, never acted on. Every input to the auto-compact threshold describes
# the window the SERVER advertises; this describes a history the MODEL failed
# on, which is a different kind of fact and a weaker one — one sample, no
# control, and no evidence about the largest prompt that WOULD have survived. It
# is kept because it is the number nobody had, not because it is authority to
# shrink the window.
#
# How weak it is, measured on this deployment's own turns.log over 2,483
# requests (2026-09-22..10-02): collapses are NOT ordered by prompt size. The
# rate per size band runs 2.3% / 7.9% / 17.2% / 15.7% / 0.0% / 15.4% / 0.0%
# (50k bands from 0 to 300k+), the largest prompt ever sent — 327,795 tokens —
# came back healthy, and the per-DAY rate swings from 0.55% to 25.98% on the
# same client and the same code. Whatever decides this is on the server side,
# not in the length. Deriving a ceiling from one collapse would therefore be
# wrong twice over, which is why the threshold below ignores this value.
_OBSERVED_COLLAPSE_TOKENS: Optional[int] = None
_OBSERVED_COLLAPSE_LOCK = threading.Lock()

# Collapses smaller than this are recorded but not written to error.log.
# Decoders also collapse for reasons that have nothing to do with length (a
# poisoned tool result, a pathological repetition in the input), and a line
# pointing at the context window would misdirect whoever reads it.
MIN_COLLAPSE_REPORT_TOKENS = 65_536


def note_decoder_collapse(prompt_tokens: int) -> None:
    """Record that a request of `prompt_tokens` came back collapsed, and say so.

    Called from the one place that sees both numbers: the stop reason and the
    prompt size of the request that earned it (services._openai_common, next to
    the turns.log line that reports the same figure).

    Keeps the LOWEST collapse seen. A later, higher one says nothing new. This
    only decides what gets reported — no threshold reads it.
    """
    global _OBSERVED_COLLAPSE_TOKENS
    if prompt_tokens <= 0:
        return
    with _OBSERVED_COLLAPSE_LOCK:
        seen = _OBSERVED_COLLAPSE_TOKENS
        if seen is not None and prompt_tokens >= seen:
            return
        _OBSERVED_COLLAPSE_TOKENS = prompt_tokens

    window = get_context_size()
    threshold = get_auto_compact_threshold(False)
    debug_log('DECODER_COLLAPSE_OBSERVED', {
        'prompt_tokens': prompt_tokens,
        'advertised_window': window,
        'threshold': threshold,
        'reported': prompt_tokens >= MIN_COLLAPSE_REPORT_TOKENS,
    })
    # Logged unconditionally, not just under debug. The collapse point of the
    # served weights is not reported by anything else, and without it the only
    # visible symptom is a turn that stops with the status bar reading 59%.
    #
    # Three numbers, no advice. Nothing here changes when compaction fires, and
    # nothing here proposes changing it: the threshold is already a cut down
    # from the window, and a single collapse is not grounds to cut it further.
    # If an operator decides otherwise, auto_compact_threshold_tokens is theirs.
    if prompt_tokens >= MIN_COLLAPSE_REPORT_TOKENS:
        log_error(
            f"decoder collapsed at {prompt_tokens} prompt tokens "
            f"(window {window}, auto-compaction fires at {threshold})",
            'compaction: collapse observed',
        )


def get_auto_compact_threshold(is_subagent: bool = False) -> int:
    """Token count at which auto-compaction fires, for one tenant of the
    server's KV pool.

    `is_subagent` matters because the two tenants are not interchangeable. The
    parent is the conversation the user is having and must be allowed the whole
    window; the sub-agent is transient and only has to fit in the space the
    parent leaves. Capping the parent to make room would be paying for the
    sub-agent with the thing the user actually wanted.

    Primarily a PERCENTAGE of the context window (AUTOCOMPACT_FRACTION), so it
    scales with the model's size instead of a fixed absolute reserve. But it is
    never allowed to sit past the point where the prompt plus a full max_tokens
    response would overflow the model — otherwise a large input slips under a
    high percentage threshold while still 400-ing the server
    (458753 + 65536 > 524288). So take the smaller of the two."""
    ctx = get_context_size()
    percent_threshold = int(ctx * AUTOCOMPACT_FRACTION)
    overflow_safe = get_effective_context_window_size() - AUTOCOMPACT_BUFFER_TOKENS
    threshold = min(percent_threshold, overflow_safe)
    # Operator ceiling (config.auto_compact_threshold_tokens): keeps every
    # tenant — the parent AND each sub-agent, which run this same check — small
    # enough that all of them fit the server's KV pool together. "All" used to
    # mean two; on this line it is the parent plus max_concurrent_subagents.
    cfg = get_config()
    cap = int(getattr(cfg, 'auto_compact_threshold_tokens', 0) or 0)
    if cap > 0:
        threshold = min(threshold, cap)
    if is_subagent:
        # Explicit operator value wins; 0 (the default) derives one from the
        # parent's threshold AFTER the operator ceiling, so the sub-agent is a
        # share of what the parent actually gets, not of the raw window.
        sub_cap = int(getattr(cfg, 'subagent_context_ceiling_tokens', 0) or 0)
        if sub_cap <= 0:
            # Two measurements, whichever binds first: what one extra tenant
            # may cost, and what all the concurrent ones may cost together.
            # The second is what makes raising the slot count safe — without
            # it the per-agent share is simply multiplied by the slots and
            # nothing says so.
            slots = max(1, int(getattr(cfg, 'max_concurrent_subagents', 1) or 1))
            share = min(SUBAGENT_THRESHOLD_FRACTION, SUBAGENT_TOTAL_FRACTION / slots)
            sub_cap = int(threshold * share)
        threshold = min(threshold, sub_cap)
    return threshold


def calculate_token_warning_state(token_usage: int) -> Dict[str, Any]:
    """Port of calculateTokenWarningState (auto-compact enabled path)."""
    threshold = get_auto_compact_threshold()
    percent_left = max(0, round(((threshold - token_usage) / threshold) * 100)) if threshold > 0 else 0
    warning_threshold = threshold - WARNING_THRESHOLD_BUFFER_TOKENS
    error_threshold = threshold - ERROR_THRESHOLD_BUFFER_TOKENS
    effective = get_effective_context_window_size()
    blocking_limit = effective - MANUAL_COMPACT_BUFFER_TOKENS
    return {
        'percent_left': percent_left,
        'is_above_warning_threshold': token_usage >= warning_threshold,
        'is_above_error_threshold': token_usage >= error_threshold,
        'is_above_auto_compact_threshold': token_usage >= threshold,
        'is_at_blocking_limit': token_usage >= blocking_limit,
    }


def is_auto_compact_enabled() -> bool:
    return getattr(get_config(), 'auto_compact', True)


def oob_scope(tool_use_context: Any) -> str:
    """The out-of-band scope for a context's tool set (see tokens.oob_scope_key)."""
    from ..utils.tokens import oob_scope_key
    try:
        tools = (tool_use_context.options or {}).get('tools', []) or []
    except Exception:
        tools = []
    return oob_scope_key(t.name for t in tools)


def should_auto_compact(messages: List[Any], is_subagent: bool = False,
                        scope: str = '') -> bool:
    """Port of shouldAutoCompact. Counts via the unified count_text_tokens
    substrate — the served model's own tokenizer when available (exact), else
    the model tokenizer (or its CJK-aware heuristic) — never raw chars/4.

    `is_subagent` selects the tenant's threshold — see get_auto_compact_threshold."""
    if not is_auto_compact_enabled():
        return False
    # tokenCountWithEstimation: last API usage + estimate of everything after it,
    # so the count grows mid-tool-loop and compaction still fires.
    # estimate_input_tokens, NOT token_count_with_estimation: this number is
    # compared against a threshold derived from the model's context window, so
    # it has to be the size of the REQUEST (history + system prompt + tool
    # block), not the size of the history alone. Using the narrower count let a
    # 458,753-token input sit under the threshold and then 400 at
    # 524,289 > 524,288 — compaction never fired, and the output cap that would
    # have caught it downstream cannot help once the INPUT alone is over.
    token_count = estimate_input_tokens(messages, scope)
    threshold = get_auto_compact_threshold(is_subagent)
    debug_log('AUTOCOMPACT_CHECK', {
        'tokens': token_count,
        'threshold': threshold,
        'effective_window': get_effective_context_window_size(),
    })
    return token_count >= threshold


# === Microcompact: content-clear old tool results (microCompact.ts core) ===
#
# Faithful port of maybeTimeBasedMicrocompact's CLEARING mechanism: keep the last
# MICROCOMPACT_KEEP_RECENT compactable tool results, and replace the CONTENT of
# older ones with a marker (structure/pairing preserved). This relieves context
# pressure WITHOUT a destructive full summary — old file reads / command output
# are dropped while recent working context stays granular.
#
# One deliberate deviation (flagged): upstream TRIGGERS this on an idle time-gap
# (server cache cold, measured from message timestamps). codemason messages carry
# no timestamp, and adding that everywhere would be disproportionate, so the
# caller runs this under TOKEN pressure instead (before the full autocompact).
# The ant-only cached-microcompact and the HISTORY_SNIP path are NOT ported
# (cache-editing API / no source in the extract).

MICROCOMPACT_KEEP_RECENT = 5
TIME_BASED_MC_CLEARED_MESSAGE = '[Old tool result content cleared]'
# COMPACTABLE_TOOLS (microCompact.ts): output-heavy read-only tools whose results
# are safe to drop once old.
_COMPACTABLE_TOOLS = {'Read', 'Bash', 'Grep', 'Glob', 'WebSearch', 'WebFetch', 'Edit', 'Write'}


def _battr(b, k, default=None):
    return b.get(k, default) if isinstance(b, dict) else getattr(b, k, default)


def _collect_compactable_tool_ids(messages) -> List[str]:
    """collectCompactableToolIds: tool_use ids of compactable tools, in order."""
    ids: List[str] = []
    for m in messages:
        if _battr(m, 'type') != 'assistant':
            continue
        msg = _battr(m, 'message')
        content = _battr(msg, 'content') if msg is not None else None
        if not isinstance(content, list):
            continue
        for b in content:
            if _battr(b, 'type') == 'tool_use' and _battr(b, 'name') in _COMPACTABLE_TOOLS:
                bid = _battr(b, 'id')
                if bid:
                    ids.append(bid)
    return ids


def _tool_result_tokens(block) -> int:
    """calculateToolResultTokens: rough token estimate of a tool_result block."""
    content = _battr(block, 'content')
    if not content:
        return 0
    if isinstance(content, str):
        return rough_token_count_estimation(content)
    if isinstance(content, list):
        total = 0
        for it in content:
            itype = _battr(it, 'type')
            if itype == 'text':
                total += rough_token_count_estimation(_battr(it, 'text', ''))
            elif itype == 'image':
                from ..utils.image_prep import image_block_tokens
                total += image_block_tokens(it)
            elif itype == 'document':
                total += 2000
        return total
    return 0


def _result_images(block) -> list:
    content = _battr(block, 'content')
    if not isinstance(content, list):
        return []
    return [it for it in content if _battr(it, 'type') == 'image']


def _is_cleared(content) -> bool:
    """A tool result microcompact already cleared: the marker alone, or the
    marker followed by the images it kept."""
    if content == TIME_BASED_MC_CLEARED_MESSAGE:
        return True
    return (isinstance(content, list) and bool(content)
            and _battr(content[0], 'type') == 'text'
            and _battr(content[0], 'text') == TIME_BASED_MC_CLEARED_MESSAGE
            and all(_battr(it, 'type') == 'image' for it in content[1:]))


def microcompact_messages(messages):
    """Clear the content of old compactable tool results (keep the last
    MICROCOMPACT_KEEP_RECENT). Returns (new_messages, tokens_freed); (messages, 0)
    when nothing is cleared."""
    from ..utils.image_prep import image_block_tokens
    compactable = _collect_compactable_tool_ids(messages)
    keep_recent = max(1, MICROCOMPACT_KEEP_RECENT)
    keep = set(compactable[-keep_recent:])
    clear = {i for i in compactable if i not in keep}
    if not clear:
        return messages, 0

    tokens_freed = 0
    out = []
    for m in messages:
        msg = _battr(m, 'message')
        content = _battr(msg, 'content') if msg is not None else None
        if _battr(m, 'type') != 'user' or not isinstance(content, list) or not isinstance(m, dict):
            out.append(m)
            continue
        touched = False
        new_content = []
        for b in content:
            bid = _battr(b, 'tool_use_id')
            if (_battr(b, 'type') == 'tool_result' and bid in clear
                    and not _is_cleared(_battr(b, 'content'))):
                # [ours] Image blocks stay. What they cost is already bounded
                # by the image limit (the oldest go out as one-line labels —
                # _openai_common.images_to_evict), and taking them out here
                # would renumber every later [Image N] the model has notes on
                # and drop the saved PATH their labels give. So only the text
                # is cleared, and only the text counts as freed.
                images = _result_images(b)
                tokens_freed += _tool_result_tokens(b) - sum(image_block_tokens(i) for i in images)
                touched = True
                nb = dict(b) if isinstance(b, dict) else {
                    'type': 'tool_result', 'tool_use_id': bid}
                nb['content'] = ([{'type': 'text', 'text': TIME_BASED_MC_CLEARED_MESSAGE}] + images
                                 if images else TIME_BASED_MC_CLEARED_MESSAGE)
                new_content.append(nb)
            else:
                new_content.append(b)
        if not touched:
            out.append(m)
            continue
        nm = dict(m)
        nmsg = dict(msg)
        nmsg['content'] = new_content
        nm['message'] = nmsg
        out.append(nm)

    if tokens_freed == 0:
        return messages, 0
    return out, tokens_freed


# === Summary prompt (prompt.ts — verbatim) ===

NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)

_DETAILED_ANALYSIS_INSTRUCTION_BASE = """Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""

BASE_COMPACT_PROMPT = f"""Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

{_DETAILED_ANALYSIS_INSTRUCTION_BASE}

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>
"""


def _conversation_language(messages: List[Any]) -> Optional[str]:
    """The language the user has been writing in, from the messages themselves.

    Read here rather than from the request-language set during streaming: by the
    time compaction runs, the last thing detected may be a tool result or an
    English summary, and after the FIRST compaction the only user message left
    is the previous summary. Scanning back over the real user turns is what
    survives repeated compaction.
    """
    from ..utils.language import detect_language
    for m in reversed(messages):
        role = m.get('type') if isinstance(m, dict) else getattr(m, 'type', None)
        if role != 'user':
            continue
        msg = m.get('message') if isinstance(m, dict) else getattr(m, 'message', None)
        content = (msg or {}).get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
        text = ''
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for b in content:
                # Tool results are the tool's words, not the user's — a page of
                # English stdout must not decide what language the user speaks.
                btype = b.get('type') if isinstance(b, dict) else getattr(b, 'type', None)
                if btype == 'text':
                    text += (b.get('text') if isinstance(b, dict) else getattr(b, 'text', '')) or ''
        lang = detect_language(text)
        if lang:
            return lang
    # Latin script carries no signal, so fall back to the configured language —
    # the same precedence effective_language() uses (conversation first, config
    # as the fallback for "always answer in X").
    from ..utils.config import get_config
    return get_config().language or None


def get_compact_prompt(custom_instructions: Optional[str] = None,
                       language: Optional[str] = None) -> str:
    prompt = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT
    # The summary does not describe the conversation, it REPLACES it. Written in
    # English it makes the whole history English, and the model goes on in
    # English for the rest of the session — with nothing left to signal
    # otherwise, since the user's own words are exactly what was just discarded.
    # Every later compaction summarises that English, so it never comes back.
    if language:
        prompt += (
            f"\n\nWrite the summary in {language}. The summary replaces the "
            f"conversation itself, and the conversation is in {language}; an "
            f"English summary would silently switch the language of everything "
            f"that follows. Keep code, identifiers, paths, commands and error "
            f"strings verbatim."
        )
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
    prompt += NO_TOOLS_TRAILER
    return prompt


def format_compact_summary(summary: str) -> str:
    """Strip the <analysis> scratchpad and turn <summary> into a header."""
    formatted = re.sub(r'<analysis>[\s\S]*?</analysis>', '', summary)
    m = re.search(r'<summary>([\s\S]*?)</summary>', formatted)
    if m:
        content = (m.group(1) or '').strip()
        # repl MUST be a function: `content` is model-authored text and may hold
        # \d, \g, C:\Users etc. which re.sub would parse as escapes in a string
        # repl (re.error: bad escape). count=1 matches JS .replace(str,str).
        formatted = re.sub(r'<summary>[\s\S]*?</summary>',
                           lambda _m: f'Summary:\n{content}', formatted, count=1)
    formatted = re.sub(r'\n\n+', '\n\n', formatted)
    return formatted.strip()


#: Opening sentence of every compaction summary.
#:
#: Also what ``session_storage._trim_to_last_compaction`` looks for: a resume
#: must start where the last compaction did, and the transcript records no flag
#: for it. Shared rather than copied, so the two cannot drift apart — a changed
#: wording here would otherwise make every later session replay history the
#: compaction had already thrown away.
COMPACT_CONTINUATION_PREFIX = (
    "This session is being continued from a previous conversation that ran out of context."
)


def get_compact_user_summary_message(summary: str, suppress_follow_up_questions: bool = False) -> str:
    formatted = format_compact_summary(summary)
    base = (
        f"{COMPACT_CONTINUATION_PREFIX} The summary below covers the earlier portion "
        f"of the conversation.\n\n{formatted}"
    )
    if suppress_follow_up_questions:
        return (
            f"{base}\n"
            "Continue the conversation from where it left off without asking the user "
            "any further questions. Resume directly — do not acknowledge the summary, "
            "do not recap what was happening, do not preface with \"I'll continue\" or "
            "similar. Pick up the last task as if the break never happened."
        )
    return base


# === Compaction execution (core of compact.ts compactConversation) ===

COMPACT_SYSTEM_PROMPT = "You are a helpful AI assistant tasked with summarizing conversations."

API_ERROR_MESSAGE_PREFIX = "Error"

# Floor for a summary we are willing to REPLACE the conversation with.
#
# Deliberately low: its only job is to catch text that formats to nothing
# (whitespace-only is truthy, and format_compact_summary trims it to ""). The
# substantive rejections are made on stop_reason, because the strings that must
# not be accepted — the stall note, a partial abort — are long enough to clear
# any length bar worth setting.
MIN_USABLE_SUMMARY_CHARS = 40


@dataclass
class CompactionResult:
    summary: str
    messages: List[Any]            # post-compact messages (replace the conversation)
    summary_response: Any = None


def _extract_text(response: Any) -> Optional[str]:
    """The first TEXT block, wherever it sits. Looking only at content[0]
    made a summary that arrived behind a thinking or tool_use block read as
    "no summary" and fail the compaction outright."""
    content = response.message.content if hasattr(response, 'message') else None
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                return block.get('text', '')
            if hasattr(block, 'type') and block.type == 'text':
                return block.text
    return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Field of a dict-shaped or object-shaped message/block."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def strip_images_for_summary(messages: List[Any]) -> List[Any]:
    """Port of stripImagesFromMessages: the summary request carries no images.

    The summarizer writes text, the images are already in the store, and with
    them the largest request a session sends would also carry up to the
    server's image limit in pixels (each up to VISION_TOKENS_PER_IMAGE_MAX
    tokens). Each image becomes "[Image N: saved at PATH]", numbered in the
    order convert_to_chat_messages numbers them, so the summary can keep the
    path and the model can Read the image again after compaction. Only the
    request copy changes; `messages` is not modified."""
    import copy
    from ..utils.image_store import stored_path

    n = 0
    seen_results: set = set()

    def note(block: Any, numbered: bool = True) -> Dict[str, Any]:
        nonlocal n
        try:
            path = stored_path(_get(block, 'source'))
        except Exception:
            path = None
        where = f': saved at {path}' if path is not None else ''
        if not numbered:
            return {'type': 'text', 'text': f'[Image{where}]'}
        n += 1
        return {'type': 'text', 'text': f'[Image {n}{where}]'}

    def is_image(block: Any) -> bool:
        return _get(block, 'type') == 'image'

    out = []
    for msg in messages:
        inner = _get(msg, 'message')
        content = _get(inner, 'content') if inner is not None else None
        if _get(msg, 'type') != 'user' or not isinstance(content, list):
            out.append(msg)
            continue
        # Numbered in convert_to_chat_messages' order: in a message with tool
        # results, the results' images first (a repeated result is skipped
        # there, so it gets no number here), then images beside them.
        has_results = any(_get(b, 'type') == 'tool_result' for b in content)
        order = ([b for b in content if _get(b, 'type') == 'tool_result']
                 + [b for b in content if is_image(b)]) if has_results else content
        notes: Dict[int, Any] = {}
        for b in order:
            if is_image(b):
                notes[id(b)] = note(b)
            elif _get(b, 'type') == 'tool_result' and isinstance(_get(b, 'content'), list):
                tid = _get(b, 'tool_use_id')
                first = tid not in seen_results
                seen_results.add(tid)
                for x in _get(b, 'content'):
                    if is_image(x):
                        notes[id(x)] = note(x, numbered=first)
            elif _get(b, 'type') == 'tool_result':
                seen_results.add(_get(b, 'tool_use_id'))
        changed = False
        blocks = []
        for b in content:
            if is_image(b):
                blocks.append(notes[id(b)])
                changed = True
            elif _get(b, 'type') == 'tool_result' and isinstance(_get(b, 'content'), list) \
                    and any(is_image(x) for x in _get(b, 'content')):
                tr = dict(b) if isinstance(b, dict) else copy.copy(b)
                items = [notes[id(x)] if is_image(x) else x for x in _get(b, 'content')]
                if isinstance(tr, dict):
                    tr['content'] = items
                else:
                    tr.content = items
                blocks.append(tr)
                changed = True
            else:
                blocks.append(b)
        if not changed:
            out.append(msg)
            continue
        new_inner = dict(inner) if isinstance(inner, dict) else copy.copy(inner)
        new_msg = dict(msg) if isinstance(msg, dict) else copy.copy(msg)
        if isinstance(new_inner, dict):
            new_inner['content'] = blocks
        else:
            new_inner.content = blocks
        if isinstance(new_msg, dict):
            new_msg['message'] = new_inner
        else:
            new_msg.message = new_inner
        out.append(new_msg)
    return out


async def compact_conversation(
    messages: List[Any],
    tool_use_context: Any,
    custom_instructions: Optional[str] = None,
    is_auto_compact: bool = True,
    on_token_update: Optional[Any] = None,
    on_thinking_update: Optional[Any] = None,
) -> CompactionResult:
    """Query the model for a summary and build the post-compact message list.

    The callbacks drive the caller's spinner. Without them this is the longest
    silent wait in a session — the whole conversation is re-prefilled and
    summarised with nothing on screen, which is indistinguishable from a hang,
    and the bigger the context (i.e. the more you needed to compact) the longer
    it looks frozen.
    """
    from .backend import query_model
    from ..utils.messages import create_user_message

    _lang = _conversation_language(messages)
    summary_request = create_user_message(get_compact_prompt(custom_instructions, _lang))

    signal = getattr(tool_use_context, 'abort_controller', None)
    model = None
    if hasattr(tool_use_context, 'options'):
        model = tool_use_context.options.get('slow_and_capable_model')

    # Cap the summary output at the reserved budget (get_effective_context_window_size
    # already subtracts MAX_OUTPUT_TOKENS_FOR_SUMMARY on the assumption the summary
    # won't exceed it — enforce that here).
    summary_output_cap = min(MAX_OUTPUT_TOKENS_FOR_SUMMARY, get_config().max_tokens)

    # PTL retry loop: if the summary request overflows context, truncate the
    # oldest rounds and retry (up to MAX_PTL_RETRIES) instead of failing straight
    # into the circuit breaker with the session stuck over-limit.
    conv = strip_images_for_summary(list(messages))
    attempt = 0
    while True:
        response = await query_model(
            messages=conv + [summary_request],
            system_prompt=[COMPACT_SYSTEM_PROMPT],
            tools=[],  # NO tools during summarization
            signal=signal,
            # no_reasoning: summarising is not a problem to deliberate over, and
            # the reasoning pass runs before a single summary token appears —
            # doubling the one wait the user is already staring at. The effort
            # level is a setting for the work, not for the bookkeeping around it.
            options={'dangerously_skip_permissions': False, 'model': model,
                     'max_tokens': summary_output_cap, 'no_reasoning': True},
            on_token_update=on_token_update,
            on_thinking_update=on_thinking_update,
        )
        summary = _extract_text(response)
        # Accepting a summary REPLACES the entire conversation, so a wrong accept
        # is unrecoverable. The old gate was "non-empty and does not start with
        # Error", which also accepts:
        #   - the stall guard's "\n\n[Stopped: ...]" text (stop_reason
        #     'stalled'). Very reachable here: this is the largest request the
        #     session ever sends, and it runs on a cold prefix.
        #   - ESC during "Compacting…" — it shares the turn's abort event and
        #     yields whatever partial text arrived as a clean end_turn.
        #   - a summary cut at the output cap ('max_tokens'), which has no
        #     </analysis> and no <summary>, so the raw SCRATCHPAD would become
        #     the new history.
        #   - whitespace-only text: truthy, formats to "".
        # A rejected summary falls through to the PTL retry (which must keep
        # seeing the raw text) and then to the raise — the pre-existing path.
        _stop = getattr(getattr(response, 'message', None), 'stop_reason', None)
        _usable_stop = (
            _stop not in ('stalled', 'max_tokens', 'empty_turn', 'repetition')
            and not getattr(response, 'is_api_error_message', False)
        )
        if (summary and _usable_stop
                and not summary.startswith(API_ERROR_MESSAGE_PREFIX)
                and len(format_compact_summary(summary)) >= MIN_USABLE_SUMMARY_CHARS):
            break
        if attempt < MAX_PTL_RETRIES and summary and _looks_like_prompt_too_long(summary):
            truncated = _truncate_head_for_ptl(conv, summary)
            if truncated is not None:
                conv = truncated
                attempt += 1
                debug_log('COMPACT_PTL_RETRY', {'attempt': attempt, 'kept_messages': len(conv)})
                continue
        # Name the reason, and put it on disk. Compaction failure used to be
        # entirely silent (the caller prints nothing when was_compacted is
        # False), so three of these in a row tripped the circuit breaker with no
        # record anywhere of what went wrong.
        _why = (
            f"compaction rejected the summary (stop_reason {_stop or 'none'}, "
            f"{len(summary) if summary else 0} chars)"
            + (", server error" if getattr(response, 'is_api_error_message', False) else "")
        )
        log_error(_why, 'compaction: no usable summary')
        raise RuntimeError(f"Compaction failed: {_why}")

    user_summary = get_compact_user_summary_message(summary, suppress_follow_up_questions=is_auto_compact)
    post_compact_messages = [create_user_message(user_summary)]

    return CompactionResult(summary=summary, messages=post_compact_messages, summary_response=response)


async def auto_compact_if_needed(
    messages: List[Any],
    tool_use_context: Any,
    tracking: Optional[Dict[str, Any]] = None,
    on_token_update: Optional[Any] = None,
    on_thinking_update: Optional[Any] = None,
    context_proven_fatal: bool = False,
) -> Dict[str, Any]:
    """Port of autoCompactIfNeeded with the consecutive-failure circuit breaker.

    Callbacks are forwarded to compact_conversation for the caller's spinner —
    auto-compaction needs them more than the manual command does, since it
    happens without being asked and its silence has no obvious cause at all.

    `context_proven_fatal` — the caller has just watched the decoder collapse on
    this exact history (query.py's repetition recovery, and nothing else).

    It exists because the threshold is the wrong authority in that one case. The
    threshold predicts whether a history is too big; a collapse is the
    observation that it was, and a prediction does not get to overrule the
    measurement it was predicting. Without it the recovery path asked
    should_auto_compact for permission to shrink a context that had just killed
    the turn, was told there was room, and re-sent it unchanged — seven times,
    every attempt collapsing in the same place.

    Under it ONLY microcompact runs — the free, local, non-destructive one. The
    full summary is deliberately not reachable this way: its own request carries
    the same history that was just proven fatal, so it would likely collapse
    too, and MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES failed summaries trip a
    one-way breaker that ends compaction for the rest of the session. Reducing
    the context must not be able to destroy the ability to reduce the context.
    """
    tracking = tracking or {}
    consecutive = tracking.get('consecutive_failures', 0)
    if consecutive >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES:
        # The breaker is one-way: only a SUCCESSFUL compaction resets the
        # counter, and once it is tripped no compaction is attempted, so the
        # reset can never happen. Every later turn then sends the full,
        # still-growing history until the server rejects it. Nothing recorded
        # that this had happened, which is why an over-context session looked
        # like a model problem. Log it once per turn it suppresses, at the one
        # place that knows — the caller only sees an empty outcome.
        log_error(
            f"auto-compaction suppressed: {consecutive} consecutive failures "
            f"(limit {MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES}) — the history is no "
            "longer being reduced and only /clear or a successful /compact recovers",
            'compaction: breaker open',
        )
        return {'was_compacted': False}

    # A sub-agent (agent_id set by the Agent tool / forked skills; None for the
    # user's own conversation) gets the tighter ceiling.
    is_subagent = getattr(tool_use_context, 'agent_id', None) is not None
    _scope = oob_scope(tool_use_context)
    if not should_auto_compact(messages, is_subagent, _scope):
        if not context_proven_fatal:
            return {'was_compacted': False}
        # Proven fatal but under the threshold. Microcompact unconditionally —
        # see context_proven_fatal above for why this is the only step taken
        # here, and why it is taken without asking.
        #
        # Not gated on the threshold, but still gated on there being something
        # to clear: microcompact_messages is idempotent (an already-cleared
        # result is skipped), so a history whose old tool results are all
        # cleared frees 0, and reporting 'was_microcompacted' for a set that did
        # not change would report progress the caller then recurses on.
        cleared, freed = microcompact_messages(messages)
        debug_log('MICROCOMPACT_PROVEN_FATAL', {
            'tokens_freed': freed,
            'threshold': get_auto_compact_threshold(is_subagent),
            'exhausted': freed == 0,
        })
        if freed == 0:
            # Nothing left to clear locally, and the full summary is not
            # reachable from here (see the docstring). The turn retries on an
            # unreduced history and will most likely collapse again.
            log_error(
                'the decoder collapsed on this history and microcompact had '
                'nothing left to clear — context was not reduced before the retry',
                'compaction: proven-fatal, exhausted',
            )
            return {'was_compacted': False}
        return {'was_compacted': False, 'was_microcompacted': True, 'messages': cleared}

    # Microcompact first: content-clear old tool results (keep the recent tail)
    # and see if that alone drops us under the threshold. If so, skip the
    # destructive full summary and keep granular context (microCompact.ts intent:
    # "if collapse gets us under the autocompact threshold, autocompact is a
    # no-op"). Old results are pre-anchor, so the usage-anchored count can't
    # see the saving — subtract tokens_freed explicitly (upstream's snipTokensFreed
    # plumbing).
    cleared, freed = microcompact_messages(messages)
    if freed > 0:
        threshold = get_auto_compact_threshold(is_subagent)
        # Same count the trigger used (estimate_input_tokens, the request
        # size), so "relieved" is judged with the ruler that said "over".
        effective = estimate_input_tokens(messages, _scope) - freed
        debug_log('MICROCOMPACT', {
            'tokens_freed': freed, 'effective': effective, 'threshold': threshold,
            'relieved': effective < threshold,
        })
        if effective < threshold:
            return {'was_compacted': False, 'was_microcompacted': True, 'messages': cleared}
        messages = cleared  # still over -> full-summarize the smaller, cleared set

    # Only around the SUMMARY. Microcompaction returns above in microseconds, so
    # pinning a label for it would flash something nobody can read; the wait
    # note set earlier already covers that case. A summary is an API round trip
    # on the whole conversation -- the longest wait in a session -- and that is
    # the one worth renaming the line for.
    _on_status = getattr(tool_use_context, 'on_status', None)
    if callable(_on_status):
        try:
            _on_status('Compacting')
        except Exception:
            pass
    try:
        result = await compact_conversation(
            messages, tool_use_context, is_auto_compact=True,
            on_token_update=on_token_update, on_thinking_update=on_thinking_update)
        # Post-compact file reconstruction (compact.ts): the summary does NOT carry
        # full file contents, so (1) CLEAR readFileState — otherwise Edit/Write's
        # read-before-edit gate would pass on a file the model no longer has, a
        # stale write — then (2) re-inject the most recent files fresh and re-mark
        # them read, so work continues without a re-read round-trip.
        _rfs = getattr(tool_use_context, 'read_file_state', None)
        if isinstance(_rfs, dict):
            pre = dict(_rfs)
            _rfs.clear()
            import os as _os
            from ..types.query import ReadFileState
            for path, full, msg in _post_compact_file_attachments(pre):
                result.messages.append(msg)
                try:
                    _rfs[path] = ReadFileState(
                        timestamp=_os.path.getmtime(path), content=full, offset=None, limit=None
                    )
                except OSError:
                    pass
        # The summary does not carry the spawn either: without this the model
        # comes out of the compaction having forgotten its agents and spawns
        # them again. Appended after the file restoration, as the source builds
        # both attachment sets for the same post-compact message list.
        result.messages.extend(create_async_agent_attachments_if_needed(tool_use_context))
        return {'was_compacted': True, 'compaction_result': result, 'consecutive_failures': 0}
    except Exception as e:
        debug_log('AUTOCOMPACT_FAILED', {'error': str(e)})
        return {'was_compacted': False, 'consecutive_failures': consecutive + 1}
    finally:
        # Hand the label back on every exit, success or failure, so the turn
        # never keeps wearing a name that belongs to a finished compaction.
        if callable(_on_status):
            try:
                _on_status(None)
            except Exception:
                pass
