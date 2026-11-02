"""
OpenAI-compatible model service for CodeMason — shared implementation.

Both backends (llama.cpp and vLLM) speak the same OpenAI /v1/chat/completions
protocol, so all of the logic — message conversion, context management, leaked
tool-call recovery, the streaming loop, live token counting, body encryption —
lives HERE and is server-agnostic. The two thin shims that import it,
services/llamacpp.py and services/vllm.py, differ by exactly one request field:

  - vLLM sets stream_options.continuous_usage_stats -> a running
    completion_tokens on every chunk, so the live spinner shows the server's
    exact count instead of a per-chunk tally.
  - llama.cpp emits usage only in the final chunk, so it leaves the flag off and
    relies on the per-chunk tally until that final exact count arrives.

Callers never import from here directly; they go through services/backend.py,
which routes to the right shim based on the detected/configured server type.
"""

import collections
import codecs
import json
import os
import re
import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4
from dataclasses import dataclass

import httpx

from ..types.messages import (
    Message as APIMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    BetaUsage,
)
from ..types.query import (
    UserMessage,
    AssistantMessage,
)
from ..utils.log import debug_log, log_error, debug_enabled
from ..utils.generators import abortable
from ..utils import tokens as _tokens_mod
from ..utils.tokens import count_text_tokens, has_model_tokenizer


# === Configuration ===

from ..utils.config import get_config as _get_config

THINKING_MODE = os.getenv('THINKING_MODE', 'false').lower() == 'true'

# Tool result truncate settings
MAX_TOOL_RESULT_LENGTH = 30000

# Headroom kept between (estimated prompt + requested output) and the model's
# context window, to absorb the drift between our token estimate and the
# server's own tokenizer (which counts a little more — the off-by-one that made
# a 458753+65536 request overflow a 524288 window). Output is capped to fit
# within this margin, never below _MIN_OUTPUT_TOKENS.
_CONTEXT_SAFETY_MARGIN = 512
_MIN_OUTPUT_TOKENS = 512

# Minimum seconds between exact re-counts of the reasoning text (see
# _recount_thinking). The spinner repaints at 10fps, so a 0.2s floor keeps the
# thinking counter visibly live while capping the re-tokenise cost.
_THINK_COUNT_EVERY = 0.2

# Reasoning-token mark at which a turn that has produced NO content is recorded
# in the error log. This only records; it never cuts.
#
# Volume alone cannot tell a runaway from a hard question. The largest
# legitimate reasoning pass on record here is 26,271 tokens (a hard question at
# reasoning_effort=high, see config.max_tokens), and the runaways observed
# 2026-09-13 reached 65,536 — a factor of two, not a cliff. Both of those turns
# were stopped by the server's own max_tokens rather than by finishing, so
# neither says where the thinking would have ended. Throwing away an hour of
# reasoning on a guess is the more expensive mistake; what matters is carrying
# it into the recovery request, which the clear_thinking gate now does.
_SILENT_REASONING_TOKENS = 40_000

# Context management — auto-detected from server via tokens.py
# Size is probed lazily: vLLM /v1/models (max_model_len) or llama.cpp /slots (n_ctx)
from ..utils.tokens import (
    estimate_input_tokens,
    oob_scope_key,
    set_out_of_band_tokens,
    has_usage_anchor,
    get_context_size,
    get_max_tokens,
    get_output_reserve,
    token_count_with_estimation,
)


async def _stall_guarded(aiter, first_secs: float, gap_secs: float,
                         started_fn, on_stall):
    """Yield from `aiter`, cutting it if it goes quiet for too long.

    A request that produces NOTHING is a different failure from one that starts
    and then goes silent, so they get separate budgets: the first is generous
    because prefill scales with the uncached prompt, the second is tight because
    a stream in flight has no reason to pause. Either way `on_stall` is called
    and iteration ends; the caller decides what that means for the turn.
    """
    it = aiter.__aiter__()
    try:
        while True:
            started = started_fn()
            budget = gap_secs if started else first_secs
            try:
                if budget and budget > 0:
                    item = await asyncio.wait_for(it.__anext__(), timeout=budget)
                else:
                    item = await it.__anext__()
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                on_stall(started, budget)
                return
            yield item
    finally:
        # Close what we wrapped. abortable() closes THIS generator, which does
        # not reach the inner one; left open, its finalizer ran at loop
        # teardown as "Task was destroyed but it is pending".
        aclose = getattr(it, 'aclose', None)
        if aclose is not None:
            try:
                await aclose()
            except BaseException:
                pass


def _write_cut_dump(kind: str, why: str, reasoning: str, content: str,
                    elapsed_s: float) -> None:
    """Write what a cut generation had produced to logs/<kind>-<ts>.txt.

    Same shape as the aborted-turn dump and for the same reason: the output is
    about to be dropped from the conversation on purpose, and dropping it from
    the record too is how a 2,500-second stall ends up needing a core dump to
    explain. Best effort — a failure here is logged, never raised.
    """
    try:
        from ..utils.config import LOG_DIR
        from ..utils.file import atomic_write_bytes
        from ..utils.log import ensure_log_dir, now_iso
        # ensure_log_dir and atomic_write_bytes, not a bare mkdir and write_text:
        # the bare pair lands the directory 0755 and the file 0644, and these
        # dumps hold a generation's whole reasoning and content — the same
        # request material every other file in logs/ is 0600 for. The atomic
        # write also means a reader never catches a half-written dump.
        ensure_log_dir()
        name = '%s-%s.txt' % (kind, now_iso().replace(':', '-').replace('.', '-'))
        atomic_write_bytes(
            str(LOG_DIR / name),
            ('cut after %.1fs\nreason: %s\nreasoning: %d chars\ncontent: %d chars\n'
             '\n===== REASONING =====\n%s\n\n===== CONTENT =====\n%s\n'
             % (elapsed_s, why, len(reasoning), len(content), reasoning, content)
             ).encode('utf-8'))
    except Exception as e:
        log_error('could not write the %s dump: %s' % (kind, e),
                  '_openai_common: cut generation')


# Window for the SINGLE-character case in _DegenerateRunGuard. A collapse onto
# one token is the only shape ever actually observed here, and real work does
# not repeat one character anywhere near this far (measured max 40), so this
# tier catches it in a fraction of the time the two-character window needs:
# ~12s against ~126s on the turns measured 2026-09-13.
SINGLE_CHAR_RUN_CHARS = 200


class _DegenerateRunGuard:
    """Abort a generation whose recent output has collapsed to almost no
    distinct characters.

    Observed 2026-09-10, twice: GLM-5.3 collapsed onto one token and emitted it
    until max_tokens — 53,859 consecutive '!' on the reasoning channel with no
    answer text at all. Nothing stopped it. The turn took 63 minutes, and the
    max_tokens recovery then ran it three more times, for 4h12m of a session
    spent generating one character.

    The server-side n-gram detector is not the answer here. It is off for GLM on
    purpose (Config.repetition_detection_enabled): enumerating an API holds a
    command skeleton fixed and varies one field, so byte-identical tool calls
    are ORDINARY there, and turning it on cut a live probe mid-enumeration on
    2026-09-06. That detector judges a loop on n-gram shape, which real work can
    share.

    This one asks a different question — how many DISTINCT characters are in the
    recent output — and real work cannot answer it the same way. Two distinct
    characters across two thousand of them is not enumeration, not a table, not
    base64, not minified code, not CJK prose. It is a stuck decoder. So the
    guard is family-independent and always on.
    """

    __slots__ = ('window', 'buf', 'counts', 'tripped', 'last_output_at')

    def __init__(self, window: int):
        self.window = max(0, int(window or 0))
        self.buf = collections.deque()
        self.counts = {}
        self.tripped = None
        # When the last OUTPUT delta arrived. None until the first one does,
        # which is what separates "still prefilling" from "started, then went
        # silent". Recorded even when the collapse guard is off: the stall
        # guards need it, and that is a separate question.
        self.last_output_at = None

    def note_progress(self) -> None:
        """Record that the server made progress, WITHOUT feeding the window.

        Every delta that carries characters — answer, reasoning, tool-call
        arguments — goes through feed(), which also fills the distinct-character
        window. vLLM's continuous_usage_stats running completion_tokens is the
        one source with no text behind it: it proves the server decoded
        something, but a synthetic character would poison a guard whose whole
        test is how many DISTINCT characters the recent output holds. So it
        marks progress here, and only that.

        Measured 2026-09-12: a turn whose entire output was tool-call arguments
        had the server counting 40 completion tokens while not one delta of
        either kind arrived. The spinner climbed (it reads the usage counter)
        and last_output_at stayed None, so the read loop kept the 600s
        first-token budget instead of the 120s gap budget and the turn sat
        wedged for 603.1s. The screen said "generating"; the guard said
        "nothing has ever arrived".
        """
        self.last_output_at = time.monotonic()

    def feed(self, text: str) -> None:
        if text:
            self.last_output_at = time.monotonic()
        if not self.window or self.tripped is not None or not text:
            return
        for ch in text:
            self.buf.append(ch)
            self.counts[ch] = self.counts.get(ch, 0) + 1
            if len(self.buf) > self.window:
                old = self.buf.popleft()
                n = self.counts.get(old, 0) - 1
                if n <= 0:
                    self.counts.pop(old, None)
                else:
                    self.counts[old] = n
            # Two tiers, because one distinct character and two are not the
            # same claim. Measured 2026-09-13 over 449 text blocks of this
            # deployment's own transcripts: the longest run real work produces
            # is 40 characters when they are all ONE character, and 47 when two
            # are allowed. Every collapse actually observed — eight in one
            # session that day, plus the 53,859 in the class docstring — was a
            # single character. So the single-character case can trip 10x
            # sooner (~12s against ~126s on the turns measured) without coming
            # near anything real, while the two-character case keeps the wide
            # window that protects rules, indentation and separator lines.
            _single = min(SINGLE_CHAR_RUN_CHARS, self.window)
            if len(self.counts) <= 1:
                _tripped = len(self.buf) >= _single
                _n = _single
            else:
                _tripped = len(self.buf) >= self.window and len(self.counts) <= 2
                _n = self.window
            if _tripped:
                seen = ', '.join(sorted(repr(c) for c in self.counts))
                self.tripped = (
                    'the model stopped making progress: the last %d characters '
                    'of output contain only %d (%s). Generation was cut here.'
                    % (_n, len(self.counts), seen))
                return


def _repetition_detection_params(cfg: Any) -> Optional[Dict[str, int]]:
    """Build vLLM's RepetitionDetectionParams from config, or None if disabled.

    Measured against the live server (2026-08-06): a forced 40x repetition was
    cut at ~140 tokens with finish_reason='repetition', while a code answer with
    the same parameter names across three classes and a 25-row table written in
    one deliberately uniform style both ran to the full budget untouched under
    every setting tried. The failure mode this guards is loud; legitimate
    repetition is not n-gram-identical.
    """
    # Model-gated: the detector's premise does not hold for every model.
    # See Config.repetition_detection_enabled.
    _enabled = getattr(cfg, 'repetition_detection_enabled', None)
    if callable(_enabled) and not _enabled():
        return None
    lo = int(getattr(cfg, 'repetition_min_pattern', 0) or 0)
    hi = int(getattr(cfg, 'repetition_max_pattern', 0) or 0)
    n = int(getattr(cfg, 'repetition_min_count', 0) or 0)
    if hi <= 0 or n < 2:
        return None
    return {'min_pattern_size': lo, 'max_pattern_size': hi, 'min_count': n}


def count_tokens_precise(text: Optional[str]) -> int:
    """Count tokens precisely. Prefers the served model's own tokenizer (exact,
    matches the server) and falls back to the CJK-aware heuristic via
    count_text_tokens."""
    if not text:
        return 0
    try:
        return count_text_tokens(text)
    except Exception:
        # Fallback to rough estimate
        return len(text) // 3


# === Utility Functions ===

def truncate_tool_result(content: str) -> str:
    """Truncate tool result if too long, keeping start and end"""
    if len(content) <= MAX_TOOL_RESULT_LENGTH:
        return content

    half_length = MAX_TOOL_RESULT_LENGTH // 2
    start = content[:half_length]
    end = content[-half_length:]
    truncated_lines = content[half_length:-half_length].count('\n')

    return f"{start}\n\n... [{truncated_lines} lines truncated] ...\n\n{end}"


def get_simple_description(tool_name: str) -> str:
    """Get simple description for a tool"""
    descriptions = {
        'dispatch_agent': 'Launch agent for complex searches',
        'Agent': 'Launch a sub-agent for batch operations or open-ended searches. IMPORTANT: write the agent prompt in the language the user is writing in, whatever that language is. Never default to English.',
        'Architect': 'Analyze technical requirements',
        'Bash': 'Execute bash commands',
        'Glob': 'Find files by pattern',
        'Grep': 'Search file contents',
        'List': 'List directory contents',
        'Read': 'Read file contents',
        'Edit': 'Edit existing files',
        'Write': 'Create or overwrite files',
        'ReadNotebook': 'Read Jupyter notebooks',
        'NotebookEditCell': 'Edit notebook cells',
        'TodoWrite': 'Create task lists',
        'TodoRead': 'Read task lists',
    }
    return descriptions.get(tool_name, f'Use the {tool_name} tool')


def clean_special_tokens(content: str, reasoning_separated: bool = False) -> str:
    """Clean special tokens from model output.

    reasoning_separated: True when the server already delivered this turn's
    thinking via the separate `reasoning_content` channel. In that case any
    <think>…</think> left in `content` is NOT thinking — it's a leaked fragment
    where the model wrapped the head of its own answer in stray tags (e.g.
    `<think>a stray thought</think>the answer continues…`). Deleting the block then truncates
    the answer. So we strip only the TAGS and keep the inner text. Full block
    deletion is reserved for servers that inline ALL reasoning (no separate
    channel) under THINKING_MODE."""
    if not content:
        return content

    cleaned = content

    # Remove gpt-oss style channel/message tokens
    cleaned = re.sub(r'<\|channel\|>[^<]*<\|message\|>[\s\S]*?<\|end\|>', '', cleaned, flags=re.IGNORECASE)

    # Remove any remaining special tokens like <|...|>
    cleaned = re.sub(r'<\|[^|]+\|>', '', cleaned)

    # Handle <think> tags. With a separate reasoning channel, never delete the
    # enclosed text (it's answer content) — drop the tags only. Otherwise fall
    # back to THINKING_MODE: strip whole inline-think blocks, or just empty ones.
    if reasoning_separated:
        # Drop the stray tags but keep the text. Collapse each tag plus the
        # horizontal whitespace touching it into a single space — the tag often
        # sits where a word break belongs (`word</think>next` -> `word next`).
        # Matching only AT the tag site leaves newlines and code-block
        # indentation everywhere else untouched.
        cleaned = re.sub(r'[ \t]*</?think>[ \t]*', ' ', cleaned, flags=re.IGNORECASE)
    elif THINKING_MODE:
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', cleaned, flags=re.IGNORECASE)
    else:
        cleaned = re.sub(r'<think>\s*</think>', '', cleaned, flags=re.IGNORECASE)

    # Remove any JSON tool calls from the text (MiniMax specific)
    cleaned = re.sub(r'\[?\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{[^}]*\}\s*\}\s*\]?', '', cleaned)

    return cleaned.strip()


# ── Leaked tool-call recovery ────────────────────────────────────────────────
# DeepSeek-V4 intermittently SERIALISES a tool call into the assistant `content`
# instead of emitting it on the structured tool_calls channel (deepseek-ai
# DeepSeek-V3#1244; vLLM#41240). finish_reason comes back 'stop', tool_calls is
# empty, and the invocation would be silently lost — the model then narrates as
# if it ran, fabricating output. DeepSeek ships no fix and no official Jinja
# template, so recovery is on us: parse the leaked markup back into real tool
# calls so NOT ONE invocation is dropped.
#
# Two leak shapes are handled:
#   1. DSML (the native V4 tool markup) — XML with the FULL-WIDTH pipe ｜ (U+FF5C):
#        <｜DSML｜tool_calls><｜DSML｜invoke name="X">
#          <｜DSML｜parameter name="k" string="true">v</｜DSML｜parameter>
#        </｜DSML｜invoke></｜DSML｜tool_calls>
#      We also accept the half-width | as a tokenizer variant, and tolerate a
#      stream truncated before the closing tag.
#   2. Bare JSON — {"name": "...", "arguments"|"parameters": {...}} — some builds
#      leak this shape instead (the MiniMax-style fallthrough).
_PIPE = r'[|｜]'
_DSML = r'<' + _PIPE + r'DSML' + _PIPE
_DSML_C = r'</' + _PIPE + r'DSML' + _PIPE
_RE_DSML_INVOKE = re.compile(
    _DSML + r'invoke\s+name="([^"]+)"\s*>(.*?)' + _DSML_C + r'invoke\s*>',
    re.DOTALL)
_RE_DSML_INVOKE_OPEN = re.compile(   # truncated: opened invoke, no closing tag
    _DSML + r'invoke\s+name="([^"]+)"\s*>(.*)\Z', re.DOTALL)
_RE_DSML_PARAM = re.compile(
    _DSML + r'parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)'
    + _DSML_C + r'parameter\s*>', re.DOTALL)
_RE_DSML_PARAM_OPEN = re.compile(    # truncated final parameter, no closing tag
    _DSML + r'parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*)\Z',
    re.DOTALL)
_RE_DSML_ANY_TAG = re.compile(r'</?' + _PIPE + r'DSML' + _PIPE + r'[^>]*>')
_RE_DSML_WRAP_SPAN = re.compile(
    _DSML + r'tool_calls\s*>.*?(?:' + _DSML_C + r'tool_calls\s*>|\Z)', re.DOTALL)


def _parse_dsml_params(body: str) -> Dict[str, Any]:
    """Parse <｜DSML｜parameter …> elements inside one invoke body into a dict.
    string="true" keeps the raw value; string="false" is JSON (fallback: raw)."""
    args: Dict[str, Any] = {}
    for name, is_str, val in _RE_DSML_PARAM.findall(body):
        if is_str == 'true':
            args[name] = val
        else:
            try:
                args[name] = json.loads(val.strip())
            except Exception:
                args[name] = val.strip()
    # A stream cut off mid-parameter leaves one unterminated element; recover it
    # too, but only if it wasn't already captured by the closed-tag pass above.
    mo = _RE_DSML_PARAM_OPEN.search(body)
    if mo and mo.group(1) not in args:
        name, is_str, val = mo.group(1), mo.group(2), mo.group(3)
        # strip a dangling partial open-tag of the next element if present
        val = _RE_DSML_ANY_TAG.sub('', val)
        if is_str == 'true':
            args[name] = val
        else:
            try:
                args[name] = json.loads(val.strip())
            except Exception:
                args[name] = val.strip()
    return args


def _scan_json_tool_calls(text: str):
    """Yield (name, args_dict, start, end) for every bare JSON tool-call object
    in text. Uses raw_decode (not regex) so nested braces parse correctly."""
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        s = text.find('{', i)
        if s == -1:
            break
        try:
            obj, end = dec.raw_decode(text, s)
        except Exception:
            i = s + 1
            continue
        if isinstance(obj, dict) and 'name' in obj and \
           ('arguments' in obj or 'parameters' in obj):
            args = obj.get('arguments', obj.get('parameters'))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(args, dict):
                yield str(obj['name']), args, s, end
        i = end


def recover_leaked_tool_calls(content: str):
    """Extract tool calls that leaked into `content`.

    Returns (recovered, residual_text):
      recovered      -> list of {'name': str, 'input': dict}
      residual_text  -> content with the leaked markup removed (the real answer)

    No-op (returns [], content) when no leak markers are present, so the normal
    path — where the model used the structured tool_calls channel — is untouched.
    """
    if not content:
        return [], content
    has_dsml = 'DSML' in content
    has_json = '"name"' in content and ('"arguments"' in content or '"parameters"' in content)
    if not has_dsml and not has_json:
        return [], content

    recovered: List[Dict[str, Any]] = []
    text = content

    if has_dsml:
        matched = False
        for m in _RE_DSML_INVOKE.finditer(text):
            recovered.append({'name': m.group(1),
                              'input': _parse_dsml_params(m.group(2))})
            matched = True
        if matched:
            text = _RE_DSML_INVOKE.sub('', text)
        # A trailing invoke can be truncated by a stream cut whether or not any
        # fully-closed invoke preceded it. Check the residual for an unterminated
        # one in BOTH cases (previously this ran only when NO closed invoke
        # matched, so a truncated invoke AFTER >=1 complete invoke was dropped).
        mo = _RE_DSML_INVOKE_OPEN.search(text)
        if mo:
            recovered.append({'name': mo.group(1),
                              'input': _parse_dsml_params(mo.group(2))})
            text = text[:mo.start()]
        # Strip any leftover wrapper / stray DSML tags from the residual answer.
        text = _RE_DSML_WRAP_SPAN.sub('', text)
        text = _RE_DSML_ANY_TAG.sub('', text)

    # Only fall back to bare-JSON scanning when DSML yielded nothing, so a normal
    # DSML answer that happens to quote JSON isn't mis-scavenged.
    if not recovered and has_json:
        spans = list(_scan_json_tool_calls(text))
        for name, args, _s, _e in spans:
            recovered.append({'name': name, 'input': args})
        for _name, _args, s, e in reversed(spans):
            text = text[:s] + text[e:]

    return recovered, text


def build_system_prompt_with_cwd_emphasis(system_prompt: List[str]) -> str:
    """Join the system prompt blocks. The name is kept; the cwd emphasis is NOT.

    This used to put the live cwd in the first bytes of the system prompt —
    ahead of everything — and rebuild it on EVERY request, tool-loop turns
    included. The system prompt is the cached prefix, so a single `cd` moved
    byte ~40 of the request and the server re-prefilled the entire
    conversation. Measured directly on this backend at 96k tokens: 72.1s when
    the front changes, against 3.4s for the same information appended to the
    end. The cost scales with the whole context, so at 200k it is minutes.

    (A 146.4s stall observed live while diagnosing this was NOT this bug — cwd
    had not moved for that request. It was almost certainly the diagnosis
    itself evicting the session's prefix with test traffic. Recorded so the
    number is not later mistaken for evidence of this code path.)

    The emphasis was there so the model would not lose track of where "." is,
    which is a real need — the model does need the CURRENT directory, not the
    one the session opened in. It now rides the turn (cli.py's <turn-env>),
    where it is appended and costs nothing. Nothing is lost: the value is still
    per-turn and still current, and <env> continues to state the session's
    starting directory once.

    Do not put it back. Anything that varies within a session belongs after the
    history, never before it.
    """
    return '\n\n'.join(system_prompt)


# === Chat Message Types ===

@dataclass
class ChatMessage:
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    # Thinking-mode round-trip. DeepSeek-V4's thinking API requires the prior
    # turn's reasoning to come back on every assistant message that made a tool
    # call; without it the model loses the conclusion it reached while thinking
    # and re-derives it from the same tool output, which is how an agent ends up
    # issuing the identical command turn after turn. Only set when the backend
    # asked for the round-trip (vLLM) — see reasoning_roundtrip. Serialized
    # under both wire names (to_dict).
    reasoning_content: Optional[str] = None
    # Multimodal content. OpenAI's chat API takes `content` as EITHER a string
    # OR an array of typed parts; images only exist in the array form. Keeping
    # them in a separate field means every text-only message serializes exactly
    # as before (a bare string), so enabling vision changes no existing request.
    content_parts: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {'role': self.role}
        if self.content_parts:
            result['content'] = self.content_parts
        elif self.content is not None:
            result['content'] = self.content
        if self.reasoning_content:
            # vLLM reads `reasoning` (0.26 parses only that key and ignores
            # `reasoning_content`, which it lists as deprecated); older servers
            # and the DeepSeek API read `reasoning_content`. Send both.
            result['reasoning'] = self.reasoning_content
            result['reasoning_content'] = self.reasoning_content
        if self.tool_calls:
            result['tool_calls'] = self.tool_calls
        if self.tool_call_id:
            result['tool_call_id'] = self.tool_call_id
        return result


# === Message Conversion ===

def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from object or dict"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)



_REPETITION_NUDGE_PREFIX = "You started repeating the same output"


def _stale_recovery_indices(messages) -> set:
    """Indices of stale Layer-2 repetition-recovery nudges (isMeta user
    messages) and the empty assistant stub each one follows. A nudge rescues
    ONE turn; left in history it is re-read every later turn as a standing
    reprimand (GLM-5.3 grew defensive: "변심 없음, 확신" on every answer). The
    nudge that is the LAST message is the live recovery attempt and stays."""
    stale = set()
    last = len(messages) - 1
    for i, m in enumerate(messages):
        if _get_attr(m, 'type') != 'user' or not _get_attr(m, 'isMeta', False) or i == last:
            continue
        content = _get_attr(_get_attr(m, 'message', {}), 'content', '')
        texts = [content] if isinstance(content, str) else [
            _get_attr(b, 'text', '') for b in (content or []) if _get_attr(b, 'type') == 'text']
        if not any(isinstance(t, str) and t.startswith(_REPETITION_NUDGE_PREFIX) for t in texts):
            continue
        stale.add(i)
        if i > 0 and _get_attr(messages[i - 1], 'type') == 'assistant':
            pc = _get_attr(_get_attr(messages[i - 1], 'message', {}), 'content', [])
            blocks = pc if isinstance(pc, list) else [{'type': 'text', 'text': pc}]
            if all(_get_attr(b, 'type') == 'text' and not str(_get_attr(b, 'text', '')).strip() for b in blocks):
                stale.add(i - 1)
    return stale


# Upper bound of what one image costs in the prompt (server VISION_MAX_TOKENS).
VISION_TOKENS_PER_IMAGE_MAX = 4096


def _image_block_to_data_url(block: Any) -> Optional[str]:
    """Anthropic image block -> OpenAI `image_url` data URL.

    Anthropic nests the bytes under `source`: {'type': 'base64', 'media_type':
    'image/png', 'data': '<b64>'}. OpenAI wants one `data:` URL. A url-sourced
    block is passed through as-is. Returns None when the block carries neither,
    so the caller can fall back to the placeholder instead of sending a part
    the server will reject."""
    source = _get_attr(block, 'source')
    if source is None:
        return None
    stype = _get_attr(source, 'type')
    if stype == 'url':
        url = _get_attr(source, 'url')
        return url or None
    data = _get_attr(source, 'data')
    if not data:
        return None
    media_type = _get_attr(source, 'media_type') or 'image/png'
    return f'data:{media_type};base64,{data}'


def _chat_message_text(msg: 'ChatMessage') -> str:
    """Plain text of a ChatMessage, whether it is a bare string or typed parts."""
    if isinstance(msg.content, str) and msg.content:
        return msg.content
    if msg.content_parts:
        return '\n'.join(p.get('text', '') for p in msg.content_parts
                          if p.get('type') == 'text').strip()
    return ''


def images_to_evict(total: int, limit: int, seen: int) -> int:
    """How many of the oldest images to send as text instead of pixels.

    The server takes at most `limit` images per request (vLLM
    --limit-mm-per-prompt). Dropping exactly the overflow would change the
    request prefix at a new place every time an image is added, and the server
    re-prefills from that point each turn. Dropping in batches of limit/2 keeps
    the evicted set fixed until the next batch is due, so the prefix moves once
    per limit/2 new images.

    `seen` is how many of the oldest images were already in a request the model
    answered. The batch rounding only takes from those: an image that arrived
    since the last answer has not been looked at yet, and dropping it early
    means the model describes it without having seen it. (It is not in any
    cached prefix yet either, so batching it buys nothing.) Unseen images go
    only when the overflow itself demands it — so a burst of new images past the
    limit is dropped exactly, and once the model has answered, the next request
    rounds the evicted set up to the batch: the prefix moves twice for that
    burst, then again once per limit/2 images."""
    if limit <= 0 or total <= limit:
        return 0
    batch = max(1, limit // 2)
    batched = min(total, -(-(total - limit) // batch) * batch)
    return max(total - limit, min(batched, seen))


def attached_image_count(messages: List[Any], limit: int) -> int:
    """Images the next request would carry. Counted on the converted request
    itself, so whatever convert_to_chat_messages skips (a duplicate tool
    result, a dropped recovery nudge, an image it cannot send) or evicts is not
    counted either. For the display only."""
    return sum(1 for m in convert_to_chat_messages(messages, [], max_images=limit)
               for p in (m.content_parts or []) if p.get('type') == 'image_url')


def _is_model_answer(msg: Any) -> bool:
    """An assistant message the model wrote, as opposed to one codemason made
    (an API error, an interrupt notice — model '<synthetic>'). Only the former
    means the images before it were looked at."""
    if _get_attr(msg, 'is_api_error_message', False):
        return False
    return _get_attr(_get_attr(msg, 'message', {}), 'model', None) != '<synthetic>'


def _evicted_image_label(n: int, block: Any, limit: int, shown: bool) -> str:
    """Text that stands in for an image not attached. It says whether the model
    was ever shown it — without that, a model that had described an image
    concluded from "no longer attached" that it never saw it, and re-read it —
    and names the copy in the session's image store (utils/image_store.py) so
    it can be Read. Deterministic, so the request prefix stays the same."""
    where = ''
    source = _get_attr(block, 'source')
    if _get_attr(source, 'type') == 'url':
        where = f'; source {_get_attr(source, "url")}'
    else:
        try:
            from ..utils.image_store import stored_path
            path = stored_path(source)
            if path is not None:
                again = 'look again' if shown else 'see it'
                where = f'; saved at {path} (Read it to {again})'
        except Exception:
            pass
    if shown:
        return (f'[Image {n}: no longer attached — you saw it earlier; dropped to stay '
                f'within the server limit of {limit} images per request{where}]')
    return (f'[Image {n}: not attached — over the server limit of {limit} images per '
            f'request, so you have not seen it{where}]')


def convert_to_chat_messages(
    messages: List[Union[UserMessage, AssistantMessage, Dict[str, Any]]],
    system_prompt: List[str],
    reasoning_roundtrip: bool = False,
    max_images: int = 0
) -> List[ChatMessage]:
    """Convert Claude-style messages to OpenAI chat format.

    reasoning_roundtrip: when True, ThinkingBlocks stored on assistant turns are
    sent back as `reasoning_content`. Off for llama.cpp, which has no such
    channel and would only see an unknown field.

    max_images: images the server takes per request (services/vision.py);
    0 = no image channel. When > 0, image blocks are sent as OpenAI `image_url`
    parts instead of being replaced with a placeholder sentence. 0 by default
    and for llama.cpp, which has no image channel — an image_url part there is
    an unknown field at best and a hard 400 at worst.

    Every attached image gets a `[Image N]` label, numbered in conversation
    order, so the model can tell images apart and refer to them (unlabelled, it
    miscounts a run of 8). Exceeding the server cap is a 400, so when the
    conversation holds more, the oldest are sent as text (images_to_evict).
    Only this request copy changes; the stored history keeps every image."""

    chat_messages: List[ChatMessage] = []
    vision = max_images > 0
    # (number, label part, image part, source block) per attached image.
    attached: List[tuple] = []
    seen = 0  # images before the last assistant message (see images_to_evict)
    # Numbers of the images that were attached to a request the model answered.
    # Eviction only grows, so this is fixed once an image's first answered
    # request is past, and its label never flips.
    shown: set = set()

    def _attach(block: Any) -> Optional[List[Dict[str, Any]]]:
        url = _image_block_to_data_url(block) if vision else None
        if not url:
            return None
        n = len(attached) + 1
        label = {'type': 'text', 'text': f'[Image {n}]'}
        part = {'type': 'image_url', 'image_url': {'url': url}}
        attached.append((n, label, part, block))
        return [label, part]

    # Add system prompt with cwd emphasis
    if system_prompt:
        full_system_prompt = build_system_prompt_with_cwd_emphasis(system_prompt)
        chat_messages.append(ChatMessage(role='system', content=full_system_prompt))

    debug_log('CONVERTING MESSAGES', {
        'total_messages': len(messages),
        'message_types': [_get_attr(m, 'type', 'unknown') for m in messages],
    })

    # Track tool calls for matching results
    processed_tool_call_ids: set = set()
    processed_assistant_uuids: set = set()
    added_tool_use_ids: set = set()

    _stale = _stale_recovery_indices(messages)
    if _stale:
        debug_log('STALE_RECOVERY_NUDGES_DROPPED', {'count': len(_stale)})
    for _msg_idx, msg in enumerate(messages):
        if _msg_idx in _stale:
            continue
        msg_type = _get_attr(msg, 'type')
        msg_message = _get_attr(msg, 'message', {})
        msg_content = _get_attr(msg_message, 'content', [])

        if msg_type == 'user':
            if isinstance(msg_content, list):
                # Check for tool_result blocks
                tool_result_blocks = [c for c in msg_content if _get_attr(c, 'type') == 'tool_result']
                text_blocks = [c for c in msg_content if _get_attr(c, 'type') == 'text']

                if tool_result_blocks:
                    # Convert tool_result to role: 'tool'
                    for block in tool_result_blocks:
                        # Images this tool returned (Read on a PNG, an MCP
                        # screenshot) go inside its own tool message, after the
                        # text. Not in a user message after the results: the
                        # chat template treats the last user message as the start
                        # of the turn and blanks the reasoning of every assistant
                        # message before it (clear_thinking), so each image read
                        # erased what the model had worked out in the tool loop.
                        tool_images: List[Dict[str, Any]] = []
                        tool_use_id = _get_attr(block, 'tool_use_id', '')
                        if tool_use_id in processed_tool_call_ids:
                            debug_log('SKIPPING DUPLICATE TOOL RESULT', {'tool_call_id': tool_use_id})
                            continue
                        processed_tool_call_ids.add(tool_use_id)

                        result_content = ''
                        block_content = _get_attr(block, 'content', '')
                        if isinstance(block_content, str):
                            result_content = block_content
                        elif isinstance(block_content, list):
                            pieces = []
                            for c in block_content:
                                if _get_attr(c, 'type') == 'image':
                                    # Never str() an image block: that pastes its whole
                                    # base64 payload into the tool text.
                                    img = _attach(c)
                                    if img:
                                        tool_images += img
                                    else:
                                        pieces.append('[Image content not supported]')
                                else:
                                    pieces.append(_get_attr(c, 'text', str(c)))
                            result_content = '\n'.join(pieces)
                        else:
                            result_content = json.dumps(block_content) if block_content else ''

                        # Use block.content (already extracted above), not msg.tool_use_result,
                        # to avoid all blocks getting the same content (bug #3 fix).

                        # Truncate if needed
                        truncated_content = truncate_tool_result(result_content)

                        if tool_images:
                            parts: List[Dict[str, Any]] = []
                            if truncated_content.strip():
                                parts.append({'type': 'text', 'text': truncated_content})
                            chat_messages.append(ChatMessage(
                                role='tool',
                                tool_call_id=tool_use_id,
                                content_parts=parts + tool_images
                            ))
                            continue

                        # Add success message for empty results to prevent LLM stalling
                        if not truncated_content or not truncated_content.strip():
                            truncated_content = '(success, no output)'

                        chat_messages.append(ChatMessage(
                            role='tool',
                            tool_call_id=tool_use_id,
                            content=truncated_content
                        ))

                    # Add text blocks as user message if present
                    text_content = ''
                    if text_blocks:
                        text_content = '\n'.join(_get_attr(c, 'text', '') for c in text_blocks).strip()
                    # An image the user added alongside the results (typed in
                    # mid-turn) rides in that user message; it used to be
                    # dropped without a word.
                    extra_parts: List[Dict[str, Any]] = []
                    for c in msg_content:
                        if _get_attr(c, 'type') == 'image':
                            img = _attach(c)
                            if img:
                                extra_parts += img
                            else:
                                text_content = (text_content + '\n[Image content not supported]').strip()
                    if extra_parts:
                        chat_messages.append(ChatMessage(role='user', content_parts=(
                            [{'type': 'text', 'text': text_content}] if text_content else []) + extra_parts))
                    elif text_content:
                        chat_messages.append(ChatMessage(role='user', content=text_content))
                else:
                    # Regular user message with content blocks
                    text_parts = []
                    mm_parts: List[Dict[str, Any]] = []
                    for c in msg_content:
                        c_type = _get_attr(c, 'type')
                        if c_type == 'text':
                            text = _get_attr(c, 'text', '')
                            text_parts.append(text)
                            if text:
                                mm_parts.append({'type': 'text', 'text': text})
                        elif c_type == 'image':
                            img = _attach(c)
                            if img:
                                mm_parts += img
                            else:
                                # No image channel (or an unreadable source): say so
                                # rather than dropping the block silently, so the
                                # model knows something was there.
                                text_parts.append('[Image content not supported]')
                                mm_parts.append({'type': 'text',
                                                 'text': '[Image content not supported]'})
                    has_image = any(p['type'] == 'image_url' for p in mm_parts)
                    if has_image:
                        chat_messages.append(ChatMessage(role='user', content_parts=mm_parts))
                    else:
                        # Text-only: keep the bare-string shape byte-identical to
                        # the pre-vision path.
                        text_content = '\n'.join(text_parts).strip()
                        if text_content:
                            chat_messages.append(ChatMessage(role='user', content=text_content))
            else:
                # String content
                if msg_content and isinstance(msg_content, str) and msg_content.strip():
                    chat_messages.append(ChatMessage(role='user', content=msg_content.strip()))

        elif msg_type == 'assistant':
            # This message answered a request holding `attached` minus the
            # images that request evicted.
            if _is_model_answer(msg):
                for k in range(images_to_evict(len(attached), max_images, seen) + 1,
                               len(attached) + 1):
                    shown.add(k)
                seen = len(attached)
            # Check for duplicate by UUID
            msg_uuid = str(_get_attr(msg, 'uuid', ''))
            if msg_uuid and msg_uuid in processed_assistant_uuids:
                debug_log('SKIPPING DUPLICATE ASSISTANT BY UUID', {'uuid': msg_uuid})
                continue
            if msg_uuid:
                processed_assistant_uuids.add(msg_uuid)

            content_text = ''
            reasoning_text = ''
            tool_calls = []
            current_tool_use_ids = []

            if isinstance(msg_content, str):
                content_text = msg_content
            elif isinstance(msg_content, list):
                for block in msg_content:
                    block_type = _get_attr(block, 'type')
                    if block_type == 'thinking':
                        reasoning_text += _get_attr(block, 'thinking', '')
                    elif block_type == 'text':
                        content_text += _get_attr(block, 'text', '') + '\n'
                    elif block_type == 'tool_use':
                        block_id = _get_attr(block, 'id', '')
                        if block_id in added_tool_use_ids:
                            debug_log('SKIPPING DUPLICATE TOOL_USE', {'id': block_id})
                            continue
                        current_tool_use_ids.append(block_id)

                        tool_calls.append({
                            'id': block_id,
                            'type': 'function',
                            'function': {
                                'name': _get_attr(block, 'name', ''),
                                'arguments': json.dumps(_get_attr(block, 'input', {}))
                            }
                        })

            # Mark tool_use IDs as added
            for tid in current_tool_use_ids:
                added_tool_use_ids.add(tid)

            # Skip empty assistant messages
            content_text = content_text.strip()
            if not content_text and not tool_calls:
                debug_log('SKIPPING EMPTY ASSISTANT MESSAGE')
                continue

            assistant_msg = ChatMessage(
                role='assistant',
                content=content_text or None,
                tool_calls=tool_calls if tool_calls else None,
                reasoning_content=(reasoning_text.strip() or None) if reasoning_roundtrip else None
            )
            chat_messages.append(assistant_msg)

            debug_log('ADDED ASSISTANT MESSAGE', {
                'uuid': msg_uuid,
                'content_len': len(content_text),
                'reasoning_len': len(reasoning_text) if reasoning_roundtrip else 0,
                'tool_calls_count': len(tool_calls)
            })

    evict = images_to_evict(len(attached), max_images, seen)
    if evict:
        gone = {id(part): (n, label, block) for n, label, part, block in attached[:evict]}
        for m in chat_messages:
            if not m.content_parts:
                continue
            kept = []
            for p in m.content_parts:
                hit = gone.get(id(p))
                if hit is None:
                    kept.append(p)
                else:
                    n, label, block = hit
                    label['text'] = _evicted_image_label(n, block, max_images, n in shown)
            m.content_parts = kept
        debug_log('IMAGES_EVICTED', {'total': len(attached), 'evicted': evict})

    # Clean up empty messages and merge consecutive assistants
    cleaned_messages = _clean_messages(chat_messages)

    # Context management
    managed_messages = _manage_context(cleaned_messages)

    # Final pass: merge consecutive assistant messages
    final_messages = _merge_consecutive_assistants(managed_messages)

    debug_log('FINAL MESSAGE SEQUENCE', {
        'count': len(final_messages),
        'roles': [{'role': m.role, 'has_tool_calls': bool(m.tool_calls)} for m in final_messages]
    })

    return final_messages


def _clean_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Clean empty messages and merge consecutive assistant messages"""
    cleaned = []

    for msg in messages:
        # Skip empty messages (but keep assistant messages with tool_calls)
        has_tool_calls = msg.role == 'assistant' and msg.tool_calls
        # A vision message keeps everything in content_parts (content is None);
        # it is not empty.
        if (not msg.content or not msg.content.strip()) and not has_tool_calls \
                and not msg.content_parts:
            continue

        # Merge consecutive assistant messages
        if cleaned and msg.role == 'assistant' and cleaned[-1].role == 'assistant':
            prev_msg = cleaned[-1]
            prev_has_tool_calls = bool(prev_msg.tool_calls)
            curr_has_tool_calls = bool(msg.tool_calls)

            if not prev_has_tool_calls and not curr_has_tool_calls:
                # Merge content
                prev_content = prev_msg.content or ''
                curr_content = msg.content or ''
                prev_msg.content = f"{prev_content}\n\n{curr_content}".strip()
                if msg.reasoning_content:
                    prev_reasoning = prev_msg.reasoning_content or ''
                    prev_msg.reasoning_content = (
                        f"{prev_reasoning}\n\n{msg.reasoning_content}".strip())
                continue

        cleaned.append(msg)

    return cleaned


def _manage_context(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Manage context by keeping recent messages within token limit.

    NOTE: this pre-send trim is local
    stopgap for servers that hard-error on overflow. It silently drops the
    oldest messages (by heuristic count, which mis-sizes the window) and, by
    keeping every request under the limit, also renders compaction's PTL retry
    unreachable. When the exact model tokenizer is available, auto-compaction
    counts accurately and fires in time, so we defer to it exactly like upstream
    and skip this trim entirely (this also lets the PTL retry actually run).
    """
    if has_model_tokenizer():
        return messages

    def count_message_tokens(msg: ChatMessage) -> int:
        tokens = count_tokens_precise(msg.content or _chat_message_text(msg))
        if msg.content_parts:
            # Each image is at most VISION_TOKENS_PER_IMAGE_MAX on the server.
            tokens += VISION_TOKENS_PER_IMAGE_MAX * sum(
                1 for p in msg.content_parts if p.get('type') == 'image_url')
        if msg.reasoning_content:
            tokens += count_tokens_precise(msg.reasoning_content)
        if msg.tool_calls:
            tokens += count_tokens_precise(json.dumps(msg.tool_calls))
        tokens += 4  # Overhead for message structure
        return tokens

    system_messages = [m for m in messages if m.role == 'system']
    conversation_messages = [m for m in messages if m.role != 'system']

    total_tokens = sum(count_message_tokens(m) for m in system_messages)
    managed_messages = list(system_messages)

    # Add recent messages from the end
    for i in range(len(conversation_messages) - 1, -1, -1):
        msg = conversation_messages[i]
        msg_tokens = count_message_tokens(msg)

        if total_tokens + msg_tokens > get_max_tokens():
            debug_log('CONTEXT LIMIT REACHED', {
                'dropped_messages': i + 1,
                'kept_messages': len(conversation_messages) - i - 1,
                'total_tokens': total_tokens
            })
            break

        total_tokens += msg_tokens
        managed_messages.insert(len(system_messages), msg)

    # Truncating from the front can orphan tool-result messages whose assistant
    # tool_calls were dropped (a leading role='tool' with no preceding
    # tool_calls). Many servers 400 on that, so strip any orphan tool results at
    # the head of the conversation. (Assistant tool_calls can't be orphaned the
    # other way: their results are newer, so they're always kept first.)
    head = len(system_messages)
    while head < len(managed_messages) and managed_messages[head].role == 'tool':
        managed_messages.pop(head)

    return managed_messages


def _merge_consecutive_assistants(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Final pass to merge any consecutive assistant messages"""
    safe_messages = []

    for msg in messages:
        if safe_messages and msg.role == 'assistant' and safe_messages[-1].role == 'assistant':
            prev_msg = safe_messages[-1]
            prev_has_tool_calls = bool(prev_msg.tool_calls)
            curr_has_tool_calls = bool(msg.tool_calls)

            # Merge content
            if msg.content:
                prev_content = prev_msg.content or ''
                prev_msg.content = f"{prev_content}\n\n{msg.content}".strip()

            # Merge reasoning alongside it — dropping it here would reintroduce
            # the very gap this field exists to close.
            if msg.reasoning_content:
                prev_reasoning = prev_msg.reasoning_content or ''
                prev_msg.reasoning_content = (
                    f"{prev_reasoning}\n\n{msg.reasoning_content}".strip())

            # Merge tool_calls
            if curr_has_tool_calls and not prev_has_tool_calls:
                prev_msg.tool_calls = msg.tool_calls
            elif curr_has_tool_calls and prev_has_tool_calls:
                existing_ids = {tc['id'] for tc in prev_msg.tool_calls}
                for tc in msg.tool_calls:
                    if tc['id'] not in existing_ids:
                        prev_msg.tool_calls.append(tc)

            debug_log('MERGED CONSECUTIVE ASSISTANTS', {
                'prev_has_tool_calls': prev_has_tool_calls,
                'curr_has_tool_calls': curr_has_tool_calls
            })
        else:
            safe_messages.append(msg)

    return safe_messages


# === Main Query Function ===

# Built tool blocks, keyed by the tool-name set they were built from.
# See the comment at the build site for why this must not be recomputed.
_TOOL_BLOCK_CACHE: Dict[str, List[Dict[str, Any]]] = {}


async def query_openai_compatible(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_prompt: List[str],
    tools: List[Any],
    signal: Optional[asyncio.Event] = None,
    options: Optional[Dict[str, Any]] = None,
    on_token_update: Optional[Callable[[int], None]] = None,
    on_content_delta: Optional[Callable[[str], None]] = None,
    on_thinking_update: Optional[Callable[[int], None]] = None,
    *,
    continuous_usage_stats: bool = False,
    max_images: int = 0,
    repetition_detection: bool = False,
    reasoning_roundtrip: bool = False,
) -> AssistantMessage:
    """
    Query an OpenAI-compatible chat-completions server (llama.cpp or vLLM).

    continuous_usage_stats: when True (vLLM), ask the server for a running
    completion_tokens on every chunk via stream_options; when False (llama.cpp),
    usage arrives only in the final chunk and the live count uses a per-chunk
    tally until then.

    reasoning_roundtrip: when True (vLLM serving DeepSeek-V4), this turn's
    reasoning is stored on the assistant message and sent back as
    `reasoning_content` on every later request. DeepSeek's thinking API requires
    it for turns that made a tool call; the public API returns 400 without it,
    while a self-hosted server simply drops the reasoning and the model silently
    loses what it worked out — it then repeats the same tool call to rediscover
    it. Off for llama.cpp, which has no reasoning channel.

    Args:
        messages: List of user and assistant messages
        system_prompt: System prompt parts
        tools: Available tools
        signal: Abort signal (asyncio.Event)
        options: Additional options
        on_token_update: Callback for token count updates
        on_thinking_update: Callback for reasoning-token count updates; fired
            only on turns where the server actually emitted reasoning

    Returns:
        AssistantMessage with the response
    """
    import time
    start_time = time.time()
    options = options or {}
    cfg = _get_config()

    try:
        chat_messages = convert_to_chat_messages(
            messages, system_prompt, reasoning_roundtrip=reasoning_roundtrip, max_images=max_images)

        # Derive the language to instruct the model in FROM THE CONVERSATION (the
        # last thing the user actually typed), so it works on a machine whose
        # config.json never set `language` — which is every machine but the one
        # this was developed on. config.language still wins when set.
        from ..utils.language import detect_language, set_request_language
        # A vision message carries its text in content_parts, not content, so
        # reading only `content` would skip the newest turn and detect the
        # language from an older one.
        _last_user = next((t for t in (_chat_message_text(m)
                                       for m in reversed(chat_messages)
                                       if m.role == 'user') if t), '')
        set_request_language(detect_language(_last_user))

        # Convert tools to OpenAI format. Prefer the faithfully-ported per-tool
        # description (tool_prompts); fall back to the short blurb.
        from ..constants.tool_prompts import TOOL_DESCRIPTIONS
        # Built once per tool-set and reused. The tool block is the FIRST thing
        # in the request, ahead of the system prompt and the whole message
        # history, so it is the base of the server's prefix cache: anything that
        # changes it throws away every cached token behind it.
        #
        # It used to be rebuilt on every request, and both build_description()
        # and input_json_schema() are resolved fresh each time — so any tool
        # reading live state silently invalidated the conversation. An audit on
        # 2026-09-06 found three doing exactly that: the Agent tool carried the
        # detected conversation language (detected, inside a tool loop, from tool
        # OUTPUT), while the Agent and Skill tools enumerated the custom agents
        # and skills under the CURRENT directory. Each was worth a full
        # re-prefill — measured at ~123 seconds on a 121k-token session.
        #
        # Freezing per tool-set makes the block structurally unable to drift
        # mid-session, which is a stronger guarantee than auditing every tool
        # forever. Sub-agents legitimately run different tool sets, hence the key.
        # The cost: a skill or agent added mid-session is not listed until
        # restart — the alternative charges a re-prefill for a listing that
        # almost never changes while a session runs.
        _tool_key = oob_scope_key(t.name for t in tools)
        tool_schemas = _TOOL_BLOCK_CACHE.get(_tool_key)
        if tool_schemas is not None:
            return_early_schemas = tool_schemas
        else:
            return_early_schemas = None
        tool_schemas = [] if return_early_schemas is None else return_early_schemas
        for tool in (tools if return_early_schemas is None else []):
            # Prefer a dynamically-built description (e.g. Skill lists available
            # skills), then the tool's static DESCRIPTION, then the map/blurb.
            description = None
            if hasattr(tool, 'build_description'):
                try:
                    description = tool.build_description()
                except Exception:
                    description = None
            description = description or getattr(tool, 'DESCRIPTION', None) or TOOL_DESCRIPTIONS.get(tool.name) or get_simple_description(tool.name)
            schema = {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': description,
                    'parameters': tool.input_json_schema()
                }
            }
            tool_schemas.append(schema)
        if return_early_schemas is None and tool_schemas:
            _TOOL_BLOCK_CACHE[_tool_key] = tool_schemas

        # Register what the estimate cannot see, now that both parts are built.
        # Cheap: the tool block is cached per tool-set and the system prompt is
        # fixed for the session, so this recomputes the same number each turn
        # and every consumer (compaction included) reads one consistent value.
        try:
            _oob = 0
            if tool_schemas:
                # json.dumps with its defaults: the same bytes the Rust port's
                # py_json_dumps produces, so both trees register one number.
                _oob += count_tokens_precise(json.dumps(tool_schemas))
            if system_prompt:
                _oob += count_tokens_precise('\n\n'.join(system_prompt))
            set_out_of_band_tokens(_tool_key, _oob)
        except Exception as _e:  # noqa: BLE001 — an estimate must not block a request
            # Logged, never swallowed. A bare `pass` here hid a NameError for
            # 24 commits (2026-09-06..12): the count was never registered and
            # every unanchored estimate ran ~30k tokens short of the request.
            log_error('out-of-band token count failed: %s' % _e,
                      '_openai_common: request assembly')

        # Per-call output cap override (e.g. compaction reserves a fixed summary
        # budget); falls back to the configured default.
        max_output_tokens = options.get('max_tokens') or cfg.max_tokens

        # Never let (prompt + requested output) exceed the model's context
        # window. The server rejects that with 400 "total ... tokens > maximum
        # context length" — e.g. 458753 input + 65536 output = 524289 > 524288.
        # It tokenizes with the model's OWN tokenizer, which routinely counts a
        # few more tokens than our estimate (that is the off-by-one above), so
        # cap the output against the estimated prompt size with a safety margin,
        # keeping a floor so a turn always gets some room to answer.
        try:
            _ctx = get_context_size()
            _input_est = estimate_input_tokens(messages, _tool_key)

            # What the estimate does NOT already contain.
            #
            # An anchored estimate starts from the server's own count of the
            # previous request, so the system prompt and the tool block are
            # inside it. The unanchored fallback is a per-block estimate of the
            # message text alone, and both are missing — here that is ~25k for
            # the tools plus the system prompt, and a fixed 512-token margin
            # cannot absorb a hole that size. Measured elsewhere on this
            # codebase: a one-line request reported an estimate of 9 tokens
            # against a real 458,753-token input and the turn was rejected at
            # 524,289 > 524,288, one token over.
            #
            # Add them ONLY when there is no anchor. Adding them always would
            # double-count on every turn of an ongoing conversation: not an
            # overflow, but a silent ~25k tax that shortens answers and drags
            # auto-compaction forward. (An earlier fix for the same crash added
            # the tool tokens unconditionally, which is why the branch is
            # spelled out here rather than left implicit.)
            #
            # The output cap is the LAST line of defence, not the first: once
            # the input alone exceeds the window, capping output to its floor
            # still overflows. should_auto_compact reads the same
            # estimate_input_tokens() for that reason.
            _anchored = has_usage_anchor(messages)

            # Scale the margin with the input. The model's OWN tokenizer counts
            # a little more than any estimate of ours, and that drift grows with
            # the prompt (chat-template wrappers, per-message role tags), so a
            # constant is too small exactly where it matters. ~1.5%, floored at
            # the old constant so short prompts behave as before.
            _margin = max(_CONTEXT_SAFETY_MARGIN, _input_est // 64)

            _room = _ctx - _input_est - _margin
            if _room < max_output_tokens:
                max_output_tokens = max(_room, _MIN_OUTPUT_TOKENS)
                debug_log('OUTPUT_CAPPED_TO_FIT_CONTEXT', {
                    'context': _ctx, 'input_est': _input_est,
                    'anchored': _anchored, 'margin': _margin,
                    'capped_max_tokens': max_output_tokens,
                })
        except Exception:  # noqa: BLE001 — never block a request on the estimate
            pass

        # Ask the server to emit a final usage chunk with REAL token counts
        # (computed by the model's own tokenizer). Without this, streamed
        # responses carry no usage and token displays fall back to crude
        # estimates (the CJK-aware char heuristic, not the exact tokenizer).
        # include_usage -> final exact counts. continuous_usage_stats (vLLM only)
        # -> a running completion_tokens on EVERY chunk, so the live spinner can
        # show the server's exact count instead of a per-chunk tally that
        # under-counts multi-token (speculative/MTP) chunks. llama.cpp ignores
        # the flag and only sends usage in the final chunk, so it stays off there.
        _stream_options = {'include_usage': True}
        if continuous_usage_stats:
            _stream_options['continuous_usage_stats'] = True

        # Fresh numbers for this request, then ask the server what it is about
        # to re-read (services.prefill_probe). Our own estimate below fills the
        # gap until the server answers, and llama.cpp has no counters so it is
        # all there ever is there.
        _tokens_mod.reset_prefill_state()
        from .prefill_probe import probe_next_request as _probe_prefill
        _probe_prefill()

        # Sampling comes from Config.effective_sampling, not the raw fields: the
        # served family decides. GLM gets the vendor's own numbers (temperature
        # 1.0 / top_p 0.95, no top_k); everything else keeps the stored values
        # it has always used. A None there means the key is not sent at all, so
        # the server's default applies instead of one we invented.
        _sampling = cfg.effective_sampling()
        request_body = {
            'model': cfg.model_name,
            'messages': [m.to_dict() for m in chat_messages],
            'temperature': _sampling['temperature'],
            'max_tokens': max_output_tokens,
            'stream': True,
            'stream_options': _stream_options,
            'top_p': _sampling['top_p'],
            # llama.cpp prompt-cache hint. vLLM ignores it (prefix caching is
            # automatic/server-side), so it's harmless there and still helps if
            # the backend is llama.cpp.
            'cache_prompt': True,
        }
        # Optional knobs. The anti-runaway penalties were the DeepSeek-V4
        # 8K-repeat guard: not in the server-side DSV4_SERVER_LOCK whitelist, so
        # they pass through even while temperature is locked server-side, and
        # mild by design so code generation isn't distorted.
        for _k in ('top_k', 'min_p', 'repetition_penalty',
                   'presence_penalty', 'frequency_penalty'):
            _v = _sampling.get(_k)
            if _v is not None:
                request_body[_k] = _v

        if tool_schemas:
            request_body['tools'] = tool_schemas

        # Thinking opt-in: only carry reasoning_effort when explicitly configured
        # (cfg.reasoning_effort in {'low','medium','high','max'}). Empty => omit the key
        # entirely, which does NOT mean "no thinking" — it hands the decision to
        # the server's own default. That default is a launch flag we do not
        # control from here (--default-chat-template-kwargs.thinking), so an empty
        # value can resolve either way: the DSV4-Flash box has been serving with
        # thinking=true/effort=low. Set this only to OVERRIDE the server, never to
        # guarantee thinking is off. OWUI drives the same knob via its account
        # param. Verified 2026-08-01 that a thinking-ON server is handled
        # correctly: reasoning arrives on the separate `reasoning_content`
        # channel, is kept out of `full_content`, and is not echoed back in
        # history — see got_reasoning below.
        # options['no_reasoning'] asks for the least thinking the server will
        # do, for calls that are bookkeeping rather than work — compaction above
        # all, where the reasoning pass runs before the first summary token and
        # doubles the wait, and repetition retries, where the reasoning channel
        # is the thing that collapsed.
        #
        # It sends 'low' rather than omitting the key. Omitting does NOT mean
        # "no thinking": the key is simply absent and the server applies its own
        # launch default, which here is
        # default_chat_template_kwargs={'enable_thinking': True,
        # 'reasoning_effort': 'high'} — so omission bought nothing.
        #
        # Measured 2026-09-13 against this deployment, one geometry question,
        # completion tokens: reasoning_effort=high 737, key omitted 636,
        # chat_template_kwargs={'enable_thinking': False} 728,
        # reasoning_effort=low 417. Omission and enable_thinking=false both land
        # next to high; only 'low' actually moves it. (An earlier note here said
        # low "injects an empty string" on this model family and so meant "no
        # instruction" — that is no longer true of the served model, and it is
        # why this call omitted the key for a month without effect.)
        # Was guarded with hasattr, which meant that while Config lacked the
        # method this quietly fell back to the global value and ignored the
        # per-model policy -- no error, just the wrong effort. The method
        # exists now (parity with Rust), so call it plainly: a missing one
        # should fail loudly, not resolve to something that merely looks right.
        _eff = cfg.effective_reasoning_effort()
        if (options or {}).get('no_reasoning'):
            request_body['reasoning_effort'] = 'low'
        elif _eff:
            request_body['reasoning_effort'] = _eff
        # clear_thinking: drop prior-turn reasoning at the template (see Config).
        #
        # Held OFF for recovery requests, because the template decides "prior
        # turn" by user message and every recovery path injects one. The nudge
        # that asks the model to carry on is itself what deletes the thinking it
        # needs to carry on FROM.
        #
        # Traced 2026-09-13 from a live session. A turn spent its whole
        # 65,536-token budget on the reasoning channel and produced no content,
        # so the server cut it at max_tokens and the max-output recovery fired.
        # The next request's prompt should have grown by that reasoning; instead
        # it FELL from 95,825 to 79,420 — the cut turn's 65,537 tokens gone, and
        # with them the 16,405 tokens of in-turn reasoning the tool loop had
        # accumulated, which is the drop exactly. The prompt the model then
        # received told it to "resume mid-thought, no recap" with the thought
        # deleted, and it redid the same work for hours.
        #
        # The trade is prompt size against state: this one request carries the
        # thinking it would otherwise lose. The cost is bounded — the reasoning
        # dies at the next ordinary user turn, as before.
        # The kwarg is named per family and both are sent, because a server that
        # does not know a name ignores it:
        #   GLM-5.3     `clear_thinking` — read by its chat_template.jinja.
        #   DeepSeek-V4 `drop_thinking`  — read by the model's own encoder
        #     (encoding_dsv4.apply_chat_template -> kwargs.get('drop_thinking',
        #     True)), verified in the deployed encoding patch. The encoder then
        #     forces it OFF whenever any message carries tools, which is every
        #     request this client makes, so on DSV4 prior reasoning survives
        #     regardless and the key is inert. Sending only clear_thinking meant
        #     saying nothing at all to that family.
        # Semantics match: true = keep only this turn's reasoning.
        #
        # The gate reads `keep_thinking`, not `no_reasoning`, though the recovery
        # paths set both. They are different intents: no_reasoning is about the
        # reasoning this request may produce, keep_thinking about the reasoning
        # earlier requests already produced. Compaction wants the first and not
        # the second — it sets no_reasoning to save the wait, and carrying every
        # earlier turn's reasoning into the one request whose purpose is to
        # shrink the history is the opposite of what it is for.
        if getattr(cfg, 'clear_thinking', True):
            _clear = not (options or {}).get('keep_thinking')
            request_body['chat_template_kwargs'] = {
                'clear_thinking': _clear,
                'drop_thinking': _clear,
            }

        # Server-side repetition detection (vLLM RepetitionDetectionParams).
        # Cuts a generation that has started repeating an n-gram, and reports
        # finish_reason='repetition'. Sent only when the backend advertises it,
        # because llama.cpp has no such field.
        #
        # This is the answer to the loop shape the repeated-tool-call cap cannot
        # see: paragraphs alternating with no tool call between them, which
        # arrives as one long assistant message and is indistinguishable from a
        # long answer until the budget runs out. Catching it at the sampler
        # stops it mid-generation instead of after the tokens are gone.
        _rd = _repetition_detection_params(cfg) if repetition_detection else None
        if _rd:
            request_body['repetition_detection'] = _rd

        debug_log('REQUEST BODY', request_body)

        # Make streaming request
        full_content = ''
        got_reasoning = False  # True once the server emits any reasoning_content
        # this turn — i.e. thinking arrived on its own channel. Any <think> left
        # in `content` is then a known Qwen3.6-MTP glitch (answer text wrongly
        # tag-wrapped; llama.cpp#22398), so clean_special_tokens keeps the inner
        # text instead of deleting it.
        tool_calls = []
        usage = None
        cached_prompt_tokens = 0  # prompt_tokens_details.cached_tokens, tracked
        # SEPARATELY from `usage` because that slot is overwritten wholesale by
        # every chunk that carries usage. With continuous_usage_stats the server
        # sends usage on EVERY chunk and only some of them carry the details
        # block, so a final chunk without it would erase a real cache hit. Keep
        # the largest value ever reported instead: cached_tokens describes the
        # prompt, which does not change mid-stream, so it never legitimately
        # shrinks.
        final_finish_reason = None  # last non-null finish_reason from the stream;
        # mapped to the Anthropic stop_reason below so query.py can detect a
        # max_output truncation ('length' -> 'max_tokens') and run recovery.
        # Bound here, not only inside the stream block: an early failure path
        # can skip that block entirely, and the assembly below reads it.
        aborted = False
        # Set by a stall guard inside the stream block, for the same reason.
        stall_reason = None
        streamed_tokens = 0  # live per-chunk tally over answer + reasoning
                             # deltas, for the spinner only; the final
                             # include_usage chunk (completion_tokens) supplies
                             # the exact end count used everywhere that matters.
        # Reasoning-token accounting. The server gives NO breakdown — vLLM's
        # usage carries only completion_tokens (verified against the DSV4-Flash
        # box: no completion_tokens_details.reasoning_tokens) — so counting the
        # reasoning text ourselves is the only way to know. Counting CHUNKS
        # instead would be badly wrong: with speculative decoding a delta
        # routinely carries several tokens (~12 chars/chunk measured), so a
        # chunk tally undercounts by multiples. Re-tokenising the whole
        # accumulated string is exact and cheap (~0.6ms per 1.7k chars), but
        # doing it per delta is O(n²), so it is throttled to _THINK_COUNT_EVERY
        # and always finalised once when the stream ends.
        reasoning_text = ''
        thinking_tokens = 0
        _last_think_count_at = 0.0
        _silent_logged = False

        def _recount_thinking(force: bool = False) -> None:
            """Refresh the exact reasoning-token count, throttled."""
            nonlocal thinking_tokens, _last_think_count_at
            if not reasoning_text:
                return
            now = time.monotonic()
            if not force and (now - _last_think_count_at) < _THINK_COUNT_EVERY:
                return
            _last_think_count_at = now
            try:
                from ..utils.tokens import count_text_tokens
                counted = count_text_tokens(reasoning_text)
            except Exception:
                return          # never let token accounting break a stream
            if counted == thinking_tokens:
                return
            thinking_tokens = counted
            if on_thinking_update:
                on_thinking_update(thinking_tokens)
            # A turn can fail by thinking without ever answering — see
            # _SILENT_REASONING_TOKENS. Record it once so an unattended run
            # leaves the record that explains the silence.
            nonlocal _silent_logged
            if (thinking_tokens >= _SILENT_REASONING_TOKENS
                    and not _silent_logged and not full_content.strip()):
                _silent_logged = True
                log_error(
                    'the model has produced %d reasoning tokens and no answer '
                    'yet. Not cut — a hard question can legitimately think this '
                    'long, and the reasoning is carried into the recovery '
                    'request if the server max_tokens ends the turn.'
                    % thinking_tokens,
                    'openai_common: silent reasoning')

        # Omit the Authorization header entirely when no key is set, so a
        # default (empty-key) install sends an anonymous request rather than a
        # malformed `Authorization:` empty header. Matches tokens.py behaviour.
        request_headers = {'Content-Type': 'application/json'}
        if cfg.model_api_key:
            request_headers['Authorization'] = f'Bearer {cfg.model_api_key}'

        # Optional end-to-end BODY encryption (utils/body_crypto). When a key is
        # configured, the request body is AES-256-GCM sealed and base_url should
        # point at the decrypt gateway; when empty this is a no-op and the
        # plaintext body is sent exactly as before (fallback).
        _aes_key = None
        if cfg.model_aes_key:
            from ..utils.body_crypto import (
                load_key as _bc_load,
                encrypt as _bc_enc,
                decrypt as _bc_dec,
                BodyCryptoError as _BCErr,
            )
            _aes_key = _bc_load(cfg.model_aes_key)
        if _aes_key is not None:
            _stream_kwargs = dict(
                content=_bc_enc(_aes_key, json.dumps(request_body)).encode('ascii'),
                headers={**request_headers,
                         'Content-Type': 'application/octet-stream',
                         'X-Body-Encrypted': 'aesgcm'},
            )
        else:
            _stream_kwargs = dict(json=request_body, headers=request_headers)

        # Large-context requests (e.g. ~1M-token auto-compaction summaries) need
        # far more than the old 600s: ~1M prefill alone is ~7 min and the summary
        # generation adds more — measured a real 1M compaction at ~410s, and 600s
        # cut it off mid-generation (empty response -> compaction failure). Default
        # 1800s covers 1M sessions out of the box; env-overridable for headroom via
        # CODEMASON_MODEL_HTTP_TIMEOUT (legacy LLAMACPP_HTTP_TIMEOUT still honoured).
        _http_timeout = float(os.getenv('CODEMASON_MODEL_HTTP_TIMEOUT')
                              or os.getenv('LLAMACPP_HTTP_TIMEOUT') or '1800')
        # Persistent keep-alive client + connect-phase retry (services.backend):
        # reuse one connection across rapid turns and absorb a momentary connect
        # blip instead of dying on the first ConnectError. Body below unchanged.
        #
        # `signal` goes in for the same reason it wraps the read loop below: the
        # retry ladder waits up to two hours, and until it was handed the abort
        # event there was NOTHING watching Esc or Ctrl+C during that wait. The
        # key set the event, the loop never read it, and the only way out of a
        # dropped connection was killing the process.
        from .backend import retrying_stream, shared_client_cm
        async with shared_client_cm() as client:
            async with retrying_stream(
                client,
                'POST',
                f"{cfg.model_base_url}/v1/chat/completions",
                _http_timeout,
                _stream_kwargs,
                signal=signal,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    # A rejected auth with no key configured is the one failure
                    # whose status alone hides the cause: the request went out
                    # with no Authorization header at all, so the server is
                    # answering "who are you", not "your key is wrong". A bare
                    # 401 is indistinguishable from a server that is down or a
                    # key that expired.
                    #
                    # Only on 401/403 and only when the key is empty: an empty
                    # key is legitimate against a local server with no auth, so
                    # warning up front would be noise on a working setup.
                    _hint = ''
                    if response.status_code in (401, 403) and not cfg.model_api_key:
                        _hint = ("\n  No API key is configured, so the request was sent "
                                 "unauthenticated. Set model_api_key in "
                                 "~/.codemason/config.json, or the environment variable "
                                 "CODEMASON_MODEL_API_KEY.")
                    raise Exception(f"Model API error: {response.status_code} - {error_text.decode('utf-8', errors='replace')}{_hint}")

                buffer = ''
                # -rt/--realtime: DSML/think/special-token-safe live renderer.
                # Holds back a trailing fragment that could still grow into a
                # marker, so nothing raw leaks even across chunk boundaries.
                _degen = _DegenerateRunGuard(cfg.degenerate_run_chars)
                rt_filter = None
                if on_content_delta is not None:
                    from ..utils.realtime_render import RealtimeStreamFilter
                    rt_filter = RealtimeStreamFilter()
                # Incremental UTF-8 decode: a multibyte char (a 3-byte CJK
                # codepoint) can straddle two aiter_bytes() chunks. Decoding each
                # chunk in isolation would turn the split halves into U+FFFD (the
                # a multi-byte character split across chunks). The incremental decoder holds an
                # incomplete trailing sequence until the next chunk completes it.
                # (Encrypted mode is base64/ASCII on the wire so it never split;
                # plaintext mode did — this makes both paths safe.)
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

                def _process_sse_line(line: str) -> None:
                    """Handle one `data: ...` SSE line. Extracted so the trailing
                    buffer (a final line with no newline) can be processed too."""
                    nonlocal full_content, final_finish_reason, got_reasoning
                    nonlocal streamed_tokens, usage, reasoning_text
                    nonlocal cached_prompt_tokens, stream_done
                    if not line.strip() or not line.startswith('data: '):
                        return

                    data = line[6:]
                    if data == '[DONE]':
                        # The terminator the protocol provides, used as one.
                        # Swallowing it left the loop to learn the response was
                        # over only when the byte stream ended, so a server that
                        # holds the connection open after sending it parked the
                        # loop inside the read until a stall guard fired:
                        # measured 2026-09-12 as a 600.0s wait that began ~3.1s
                        # into a turn whose usage chunk had ALREADY reported the
                        # final token count.
                        stream_done = True
                        return

                    # Encrypted stream: each SSE event is an independent
                    # AES-GCM message; decrypt back to the original JSON line.
                    if _aes_key is not None:
                        try:
                            data = _bc_dec(_aes_key, data).decode('utf-8')
                        except _BCErr as _e:
                            raise Exception(f"stream body decryption failed: {_e}")

                    try:
                        chunk_data = json.loads(data)
                    except json.JSONDecodeError:
                        return
                    # Server-side error delivered INSIDE a 200 SSE stream (the
                    # HTTP status was already 200). vLLM emits {"error": {...}} as
                    # a data chunk — with no `choices` — for e.g. an unsupported
                    # sampling param under speculative decoding. Without catching
                    # it here the loop drains to EMPTY content (no delta, no
                    # finish_reason) — the invisible blank-response failure.
                    if isinstance(chunk_data, dict) and chunk_data.get('error'):
                        _err = chunk_data['error']
                        _emsg = _err.get('message') if isinstance(_err, dict) else str(_err)
                        debug_log('STREAM_ERROR_CHUNK', {'error': _err})
                        raise Exception(f"Model API error (in-stream): {_emsg}")
                    # The final include_usage chunk has choices: [] (empty) —
                    # guard against IndexError; its usage is captured below.
                    _choices = chunk_data.get('choices') or []
                    choice = _choices[0] if _choices else {}
                    delta = choice.get('delta', {})

                    finish_reason = choice.get('finish_reason')
                    if finish_reason:
                        final_finish_reason = finish_reason
                        debug_log('STREAM_FINISH', {
                            'finish_reason': finish_reason,
                            'content_length': len(full_content),
                            'tool_calls_count': len(tool_calls),
                        })

                    # DeepSeek-V4 streams `reasoning_content`; the GLM-5.3 vLLM fork
                    # streams `reasoning`. Same channel, accept both.
                    _rc = delta.get('reasoning_content') or delta.get('reasoning')
                    if _rc:
                        got_reasoning = True
                        reasoning_text += _rc
                        _degen.feed(_rc)
                        _recount_thinking()
                        debug_log('REASONING_CONTENT_CHUNK', {
                            'length': len(_rc),
                        })

                    if 'content' in delta and delta['content']:
                        full_content += delta['content']
                        _degen.feed(delta['content'])
                        # -rt: stream the safe portion to the screen now.
                        if rt_filter is not None:
                            safe = rt_filter.feed(delta['content'])
                            if safe:
                                on_content_delta(safe)

                    # Advance the live spinner counter for BOTH answer and
                    # reasoning deltas — reasoning is real generation, so the count
                    # must not freeze during long thinking. Per-chunk tally; the
                    # exact end count comes from the final include_usage chunk.
                    if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning'):
                        _tokens_mod.mark_output_started()
                        streamed_tokens += 1
                        if on_token_update:
                            on_token_update(streamed_tokens)

                    if 'tool_calls' in delta:
                        # A tool-call delta is output too. Without this a turn
                        # whose whole answer is a tool call looked, to the
                        # stall guards, like a server that never answered
                        # (2026-09-12: 603.1s on the first-token budget), and
                        # a decoder that collapsed INSIDE an argument string
                        # was invisible to the degenerate-run guard.
                        _degen.note_progress()
                        for tc in delta['tool_calls']:
                            idx = tc.get('index', 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({
                                    'id': '',
                                    'type': 'function',
                                    'function': {'name': '', 'arguments': ''}
                                })
                            if tc.get('id'):
                                tool_calls[idx]['id'] = tc['id']
                            if tc.get('function', {}).get('name'):
                                tool_calls[idx]['function']['name'] = tc['function']['name']
                            _args = tc.get('function', {}).get('arguments')
                            # str only. 'arguments' is a string by the wire
                            # format, but a server that sends it as an object
                            # would raise TypeError on the += and take down the
                            # whole stream. Leaving the slot empty instead lets
                            # the assembly below report ONE lost call, which is
                            # recoverable, rather than losing the turn.
                            if _args and isinstance(_args, str):
                                tool_calls[idx]['function']['arguments'] += _args
                                _degen.feed(_args)

                    if 'usage' in chunk_data and chunk_data['usage']:
                        usage = chunk_data['usage']
                        # OpenAI-compatible servers report a prefix-cache hit as
                        # prompt_tokens_details.cached_tokens (a SUBSET of
                        # prompt_tokens). Nothing here ever read it, so every
                        # response recorded cache_read_input_tokens=0 and a fully
                        # warm 234k-token turn was indistinguishable in the logs
                        # from a cold one that re-prefilled all 234k. That is
                        # exactly the measurement needed when a turn stalls for
                        # minutes: a cold prefill of a ~200k prompt is ~3 minutes
                        # of silence, and without this field there is no way to
                        # tell it apart from a hung server.
                        _ptd = (usage.get('prompt_tokens_details')
                                if isinstance(usage, dict) else None)
                        _cached = (_ptd.get('cached_tokens')
                                   if isinstance(_ptd, dict) else None)
                        if isinstance(_cached, int) and _cached > cached_prompt_tokens:
                            cached_prompt_tokens = _cached
                        # vLLM continuous_usage_stats: running completion_tokens on
                        # each chunk -> use it as the EXACT live count (the tally
                        # above under-counts multi-token chunks). Monotonic guard;
                        # harmless on servers that only send usage in the final chunk.
                        _ct = usage.get('completion_tokens') if isinstance(usage, dict) else None
                        if isinstance(_ct, int) and _ct > streamed_tokens:
                            streamed_tokens = _ct
                            # The stall guards have to see this too, or a turn
                            # that emits only tool-call arguments looks exactly
                            # like a server that never answered. See
                            # _DegenerateRunGuard.note_progress.
                            _degen.note_progress()
                            if on_token_update:
                                on_token_update(streamed_tokens)

                # Race every read against the abort signal (utils.generators.
                # abortable) instead of only polling BETWEEN chunks. aiter_bytes()
                # blocks until the server sends something, and during PREFILL it
                # sends nothing at all — 122k tokens is ~80s at ~1.5k tok/s — so
                # the poll below never ran and Esc/Ctrl+C looked completely dead
                # until the first token landed (worst case the 1800s HTTP
                # timeout). Racing force-cancels the wedged read after the grace
                # window, tearing the connection down at once, so an interrupt
                # takes effect immediately no matter where the stream is. This is
                # the same fix the tool dispatch point already applies to wedged
                # tools; the model stream was simply never covered by it.
                aborted = False
                stream_done = False
                # Set by a stall guard; carries the line the user is shown. Kept
                # separate from the degenerate guard because the two end the
                # turn differently: a collapsed decoder is handed to Layer-2
                # recovery, a server that simply stopped sending is not.
                stall_reason = None
                # A cancelled read surfaces as an httpx error on the way out.
                # During an abort that is the expected outcome, not a failure —
                # the connection was torn down on purpose — and letting it escape
                # reaches the event loop, where prompt_toolkit prints "Unhandled
                # exception" and halts the REPL for a keypress. Only a stream that
                # broke with no abort pending is a real error.
                # Stall guards. See Config.first_token_timeout_secs: a request
                # that produces nothing and a stream that starts then goes quiet
                # are different failures and get different budgets.
                _stalled = {}

                def _note_stall(_started, _budget):
                    _stalled['started'] = _started
                    _stalled['budget'] = _budget

                _guarded = _stall_guarded(
                    response.aiter_bytes(),
                    float(cfg.first_token_timeout_secs or 0),
                    float(cfg.token_gap_timeout_secs or 0),
                    lambda: _degen.last_output_at is not None,
                    _note_stall,
                )
                try:
                    async for chunk_bytes in abortable(_guarded, signal):
                        # Still poll per chunk: a signal set while THIS chunk was in
                        # flight must not be processed as if it arrived before it.
                        if signal and signal.is_set():
                            debug_log('STREAM_ABORTED', {'content_length': len(full_content)})
                            aborted = True
                            break

                        chunk = decoder.decode(chunk_bytes)
                        buffer += chunk
                        lines = buffer.split('\n')
                        buffer = lines.pop()

                        for line in lines:
                            _process_sse_line(line)

                        # Degenerate-run guard tripped. This does NOT abort the
                        # turn: it takes over the signal the server-side detector
                        # used to send, so the Layer-2 recovery in query.py runs
                        # exactly as it always did — discard the looping tool
                        # calls, nudge the model to do something different, up to
                        # REPETITION_RECOVERY_LIMIT times, without the user
                        # seeing a stopped turn at all.
                        #
                        # That ladder went dark for GLM on 2026-09-06, when
                        # repetition detection was defaulted off for the family
                        # to stop it cutting a live API enumeration. Nothing else
                        # can produce finish_reason 'repetition', so switching
                        # the detector off also switched the RECOVERY off, and a
                        # loop that used to be caught at ~140 tokens and quietly
                        # nudged instead ran to max_tokens: 63 minutes a request,
                        # three recoveries behind it. The guard is the
                        # client-side replacement for that signal, and it cannot
                        # false-positive the way the n-gram detector did.
                        #
                        # The collapsed output itself is written to disk and then
                        # dropped. Left in history the model reads its own
                        # garbage back on every later turn — the same reason
                        # _stale_recovery_indices prunes spent nudges.
                        if _degen.tripped is not None:
                            log_error(_degen.tripped,
                                      '_openai_common: degenerate run')
                            debug_log('DEGENERATE_RUN',
                                      {'reason': _degen.tripped})
                            _write_cut_dump('degenerate', _degen.tripped,
                                            reasoning_text, full_content,
                                            time.time() - start_time)
                            reasoning_text = ''
                            full_content = ''
                            tool_calls = []
                            final_finish_reason = 'repetition'
                            break

                        # The response said it was over. Checked AFTER the
                        # degenerate guard so a collapse spotted in the same
                        # batch still wins, and after the whole batch so nothing
                        # queued behind [DONE] is dropped.
                        if stream_done:
                            debug_log('STREAM_DONE',
                                      {'content_length': len(full_content)})
                            break

                except Exception as _stream_exc:
                    if signal is not None and signal.is_set():
                        debug_log('STREAM_ABORTED', {'via': type(_stream_exc).__name__})
                        aborted = True
                    else:
                        raise
                # A stall guard fired: say so, keep whatever did arrive on
                # disk, and end the turn. Unlike the degenerate-run guard this
                # is NOT handed to Layer-2 recovery — nothing here suggests the
                # model is looping, and nudging a server that never answered
                # would just spend the recovery budget on a transport problem.
                if _stalled:
                    _why = (
                        'the model went silent mid-answer: no output for %.0fs '
                        'after %d characters. Generation was cut here.'
                        % (_stalled['budget'],
                           len(full_content) + len(reasoning_text))
                        if _stalled['started'] else
                        'the model produced nothing for %.0fs — the prompt was '
                        'accepted but no token ever arrived. Generation was cut '
                        'here.' % _stalled['budget'])
                    log_error(_why, '_openai_common: stalled stream')
                    debug_log('STREAM_STALLED',
                              {'reason': _why, 'started': _stalled['started']})
                    _write_cut_dump('stalled', _why, reasoning_text,
                                    full_content, time.time() - start_time)
                    # NOT signal.set(). `signal` is the turn's shared abort
                    # event — every tool, every subagent, the subagent gate,
                    # compaction, TaskOutput and the Esc key read it — so
                    # setting it to end a wedged read also cancelled work that
                    # had nothing to do with this request, and left the turn
                    # indistinguishable from one the user had stopped, printed
                    # as "[Request interrupted by user]". _stall_guarded has
                    # already ended iteration, so nothing needs cancelling here:
                    # the flag below is enough, and the reason travels with it.
                    stall_reason = _why
                    aborted = True

                # abortable() ends the loop by RETURNING when it force-cancels a
                # wedged read, so the in-loop check above never runs in exactly
                # the prefill case this fixes. Re-derive the flag from the signal
                # or the trailing buffer below would be parsed as a clean finish.
                if not aborted and signal is not None and signal.is_set():
                    debug_log('STREAM_ABORTED', {'content_length': len(full_content)})
                    aborted = True

                # An aborted stream left the underlying connection half-read.
                # Close the response HERE, inside our guard, swallowing the
                # teardown. Otherwise the `async with retrying_stream(...)` exit
                # below runs response.aclose() itself, where httpx maps the torn
                # read to its own exception — and raised from __aexit__ it escapes
                # PAST every try in this function to the event loop, where
                # prompt_toolkit prints "Unhandled exception" and freezes the REPL
                # on a keypress (the Ctrl+C-mid-stream bug). aclose() sets
                # is_closed before it may raise and is idempotent, so the
                # context-manager exit then no-ops and nothing escapes.
                if aborted:
                    try:
                        await response.aclose()
                    except BaseException:
                        pass

                # Process a final SSE line the stream left WITHOUT a trailing
                # newline. Some servers send the last include_usage chunk
                # unterminated; dropping it lost the exact token counts (they then
                # fell back to estimates). Skip on abort — the stream was cut.
                if not aborted and buffer.strip():
                    _process_sse_line(buffer)

                # Settle the reasoning count: the throttle above may have skipped
                # the last deltas, and on a SHORT think it may never have fired at
                # all. No-op when the turn produced no reasoning.
                _recount_thinking(force=True)
                if thinking_tokens:
                    debug_log('THINKING_TOKENS', {
                        'tokens': thinking_tokens,
                        'chars': len(reasoning_text),
                    })

        # -rt: emit any held-back tail (never resolved into a marker → real text).
        if rt_filter is not None:
            tail = rt_filter.flush()
            if tail:
                on_content_delta(tail)

        duration_ms = int((time.time() - start_time) * 1000)

        # [DIAGNOSTIC PROBE] DSML content-leak capture. Env-gated, default off.
        # Set CODEMASON_RAW_DUMP=/path/to/file to append the PRE-clean raw content
        # plus finish_reason and the count of STRUCTURED tool_calls the stream
        # parser saw. Purpose: detect the DeepSeek-V4 bug where a tool call is
        # serialised into `content` (finish_reason='stop', 0 structured calls)
        # instead of the tool_calls channel — before clean_special_tokens erases
        # the DSML/JSON markup. Records only; changes no behaviour.
        _dump_path = os.getenv('CODEMASON_RAW_DUMP')
        if _dump_path:
            try:
                _rec = {
                    'ts': time.time(),
                    'finish_reason': final_finish_reason,
                    'n_structured_tool_calls': len(tool_calls),
                    'raw_content': full_content,
                }
                with open(_dump_path, 'a') as _f:
                    _f.write(json.dumps(_rec, ensure_ascii=False) + '\n')
            except Exception:
                pass

        # Build response content
        # An interrupted turn drops everything it had produced. That is correct
        # for the CONVERSATION — folding a half-finished answer into the history
        # would change the next request's prompt and cost tokens for work the
        # user just stopped — but it also means the evidence goes with it.
        #
        # Measured 2026-09-07: a turn ran 37.7 minutes and produced 36,946
        # tokens (4,619 of them reasoning) with no text and no tool call on
        # screen, and ESC left one line in the transcript: "[Request interrupted
        # by user]". Whether it was a genuine long derivation or a loop
        # repeating itself could only be argued from the token rate, because not
        # one character survived.
        #
        # So: to disk, always. The history is untouched.
        #
        # This was behind -dbg until 2026-09-10, which put it in the same trap
        # the comment above describes: the flag has to be set BEFORE the run
        # that goes wrong, and nobody launches an ordinary session expecting to
        # need it. That day a turn sat for 2,500s emitting nothing; the flag was
        # not on, so ESC would again have left one transcript line. The evidence
        # came from a core dump of the live process instead, which is not a
        # procedure anyone should need. An abort is a rare, one-shot event — it
        # costs nothing to record unconditionally, exactly like turns.log.
        #
        # A stall is not written here a second time: _write_cut_dump already put
        # it in stalled-<ts>.txt with its reason.
        if aborted and not stall_reason:
            try:
                from ..utils.config import LOG_DIR
                from ..utils.file import atomic_write_bytes
                from ..utils.log import ensure_log_dir, now_iso
                # Same two helpers, same reason, as the cut dump above.
                ensure_log_dir()
                _name = 'aborted-%s.txt' % now_iso().replace(':', '-').replace('.', '-')
                _elapsed = time.time() - start_time
                atomic_write_bytes(
                    str(LOG_DIR / _name),
                    ('aborted after %.1fs\nreasoning: %d chars\ncontent: %d chars\n'
                     '\n===== REASONING =====\n%s\n\n===== CONTENT =====\n%s\n'
                     % (_elapsed, len(reasoning_text), len(full_content),
                        reasoning_text, full_content)).encode('utf-8'))
                log_error(
                    'turn interrupted after %.0fs — %d reasoning + %d content chars saved to %s'
                    % (_elapsed, len(reasoning_text), len(full_content), LOG_DIR / _name),
                    '_openai_common: aborted turn')
            except Exception as _e:
                # Must be visible too: a dump that silently failed to write is
                # the same lost evidence as no dump at all.
                log_error('could not write the aborted-turn dump: %s' % _e,
                          '_openai_common: aborted turn')

        content_blocks = []

        # Keep this turn's thinking on the message so the next request can send
        # it back (see ChatMessage.reasoning_content). It leads the block list
        # because that is the order the model produced it in: think, then answer
        # or call a tool. Storing it costs context but losing it costs the
        # conclusion — the model re-runs the same tool call to rediscover what it
        # already knew. Gated so a backend without the channel never stores it.
        if reasoning_roundtrip and reasoning_text.strip():
            content_blocks.append(ThinkingBlock(type='thinking',
                                                thinking=reasoning_text))

        # Recover any tool call that leaked into `content` (DeepSeek-V4 #1244)
        # BEFORE cleaning strips the markup. On the normal path — model used the
        # structured tool_calls channel — this is a no-op and `residual` == raw.
        recovered_calls, residual_content = recover_leaked_tool_calls(full_content)
        if recovered_calls:
            debug_log('RECOVERED_LEAKED_TOOL_CALLS', {
                'count': len(recovered_calls),
                'names': [c['name'] for c in recovered_calls],
                'finish_reason': final_finish_reason,
                'n_structured': len(tool_calls),
            })

        # Clean and add text content (from the residual, sans leaked tool markup)
        if residual_content:
            cleaned_content = clean_special_tokens(residual_content, reasoning_separated=got_reasoning)
            if got_reasoning and '<think>' in residual_content:
                # A leaked-think glitch (llama.cpp#22398): record it so the
                # frequency is visible, and confirm we preserved the inner text.
                debug_log('INLINE_THINK_RECOVERED', {
                    'raw_len': len(residual_content),
                    'cleaned_len': len(cleaned_content),
                })
            if cleaned_content:
                content_blocks.append(TextBlock(type='text', text=cleaned_content))

        # Signatures of structured tool calls, so a recovered duplicate (same
        # call echoed both on-channel and in content) isn't dispatched twice.
        _seen_sigs = set()

        # Tool calls the server sent that could not be turned into an
        # invocation. Collected rather than discarded: a dropped call is how a
        # turn ends having ANNOUNCED work it never did, which reads to the user
        # as the model quietly stopping. Everything that can go wrong upstream —
        # a stream cut mid 'arguments' so the JSON is truncated, an 'arguments'
        # that arrives as an object instead of a string, a server that numbers
        # tool_calls from a nonzero index — converges HERE, so this one count
        # covers all of them without having to tell them apart.
        lost_tool_calls = []

        # Add structured tool calls (the normal channel)
        for tc in tool_calls:
            name = tc['function'].get('name') or ''
            args = tc['function'].get('arguments')
            # An index filler from the delta loop: a server that opens at index
            # 1 leaves an untouched slot at 0. It is not a call the model made,
            # so it is neither dispatched nor counted as lost.
            if not name and not args:
                continue
            try:
                # TypeError as well as JSONDecodeError: 'arguments' is a string
                # by the wire format, but a server that sends it as an object
                # reaches json.loads with a dict, and an uncaught TypeError here
                # takes down the whole turn rather than losing one call.
                tool_input = json.loads(args)
                content_blocks.append(ToolUseBlock(
                    type='tool_use',
                    id=tc['id'] or str(uuid4()),
                    name=name,
                    input=tool_input
                ))
                _seen_sigs.add((name, json.dumps(tool_input, sort_keys=True)))
            except (json.JSONDecodeError, TypeError) as e:
                lost_tool_calls.append(f'{name or "<unnamed>"}: {e}')
                debug_log('FAILED TO PARSE TOOL ARGUMENTS', {'tool_call': tc})

        # Add recovered (leaked) tool calls, de-duplicated against structured ones
        for rc in recovered_calls:
            sig = (rc['name'], json.dumps(rc['input'], sort_keys=True, default=str))
            if sig in _seen_sigs:
                continue
            _seen_sigs.add(sig)
            content_blocks.append(ToolUseBlock(
                type='tool_use',
                id=str(uuid4()),
                name=rc['name'],
                input=rc['input'],
            ))

        # A lost call goes to the ERROR log, not just the debug log: debug_log
        # only writes when CODEMASON_DEBUG is set, so under an unattended run
        # the one record of why the turn went quiet is the one record that does
        # not exist.
        # But NOT when the user aborted. Cutting the stream mid-tool-call
        # leaves the arguments half-written by definition, so the parse
        # "failure" is the expected outcome of pressing ESC, not a defect.
        # Logging it as one cost real time: 56 of 60 such records in a real
        # user's log sat at column 0 or column 9 — the first argument fragment,
        # the signature of a cut, not of a model that emits bad JSON — and they
        # were read as evidence of a truncation bug that did not exist.
        if lost_tool_calls and not aborted:
            log_error(
                'dropped %d tool call(s) the model sent — arguments did not parse: %s'
                % (len(lost_tool_calls), '; '.join(lost_tool_calls)),
                '_openai_common: tool call assembly',
            )

        # Say what happened IN THE TRANSCRIPT, where the person reading it is.
        # Until this, the only record of a stall was one entry in error.log, so
        # the turn showed on screen as an interrupt and left the user to
        # conclude they had stopped it themselves.
        if stall_reason:
            content_blocks = [c for c in content_blocks
                              if not (c.type == 'text' and not c.text)]
            content_blocks.append(
                TextBlock(type='text', text='\n\n[Stopped: %s]' % stall_reason))

        # Ensure at least one content block
        if not content_blocks:
            content_blocks.append(TextBlock(type='text', text=''))

        debug_log('FINAL CONTENT ARRAY', {
            'content_length': len(content_blocks),
            'content_types': [c.type for c in content_blocks]
        })

        # Map the OpenAI/llama.cpp finish_reason to the Anthropic stop_reason
        # (substrate boundary). 'length' means the output hit the token cap and
        # was cut mid-stream -> 'max_tokens', which query.py's max_output
        # recovery keys on to resume the model instead of ending the turn.
        # A completed tool call -> 'tool_use'; everything else -> 'end_turn'.
        # A recovered leaked call is a real invocation too — it must drive the
        # 'tool_use' stop_reason so query.py dispatches it instead of ending the
        # turn (otherwise the recovered call sits in content and never runs).
        # 'repetition' (vLLM) is checked FIRST and always ends the turn. It must
        # not reach the two branches below: 'max_tokens' would hand it to the
        # max_output recovery, which resumes the model from where it stopped —
        # i.e. resumes the loop — and 'tool_use' would dispatch a tool call the
        # model emitted while already repeating itself. The cut also leaves the
        # text ending mid-thought, so it is labelled: the note is what the model
        # reads on the next turn instead of an answer that simply stops, and it
        # names the cause rather than leaving it to infer one.
        if final_finish_reason == 'repetition':
            # Surface the raw vLLM signal; query.py owns the policy (layer-2
            # recovery: nudge the model to self-correct N times before giving
            # up). The baked-in end_turn + note moved there.
            stop_reason = 'repetition'
        elif final_finish_reason == 'length':
            stop_reason = 'max_tokens'
        elif any(c.type == 'tool_use' for c in content_blocks):
            # Derived from what was actually BUILT, never from what arrived. The
            # two used to be able to disagree: the old test was "did the server
            # send tool calls", so a call that failed to parse still set
            # 'tool_use' while producing no tool_use block, and query.py read
            # that as a turn with nothing to dispatch and ended silently.
            # Asking the content list removes the disagreement by construction.
            #
            # Deliberately ABOVE the stall branch: a tool call the model
            # finished emitting is real work, fully received and fully parsed,
            # and the stall happened after it. A truncated call never becomes a
            # tool_use block, so what reaches here is only ever complete.
            stop_reason = 'tool_use'
        elif stall_reason:
            # A stall guard cut the stream. Surface the raw signal and let
            # query.py own the policy, exactly as 'repetition' does — but the
            # policy there is deliberately different: a stall is NOT handed to
            # Layer-2 recovery, because nothing about a server that stopped
            # sending suggests the model is looping.
            #
            # Without this branch a stall with no output lands on 'empty_turn'
            # and gets nudged, spending the recovery budget on a transport
            # problem; before the shared abort event stopped being set, it came
            # out one step earlier still, as "[Request interrupted by user]".
            stop_reason = 'stalled'
        elif lost_tool_calls and not aborted:
            # The model DID call a tool and none of it survived. Not an
            # end_turn — say so, and let query.py own the recovery policy.
            #
            # Unless the user aborted: 'tool_parse_failed' is the signal
            # query.py keys its retry policy on, and re-issuing a call the user
            # just stopped is the opposite of what they asked for.
            stop_reason = 'tool_parse_failed'
        elif not full_content.strip():
            # Finished cleanly having produced nothing at all: no text, no tool
            # call, nothing lost in parsing. That is not a turn ending, it is a
            # turn that never happened.
            #
            # Observed 2026-09-07 in an unattended run: after a command returned
            # an empty result (a sed range past the end of the file) the model
            # spent 125 reasoning tokens and answered with zero characters.
            # finish_reason said "stop", so the turn ended and the session sat
            # waiting for the user — the silent stop that -dsp exists to avoid,
            # with nothing on screen to say why.
            #
            # The tool-call check above has to come first: answering with a tool
            # call and no prose is normal and must not land here.
            stop_reason = 'empty_turn'
        else:
            stop_reason = 'end_turn'

        # Prompt-token split for BetaUsage below. Clamped because a server
        # reporting more cached than prompt tokens would drive input_tokens
        # negative and corrupt every total derived from it; the smaller,
        # self-consistent figure is the one to trust.
        _prompt_tokens = usage.get('prompt_tokens', 0) if usage else 0
        _cached_tokens = min(cached_prompt_tokens, _prompt_tokens)

        # The SERVER-raised repetition signal used to leave the collapsed
        # generation in the message, while the CLIENT-side detector clears its
        # buffers for exactly the reason documented there: a collapsed
        # generation in history is read back on every later turn. The asymmetry
        # meant a server-side cut put the '!!!!!'/'★★★★' run into history and
        # re-sent it on every subsequent request — and because the stale-recovery
        # prune only removes an EMPTY assistant stub, it took out the explaining
        # nudge and left the garbage behind. Same treatment on both paths: the
        # run has already been streamed to the screen, so nothing the user
        # should see is lost by not storing it.
        if stop_reason == 'repetition':
            content_blocks = []

        # A collapsed decoder is the only direct evidence of where THIS model
        # stops working, as opposed to where the server stops accepting.
        # Recorded here because this is the one place holding both halves of the
        # measurement — the stop reason and the prompt size that earned it.
        #
        # Only "repetition" counts. A max_tokens or stalled turn says nothing
        # about the window, and treating either as a collapse would record a
        # ceiling from evidence that is not about length at all.
        from .compaction import note_decoder_collapse as _note_collapse
        _note_collapse(_prompt_tokens)

        # Create response message
        api_message = APIMessage(
            id=str(uuid4()),
            type='message',
            role='assistant',
            content=content_blocks,
            model=cfg.model_name,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=BetaUsage(
                # Anthropic accounting, which every consumer in this tree
                # assumes: input_tokens EXCLUDES the cached prefix, and the
                # total is input + cache_read + cache_creation + output
                # (utils/tokens.py _usage_total, utils/formatting.py
                # format_token_display). OpenAI accounting is the opposite —
                # prompt_tokens INCLUDES cached_tokens — so the cached part is
                # SUBTRACTED here, not added. Reporting the raw prompt_tokens
                # next to a non-zero cache_read would double-count the prefix
                # and inflate the context-fill percentage that drives
                # auto-compact: a warm 234k turn would read as 467k and
                # summarise away a conversation that had not filled the window.
                # The sum is unchanged by this split, so anchoring and the
                # default footer read exactly as before; only the previously
                # invisible cache figure is new.
                input_tokens=max(0, _prompt_tokens - _cached_tokens),
                output_tokens=usage.get('completion_tokens', 0) if usage else 0,
                cache_read_input_tokens=_cached_tokens,
            )
        )

        # One line per request, always. See utils.log.log_turn for why this is
        # unconditional rather than threshold- or debug-gated.
        #
        # The three numbers a slow turn is actually about, side by side: how much
        # of the prompt the server had to recompute (prefill, straight from
        # vLLM's own counters), how much it thought, and how much it wrote. At
        # the measured 1,335 tok/s prefill and 35 tok/s decode, those three
        # explain essentially all of any wait, and having them written down is
        # what settles "it was prefilling" without anyone having to reconstruct
        # it from timings afterwards.
        try:
            from ..utils.log import log_turn as _log_turn
            _out = api_message.usage.output_tokens
            _prefill = _tokens_mod.pending_prefill()
            _secs = duration_ms / 1000.0
            _log_turn({
                'prompt': _prompt_tokens,
                'prefill': _prefill,
                'cached': _tokens_mod.pending_cached(),
                'measured': str(_tokens_mod.prefill_is_measured()).lower(),
                'think': thinking_tokens,
                'out': _out,
                'dur_s': '%.1f' % _secs,
                # What the time was spent on, at the rates measured on this
                # deployment. A turn whose seconds are not covered by these is
                # the interesting one.
                'est_prefill_s': '%.1f' % (_prefill / 1335.0),
                'est_decode_s': '%.1f' % (_out / 35.0),
                'stop': api_message.stop_reason or '',
                'aborted': str(aborted).lower(),
            })
        except Exception as _e:
            debug_log('TURN_LOG_FAILED', {'error': str(_e)})

        return AssistantMessage(
            message=api_message,
            cost_usd=0.0,
            duration_ms=duration_ms,
            type='assistant',
            uuid=uuid4()
        )

    except Exception as e:
        # repr(e), not str(e): many network exceptions (ReadTimeout, ConnectError,
        # RemoteProtocolError, ...) have an EMPTY str(), which turned both the log
        # and the user-facing message into a blank "Error connecting to the model
        # server:"
        # with no way to tell what actually failed. repr() always carries the type.
        debug_log('ERROR', {'error': repr(e)})
        duration_ms = int((time.time() - start_time) * 1000)

        error_message = APIMessage(
            id=str(uuid4()),
            type='message',
            role='assistant',
            content=[TextBlock(type='text', text=f"Error connecting to the model server: {repr(e)}")],
            model='<synthetic>',   # API-error message carries zero usage; flag synthetic so it can't anchor the token count
            stop_reason='end_turn',
            stop_sequence=None,
            usage=BetaUsage()
        )

        return AssistantMessage(
            message=error_message,
            cost_usd=0.0,
            duration_ms=duration_ms,
            type='assistant',
            uuid=uuid4(),
            is_api_error_message=True
        )
