"""
Configuration management for CodeMason
Equivalent to src/utils/config.ts
"""

import json
import os
import string
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional


# Config directory
# ─────────────────────────────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH for every codemason config/state location.
# One env var (CODEMASON_CONFIG_DIR) relocates the whole tree; default ~/.codemason.
# All modules (env, log, hooks, keep_going_installer) import these — never
# redefine a path elsewhere.
# ─────────────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(os.environ.get('CODEMASON_CONFIG_DIR') or (Path.home() / '.codemason'))
CONFIG_FILE = CONFIG_DIR / 'config.json'
SETTINGS_FILE = CONFIG_DIR / 'settings.json'
MEMORY_DIR = CONFIG_DIR / 'memory'
# Where the logs go. CODEMASON_LOG_DIR relocates them; -dbg points it at a
# readable /tmp directory so a second pair of eyes can tail the run live
# instead of asking the user to copy lines out of a terminal.
LOG_DIR = Path(os.environ['CODEMASON_LOG_DIR']) if os.environ.get('CODEMASON_LOG_DIR') else CONFIG_DIR / 'logs'
HOOKS_DIR = CONFIG_DIR / 'hooks'
# Every inter-process lock in the system lives here, named after what it
# protects, so there is one directory to audit and no .lock files scattered
# through the tree the user reads. The files are empty and are never deleted —
# see utils/file.FileLock for why that is the safe shape and not residue.
LOCKS_DIR = CONFIG_DIR / 'locks'


# Folding is ASCII-only on purpose. The two builds must agree byte for byte, and
# full Unicode lowercasing has cases (the Turkish dotted I) where Rust's
# to_lowercase and Python's str.lower need not — exactly the one-byte difference
# endpoint_tag warns about. Every byte either build folds here is ASCII.
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _fold_endpoint(url: str) -> str:
    """`url` with its scheme and host lowercased and its path left alone.

    Hand-rolled rather than handed to a URL parser so the Rust build can land on
    the same bytes from the same few lines: find the authority, fold that, copy
    the rest. The separators are ASCII, so both builds slice at the same index.
    """
    start = url.find('://')
    start = start + 3 if start >= 0 else 0
    cut = url.find('/', start)
    if cut < 0:
        cut = len(url)
    return url[:cut].translate(_ASCII_LOWER) + url[cut:]


def endpoint_tag(base_url: str = '') -> str:
    """Short fingerprint of a model endpoint, for keying anything whose real
    owner is the SERVER rather than the user or the session.

    Hashed rather than used raw because a URL is not a filename, and normalized
    first (trim, drop a trailing '/') so the same server written two ways is one
    tag. The Rust client computes this identically — anything keyed by it is
    shared between the builds, so a one-byte difference would split the
    namespace in half without any error.

    Case is folded over the scheme and host, which are case-insensitive, so
    'http://LocalHost:8000' and 'http://localhost:8000' are one pool rather than
    two sessions rationed against one GPU as if it were two. The PATH keeps its
    case: a path is case-sensitive, and merging two endpoints that differ there
    would make each wait on a cache the other never touches.

    Spellings that differ by more than case (localhost vs 127.0.0.1, a proxy vs
    the server behind it) read as two endpoints no matter what; point both
    sessions at one string if they share a server.
    """
    url = _fold_endpoint((base_url or get_config().model_base_url or '').strip().rstrip('/'))
    if not url:
        return 'none'
    import hashlib
    return hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]


def lock_path(name: str) -> str:
    """The lock sidecar for a shared file. `name` is what it protects
    ('config.json'); the Rust client derives the same path from the same name,
    so the two builds exclude each other and not just their own kind."""
    return str(LOCKS_DIR / f'{name}.lock')


def config_lock_path() -> str:
    """The lock the config writer takes. Its own function so the test pins the
    path the writer actually uses, rather than one spelled the same way twice."""
    return lock_path(CONFIG_FILE.name)


# Legacy config.json keys -> their neutral replacements (the llamacpp_* -> model_*
# rename). Consumed by Config.from_dict so an older on-disk config still loads.
_LEGACY_CONFIG_KEYS = {
    'llamacpp_base_url': 'model_base_url',
    'llamacpp_model': 'model_name',
    'llamacpp_api_key': 'model_api_key',
    'llamacpp_aes_key': 'model_aes_key',
}


def _env_first(default, *names):
    """Return the first environment variable that is set among `names` (new
    canonical name first, legacy alias last), else `default`. Lets the renamed
    CODEMASON_MODEL_* vars take over while the old LLAMACPP_* still resolve, so an
    existing .env / launch script keeps working unchanged."""
    for _n in names:
        _v = os.getenv(_n)
        if _v is not None:
            return _v
    return default


# The on-disk schema version this build writes. Bump it when a default changes
# in a way an existing config.json must pick up, and add the step to
# _MIGRATIONS. A file without the key predates versioning and counts as 1.
CONFIG_VERSION = 3

# Defaults that were later found to be defects. A config written by an older
# version holds them as if the user had chosen them — first run materializes
# every default into the file — so a fixed default would never reach that user.
# Each step replaces a value ONLY when it still equals what the old version
# wrote; anything else is the user's choice and is left alone. Runs once per
# file: the version is stamped forward afterwards.
_MIGRATIONS = {
    2: (
        # Truncated long reasoning turns mid-answer (default until 2026-08-13).
        ('max_tokens', (16384,), 65536),
        # A turn cap that silently ended -dsp runs at turn 200 (default for
        # 2h40m on 2026-09-04).
        ('max_turns', (200,), None),
        # "Send no effort" (default until 2026-08-06); policy is high, always.
        ('reasoning_effort', ('',), 'high'),
        # The Rust build's name for auto (default until 2026-10-05).
        ('sampling_profile', ('vendor',), 'auto'),
    ),
    3: (
        # The serial line's slot count. 1 was its default and 2 its only other
        # sensible value there; on this line the tools are concurrency-safe, so a
        # file carried over from main would let the code dispatch three spawns
        # and the gate still admit one — parallel on paper, serial in fact, with
        # nothing reporting it. Anything at 3 or above is a deliberate choice and
        # is left alone.
        ('max_concurrent_subagents', (1, 2), 3),
    ),
}

# Fields whose default is None: the type a non-null value must have.
_OPTIONAL_FIELD_TYPES = {'max_turns': int, 'language': str, 'ui_language': str}
_TRUE_WORDS = {'true', '1', 'yes', 'on'}
_FALSE_WORDS = {'false', '0', 'no', 'off'}


def _coerce_field(name: str, value: Any, default: Any):
    """(ok, value) — `value` read as the type `name` expects.

    Someone who hand-edits config.json without knowing the schema writes
    "false" for false and "16384" for 16384. Taken as-is, "false" is a truthy
    string: `"dangerously_skip_permissions": "false"` turned permission prompts
    OFF. So the common spellings are read as what they mean, and anything that
    cannot be read is reported and left to the default.
    """
    kind = _OPTIONAL_FIELD_TYPES.get(name)
    if kind is not None:
        if value is None:
            return True, None
    else:
        kind = type(default)
    if kind is bool:
        if isinstance(value, bool):
            return True, value
        if isinstance(value, (int, float)) and value in (0, 1):
            return True, bool(value)
        if isinstance(value, str) and value.strip().lower() in _TRUE_WORDS | _FALSE_WORDS:
            return True, value.strip().lower() in _TRUE_WORDS
        return False, None
    if kind is int:
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            return True, value
        if isinstance(value, float) and value.is_integer():
            return True, int(value)
        if isinstance(value, str):
            try:
                f = float(value.strip())
            except ValueError:
                return False, None
            return (True, int(f)) if f.is_integer() else (False, None)
        return False, None
    if kind is float:
        if isinstance(value, bool):
            return False, None
        if isinstance(value, (int, float)):
            return True, float(value)
        if isinstance(value, str):
            try:
                return True, float(value.strip())
            except ValueError:
                return False, None
        return False, None
    if kind is str:
        if isinstance(value, str):
            return True, value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, str(value)
        return False, None
    if kind is dict:
        return (True, value) if isinstance(value, dict) else (False, None)
    return True, value


def _say(text: str, **kwargs) -> None:
    """One line to stderr, in the UI language. Never raises: a config problem
    is reported on the way to running, not turned into a second failure."""
    try:
        from .i18n import gettext
        line = gettext(text, **kwargs)
    except Exception:
        try:
            line = text.format(**kwargs)
        except Exception:
            line = text
    try:
        import sys
        sys.stderr.write(line + '\n')
    except Exception:
        pass


@dataclass
class Config:
    """CodeMason configuration"""
    # Model-server settings (backend-agnostic — llama.cpp or vLLM, selected by
    # `backend` below). Defaults are the live DSV4 endpoint so a fresh checkout
    # (no ~/.codemason/config.json — e.g. the dev dir unzipped on another machine)
    # just works with zero setup. Override per-machine via ~/.codemason/config.json
    # or env (CODEMASON_MODEL_BASE_URL / CODEMASON_MODEL_NAME / CODEMASON_MODEL_API_KEY;
    # the legacy LLAMACPP_BASE_URL / LLAMACPP_MODEL / LLAMACPP_API_KEY still work).
    # Key is hardcoded here because this repo is PRIVATE; move it to env if it ever
    # goes public.
    # Schema version of this file (see CONFIG_VERSION / _MIGRATIONS). Not a
    # setting: it records which defaults the file has already been moved past.
    config_version: int = CONFIG_VERSION

    model_base_url: str = 'https://llm.systemboan.net'
    model_name: str = 'glm-5.3-flash'
    # No baked-in credential (2026-09-14). A key committed here ships in the
    # source and in every checkout, and rotating it means editing and
    # redistributing code — which is what it cost when this one was read out
    # loud during a screen recording. Existing installs already carry it in
    # ~/.codemason/config.json; this default only affects a fresh checkout,
    # which should be told the key once rather than born knowing it. Matches the
    # Rust client, which has never hardcoded one.
    model_api_key: str = ''

    # Optional symmetric key (base64 or hex, 32 bytes) for end-to-end BODY
    # encryption of the request/response with the vLLM endpoint. When set,
    # base_url should point at the decrypt gateway (deploy/encrypt_gateway.py)
    # sitting in front of vLLM. When EMPTY (default), nothing changes — bodies are
    # sent in plaintext exactly as before. See utils/body_crypto.py for the model:
    # the key is never transmitted, so it survives a MITM even with a trusted-CA
    # interception. Env override: CODEMASON_MODEL_AES_KEY (legacy LLAMACPP_AES_KEY).
    model_aes_key: str = ''

    # Model server backend. Selects the service implementation (services/vllm.py
    # vs services/llamacpp.py) and thus which server-specific request options are
    # sent. Set this when you configure the endpoint — you already know whether
    # the URL is a vLLM or a llama-server. 'auto' probes the server once
    # (/slots => llama.cpp, else vLLM). Env override: CODEMASON_BACKEND.
    backend: str = 'vllm'   # 'vllm' | 'llamacpp' | 'auto'  (DSV4 endpoint = vllm)

    # Directory of the served model. When set and it contains a tokenizer.json,
    # token counting uses the model's OWN tokenizer (exact, matches the server)
    # instead of a character heuristic (which mis-counts Korean / code for
    # DSV4). Loading the real tokenizer also switches OFF the non-canonical
    # _manage_context pre-send trim (see services/vllm.py) so the faithful
    # auto-compaction port is the only context defense, exactly like upstream.
    # Empty (or no tokenizer.json) => fall back to heuristic + _manage_context.
    # Env override: CODEMASON_MODEL_DIR.
    model_dir: str = ''

    # LLM request parameters
    # 65536 — a reasoning model needs room to think before it answers, and this
    # ceiling is the only brake left.
    #
    # 16384 was the server's -n and looked like the matching value, but it is not
    # the same quantity: -n bounds what the server will emit, this bounds one
    # request, and with thinking on the reasoning trace is counted in it. A
    # measured hard maths question at reasoning_effort=high spent 26,271
    # completion tokens (71.5K characters of reasoning, 781 of answer) and
    # finished on its own — under 16384 it would have been cut off mid-thought,
    # and a truncated turn looks like a model that cannot answer rather than a
    # limit that was set too low.
    #
    # Not larger, though the vLLM recipe permits up to 384K at high/max: the
    # ceiling is what stops a degenerate loop from running until the context ends,
    # and 2.5x the measured worst case keeps that brake meaningful. Raise it per
    # request for the rare job that needs more.
    max_tokens: int = 65536
    # 0.6 — the value this deployment actually runs on, and has for months.
    #
    # This briefly read 1.0, on the argument that a prose loop had been seen at
    # 0.6 so the temperature must be what caused it. That attributed the loop to
    # the wrong thing. The loop was diagnosed to reasoning_effort=low — which for
    # this model injects an EMPTY prefix rather than a "be brief" instruction, so
    # it leaves the model on its trained default and contributes nothing — in
    # combination with uninformative tool results. Raising effort to high and
    # letting the server cut an actual n-gram cycle (repetition_* below) is what
    # fixed it; the temperature was never the lever.
    #
    # Left at 1.0, a default nobody runs also becomes a trap: the on-disk config
    # is what every session uses, and any path that rewrites that file drops the
    # deployment onto a temperature it was never tested at.
    #
    # The penalties below are NOT the lever here either — see why they stay off,
    # and repetition_* further down for what catches a loop that still happens.
    # The auto profile (recommended_sampling) is what actually runs; these are
    # the starting point someone switching to "manual" inherits. GLM's vendor
    # numbers — temperature 1.0, top_p 0.95, no top_k — are the RECOMMENDED
    # values and live in recommended_sampling, not here. This field is the
    # DEFAULT: what a fresh config.json is stamped with, and what this
    # deployment is actually run on. Keeping the two apart is the point of the
    # paragraph above — a default nobody runs is a trap, because any path that
    # rewrites the file drops the session onto it.
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 0.95               # the working value
    # 0, and not by preference: DeepSeek-V4's own guidance (Unsloth's "how to run
    # locally" page for the V4 family) recommends temperature 1.0, top_p 1.0 and
    # min_p 0.01. The last one is not reachable here — any min_p > 0 is rejected
    # outright (400) under MTP speculative decoding on this server, in vLLM's
    # _validate_spec_decode — and MTP is worth more than the recommendation. So
    # this is a thing given up knowingly, recorded here because the reason lived
    # only in somebody's memory for months. Runaway control is done with the
    # penalties below instead. (Turning MTP off would make 0.01 available.)
    min_p: float = 0.0
    # Anti-repetition penalties — DISABLED (experiment result). repetition/
    # frequency penalties treat a tool call's CLOSING tags (structurally
    # identical each call, e.g. </｜DSML｜invoke>) as "repetition" and suppress
    # those tokens, so the model can't close the tag → malformed/unclosed tool
    # calls + premature EOS (the DSV4 tool-call-loss we patched around).
    #
    # This used to add "temp 0.6 (non-greedy) already breaks token-level runaway,
    # so no penalty is needed". The first half is false — a prose loop ran at 0.6
    # — but the conclusion still holds, because the fix is temperature (now 1.0),
    # not these. A penalty that cannot tell a repeated argument from a repeated
    # closing tag buys repetition control at the price of malformed tool calls,
    # which is the worse failure: a loop wastes a turn, a broken tool call
    # corrupts one. Keep all three at their no-op values.
    repetition_penalty: float = 1.0   # 1.0 = off (was 1.05)
    presence_penalty: float = 0.0     # 0.0 = off (was 0.3)
    frequency_penalty: float = 0.0    # 0.0 = off (was 0.3)

    # Server-side loop detection (vLLM RepetitionDetectionParams). Ends the
    # generation once the same token n-gram has repeated, and reports
    # finish_reason='repetition'. Ignored by llama.cpp, which has no such field.
    #
    # This is what a penalty was reached for and could not safely do. A penalty
    # reweights every token, so it cannot tell a repeated argument from a
    # repeated closing tag; detection watches for an exact n-gram cycle and does
    # nothing at all until one appears, so well-formed output is untouched.
    #
    # Values measured against the live server (2026-08-06). A forced 40x
    # repetition was cut at ~140 tokens under every setting tried, while two
    # deliberately repetitive-but-legitimate answers — three classes sharing
    # parameter names, and a 25-row table written in one uniform style — ran to
    # the full budget uncut under all of them. So the choice is about margin,
    # not about which one works:
    #   min_pattern 8  — ignores short runs (newlines, rules, punctuation)
    #   max_pattern 512 — long enough for a whole repeated paragraph
    #   min_count   4  — saying something twice is emphasis, four times is a loop
    # 0 for max_pattern (or min_count < 2) disables it.
    repetition_min_pattern: int = 8
    repetition_max_pattern: int = 512
    repetition_min_count: int = 4
    # Per-model on/off for the loop detector, keyed by served model id.
    # Missing key => the family default in repetition_detection_enabled().
    repetition_detection_per_model: dict = field(default_factory=dict)
    # Thinking (reasoning) effort carried on every request. Empty = "don't send
    # reasoning_effort", which defers to the server default — NOT a guarantee of
    # no thinking. The server default lives in the vLLM launch flags
    # (--default-chat-template-kwargs.thinking), outside this config, and the
    # DSV4-Flash box has been serving thinking=true/effort=low.
    #
    # 'high' rather than empty or 'low', which reads backwards until you look at
    # what the levels actually inject. On the DSV4-Flash 0731 family the table in
    # the model's own encoding_dsv4.py is:
    #
    #     'low':  ''                      <- literally nothing
    #     'high': "...Avoid repeating settled points or narrating redundant
    #              alternatives. HARD LIMIT: finish reasoning within about 1,500
    #              tokens... Never consume the whole output budget with reasoning."
    #
    # So 'low' is not a short leash, it is no leash: reasoning length falls back
    # to whatever the model was trained to prefer, and this family prefers deep
    # deliberation. The anti-repetition wording and the budget only exist at
    # 'high' and above. Measured on one hard prompt (2026-08-06): 'low' spent its
    # entire 6000-token budget reasoning and emitted no answer at all, finishing
    # on `length`; 'high' closed its reasoning and answered in 79s, on `stop`.
    # Raising the level is what puts a ceiling on it. Env: REASONING_EFFORT.
    reasoning_effort: str = 'high'
    # Per-model override keyed by served model id (model_name). Policy
    # 2026-09-03: DSV4-Flash = high always; glm-5.3-flash = high or max.
    reasoning_effort_per_model: dict = field(default_factory=dict)
    # Ask the server's chat template to DROP reasoning from earlier user turns
    # (clear_thinking). In-turn (tool-loop) reasoning is always kept. Prior-turn
    # reasoning made GLM-5.3 cling to old conclusions; DeepSeek-V4 drops it by
    # default and ignores this key. Policy 2026-09-03: default True.
    clear_thinking: bool = True

    # 'auto'   = per-family vendor defaults win (see effective_sampling).
    # 'manual' = send the stored values above verbatim, whatever the model is.
    # Anything else is treated as 'auto'.
    sampling_profile: str = 'auto'

    # Degenerate-run guard: abort a generation once the last N characters of its
    # output collapse to almost no distinct characters. 0 disables it. See
    # _openai_common._DegenerateRunGuard for why this exists and why it cannot
    # fire on real work.
    degenerate_run_chars: int = 2000

    # Stall guards, in seconds. 0 disables either one.
    #
    # first_token: how long a request may produce NOTHING before it is cut.
    # Generous, because prefill legitimately scales with the uncached prompt — a
    # real 1M-context compaction was measured at ~410s before its first token —
    # so this is 1.5x that, not a tight bound.
    #
    # token_gap: how long a stream that HAS started may then go silent. Once
    # tokens are flowing, a two-minute hole is pathological, not slow.
    #
    # Observed 2026-09-10: a request whose prefill completed normally (1,541 new
    # + 84,480 cached, measured) then produced zero tokens for 21 minutes.
    # Nothing noticed. The HTTP timeout only fires when no BYTES arrive at all,
    # so an SSE keep-alive would hold it open indefinitely; the turn ended when
    # the operator pressed ESC.
    first_token_timeout_secs: float = 600.0
    token_gap_timeout_secs: float = 120.0

    def effective_reasoning_effort(self) -> str:
        """The reasoning effort actually sent, per served model.

        ``reasoning_effort_per_model`` is keyed by served model id and wins
        over the global setting, because the right level is a property of the
        model: policy 2026-09-03 is DSV4-Flash high always, glm-5.3-flash high
        or max. An empty result means "omit the key" and hands the decision to
        the server, which is NOT the same as "no thinking".

        Parity with Rust ``Config::effective_reasoning_effort``. The field
        existed here with nothing reading it, so the per-model policy was
        silently ignored and the startup banner -- which does call this --
        died with AttributeError on every launch.
        """
        model = (self.model_name or '').strip()
        if model:
            per_model = self.reasoning_effort_per_model.get(model)
            if per_model is not None:
                return str(per_model).strip().lower()
        return self.reasoning_effort

    def effective_sampling(self) -> dict:
        """The sampling knobs actually sent for one request.

        A key that is absent from the returned dict is NOT sent, which hands
        that decision to the server.

        THREE SOURCES, IN THIS ORDER OF AUTHORITY

        1. The environment wins outright. A knob exported for a run is a
           deliberate, per-run instruction and outranks everything below.
        2. sampling_profile = 'manual' means the operator has taken the wheel:
           config.json is sent verbatim, whatever the served model is.
        3. sampling_profile = 'auto' (the default, and what any unrecognised
           value falls back to) means the client decides, per family.
           config.json is NOT consulted — that is the point of auto. Someone who
           just wants the thing to work should not have to learn which number
           belongs to which model, and a config file that fills itself in cannot
           tell a value someone chose from one it wrote itself.

        THE AUTO TABLE

        GLM: the vendor's published numbers — z.ai's "Migrate to GLM-5.3" guide
        and the zai-org/GLM-5.3-Flash card both give temperature 1.0 with top_p
        0.95, and the guide adds "it is not recommended to adjust both
        simultaneously". No top_k (40 narrows the candidate set on top of top_p,
        which is exactly the "adjust both" the vendor warns against), no
        penalties (the vendor recommends none).

        Everything else, DeepSeek-V4 above all: temperature 0.6 with top_k 40 and
        top_p 0.95. Not a vendor number — it is what this deployment has run on
        for months and what gets handed to anyone setting the client up against
        the same model. A recommendation earns its place by having been run, not
        by being published.

        min_p stays 0 in both, and that is a concession, not a preference:
        DeepSeek-V4's own guidance (Unsloth's run-locally page for the V4 family)
        asks for temperature 1.0, top_p 1.0 and min_p 0.01. min_p > 0 is rejected
        outright (400) under MTP speculative decoding, in vLLM's
        _validate_spec_decode, and MTP is worth more than the recommendation.
        Turning MTP off is what would make 0.01 available. (The DeepSeek
        temperature here is 0.6 rather than the recommended 1.0 for the reason
        above it: measured beats published on the deployment it is about.)

        The penalties stay off in both: a multiplicative repetition penalty
        suppresses the structurally-identical CLOSING tags of tool calls and
        leaves them unclosed. Reach for them through 'manual'.

        (Measured 2026-09-14 on the live GLM deployment, counting answers
        containing U+FFFD: temperature 1.0 broke 7 of 8, 0.6 + top_k 40 broke 1
        of 8. That is a symptom rate, not a cause — the defect was the sampler
        letting a Latin token follow the lead byte of a Korean syllable, fixed
        server-side — so it is not a reason to hold GLM off its vendor number.)
        """
        if str(self.sampling_profile or '').strip().lower() == 'manual':
            out = {
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'min_p': self.min_p,
                'repetition_penalty': self.repetition_penalty,
                'presence_penalty': self.presence_penalty,
                'frequency_penalty': self.frequency_penalty,
            }
        else:
            out = self.recommended_sampling(self.model_name)
        self._apply_sampling_env(out)
        return out

    @staticmethod
    def recommended_sampling(model_name: str) -> dict:
        """The client's own recommendation for a served model — the auto table."""
        if 'glm' in (model_name or '').strip().lower():
            return {'temperature': 1.0, 'top_p': 0.95, 'min_p': 0.0}
        return {'temperature': 0.6, 'top_p': 0.95, 'top_k': 40, 'min_p': 0.0}

    @staticmethod
    def _apply_sampling_env(out: dict) -> None:
        """The environment has the last word, in every profile.

        Both the bare name and a CODEMASON_-prefixed one are read, the prefixed
        one first: TEMPERATURE is conventional enough to collide with something
        else in a shell, and someone who needs to be unambiguous should have a
        way to say so.
        """
        def _env(name):
            for key in (f'CODEMASON_{name}', name):
                raw = os.getenv(key)
                if raw is None or not raw.strip():
                    continue
                try:
                    return float(raw.strip())
                except ValueError:
                    continue
            return None

        for key in ('TEMPERATURE', 'TOP_P', 'MIN_P', 'REPETITION_PENALTY',
                    'PRESENCE_PENALTY', 'FREQUENCY_PENALTY'):
            v = _env(key)
            if v is not None:
                out[key.lower()] = v
        v = _env('TOP_K')
        if v is not None:
            # A negative or zero top_k is how a shell says "do not send it".
            if v >= 1:
                out['top_k'] = int(v)
            else:
                out.pop('top_k', None)

    def repetition_detection_enabled(self) -> bool:
        """Whether to ask the server for loop detection on the CURRENT model.

        The detector was measured into place on 2026-08-06 against the DeepSeek
        substrate, where a forced 40x repetition was cut at ~140 tokens. Its
        stated premise is that "legitimate repetition is not n-gram-identical".

        GLM-5.3-Flash breaks that premise in ordinary work. Enumerating an API
        means holding a command skeleton fixed and varying one field, so six
        consecutive tool calls are byte-identical for a hundred-odd tokens
        apiece while the ANSWERS advance. Observed 2026-09-06: a live probe of a
        router's ubus endpoints was cut mid-enumeration having just established
        that three of them accept unauthenticated calls — real progress, judged
        a loop on shape alone. The recovery nudges then told a model that had
        done nothing wrong to stop repeating itself, six times, and the turn
        ended.

        So the default is per family: on for DeepSeek, off for GLM. An explicit
        entry in repetition_detection_per_model overrides either way.
        """
        m = (self.model_name or '').strip()
        if m and isinstance(self.repetition_detection_per_model, dict):
            v = self.repetition_detection_per_model.get(m)
            if v is None:
                v = self.repetition_detection_per_model.get(m.lower())
            if v is not None:
                return bool(v)
        return 'glm' not in m.lower()
    # Remote notifications (codemason.remote). Seconds a turn must run before it
    # is worth a buzz. 0 = every turn, negative = never.
    #
    # 0, because the channel turned out to be for continuity rather than alerts.
    # The old default of 60 assumed you only care about work you walked away
    # from, and it silently dropped every shorter turn — which meant a session
    # driven from the keyboard reached the phone as nothing at all, and could
    # not be picked up there however well the inbound side worked. Moving
    # between desk and couch mid-session needs the whole conversation to be
    # present on both, not just the slow parts of it. Notification fatigue is a
    # phone-side setting; a missing conversation cannot be fixed phone-side.
    #
    # Also read by remote.hooks.turn_report_delay() to decide when a live
    # progress message may start posting, so this one number governs both
    # halves and they cannot disagree.
    # Env: CODEMASON_NOTIFY_AFTER.
    remote_notify_after: float = 0.0
    # How long the remote channel keeps files that hold your own words —
    # undrained inboxes, and messages parked on a picker waiting to be routed.
    #
    # Long on purpose. Sweeping is a backstop for runs that crashed, not tidying,
    # and deleting content on a timer destroys data nobody asked to lose. It was
    # a day; a message offered with "run it now?" was swept an hour later, which
    # answered a question meant for the sender. Explicit removal is `/remote
    # clean`, which ignores this entirely.
    #
    # In-flight coordination (question claims, forwarded answers) is not covered
    # by this — those are consumed within one prompt round and keeping them
    # longer preserves nothing. Env: CODEMASON_REMOTE_RETENTION_DAYS.
    remote_retention_days: float = 30.0
    # How long a remote permission prompt stays answerable, in seconds.
    #
    # Waiting is not free: the prompt holds the exclusive poll lock the whole
    # time, so every other session goes without remote input until it resolves.
    # And a long window only pays off if the answer actually arrives late — in
    # practice you answer within half an hour or you are not answering at all,
    # so the extra hours buy nothing and cost the lock.
    #
    # Expiry is announced (a fresh push, not a silent message edit), so a short
    # window never leaves you wondering whether it is still waiting.
    # 0 or negative disables remote prompting. Env: CODEMASON_ASK_TIMEOUT.
    remote_ask_timeout: float = 1800.0
    # Deliver messages that arrived while no session was listening?
    #
    # Off. Telegram holds updates ~24h, so starting a session used to replay
    # everything sent since the last one ended — including a queued `/clear`,
    # which wiped a conversation the moment the session came up. Nothing written
    # for a session that no longer exists should run in a new one, and an
    # instruction worth repeating takes a second to repeat.
    # Env: CODEMASON_REPLAY_BACKLOG=1 to restore.
    remote_replay_backlog: bool = False
    # Shade the echoed prompt so it reads apart from the answer. Once a
    # conversation is long enough, what you asked and what came back run together
    # into one column of text; a background gives the eye the boundary. The colour
    # follows `theme`, and the foreground is set alongside it so contrast holds on
    # any terminal. False leaves the terminal's own colours alone.
    input_highlight: bool = True
    # That background/foreground. Empty means "pick by theme". Terminal
    # backgrounds vary too much for one grey to be right everywhere, so it is a
    # value rather than a constant.
    input_highlight_bg: str = ''
    input_highlight_fg: str = ''
    # Answers longer than this (characters) are attached as a file instead of
    # sent inline. Telegram caps a message at 4096, so long answers used to arrive
    # with the conclusion cut off — which for "analyse this project" is the part
    # that mattered. The chat keeps an excerpt; the file carries the whole thing,
    # and a file is what scrolls, searches and saves on a phone. 0 = always
    # inline, and therefore truncated.
    remote_file_threshold: int = 3000
    # Answers longer than this arrive in a quote box, which Telegram draws in a
    # smaller face. Boxed, not folded: an expandable quote closed again on its
    # own and cost a tap on every look. 0 = never box.
    remote_collapse_over: int = 1200
    # Extension for the attachment. The content is Markdown either way: .md
    # renders if a viewer app is installed, while .txt is likelier to open in
    # whatever the phone already has.
    remote_file_ext: str = '.md'
    # How much of the answer stays in the chat when the rest goes to a file. A
    # conclusion usually lives in the opening, so this much is generally enough to
    # read without opening anything — and it arrives folded, so length costs
    # nothing on screen.
    remote_excerpt: int = 3000

    # The whole phone channel, on or off. Off stops alerts, the spinner,
    # approvals, questions and inbound messages together — there is one gate
    # (remote.notify.off_reason) precisely because a half-off channel is worse
    # than either state. With no credentials it is off regardless, and /remote on
    # refuses rather than claiming otherwise. Toggle: /remote on|off. This run
    # only: --no-remote. Env: CODEMASON_REMOTE=0
    remote_enabled: bool = True

    # Read another product's config directories (~/.claude, <cwd>/.claude) for
    # skills and sub-agent definitions. OFF by default: this is our program, and
    # another tool's layout is an opt-in convenience for someone migrating, not
    # something to inherit silently. Env: CODEMASON_IMPORT_CLAUDE_DIRS.
    import_claude_dirs: bool = False
    # Widest table, in terminal cells, that is still worth drawing as a grid on
    # the phone. Telegram has no tables, so the best available is an aligned
    # monospace block — and alignment is only a gain while the line fits the
    # screen. Past that the client wraps it (columns become mush) or scrolls it
    # sideways (the values you are comparing go off-screen), and a wider table
    # reads better re-laid one row at a time. Depends on your device and font
    # size, so it is a setting: raise it on a tablet, lower it if you run large
    # text. 0 or negative always draws the grid. Env: CODEMASON_TABLE_WIDTH.
    remote_table_width: int = 36

    # UI settings
    theme: str = 'dark'
    verbose: bool = False
    # -rt/--realtime: stream the reply to the screen as it arrives. On by
    # default — a local model's first token comes back fast but the whole answer
    # does not, and waiting at a blank screen for it reads as a hang. --no-rt
    # turns it off for a console that cannot repaint.
    realtime: bool = True
    #                                   (DSML/think/special-token-safe). Off = buffer
    #                                   the whole turn then render markdown at once.

    # Permission settings
    dangerously_skip_permissions: bool = False

    # Feature flags
    auto_compact: bool = True
    # Spinner label while a tool runs. Default: the tool's name ("Bash…").
    #
    # Upstream picks a decorative verb at random per turn ("Bloviating",
    # "Boondoggling") on the reasoning that the tool is already named on the
    # line above. In a long autonomous run that line has scrolled away, and then
    # the only thing on screen is a word chosen at random — which reads as the
    # model doing something arbitrary rather than a tool waiting. Naming the
    # tool costs nothing and answers the question being asked. Set True for
    # upstream's decorative behaviour. Env: CODEMASON_SPINNER_RANDOM_VERB.
    spinner_random_verb: bool = False
    # Absolute ceiling for auto-compaction, in tokens; 0 = window-derived only.
    #
    # The window-derived threshold lets ONE conversation grow to nearly the whole
    # model window. Against a server whose KV pool holds ~1.6 windows (measured
    # 2026-09-07: 847,533 tokens for a 524,288 window) that leaves no room for a
    # sub-agent: the moment one starts working, the parent's blocks are evicted,
    # and when the sub-agent returns the parent re-prefills from zero — 200s at
    # 267k. Two tenants can only coexist if 2 * threshold < pool, and only the
    # operator knows the pool.
    auto_compact_threshold_tokens: int = 0
    # How many sub-agents (Agent tool, forked skills) may run at once, across
    # every codemason process on this machine — each is a separate cache tenant
    # competing with the parent for the same KV pool.
    #
    # 3 on this line, where the tools are concurrency-safe and a spawn can
    # actually overlap. Measured 2026-10-10 against the 2x-Spark lane at KV 12GB
    # (1,356,053 tokens): a 400,708-token parent still answered from cache at
    # 98.9% after three 133k sub-agents ran to completion, and three agents
    # finished the same work in 76.0s where they took 121.5s one at a time.
    # Against a smaller pool this is the number to lower — subagent_gate's
    # docstring says how to read it.
    max_concurrent_subagents: int = 3
    # Context ceiling for SUB-AGENTS only, in tokens. 0 (the default) does NOT
    # mean unlimited: it derives one from the parent's own threshold
    # (SUBAGENT_THRESHOLD_FRACTION in services.compaction), so an untuned
    # install still keeps the sub-agent small enough not to evict the parent.
    # Set a positive value to override that derivation — do that only from a
    # measurement on the server you actually serve against, never from
    # arithmetic over its advertised numbers (see below).
    #
    # The parent must be free to use the whole window — a 512K conversation is
    # the point of a 512K model. The sub-agent is what has to fit in what is
    # left. Blocks, not tokens, are the currency: a conversation of L tokens
    # holds L/4608 attention blocks plus L/VLLM_PREFIX_CACHE_RETENTION_INTERVAL
    # Mamba checkpoints, out of a 215-block pool on this deployment. Mamba
    # state does not accumulate while a request runs; each superseded block is
    # freed as the state advances (mamba_cache_mode "align"), so only the
    # retained checkpoints scale with length.
    #
    # Measured 2026-09-13 at retention 18432 with the parent parked at 441,165
    # tokens: a concurrent conversation of 270,000 left the parent fully
    # cached in 10 of 10 trials; 280,000 cost it one attention block every
    # time; 290,000 survived once and wiped it twice; 296,945 and up wiped it
    # outright with a ~360s re-prefill. Eviction near the cliff is
    # all-or-nothing — a Mamba hit needs the single state block ending on the
    # boundary, so one lost checkpoint zeroes the whole hit — and the boundary
    # is probabilistic, so a value is only safe when it survives repeatedly,
    # never on one green run.
    #
    # Note what this ceiling has to cover: a sub-agent's peak footprint is this
    # value PLUS the output that turn generates, so budget
    # ceiling + subagent_max_tokens against the measured limit, not the ceiling
    # alone. This deployment runs 230,000 + 32,768 = 262,768 against 270,000.
    #
    # Do NOT derive a value from the server's "GPU KV cache size: N tokens"
    # log line. That N is int(max_concurrency * max_model_len) where
    # max_concurrency is pool_blocks / blocks_for_one_max_len_request; it
    # describes one request's worst case, not how much conversation can sit in
    # the cache at once, and reading it as the latter is what produced the
    # wrong numbers this comment replaces.
    subagent_context_ceiling_tokens: int = 0
    # Output budget for SUB-AGENT requests, in tokens; 0 (the default) = same as
    # the parent's max_tokens.
    #
    # Worth setting below max_tokens because the two reserves are paid for
    # differently. A sub-agent's peak KV footprint is its compaction ceiling
    # PLUS whatever that turn generates, so the full max_tokens has to be held
    # back from subagent_context_ceiling_tokens even though it is almost never
    # spent: measured over this deployment's turn log the median output is 81
    # tokens and the largest 1,860. What justifies reserving it at all is the
    # tail, not the mean — 2026-09-10 the model collapsed onto one token and
    # rode it to max_tokens twice, 53,859 consecutive '!' for 63 minutes a turn.
    #
    # So cap the sub-agent instead of hoping: a smaller budget shrinks the
    # reserve one-for-one and buys the same number of tokens back as working
    # context, which is what a sub-agent that reads whole sources actually
    # needs. Nothing is truncated by the cap — hitting it yields
    # stop_reason=max_tokens and the recovery path resumes the generation — but
    # leave headroom for reasoning, which is billed as output and runs long at
    # high effort.
    subagent_max_tokens: int = 0

    # Legacy encoding to run a SECOND ripgrep pass in, or empty to skip it.
    #
    # ripgrep's automatic detection only looks at a BOM, so a CP949 file is
    # searched as if it were UTF-8 and Hangul in it matches nothing. Passing
    # --encoding is not a fix on its own either — measured, `-E euc-kr` finds the
    # CP949 file and MISSES the UTF-8 one, trading one blind spot for another.
    # Running both passes and merging is what covers a mixed tree, and it is
    # nearly free: measured 4ms -> 8ms over a whole repository, against a model
    # turn that costs seconds.
    #
    # On by default for that reason — a setting nobody knows about would leave
    # the feature unreachable where users meet it. Empty turns the second pass
    # off. Env: CODEMASON_GREP_LEGACY_ENCODING.
    grep_legacy_encoding: str = 'euc-kr'

    # Wall-clock cap on ONE background shell task, in seconds. 0 disables it.
    #
    # A background task has no other brake. The foreground Bash tool caps at 10
    # minutes and then ADOPTS the still-running process as a background task, so
    # the command the model expected to be quick is exactly the one that ends up
    # here unbounded. The disk cap above already stops it filling the disk; this
    # stops it holding a core forever.
    #
    # Six hours, not something tight: this is a runaway guard, not a scheduling
    # policy. A real build, test suite or dev server has to finish comfortably
    # inside it, because killing legitimate work is a worse failure than the
    # runaway it prevents. Raise it, or set 0, for a job that genuinely runs
    # longer. Env: CODEMASON_BACKGROUND_TASK_TIMEOUT.
    # Training cutoff of the served model, e.g. "March 2025". Empty (default)
    # leaves the "Assistant knowledge cutoff is ..." line out of the environment
    # block entirely, which is what happened before this existed: the cutoff was
    # looked up in a hardcoded table of another product's model ids, so it never
    # matched anything we serve and the line was never emitted. Fill it in for
    # the model you actually run. Env: CODEMASON_MODEL_KNOWLEDGE_CUTOFF.
    model_knowledge_cutoff: str = ''

    background_task_timeout: float = 21600.0

    # Engine-side runaway guard (QueryParams.maxTurns): cap on the number of
    # recursive tool turns in a single user prompt. Env: CODEMASON_MAX_TURNS;
    # None disables it.
    #
    # This is a BACKSTOP, not the main defence. Two cheaper guards run first and
    # catch the common loops: an identical (tool, args) call is refused after
    # REPEATED_TOOL_CALL_LIMIT, and Layer-2 repetition recovery rewrites a turn
    # that keeps repeating itself. What neither sees is a loop whose arguments
    # change every time — reading one more file, forever — and that is what this
    # stops.
    #
    # OFF by default, and that is the considered answer, not an oversight.
    #
    # I set it to 200 during the audit, reasoning that -dsp plus keep-going left
    # nothing between a session and an unbounded loop. That reasoning ignored
    # what -dsp is FOR here: the user starts a long job, walks away for hours,
    # and comes back to the result. A cap does not prompt — it silently ENDS the
    # turn mid-task. So the failure it creates is exactly the one that hurts most
    # in the only mode where it would ever fire: hours later, the work is half
    # done, and the transcript just says it stopped.
    #
    # The loops it was meant to catch are already caught, and more cheaply: an
    # identical (tool, args) call is refused after the repeated-call limit, and
    # Layer-2 repetition recovery rewrites a turn that keeps repeating itself.
    # What a turn cap adds over those is a loop whose arguments change every time
    # — and paying for that with "your six-hour run stopped at turn 200" is not a
    # trade worth making.
    #
    # Set it to a number if you want the cap for a specific run. A null on disk
    # means unset, which is off.
    max_turns: Optional[int] = None

    # "keep-going" autonomy: when on, the installed Stop hook nudges the model
    # to keep working past a natural turn-end until the task is done (see
    # ~/.codemason/hooks/keep_going.py). This is the PERSISTENT default; a
    # session can override it with --keep-going / --no-keep-going or the REPL
    # `keepgoing on|off` command (both via the CODEMASON_KEEP_GOING env var).
    # Default OFF: the nudge existed to stop a *lightweight* model from quitting
    # mid-task; on the current capable backend (DSV4) it's unneeded. Re-enable
    # per-session with `keepgoing on` or config.json if ever wanted.
    keep_going: bool = False

    # Auto-run the active goal's completion gates when a turn ends (see
    # query.py). A failing gate feeds its output back and the turn continues, so
    # the agent iterates until the gates pass instead of stopping at a
    # self-declared "done". Only fires when a goal HAS gates (opt-in). Toggle with
    # `/goal gate auto on|off`.
    goal_gate_autorun: bool = True

    # Preferred response language (settings.language). When set (e.g.
    # "Korean", "japanese"), the system prompt gets a "# Language" section telling
    # the model to respond in it. None = model default (no forced language).
    language: Optional[str] = None
    # The program's own language — menus, status, alerts. Empty follows
    # `language`, then the OS locale, then English. It is separate from `language`
    # because the two can legitimately disagree: answer me in Korean but show me
    # the menus in English, or the reverse. Naming a language nobody has
    # translated is safe; it falls back to English string by string.
    ui_language: Optional[str] = None

    # WebSearch: self-hosted SearXNG as the 0-priority backend (tools/
    # web_search_tool.py). When reachable, its JSON API replaces the client-side
    # ddgs scraping for that query, sidestepping the per-IP anti-scraping limits;
    # when absent/down/JSON-disabled, WebSearch falls back to ddgs. Empty URL
    # disables SearXNG (ddgs-only). Env overrides: CODEMASON_SEARXNG_URL /
    # CODEMASON_SEARXNG_TIMEOUT / CODEMASON_SEARXNG_LANG. On the same host as the
    # instance, http://127.0.0.1:8888 skips the reverse proxy and is fastest.
    searxng_url: str = 'https://searxng.systemboan.net'
    searxng_timeout: float = 8.0      # per-request timeout (seconds)
    searxng_lang: str = 'auto'        # SearXNG search language ('auto' = detect)
    # API-key auth for the PUBLICLY-exposed instance. SearXNG has no native auth, so
    # the reverse proxy (nginx / NPM) gates access on this shared-secret header
    # (X-API-Key) and 403s anything without it; codemason sends it on every SearXNG
    # request so it passes that gate. NOT hardcoded (2026-09-14): a shared secret
    # in source is one screen recording away from being public, and rotating it
    # then means a code change on every machine. Set it in
    # ~/.codemason/config.json next to the nginx/NPM rule's value.
    # Empty = send no header (LAN/VPN-only instances).
    # Env override: CODEMASON_SEARXNG_API_KEY.
    searxng_api_key: str = ''

    # WebSearch rate-limit posture (ddgs fallback path). Every knob keeps a single
    # logical WebSearch from turning into a request BURST that trips per-IP
    # anti-scraping limits. Defaults are tuned for a busy coding agent; raise the
    # throttle / lower expansions on a machine that still gets limited. Env
    # overrides: CODEMASON_WEBSEARCH_* (BACKENDS/MAX_EXPANSIONS/THROTTLE_MIN/
    # THROTTLE_MAX/RETRY_ATTEMPTS/BACKOFF_BASE/BACKOFF_CAP/CACHE_TTL/CACHE_DIR).
    websearch_backends: str = 'duckduckgo, brave, mojeek, google'  # ddgs merged engines
    websearch_max_expansions: int = 1        # extra queries the planner may add (0 = verbatim only)
    websearch_throttle_min: float = 0.4      # inter-query ddgs delay, jitter low (seconds)
    websearch_throttle_max: float = 1.2      # inter-query ddgs delay, jitter high (0 disables)
    websearch_retry_attempts: int = 3        # per-query attempts (exponential backoff)
    websearch_backoff_base: float = 0.8      # backoff base (seconds)
    websearch_backoff_cap: float = 8.0       # backoff ceiling (seconds)
    websearch_cache_ttl: int = 6 * 60 * 60   # result cache lifetime seconds (0 disables)
    websearch_cache_dir: str = ''            # cache root ('' = <CONFIG_DIR>/cache/websearch)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], warnings: Optional[list] = None) -> 'Config':
        # Migrate legacy config.json keys, then filter to only known fields.
        # An on-disk config written before the llamacpp_* -> model_* rename still
        # carries the old keys; without this migration the unknown-key filter
        # below would drop a user's saved endpoint/model/key and silently fall
        # back to defaults. get_config() then rewrites the file forward with the
        # new keys, so this runs at most once per machine.
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        migrated = dict(data) if isinstance(data, dict) else {}
        for _old, _new in _LEGACY_CONFIG_KEYS.items():
            if _old in migrated and _new not in migrated:
                migrated[_new] = migrated.pop(_old)
        defaults = cls()
        filtered_data = {}
        for k, v in migrated.items():
            if k not in known_fields:
                continue
            ok, value = _coerce_field(k, v, getattr(defaults, k))
            if ok:
                filtered_data[k] = value
            elif warnings is not None:
                warnings.append((k, v, getattr(defaults, k)))
        return cls(**filtered_data)


# Global config instance
_config: Optional[Config] = None

# Fields the environment overrode, holding the value the file had before it —
# what save_config() writes back so env stays runtime-only. See get_config().
_env_shadow: Dict[str, Any] = {}
# The config as it was on disk when this process loaded it. Used as the base
# of a 3-way merge in save_config so a concurrent session's settings are not
# clobbered — see the comment there.
_file_baseline: Dict[str, Any] = {}
# config.json exists but could not be parsed. This session runs on defaults and
# must not write: the next save would replace the user's file — API key, base
# URL, language — with those defaults. Cleared only by reset_config().
_load_failed: bool = False
_load_failed_reported: bool = False


def ensure_config_dir() -> None:
    """Ensure config directory exists"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get the current configuration"""
    global _config

    if _config is not None:
        return _config

    ensure_config_dir()

    global _load_failed
    needs_write = False
    bad_values: list = []
    migrated_notes: list = []
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _config = Config.from_dict(data, bad_values)
            migrated_notes = _apply_migrations(_config, data)
            # A changed value, or a version stamp the file does not carry yet.
            if migrated_notes or (isinstance(data, dict)
                                  and data.get('config_version') != _config.config_version):
                needs_write = True
            # Backfill: if the on-disk file predates any config field (e.g. the
            # searxng_* settings added later), write the full set back so the
            # missing keys are materialized with their defaults. User-set values
            # are preserved (from_dict kept them); only absent keys get defaults.
            # Idempotent — once every key is present this never rewrites.
            known = {fld.name for fld in Config.__dataclass_fields__.values()}
            if not isinstance(data, dict) or (known - set(data.keys())):
                needs_write = True
        except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError) as e:
            # Don't silently reset a corrupt config. It used to go only to
            # error.log, so a stray comma meant running on defaults — no API
            # key, someone else's server — with nothing on screen, and the next
            # /dsp or /effort then saved those defaults over the user's file.
            from .log import log_error
            log_error(e, "Failed to load config; using defaults")
            _config = Config()
            _load_failed = True
            _say("[codemason] Could not read {path}: {error}", path=str(CONFIG_FILE), error=str(e))
            _say("[codemason] Running on defaults. Settings changed in this session will not be saved "
                 "until the file is fixed (it is left untouched).")
    else:
        _config = Config()
        # First run: materialize config.json so the user has a real file to edit
        # (and can see where it lives) instead of silently running on in-memory
        # defaults with no file present. ensure_config_dir() already made the dir.
        needs_write = True

    if needs_write:
        # Persist BEFORE env overrides so env stays runtime-only (never baked into
        # the file) — matches how the model_* env overrides have always behaved.
        #
        # Through the same merge + atomic write save_config() uses, not a plain
        # open('w'). This path runs on every version upgrade (it exists to
        # materialize keys a newer version added), and a plain dump of our own
        # view DROPS any key we do not know — which is exactly the loss the merge
        # was added to prevent, on the one path most likely to meet a file
        # written by a different version. It was also non-atomic (a crash
        # mid-write left an unparseable config) and created the file 0644, in a
        # file that holds the API key.
        #
        # The baseline is empty here on purpose: this is a first materialization
        # or a backfill, so every key counts as ours to write. Unknown keys
        # already on disk are still preserved by the merge.
        try:
            _write_config_atomic(_config.to_dict(), baseline={})
        except OSError as e:
            from .log import log_error
            log_error(e, "Failed to write config")

    # What the file said, captured before the env overlay goes on. save_config()
    # writes these back over any field env changed, so a variable set for one
    # run stays runtime-only instead of being baked in by the next save.
    #
    # The write above is already ordered to keep env out of the file, but that
    # only covers the write that happens right here. `_config` is a singleton and
    # the overrides below mutate it in place, so every LATER save — /dsp on,
    # the model auto-heal in tokens.py — was dumping env-derived values into
    # config.json as if the user had chosen them. Exporting a base URL for one
    # session and toggling /dsp was enough to make that URL permanent, which on a
    # config carried between machines is the difference between "runs anywhere"
    # and "silently points at the box it was last used on".
    global _env_shadow
    _env_shadow = {}
    _pre_env = _config.to_dict()
    global _file_baseline
    _file_baseline = dict(_pre_env)

    def _warn_plaintext_endpoint(url: str, label: str) -> None:
        """Warn once when a bearer token would cross a network in the clear.

        The Authorization header is attached OUTSIDE the body-encryption branch,
        so an http:// endpoint sends the API key — and, without body encryption,
        the whole conversation, file contents and generated code — as plaintext.
        No scheme check existed anywhere for this value.

        Loopback is exempt and silent: a local vLLM on 127.0.0.1 is a normal
        setup and nothing leaves the machine. This warns rather than refuses,
        because a gateway reachable only over a trusted link is a deliberate
        deployment, and breaking a working one is not this function's call.
        """
        try:
            from urllib.parse import urlparse
            u = urlparse((url or '').strip())
            if u.scheme != 'http':
                return
            host = (u.hostname or '').lower()
            if host in ('localhost', '127.0.0.1', '::1') or host.startswith('127.'):
                return
            key = f'_warned_plaintext_{label}'
            if getattr(_warn_plaintext_endpoint, key, False):
                return
            setattr(_warn_plaintext_endpoint, key, True)
            # A diagnostic, not output: stdout is the answer channel, and in
            # `-p --output-format json` a line there makes the JSON unparseable.
            # (patch_stdout covers stderr too, so the REPL prompt still repaints.)
            import sys
            print(
                f"  Warning: {label} is http:// ({host}). The API key is sent as a "
                f"header, which body encryption does not cover, so it crosses the "
                f"network in the clear. Use https:// unless the link is trusted.",
                file=sys.stderr,
            )
        except Exception:
            pass

    # Override with environment variables
    _config.model_base_url = _env_first(_config.model_base_url, 'CODEMASON_MODEL_BASE_URL', 'LLAMACPP_BASE_URL')
    _config.model_name = _env_first(_config.model_name, 'CODEMASON_MODEL_NAME', 'LLAMACPP_MODEL')
    _config.model_api_key = _env_first(_config.model_api_key, 'CODEMASON_MODEL_API_KEY', 'LLAMACPP_API_KEY')
    _config.model_aes_key = _env_first(_config.model_aes_key, 'CODEMASON_MODEL_AES_KEY', 'LLAMACPP_AES_KEY')
    _warn_plaintext_endpoint(_config.model_base_url, 'model_base_url')
    _v = (os.getenv('CODEMASON_IMPORT_CLAUDE_DIRS') or '').strip().lower()
    if _v:
        _config.import_claude_dirs = _v in ('1', 'true', 'yes', 'on')
    _config.backend = os.getenv('CODEMASON_BACKEND', _config.backend)
    _config.model_dir = os.getenv('CODEMASON_MODEL_DIR', _config.model_dir)
    if os.getenv('CODEMASON_MAX_TURNS'):
        try:
            _config.max_turns = int(os.getenv('CODEMASON_MAX_TURNS'))
        except ValueError:
            pass
    if os.getenv('MAX_TOKENS'):
        _config.max_tokens = int(os.getenv('MAX_TOKENS'))
    if os.getenv('TEMPERATURE'):
        _config.temperature = float(os.getenv('TEMPERATURE'))
    if os.getenv('REASONING_EFFORT') is not None:
        _re = os.getenv('REASONING_EFFORT').strip().lower()
        # '', 'off', 'none', 'false' all disable; otherwise must be a valid level.
        _config.reasoning_effort = _re if _re in ('low', 'medium', 'high', 'max') else ''
    if os.getenv('CODEMASON_REPLAY_BACKLOG') is not None:
        _config.remote_replay_backlog = os.getenv(
            'CODEMASON_REPLAY_BACKLOG', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if os.getenv('CODEMASON_ASK_TIMEOUT'):
        try:
            _config.remote_ask_timeout = float(os.getenv('CODEMASON_ASK_TIMEOUT'))
        except ValueError:
            pass
    if os.getenv('CODEMASON_NOTIFY_AFTER'):
        try:
            _config.remote_notify_after = float(os.getenv('CODEMASON_NOTIFY_AFTER'))
        except ValueError:
            pass
    _config.grep_legacy_encoding = os.getenv(
        'CODEMASON_GREP_LEGACY_ENCODING', _config.grep_legacy_encoding)
    _config.model_knowledge_cutoff = os.getenv(
        'CODEMASON_MODEL_KNOWLEDGE_CUTOFF', _config.model_knowledge_cutoff)
    if os.getenv('CODEMASON_BACKGROUND_TASK_TIMEOUT'):
        try:
            _config.background_task_timeout = float(
                os.getenv('CODEMASON_BACKGROUND_TASK_TIMEOUT'))
        except ValueError:
            pass
    if os.getenv('CODEMASON_REMOTE_RETENTION_DAYS'):
        try:
            _config.remote_retention_days = float(
                os.getenv('CODEMASON_REMOTE_RETENTION_DAYS'))
        except ValueError:
            pass
    if os.getenv('CODEMASON_TABLE_WIDTH'):
        try:
            _config.remote_table_width = int(os.getenv('CODEMASON_TABLE_WIDTH'))
        except ValueError:
            pass
    _config.searxng_url = os.getenv('CODEMASON_SEARXNG_URL', _config.searxng_url)
    _config.searxng_lang = os.getenv('CODEMASON_SEARXNG_LANG', _config.searxng_lang)
    _config.searxng_api_key = os.getenv('CODEMASON_SEARXNG_API_KEY', _config.searxng_api_key)
    if os.getenv('CODEMASON_SEARXNG_TIMEOUT'):
        try:
            _config.searxng_timeout = float(os.getenv('CODEMASON_SEARXNG_TIMEOUT'))
        except ValueError:
            pass
    _config.websearch_backends = os.getenv('CODEMASON_WEBSEARCH_BACKENDS', _config.websearch_backends)
    _config.websearch_cache_dir = os.getenv('CODEMASON_WEBSEARCH_CACHE_DIR', _config.websearch_cache_dir)
    for _env, _attr, _cast in (
        ('CODEMASON_WEBSEARCH_MAX_EXPANSIONS', 'websearch_max_expansions', int),
        ('CODEMASON_WEBSEARCH_THROTTLE_MIN', 'websearch_throttle_min', float),
        ('CODEMASON_WEBSEARCH_THROTTLE_MAX', 'websearch_throttle_max', float),
        ('CODEMASON_WEBSEARCH_RETRY_ATTEMPTS', 'websearch_retry_attempts', int),
        ('CODEMASON_WEBSEARCH_BACKOFF_BASE', 'websearch_backoff_base', float),
        ('CODEMASON_WEBSEARCH_BACKOFF_CAP', 'websearch_backoff_cap', float),
        ('CODEMASON_WEBSEARCH_CACHE_TTL', 'websearch_cache_ttl', int),
    ):
        _v = os.getenv(_env)
        if _v is not None:
            try:
                setattr(_config, _attr, _cast(_v))
            except ValueError:
                pass

    _env_shadow = {k: v for k, v in _pre_env.items()
                   if getattr(_config, k, v) != v}

    for key, value, default in bad_values:
        _say("[codemason] config.json: \"{key}\" = {value} is not a valid value for this setting; "
             "using the default ({default}).",
             key=key, value=json.dumps(value, ensure_ascii=False),
             default=json.dumps(default, ensure_ascii=False))
    for key, old, new in migrated_notes:
        _say("[codemason] config.json updated: \"{key}\" {old} -> {new} (an old default that has since been fixed).",
             key=key, old=json.dumps(old, ensure_ascii=False), new=json.dumps(new, ensure_ascii=False))
    return _config


def _apply_migrations(cfg: Config, data: Any) -> list:
    """Move a file written by an older version past defaults that were fixed.

    Returns (key, old, new) for every value changed. Only a value that still
    equals what the old version wrote is replaced; the version is stamped to
    CONFIG_VERSION (never lowered, so a newer build's file keeps its number).
    """
    if not isinstance(data, dict):
        return []
    raw = data.get('config_version')
    file_version = raw if isinstance(raw, int) and not isinstance(raw, bool) else 1
    notes = []
    for version in sorted(_MIGRATIONS):
        if file_version >= version:
            continue
        for key, olds, new in _MIGRATIONS[version]:
            if key in data and getattr(cfg, key) in olds:
                notes.append((key, getattr(cfg, key), new))
                setattr(cfg, key, new)
    cfg.config_version = max(file_version, CONFIG_VERSION)
    return notes


def _write_config_atomic(data: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    """Merge `data` over the file as it is now, then write it atomically at 0600.

    Shared by save_config() and the first-run/backfill path so the two cannot
    diverge in how they treat unknown keys, durability, or file mode — they did,
    and the backfill was the weaker one.
    """
    from .file import FileLock, atomic_write_bytes
    # The merge below reads the file and the write replaces it. Without the lock
    # those two steps straddle another process's identical pair, and the later
    # rename wins outright: measured 150/150 edits lost with two processes saving
    # one setting each at the same moment. The lock makes read-merge-write one
    # step against every other codemason process, Python or Rust. Not getting it
    # (timeout) degrades to exactly the old behaviour — see utils/file.
    with FileLock(config_lock_path()):
        merged = _merge_over_disk(data, baseline)
        atomic_write_bytes(
            str(CONFIG_FILE),
            json.dumps(merged, indent=2, ensure_ascii=False).encode('utf-8'),
        )


def _merge_over_disk(mine: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay only the keys this process changed onto the current file.

    `baseline` is what the file said when we loaded it (empty for the
    first-run backfill, which has none). A key whose value still equals the
    baseline was not touched here, so the on-disk value — which may be newer,
    from another session — wins. A key that differs is our edit and is written.
    Unknown keys already on disk are kept untouched.

    Best effort: if the file cannot be read we fall back to writing our own
    view, which is exactly the old behaviour and no worse.

    Call it under the config lock (see _write_config_atomic): on its own it
    narrows the window for a lost update but does not close it.
    """
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            on_disk = json.load(f)
        if not isinstance(on_disk, dict):
            return mine
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return mine
    return _merge_body(mine, baseline, on_disk)


def _merge_body(mine: Dict[str, Any], baseline: Dict[str, Any],
                on_disk: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(on_disk)
    for key, value in mine.items():
        if key not in baseline or baseline[key] != value:
            merged[key] = value      # our edit
        elif key not in merged:
            merged[key] = value      # absent on disk: materialize the default
    return merged


def save_config(config: Optional[Config] = None) -> bool:
    """Save configuration to file.

    Returns whether the file was written — False when config.json could not be
    read at startup (see _load_failed) or the write failed — so a command can
    say "saved" only when it was.
    """
    global _config

    if config is not None:
        _config = config

    if _config is None:
        return False

    if _load_failed:
        # See _load_failed: writing now would replace the user's unreadable
        # file with defaults. Say so once instead of failing silently.
        global _load_failed_reported
        if not _load_failed_reported:
            _load_failed_reported = True
            _say("[codemason] Not saved: {path} could not be read when this session started. "
                 "Fix the file and restart; your change applies to this session only.",
                 path=str(CONFIG_FILE))
        return False

    ensure_config_dir()

    try:
        # ensure_ascii=False + utf-8 to match get_config()'s writer — otherwise a
        # re-save escaped non-ASCII values (e.g. language: "한국어") to \uXXXX and
        # churned the file / broke the round-trip the migration path assumes.
        data = _config.to_dict()
        # Put back what the file said for anything the environment overrode,
        # so an env var set for one run never becomes a stored setting.
        data.update(_env_shadow)
        # 3-way merge against the file as it is RIGHT NOW, with the copy we
        # loaded at startup as the base. Writing `data` wholesale is a
        # lost update: this process's in-memory copy was read at startup, so
        # another session that changed a DIFFERENT setting in the meantime had
        # its change silently reverted to whatever we happened to load. With
        # several sessions open at once (which is normal here) settings quietly
        # rolled back for no visible reason.
        #
        # Only the keys THIS process actually changed are ours to write; every
        # other key takes the on-disk value. Keys we have never heard of — a
        # newer version's — are preserved rather than dropped.
        #
        # Both of those — the merge and the atomic replace — belong to
        # _write_config_atomic, which also holds the lock that makes the pair
        # one step against the other processes. This path used to inline its own
        # copy of them and so missed anything added there: that is how it ended
        # up as the only writer with no lock.
        _write_config_atomic(data, _file_baseline)
        written = True
    except Exception as e:
        from .log import log_error
        log_error(e, "Failed to save config")
        written = False

    # The UI language is resolved once and cached; a saved config that changes it
    # would otherwise not take effect until the next run.
    try:
        from . import i18n
        i18n.reset()
    except Exception:
        pass
    return written


def saved_label(saved: bool) -> str:
    """The word a command puts after a setting it just changed: "saved", or why
    not. Pairs with save_config()'s return value."""
    from .i18n import gettext as _t
    return _t('saved') if saved else _t('not saved: config.json could not be read')


def set_runtime_override(field: str, value: Any) -> None:
    """Apply a setting for this run only, without letting it reach the file.

    Command-line flags are the same kind of input as an environment variable —
    true for one invocation, not a stored preference — and they reach the config
    the same way, by mutating the singleton. So they need the same shadow: `-rt`
    passed once must not become the setting for every later run just because
    something saved in between.

    Use for flags. A setting the user changed on purpose (`/dsp on`) is a real
    preference and goes through clear_runtime_override() + save_config().
    """
    cfg = get_config()
    if field not in Config.__dataclass_fields__:
        return
    _env_shadow.setdefault(field, getattr(cfg, field))
    setattr(cfg, field, value)


def clear_runtime_override(field: str) -> None:
    """Let a field be persisted again, for a value the user deliberately set."""
    _env_shadow.pop(field, None)


def reset_config() -> None:
    """Reset configuration to defaults"""
    global _config, _env_shadow, _load_failed
    _env_shadow = {}
    _config = Config()
    # An explicit reset is the one write that may replace an unreadable file.
    _load_failed = False
    save_config()
