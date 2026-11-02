"""Behaviours the Rust client pins with unit tests, asserted here too.

The two clients are meant to behave the same; the Rust side has 426 tests and
this side had three. Each case below is a port of a named Rust test, so a
divergence shows up as a failure here instead of as a difference in the field.
"""
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if sys.platform == 'win32':
    # Our own case names carry em dashes and Korean, and the console
    # encoder here is cp949: the first such line would raise
    # UnicodeEncodeError inside the reporter and stop the run mid-suite,
    # blaming whatever case it was printing. Pinned inline, importing
    # nothing -- several of these files must set CODEMASON_CONFIG_DIR
    # before anything under codemason/ is imported.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

from codemason.services._openai_common import (      # noqa: E402
    SINGLE_CHAR_RUN_CHARS,
    _DegenerateRunGuard,
    _SILENT_REASONING_TOKENS,
    clean_special_tokens,
    recover_leaked_tool_calls,
    truncate_tool_result,
)
from codemason.utils.config import (                 # noqa: E402
    Config,
    LOCKS_DIR,
    config_lock_path,
    endpoint_tag,
    lock_path,
)
from codemason.utils.session_storage import (            # noqa: E402
    get_project_dir,
    project_lock,
    sanitize_path,
)
from codemason.utils.subagent_gate import _slot_path  # noqa: E402

results = []


def check(name, cond, detail=''):
    ok = bool(cond)
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'   [{detail}]' if detail and not ok else ''))
    results.append(ok)


# ── truncate_tool_result (rs: truncate_keeps_short / truncate_long_marks_lines)
check('truncate: short content is untouched', truncate_tool_result('short') == 'short')
_long = ''.join('\n' if i % 100 == 0 else 'x' for i in range(40000))
_out = truncate_tool_result(_long)
check('truncate: long content is marked and shortened',
      'lines truncated' in _out and len(_out) < len(_long))

# ── clean_special_tokens (rs: clean_strips_special_tokens)
check('clean: strips <|…|> markers', clean_special_tokens('a<|foo|>b', False) == 'ab',
      clean_special_tokens('a<|foo|>b', False))
check('clean: drops an empty think block', clean_special_tokens('x<think>  </think>y', False) == 'xy',
      clean_special_tokens('x<think>  </think>y', False))
check('clean: a separated-reasoning close tag becomes a space',
      clean_special_tokens('word</think>next', True) == 'word next',
      clean_special_tokens('word</think>next', True))

# ── recover_leaked_tool_calls (rs: dsml_recovery_basic / dsml_json_param_parsed /
#    json_leak_recovery / no_leak_is_noop)
D = '<｜DSML｜'
_content = (f'before {D}tool_calls>{D}invoke name="Bash">'
            f'{D}parameter name="command" string="true">ls -la</｜DSML｜parameter>'
            f'</｜DSML｜invoke></｜DSML｜tool_calls> after')
_calls, _residual = recover_leaked_tool_calls(_content)
check('DSML leak: one call recovered', len(_calls) == 1, len(_calls))
if _calls:
    _c = _calls[0]
    _name = _c.get('name') if isinstance(_c, dict) else getattr(_c, 'name', None)
    _input = _c.get('input') if isinstance(_c, dict) else getattr(_c, 'input', None)
    check('DSML leak: tool name', _name == 'Bash', _name)
    check('DSML leak: argument value', (_input or {}).get('command') == 'ls -la', _input)
check('DSML leak: prose survives on both sides',
      'before' in _residual and 'after' in _residual, _residual)
check('DSML leak: no markup left in the answer', 'DSML' not in _residual, _residual)

_calls2, _ = recover_leaked_tool_calls(
    f'{D}invoke name="T">{D}parameter name="n" string="false">42</｜DSML｜parameter></｜DSML｜invoke>')
_in2 = (_calls2[0].get('input') if isinstance(_calls2[0], dict) else getattr(_calls2[0], 'input', {})) if _calls2 else {}
check('DSML leak: a non-string parameter is parsed as JSON', _in2.get('n') == 42, _in2)

_calls3, _res3 = recover_leaked_tool_calls(
    'I\'ll run it {"name": "Bash", "arguments": {"command": "echo {nested}"}} done')
check('JSON leak: one call recovered', len(_calls3) == 1, len(_calls3))
if _calls3:
    _in3 = _calls3[0].get('input') if isinstance(_calls3[0], dict) else getattr(_calls3[0], 'input', {})
    check('JSON leak: nested braces survive', (_in3 or {}).get('command') == 'echo {nested}', _in3)
check('JSON leak: the serialised call is removed from the answer',
      'arguments' not in _res3, _res3)

_calls4, _res4 = recover_leaked_tool_calls('plain answer')
check('no leak: nothing recovered, answer untouched', not _calls4 and _res4 == 'plain answer')

# ── degenerate guard (rs: a_collapsed_decoder_is_cut / two_distinct_characters…/
#    a_two_character_cycle_also_trips / real_output_does_not_trip_the_guard)
g = _DegenerateRunGuard(2000)
g.feed('!' * (SINGLE_CHAR_RUN_CHARS - 1))
check('degenerate: one character under the short window does not trip', g.tripped is None)
g.feed('!')
check('degenerate: one character at the short window trips', g.tripped is not None)

g = _DegenerateRunGuard(2000)
g.feed('-=' * 999)
check('degenerate: two characters do not trip at the short window', g.tripped is None)
g.feed('-=')
check('degenerate: two characters trip at the full window', g.tripped is not None)

g = _DegenerateRunGuard(2000)
g.feed('ab' * 1500)
check('degenerate: a two-character cycle trips', g.tripped is not None)

for sample in ['fn main() { println!("hello"); } // ordinary code, repeated. ',
               '안녕하세요 반갑습니다. 오늘은 날씨가 좀 춥습니다. ',
               'curl -s https://example.com/a/b/c?x=1 | jq .data[] ']:
    g = _DegenerateRunGuard(2000)
    fed = ''
    while len(fed) < 8000:
        fed += sample
    g.feed(fed)
    check(f'degenerate: real output does not trip ({sample[:18]}…)', g.tripped is None)

# ── ceiling (rs: the_longest_real_reasoning_pass_stays_under_the_ceiling)
check('silent-reasoning ceiling clears the largest legitimate pass on record',
      26271 < _SILENT_REASONING_TOKENS, _SILENT_REASONING_TOKENS)

# ── config gates (rs: auto_ignores_the_file_and_uses_the_family_table /
#    an_unknown_profile_falls_back_to_auto / manual_sends_the_file_verbatim /
#    the_environment_has_the_last_word)
for _name, _temp, _topk in [('glm-5.3-flash', 1.0, None), ('GLM-5.3', 1.0, None),
                            ('DSV4-Flash', 0.6, 40), ('deepseek-v4', 0.6, 40), ('', 0.6, 40)]:
    c = Config()
    c.model_name = _name
    c.temperature, c.top_k, c.top_p, c.frequency_penalty = 0.123, 7, 0.111, 0.9
    s = c.effective_sampling()
    check(f'auto: family table for {_name or "(unset)"}',
          s.get('temperature') == _temp and s.get('top_k') == _topk
          and s.get('top_p') == 0.95 and s.get('min_p') == 0.0
          and 'frequency_penalty' not in s, s)

for _profile in ['', 'auto', 'AUTO', 'vendor', 'typo']:
    c = Config()
    c.model_name = 'glm-5.3-flash'
    c.sampling_profile = _profile
    c.temperature = 0.123
    check(f'auto: an unknown profile falls back to it ({_profile!r})',
          c.effective_sampling().get('temperature') == 1.0)

c = Config()
c.model_name = 'glm-5.3-flash'
c.sampling_profile = 'manual'
c.temperature, c.top_k, c.frequency_penalty = 0.42, 7, 0.3
s = c.effective_sampling()
check('manual: the file is sent verbatim',
      s.get('temperature') == 0.42 and s.get('top_k') == 7
      and s.get('frequency_penalty') == 0.3, s)

os.environ['TEMPERATURE'] = '0.6'
os.environ['TOP_K'] = '40'
c = Config(); c.model_name = 'glm-5.3-flash'
_auto = c.effective_sampling()
c.sampling_profile = 'manual'; c.temperature = 0.42
_manual = c.effective_sampling()
os.environ.pop('TEMPERATURE', None); os.environ.pop('TOP_K', None)
check('env: beats the auto table', _auto.get('temperature') == 0.6, _auto)
check('env: can add a knob auto omits', _auto.get('top_k') == 40, _auto)
check('env: beats the file too', _manual.get('temperature') == 0.6, _manual)

c = Config(); c.model_name = 'glm-5.3-flash'
c2 = Config(); c2.model_name = 'DSV4-Flash'
check('detector: off for GLM', not c.repetition_detection_enabled())
check('detector: on for DeepSeek', c2.repetition_detection_enabled())

# ── realtime display filter (rs: realtime_render tests)
from codemason.utils.realtime_render import (           # noqa: E402
    MarkdownBlockBuffer,
    RealtimeStreamFilter,
    _tail_partial,
)

f = RealtimeStreamFilter()
check('filter: an unclosed pipe marker holds nothing back yet', f.feed('<|tool') == '',
      repr(f.feed('')))
f = RealtimeStreamFilter()
_longtxt = '<|' + 'a' * 4000
_o = f.feed(_longtxt) + f.flush()
check("filter: prose starting '<|' survives instead of being held", _o == _longtxt, len(_o))

f = RealtimeStreamFilter()
check('filter: a DSML invoke suppresses the rest of the answer',
      f.feed('hello <｜DSML｜invoke name="Bash"> more') == 'hello ')
check('filter: nothing escapes after the suppression', f.feed('anything') == '' and f.flush() == '')

f = RealtimeStreamFilter()
check('filter: an orphan closing tag suppresses the spill',
      f.feed('text </｜DSML｜parameter> spill') == 'text ')

f = RealtimeStreamFilter()
_o = f.feed('a<thi') + f.feed('nk>hidden</th') + f.feed('ink>b') + f.flush()
check('filter: a think block split across chunks is dropped whole', _o == 'ab', repr(_o))

f = RealtimeStreamFilter()
_o = f.feed('x<|en') + f.feed('d|>y') + f.flush()
check('filter: a special token split across chunks is stripped', _o == 'xy', repr(_o))

f = RealtimeStreamFilter()
_o = f.feed('a < b and 1<2') + f.flush()
check('filter: real less-than signs are emitted', _o == 'a < b and 1<2', repr(_o))

f = RealtimeStreamFilter()
_o = f.feed('dessin</arg_value></tool_call>') + f.flush()
check('filter: orphan tool tags are stripped without eating the answer',
      _o.strip() == 'dessin', repr(_o))

f = RealtimeStreamFilter()
_o = ''.join(f.feed(c) for c in ['ok</arg_', 'value><tool_', 'call>rest']) + f.flush()
check('filter: a tag split across chunks is still stripped',
      'arg_value' not in _o and 'tool_call' not in _o and 'ok' in _o and 'rest' in _o, repr(_o))

f = RealtimeStreamFilter()
_o = f.feed('if a < b && c > d then') + f.flush()
check('filter: real angle brackets are untouched', 'a < b' in _o, repr(_o))

_s = 'hello </thi'
check('tail_partial: keeps exactly the partial marker',
      _s[len(_s) - _tail_partial(_s, '</think>'):] == '</thi',
      repr(_s[len(_s) - _tail_partial(_s, '</think>'):]))

b = MarkdownBlockBuffer()
_blocks = b.feed('para1 line\n\n```py\ncode\n\nmore\n```\npara2')
check('block buffer: paragraphs and fences split correctly',
      _blocks == ['para1 line', '```py\ncode\n\nmore\n```'], _blocks)
check('block buffer: the tail is flushed', b.flush() == 'para2', repr(b.flush()))


# ── compaction (rs: ptl_gap_* / looks_like_ptl / format_summary_strips_analysis /
#    subagent_default_is_derived… / the_token_bar_width_can_never_underflow)
from codemason.services import compaction as _compaction      # noqa: E402
from codemason.services.compaction import (              # noqa: E402
    SUBAGENT_THRESHOLD_FRACTION,
    SUBAGENT_TOTAL_FRACTION,
    _looks_like_prompt_too_long,
    _parse_ptl_token_gap,
    calculate_token_warning_state,
    format_compact_summary,
    get_auto_compact_threshold,
    note_decoder_collapse,
)

# rs: a_collapse_is_reported_and_changes_nothing.
#
# An observed collapse is RECORDED and must change nothing. It briefly moved the
# threshold in the Rust build and that was wrong: deriving a ceiling from one
# collapse at 264,197 on a 524,288 window took the threshold from 445,752 to
# 211,357 - 40% of a window the operator is paying for, discarded on one sample.
#
# This deployment's own turns.log then showed the sample is not even ordered by
# size: 327,795 tokens came back healthy, two 50k bands collapse at 0.0%, and
# the per-day rate swings 0.55%..25.98% on identical code. Recording it is
# useful; acting on it is not.
_collapse_before = get_auto_compact_threshold(False)
_compaction._OBSERVED_COLLAPSE_TOKENS = None
note_decoder_collapse(264_197)
check('collapse: an observed collapse does not move the threshold',
      get_auto_compact_threshold(False) == _collapse_before, get_auto_compact_threshold(False))
check('collapse: but it IS recorded',
      _compaction._OBSERVED_COLLAPSE_TOKENS == 264_197, _compaction._OBSERVED_COLLAPSE_TOKENS)
note_decoder_collapse(400_000)
check('collapse: a later, higher collapse says nothing new',
      _compaction._OBSERVED_COLLAPSE_TOKENS == 264_197, _compaction._OBSERVED_COLLAPSE_TOKENS)
note_decoder_collapse(200_000)
check('collapse: a new low replaces it',
      _compaction._OBSERVED_COLLAPSE_TOKENS == 200_000, _compaction._OBSERVED_COLLAPSE_TOKENS)
note_decoder_collapse(0)
check('collapse: a non-positive prompt is ignored',
      _compaction._OBSERVED_COLLAPSE_TOKENS == 200_000, _compaction._OBSERVED_COLLAPSE_TOKENS)
check('collapse: three collapses, threshold still untouched',
      get_auto_compact_threshold(False) == _collapse_before, get_auto_compact_threshold(False))
_compaction._OBSERVED_COLLAPSE_TOKENS = None

check('ptl gap: vLLM wording',
      _parse_ptl_token_gap("This model's maximum context length is 262144 tokens. However, "
                           "you requested a total of at least 270000 tokens") == 270000 - 262144,
      _parse_ptl_token_gap("This model's maximum context length is 262144 tokens. However, "
                           "you requested a total of at least 270000 tokens"))
check('ptl gap: input/output wording',
      _parse_ptl_token_gap('maximum context length is 1000 tokens, got 900 input tokens '
                           'and 200 output tokens') == 100,
      _parse_ptl_token_gap('maximum context length is 1000 tokens, got 900 input tokens '
                           'and 200 output tokens'))
check('ptl gap: unparseable stays None',
      _parse_ptl_token_gap('prompt is too long') is None and _parse_ptl_token_gap(None) is None)
check('ptl detect: recognises the overflow error',
      _looks_like_prompt_too_long('Error: maximum context length exceeded'))
check('ptl detect: an unrelated error is not one',
      not _looks_like_prompt_too_long('connection refused'))

_sum = format_compact_summary('<analysis>thinking</analysis>\n\n<summary>\n'
                              'the facts $1 \\g<0>\n</summary>')
check('summary: analysis is stripped and the body survives verbatim',
      _sum.startswith('Summary:') and 'the facts $1 \\g<0>' in _sum and 'analysis' not in _sum,
      repr(_sum[:80]))

# Spawning a sub-agent is never prompted, on BOTH lines, and TWO independent
# gates have to agree on that. `needs_permissions()` is the one query.py
# actually consults, and it is what made this correct all along; check_permissions
# is the source's own method and the answer if that gate ever moves. The
# original's only other branch there routes through the auto-mode security
# classifier, guarded by a compile-time constant that is false in the external
# build, so allow is the whole of it.
#
# Both are pinned because neither is exercised by an end-to-end run: every one
# of those used -dsp, where the permission flow is bypassed entirely.
import asyncio as _asyncio_perm

from codemason.tools.agent_tool import AgentTool as _AgentToolPerm

_perm = _asyncio_perm.run(_AgentToolPerm.check_permissions({'prompt': 'p'}, None))
check('agent: spawning a sub-agent is auto-approved, never prompted',
      _perm.behavior == 'allow', _perm.behavior)
check('...by the gate query actually consults, needs_permissions',
      _AgentToolPerm.needs_permissions() is False,
      _AgentToolPerm.needs_permissions())
check('...and the tool is NOT read-only, which is why the other gate matters',
      _AgentToolPerm.is_read_only() is False, _AgentToolPerm.is_read_only())
# The Rust checkout, wherever it is on this machine. Hardcoding one absolute
# path meant this open raised FileNotFoundError at module level everywhere
# else, taking the rest of the file's cases down with it.
def _rust_file(rel):
    roots = []
    env = os.environ.get('CODEMASON_RUST_ROOT')
    if env:
        roots.append(env)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots += [
        os.path.join(os.path.dirname(here), 'codemason_v2.0'),  # sibling checkout
        '/home/user/work/codemason_v2.0',
        r'C:\root\codemason_v2.0',
    ]
    for r in roots:
        cand = os.path.join(r, *rel.split('/'))
        if os.path.exists(cand):
            return cand
    return None


_rust_agent_tool = _rust_file('src/tools/agent_tool.rs')
if _rust_agent_tool is None:
    print('SKIP  ...and the Rust line overrides it too, to the same answer'
          ' (no Rust checkout found; set CODEMASON_RUST_ROOT)')
else:
    _rust_src = open(_rust_agent_tool, encoding='utf-8').read()
    check('...and the Rust line overrides it too, to the same answer',
          'async fn check_permissions' in _rust_src
          and 'PermissionResult::allow()' in _rust_src)

_parent = 445752
_derived = int(_parent * SUBAGENT_THRESHOLD_FRACTION)
check('subagent: the derived ceiling lowers the threshold and matches the Rust number',
      _derived == 133725, _derived)

# Raising the slot count must not multiply the sub-agent footprint: the
# per-agent fraction was measured with ONE extra tenant, so on its own it says
# nothing about concurrency (rs: the_subagent_share_shrinks_as_the_slot_count_grows).
def _cap(slots):
    share = min(SUBAGENT_THRESHOLD_FRACTION,
                SUBAGENT_TOTAL_FRACTION / max(1, slots))
    return int(_parent * share)


check('subagent: the measured slot counts keep the measured per-agent cap',
      _cap(0) == _cap(1) == _cap(2) == _cap(3) == 133725,
      [_cap(n) for n in (0, 1, 2, 3)])
check('subagent: a fourth slot costs each agent something',
      _cap(4) < _cap(3), (_cap(4), _cap(3)))
_budget = int(_parent * SUBAGENT_TOTAL_FRACTION)
check('subagent: the combined footprint stops growing past the measured total',
      all(_cap(n) * n <= _budget + n for n in range(4, 13)),
      [(n, _cap(n) * n) for n in range(4, 13)])

for _usage in [0, 1, 1000, 32000, 100000, 10000000]:
    _st = calculate_token_warning_state(_usage)
    _left = _st['percent_left'] if isinstance(_st, dict) else getattr(_st, 'percent_left', None)
    _ok = _left is not None and 0 <= _left <= 100 and (20 * (100 - _left) // 100) <= 20
    check(f'token bar: width stays in range at usage {_usage}', _ok, _left)

# ── query helpers (rs: tool_call_key_ignores_description / truncation_does_not_split… /
#    the_nudge_quotes_what_the_failing_calls_returned / every_stage_asks_for_something_different)
from codemason.query import (                            # noqa: E402
    REPETITION_LAST_RESORT_ATTEMPT,
    REPETITION_RECOVERY_LIMIT,
    _REPETITION_RECOVERY_PROMPT,
    _repetition_stage_prompt,
    _tool_call_key,
    _truncate_chars,
)

check('tool key: the description field is not part of the identity',
      _tool_call_key('Bash', {'command': 'ls', 'description': 'one'})
      == _tool_call_key('Bash', {'command': 'ls', 'description': 'two'}))
_ko = '실행이 되돌려졌습니다' * 200
_cut = _truncate_chars(_ko, 300)
check('truncate: a multibyte character is never split',
      len(_cut) == 304 and _cut.startswith('실행이'), len(_cut))
# The turn gets THREE requests: the one that collapsed, one plain retry, and one
# on a reduced context. Pinned as a total, because that is the number the
# measurement is about — 94.9% of 176 collapse runs escaped on the retry,
# nothing escaped by nudge after the 6th attempt, and the reduction cured both
# collapses that got that far on its first try.
#
# This replaces a `LIMIT >= 4` check that encoded the opposite belief ("the
# budget is worth spending"). It was reasonable and it was wrong: the talking
# never worked, and holding the reduction behind it cost ~90 seconds and 6
# requests per collapse. Raising the budget again needs new numbers in
# turns.log, not a new argument.
_requests, _count = 1, 0          # 1 = the request that collapsed
while _count <= REPETITION_LAST_RESORT_ATTEMPT:
    _requests += 1
    _count += 1
check('recovery ladder: the turn gets exactly three requests',
      _requests == 3, _requests)
# At least one plain retry must come before the reduction, or every collapse
# would pay the irreversible step — including the 94.9% a retry already cures.
check('recovery ladder: the first retry does not reduce the context',
      REPETITION_RECOVERY_LIMIT < REPETITION_LAST_RESORT_ATTEMPT,
      (REPETITION_RECOVERY_LIMIT, REPETITION_LAST_RESORT_ATTEMPT))
# 0..=LAST_RESORT, not 0..=LIMIT: the reducing attempt carries a nudge too, and
# it is the one the turn most depends on, so it belongs in this check. Every
# REACHABLE stage is counted and they must all differ.
_stages = [_repetition_stage_prompt(i) for i in range(REPETITION_LAST_RESORT_ATTEMPT + 1)]
check('recovery ladder: the stages do not collapse into one instruction',
      len(set(_stages)) == len(_stages), (len(set(_stages)), len(_stages)))
check('recovery ladder: stage 1 stays the mild one',
      _stages[0] == _REPETITION_RECOVERY_PROMPT)
# The attempt that is actually last, not LIMIT - 1: with the budget cut to one
# nudge that expression goes negative and quietly tests the MILD stage.
_last = _repetition_stage_prompt(REPETITION_LAST_RESORT_ATTEMPT)
check('recovery ladder: the last stage releases the work',
      ('continue with' in _last or 'move on' in _last) and 'Do not run another tool' in _last,
      _last[:90])

# --- the token heuristic, ported from tokens.rs ---------------------------
#
# Ports of dense_scripts_are_counted_one_token_per_char,
# the_non_latin_alphabets_are_dense_too, symbols_and_emoji_are_charged_two and
# latinish_text_keeps_the_chars_per_token_rule. The heuristic decides when
# compaction fires, so a divergence here is a divergence in when the two clients
# summarize.
from codemason.utils.tokens import (                 # noqa: E402
    _heuristic_token_count,
    _tally_scripts,
    has_model_tokenizer,
    oob_scope_key,
    set_out_of_band_tokens,
    _out_of_band_for,
)

_korean = '안녕하세요반갑습니다'
check('heuristic: dense scripts are one token per char',
      _tally_scripts(_korean) == (0, 10, 0), _tally_scripts(_korean))
if not has_model_tokenizer():
    check('heuristic: 10 Hangul syllables ~ 10 tokens',
          _heuristic_token_count(_korean, 4) == 10, _heuristic_token_count(_korean, 4))
    check('heuristic: latin text keeps the chars/4 rule',
          _heuristic_token_count('abcdefgh', 4) == 2, _heuristic_token_count('abcdefgh', 4))
    check('heuristic: an emoji is charged two',
          _heuristic_token_count('\U0001F600', 4) == 2,
          _heuristic_token_count('\U0001F600', 4))

_alphabets_ok = True
for _name, _sample, _expect in [
    ('Cyrillic', 'привет', 6),
    ('Greek', 'καλημέρα', 8),
    ('Arabic', 'مرحبا', 5),
    ('Hebrew', 'שלום', 4),
    ('Thai', 'สวัสดี', 6),
    ('Devanagari', 'नमस्ते', 6),
    # CJK punctuation: the old range began at 0x3040 and left these at chars/4,
    # though they end every Japanese sentence.
    ('JP punctuation', '。、「」', 4),
    # Astral ideographs: the old list stopped at 0xFFEF.
    ('CJK Ext B', '\U00020000\U00020001', 2),
]:
    _lat, _dense, _sym = _tally_scripts(_sample)
    if _dense != _expect or _lat != 0:
        _alphabets_ok = False
        print(f'      {_name} {_sample!r}: latin={_lat} dense={_dense} want dense={_expect}')
check('heuristic: the non-latin alphabets are dense too', _alphabets_ok)

_a, _b = oob_scope_key(['Bash', 'Read']), oob_scope_key(['Bash'])
set_out_of_band_tokens(_a, 25000)
set_out_of_band_tokens(_b, 3000)
check('out-of-band tokens are per tool-set, not per process',
      _out_of_band_for(_a) == 25000 and _out_of_band_for(_b) == 3000,
      (_out_of_band_for(_a), _out_of_band_for(_b)))
check('out-of-band: an unregistered scope falls back to the largest, not to 0',
      _out_of_band_for('nothing-registered-under-this') == 25000)


# effective_reasoning_effort: the per-model override wins over the global
# setting, matching Rust's Config::effective_reasoning_effort. The field
# existed here for weeks with nothing reading it -- the request builder hid
# the missing method behind hasattr and silently used the global value, and
# the startup banner died with AttributeError on every launch.
_c = Config()
_c.model_name = 'glm-5.3-flash'
_c.reasoning_effort = 'high'
_c.reasoning_effort_per_model = {'glm-5.3-flash': 'MAX'}
check('effective_reasoning_effort: per-model wins, normalized',
      _c.effective_reasoning_effort() == 'max', _c.effective_reasoning_effort())
_c.reasoning_effort_per_model = {'some-other-model': 'low'}
check('effective_reasoning_effort: unrelated per-model entry is ignored',
      _c.effective_reasoning_effort() == 'high', _c.effective_reasoning_effort())
_c.model_name = ''
_c.reasoning_effort_per_model = {'glm-5.3-flash': 'low'}
check('effective_reasoning_effort: no model name falls back to the global',
      _c.effective_reasoning_effort() == 'high', _c.effective_reasoning_effort())
_c.model_name = 'glm-5.3-flash'
_c.reasoning_effort = ''
_c.reasoning_effort_per_model = {}
check('effective_reasoning_effort: empty stays empty (server decides)',
      _c.effective_reasoning_effort() == '', _c.effective_reasoning_effort())

# ── Cross-build file-sharing contract ───────────────────────────────────────
#
# These are not implementation details: both builds run against ONE ~/.codemason,
# so a path that differs by a byte means the two never exclude each other and
# never share a pool — with nothing in either suite noticing. The Rust side pins
# the same literals (utils/config.rs: the_config_lock_path_is_the_one_the_python
# _build_uses, the_endpoint_tag_matches_the_python_build).
check('lock dir: every lock lives in <config>/locks',
      LOCKS_DIR.name == 'locks' and LOCKS_DIR.parent.name == '.codemason'
      or LOCKS_DIR.name == 'locks',                 # CODEMASON_CONFIG_DIR relocates it
      str(LOCKS_DIR))
check('lock path: config.json -> locks/config.json.lock',
      lock_path('config.json') == str(LOCKS_DIR / 'config.json.lock'),
      lock_path('config.json'))
check('lock path: the one the config WRITER takes, not just one spelled alike',
      config_lock_path() == str(LOCKS_DIR / 'config.json.lock'), config_lock_path())
check('lock path: settings.json and mcp.json follow the same rule',
      lock_path('settings.json') == str(LOCKS_DIR / 'settings.json.lock')
      and lock_path('mcp.json') == str(LOCKS_DIR / 'mcp.json.lock'),
      lock_path('settings.json') + ' | ' + lock_path('mcp.json'))

# The endpoint tag keys the sub-agent pool, which both builds contend for.
check('endpoint tag: sha256[:12] of the normalized URL',
      endpoint_tag('https://llm.systemboan.net') == '4214d21b0e48',
      endpoint_tag('https://llm.systemboan.net'))
check('endpoint tag: second reference value',
      endpoint_tag('http://127.0.0.1:8000/v1') == '87459d11adee',
      endpoint_tag('http://127.0.0.1:8000/v1'))
check('endpoint tag: trimmed and trailing slash dropped (one server, one pool)',
      endpoint_tag('  https://llm.systemboan.net/  ') == endpoint_tag('https://llm.systemboan.net'))
# Scheme and host are case-insensitive, so folding them keeps one server one pool.
# The PATH is not folded: it is case-sensitive, and two endpoints differing only
# there would otherwise be rationed against a cache neither shares.
check('endpoint tag: scheme and host fold case (one server, one pool)',
      endpoint_tag('https://LLM.SystemBoan.NET') == '4214d21b0e48'
      and endpoint_tag('http://LocalHost:8000') == endpoint_tag('http://localhost:8000'),
      endpoint_tag('https://LLM.SystemBoan.NET'))
check('endpoint tag: the path keeps its case (two endpoints, two pools)',
      endpoint_tag('http://h:8000/v1') != endpoint_tag('http://h:8000/V1'))
check('endpoint tag: no endpoint -> "none"', endpoint_tag('   ') == 'none')
check('sub-agent slot: subagent.<endpoint>.<i>.lock',
      _slot_path('abc123', 2) == str(LOCKS_DIR / 'subagent.abc123.2.lock'),
      _slot_path('abc123', 2))

# Per-project documents are keyed by the sanitized cwd, like the transcripts.
# Literal paths, not sanitize_path on both sides: deriving it in each build would
# only assert that each build agrees with itself.
check('project lock: <sanitized-cwd>.<document>.lock',
      project_lock('goal-gates.txt', '/tmp/x') == str(LOCKS_DIR / '-tmp-x.goal-gates.txt.lock'),
      project_lock('goal-gates.txt', '/tmp/x'))
check('project lock: second reference value',
      project_lock('settings.json', '/home/user/work')
      == str(LOCKS_DIR / '-home-user-work.settings.json.lock'),
      project_lock('settings.json', '/home/user/work'))
_goal_dir = str(get_project_dir('/tmp/x'))
check('project goal lives in the project dir, not the config dir',
      os.path.basename(_goal_dir) == '-tmp-x'
      and os.path.basename(os.path.dirname(_goal_dir)) == 'projects',
      _goal_dir)

print('\n' + (f'ALL PASS ({len(results)})' if all(results)
              else f'FAILURES: {results.count(False)} / {len(results)}'))
sys.exit(0 if all(results) else 1)
