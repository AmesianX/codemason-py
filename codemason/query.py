"""
Query Loop for CodeMason
Equivalent to src/query.ts
"""

import asyncio
import hashlib
import json
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set, Type, Union
from uuid import uuid4

from .types.messages import Message as APIMessage, ToolUseBlock, TextBlock, BetaUsage
from .types.query import (
    UserMessage,
    AssistantMessage,
    ProgressMessage,
    Message,
    ToolUseContext,
    FullToolUseResult,
)
from .tools.base import Tool, ToolResult
from .services.backend import query_model
from .utils.log import debug_log
from .utils.messages import (
    create_user_message,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
)
from .utils.generators import abortable, all as all_generators
from .services.compaction import auto_compact_if_needed


MAX_TOOL_USE_CONCURRENCY = 10

# query.ts:164. When a turn ends because the model hit the output token limit
# (finish_reason=length -> stop_reason=max_tokens) rather than stopping
# naturally, the response was cut mid-thought. We inject a resume nudge and
# re-query, up to this many times per stall, so a weak local model that keeps
# bumping the cap continues toward the goal instead of silently halting.
MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3

# Cap on how many times, within ONE user turn, a failing goal gate may push the
# turn to continue. Bounds the auto-gate loop so a gate that can't pass doesn't
# spin forever; a new user message resets the count (via process_prompt).
GOAL_GATE_RECOVERY_LIMIT = 5

_GATE_CONTINUATION_TEMPLATE = (
    "[Continuing toward your standing goal — a completion gate failed]\n"
    "Goal: {goal}\n\n"
    "This goal is complete only when its gate commands pass, and "
    "{n} just failed (auto-check {attempt}/{limit}):\n{failures}\n"
    "Fix the underlying problem so the gate passes, then it will be re-checked. "
    "Do not declare the goal done while any gate fails. If a gate itself is "
    "wrong or genuinely cannot pass, say so clearly and stop."
)


def _format_gate_failures(goal: str, failed: list, attempt: int, limit: int) -> str:
    """Build the continuation prompt for one or more failed gates."""
    blocks = []
    for cmd, code, tail in failed:
        tail = (tail or "").strip()
        body = f"  $ {cmd}\n  exit code: {code}"
        if tail:
            body += "\n  output (tail):\n  ```\n" + tail + "\n  ```"
        blocks.append(body)
    return _GATE_CONTINUATION_TEMPLATE.format(
        goal=goal, n=len(failed), attempt=attempt, limit=limit,
        failures="\n".join(blocks),
    )


# query.ts:1209 — the verbatim recovery prompt (isMeta user message).
# The cut turn's reasoning is now carried into this request — recovery requests
# hold clear_thinking OFF, see services/_openai_common. So the instruction is to
# CONTINUE from that thinking, not to start over without it.
#
# It used to say "no recap of what you were doing", written when the thinking was
# in fact gone: the recovery injects a user message, the server template treats
# everything before a user message as a prior turn, and clear_thinking dropped
# it. Traced 2026-09-13 — a turn spent its whole 65,536-token budget on the
# reasoning channel while working out why a Foundry fork test reverted, and the
# next request's prompt FELL by 16,405 tokens instead of rising. The model was
# told to pick up mid-thought with the thought deleted, and redid the same work
# for hours.
_MAX_OUTPUT_RECOVERY_PROMPT = (
    "Output token limit hit mid-thought. Your reasoning from before the cut is "
    "still in this conversation — read it and continue from where it left off "
    "rather than starting the analysis again. No apology. Write your "
    "conclusions down as you reach them instead of holding them in reasoning, "
    "so the next cut cannot cost them. Break the remaining work into smaller "
    "pieces."
)

# Layer-2 repetition recovery (enhancement; not from upstream). vLLM's detector
# cuts a degenerate n-gram loop mid-generation. Rather than end the turn and
# hand a "[Generation stopped]" note to the USER, the loop gets this many
# chances to self-correct — each time the degenerate emission's tool calls are
# discarded and a nudge is injected so the model realizes it looped and changes
# approach WITHIN the same turn. Only after the budget is spent does the turn
# end with a visible note.
# Attempts before the turn is allowed to end. Was 2, which under -dsp meant an
# unattended run stopped with work half-done — the exact failure that mode
# exists to avoid. The budget is now spent on ESCALATING instructions rather
# than repeating one, because repeating the same nudge to a model that is
# already stuck is itself a loop.
#
# READ THE NUMBER CAREFULLY: this is the limit, not the attempt count. The call
# sites test `count <= LIMIT`, so the model gets LIMIT + 1 = 7 attempts, and the
# doc above ("this many chances") read as 6 for long enough to be worth spelling
# out. The extra attempt is deliberate — it is the one that tells the model to
# ABANDON this line of work — and tests/test_parity_with_rust.py pins the
# relationship so the two cannot drift apart again.
#
# WAS 6, giving 8 nudge attempts and a 9th for the reduction. Cut to 0 — ONE
# nudge attempt, then the reduction — on 176 measured collapse runs from
# turns.log, 2026-09-22 to 2026-10-02 (2,483 requests):
#
#   attempt   runs entering   escaped by a nudge
#         2             176          167  (94.9%)
#         3               7            0
#         4               7            2
#         5               4            0
#         6               4            1   <- the most that ever worked
#         7               3            0
#         8               3            0
#
# One plain retry cures 94.9% of collapses. Past that the ladder is close to
# worthless: attempts 7 and 8 have never once saved a turn, and the two
# collapses that reached them (2026-10-02 16:03 and 16:09, at 289,350 and
# 290,984 prompt tokens) were cured by the reduction on its first try, both
# times. The 16-failure run on 2026-10-01 was the /ab abliteration bug, not
# length, and is excluded.
#
# So the nudges were costing 6 requests and ~90 seconds per collapse to buy a
# measured 0, and the step that does work was held back behind them. The budget
# now buys exactly what the data supports: retry once, then reduce.
#
# Also measured and NOT acted on: prompt size separates the two populations
# sharply (every nudge-only recovery was at <= 162,688 tokens; both collapses
# above 250,000 needed the reduction). Gating on size was rejected because the
# gate would have to sit in the 163k-289k range, where there are zero
# observations — a predicted threshold, which is the thing this file keeps
# learning not to trust. Two failed attempts is a measurement.
REPETITION_RECOVERY_LIMIT = 0

# One attempt PAST the nudge ladder, spent on cutting the context down instead
# of on another sentence about it. Taken only when every nudge has failed and
# the sole remaining alternative is ending the turn.
#
# Why last and not first. Talking to the model is free; microcompact is not — it
# replaces the body of every compactable tool result but the newest
# MICROCOMPACT_KEEP_RECENT and cannot be undone (it is idempotent, so a second
# pass frees nothing). Spending that on the FIRST collapse would destroy the
# history in the ~50% of cases a plain retry already recovers, and in every
# collapse whose cause was never length at all — a poisoned tool result, a
# pathological repetition in the input.
#
# Here the cost is close to zero, because the measured alternative is
# _REPETITION_GIVEUP_NOTE and a half-finished unattended run. Nudges first,
# reduction only when they are exhausted.
REPETITION_LAST_RESORT_ATTEMPT = REPETITION_RECOVERY_LIMIT + 1
_REPETITION_RECOVERY_PROMPT = (
    "You started repeating the same output or tool call and the server cut the "
    "generation. Stop — the current approach has stopped making progress. Do "
    "something DIFFERENT this step: a different tool, a different command or "
    "input, or state your conclusion from what you already know. Do not repeat "
    "the last action."
)
# Later stages of the same nudge. Repeating stage 1 at a model that ignored
# stage 1 is the same mistake the model is making, so each stage asks for
# something categorically different: first vary the action, then abandon the
# goal of this step, then abandon the line of work entirely and carry on with
# the rest of the task. The last one is what keeps an unattended run moving:
# the turn does not end, the work does.
_REPETITION_ESCALATION_PROMPTS = (
    "You have been cut short repeatedly now. Changing the command is not "
    "enough — the GOAL of this step is what is not working. Do not attempt it "
    "again in another form. Write down plainly what you have established so "
    "far and what remains unknown, then pick a different part of the task and "
    "continue.",
    "This line of work has failed several times and is not going to succeed as "
    "approached. Abandon it now. State your conclusion from what you already "
    "know — \"this could not be verified\" is a valid and useful conclusion — "
    "and move on to the next item of work. Do not return to this one.",
    "Stop this line of work completely. Do not run another tool against it. "
    "Summarise what you found, mark it unresolved, and continue with whatever "
    "else the task requires. If nothing else remains, say so and finish.",
)


def _repetition_stage_prompt(attempt: int) -> str:
    """The nudge for `attempt` (0-based), escalating and then holding at the last.

    The last-resort attempt gets the FINAL rung, not the rung its index would
    pick. It is now the last attempt of the turn as well as the reduced one, and
    the final rung is the only one that hands an unattended run somewhere to go
    ("mark it unresolved, and continue with whatever else the task requires").
    It is also the text that was measured working: both rescues on 2026-10-02
    ran on it (by clamping, at attempt 8) and came back with think=503 / 1,276
    output and a tool call, from eight consecutive 200-character runs of '!'.

    With REPETITION_RECOVERY_LIMIT at 0 the two middle rungs are unreachable.
    They are kept, not deleted: they are the ladder the budget would climb again
    if the measurement that cut it ever stops holding, and the constant's
    comment says what to re-measure first.
    """
    if attempt <= 0:
        return _REPETITION_RECOVERY_PROMPT
    if attempt >= REPETITION_LAST_RESORT_ATTEMPT:
        return _REPETITION_ESCALATION_PROMPTS[-1]
    return _REPETITION_ESCALATION_PROMPTS[
        min(attempt - 1, len(_REPETITION_ESCALATION_PROMPTS) - 1)]


_REPETITION_GIVEUP_NOTE = (
    "\n\n[Stopped: the model kept repeating itself across several attempts and "
    "could not break out of the loop. The output above is incomplete — this "
    "line of work needs a different approach or more information.]"
)

# Header for the failing tool results appended to the repetition nudge.
#
# The nudge alone says "you repeated yourself, do something different", which
# names the symptom and leaves the model to rediscover the cause. The cause is
# almost always in the results it just got: a command that errors, or answers
# something other than what it was asked, returns the model to the identical
# state every round, so it re-emits the identical plan. (The repeated-call cap
# note below records the same finding from the other direction.)
#
# Observed working: given the nudge alone, one recovery turn produced 3611
# characters of reasoning that opened by naming the loop itself — the model
# could see it once told to look. Handing it the failures directly starts it
# from there instead of hoping it gets there.
_REPETITION_FAILURES_HEADER = (
    "\n\nThese are the results the repeated attempts were getting back. The loop "
    "is most likely here — an error, or output that does not answer what the call "
    "was for, leaves you in the same position each round:\n"
)
# How many recent failing tool results to quote, newest last.
REPETITION_FAILURES_MAX = 3
# Characters kept from each quoted result.
REPETITION_FAILURE_CHARS = 300


def _truncate_chars(s: str, max_chars: int) -> str:
    """First `max_chars` characters of `s`, marked when it was cut."""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + ' […]'


def _recent_tool_failures(messages) -> list:
    """Recent tool results that failed or came back empty, oldest first.

    Walks back from the tail over user messages only, so it sees this turn's
    results and stops at the previous assistant text. Empty output counts: a
    command that prints nothing answers nothing, and that is the same dead end
    as an error for a model deciding what to do next.
    """
    out = []
    for msg in reversed(messages):
        role = msg.get('message', {}).get('role') if isinstance(msg, dict) else None
        if role != 'user':
            continue
        content = msg['message'].get('content')
        if not isinstance(content, list):
            continue
        for b in reversed(content):
            if not isinstance(b, dict) or b.get('type') != 'tool_result':
                continue
            raw = b.get('content')
            if isinstance(raw, list):
                text = '\n'.join(
                    x.get('text', '') for x in raw if isinstance(x, dict)
                )
            else:
                text = raw if isinstance(raw, str) else ''
            trimmed = text.strip()
            if not b.get('is_error') and trimmed:
                continue
            out.append(
                '(no output)' if not trimmed
                else _truncate_chars(trimmed, REPETITION_FAILURE_CHARS)
            )
            if len(out) >= REPETITION_FAILURES_MAX:
                out.reverse()
                return out
    out.reverse()
    return out


def _repetition_recovery_prompt(messages, attempt: int = 0) -> str:
    """The repetition nudge for this attempt, with the failing results appended."""
    base = _repetition_stage_prompt(attempt)
    failures = _recent_tool_failures(messages)
    if not failures:
        return base
    body = '\n'.join('- %s' % f for f in failures)
    return base + _REPETITION_FAILURES_HEADER + body


# --- Lost-tool-call recovery ----------------------------------------------
#
# Layer-2 (Rust+Python enhancement, not in upstream): the model invoked a tool
# and NONE of the invocation survived assembly — _openai_common surfaces
# stop_reason 'tool_parse_failed' when it had tool calls to build and produced
# no tool_use block from any of them. Every upstream cause (a stream cut mid
# 'arguments', a non-string 'arguments', a malformed fragment) lands on that one
# signal, so the recovery does not need to know which one it was.
#
# Re-asking is safe here in a way it is not for a PARTIAL loss: with no tool_use
# block, nothing from this turn ran, so there is nothing to run twice. A partial
# loss deliberately does NOT come through this path — it is logged and the
# surviving calls proceed, because re-asking would re-execute them.
#
# The budget exists because the trigger can be deterministic: a server that
# always serialises 'arguments' as an object would fail identically forever, and
# an unattended run must stop with a stated reason rather than spin.
TOOL_PARSE_RECOVERY_LIMIT = 2

# Empty-turn recovery: attempts before the turn is allowed to end.
#
# One, not the two or six the other layers get. A turn that produced nothing at
# all is rarer than a lost tool call or a repetition loop, and asking twice buys
# nothing: if the model had no answer the first time it does not acquire one by
# being asked the same question again. One nudge is the difference between "it
# stopped for no reason" and "it was asked and had nothing", which is the
# distinction worth a request.
EMPTY_TURN_RECOVERY_LIMIT = 1
_EMPTY_TURN_RECOVERY_PROMPT = (
    "Your last turn was empty — no text, no tool call, nothing. Say what you "
    "found, and what you are doing next. If the last thing you tried was a dead "
    "end, say so plainly and take a different approach; if the work is finished, "
    "say that instead. Do not answer with nothing again."
)
_EMPTY_TURN_GIVEUP_NOTE = (
    "\n\n[Stopped: the model returned an empty turn — no text, no tool call — "
    "and did so again when asked. Nothing was done. Whatever it was last looking "
    "at is above.]"
)
_TOOL_PARSE_RECOVERY_PROMPT = (
    "Your last tool call did not arrive intact — the arguments could not be "
    "read, so nothing ran. Nothing from that step took effect. Issue the call "
    "again now, as a normal tool call; if it carried a large argument, send a "
    "smaller one or split the work into more than one call."
)
_TOOL_PARSE_GIVEUP_NOTE = (
    "\n\n[Stopped: the model called a tool but the call could not be read — the "
    "arguments did not parse — and it did not come through on retry. Nothing "
    "from that step ran. See the error log for the raw failure.]"
)

# The max_tokens layer had no give-up note, alone among the four. An exhausted
# budget therefore ended the turn on a sentence cut mid-word, with nothing
# anywhere saying the answer was incomplete — under -dsp indistinguishable from
# a finished one.
_MAX_OUTPUT_GIVEUP_NOTE = (
    "\n\n[Stopped: the model ran into the output limit on every attempt, so the "
    "text above is cut off mid-answer and is incomplete. Ask for the rest, or "
    "for it in smaller pieces.]"
)

# --- Repeated-tool-call cap -----------------------------------------------
#
# A safety net, not a fix. The underlying failure is the model degenerating into
# byte-identical output, and it is not ours to fix here.
#
# An earlier version of this comment blamed the KV cache — 4-bit collapsing under
# long agentic context where fp8 stays clean — and said the server had dealt with
# it. That was wrong on the facts: the server had been serving fp8 the whole time,
# so nothing had been dealt with. What the evidence actually shows (2026-08-06):
#
#   - It is not context length. The same runaway reproduces at ~60 tokens of
#     context, and a run that had been looping recognised the loop and recovered
#     on its own at 158k — deep past the model's 64k trained window.
#   - The strongest single factor is the reasoning-effort level. On this model
#     family `low` injects an EMPTY string, so nothing constrains deliberation;
#     the anti-repetition wording and the reasoning budget live in `high`/`max`.
#     Measured on one hard prompt: `low` burned all 6000 output tokens on
#     reasoning and emitted no answer at all, `high` finished in 79s with one.
#   - The proximate trigger is an uninformative tool result. A command whose
#     output does not answer the question it was issued for returns the model to
#     the identical state each round, so it re-emits the identical plan.
#
# All this cap does is stop a degenerate turn from grinding the same tool call
# forever while nobody is watching.
#
# Counted by BATCH: the whole set of tool calls in one assistant turn, keyed on
# (name, canonical input), contiguous at the tail of the history.
#
# Two stages, following the shape recommended on QwenLM/qwen-code#4695 — the
# same failure reported against this model family, where a client watched 43
# identical calls go by with no circuit breaker.
#
# The nudge exists because ending the turn is not the same as breaking the loop,
# and we have watched the model break its own loop the moment it noticed one:
# after many wasted rounds it wrote "I have fallen into a loop repeating the
# same cast command", dropped the blocked sub-goal and finished. Naming the loop
# is the trigger. The nudge just says it early instead of waiting for the model
# to arrive there on its own, and says it in the one place the model cannot skim
# past — where the tool result it asked for should have been.
#
# The first three run: polling the same command a few times is ordinary (a build,
# a service coming up), and a guard that fires on three would break it.
REPEATED_TOOL_CALL_NUDGE = 4    # not run; the result slot carries a warning
REPEATED_TOOL_CALL_LIMIT = 6    # not run; the turn ends

# What the model reads instead of the result it expected. Addressed to the
# model, so English, like _MAX_OUTPUT_RECOVERY_PROMPT above.
_REPEATED_TOOL_CALL_NUDGE_PROMPT = (
    "This exact call, with these exact arguments, has now been made {n} times "
    "in a row, and it was not run again. Repeating it will not return anything "
    "different.\n"
    "If the output you needed was missing from the earlier results, the command "
    "itself is what has to change — a different query, or one that does not "
    "discard the part you are looking for. If the answer is simply not "
    "reachable this way, say what is blocked and answer in text instead. Do not "
    "issue this call again."
)

# Injected into a SUB-AGENT when the user has interjected in the MAIN conversation
# and a steering line is waiting to be delivered. Tool execution can't be
# interrupted mid-call, so a long sub-agent would otherwise hold the main turn
# (and the user's message) until it fully finished. This asks the sub-agent to
# wrap up cooperatively — summarize what it has so far and stop — so the main turn
# reaches its gate and delivers the interjection promptly, without discarding the
# work the sub-agent already did.
# Injected after a mid-turn interjection in the MAIN conversation.
#
# The steering text used to go in bare. To the model that is indistinguishable
# from a fresh question, so it answers and ends the turn — and the work it was
# running is silently abandoned. Observed: the user asked a check-in question
# during a research run, got a full answer, and the run stopped there with
# nothing said about it. Under -dsp nobody is watching, so that is a halt.
#
# The subagent path already had the opposite instruction (_AGENT_STEER_WRAPUP,
# "wrap up now"), because a subagent's caller IS waiting. The main path had
# neither, so the model was left to guess, and guessing "this is a new turn" is
# the reasonable reading of a bare message.
#
# Deliberately not an unconditional "keep going": an interjection is often a
# redirect or a stop, and overriding that would be worse than halting. The
# instruction is to resume UNLESS the message itself said otherwise.
_MAIN_STEER_RESUME = (
    "The message above arrived while you were working. The task you were "
    "running is NOT finished and no one has cancelled it. Answer the message "
    "briefly, then GO BACK to what you were doing and carry on in the same "
    "turn — do not stop to summarise, and do not re-plan. The exception is the "
    "message itself: if it told you to stop, or to do something else instead, "
    "follow it and ignore this note."
)

_AGENT_STEER_WRAPUP = (
    "The user has just interjected in the main conversation and is waiting on it. "
    "Stop starting new work now: do NOT call any more tools. Immediately write up "
    "what you have found SO FAR as your final answer — partial results are fine and "
    "expected — then end your turn."
)


def _msg_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field off a message/block that may be a dataclass or a dict.

    query() sees both: assistant turns are AssistantMessage dataclasses, tool
    results are plain dicts, and a /resume'd history is dicts all the way down.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tool_batch_signature(msg: Any) -> Optional[str]:
    """Canonical signature of an assistant turn's tool calls, or None.

    None means "not a tool-calling assistant turn" — neutral, skipped by the
    caller rather than treated as a break in the run.
    """
    if _msg_attr(msg, 'type') != 'assistant':
        return None
    content = _msg_attr(_msg_attr(msg, 'message', {}), 'content', []) or []
    if not isinstance(content, list):
        return None

    calls = []
    for block in content:
        if _msg_attr(block, 'type') != 'tool_use':
            continue
        calls.append((
            _msg_attr(block, 'name', '') or '',
            json.dumps(_msg_attr(block, 'input', {}) or {},
                       sort_keys=True, ensure_ascii=False, default=str),
        ))
    if not calls:
        return None
    return json.dumps(calls, ensure_ascii=False)


def _count_repeated_tool_batches(messages: List[Message], signature: str) -> int:
    """How many assistant turns at the tail of `messages` issued this exact batch.

    Walks backward, skipping anything that is not a tool-calling assistant turn
    (tool results sit between the repeats by construction), and stops at the
    first DIFFERING batch. Contiguity at the tail is the only thing that ends a
    run: a message the user typed does NOT reset it, because after /resume "해"
    re-enters the identical dead end and a reset there just restarts the loop.
    A user turn that genuinely changes the situation changes what the model does,
    which changes the signature, which ends the run anyway.
    """
    run = 0
    for msg in reversed(messages):
        sig = _tool_batch_signature(msg)
        if sig is None:
            continue
        if sig != signature:
            break
        run += 1
    return run


# === Type Definitions ===

CanUseToolFn = Callable[
    [Type[Tool], Dict[str, Any], ToolUseContext, AssistantMessage],
    Awaitable[Dict[str, Any]]
]


# === Helper Functions ===

def create_assistant_message_obj(text: str) -> AssistantMessage:
    """Create a SYNTHETIC assistant message (used only for interrupt notices).

    model MUST be '<synthetic>' (not the real model): these carry zero usage, so
    if they aren't flagged synthetic the token-usage anchor walk latches onto
    their 0 and the running token count collapses -> autocompact stops firing.
    """
    return AssistantMessage(
        message=APIMessage(
            id=str(uuid4()),
            type='message',
            role='assistant',
            content=[TextBlock(type='text', text=text)],
            model='<synthetic>',
            stop_reason='end_turn',
            stop_sequence=None,
            usage=BetaUsage()
        ),
        cost_usd=0.0,
        duration_ms=0,
        type='assistant',
        uuid=uuid4()
    )


def format_system_prompt_with_context(
    system_prompt: List[str],
    context: Dict[str, str]
) -> str:
    """Format system prompt with context"""
    parts = list(system_prompt)

    if context:
        context_parts = []
        for key, value in context.items():
            if value:
                context_parts.append(f"<{key}>{value}</{key}>")
        if context_parts:
            parts.append("\n".join(context_parts))

    return "\n\n".join(parts)


# === Main Query Function ===

async def _compact_before_recursion(head, tail, tool_use_context,
                                    context_proven_fatal: bool = False):
    """Compact before a recursion, exactly as the tool-result path does.

    Every recursion in this file appends to the history and calls back into
    query(), but only the tool-result path ever asked whether the history had
    outgrown the window first. The recovery paths — repetition, empty turn,
    tool-parse failure, the repeated-call cap, max_tokens, Stop-hook blocking,
    goal gates — recursed straight past that check, and each of them appends TWO
    messages (the failed assistant turn plus a nudge) to a history that was
    already at the threshold. A turn that collapsed at 158k therefore made its
    next attempt at 160k, then 162k, with the collapse getting more likely at
    every step. Measured 2026-09-28: 158k -> 186k across one morning, no
    compaction at any point, every turn ending in a degenerate emission.

    `head` is summarized when compaction runs; `tail` is appended AFTERWARDS and
    survives. The nudge belongs in the tail — a full compaction replaces the
    whole conversation with a single summary message, so a nudge inside `head` is
    a recursion with no instruction left in it, which is worse than not
    compacting at all.
    """
    next_messages = list(head)
    _tracking = getattr(tool_use_context, '_autocompact_tracking', None) or {}
    _compact = await auto_compact_if_needed(
        next_messages, tool_use_context, _tracking,
        context_proven_fatal=context_proven_fatal)
    if 'consecutive_failures' in _compact:
        _tracking['consecutive_failures'] = _compact['consecutive_failures']
        try:
            tool_use_context._autocompact_tracking = _tracking
        except Exception:
            pass
    if _compact.get('was_compacted'):
        next_messages = list(_compact['compaction_result'].messages)
        # The summary goes into the recursion and dies with it, exactly like a
        # microcompaction -- but this one the REPL cannot replay, because a
        # summary is an API round trip and not a function of the history. So
        # record that a summary is owed and let the turn boundary make it.
        #
        # Nothing was raised here at all before. The REPL then carried the FULL
        # history into the next turn while the usage anchor described the
        # summarized prompt -- a few thousand tokens against four hundred
        # thousand -- so the gauge read near-empty while every request was
        # refused for length: the same defect as the microcompaction one, with
        # the divergence an order of magnitude wider.
        _ns = getattr(tool_use_context, '_needs_summary', None)
        if isinstance(_ns, dict):
            _ns['raised'] = True
        print('  ⎿  Auto compacted — conversation summarized. Continuing...\n')
    elif _compact.get('was_microcompacted'):
        next_messages = list(_compact['messages'])
        # This set goes into the recursion and is then dropped: the history
        # belongs to the REPL and nothing here can replace it. So tell the REPL
        # to redo the reduction where it does stick — for EVERY microcompaction.
        #
        # This was once gated on context_proven_fatal, reasoning that the other
        # eight call sites reduce a HEALTHY history and "the turn-boundary check
        # reaches it on its own terms". It cannot. The mid-turn requests go out
        # TRIMMED, so the usage anchor describes the slim prompt while this
        # history is still the fat one; should_auto_compact then weighs the slim
        # number and declines. Worse, the only writer of _proven_fatal is the
        # repetition recovery, so once collapses stopped the gate never opened
        # again and ordinary reductions were dropped every time.
        #
        # Measured in the Rust port 2026-10-03 18:07: message 7 returned at its
        # original 17,958 bytes — the same hash it carried before the 17:55
        # reduction cleared it to 114 — the gauge read 262k, and the server was
        # handed 458,753 input + 65,536 reserved output. 524,289 against a
        # 524,288 window: 400.
        _nr = getattr(tool_use_context, '_needs_reduction', None)
        if isinstance(_nr, dict):
            _nr['raised'] = True
        # _proven_fatal keeps its own, narrower meaning: the decoder died here.
        if context_proven_fatal:
            _flag = getattr(tool_use_context, '_proven_fatal', None)
            if isinstance(_flag, dict):
                _flag['raised'] = True
        print('  ⎿  Microcompacted — old tool results cleared, recent context kept.\n')
    return [*next_messages, *tail]


async def query(
    messages: List[Message],
    system_prompt: List[str],
    context: Dict[str, str],
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
    max_output_recovery_count: int = 0,
    repetition_recovery_count: int = 0,
    tool_parse_recovery_count: int = 0,
    empty_turn_recovery_count: int = 0,
    # Omit reasoning_effort for THIS request only. Set by the repetition branch
    # when it retries a collapsed turn; every other recursion leaves it at the
    # default, so reasoning returns the moment the model is producing content
    # again. The recovery COUNTER cannot stand in for this: it stays spent for
    # the rest of the turn, which would leave the model working blind through
    # every tool call after one hiccup.
    #
    # Measured 2026-09-13: eight consecutive repetition retries on one turn
    # each produced 2000 characters of a single repeated token on the reasoning
    # channel and ZERO characters of content, ~125s apiece. The nudge between
    # attempts is a user message and cannot reach the channel that is stuck, so
    # the final attempt — the one meant to make the model abandon this line and
    # carry on — produced no text either and the turn ended.
    suppress_reasoning: bool = False,
    stop_hook_active: bool = False,
    turn_count: int = 0,
) -> AsyncGenerator[Message, None]:
    """
    Main query loop that handles tool calls recursively.
    Equivalent to src/query.ts query()
    """
    # Snapshot the input list. The REPL consumes this generator with
    # `for m in query(messages): messages.append(m)` — i.e. it mutates the SAME
    # list we hold. Without this copy, `next_messages = [*messages, assistant,
    # *results]` below would re-include the assistant + tool_results the REPL
    # already appended during iteration, sending them to the model twice
    # (inflating tokens and auto-compact). Our own history is threaded explicitly
    # via next_messages, so a frozen entry snapshot is what we want.
    messages = list(messages)

    # maxTurns runaway guard (QueryParams.maxTurns): stop after N recursive tool
    # turns in one prompt. turn_count increments at each recursion; the first call
    # (0) always proceeds. 0 (or None) = no cap. Guards against a keep-going
    # loop whose arguments change every turn, which the two repetition guards
    # below cannot see because no call repeats exactly.
    from .utils.config import get_config as _get_config_mt
    _max_turns = _get_config_mt().max_turns
    if _max_turns is not None and _max_turns > 0 and turn_count >= _max_turns:
        yield create_assistant_message_obj(
            f"[Reached the maximum of {_max_turns} turns for this request. Stopping.]"
        )
        return

    full_system_prompt = format_system_prompt_with_context(system_prompt, context)

    # Check for abort signal
    if tool_use_context.abort_controller and tool_use_context.abort_controller.is_set():
        yield create_assistant_message_obj(INTERRUPT_MESSAGE)
        return

    # Query the model
    #
    # `/ab auto`: the turn's first request is the one deciding whether to
    # accept, and the only one that wants the refusal edit. On here, off as soon
    # as it answers (below) — the two are paired inside this request, so no
    # outcome can leave the switch on. See services/ab.py.
    from .services import ab as _ab
    _ab_edit = await _ab.begin_request()
    assistant_message = await query_model(
        messages=messages,
        system_prompt=[full_system_prompt],
        tools=tool_use_context.options.get('tools', []),
        signal=tool_use_context.abort_controller,
        options={
            'dangerously_skip_permissions': tool_use_context.options.get('dangerously_skip_permissions', False),
            'model': tool_use_context.options.get('slow_and_capable_model'),
            # NOTE (2026-09-28): every recovery recursion now passes
            # suppress_reasoning=False, so both of these are False for
            # conversation requests. Compaction sets its own no_reasoning
            # directly and is unaffected.
            #
            # Why: these two are the only things that differ between a normal
            # request and a recovery request, and BOTH change how the server
            # renders the prompt — keep_thinking through chat_template_kwargs
            # {clear_thinking, drop_thinking}, and no_reasoning through
            # reasoning_effort, which this deployment takes as a template kwarg
            # too. Either one re-renders the whole conversation, so the token
            # sequence diverges near the HEAD of the prompt and the server's
            # prefix cache misses completely. Measured on this install: recovery
            # requests logged cached=0 and "cold prefill: 164109 of 164109
            # prompt tokens must be recomputed (0.0% served from cache) — about
            # 123s before the first token", twice per repetition event (the
            # flags flip back on the next normal recursion). ~1.3M prefill
            # tokens to answer one nudge, and two minutes of a blank screen
            # indistinguishable from a hang.
            #
            # What it cost to stop: the thinking this comment used to preserve.
            # For the repetition path there is nothing left to preserve — the
            # collapsed generation is now cleared where the server raises it
            # (_openai_common), and that reasoning channel WAS the garbage. Low
            # effort did not help either: those recovery calls came back with
            # think=26 and collapsed anyway, half the time. If a future path
            # genuinely needs the cut turn's reasoning, inline it as text in the
            # appended nudge rather than flipping a flag that re-renders 160k
            # tokens. To revert: set these five recursions back to
            # suppress_reasoning=True.
            'no_reasoning': suppress_reasoning,
            # ...and keeps the thinking it has to resume FROM. Same flag, two
            # different moments; they are separate options because compaction
            # sets only the first (see the chat_template_kwargs gate).
            'keep_thinking': suppress_reasoning,
            # A sub-agent (agent_id set by the Agent tool / forked skills) may
            # run on a smaller output budget than the parent — see
            # config.subagent_max_tokens. 0 or unset leaves cfg.max_tokens.
            **({'max_tokens': _sub_max}
               if getattr(tool_use_context, 'agent_id', None) is not None
               and (_sub_max := int(getattr(_get_config_mt(), 'subagent_max_tokens', 0) or 0)) > 0
               else {}),
        },
        on_token_update=tool_use_context.on_token_update,
        on_content_delta=tool_use_context.on_content_delta,
        on_thinking_update=getattr(tool_use_context, 'on_thinking_update', None),
    )
    # Off, whatever came back: a tool call, a sentence, an empty turn, an abort,
    # or a generation the degenerate guard cut. Every recursion below —
    # repetition retries included — runs on the restored weights.
    await _ab.end_request(_ab_edit)

    # Check for abort after query
    if tool_use_context.abort_controller and tool_use_context.abort_controller.is_set():
        yield create_assistant_message_obj(INTERRUPT_MESSAGE)
        return

    # Layer-2 repetition recovery. _openai_common surfaces stop_reason
    # 'repetition' (the raw vLLM signal) instead of baking in end_turn; the
    # POLICY lives here. The degenerate emission's tool calls are discarded
    # (executing a call the model made while looping just extends the loop) and,
    # while the budget lasts, a nudge is injected so the model breaks out on its
    # own — the USER never sees a "stopped" note. stop_reason is normalized to
    # end_turn so the transcript stays standard, and the dangling tool_use
    # blocks are stripped so history stays balanced.
    _is_repetition = getattr(assistant_message.message, 'stop_reason', None) == 'repetition'
    if _is_repetition:
        assistant_message.message.content = [
            b for b in assistant_message.message.content
            if not (hasattr(b, 'type') and b.type == 'tool_use')
        ]
        assistant_message.message.stop_reason = 'end_turn'
        if repetition_recovery_count > REPETITION_LAST_RESORT_ATTEMPT:
            # Nudges AND the last-resort reduction are both spent: surface the
            # note in the yielded message and end.
            assistant_message.message.content.append(
                TextBlock(type='text', text=_REPETITION_GIVEUP_NOTE))

    # Layer-2 lost-tool-call recovery, shaped exactly like the repetition path
    # above: _openai_common surfaces the raw signal, the POLICY lives here.
    # There are no tool_use blocks to strip — their absence IS the signal — so
    # this only normalizes stop_reason so the transcript stays standard, and
    # appends the give-up note once the budget is spent. The note matters more
    # here than for repetition: a lost call leaves text that reads like work
    # about to happen, so silence is indistinguishable from success to anyone
    # reading it hours later.
    _is_empty_turn = (
        getattr(assistant_message.message, 'stop_reason', None) == 'empty_turn')
    if _is_empty_turn:
        assistant_message.message.stop_reason = 'end_turn'
        if empty_turn_recovery_count > EMPTY_TURN_RECOVERY_LIMIT:
            assistant_message.message.content.append(
                TextBlock(type='text', text=_EMPTY_TURN_GIVEUP_NOTE))

    _is_tool_parse_failure = (
        getattr(assistant_message.message, 'stop_reason', None) == 'tool_parse_failed')
    if _is_tool_parse_failure:
        assistant_message.message.stop_reason = 'end_turn'
        if tool_parse_recovery_count > TOOL_PARSE_RECOVERY_LIMIT:
            assistant_message.message.content.append(
                TextBlock(type='text', text=_TOOL_PARSE_GIVEUP_NOTE))

    # A stall guard cut the stream (_openai_common: stalled stream). Shaped like
    # the three above — _openai_common surfaces the raw signal, the POLICY lives
    # here — but the policy is deliberately different: a stall is NOT handed to
    # Layer-2 recovery. Nothing about a server that stopped sending suggests the
    # model is looping, and nudging it would spend the recovery budget on a
    # transport problem. The note naming the cause is already in the message
    # (_openai_common wrote it there, where the elapsed seconds are known), so
    # this only normalizes stop_reason to keep the transcript standard.
    #
    # Reaching here at all is the point: a stall used to set the turn's shared
    # abort event, so it came out of the is_aborted() check above as
    # "[Request interrupted by user]" and never got this far.
    if getattr(assistant_message.message, 'stop_reason', None) == 'stalled':
        assistant_message.message.stop_reason = 'end_turn'

    # Only when nothing will be dispatched: a truncated turn that still carries a
    # readable tool call continues normally and is not "stopped".
    _max_tokens_spent = (
        getattr(assistant_message.message, 'stop_reason', None) == 'max_tokens'
        and max_output_recovery_count >= MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
        and not any(
            hasattr(b, 'type') and b.type == 'tool_use'
            for b in assistant_message.message.content
        )
    )
    if _max_tokens_spent:
        assistant_message.message.content.append(
            TextBlock(type='text', text=_MAX_OUTPUT_GIVEUP_NOTE))

    yield assistant_message

    if _is_repetition:
        # <= , not < : the last attempt is the one that tells the model to
        # ABANDON this line of work and carry on with the rest of the task.
        # Ending the turn instead is the failure an unattended run cannot
        # survive — the session goes quiet with hours of work left and nobody
        # watching to restart it.
        # Against REPETITION_LAST_RESORT_ATTEMPT, not the raw limit, so the
        # ladder gets its full run of nudges AND one further attempt on a
        # reduced context before the turn is allowed to end. Ending it is the
        # worst available outcome, so it is the last thing tried, never the
        # second.
        if repetition_recovery_count <= REPETITION_LAST_RESORT_ATTEMPT:
            debug_log('REPETITION_RECOVERY', {
                'attempt': repetition_recovery_count + 1,
                'limit': REPETITION_RECOVERY_LIMIT,
            })
            _nudge = create_user_message(_repetition_recovery_prompt(messages, repetition_recovery_count))
            _nudge['isMeta'] = True
            yield _nudge
            # context_proven_fatal: only on the LAST attempt, and nowhere else
            # in this file. True here because this history has been
            # DEMONSTRATED to kill the decoder and every nudge has already
            # failed on it. False for the attempts before it, so the ~50% of
            # collapses a plain retry recovers keep their tool results; false at
            # the other recovery call sites because nothing has been
            # demonstrated there at all — an empty turn, a truncated tool
            # argument, a repeated call, a Stop hook or a failing gate each say
            # the model did something wrong, not that this context cannot be
            # decoded.
            _next = await _compact_before_recursion(
                [*messages, assistant_message], [_nudge], tool_use_context,
                context_proven_fatal=(
                    repetition_recovery_count == REPETITION_LAST_RESORT_ATTEMPT))
            async for msg in query(
                _next, system_prompt, context, can_use_tool, tool_use_context,
                max_output_recovery_count=max_output_recovery_count,
                repetition_recovery_count=repetition_recovery_count + 1,
                suppress_reasoning=False,
                tool_parse_recovery_count=tool_parse_recovery_count,
                empty_turn_recovery_count=empty_turn_recovery_count,
                stop_hook_active=stop_hook_active,
                turn_count=turn_count + 1,
            ):
                yield msg
        # budget spent (note already shown) OR recovery done: end the turn.
        return

    if _is_empty_turn:
        # <= for the same reason as the paths below: spend the last attempt
        # asking, not on ending the run in silence.
        if empty_turn_recovery_count <= EMPTY_TURN_RECOVERY_LIMIT:
            debug_log('EMPTY_TURN_RECOVERY', {
                'attempt': empty_turn_recovery_count + 1,
                'limit': EMPTY_TURN_RECOVERY_LIMIT,
            })
            _nudge = create_user_message(_EMPTY_TURN_RECOVERY_PROMPT)
            _nudge['isMeta'] = True
            yield _nudge
            _next = await _compact_before_recursion(
                [*messages, assistant_message], [_nudge], tool_use_context)
            async for msg in query(
                _next, system_prompt, context, can_use_tool, tool_use_context,
                max_output_recovery_count=max_output_recovery_count,
                # 빈 턴 회복: 추론만 하고 내용을 못 낸 실패라 붕괴와 같은 계열이다.
                # 회복 요청은 low 로 — 추론이 막힌 상태면 회복 자체가
                # 같은 자리에서 무너진다 (2026-09-13 실측).
                suppress_reasoning=False,
                repetition_recovery_count=repetition_recovery_count,
                tool_parse_recovery_count=tool_parse_recovery_count,
                empty_turn_recovery_count=empty_turn_recovery_count + 1,
                stop_hook_active=stop_hook_active,
                turn_count=turn_count + 1,
            ):
                yield msg
        # budget spent (note already shown) OR recovery done: end the turn.
        return

    if _is_tool_parse_failure:
        # <= for the same reason as the repetition path above: spend the last
        # attempt telling the model to move on, not on ending the run.
        if tool_parse_recovery_count <= TOOL_PARSE_RECOVERY_LIMIT:
            debug_log('TOOL_PARSE_RECOVERY', {
                'attempt': tool_parse_recovery_count + 1,
                'limit': TOOL_PARSE_RECOVERY_LIMIT,
            })
            _nudge = create_user_message(_TOOL_PARSE_RECOVERY_PROMPT)
            _nudge['isMeta'] = True
            yield _nudge
            _next = await _compact_before_recursion(
                [*messages, assistant_message], [_nudge], tool_use_context)
            async for msg in query(
                _next, system_prompt, context, can_use_tool, tool_use_context,
                max_output_recovery_count=max_output_recovery_count,
                # 도구 인자 파싱 실패 회복: 인자가 잘린 턴을 다시 시킨다.
                # 회복 요청은 low 로 — 추론이 막힌 상태면 회복 자체가
                # 같은 자리에서 무너진다 (2026-09-13 실측).
                suppress_reasoning=False,
                repetition_recovery_count=repetition_recovery_count,
                tool_parse_recovery_count=tool_parse_recovery_count + 1,
                empty_turn_recovery_count=empty_turn_recovery_count,
                stop_hook_active=stop_hook_active,
                turn_count=turn_count + 1,
            ):
                yield msg
        # budget spent (note already shown) OR recovery done: end the turn.
        return

    # Check for tool use
    tool_use_messages = [
        block for block in assistant_message.message.content
        if hasattr(block, 'type') and block.type == 'tool_use'
    ]

    # If no tool use, the turn would end here — but first check whether it
    # ended because the output was TRUNCATED at the token cap rather than the
    # model finishing. query.ts:1188-1234: on a max_output_tokens turn-end,
    # inject the resume nudge and re-query (up to MAX_OUTPUT_TOKENS_RECOVERY_LIMIT)
    # so the model picks up mid-thought instead of stalling. Detected via the
    # stop_reason the llama.cpp layer now maps from finish_reason='length'.
    if not tool_use_messages:
        _stop_reason = getattr(assistant_message.message, 'stop_reason', None)
        if (
            _stop_reason == 'max_tokens'
            and max_output_recovery_count < MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
        ):
            debug_log('MAX_OUTPUT_RECOVERY', {
                'attempt': max_output_recovery_count + 1,
                'limit': MAX_OUTPUT_TOKENS_RECOVERY_LIMIT,
            })
            # Yield the nudge as a real user message so it sits between the
            # truncated assistant turn and the continuation (preserving message
            # alternation in the caller's durable history). isMeta=True is what
            # hides it from the UI (matching query.ts:1228); the CLI's user
            # branch also only renders tool_result blocks, so a text-only meta
            # message stays in history without being displayed either way.
            # NOTE (substrate divergence): query.ts WITHHOLDS the
            # max_output_tokens error from the stream and only surfaces it if
            # recovery exhausts. On the llama.cpp substrate there is no separate
            # withheld error object — the truncated turn is an ordinary assistant
            # message (stop_reason=max_tokens) already yielded above — so the
            # port resumes from it directly rather than withholding/replaying.
            recovery_message = create_user_message(_MAX_OUTPUT_RECOVERY_PROMPT)
            recovery_message['isMeta'] = True
            yield recovery_message
            _next = await _compact_before_recursion(
                [*messages, assistant_message], [recovery_message], tool_use_context)
            async for msg in query(
                messages=_next,
                system_prompt=system_prompt,
                context=context,
                can_use_tool=can_use_tool,
                tool_use_context=tool_use_context,
                max_output_recovery_count=max_output_recovery_count + 1,
                # 출력 한도 회복: 추론이 예산을 먹고 한도에 닿은 경우가 있다.
                # 회복 요청은 low 로 — 추론이 막힌 상태면 회복 자체가
                # 같은 자리에서 무너진다 (2026-09-13 실측).
                suppress_reasoning=False,
                repetition_recovery_count=repetition_recovery_count,
                tool_parse_recovery_count=tool_parse_recovery_count,
                empty_turn_recovery_count=empty_turn_recovery_count,
                turn_count=turn_count + 1,
                # Preserve stop_hook_active across recovery too (the tool-result
                # recursion does): otherwise a max_tokens recovery inside a
                # stop-hook continuation resets the flag, and the eventual
                # end_turn looks like a "first stop" and can fire an extra
                # continuation the original turn would have suppressed.
                stop_hook_active=stop_hook_active,
            ):
                yield msg
            return

        # API-error guard (query.ts:1258-1265). If the turn ended because the
        # model call itself errored (connection/rate-limit/auth/etc.), do NOT
        # run Stop hooks: the model produced no real response, and a blocking
        # Stop hook would create a death spiral (error -> hook blocks -> retry
        # -> error -> …). The error path sets is_api_error_message on the
        # assistant message (llamacpp.py); the original returns {reason:'completed'}.
        if getattr(assistant_message, 'is_api_error_message', False):
            return

        # Goal completion gates. Before the turn ends, if the standing goal has
        # gates and auto-run is on, run them: a failure feeds its output back and
        # the turn continues (bounded per user turn by GOAL_GATE_RECOVERY_LIMIT),
        # so the agent iterates until the gates pass instead of stopping at a
        # self-declared "done". No goal / no gates / autorun off => no-op.
        try:
            from .utils.goal import get_goal as _gg, get_gates as _ggates, run_gates as _rgates
            from .utils.config import get_config as _gcfg
            from .utils.state import get_cwd as _gcwd
            if _gg() and _ggates() and getattr(_gcfg(), 'goal_gate_autorun', True):
                _gr = getattr(tool_use_context, '_gate_recovery', 0)
                if _gr < GOAL_GATE_RECOVERY_LIMIT:
                    _gres = await asyncio.to_thread(_rgates, _gcwd())
                    _failed = [(c, code, tail) for (c, ok, code, tail) in _gres if not ok]
                    if _failed:
                        tool_use_context._gate_recovery = _gr + 1
                        debug_log('GOAL_GATE_FAILED', {
                            'attempt': _gr + 1, 'limit': GOAL_GATE_RECOVERY_LIMIT,
                            'failed': len(_failed),
                        })
                        _gmsg = create_user_message(_format_gate_failures(
                            _gg(), _failed, _gr + 1, GOAL_GATE_RECOVERY_LIMIT))
                        _gmsg['isMeta'] = True
                        yield _gmsg
                        # CARRIED, not reset/defaulted. tool_parse was passed 0
                        # and max_output was not passed at all (so it defaulted
                        # to 0), which handed every gate attempt a fresh recovery
                        # budget: a model collapsing on the same gate could spend
                        # REPETITION_RECOVERY_LIMIT nudges per attempt, times
                        # GOAL_GATE_RECOVERY_LIMIT attempts, on one stuck turn.
                        # The budgets exist to end a turn the model cannot
                        # escape; a failing gate is evidence it has not escaped.
                        async for msg in query(
                            messages=await _compact_before_recursion(
                                [*messages, assistant_message], [_gmsg], tool_use_context),
                            system_prompt=system_prompt, context=context,
                            can_use_tool=can_use_tool, tool_use_context=tool_use_context,
                            stop_hook_active=stop_hook_active,
                            max_output_recovery_count=max_output_recovery_count,
                            repetition_recovery_count=repetition_recovery_count, turn_count=turn_count + 1,
                            tool_parse_recovery_count=tool_parse_recovery_count,
                            empty_turn_recovery_count=empty_turn_recovery_count,
                        ):
                            yield msg
                        return
                    # All gates pass — clear the counter for the next turn.
                    tool_use_context._gate_recovery = 0
        except Exception as _gerr:  # noqa: BLE001 - gates must never break the turn
            debug_log('GOAL_GATE_ERROR', {'error': str(_gerr)})

        # Stop hooks (query.ts:1267 -> stopHooks.ts handleStopHooks). The turn
        # would end here naturally. Give configured Stop hooks a chance to keep
        # the model working: a hook that BLOCKS (exit 2, or stdout
        # {"decision":"block","reason":...}) injects that feedback and we
        # recurse — the model resumes instead of stopping. A hook returning
        # {"continue":false} ends the turn (prevent_continuation). stop_hook_active
        # is threaded through so a hook can tell it is already inside a stop-hook
        # continuation and stop looping (the hook owns loop-termination, per
        # Claude Code's documented contract — there is no engine-side turn cap
        # here, matching the original). No configured Stop hook => unchanged
        # behavior (turn ends), exactly like Claude Code with no hook.
        from .utils.hooks import handle_stop_hooks, has_hook_for_event
        if has_hook_for_event('Stop'):
            _sr = await handle_stop_hooks(
                messages, assistant_message, tool_use_context, stop_hook_active
            )
            if _sr['prevent_continuation']:
                debug_log('STOP_HOOK_PREVENTED', {})
                return
            if _sr['blocking_errors']:
                debug_log('STOP_HOOK_BLOCKING', {'count': len(_sr['blocking_errors'])})
                for _bm in _sr['blocking_errors']:
                    yield _bm
                async for msg in query(
                    messages=await _compact_before_recursion(
                        [*messages, assistant_message], list(_sr['blocking_errors']),
                        tool_use_context),
                    system_prompt=system_prompt,
                    context=context,
                    can_use_tool=can_use_tool,
                    tool_use_context=tool_use_context,
                    stop_hook_active=True,
                    # Carried: omitting it defaulted the count to 0, so a
                    # stop-hook continuation re-armed the max_tokens budget.
                    max_output_recovery_count=max_output_recovery_count,
                    repetition_recovery_count=repetition_recovery_count,
                    tool_parse_recovery_count=tool_parse_recovery_count,
                    empty_turn_recovery_count=empty_turn_recovery_count,
                    turn_count=turn_count + 1,
                ):
                    yield msg
        return

    # Repeated-tool-call cap (see REPEATED_TOOL_CALL_LIMIT). Checked before the
    # permission/execution path, so the batch that trips it never runs.
    _batch_sig = _tool_batch_signature(assistant_message)
    if _batch_sig is not None:
        # +1 for the batch we are holding, which is not in `messages` yet.
        _repeat_count = _count_repeated_tool_batches(messages, _batch_sig) + 1
        if _repeat_count >= REPEATED_TOOL_CALL_LIMIT:
            _names = ', '.join(
                sorted({getattr(tu, 'name', '') or '?' for tu in tool_use_messages})
            )
            debug_log('REPEATED_TOOL_CALL_CAPPED', {
                'tools': _names, 'consecutive': _repeat_count,
                'limit': REPEATED_TOOL_CALL_LIMIT,
            })
            print(
                f'  ⎿  Stopped — the same {_names} call was repeated '
                f'{_repeat_count} times. Not running it again.\n'
            )
            # DIVERGENCE FROM THE RUST PORT, deliberate and kept.
            #
            # Rust decides this cap BEFORE yielding the assistant message, so it
            # can strip the tool_use blocks off and fold the note into that same
            # message — one message, no pairing to answer. Here the message went
            # out ~170 lines earlier and is already in the caller's history and
            # on disk, so the blocks cannot be taken back; the only correct move
            # left is to answer them, which is what this does.
            #
            # The two therefore write different transcripts for the same event.
            # That was reviewed and left alone: both are self-consistent, each
            # loads fine in the other on --resume, and the model-visible ending
            # text is byte-identical (verified). Moving the decision above the
            # yield to match would restructure a hot path for a cosmetic result,
            # with the failure mode being the orphaned tool_use that poisons a
            # whole session — the exact bug this block exists to prevent.
            # One tool_result per tool_use before ending the turn, for the same
            # reason the nudge path below does it. `assistant_message` was
            # already yielded above and is in the caller's history and on disk
            # with N tool_use blocks; ending here without answering them leaves
            # the pairing broken forever. Nothing downstream repairs it —
            # _manage_context only strips orphan tool RESULTS at the head, and
            # its comment that assistant tool_calls cannot be orphaned the other
            # way is exactly what this path violates — so every later request in
            # the session, and after /resume, is rejected by the backend.
            for _tu in tool_use_messages:
                yield create_user_message([{
                    'type': 'tool_result',
                    'content': (
                        f'Stopped: this call was repeated {_repeat_count} times in a '
                        f'row and was not run again.'
                    ),
                    'is_error': True,
                    'tool_use_id': _tu.id,
                }])
            # Blocking the call is right (identical arguments cannot produce a
            # new answer), but ENDING THE TURN here is not: an unattended run
            # dies mid-task over one stuck step it could have walked around.
            # Share the repetition budget — same class of failure — and hand the
            # model a directive instead of a full stop.
            if repetition_recovery_count <= REPETITION_RECOVERY_LIMIT:
                _cap_nudge = create_user_message(
                    f'The same {_names} call was issued {_repeat_count} times in a row '
                    f'with identical arguments, so it was blocked rather than run again '
                    f'— an identical call cannot return a different answer. Do not '
                    f'reissue it. Either change the arguments substantively, use a '
                    f'different tool, or state your conclusion from what you already '
                    f'have ("this could not be determined" is a valid conclusion) and '
                    f'continue with the rest of the task.'
                )
                _cap_nudge['isMeta'] = True
                yield _cap_nudge
                _next = await _compact_before_recursion(
                    [*messages, assistant_message], [_cap_nudge], tool_use_context)
                async for msg in query(
                    _next, system_prompt, context, can_use_tool, tool_use_context,
                    max_output_recovery_count=max_output_recovery_count,
                    # 동일 도구 반복 회복: repetition 예산을 공유하므로 여기서 붕괴가 나면 토큰 붕괴 쪽 예산까지 태운다.
                    # 회복 요청은 low 로 — 추론이 막힌 상태면 회복 자체가
                    # 같은 자리에서 무너진다 (2026-09-13 실측).
                    suppress_reasoning=False,
                    repetition_recovery_count=repetition_recovery_count + 1,
                    tool_parse_recovery_count=tool_parse_recovery_count,
                    empty_turn_recovery_count=empty_turn_recovery_count,
                    stop_hook_active=stop_hook_active,
                    turn_count=turn_count + 1,
                ):
                    yield msg
            return

        if _repeat_count >= REPEATED_TOOL_CALL_NUDGE:
            _names = ', '.join(
                sorted({getattr(tu, 'name', '') or '?' for tu in tool_use_messages})
            )
            debug_log('REPEATED_TOOL_CALL_NUDGED', {
                'tools': _names, 'consecutive': _repeat_count,
                'nudge': REPEATED_TOOL_CALL_NUDGE,
                'limit': REPEATED_TOOL_CALL_LIMIT,
            })
            print(
                f'  ⎿  Skipped — the same {_names} call, {_repeat_count} times '
                f'in a row. Asked for a different approach.\n'
            )
            # One per tool_use, or the tool_use <-> tool_result pairing breaks
            # and the next request is malformed. is_error, because from the
            # model's side the call did not produce what it asked for.
            _nudges = [
                create_user_message([{
                    'type': 'tool_result',
                    'content': _REPEATED_TOOL_CALL_NUDGE_PROMPT.format(n=_repeat_count),
                    'is_error': True,
                    'tool_use_id': tu.id,
                }])
                for tu in tool_use_messages
            ]
            for _nm in _nudges:
                yield _nm
            # Same continuation shape as the stop-hook blocking path above: the
            # turn goes on, with the warning where the results would have been.
            async for msg in query(
                messages=await _compact_before_recursion(
                    [*messages, assistant_message], list(_nudges), tool_use_context),
                system_prompt=system_prompt,
                context=context,
                can_use_tool=can_use_tool,
                tool_use_context=tool_use_context,
                stop_hook_active=stop_hook_active,
                # Carried, as the Rust port does: omitting it defaulted the
                # count to 0 and re-armed the max_tokens budget every time a
                # repeated call was nudged.
                max_output_recovery_count=max_output_recovery_count,
                repetition_recovery_count=repetition_recovery_count,
                tool_parse_recovery_count=0,
                empty_turn_recovery_count=0,
                turn_count=turn_count + 1,
            ):
                yield msg
            return

    tool_results: List[Dict[str, Any]] = []

    # Run tools concurrently only when ALL are concurrency-safe (the scheduler
    # in toolOrchestration.ts batches consecutive concurrency-safe calls);
    # otherwise serialize. isConcurrencySafe defaults to is_read_only but tools
    # like TaskStop are safe without being read-only.
    tools = tool_use_context.options.get('tools', [])
    all_concurrency_safe = all(
        _is_tool_concurrency_safe(block.name, tools, getattr(block, 'input', None))
        for block in tool_use_messages
    )

    try:
        if all_concurrency_safe:
            async for message in run_tools_concurrently(
                tool_use_messages,
                assistant_message,
                can_use_tool,
                tool_use_context,
            ):
                yield message
                if isinstance(message, dict) and message.get('type') == 'user':
                    tool_results.append(message)
        else:
            async for message in run_tools_serially(
                tool_use_messages,
                assistant_message,
                can_use_tool,
                tool_use_context,
            ):
                yield message
                if isinstance(message, dict) and message.get('type') == 'user':
                    tool_results.append(message)
    except Exception as _tool_exc:
        # The pairing guarantee below only runs if the loops finish normally, so
        # an exception escaping a tool generator (generators.py re-raises one
        # carried as a value) skipped it entirely: the assistant tool_use blocks
        # are already in the caller's history and on disk, and none get results.
        # That is the same permanent session poisoning as the repeated-call cap,
        # by a second route. Answer what is unanswered, then let the exception
        # continue — the caller still gets to see and report it.
        #
        # Only Exception: GeneratorExit and CancelledError mean we are being
        # closed or cancelled, and yielding there raises instead of helping.
        _answered = {
            b.get('tool_use_id')
            for r in tool_results if isinstance(r, dict)
            for b in (r.get('message', {}).get('content', []) or [])
            if isinstance(b, dict) and b.get('tool_use_id')
        }
        for _tu in tool_use_messages:
            if _tu.id not in _answered:
                yield create_user_message([{
                    'type': 'tool_result',
                    'content': f'Tool execution failed: {_tool_exc}',
                    'is_error': True,
                    'tool_use_id': _tu.id,
                }])
        raise

    # Guarantee tool_use <-> tool_result pairing (yieldMissingToolResultBlocks,
    # query.ts:123). Any tool_use that produced no result — dropped, progress-only,
    # or aborted mid-flight — gets a synthetic error tool_result, so the next
    # query never sends an orphan tool_use (which the model server rejects).
    aborted = bool(
        tool_use_context.abort_controller and tool_use_context.abort_controller.is_set()
    )
    answered_ids = set()
    for r in tool_results:
        msg = r.get('message', {}) if isinstance(r, dict) else {}
        for block in (msg.get('content', []) or []):
            tid = block.get('tool_use_id') if isinstance(block, dict) else None
            if tid:
                answered_ids.add(tid)
    # The TS yieldMissingToolResultBlocks is only ever called with a concrete
    # reason (abort -> 'Interrupted by user', or a model/runtime error string);
    # it never invents a "looks normal" filler. On abort we reuse the verbatim
    # 'Interrupted by user'. A non-abort orphan here means a tool produced no
    # result outside any known error/abort path — an internal anomaly. We still
    # fill it (so the next request has no orphan tool_use that would 400), but
    # with a neutral, non-confident message and a debug_log so the underlying
    # bug is traceable rather than masked as normal operation.
    for tu in tool_use_messages:
        if tu.id not in answered_ids:
            if aborted:
                missing_error = 'Interrupted by user'
            else:
                missing_error = 'Tool did not return a result.'
                debug_log('TOOL_RESULT_ORPHAN', {
                    'tool_use_id': tu.id,
                    'tool_name': getattr(tu, 'name', None),
                    'note': 'tool_use had no matching tool_result outside abort/error path',
                })
            missing = create_user_message([{
                'type': 'tool_result',
                'content': missing_error,
                'is_error': True,
                'tool_use_id': tu.id,
            }])
            yield missing
            tool_results.append(missing)

    # Check for abort after tool execution
    if aborted:
        yield create_assistant_message_obj(INTERRUPT_MESSAGE_FOR_TOOL_USE)
        return

    # Sort tool results to match order of tool use messages
    def get_tool_use_id(r):
        content = r.get('message', {}).get('content', [])
        if content and isinstance(content, list) and len(content) > 0:
            return content[0].get('tool_use_id', '')
        return ''

    ordered_results = sorted(
        tool_results,
        key=lambda r: next(
            (i for i, tu in enumerate(tool_use_messages) if tu.id == get_tool_use_id(r)),
            len(tool_use_messages)
        )
    )

    # Auto-compact before recursive query. When the running token
    # count crosses the threshold, the conversation is summarized and the whole
    # history is replaced by that summary.
    # The consecutive-failure tracking is threaded across turns on the shared
    # tool_use_context inside the helper, so the
    # MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES circuit breaker can actually trip
    # (autoCompact.ts threads this through State at every continue site;
    # previously the returned count was discarded so the breaker never
    # accumulated).
    next_messages = await _compact_before_recursion(
        [*messages, assistant_message, *ordered_results], [], tool_use_context)

    # Mid-turn notification drain (query.ts mid-turn gate, ~lines 1550-1643):
    # surface background-task completions that landed during this turn by
    # injecting their <task-notification> blocks into the next turn. Each block
    # is BOTH appended to next_messages (so this turn's recursive model call sees
    # it) AND yielded (so the caller's durable history — cli.repl_loop's
    # `messages`, built from yields — keeps it across user turns; the source
    # pushes them onto toolResults, which becomes the next state's messages).
    # Yielded once here; the recursive query won't re-yield a pre-existing input
    # message, so the model sees each exactly once. Appended AFTER auto-compact
    # so a fresh completion isn't summarized away.
    #
    # The REPL also drains between user prompts (cli.repl_loop) for completions
    # that arrive while the model is idle. A single-shot `-p` run that ends before
    # a background task finishes is covered by print_runner's waitingForAgents
    # loop (post-final-turn completions).
    # Only the MAIN thread drains. A sub-agent runs this same query() code, and
    # the notification queue is process-global and untagged, so draining inside a
    # sub-agent would let it SWALLOW the main conversation's background-task
    # completions (they'd never reach the user). The original scopes by agentId
    # (main = agentId undefined); with an untagged queue, gating on "not a
    # sub-agent" is the faithful equivalent. Sub-agent-spawned completions are
    # surfaced by the main loop's next drain instead.
    if not getattr(tool_use_context, 'agent_id', None):
        from .utils.message_queue_manager import drain_pending_notifications
        for _block in drain_pending_notifications():
            _notif = create_user_message(_block)
            next_messages = [*next_messages, _notif]
            yield _notif

    # === Mid-turn message to a sub-agent (SendMessage) ===
    # The mirror of the block above, and the exact opposite gate: ONLY a
    # sub-agent drains here, and only its own queue. The parent queued this with
    # SendMessage while the agent was already mid-turn, so it could not be
    # handed to the agent's message list directly — the loop running the agent
    # owns that list. This is the tool-round boundary the queue was waiting for.
    #
    # Nothing can be stolen the way an untagged notification queue can be: the
    # queue lives on the agent's own task, so the only reader is the agent the
    # message was addressed to.
    #
    # The source drains it inside its per-turn attachment builder, which runs
    # before every turn including a sub-agent's; this is codemason's equivalent
    # per-turn injection point. The framing is wrapCommandText's 'coordinator'
    # case verbatim — it has to say not just what arrived but when to deal with
    # it, or a mid-task agent treats a new instruction as idle context.
    else:
        from .state.app_state import get_app_state as _get_app_state
        from .state.app_state import set_app_state as _set_app_state
        from .tasks.local_agent_task import drain_pending_messages
        from .utils.messages import wrap_command_text, wrap_messages_in_system_reminder
        _get = getattr(tool_use_context, 'get_app_state', None) or _get_app_state
        _set = getattr(tool_use_context, 'set_app_state', None) or _set_app_state
        for _queued in drain_pending_messages(
            getattr(tool_use_context, 'agent_id', None), _get, _set
        ):
            _sent = create_user_message(wrap_command_text(_queued, 'coordinator'))
            _sent = wrap_messages_in_system_reminder([_sent])[0]
            _sent['isMeta'] = True
            next_messages = [*next_messages, _sent]
            yield _sent

    # === Mid-turn user steering (interactive REPL) ===
    # A line the user typed WHILE this turn was running. The MAIN turn injects it
    # here — the same gate the task-notification drain uses — so the model sees it
    # at its very next step (true mid-turn interjection, no abort). Appended AFTER
    # auto-compact so a line typed right before the threshold isn't summarized
    # away, and yielded once so the caller's durable history keeps it. Each
    # injected line is dropped from the REPL's parked-input display and echoed to
    # scrollback via the router; anything still queued when a turn ends before
    # another step could inject it is reconciled by the REPL's turn-boundary
    # handler.
    #
    # A SUB-AGENT must never CONSUME the user's steering (it belongs to the main
    # conversation, and the queue is process-global and untagged). But a long
    # sub-agent would otherwise hold the main turn — and the user's message —
    # until it fully finished. So when a line is waiting, the sub-agent peeks (does
    # not drain) and wraps up cooperatively: one instruction to summarize now, then
    # it ends and the main turn reaches this gate and delivers the interjection.
    from .utils.steering_queue import drain_steering_messages, has_steering_messages
    if not getattr(tool_use_context, 'agent_id', None):
        _steer_router = getattr(tool_use_context, 'input_router', None)
        _steered = False
        for _steer_text in drain_steering_messages():
            _steer_msg = create_user_message(_steer_text)
            next_messages = [*next_messages, _steer_msg]
            yield _steer_msg
            _steered = True
            if _steer_router is not None:
                try:
                    _steer_router.note_steer_injected(_steer_text)
                except Exception:
                    pass
        # One note per drain, after the interjections, so the model knows the
        # turn is still live. Appended, so the cached prefix is intact.
        if _steered:
            _resume = create_user_message(_MAIN_STEER_RESUME)
            _resume['isMeta'] = True
            next_messages = [*next_messages, _resume]
            yield _resume
    elif has_steering_messages():
        if not getattr(tool_use_context, '_steer_wrapup', False):
            # First time we notice the interjection: ask this sub-agent to finish.
            try:
                tool_use_context._steer_wrapup = True
            except Exception:
                pass
            _wrap = create_user_message(_AGENT_STEER_WRAPUP)
            _wrap['isMeta'] = True
            next_messages = [*next_messages, _wrap]
            yield _wrap
        else:
            # Already asked to wrap up, yet it called tools again — force-stop so
            # the main turn is not held any longer. The Agent tool finalizes on the
            # last assistant text (or a synthetic "Done (N tool uses)" summary).
            return

    # Recursive query with tool results. Preserve stop_hook_active: once inside a
    # stop-hook continuation, it must survive subsequent tool turns, or the next
    # Stop-hook check sees it as False ("first stop") and can loop forever.
    async for msg in query(
        messages=next_messages,
        system_prompt=system_prompt,
        context=context,
        can_use_tool=can_use_tool,
        tool_use_context=tool_use_context,
        stop_hook_active=stop_hook_active,
        # Reset, like tool_parse below. The budget is there to end a turn the
        # model cannot escape, not to count the times it escaped: a tool-using
        # turn means it produced content and a well-formed call, so whatever it
        # was stuck on is behind it.
        #
        # Carrying the counts forward instead made a working recovery fail by
        # succeeding. Observed 2026-09-13: a session collapsed and recovered
        # eight separate times, each on the first retry, and then stopped with
        # "the model kept repeating itself across several attempts and could not
        # break out of the loop" — the one outcome an unattended run cannot
        # survive, reached by a guard that had worked every single time.
        repetition_recovery_count=0,
        tool_parse_recovery_count=0,
        empty_turn_recovery_count=0,
        # NOT reset, and previously not passed at all — which defaulted it to 0
        # and re-armed the max_tokens budget on every tool turn. The three above
        # are reset because a well-formed tool call proves the model escaped what
        # it was stuck on; hitting the OUTPUT limit proves nothing of the kind,
        # and a long answer can hit it turn after turn. The Rust port carries it
        # here for the same reason.
        max_output_recovery_count=max_output_recovery_count,
        turn_count=turn_count + 1,
    ):
        yield msg


def _is_tool_concurrency_safe(tool_name: str, tools: List[Type[Tool]],
                              input_data=None) -> bool:
    """Whether a tool call may run concurrently (tool.isConcurrencySafe).

    Resolves by name or alias. Mirrors the source's conservative fallback: if
    isConcurrencySafe raises (e.g. shell-quote parse failure), treat as not safe.
    """
    for tool in tools:
        if tool.matches_name(tool_name):
            try:
                inp = input_data if isinstance(input_data, dict) else None
                return tool.is_concurrency_safe(inp)
            except Exception:  # noqa: BLE001 - be conservative on error
                return False
    # Unknown/unresolved tool: fail closed (source findToolByName -> undefined ->
    # isConcurrencySafe=false). It errors independently in run_tool_use anyway;
    # treating it as unsafe just keeps a mixed batch serial.
    return False


# === Tool Execution ===

async def run_tools_concurrently(
    tool_use_messages: List[ToolUseBlock],
    assistant_message: AssistantMessage,
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> AsyncGenerator[Message, None]:
    """Run multiple tools concurrently. Equivalent to src/query.ts runToolsConcurrently()"""
    sibling_ids = set(tu.id for tu in tool_use_messages)

    # Two identical calls in ONE turn run at the same time, so neither can see
    # the other's result and the history check waves both through. They cannot
    # return different things, so keep the first and answer the rest here —
    # decided up front, before anything is dispatched.
    unique: List[ToolUseBlock] = []
    duplicates: List[ToolUseBlock] = []
    seen_keys = set()
    for tool_use in tool_use_messages:
        key = _tool_call_key(tool_use.name,
                             tool_use.input if isinstance(tool_use.input, dict) else {})
        if key in seen_keys:
            duplicates.append(tool_use)
        else:
            seen_keys.add(key)
            unique.append(tool_use)

    for tool_use in duplicates:
        debug_log('TOOL_REPEAT_BLOCKED', {'tool': tool_use.name, 'reason': 'same_batch'})
        yield create_user_message([{
            'type': 'tool_result',
            'content': (
                f"Blocked: an identical {tool_use.name} call with these exact "
                f"arguments was already issued in this same turn. Duplicating it "
                f"cannot return anything different — use the other call's result."
            ),
            'is_error': True,
            'tool_use_id': tool_use.id,
        }])

    generators = [
        run_tool_use(
            tool_use,
            sibling_ids,
            assistant_message,
            can_use_tool,
            tool_use_context,
        )
        for tool_use in unique
    ]

    async for message in all_generators(generators, MAX_TOOL_USE_CONCURRENCY):
        yield message


async def run_tools_serially(
    tool_use_messages: List[ToolUseBlock],
    assistant_message: AssistantMessage,
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> AsyncGenerator[Message, None]:
    """Run tools one at a time. Equivalent to src/query.ts runToolsSerially()"""
    sibling_ids = set(tu.id for tu in tool_use_messages)

    for tool_use in tool_use_messages:
        async for message in run_tool_use(
            tool_use,
            sibling_ids,
            assistant_message,
            can_use_tool,
            tool_use_context,
        ):
            yield message


# === Repeated-tool-call guard ===
#
# A model that loses its own reasoning between turns (or simply refuses to give
# up on a dead end) re-issues the identical call and gets the identical answer,
# learning nothing while the context fills. Observed in the field: five straight
# rounds of one curl whose three URLs all returned the same 8804 bytes, ending
# only when the server's repetition detector cut the output.
#
# The guard blocks a call only when BOTH the arguments and the previous result
# repeat, so an honest retry — rebuild, re-read a file being edited, poll a job —
# never trips it: its result changes, which resets the counter.
_TOOL_REPEAT_HISTORY: Dict[str, Dict[str, Any]] = {}
_TOOL_REPEAT_LIMIT = 2   # identical (args, result) pairs tolerated; the next
                         # attempt is refused


def reset_tool_repeat_history() -> None:
    """Forget every recorded call. Called when the conversation resets."""
    _TOOL_REPEAT_HISTORY.clear()


# Fields that label a call for the human reading the transcript but do not
# change what it does. They must not enter the identity of a call: a model
# repeating one command numbers its own attempts ("1st curl request", "2nd curl
# request"), which would make every repeat look like a brand-new call and defeat
# the guard entirely.
_TOOL_KEY_IGNORED_FIELDS = frozenset({'description'})


def _tool_call_key(tool_name: str, input_data: Any) -> str:
    if isinstance(input_data, dict):
        input_data = {k: v for k, v in input_data.items()
                      if k not in _TOOL_KEY_IGNORED_FIELDS}
    try:
        args = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        args = repr(input_data)
    return f"{tool_name}:{args}"


def _tool_result_digest(message: Any) -> Optional[str]:
    """Digest of a tool_result payload, or None if this message isn't one."""
    content = _msg_attr(_msg_attr(message, 'message', None), 'content', None)
    if not isinstance(content, list):
        return None
    for block in content:
        if _msg_attr(block, 'type') != 'tool_result':
            continue
        payload = _msg_attr(block, 'content', '')
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                payload = repr(payload)
        return hashlib.sha1(payload.encode('utf-8', 'replace')).hexdigest()
    return None


async def run_tool_use(
    tool_use: ToolUseBlock,
    sibling_tool_use_ids: Set[str],
    assistant_message: AssistantMessage,
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> AsyncGenerator[Message, None]:
    """Execute a single tool use. Equivalent to src/query.ts runToolUse()"""
    from .utils.log import log_error

    tool_name = tool_use.name
    tools = tool_use_context.options.get('tools', [])
    # Resolve by name OR alias (toolMatchesName) so deprecated names like
    # KillShell -> TaskStop / BashOutputTool -> TaskOutput keep working.
    tool = next((t for t in tools if t.matches_name(tool_name)), None)

    # Check if the tool exists
    if not tool:
        debug_log('TOOL_USE_ERROR', {'error': f'No such tool available: {tool_name}'})
        yield create_user_message([{
            'type': 'tool_result',
            'content': f'Error: No such tool available: {tool_name}',
            'is_error': True,
            'tool_use_id': tool_use.id,
        }])
        return

    tool_input = tool_use.input if isinstance(tool_use.input, dict) else {}

    # Refuse a call that has already been made with these exact arguments and
    # returned the same thing every time. Answering with the cached result would
    # keep the loop turning, so say plainly that the result cannot change.
    repeat_key = _tool_call_key(tool_name, tool_input)
    prior = _TOOL_REPEAT_HISTORY.get(repeat_key)
    if prior and prior.get('count', 0) >= _TOOL_REPEAT_LIMIT:
        # Refuse THIS attempt, then forget the streak. The refusal is a brake,
        # not a ban: whether the result would have changed is unknowable without
        # running the call, so a permanent block would strand any legitimate
        # retry — two failing builds, a fix, and the third build refused. A model
        # genuinely stuck gets braked again two calls later, so a real loop still
        # cannot run away.
        _TOOL_REPEAT_HISTORY.pop(repeat_key, None)
        debug_log('TOOL_REPEAT_BLOCKED', {
            'tool': tool_name,
            'count': prior.get('count'),
        })
        yield create_user_message([{
            'type': 'tool_result',
            'content': (
                f"Blocked: {tool_name} was already called {prior['count']} times "
                f"with these exact arguments, and every call returned the same "
                f"result. Running it again cannot produce anything new. "
                f"This approach is not working — change it: use different "
                f"arguments, a different tool, or state what is blocking you."
            ),
            'is_error': True,
            'tool_use_id': tool_use.id,
        }])
        return

    try:
        # Check for abort
        if tool_use_context.abort_controller and tool_use_context.abort_controller.is_set():
            yield create_user_message([{
                'type': 'tool_result',
                'content': 'Tool execution was cancelled',
                'is_error': True,
                'tool_use_id': tool_use.id,
            }])
            return

        async for message in check_permissions_and_call_tool(
            tool,
            tool_use.id,
            sibling_tool_use_ids,
            tool_input,
            tool_use_context,
            can_use_tool,
            assistant_message,
        ):
            digest = _tool_result_digest(message)
            if digest is not None:
                seen = _TOOL_REPEAT_HISTORY.get(repeat_key)
                if seen and seen.get('digest') == digest:
                    seen['count'] = seen.get('count', 1) + 1
                else:
                    # First time, or the result changed — the call is making
                    # progress, so start counting again.
                    _TOOL_REPEAT_HISTORY[repeat_key] = {'digest': digest, 'count': 1}
            yield message

    except Exception as e:
        log_error(e)
        yield create_user_message([{
            'type': 'tool_result',
            'content': format_error(e),
            'is_error': True,
            'tool_use_id': tool_use.id,
        }])


async def check_permissions_and_call_tool(
    tool: Type[Tool],
    tool_use_id: str,
    sibling_tool_use_ids: Set[str],
    input_data: Dict[str, Any],
    context: ToolUseContext,
    can_use_tool: CanUseToolFn,
    assistant_message: AssistantMessage,
) -> AsyncGenerator[Message, None]:
    """Check permissions and call tool. Equivalent to src/query.ts checkPermissionsAndCallTool()"""

    # Plan mode: block file-modifying tools while in plan mode (permission-mode
    # 'plan' denies edits until ExitPlanMode approval). The original permits the
    # write when a permission bypass is available, so honor
    # dangerously_skip_permissions as the same escape hatch and skip the hard
    # block in that case.
    #
    # DEVIATION (documented): the gate keys off class-level tool.is_read_only().
    # The source is INPUT-aware for Bash (BashTool.isReadOnly(input) →
    # checkReadOnlyConstraints), so read-only shell commands (`git log`, `ls`,
    # `rg`) are allowed during planning. codemason's BashTool.is_read_only() is
    # unconditionally False, so ALL Bash is blocked in plan mode — the model uses
    # Glob/Grep/Read (allowed) for exploration instead. Faithfully closing this
    # requires porting the ~4,700-line read-only-command validator
    # (readOnlyCommandValidation.ts/readOnlyValidation.ts/words.ts); a hardcoded
    # mini-allowlist is deliberately NOT done — misclassifying a write command as
    # read-only would let it run during plan mode (a security regression). Tracked
    # as a separate task.
    from .utils.plan_mode import is_plan_mode_active
    _PLAN_MODE_ALLOWED = {'EnterPlanMode', 'ExitPlanMode'}
    _plan_bypass = context.options.get('dangerously_skip_permissions', False)
    if (
        is_plan_mode_active()
        and not tool.is_read_only()
        and tool.name not in _PLAN_MODE_ALLOWED
        and not _plan_bypass
    ):
        yield create_user_message([{
            'type': 'tool_result',
            'content': (
                f'{tool.name} is blocked in plan mode — no file modifications are '
                'allowed. Finish exploring and designing, then call ExitPlanMode to '
                'present your plan for approval before coding.'
            ),
            'is_error': True,
            'tool_use_id': tool_use_id,
        }])
        return

    # Validate input
    validation_result = await tool.validate_input(input_data, context)
    if hasattr(validation_result, 'result') and not validation_result.result:
        debug_log('TOOL_USE_ERROR', {'error': validation_result.message})
        yield create_user_message([{
            'type': 'tool_result',
            'content': validation_result.message,
            'is_error': True,
            'tool_use_id': tool_use_id,
        }])
        return

    # Check permissions. The whole flow is bypassed under
    # dangerously_skip_permissions (faithful: the original never consults
    # checkPermissions in that mode). Otherwise the tool's own check_permissions
    # (Tool.ts checkPermissions) returns a 3-state decision:
    #   deny  -> hard-block with the tool's message (no prompt)
    #   ask   -> defer to the interactive prompt (can_use_tool), surfacing the
    #            tool's message first so the user knows WHY (e.g. a dangerous
    #            rm path) — this is where Bash's checkPathConstraints lands
    #   allow -> proceed without a prompt
    should_skip = context.options.get('dangerously_skip_permissions', False)
    if (not should_skip or tool.always_ask()) and tool.needs_permissions():
        perm = await tool.check_permissions(input_data, context)
        if getattr(perm, 'behavior', 'ask') == 'deny':
            yield create_user_message([{
                'type': 'tool_result',
                'content': perm.message or 'Permission denied',
                'is_error': True,
                'tool_use_id': tool_use_id,
            }])
            return
        if getattr(perm, 'behavior', 'ask') == 'ask':
            if perm.message:
                # Surface the security reason at the prompt (engine prints the
                # one-line notice the same way it announces auto-compaction).
                print(f'  {perm.message}')
            permission_result = await can_use_tool(tool, input_data, context, assistant_message)
            if not permission_result.get('result', True):
                yield create_user_message([{
                    'type': 'tool_result',
                    'content': permission_result.get('message', 'Permission denied'),
                    'is_error': True,
                    'tool_use_id': tool_use_id,
                }])
                return

    # Call the tool with a per-call context carrying this tool_use id (the
    # source clones toolUseContext with toolUseId per call). replace() shares the
    # mutable read_file_state dict + abort_controller, so cross-tool state and
    # interrupts still work; only tool_use_id differs.
    from dataclasses import replace as _replace
    call_context = _replace(context, tool_use_id=tool_use_id)
    # Wrap the tool's stream so ESC (abort_controller) cancels it mid-await instead
    # of only being noticed after the tool returns. This is the single dispatch
    # point every tool funnels through — serial, concurrent, and nested-agent
    # tools alike — so one wrap makes them all interruptible (see generators.abortable).
    _abort_signal = getattr(call_context, 'abort_controller', None)
    try:
        async for result in abortable(
            tool.call(input_data, call_context, can_use_tool), _abort_signal
        ):
            if result.type == 'result':
                result_content = result.result_for_assistant
                image_blocks = []
                if isinstance(result_content, list):
                    # Image blocks (Read on an image with a vision server) stay
                    # blocks; never str() them — that pastes the whole base64
                    # payload into the text (and trips the persist-to-disk cap).
                    image_blocks = [b for b in result_content
                                    if isinstance(b, dict) and b.get('type') == 'image']
                    result_content = '\n'.join(
                        block.get('text', str(block)) if isinstance(block, dict) else str(block)
                        for block in result_content
                        if not (isinstance(block, dict) and block.get('type') == 'image')
                    )

                # Persist oversized results to disk and replace with a
                # <persisted-output> preview (processToolResultBlock →
                # maybePersistLargeToolResult). Also injects a marker for empty
                # results so the model always has something to react to. Applied
                # only to successful results — error/abort blocks below are small.
                from .utils.tool_result_storage import process_tool_result_content
                content_str = process_tool_result_content(
                    str(result_content) if result_content is not None else '',
                    tool.name,
                    tool_use_id,
                    getattr(tool, 'max_result_size_chars', 100_000),
                )

                yield create_user_message([{
                    'type': 'tool_result',
                    'content': ([{'type': 'text', 'text': content_str}] + image_blocks
                                if image_blocks else content_str),
                    'tool_use_id': tool_use_id,
                    # The tool's structured Output, carried alongside the model-facing
                    # content (the original's message.toolUseResult). The display layer
                    # renders FROM this via Tool.render_tool_result_message; the API
                    # serializers only read type/content/tool_use_id, so it never
                    # reaches the model.
                    'data': result.data,
                }])
                return

            elif result.type == 'progress':
                # Forward progress message to CLI for display
                yield {
                    'type': 'progress',
                    'result_for_assistant': result.result_for_assistant if hasattr(result, 'result_for_assistant') else '',
                    'data': result.data if hasattr(result, 'data') else {},
                }

        # The tool generator finished without yielding a result. That's an
        # anomaly, not a success — the original synthesizes an is_error orphan
        # result here (never fabricates a passing "Tool completed"). Report it as
        # an error so the model doesn't believe a no-op tool succeeded.
        yield create_user_message([{
            'type': 'tool_result',
            'content': f'Error: {tool.name} returned no result.',
            'is_error': True,
            'tool_use_id': tool_use_id,
        }])

    except Exception as e:
        debug_log('TOOL_USE_ERROR', {'error': str(e)})
        yield create_user_message([{
            'type': 'tool_result',
            'content': format_error(e),
            'is_error': True,
            'tool_use_id': tool_use_id,
        }])


def format_error(error: Exception) -> str:
    """Format an error for display"""
    parts = [str(error)]

    if hasattr(error, 'stderr') and error.stderr:
        parts.append(str(error.stderr))
    if hasattr(error, 'stdout') and error.stdout:
        parts.append(str(error.stdout))

    full_message = '\n'.join(filter(None, parts))

    if len(full_message) <= 10000:
        return full_message

    # Truncate long messages
    half_length = 5000
    start = full_message[:half_length]
    end = full_message[-half_length:]
    return f"{start}\n\n... [{len(full_message) - 10000} characters truncated] ...\n\n{end}"
