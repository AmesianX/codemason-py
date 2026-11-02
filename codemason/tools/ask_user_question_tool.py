"""
AskUserQuestion Tool for CodeMason
Based on src/tools/AskUserQuestionTool/ from Claude Code

WHAT (schema, description, prompt, semantics) is ported verbatim from the
original. The HOW (the multiple-choice UI) is implemented on codemason's
terminal interface instead of the original's Ink/React component — the
terminal interface is part of the preserved local substrate.
"""

import os
import sys
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.tty_compat import termios

from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
    ValidationResultFailure,
)
from ..utils.formatting import colorize, Colors, safe_print
from ..utils.i18n import _


ASK_USER_QUESTION_TOOL_NAME = 'AskUserQuestion'
ASK_USER_QUESTION_TOOL_CHIP_WIDTH = 12

DESCRIPTION = (
    'Asks the user multiple choice questions to gather information, clarify '
    'ambiguity, understand preferences, make decisions or offer them choices.'
)

ASK_USER_QUESTION_TOOL_PROMPT = """Use this tool when you need to ask the user questions during execution. This allows you to:
1. Gather user preferences or requirements
2. Clarify ambiguous instructions
3. Get decisions on implementation choices as you work
4. Offer choices to the user about what direction to take.

Usage notes:
- Users will always be able to select "Other" to provide custom text input
- Use multiSelect: true to allow multiple answers to be selected for a question
- If you recommend a specific option, make that the first option in the list and add "(Recommended)" at the end of the label

When to ask:
- Ask at the START. If anything is unclear, or a decision is needed that only the user can make, ask it BEFORE beginning any work — everything you need, in one call. A question belongs at the point where the answer would still change what you do.
- Once the work itself is underway, avoid this tool. Stopping a running task to ask is almost never worth it: take the reasonable default, finish, and say plainly what you assumed and where. Ask mid-work only when the choice genuinely cannot be made without the user AND getting it wrong would destroy work or waste a large amount of it.
- Expect nobody to be watching. This is often started and left running for hours. A question is re-sent after 10 minutes and again after 5 more; if there is still no answer after 3 more, the recommended option is taken automatically and the work continues. So a question asked with nobody there costs 18 minutes and then answers itself — which is why it belongs at the start, and why your genuine recommendation must be the first option.

Plan mode note: In plan mode, use this tool to clarify requirements or choose between approaches BEFORE finalizing your plan. Do NOT use this tool to ask "Is my plan ready?" or "Should I proceed?" - use ExitPlanMode for plan approval. IMPORTANT: Do not reference "the plan" in your questions (e.g., "Do you have feedback about the plan?", "Does the plan look good?") because the user cannot see the plan in the UI until you call ExitPlanMode. If you need plan approval, use ExitPlanMode instead.
"""


# PREVIEW_FEATURE_PROMPT.markdown, ported verbatim. codemason renders preview
# text directly in the terminal (monospace/markdown-ish), so the markdown
# variant applies. Appended to prompt() when previews are supported.
PREVIEW_FEATURE_PROMPT_MARKDOWN = """
Preview feature:
Use the optional `preview` field on options when presenting concrete artifacts that users need to visually compare:
- ASCII mockups of UI layouts or components
- Code snippets showing different implementations
- Diagram variations
- Configuration examples

Preview content is rendered as markdown in a monospace box. Multi-line text with newlines is supported. When any option has a preview, the UI switches to a side-by-side layout with a vertical option list on the left and preview on the right. Do not use previews for simple preference questions where labels and descriptions suffice. Note: previews are only supported for single-select questions (not multiSelect).
"""


# === Input schema (ported from the zod schema) ===

class QuestionOption(BaseModel):
    label: str = Field(
        ...,
        description="The display text for this option that the user will see and "
        "select. Should be concise (1-5 words) and clearly describe the choice.",
    )
    description: str = Field(
        ...,
        description="Explanation of what this option means or what will happen if "
        "chosen. Useful for providing context about trade-offs or implications.",
    )
    preview: Optional[str] = Field(
        default=None,
        description="Optional preview content rendered when this option is focused. "
        "Use for mockups, code snippets, or visual comparisons that help users "
        "compare options.",
    )


class Question(BaseModel):
    question: str = Field(
        ...,
        description='The complete question to ask the user. Should be clear, '
        'specific, and end with a question mark. Example: "Which library should we '
        'use for date formatting?" If multiSelect is true, phrase it accordingly, '
        'e.g. "Which features do you want to enable?"',
    )
    header: str = Field(
        ...,
        description=f"Very short label displayed as a chip/tag (max "
        f"{ASK_USER_QUESTION_TOOL_CHIP_WIDTH} chars). Examples: \"Auth method\", "
        f"\"Library\", \"Approach\".",
    )
    options: List[QuestionOption] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="The available choices for this question. Must have 2-4 options. "
        "Each option should be a distinct, mutually exclusive choice (unless "
        "multiSelect is enabled). There should be no 'Other' option, that will be "
        "provided automatically.",
    )
    multiSelect: bool = Field(
        default=False,
        description="Set to true to allow the user to select multiple options instead "
        "of just one. Use when choices are not mutually exclusive.",
    )


class AskUserQuestionInput(ToolInputSchema):
    questions: List[Question] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="Questions to ask the user (1-4 questions)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for tracking and analytics purposes. Not "
        "displayed to user.",
    )


def _enable_terminal_echo() -> Optional[tuple]:
    """Best-effort: restore canonical mode + echo so input() is visible even when
    the REPL's ESC monitor put the terminal in cbreak mode. Returns prior settings."""
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] |= (termios.ECHO | termios.ICANON)
        termios.tcsetattr(fd, termios.TCSANOW, new)
        return (fd, old)
    except (termios.error, ValueError, OSError):
        return None


def _restore_terminal(state: Optional[tuple]) -> None:
    if not state:
        return
    fd, old = state
    try:
        termios.tcsetattr(fd, termios.TCSANOW, old)
    except (termios.error, ValueError, OSError):
        pass


class AskUserQuestionTool(Tool):
    """Ask the user multiple-choice questions on the terminal."""

    name = ASK_USER_QUESTION_TOOL_NAME
    input_schema = AskUserQuestionInput
    DESCRIPTION = DESCRIPTION

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        # AskUserQuestionTool.tsx draws its own question/options UI in call();
        # echoing the questions array here would print it twice.
        return ''

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """What the screen shows once the question is answered.

        The model-facing result is a sentence written for the model — "User has
        answered your questions: … You can now continue with the user's answers
        in mind." Printing it verbatim put a paragraph of English instructions in
        the middle of a Korean session, restating an answer the user had just
        given themselves. They know what they picked; what is worth one line is
        which answer landed against which question.
        """
        if not isinstance(data, dict):
            return ''
        answers = data.get('answers') or {}
        if not answers:
            return ''
        remote = set(data.get('remote_answers') or ())

        def _line(q, a):
            # Say where the answer came from only when it was not this terminal —
            # you already know what you typed here.
            return f'-> {a}' + (f'  {_("(remote)")}' if q in remote else '')

        if len(answers) == 1:
            q, a = next(iter(answers.items()))
            return _line(q, a)
        return '\n'.join(f'-> {q}: {a}' + (f'  {_("(remote)")}' if q in remote else '')
                         for q, a in answers.items())

    @classmethod
    async def description(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return ASK_USER_QUESTION_TOOL_PROMPT + PREVIEW_FEATURE_PROMPT_MARKDOWN

    @classmethod
    def is_read_only(cls) -> bool:
        return True

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def needs_permissions(cls) -> bool:
        # The tool IS the user interaction; no separate approval prompt.
        return False

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        # A friendly label for the tool-use header (the original's Ink UI labels
        # the question component; the terminal substrate shows this instead of an
        # empty `[]`). The interactive question body is rendered by call().
        return _('Question')

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        questions = input_data.get('questions', [])
        if not questions:
            return ValidationResultFailure(result=False, message='At least one question is required')
        # Uniqueness: question texts unique; option labels unique within a question.
        qtexts = [q.get('question') if isinstance(q, dict) else getattr(q, 'question', None) for q in questions]
        if len(qtexts) != len(set(qtexts)):
            return ValidationResultFailure(result=False, message='Question texts must be unique')
        for q in questions:
            opts = q.get('options', []) if isinstance(q, dict) else getattr(q, 'options', [])
            labels = [o.get('label') if isinstance(o, dict) else getattr(o, 'label', None) for o in opts]
            if len(labels) != len(set(labels)):
                return ValidationResultFailure(result=False, message='Option labels must be unique within each question')
        return ValidationResultSuccess()

    @staticmethod
    def _q_field(q, name, default):
        return q.get(name, default) if isinstance(q, dict) else getattr(q, name, default)

    # Sentinel: the phone answered, so this question is already recorded and the
    # local selection means nothing. Distinct from an empty selection, which is a
    # real local answer ("no answer").
    _ANSWERED_REMOTELY = object()

    # How long to wait before re-asking, and before deciding for ourselves.
    #
    # A question with nobody in front of it used to wait forever, which in the
    # mode this is mostly used in (-dsp, started and left for hours) means the
    # whole job is parked on a prompt no one will see. Waiting is right; waiting
    # forever is not. So: ask, re-ask after 10 minutes, re-ask after 5 more, and
    # after 3 more take the recommended option and keep going. Each step
    # re-sends to the phone, so someone who picks it up late still finds the
    # question waiting rather than having missed it once.
    from ..utils.escalation import ASK_LADDER_SECONDS as _ESCALATION_SECONDS

    @staticmethod
    def _recommended_option_index(labels: List[str]) -> int:
        """The option to take when nobody answers: the one marked recommended,
        else the first.

        Not a guess about which choice is "safest" in general — it is the
        convention this tool's own description asks callers to follow ("If you
        recommend a specific option, make that the first option in the list and
        add (Recommended) at the end of the label"). Falling back to the first
        option honours the same convention when the marker is absent.
        """
        for i, label in enumerate(labels):
            low = label.lower()
            if '(recommended)' in low or '(추천)' in label:
                return i
        return 0

    @classmethod
    def _escalation_total_minutes(cls) -> int:
        from ..utils.escalation import ask_total_minutes
        return ask_total_minutes()

    @classmethod
    async def _race_local_and_remote(cls, local_coro, qtext: str, header: str,
                                     labels: List[str], descs: List[str],
                                     remote_ok: bool, answers: Dict[str, str],
                                     closer=None, remote_marks=None):
        """Run the local picker and (when configured) the phone at once.

        Returns the local selection, or `_ANSWERED_REMOTELY` once the phone has
        answered — in which case `answers` already holds it and `remote_marks`
        notes which question it was, so the result line can say where the answer
        came from. A remote timeout is not an answer: it drops back to waiting on
        the picker exactly as if the phone had never been asked.
        """
        import asyncio

        local = asyncio.ensure_future(local_coro)
        try:
            # Ask, re-ask, re-ask, then decide.
            for step, wait in enumerate(cls._ESCALATION_SECONDS):
                try:
                    return await asyncio.wait_for(
                        cls._race_once(local, qtext, header, labels, descs,
                                       remote_ok, answers, closer, remote_marks),
                        timeout=wait)
                except asyncio.TimeoutError:
                    if step + 1 < len(cls._ESCALATION_SECONDS):
                        from ..utils.escalation import ask_minutes_left_after
                        left = ask_minutes_left_after(step)
                        print('  \u23bf  ' + _(
                            'Still waiting on an answer — asking again '
                            '({left} min left before choosing).'
                        ).format(left=left))

            # Nobody answered. Take the recommended option rather than parking
            # the run on a prompt no one is going to see.
            idx = cls._recommended_option_index(labels)
            label = labels[idx] if idx < len(labels) else ''
            if closer is not None:
                closer.set()
                try:
                    await asyncio.wait({local}, timeout=3)
                except Exception:  # noqa: BLE001
                    pass
            print('  \u23bf  ' + _(
                'No answer after {mins} min — going with the recommended option: {label}'
            ).format(mins=cls._escalation_total_minutes(), label=label))
            return [idx]
        finally:
            if not local.done():
                local.cancel()

    @classmethod
    async def _race_once(cls, local, qtext: str, header: str,
                         labels: List[str], descs: List[str],
                         remote_ok: bool, answers: Dict[str, str],
                         closer=None, remote_marks=None):
        """One pass of the local/remote race, with no time limit of its own —
        the caller bounds it. Returning here means somebody actually answered.

        Never awaits `local` directly and never cancels it: this coroutine is
        cancelled at each escalation deadline, and a bare `await local` would
        take the picker down with it, losing an answer the user was in the
        middle of typing.
        """
        import asyncio
        import os

        from ..remote import questions as _remote_q

        if not remote_ok:
            await asyncio.wait({local})
            return local.result()
        remote = asyncio.ensure_future(_remote_q.ask(
            qtext, header, labels, descriptions=descs, cwd=os.getcwd()))
        try:
            done, _pending = await asyncio.wait(
                {local, remote}, return_when=asyncio.FIRST_COMPLETED)
            if local in done:
                return local.result()
            chosen = remote.result()
            if not chosen:
                # phone gave up; the picker still owns it
                await asyncio.wait({local})
                return local.result()
            answers[qtext] = chosen
            # Close the picker through prompt_toolkit rather than cancelling the
            # task around it — a cancelled Application never restores the screen,
            # and the next line printed ends up inside the prompt instead of
            # above it. Then say what happened: the picker vanishing on its own
            # is not an explanation.
            if closer is not None:
                closer.set()
                try:
                    await asyncio.wait_for(local, timeout=3)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
            if remote_marks is not None:
                remote_marks.add(qtext)
            return cls._ANSWERED_REMOTELY
        finally:
            # `local` is owned by the caller across escalation steps — only the
            # phone request belongs to this pass.
            if not remote.done():
                remote.cancel()

    @classmethod
    async def _call_with_arrow_picker(
        cls, questions: List[Any], router: Any
    ) -> AsyncGenerator[ToolResult, None]:
        """Arrow-key (↑/↓/enter, space to toggle for multi) picker path for the
        interactive REPL. Pauses the persistent prompt for the duration so the
        picker Application owns the terminal, then restores it.

        With a phone configured the same question also goes there and the first
        answer wins — the same race the permission prompt runs, for the same
        reason: a turn stopped on a question is stopped whether or not anyone is
        sitting at the terminal, and neither side should own the decision."""
        import asyncio

        from ..utils.arrow_picker import arrow_select
        from ..remote import questions as _remote_q

        answers: Dict[str, str] = {}
        remote_marks: set = set()          # questions the phone answered
        OTHER = _('Other (type your own answer)')
        remote_ok = _remote_q.is_available()

        await router.pause_prompt()
        try:
            for q in questions:
                qtext = cls._q_field(q, 'question', '')
                header = cls._q_field(q, 'header', '')
                options = cls._q_field(q, 'options', [])
                multi = cls._q_field(q, 'multiSelect', False)

                labels = [cls._q_field(o, 'label', '') for o in options]
                descs = [cls._q_field(o, 'description', '') for o in options]
                previews = [cls._q_field(o, 'preview', None) for o in options]

                disp = labels + [OTHER]
                disp_descs = descs + [None]
                disp_prev = previews + [None]

                closer = asyncio.Event()
                sel = await cls._race_local_and_remote(
                    arrow_select(qtext, disp, descriptions=disp_descs,
                                 previews=disp_prev, multi=bool(multi),
                                 header=header, close_when=closer),
                    qtext, header, labels, descs, remote_ok, answers, closer,
                    remote_marks)
                if sel is cls._ANSWERED_REMOTELY:
                    continue
                if not sel:
                    answers[qtext] = '(no answer)'
                    continue

                chosen: List[str] = []
                for idx in sel:
                    if idx == len(labels):  # the appended "Other" entry
                        # The picker Application has exited, so the terminal is
                        # back in cooked mode and a plain input() is safe here.
                        try:
                            custom = input('  ' + _('Your answer: ')).strip()
                        except (EOFError, KeyboardInterrupt):
                            custom = ''
                        if custom:
                            chosen.append(custom)
                    elif 0 <= idx < len(labels):
                        chosen.append(labels[idx])
                answers[qtext] = ', '.join(chosen) if chosen else '(no answer)'
        finally:
            router.resume_prompt()

        answers_text = ', '.join(f'"{q}"="{a}"' for q, a in answers.items())
        yield ToolResultSuccess(
            type='result',
            result_for_assistant=(
                f"User has answered your questions: {answers_text}. "
                "You can now continue with the user's answers in mind."
            ),
            data={'questions': questions, 'answers': answers,
                  'remote_answers': sorted(remote_marks)},
        )

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        questions = input_data.get('questions', [])
        answers: Dict[str, str] = {}

        # Interactive REPL: use the arrow-key picker (↑/↓/enter) instead of a raw
        # input() that fights the persistent PromptSession for stdin. The picker
        # is its own prompt_toolkit Application, so the REPL prompt must be paused
        # first (one Application per terminal) — exactly like the resume picker.
        # input_router is None in -p / --plain mode, where we fall through to the
        # numbered prompt below.
        router = getattr(context, 'input_router', None) if context is not None else None
        if router is not None:
            async for r in cls._call_with_arrow_picker(questions, router):
                yield r
            return

        echo_state = _enable_terminal_echo()
        try:
            for q in questions:
                qtext = q.get('question', '') if isinstance(q, dict) else getattr(q, 'question', '')
                header = q.get('header', '') if isinstance(q, dict) else getattr(q, 'header', '')
                options = q.get('options', []) if isinstance(q, dict) else getattr(q, 'options', [])
                multi = q.get('multiSelect', False) if isinstance(q, dict) else getattr(q, 'multiSelect', False)

                print()
                if header:
                    print(colorize(f"[{header}]", Colors.CYAN))
                safe_print(colorize(qtext, Colors.BOLD))

                opt_labels: List[str] = []
                for i, o in enumerate(options, 1):
                    label = o.get('label', '') if isinstance(o, dict) else getattr(o, 'label', '')
                    desc = o.get('description', '') if isinstance(o, dict) else getattr(o, 'description', '')
                    preview = o.get('preview', None) if isinstance(o, dict) else getattr(o, 'preview', None)
                    opt_labels.append(label)
                    safe_print(f"  {colorize(str(i), Colors.BOLD)}. {label}")
                    if desc:
                        safe_print(f"     {colorize(desc, Colors.DIM)}")
                    if preview:
                        for line in str(preview).split('\n'):
                            print(f"       {colorize(line, Colors.DIM)}")
                other_idx = len(options) + 1
                print(f'  {colorize(str(other_idx), Colors.BOLD)}. ' + _('Other (type your own answer)'))

                # Headless (-p): there is no terminal to answer with, so the phone
                # is the only way this question gets answered at all. Asked before
                # the prompt below rather than racing it, because the local half
                # here is a blocking input() on a thread that cannot be cancelled.
                if not sys.stdin.isatty():
                    from ..remote import questions as _remote_q
                    descs = [o.get('description', '') if isinstance(o, dict)
                             else getattr(o, 'description', '') for o in options]
                    remote = await _remote_q.ask(qtext, header, opt_labels,
                                                 descriptions=descs, cwd=os.getcwd())
                    if remote:
                        print(f"  {remote}  " + _("(remote)"))
                        answers[qtext] = remote
                        continue

                hint = _('numbers comma-separated') if multi else _('a number')
                while True:
                    try:
                        raw = input(colorize('  ' + _('Select {hint}: ', hint=hint), Colors.CYAN)).strip()
                    except (EOFError, KeyboardInterrupt):
                        # Nothing is going to arrive on stdin — EOF is not a slow
                        # typist. Looping on it spun a headless run at 100% CPU
                        # forever; the question is simply unanswerable here.
                        answers[qtext] = '(no answer)'
                        break
                    if not raw:
                        continue
                    picks = [p.strip() for p in raw.split(',')] if multi else [raw]
                    chosen: List[str] = []
                    valid = True
                    for p in picks:
                        if not p.isdigit():
                            valid = False
                            break
                        n = int(p)
                        if n == other_idx:
                            custom = input(colorize('  ' + _('Your answer: '), Colors.CYAN)).strip()
                            if custom:
                                chosen.append(custom)
                        elif 1 <= n <= len(opt_labels):
                            chosen.append(opt_labels[n - 1])
                        else:
                            valid = False
                            break
                    if valid and chosen:
                        answers[qtext] = ', '.join(chosen)
                        break
        finally:
            _restore_terminal(echo_state)

        # Result for the assistant — verbatim format of mapToolResultToToolResult
        # BlockParam (AskUserQuestionTool.tsx:228-241): `"q"="a", "q2"="a2"`.
        # (Annotations preview/notes are an SDK-side concept; the terminal UI
        # here collects only the chosen answer, so there are none to append.)
        answers_text = ', '.join(f'"{qtext}"="{ans}"' for qtext, ans in answers.items())
        result_text = (
            f"User has answered your questions: {answers_text}. "
            "You can now continue with the user's answers in mind."
        )

        yield ToolResultSuccess(
            type='result',
            result_for_assistant=result_text,
            data={'questions': questions, 'answers': answers},
        )
