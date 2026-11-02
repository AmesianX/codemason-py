"""
Tab / live completion for the interactive REPL prompt.

Wired into the PromptSession as ``completer=``. Two kinds of completion:

  * **Slash commands** — when the line's first token starts with ``/``, complete
    the REPL built-ins (/help, /clear, …) and every discovered skill name
    (``/my-skill``), shown with its description. This is live (as-you-type).

  * **File paths** — otherwise, on an explicit Tab, complete the filesystem path
    under the cursor (``codemason/uti`` → ``codemason/utils/``). Gated to Tab (or
    clearly path-like tokens) so it never interrupts normal prose typing.

get_completions never raises — a completion error must not break the prompt.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List

from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, Suggestion
from prompt_toolkit.completion import Completer, Completion

# Built-in slash commands and a one-line meta for the menu. Keep in sync with
# cli._BUILTIN_COMMANDS / the /help text.
_BUILTIN_META: Dict[str, str] = {
    "help": "show help", "exit": "quit", "quit": "quit",
    "clear": "new session (clear history)", "compact": "summarize to free context",
    "resume": "pick a stored session to continue", "expand": "print last N tool outputs",
    "export": "save the conversation to Markdown", "language": "set UI language",
    "lang": "set UI language", "keepgoing": "toggle keep-going",
    "dsp": "toggle skipping tool approvals",
    "effort": "reasoning effort: low|medium|high|off",
    "reasoning": "reasoning effort: low|medium|high|off",
    "think": "reasoning effort: low|medium|high|off",
    "goal": "set/show/clear a persistent goal",
    "remote": "remote channel status/toggle",
    "remote-control": "remote channel status/toggle", "rc": "remote channel (alias)",
}


# Commands whose single argument has a fixed set of options — completed after
# `/<cmd> `. Keep in sync with the handlers in cli._dispatch_builtin.
_EFFORT_ARGS = ["low", "medium", "high", "off", "status"]
_ONOFF_ARGS = ["on", "off", "status"]
_COMMAND_ARGS: Dict[str, List[str]] = {
    "effort": _EFFORT_ARGS, "reasoning": _EFFORT_ARGS, "think": _EFFORT_ARGS,
    "dsp": _ONOFF_ARGS, "keepgoing": _ONOFF_ARGS,
    "remote": _ONOFF_ARGS, "remote-control": _ONOFF_ARGS, "rc": _ONOFF_ARGS,
}


def _skill_completions() -> List[tuple]:
    """(name, description) for every discovered skill; empty on any error."""
    try:
        from .skills import discover_skills
        return [(s.name, (s.description or "").strip()) for s in discover_skills()]
    except Exception:  # noqa: BLE001
        return []


def _path_completions(word: str) -> Iterable[Completion]:
    """Complete a filesystem path fragment (the token under the cursor)."""
    expanded = os.path.expanduser(word)
    dirname, partial = os.path.split(expanded)
    search_dir = dirname or "."
    try:
        entries = os.listdir(search_dir)
    except OSError:
        return
    # Preserve the user's typed ~ / dir prefix when inserting.
    prefix = word[: len(word) - len(partial)]
    for name in sorted(entries):
        if name.startswith("."):
            if not partial.startswith("."):
                continue  # hide dotfiles unless the user started typing one
        if not name.startswith(partial):
            continue
        full = os.path.join(search_dir, name)
        display = name + ("/" if os.path.isdir(full) else "")
        yield Completion(
            prefix + display,
            start_position=-len(word),
            display=display,
        )


def _looks_like_path(word: str) -> bool:
    return bool(word) and (
        "/" in word or word.startswith(("~", ".", "./", "../"))
    )


class CodemasonCompleter(Completer):
    """Slash-command/skill completion (live) + path completion (on Tab)."""

    def get_completions(self, document, complete_event):  # noqa: D401
        try:
            text = document.text_before_cursor
            stripped = text.lstrip()

            if stripped.startswith("/") and "\n" not in stripped:
                # First token still being typed (no space yet): complete the
                # command / skill name.
                if " " not in stripped:
                    word = stripped                # includes the leading '/'
                    prefix = word[1:].lower()
                    for name, meta in _BUILTIN_META.items():
                        if name.startswith(prefix):
                            yield Completion("/" + name, start_position=-len(word),
                                             display="/" + name, display_meta=meta)
                    for name, desc in _skill_completions():
                        if name.lower().startswith(prefix):
                            yield Completion("/" + name, start_position=-len(word),
                                             display="/" + name, display_meta=desc)
                    return
                # Past the command: complete its argument from the fixed option
                # set, but only for the FIRST argument (single-arg commands).
                parts = stripped.split()
                opts = _COMMAND_ARGS.get(parts[0][1:].lower())
                if opts:
                    args = parts[1:]
                    # args fully entered (not counting the partial being typed)
                    completed = len(args) - (0 if stripped.endswith(" ") else 1)
                    if completed == 0:  # only the first argument
                        arg = "" if stripped.endswith(" ") else parts[-1].lower()
                        for opt in opts:
                            if opt.startswith(arg):
                                yield Completion(opt, start_position=-len(arg), display=opt)
                return

            # File path — the last whitespace-delimited token. Offer it on an
            # explicit Tab, or immediately when the token is clearly a path.
            if text and not text[-1].isspace():
                word = text.split()[-1]
            else:
                word = ""
            if word and (complete_event.completion_requested or _looks_like_path(word)):
                yield from _path_completions(word)
        except Exception:  # noqa: BLE001 - never break the prompt
            return


def build_completer() -> Completer:
    return CodemasonCompleter()


class CodemasonAutoSuggest(AutoSuggest):
    """Inline "ghost text" suggestion for the next input (accept with →).

    - While typing a ``/`` command, suggest the first matching built-in or skill
      name (``/sy`` → ghosts ``stematic-debugging``).
    - Otherwise, fall back to the most recent history line that starts with what
      you've typed (fish-shell style).
    """

    def __init__(self):
        self._from_history = AutoSuggestFromHistory()

    def get_suggestion(self, buffer, document):  # noqa: D401
        try:
            text = document.text
            if not text or "\n" in text:
                return None

            stripped = text.lstrip()
            if stripped.startswith("/") and " " not in stripped:
                prefix = stripped[1:].lower()
                if prefix:
                    names = list(_BUILTIN_META.keys()) + [n for n, _ in _skill_completions()]
                    for name in names:
                        if name.lower().startswith(prefix) and len(name) > len(prefix):
                            return Suggestion(name[len(prefix):])
                return None

            # non-slash: history-based suggestion
            return self._from_history.get_suggestion(buffer, document)
        except Exception:  # noqa: BLE001 - never break the prompt
            return None


def build_auto_suggest() -> AutoSuggest:
    return CodemasonAutoSuggest()
