"""
``codemason skill`` management subcommand.

Intercepted at the top of cli.main() (before argparse, which owns the positional
``prompt``). The skill *runtime* — discovery, frontmatter parsing, argument
substitution, the Skill tool — already lives in utils/skills.py and
tools/skill_tool.py; this only adds the ergonomics for authoring and managing
the SKILL.md files it reads, so a user doesn't have to hand-create them.

Skills live in ``<skills_dir>/<name>/SKILL.md`` where skills_dir is
``~/.codemason/skills`` (user, default) or ``<cwd>/.codemason/skills``
(``--project``). ``--claude`` writes to the matching ``.claude`` dir instead;
naming it on the command line is itself the opt-in, but the runtime only READS
those dirs when config.import_claude_dirs is on.

    codemason skill list
    codemason skill show <name>
    codemason skill new <name> [--description D] [--arguments "a b"]
                                [--context inline|fork] [--agent NAME] [--project]
    codemason skill remove <name> [--project]
    codemason skill enable <name>
    codemason skill disable <name>
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .config import CONFIG_DIR
from .state import get_cwd


def _roots() -> List[Tuple[str, Path]]:
    """(label, dir) for every skill root, in the same first-wins precedence the
    runtime uses (utils/skills._skill_dirs): user before project, native
    ``.codemason`` before an imported ``.claude``. The imported roots appear
    only when config.import_claude_dirs is on, so `list` shows what will
    actually load rather than a root the runtime is going to skip."""
    from .skills import bundled_skills_dir
    from .config import get_config
    home = Path(os.path.expanduser("~"))
    cwd = Path(get_cwd())
    import_foreign = bool(getattr(get_config(), "import_claude_dirs", False))
    roots = [("user:.codemason", CONFIG_DIR / "skills")]
    if import_foreign:
        roots.append(("user:.claude", home / ".claude" / "skills"))
    roots.append(("project:.codemason", cwd / ".codemason" / "skills"))
    if import_foreign:
        roots.append(("project:.claude", cwd / ".claude" / "skills"))
    roots.append(("bundled", bundled_skills_dir()))
    # De-dupe by path (e.g. run from home: user and project resolve equal).
    out: List[Tuple[str, Path]] = []
    seen = set()
    for label, d in roots:
        if str(d) in seen:
            continue
        seen.add(str(d))
        out.append((label, d))
    return out


def _target_dir(project: bool, claude: bool) -> Path:
    """Destination root for `new`/`remove` given the scope/compat flags."""
    home = Path(os.path.expanduser("~"))
    cwd = Path(get_cwd())
    if project:
        return (cwd / ".claude" / "skills") if claude else (cwd / ".codemason" / "skills")
    return (home / ".claude" / "skills") if claude else (CONFIG_DIR / "skills")


def _skill_md(name: str, project: bool, claude: bool = False) -> Path:
    return _target_dir(project, claude) / name / "SKILL.md"


def _find_skill_md(name: str) -> Optional[Path]:
    """Return the SKILL.md path for ``name`` in precedence order across roots."""
    for _label, base in _roots():
        p = base / name / "SKILL.md"
        if p.is_file():
            return p
    return None


def _scope_of(path: Path) -> str:
    s = str(path)
    for label, base in _roots():
        if s.startswith(str(base)):
            return label
    return "?"


# ── list ────────────────────────────────────────────────────────────────────
def _cmd_list(args: List[str]) -> int:
    from .skills import discover_skills, model_invocable_skills, refresh_skills
    refresh_skills()
    skills = discover_skills()
    if not skills:
        print("No skills found. Create one with: codemason skill new <name>")
        print("  (searched " + ", ".join(str(d) for _l, d in _roots()) + ")")
        return 0
    invocable = {s.name for s in model_invocable_skills()}
    for s in skills:
        scope = _scope_of(Path(s.path))
        tags = []
        if s.context == "fork":
            tags.append("fork")
        if s.name not in invocable:
            tags.append("not-model-invocable")
        if s.argument_names:
            tags.append("args:" + ",".join(s.argument_names))
        suffix = ("  [" + ", ".join(tags) + "]") if tags else ""
        desc = (s.description or "").strip()
        print(f"  {s.name:<24} ({scope}) {desc}{suffix}")
    return 0


# ── show ────────────────────────────────────────────────────────────────────
def _cmd_show(args: List[str]) -> int:
    if not args:
        print("usage: codemason skill show <name>", file=sys.stderr)
        return 2
    name = args[0]
    p = _find_skill_md(name)
    if p is None:
        print(f"error: no such skill: {name}", file=sys.stderr)
        return 1
    print(f"# {name}  ({_scope_of(p)})")
    print(f"# {p}\n")
    print(p.read_text(encoding="utf-8", errors="replace"))
    return 0


# ── new ─────────────────────────────────────────────────────────────────────
_TEMPLATE = """\
---
name: {name}
description: {description}
{extra}---
<!-- The body below IS the skill: it expands into the prompt when the skill is
     invoked. Reference arguments as ${{1}}, ${{2}} (or names you declared in
     `arguments:`), and the skill's own directory as ${{CODEMASON_SKILL_DIR}}. -->

When this skill runs, do the following:

1. ...
2. ...
"""


def _cmd_new(args: List[str]) -> int:
    if not args:
        print("usage: codemason skill new <name> [--description D] [--arguments \"a b\"] "
              "[--context inline|fork] [--agent NAME] [--project] [--claude]", file=sys.stderr)
        return 2
    name = args[0]
    rest = args[1:]
    description = "TODO: one line on when to use this skill."
    arguments = agent = None
    context = "inline"
    project = False
    claude = False
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--description":
            i += 1; description = rest[i]
        elif tok == "--arguments":
            i += 1; arguments = rest[i]
        elif tok == "--context":
            i += 1; context = rest[i]
        elif tok == "--agent":
            i += 1; agent = rest[i]
        elif tok == "--project":
            project = True
        elif tok == "--claude":
            claude = True
        else:
            print(f"unknown option: {tok}", file=sys.stderr); return 2
        i += 1
    if context not in ("inline", "fork"):
        print("error: --context must be 'inline' or 'fork'", file=sys.stderr)
        return 2
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        print("error: skill name must be [A-Za-z0-9._-]", file=sys.stderr)
        return 2

    path = _skill_md(name, project, claude)
    if path.exists():
        print(f"error: skill already exists: {path}", file=sys.stderr)
        return 1
    extra = ""
    if arguments:
        extra += f"arguments: {arguments}\n"
    if context == "fork":
        extra += "context: fork\n"
    if agent:
        extra += f"agent: {agent}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE.format(name=name, description=description, extra=extra),
                    encoding="utf-8")
    print(f"Created skill '{name}' ({_scope_of(path)}) at:")
    print(f"  {path}")
    print("Edit the body, then it is available to the model and via /"+name)
    return 0


# ── remove ──────────────────────────────────────────────────────────────────
def _cmd_remove(args: List[str]) -> int:
    project = "--project" in args
    claude = "--claude" in args
    names = [a for a in args if not a.startswith("--")]
    if not names:
        print("usage: codemason skill remove <name> [--project] [--claude]", file=sys.stderr)
        return 2
    name = names[0]
    path = _skill_md(name, project, claude)
    if not path.exists():
        # fall back to wherever it actually is
        found = _find_skill_md(name)
        if found is None:
            print(f"error: no such skill: {name}", file=sys.stderr)
            return 1
        path = found
    skill_dir = path.parent
    shutil.rmtree(skill_dir)
    print(f"Removed skill '{name}' ({skill_dir})")
    return 0


# ── enable / disable (toggle disable-model-invocation frontmatter) ───────────
_FM_RE = re.compile(r"^(---\s*\n)(.*?\n)(---\s*\n)", re.DOTALL)


def _set_disable_flag(text: str, disable: bool) -> Tuple[str, bool]:
    """Add/remove ``disable-model-invocation: true`` in the frontmatter block.

    Returns (new_text, changed). Targeted line edit — the rest of the file
    (body, other frontmatter keys, comments) is left byte-for-byte intact.
    """
    m = _FM_RE.match(text)
    if not m:
        return text, False
    open_d, body, close_d = m.group(1), m.group(2), m.group(3)
    lines = body.split("\n")
    kept = [ln for ln in lines if not re.match(r"\s*disable-model-invocation\s*:", ln)]
    had = len(kept) != len(lines)
    if disable:
        # ensure exactly one flag line
        if kept and kept[-1] == "":
            kept.insert(len(kept) - 1, "disable-model-invocation: true")
        else:
            kept.append("disable-model-invocation: true")
        changed = True
    else:
        changed = had  # removing only changes if it was there
    new_body = "\n".join(kept)
    new_text = open_d + new_body + close_d + text[m.end():]
    return new_text, changed


def _toggle(name: str, disable: bool) -> int:
    p = _find_skill_md(name)
    if p is None:
        print(f"error: no such skill: {name}", file=sys.stderr)
        return 1
    text = p.read_text(encoding="utf-8")
    new_text, changed = _set_disable_flag(text, disable)
    if not changed:
        print(f"skill '{name}' already "
              f"{'disabled' if disable else 'enabled'} for model invocation.")
        return 0
    p.write_text(new_text, encoding="utf-8")
    print(f"{'Disabled' if disable else 'Enabled'} model invocation for skill '{name}'.")
    return 0


def _cmd_enable(args: List[str]) -> int:
    if not args:
        print("usage: codemason skill enable <name>", file=sys.stderr); return 2
    return _toggle(args[0], disable=False)


def _cmd_disable(args: List[str]) -> int:
    if not args:
        print("usage: codemason skill disable <name>", file=sys.stderr); return 2
    return _toggle(args[0], disable=True)


def handle_skill_cli(argv: List[str]) -> int:
    """Dispatch ``codemason skill <sub> ...``. ``argv`` is sys.argv[2:]."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("codemason skill — author and manage skills\n"
              "  roots (first-wins): ~/.codemason/skills, "
              "<cwd>/.codemason/skills\n"
              "  (~/.claude/skills and <cwd>/.claude/skills are read only with "
              "import_claude_dirs on)\n\n"
              "  list                     list discovered skills\n"
              "  show <name>              print a skill's SKILL.md\n"
              "  new <name> [opts]        scaffold a new skill (--project, --claude)\n"
              "  remove <name>            delete a skill (--project, --claude)\n"
              "  enable <name>            allow model invocation\n"
              "  disable <name>           block model invocation\n")
        return 0
    sub, rest = argv[0], argv[1:]
    dispatch = {
        "list": _cmd_list, "ls": _cmd_list,
        "show": _cmd_show, "cat": _cmd_show,
        "new": _cmd_new, "create": _cmd_new,
        "remove": _cmd_remove, "rm": _cmd_remove,
        "enable": _cmd_enable,
        "disable": _cmd_disable,
    }
    fn = dispatch.get(sub)
    if fn is None:
        print(f"unknown skill subcommand: {sub}", file=sys.stderr)
        return 2
    return fn(rest)
