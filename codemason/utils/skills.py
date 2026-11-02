"""
Skill discovery + frontmatter parsing for CodeMason.
Based on Claude Code's skill loading (utils/frontmatterParser.ts + skills/loadSkillsDir.ts).
A skill is a markdown file `<skills_dir>/<name>/SKILL.md` with YAML frontmatter
(description, context: inline|fork, agent, ...) plus a body that is the skill's
prompt. `context: inline` (default) expands into the conversation; `context: fork`
runs the skill in a sub-agent.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .state import get_cwd


# utils/frontmatterParser.ts FRONTMATTER_REGEX
_FRONTMATTER_REGEX = re.compile(r'^---\s*\n([\s\S]*?)---\s*\n?')

# frontmatterParser.ts YAML_SPECIAL_CHARS — chars/sequences that require quoting
# in an unquoted YAML value. Note ': ' (colon+space) is matched as a sequence so
# that '12:34' times and 'https://' URLs stay unquoted.
_YAML_SPECIAL_CHARS = re.compile(r'[{}\[\]*&#!|>%@`]|: ')

# frontmatterParser.ts quoteProblematicValues line matcher: simple `key: value`
# lines (not indented, not list items, not block scalars).
_SIMPLE_KV_LINE = re.compile(r'^([a-zA-Z_-]+):\s+(.+)$')


def _quote_problematic_values(frontmatter_text: str) -> str:
    """Port of frontmatterParser.ts quoteProblematicValues (lines 85-121).

    Pre-process frontmatter text to quote values that contain special YAML
    characters, so glob patterns like **/*.{ts,tsx} parse correctly."""
    out: List[str] = []
    for line in frontmatter_text.split('\n'):
        match = _SIMPLE_KV_LINE.match(line)
        if match:
            key, value = match.group(1), match.group(2)
            if not key or not value:
                out.append(line)
                continue
            # Skip if already quoted.
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                out.append(line)
                continue
            # Quote if it contains special YAML characters. Escape backslashes
            # first, then double-quotes.
            if _YAML_SPECIAL_CHARS.search(value):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                out.append(f'{key}: "{escaped}"')
                continue
        out.append(line)
    return '\n'.join(out)


def parse_frontmatter(markdown: str) -> Tuple[Dict, str]:
    """parseFrontmatter — (frontmatter dict, content without frontmatter).

    On YAML failure, retries after quoting problematic values (so a single
    special char in one value no longer nukes the whole frontmatter)."""
    m = _FRONTMATTER_REGEX.match(markdown)
    if not m:
        return {}, markdown
    fm_text = m.group(1) or ''
    content = markdown[len(m.group(0)):]
    fm: Dict = {}
    try:
        parsed = yaml.safe_load(fm_text)
        if isinstance(parsed, dict):
            fm = parsed
    except Exception:
        # YAML parsing failed - try again after quoting problematic values.
        try:
            parsed = yaml.safe_load(_quote_problematic_values(fm_text))
            if isinstance(parsed, dict):
                fm = parsed
        except Exception:
            fm = {}
    return fm, content


def _parse_bool_frontmatter(value) -> bool:
    """frontmatterParser.ts parseBooleanFrontmatter: true only for True/'true'."""
    return value is True or value == 'true'


def parse_argument_names(value) -> List[str]:
    """argumentSubstitution.ts parseArgumentNames: frontmatter `arguments` as a
    space-separated string OR a list. Drops empty + numeric-only names (those
    collide with the $0/$1 indexed shorthand)."""
    if not value:
        return []
    def _valid(n) -> bool:
        return isinstance(n, str) and n.strip() != '' and not re.fullmatch(r'\d+', n.strip())
    if isinstance(value, list):
        return [n.strip() for n in value if _valid(n)]
    if isinstance(value, str):
        return [n for n in re.split(r'\s+', value.strip()) if _valid(n)]
    return []


def _parse_skill_paths(fm: Dict) -> List[str]:
    """Port of loadSkillsDir.ts parseSkillPaths. Returns the list of path
    patterns (or empty). Substrate gap: codemason cannot yet detect which files
    the model touches, so skills with non-empty paths are activated nowhere and
    are excluded from the default listing (see discover_skills)."""
    raw = fm.get('paths')
    if not raw:
        return []
    if isinstance(raw, list):
        items = [str(p) for p in raw]
    else:
        # Comma-separated string. (We don't expand brace patterns here — the
        # presence/absence of patterns is all that matters for the listing.)
        items = [p.strip() for p in str(raw).split(',')]
    out: List[str] = []
    for p in items:
        # Strip /** suffix to match ignore-library path semantics.
        if p.endswith('/**'):
            p = p[:-3]
        if p:
            out.append(p)
    return out


@dataclass
class Skill:
    name: str
    description: str
    content: str
    context: str = 'inline'        # 'inline' | 'fork'
    agent: Optional[str] = None
    path: str = ''
    display_name: Optional[str] = None
    disable_model_invocation: bool = False
    paths: List[str] = field(default_factory=list)
    argument_names: List[str] = field(default_factory=list)
    base_dir: str = ''             # the skill's directory; ${CODEMASON_SKILL_DIR}


def bundled_skills_dir() -> Path:
    """The skills shipped inside the codemason package (``codemason/skills``).

    Lowest-precedence root, so a user/project skill of the same name overrides
    the bundled one. skills.py lives in ``codemason/utils/`` → its parent's
    parent is the package root."""
    return Path(__file__).resolve().parent.parent / 'skills'


def _skill_dirs() -> List[Path]:
    """Skill search roots, in first-wins precedence order.

    By default only our own dirs are searched, user before project:

        user/.codemason  ->  project/.codemason  ->  bundled

    With config.import_claude_dirs on, each scope also searches the matching
    `.claude` dir, so skills authored for the other tool are picked up:

        user/.codemason  ->  user/.claude  ->  project/.codemason  ->  project/.claude

    The native dir wins over an imported one on a name collision, and user wins
    over project (matching the original's managed -> user -> project order;
    codemason has no managed dir). The user-level `.codemason` root honors
    CODEMASON_CONFIG_DIR via CONFIG_DIR.
    """
    from .config import CONFIG_DIR  # local import: config.py must load first
    home = Path(os.path.expanduser('~'))
    cwd = Path(get_cwd())
    # Another product's directories are read ONLY when asked for. This is our
    # program; inheriting a foreign layout by default means a skill we never
    # shipped can appear in our tool list without the user choosing that.
    from .config import get_config
    _import_foreign = bool(getattr(get_config(), 'import_claude_dirs', False))
    candidates = [CONFIG_DIR / 'skills']              # user, native
    if _import_foreign:
        candidates.append(home / '.claude' / 'skills')   # user, opt-in import
    candidates.append(cwd / '.codemason' / 'skills')     # project, native
    if _import_foreign:
        candidates.append(cwd / '.claude' / 'skills')    # project, opt-in import
    candidates.append(bundled_skills_dir())              # shipped (lowest)
    dirs: List[Path] = []
    seen = set()
    for d in candidates:
        rp = str(d)
        if rp in seen:
            continue
        seen.add(rp)
        if d.is_dir():
            dirs.append(d)
    return dirs


_discover_skills_cache: Optional[List["Skill"]] = None
_discover_skills_sig: Optional[tuple] = None


def refresh_skills() -> None:
    """Drop the discover_skills() cache (call if skill dirs change mid-session)."""
    global _discover_skills_cache, _discover_skills_sig
    _discover_skills_cache = None
    _discover_skills_sig = None


def _skills_signature() -> tuple:
    """A cheap fingerprint of the skill roots and their immediate skill dirs, so
    a skill CREATED or REMOVED mid-session (which changes a root's mtime and adds
    /drops a subdir) invalidates the cache automatically — without re-parsing
    every SKILL.md on each call. Content edits to an existing SKILL.md still need
    refresh_skills() (or a `codemason skill` run, which is a fresh process)."""
    sig: List[tuple] = []
    for d in _skill_dirs():
        try:
            sig.append((str(d), d.stat().st_mtime_ns))
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    try:
                        md = sub / 'SKILL.md'
                        sig.append((str(sub), sub.stat().st_mtime_ns,
                                    md.stat().st_mtime_ns if md.is_file() else 0))
                    except OSError:
                        continue
        except OSError:
            continue
    return tuple(sig)


def discover_skills() -> List[Skill]:
    """Scan the skill dirs for `<name>/SKILL.md` files (cached, auto-refreshed).

    Memoized per process, but the cache is invalidated automatically when the
    skill directories change (a skill added or removed), keyed on a cheap mtime
    signature — so a skill created mid-session becomes visible without a restart.
    Call refresh_skills() to force a re-scan (e.g. after editing a SKILL.md's
    contents in place)."""
    global _discover_skills_cache, _discover_skills_sig
    sig = _skills_signature()
    if _discover_skills_cache is not None and sig == _discover_skills_sig:
        return _discover_skills_cache
    _discover_skills_sig = sig
    skills: Dict[str, Skill] = {}
    for d in _skill_dirs():
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for sub in entries:
            skill_md = sub / 'SKILL.md'
            if not (sub.is_dir() and skill_md.is_file()):
                continue
            name = sub.name
            if name in skills:
                # First-wins: an earlier dir (user) already provided this name.
                continue
            try:
                raw = skill_md.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            fm, content = parse_frontmatter(raw)
            display_name = (
                str(fm.get('name')) if fm.get('name') is not None else None
            )
            desc = str(fm.get('description') or '').strip()
            ctx = fm.get('context') or 'inline'
            ctx = ctx if ctx in ('inline', 'fork') else 'inline'
            skills[name] = Skill(
                name=name, description=desc, content=content.strip(),
                context=ctx, agent=fm.get('agent'), path=str(skill_md),
                display_name=display_name,
                disable_model_invocation=_parse_bool_frontmatter(
                    fm.get('disable-model-invocation')
                ),
                paths=_parse_skill_paths(fm),
                argument_names=parse_argument_names(fm.get('arguments')),
                base_dir=str(sub),
            )
    _discover_skills_cache = list(skills.values())
    return _discover_skills_cache


def model_invocable_skills() -> List[Skill]:
    """Skills the model may invoke via the Skill tool: exclude
    disable-model-invocation skills and conditional (paths) skills.

    - disable-model-invocation: excluded from model invocation entirely
      (loadSkillsDir.ts / commands.ts getSkillToolCommands).
    - non-empty paths: conditional skills that only activate when matching files
      are touched (loadSkillsDir.ts:771-794). Substrate gap: codemason can't
      detect touched files, so these are excluded from the listing here."""
    return [
        s for s in discover_skills()
        if not s.disable_model_invocation and not s.paths
    ]


def get_skill(name: str) -> Optional[Skill]:
    """Look up a model-invocable skill by name (excludes disable-model-invocation
    and conditional/paths skills)."""
    for s in model_invocable_skills():
        if s.name == name:
            return s
    return None
