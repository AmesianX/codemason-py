"""Read-only external-command validation (adapted from readOnlyCommandValidation.ts).

Shared engine + data for deciding whether an external command (git, gh, docker,
rg, pyright, ...) is read-only. This module is the ENGINE (flag validation, UNC
detection, config type); the per-command allowlists (GIT/GH/DOCKER/...) live
alongside and are consumed by both the Bash and PowerShell tools.

Cmdlets/commands are matched case-insensitively by their callers.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..platform import is_windows

# FlagArgType: 'none' | 'number' | 'string' | 'char' | '{}' | 'EOF'
FlagArgType = str


@dataclass
class ExternalCommandConfig:
    """Safe-flag config for a read-only external command."""

    safe_flags: Dict[str, FlagArgType] = field(default_factory=dict)
    # Returns True if the command is dangerous, False if safe. args = tokens
    # AFTER the command name (e.g. after "git branch").
    additional_command_is_dangerous_callback: Optional[Callable[[str, List[str]], bool]] = None
    # When False, do NOT respect POSIX `--` end-of-options (keep checking flags).
    respects_double_dash: bool = True


# Cross-platform external tools identical in bash and PowerShell on Windows.
# Cross-platform external tools identical in bash and PowerShell on Windows.
#
# These two are allowed WITHOUT flag validation, unlike every other docker
# subcommand (see DOCKER_READ_ONLY_COMMANDS, which puts `logs` and `inspect`
# through validate_flags). That asymmetry is deliberate, and it has been filed
# as an inconsistency once — the reasoning is written down here so it is not
# filed again.
#
# `ps` and `images` are read-only COMMANDS, not merely commands with read-only
# flags: nothing in either flag set writes. ps takes -a/--all, -f/--filter,
# --format, -n/--last, -l/--latest, --no-trunc, -q/--quiet, -s/--size; images
# takes -a/--all, --digests, -f/--filter, --format, --no-trunc, -q/--quiet.
# There is no mutating flag to omit from an allowlist, so an allowlist would buy
# no safety.
#
# It would cost something, though. Every flag docker adds later would become a
# permission prompt for a command that cannot do anything, and the list would
# need chasing upstream forever. Security that costs more usability than it buys
# is not a trade worth making, so the state and its price are recorded rather
# than "fixed".
#
# `$` is still rejected before this is reached, so the substitution route in is
# closed independently of the flag surface.
EXTERNAL_READONLY_COMMANDS: List[str] = ["docker ps", "docker images"]


# =============================================================================
# UNC path detection (shared across Bash and PowerShell).
# =============================================================================
_UNC_BACKSLASH_RE = re.compile(r"\\\\[^\s\\/]+(?:@(?:\d+|ssl))?(?:[\\/]|$|\s)", re.IGNORECASE)
_UNC_FORWARDSLASH_RE = re.compile(r"(?<!:)//[^\s\\/]+(?:@(?:\d+|ssl))?(?:[\\/]|$|\s)", re.IGNORECASE)
_UNC_MIXED_RE = re.compile(r"/\\{2,}[^\s\\/]")
_UNC_REVERSE_MIXED_RE = re.compile(r"\\{2,}/[^\s\\/]")
_UNC_SSL_PORT_RE = re.compile(r"@SSL@\d+", re.IGNORECASE)
_UNC_PORT_SSL_RE = re.compile(r"@\d+@SSL", re.IGNORECASE)
_UNC_DAV_RE = re.compile(r"DavWWWRoot", re.IGNORECASE)
_UNC_IPV4_BS_RE = re.compile(r"^\\\\(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\\/]")
_UNC_IPV4_FS_RE = re.compile(r"^//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\\/]")
_UNC_IPV6_BS_RE = re.compile(r"^\\\\(\[[\da-fA-F:]+\])[\\/]")
_UNC_IPV6_FS_RE = re.compile(r"^//(\[[\da-fA-F:]+\])[\\/]")


def contains_vulnerable_unc_path(path_or_command: str) -> bool:
    """True if the string contains a UNC path that could trigger a network
    request (NTLM/Kerberos leak, WebDAV). Windows-only."""
    if not is_windows():
        return False
    if _UNC_BACKSLASH_RE.search(path_or_command):
        return True
    if _UNC_FORWARDSLASH_RE.search(path_or_command):
        return True
    if _UNC_MIXED_RE.search(path_or_command):
        return True
    if _UNC_REVERSE_MIXED_RE.search(path_or_command):
        return True
    if _UNC_SSL_PORT_RE.search(path_or_command) or _UNC_PORT_SSL_RE.search(path_or_command):
        return True
    if _UNC_DAV_RE.search(path_or_command):
        return True
    if _UNC_IPV4_BS_RE.search(path_or_command) or _UNC_IPV4_FS_RE.search(path_or_command):
        return True
    if _UNC_IPV6_BS_RE.search(path_or_command) or _UNC_IPV6_FS_RE.search(path_or_command):
        return True
    return False


# =============================================================================
# Flag validation.
# =============================================================================
FLAG_PATTERN = re.compile(r"^-[a-zA-Z0-9_-]")
# ASCII-only + full-string: JS `/^\d+$/.test()` matches only ASCII 0-9 and does
# not accept a trailing newline. Python's default `\d` is Unicode and `$` matches
# before a trailing '\n', which would let `"12\n"` / fullwidth `"１２"` slip
# through as read-only. re.ASCII + fullmatch restores TS semantics.
_NUMERIC_RE = re.compile(r"\d+", re.ASCII)
# re.ASCII (Python \d is Unicode, so `-١٢` Arabic-Indic digits would match) and
# use fullmatch below (not .match with a trailing `$`, which also allows `-5\n`).
_GIT_NUM_SHORTHAND_RE = re.compile(r"-\d+", re.ASCII)
_SORT_REVERSE_RE = re.compile(r"^-[a-zA-Z]")


def validate_flag_argument(value: str, arg_type: FlagArgType) -> bool:
    """Validate a flag argument against its expected type."""
    if arg_type == "none":
        return False  # should not be called for 'none'
    if arg_type == "number":
        return bool(_NUMERIC_RE.fullmatch(value))
    if arg_type == "string":
        return True
    if arg_type == "char":
        return len(value) == 1
    if arg_type == "{}":
        return value == "{}"
    if arg_type == "EOF":
        return value == "EOF"
    return False


def validate_flags(
    tokens: List[str],
    start_index: int,
    config: ExternalCommandConfig,
    command_name: Optional[str] = None,
    raw_command: Optional[str] = None,
    xargs_target_commands: Optional[List[str]] = None,
) -> bool:
    """Validate the flags/args of a tokenized command against `config`.

    Faithful port of the flag-walking loop (validateFlags), including the
    security-hardened cases: xargs target detection, `-E=` empty-inline-value
    (no next-token consume), git `-<n>` shorthand, grep/rg attached-numeric
    (`-A20`), and bundled-flag rejection when any bundled flag takes an arg.
    """
    i = start_index
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if not token:
            i += 1
            continue

        # xargs: once the target command appears, stop validating flags.
        if xargs_target_commands and command_name == "xargs" and (not token.startswith("-") or token == "--"):
            if token == "--" and i + 1 < n:
                i += 1
                token = tokens[i]
            if token and token in xargs_target_commands:
                break
            return False

        if token == "--":
            if config.respects_double_dash is not False:
                i += 1
                break  # everything after -- is arguments
            i += 1
            continue

        if token.startswith("-") and len(token) > 1 and FLAG_PATTERN.match(token):
            has_equals = "=" in token
            parts = token.split("=")
            flag = parts[0]
            inline_value = "=".join(parts[1:])

            if not flag:
                return False

            flag_arg_type = config.safe_flags.get(flag)

            if not flag_arg_type:
                # git -<number> shorthand for -n <number>
                if command_name == "git" and _GIT_NUM_SHORTHAND_RE.fullmatch(flag):
                    i += 1
                    continue
                # grep/rg attached numeric (-A20)
                if command_name in ("grep", "rg") and flag.startswith("-") and not flag.startswith("--") and len(flag) > 2:
                    potential_flag = flag[:2]
                    potential_value = flag[2:]
                    pf_type = config.safe_flags.get(potential_flag)
                    if pf_type and _NUMERIC_RE.fullmatch(potential_value):
                        if pf_type in ("number", "string"):
                            if validate_flag_argument(potential_value, pf_type):
                                i += 1
                                continue
                            return False
                # bundled single-letter flags (-nr): all must be 'none'.
                if flag.startswith("-") and not flag.startswith("--") and len(flag) > 2:
                    for j in range(1, len(flag)):
                        single_flag = "-" + flag[j]
                        flag_type = config.safe_flags.get(single_flag)
                        if not flag_type:
                            return False
                        if flag_type != "none":
                            return False  # arg-taking flag in a bundle
                    i += 1
                    continue
                return False  # unknown flag

            if flag_arg_type == "none":
                if has_equals:
                    return False  # flag should not have a value
                i += 1
            else:
                if has_equals:
                    arg_value = inline_value
                    i += 1
                else:
                    nxt = tokens[i + 1] if i + 1 < n else None
                    if (
                        i + 1 >= n
                        or (nxt and nxt.startswith("-") and len(nxt) > 1 and FLAG_PATTERN.match(nxt))
                    ):
                        return False  # missing required argument
                    arg_value = tokens[i + 1] or ""
                    i += 2

                # string args starting with '-' rejected (type-confusion), except
                # git --sort reverse (-refname).
                if flag_arg_type == "string" and arg_value.startswith("-"):
                    if flag == "--sort" and command_name == "git" and _SORT_REVERSE_RE.match(arg_value):
                        pass  # reverse sort key
                    else:
                        return False

                if not validate_flag_argument(arg_value, flag_arg_type):
                    return False
        else:
            i += 1  # non-flag positional arg — allowed

    return True

# =============================================================================
# Per-command danger callbacks (hand-ported; the data records above wire these).
# =============================================================================

def _git_reflog_cb(raw_command: str, args: List[str]) -> bool:
    """Block reflog expire/delete/exists (write to .git/logs/**); show/refs safe."""
    dangerous = {"expire", "delete", "exists"}
    for token in args:
        if not token or token.startswith("-"):
            continue
        if token in dangerous:
            return True
        return False  # first positional is show/ref → safe
    return False  # bare `git reflog` = show


def _git_remote_show_cb(raw_command: str, args: List[str]) -> bool:
    """Allow only optional -n then exactly one remote-name-looking positional."""
    positional = [a for a in args if a != "-n"]
    if len(positional) != 1:
        return True
    # fullmatch, not match+'$': Python '$' matches before a trailing newline, so
    # `origin\n` would pass and auto-allow. JS `/^[..]+$/.test` rejects it.
    return not re.fullmatch(r"[a-zA-Z0-9_-]+", positional[0])


def _git_remote_cb(raw_command: str, args: List[str]) -> bool:
    """Allow only bare `git remote` or `-v`/`--verbose`."""
    return any(a != "-v" and a != "--verbose" for a in args)


def _git_tag_cb(raw_command: str, args: List[str]) -> bool:
    """Block tag creation: a positional without -l/--list is a tag name to create."""
    flags_with_args = {"--contains", "--no-contains", "--merged", "--no-merged",
                       "--points-at", "--sort", "--format", "-n"}
    i = 0
    seen_list = False
    seen_dash_dash = False
    while i < len(args):
        token = args[i]
        if not token:
            i += 1
            continue
        if token == "--" and not seen_dash_dash:
            seen_dash_dash = True
            i += 1
            continue
        if not seen_dash_dash and token.startswith("-"):
            if token == "--list" or token == "-l":
                seen_list = True
            elif len(token) > 2 and token[1] != "-" and "=" not in token and "l" in token[1:]:
                seen_list = True
            if "=" in token:
                i += 1
            elif token in flags_with_args:
                i += 2
            else:
                i += 1
        else:
            if not seen_list:
                return True  # positional without --list = tag creation
            i += 1
    return False


def _git_branch_cb(raw_command: str, args: List[str]) -> bool:
    """Block branch creation: a positional without --list/filtering flag is a new branch."""
    flags_with_args = {"--contains", "--no-contains", "--points-at", "--sort"}
    flags_with_optional_args = {"--merged", "--no-merged"}
    i = 0
    last_flag = ""
    seen_list = False
    seen_dash_dash = False
    while i < len(args):
        token = args[i]
        if not token:
            i += 1
            continue
        if token == "--" and not seen_dash_dash:
            seen_dash_dash = True
            last_flag = ""
            i += 1
            continue
        if not seen_dash_dash and token.startswith("-"):
            if token == "--list" or token == "-l":
                seen_list = True
            elif len(token) > 2 and token[1] != "-" and "=" not in token and "l" in token[1:]:
                seen_list = True
            if "=" in token:
                last_flag = token.split("=")[0]
                i += 1
            elif token in flags_with_args:
                last_flag = token
                i += 2
            else:
                last_flag = token
                i += 1
        else:
            last_flag_optional = last_flag in flags_with_optional_args
            if not seen_list and not last_flag_optional:
                return True  # positional without --list/filter = branch creation
            i += 1
    return False


def _gh_is_dangerous_cb(raw_command: str, args: List[str]) -> bool:
    """Block gh repo/URL exfil: HOST/OWNER/REPO (2+ slashes), URLs, SSH-style."""
    for token in args:
        if not token:
            continue
        value = token
        if token.startswith("-"):
            eq = token.find("=")
            if eq == -1:
                continue
            value = token[eq + 1:]
            if not value:
                continue
        if "/" not in value and "://" not in value and "@" not in value:
            continue
        if "://" in value:
            return True
        if "@" in value:
            return True
        if value.count("/") >= 2:
            return True
    return False


def _pyright_cb(raw_command: str, args: List[str]) -> bool:
    """Block pyright --watch/-w (long-running, not read-only-and-exit)."""
    return any(t == "--watch" or t == "-w" for t in args)


# =============================================================================
# Read-only command allowlists (data extracted verbatim from source; callbacks
# wired by reference). GIT/GH/DOCKER/RIPGREP/PYRIGHT.
# =============================================================================
GIT_READ_ONLY_COMMANDS: Dict[str, ExternalCommandConfig] = {
    'git diff': ExternalCommandConfig(safe_flags={'--stat': 'none', '--numstat': 'none', '--shortstat': 'none', '--name-only': 'none', '--name-status': 'none', '--color': 'none', '--no-color': 'none', '--dirstat': 'none', '--summary': 'none', '--patch-with-stat': 'none', '--word-diff': 'none', '--word-diff-regex': 'string', '--color-words': 'none', '--no-renames': 'none', '--no-ext-diff': 'none', '--check': 'none', '--ws-error-highlight': 'string', '--full-index': 'none', '--binary': 'none', '--abbrev': 'number', '--break-rewrites': 'none', '--find-renames': 'none', '--find-copies': 'none', '--find-copies-harder': 'none', '--irreversible-delete': 'none', '--diff-algorithm': 'string', '--histogram': 'none', '--patience': 'none', '--minimal': 'none', '--ignore-space-at-eol': 'none', '--ignore-space-change': 'none', '--ignore-all-space': 'none', '--ignore-blank-lines': 'none', '--inter-hunk-context': 'number', '--function-context': 'none', '--exit-code': 'none', '--quiet': 'none', '--cached': 'none', '--staged': 'none', '--pickaxe-regex': 'none', '--pickaxe-all': 'none', '--no-index': 'none', '--relative': 'string', '--diff-filter': 'string', '-p': 'none', '-u': 'none', '-s': 'none', '-M': 'none', '-C': 'none', '-B': 'none', '-D': 'none', '-l': 'none', '-S': 'string', '-G': 'string', '-O': 'string', '-R': 'none'}, additional_command_is_dangerous_callback=None),
    'git log': ExternalCommandConfig(safe_flags={'--oneline': 'none', '--graph': 'none', '--decorate': 'none', '--no-decorate': 'none', '--date': 'string', '--relative-date': 'none', '--all': 'none', '--branches': 'none', '--tags': 'none', '--remotes': 'none', '--since': 'string', '--after': 'string', '--until': 'string', '--before': 'string', '--max-count': 'number', '-n': 'number', '--stat': 'none', '--numstat': 'none', '--shortstat': 'none', '--name-only': 'none', '--name-status': 'none', '--color': 'none', '--no-color': 'none', '--patch': 'none', '-p': 'none', '--no-patch': 'none', '--no-ext-diff': 'none', '-s': 'none', '--author': 'string', '--committer': 'string', '--grep': 'string', '--abbrev-commit': 'none', '--full-history': 'none', '--dense': 'none', '--sparse': 'none', '--simplify-merges': 'none', '--ancestry-path': 'none', '--source': 'none', '--first-parent': 'none', '--merges': 'none', '--no-merges': 'none', '--reverse': 'none', '--walk-reflogs': 'none', '--skip': 'number', '--max-age': 'number', '--min-age': 'number', '--no-min-parents': 'none', '--no-max-parents': 'none', '--follow': 'none', '--no-walk': 'none', '--left-right': 'none', '--cherry-mark': 'none', '--cherry-pick': 'none', '--boundary': 'none', '--topo-order': 'none', '--date-order': 'none', '--author-date-order': 'none', '--pretty': 'string', '--format': 'string', '--diff-filter': 'string', '-S': 'string', '-G': 'string', '--pickaxe-regex': 'none', '--pickaxe-all': 'none'}, additional_command_is_dangerous_callback=None),
    'git show': ExternalCommandConfig(safe_flags={'--oneline': 'none', '--graph': 'none', '--decorate': 'none', '--no-decorate': 'none', '--date': 'string', '--relative-date': 'none', '--stat': 'none', '--numstat': 'none', '--shortstat': 'none', '--name-only': 'none', '--name-status': 'none', '--color': 'none', '--no-color': 'none', '--patch': 'none', '-p': 'none', '--no-patch': 'none', '--no-ext-diff': 'none', '-s': 'none', '--abbrev-commit': 'none', '--word-diff': 'none', '--word-diff-regex': 'string', '--color-words': 'none', '--pretty': 'string', '--format': 'string', '--first-parent': 'none', '--raw': 'none', '--diff-filter': 'string', '-m': 'none', '--quiet': 'none'}, additional_command_is_dangerous_callback=None),
    'git shortlog': ExternalCommandConfig(safe_flags={'--all': 'none', '--branches': 'none', '--tags': 'none', '--remotes': 'none', '--since': 'string', '--after': 'string', '--until': 'string', '--before': 'string', '-s': 'none', '--summary': 'none', '-n': 'none', '--numbered': 'none', '-e': 'none', '--email': 'none', '-c': 'none', '--committer': 'none', '--group': 'string', '--format': 'string', '--no-merges': 'none', '--author': 'string'}, additional_command_is_dangerous_callback=None),
    'git reflog': ExternalCommandConfig(safe_flags={'--oneline': 'none', '--graph': 'none', '--decorate': 'none', '--no-decorate': 'none', '--date': 'string', '--relative-date': 'none', '--all': 'none', '--branches': 'none', '--tags': 'none', '--remotes': 'none', '--since': 'string', '--after': 'string', '--until': 'string', '--before': 'string', '--max-count': 'number', '-n': 'number', '--author': 'string', '--committer': 'string', '--grep': 'string'}, additional_command_is_dangerous_callback=_git_reflog_cb),
    'git stash list': ExternalCommandConfig(safe_flags={'--oneline': 'none', '--graph': 'none', '--decorate': 'none', '--no-decorate': 'none', '--date': 'string', '--relative-date': 'none', '--all': 'none', '--branches': 'none', '--tags': 'none', '--remotes': 'none', '--max-count': 'number', '-n': 'number'}, additional_command_is_dangerous_callback=None),
    'git ls-remote': ExternalCommandConfig(safe_flags={'--branches': 'none', '-b': 'none', '--tags': 'none', '-t': 'none', '--heads': 'none', '-h': 'none', '--refs': 'none', '--quiet': 'none', '-q': 'none', '--exit-code': 'none', '--get-url': 'none', '--symref': 'none', '--sort': 'string'}, additional_command_is_dangerous_callback=None),
    'git status': ExternalCommandConfig(safe_flags={'--short': 'none', '-s': 'none', '--branch': 'none', '-b': 'none', '--porcelain': 'none', '--long': 'none', '--verbose': 'none', '-v': 'none', '--untracked-files': 'string', '-u': 'string', '--ignored': 'none', '--ignore-submodules': 'string', '--column': 'none', '--no-column': 'none', '--ahead-behind': 'none', '--no-ahead-behind': 'none', '--renames': 'none', '--no-renames': 'none', '--find-renames': 'string', '-M': 'string'}, additional_command_is_dangerous_callback=None),
    'git blame': ExternalCommandConfig(safe_flags={'--color': 'none', '--no-color': 'none', '-L': 'string', '--porcelain': 'none', '-p': 'none', '--line-porcelain': 'none', '--incremental': 'none', '--root': 'none', '--show-stats': 'none', '--show-name': 'none', '--show-number': 'none', '-n': 'none', '--show-email': 'none', '-e': 'none', '-f': 'none', '--date': 'string', '-w': 'none', '--ignore-rev': 'string', '--ignore-revs-file': 'string', '-M': 'none', '-C': 'none', '--score-debug': 'none', '--abbrev': 'number', '-s': 'none', '-l': 'none', '-t': 'none'}, additional_command_is_dangerous_callback=None),
    'git ls-files': ExternalCommandConfig(safe_flags={'--cached': 'none', '-c': 'none', '--deleted': 'none', '-d': 'none', '--modified': 'none', '-m': 'none', '--others': 'none', '-o': 'none', '--ignored': 'none', '-i': 'none', '--stage': 'none', '-s': 'none', '--killed': 'none', '-k': 'none', '--unmerged': 'none', '-u': 'none', '--directory': 'none', '--no-empty-directory': 'none', '--eol': 'none', '--full-name': 'none', '--abbrev': 'number', '--debug': 'none', '-z': 'none', '-t': 'none', '-v': 'none', '-f': 'none', '--exclude': 'string', '-x': 'string', '--exclude-from': 'string', '-X': 'string', '--exclude-per-directory': 'string', '--exclude-standard': 'none', '--error-unmatch': 'none', '--recurse-submodules': 'none'}, additional_command_is_dangerous_callback=None),
    'git config --get': ExternalCommandConfig(safe_flags={'--local': 'none', '--global': 'none', '--system': 'none', '--worktree': 'none', '--default': 'string', '--type': 'string', '--bool': 'none', '--int': 'none', '--bool-or-int': 'none', '--path': 'none', '--expiry-date': 'none', '-z': 'none', '--null': 'none', '--name-only': 'none', '--show-origin': 'none', '--show-scope': 'none'}, additional_command_is_dangerous_callback=None),
    'git remote show': ExternalCommandConfig(safe_flags={'-n': 'none'}, additional_command_is_dangerous_callback=_git_remote_show_cb),
    'git remote': ExternalCommandConfig(safe_flags={'-v': 'none', '--verbose': 'none'}, additional_command_is_dangerous_callback=_git_remote_cb),
    'git merge-base': ExternalCommandConfig(safe_flags={'--is-ancestor': 'none', '--fork-point': 'none', '--octopus': 'none', '--independent': 'none', '--all': 'none'}, additional_command_is_dangerous_callback=None),
    'git rev-parse': ExternalCommandConfig(safe_flags={'--verify': 'none', '--short': 'string', '--abbrev-ref': 'none', '--symbolic': 'none', '--symbolic-full-name': 'none', '--show-toplevel': 'none', '--show-cdup': 'none', '--show-prefix': 'none', '--git-dir': 'none', '--git-common-dir': 'none', '--absolute-git-dir': 'none', '--show-superproject-working-tree': 'none', '--is-inside-work-tree': 'none', '--is-inside-git-dir': 'none', '--is-bare-repository': 'none', '--is-shallow-repository': 'none', '--is-shallow-update': 'none', '--path-prefix': 'none'}, additional_command_is_dangerous_callback=None),
    'git rev-list': ExternalCommandConfig(safe_flags={'--all': 'none', '--branches': 'none', '--tags': 'none', '--remotes': 'none', '--since': 'string', '--after': 'string', '--until': 'string', '--before': 'string', '--max-count': 'number', '-n': 'number', '--author': 'string', '--committer': 'string', '--grep': 'string', '--count': 'none', '--reverse': 'none', '--first-parent': 'none', '--ancestry-path': 'none', '--merges': 'none', '--no-merges': 'none', '--min-parents': 'number', '--max-parents': 'number', '--no-min-parents': 'none', '--no-max-parents': 'none', '--skip': 'number', '--max-age': 'number', '--min-age': 'number', '--walk-reflogs': 'none', '--oneline': 'none', '--abbrev-commit': 'none', '--pretty': 'string', '--format': 'string', '--abbrev': 'number', '--full-history': 'none', '--dense': 'none', '--sparse': 'none', '--source': 'none', '--graph': 'none'}, additional_command_is_dangerous_callback=None),
    'git describe': ExternalCommandConfig(safe_flags={'--tags': 'none', '--match': 'string', '--exclude': 'string', '--long': 'none', '--abbrev': 'number', '--always': 'none', '--contains': 'none', '--first-match': 'none', '--exact-match': 'none', '--candidates': 'number', '--dirty': 'none', '--broken': 'none'}, additional_command_is_dangerous_callback=None),
    'git cat-file': ExternalCommandConfig(safe_flags={'-t': 'none', '-s': 'none', '-p': 'none', '-e': 'none', '--batch-check': 'none', '--allow-undetermined-type': 'none'}, additional_command_is_dangerous_callback=None),
    'git for-each-ref': ExternalCommandConfig(safe_flags={'--format': 'string', '--sort': 'string', '--count': 'number', '--contains': 'string', '--no-contains': 'string', '--merged': 'string', '--no-merged': 'string', '--points-at': 'string'}, additional_command_is_dangerous_callback=None),
    'git grep': ExternalCommandConfig(safe_flags={'-e': 'string', '-E': 'none', '--extended-regexp': 'none', '-G': 'none', '--basic-regexp': 'none', '-F': 'none', '--fixed-strings': 'none', '-P': 'none', '--perl-regexp': 'none', '-i': 'none', '--ignore-case': 'none', '-v': 'none', '--invert-match': 'none', '-w': 'none', '--word-regexp': 'none', '-n': 'none', '--line-number': 'none', '-c': 'none', '--count': 'none', '-l': 'none', '--files-with-matches': 'none', '-L': 'none', '--files-without-match': 'none', '-h': 'none', '-H': 'none', '--heading': 'none', '--break': 'none', '--full-name': 'none', '--color': 'none', '--no-color': 'none', '-o': 'none', '--only-matching': 'none', '-A': 'number', '--after-context': 'number', '-B': 'number', '--before-context': 'number', '-C': 'number', '--context': 'number', '--and': 'none', '--or': 'none', '--not': 'none', '--max-depth': 'number', '--untracked': 'none', '--no-index': 'none', '--recurse-submodules': 'none', '--cached': 'none', '--threads': 'number', '-q': 'none', '--quiet': 'none'}, additional_command_is_dangerous_callback=None),
    'git stash show': ExternalCommandConfig(safe_flags={'--stat': 'none', '--numstat': 'none', '--shortstat': 'none', '--name-only': 'none', '--name-status': 'none', '--color': 'none', '--no-color': 'none', '--patch': 'none', '-p': 'none', '--no-patch': 'none', '--no-ext-diff': 'none', '-s': 'none', '--word-diff': 'none', '--word-diff-regex': 'string', '--diff-filter': 'string', '--abbrev': 'number'}, additional_command_is_dangerous_callback=None),
    'git worktree list': ExternalCommandConfig(safe_flags={'--porcelain': 'none', '-v': 'none', '--verbose': 'none', '--expire': 'string'}, additional_command_is_dangerous_callback=None),
    'git tag': ExternalCommandConfig(safe_flags={'-l': 'none', '--list': 'none', '-n': 'number', '--contains': 'string', '--no-contains': 'string', '--merged': 'string', '--no-merged': 'string', '--sort': 'string', '--format': 'string', '--points-at': 'string', '--column': 'none', '--no-column': 'none', '-i': 'none', '--ignore-case': 'none'}, additional_command_is_dangerous_callback=_git_tag_cb),
    'git branch': ExternalCommandConfig(safe_flags={'-l': 'none', '--list': 'none', '-a': 'none', '--all': 'none', '-r': 'none', '--remotes': 'none', '-v': 'none', '-vv': 'none', '--verbose': 'none', '--color': 'none', '--no-color': 'none', '--column': 'none', '--no-column': 'none', '--abbrev': 'number', '--no-abbrev': 'none', '--contains': 'string', '--no-contains': 'string', '--merged': 'none', '--no-merged': 'none', '--points-at': 'string', '--sort': 'string', '--show-current': 'none', '-i': 'none', '--ignore-case': 'none'}, additional_command_is_dangerous_callback=_git_branch_cb),
}

GH_READ_ONLY_COMMANDS: Dict[str, ExternalCommandConfig] = {
    'gh pr view': ExternalCommandConfig(safe_flags={'--json': 'string', '--comments': 'none', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh pr list': ExternalCommandConfig(safe_flags={'--state': 'string', '-s': 'string', '--author': 'string', '--assignee': 'string', '--label': 'string', '--limit': 'number', '-L': 'number', '--base': 'string', '--head': 'string', '--search': 'string', '--json': 'string', '--draft': 'none', '--app': 'string', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh pr diff': ExternalCommandConfig(safe_flags={'--color': 'string', '--name-only': 'none', '--patch': 'none', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh pr checks': ExternalCommandConfig(safe_flags={'--watch': 'none', '--required': 'none', '--fail-fast': 'none', '--json': 'string', '--interval': 'number', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh issue view': ExternalCommandConfig(safe_flags={'--json': 'string', '--comments': 'none', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh issue list': ExternalCommandConfig(safe_flags={'--state': 'string', '-s': 'string', '--assignee': 'string', '--author': 'string', '--label': 'string', '--limit': 'number', '-L': 'number', '--milestone': 'string', '--search': 'string', '--json': 'string', '--app': 'string', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh repo view': ExternalCommandConfig(safe_flags={'--json': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh run list': ExternalCommandConfig(safe_flags={'--branch': 'string', '-b': 'string', '--status': 'string', '-s': 'string', '--workflow': 'string', '-w': 'string', '--limit': 'number', '-L': 'number', '--json': 'string', '--repo': 'string', '-R': 'string', '--event': 'string', '-e': 'string', '--user': 'string', '-u': 'string', '--created': 'string', '--commit': 'string', '-c': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh run view': ExternalCommandConfig(safe_flags={'--log': 'none', '--log-failed': 'none', '--exit-status': 'none', '--verbose': 'none', '-v': 'none', '--json': 'string', '--repo': 'string', '-R': 'string', '--job': 'string', '-j': 'string', '--attempt': 'number', '-a': 'number'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh auth status': ExternalCommandConfig(safe_flags={'--active': 'none', '-a': 'none', '--hostname': 'string', '-h': 'string', '--json': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh pr status': ExternalCommandConfig(safe_flags={'--conflict-status': 'none', '-c': 'none', '--json': 'string', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh issue status': ExternalCommandConfig(safe_flags={'--json': 'string', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh release list': ExternalCommandConfig(safe_flags={'--exclude-drafts': 'none', '--exclude-pre-releases': 'none', '--json': 'string', '--limit': 'number', '-L': 'number', '--order': 'string', '-O': 'string', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh release view': ExternalCommandConfig(safe_flags={'--json': 'string', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh workflow list': ExternalCommandConfig(safe_flags={'--all': 'none', '-a': 'none', '--json': 'string', '--limit': 'number', '-L': 'number', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh workflow view': ExternalCommandConfig(safe_flags={'--ref': 'string', '-r': 'string', '--yaml': 'none', '-y': 'none', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh label list': ExternalCommandConfig(safe_flags={'--json': 'string', '--limit': 'number', '-L': 'number', '--order': 'string', '--search': 'string', '-S': 'string', '--sort': 'string', '--repo': 'string', '-R': 'string'}, additional_command_is_dangerous_callback=_gh_is_dangerous_cb),
    'gh search repos': ExternalCommandConfig(safe_flags={'--archived': 'none', '--created': 'string', '--followers': 'string', '--forks': 'string', '--good-first-issues': 'string', '--help-wanted-issues': 'string', '--include-forks': 'string', '--json': 'string', '--language': 'string', '--license': 'string', '--limit': 'number', '-L': 'number', '--match': 'string', '--number-topics': 'string', '--order': 'string', '--owner': 'string', '--size': 'string', '--sort': 'string', '--stars': 'string', '--topic': 'string', '--updated': 'string', '--visibility': 'string'}, additional_command_is_dangerous_callback=None),
    'gh search issues': ExternalCommandConfig(safe_flags={'--app': 'string', '--assignee': 'string', '--author': 'string', '--closed': 'string', '--commenter': 'string', '--comments': 'string', '--created': 'string', '--include-prs': 'none', '--interactions': 'string', '--involves': 'string', '--json': 'string', '--label': 'string', '--language': 'string', '--limit': 'number', '-L': 'number', '--locked': 'none', '--match': 'string', '--mentions': 'string', '--milestone': 'string', '--no-assignee': 'none', '--no-label': 'none', '--no-milestone': 'none', '--no-project': 'none', '--order': 'string', '--owner': 'string', '--project': 'string', '--reactions': 'string', '--repo': 'string', '-R': 'string', '--sort': 'string', '--state': 'string', '--team-mentions': 'string', '--updated': 'string', '--visibility': 'string'}, additional_command_is_dangerous_callback=None),
    'gh search prs': ExternalCommandConfig(safe_flags={'--app': 'string', '--assignee': 'string', '--author': 'string', '--base': 'string', '-B': 'string', '--checks': 'string', '--closed': 'string', '--commenter': 'string', '--comments': 'string', '--created': 'string', '--draft': 'none', '--head': 'string', '-H': 'string', '--interactions': 'string', '--involves': 'string', '--json': 'string', '--label': 'string', '--language': 'string', '--limit': 'number', '-L': 'number', '--locked': 'none', '--match': 'string', '--mentions': 'string', '--merged': 'none', '--merged-at': 'string', '--milestone': 'string', '--no-assignee': 'none', '--no-label': 'none', '--no-milestone': 'none', '--no-project': 'none', '--order': 'string', '--owner': 'string', '--project': 'string', '--reactions': 'string', '--repo': 'string', '-R': 'string', '--review': 'string', '--review-requested': 'string', '--reviewed-by': 'string', '--sort': 'string', '--state': 'string', '--team-mentions': 'string', '--updated': 'string', '--visibility': 'string'}, additional_command_is_dangerous_callback=None),
    'gh search commits': ExternalCommandConfig(safe_flags={'--author': 'string', '--author-date': 'string', '--author-email': 'string', '--author-name': 'string', '--committer': 'string', '--committer-date': 'string', '--committer-email': 'string', '--committer-name': 'string', '--hash': 'string', '--json': 'string', '--limit': 'number', '-L': 'number', '--merge': 'none', '--order': 'string', '--owner': 'string', '--parent': 'string', '--repo': 'string', '-R': 'string', '--sort': 'string', '--tree': 'string', '--visibility': 'string'}, additional_command_is_dangerous_callback=None),
    'gh search code': ExternalCommandConfig(safe_flags={'--extension': 'string', '--filename': 'string', '--json': 'string', '--language': 'string', '--limit': 'number', '-L': 'number', '--match': 'string', '--owner': 'string', '--repo': 'string', '-R': 'string', '--size': 'string'}, additional_command_is_dangerous_callback=None),
}

DOCKER_READ_ONLY_COMMANDS: Dict[str, ExternalCommandConfig] = {
    'docker logs': ExternalCommandConfig(safe_flags={'--follow': 'none', '-f': 'none', '--tail': 'string', '-n': 'string', '--timestamps': 'none', '-t': 'none', '--since': 'string', '--until': 'string', '--details': 'none'}, additional_command_is_dangerous_callback=None),
    'docker inspect': ExternalCommandConfig(safe_flags={'--format': 'string', '-f': 'string', '--type': 'string', '--size': 'none', '-s': 'none'}, additional_command_is_dangerous_callback=None),
}

RIPGREP_READ_ONLY_COMMANDS: Dict[str, ExternalCommandConfig] = {
    'rg': ExternalCommandConfig(safe_flags={'-e': 'string', '--regexp': 'string', '-f': 'string', '-i': 'none', '--ignore-case': 'none', '-S': 'none', '--smart-case': 'none', '-F': 'none', '--fixed-strings': 'none', '-w': 'none', '--word-regexp': 'none', '-v': 'none', '--invert-match': 'none', '-c': 'none', '--count': 'none', '-l': 'none', '--files-with-matches': 'none', '--files-without-match': 'none', '-n': 'none', '--line-number': 'none', '-o': 'none', '--only-matching': 'none', '-A': 'number', '--after-context': 'number', '-B': 'number', '--before-context': 'number', '-C': 'number', '--context': 'number', '-H': 'none', '-h': 'none', '--heading': 'none', '--no-heading': 'none', '-q': 'none', '--quiet': 'none', '--column': 'none', '-g': 'string', '--glob': 'string', '-t': 'string', '--type': 'string', '-T': 'string', '--type-not': 'string', '--type-list': 'none', '--hidden': 'none', '--no-ignore': 'none', '-u': 'none', '-m': 'number', '--max-count': 'number', '-d': 'number', '--max-depth': 'number', '-a': 'none', '--text': 'none', '-z': 'none', '-L': 'none', '--follow': 'none', '--color': 'string', '--json': 'none', '--stats': 'none', '--help': 'none', '--version': 'none', '--debug': 'none', '--': 'none'}, additional_command_is_dangerous_callback=None),
}

PYRIGHT_READ_ONLY_COMMANDS: Dict[str, ExternalCommandConfig] = {
    'pyright': ExternalCommandConfig(safe_flags={'--outputjson': 'none', '--project': 'string', '-p': 'string', '--pythonversion': 'string', '--pythonplatform': 'string', '--typeshedpath': 'string', '--venvpath': 'string', '--level': 'string', '--stats': 'none', '--verbose': 'none', '--version': 'none', '--dependencies': 'none', '--warnings': 'none'}, additional_command_is_dangerous_callback=_pyright_cb, respects_double_dash=False),
}
