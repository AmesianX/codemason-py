"""
Tool descriptions — based on Claude Code tool prompts.

Source of truth: extracted/source/src/tools/<Tool>/prompt.ts

These are the descriptions sent to the model in the tool schema (the original
sends each tool's full prompt, not a one-line blurb). Dynamic/feature-gated bits
are resolved for the EXTERNAL local build:
  - hasEmbeddedSearchTools() = False  → steer toward Glob/Grep
  - SandboxManager disabled            → no sandbox section
  - isPDFSupported() = True            → include PDF guidance
  - USER_TYPE != 'ant'                 → no ant-only hints
"""

from .prompts import (
    BASH_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
    FILE_EDIT_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
    AGENT_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
)


# === Read (FileReadTool/prompt.ts renderPromptTemplate) ===

MAX_LINES_TO_READ = 2000

READ_DESCRIPTION = f"""Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to {MAX_LINES_TO_READ} lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Results are returned using cat -n format, with line numbers starting at 1
- This tool allows the assistant to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as the assistant is a multimodal LLM.
- This tool can read PDF files (.pdf). For large PDFs (more than 10 pages), you MUST provide the pages parameter to read specific page ranges (e.g., pages: "1-5"). Reading a large PDF without the pages parameter will fail. Maximum 20 pages per request.
- This tool can read Jupyter notebooks (.ipynb files) and returns all cells with their outputs, combining code, text, and visualizations.
- This tool can only read files, not directories. To read a directory, use an ls command via the {BASH_TOOL_NAME} tool.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents."""


# === Write (FileWriteTool/prompt.ts getWriteToolDescription) ===

WRITE_DESCRIPTION = f"""Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the {FILE_READ_TOOL_NAME} tool first to read the file's contents. This tool will fail if you did not read the file first.
- Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""


# === Edit (FileEditTool/prompt.ts getDefaultEditDescription) ===
# prefixFormat resolves to 'line number + tab': for external builds
# isCompactLinePrefixEnabled() defaults to True (killswitch off), and this
# matches the actual tab-prefixed Read output.

EDIT_DESCRIPTION = f"""Performs exact string replacements in files.

Usage:
- You must use your `{FILE_READ_TOOL_NAME}` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + tab. Everything after that is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""


# === Glob (GlobTool/prompt.ts DESCRIPTION) ===

GLOB_DESCRIPTION = f"""- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the {AGENT_TOOL_NAME} tool instead"""


# === Grep (GrepTool/prompt.ts getDescription) ===

GREP_DESCRIPTION = f"""A powerful search tool built on ripgrep

  Usage:
  - ALWAYS use {GREP_TOOL_NAME} for search tasks. NEVER invoke `grep` or `rg` as a {BASH_TOOL_NAME} command. The {GREP_TOOL_NAME} tool has been optimized for correct permissions and access.
  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
  - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
  - Use {AGENT_TOOL_NAME} tool for open-ended searches requiring multiple rounds
  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{{\\}}` to find `interface{{}}` in Go code)
  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{{[\\s\\S]*?field`, use `multiline: true`
"""


# === Bash (BashTool/prompt.ts getSimplePrompt) ===
# Resolved external: embedded=False, sandbox disabled, MONITOR_TOOL feature off.
# Default timeout 120000ms (2 min), max 600000ms (10 min).
# The external prompt appends getCommitAndPRInstructions() — for external builds
# (USER_TYPE != 'ant') with shouldIncludeGitInstructions() defaulting True, that
# is the FULL inline git commit/PR workflow block (prompt.ts:81-160), ported
# verbatim below and appended to BASH_DESCRIPTION. Attribution texts are the
# standard external values (Co-Authored-By commit line + "🤖 Generated with ..."
# PR line) that getAttributionTexts() emits for an external build.

_BASH_DEFAULT_TIMEOUT_MS = 120000
_BASH_MAX_TIMEOUT_MS = 600000

# Standard external attribution texts (getAttributionTexts external default).
# The template emits these as the commit message footer and PR body footer.
_COMMIT_ATTRIBUTION = "Co-Authored-By: CodeMason <noreply@codemason.local>"
_PR_ATTRIBUTION = "🤖 Generated with CodeMason"

BASH_DESCRIPTION = f"""Executes a given bash command and returns its output.

The working directory persists between commands, but shell state does not. The shell environment is initialized from the user's profile (bash or zsh).

IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience for the user:

 - File search: Use {GLOB_TOOL_NAME} (NOT find or ls)
 - Content search: Use {GREP_TOOL_NAME} (NOT grep or rg)
 - Read files: Use {FILE_READ_TOOL_NAME} (NOT cat/head/tail)
 - Edit files: Use {FILE_EDIT_TOOL_NAME} (NOT sed/awk)
 - Write files: Use {FILE_WRITE_TOOL_NAME} (NOT echo >/cat <<EOF)
 - Communication: Output text directly (NOT echo/printf)
While the {BASH_TOOL_NAME} tool can do similar things, it's better to use the built-in tools as they provide a better user experience and make it easier to review tool calls and give permission.

# Instructions
 - If your command will create new directories or files, first use this tool to run `ls` to verify the parent directory exists and is the correct location.
 - Always quote file paths that contain spaces with double quotes in your command (e.g., cd "path with spaces/file.txt")
 - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.
 - You may specify an optional timeout in milliseconds (up to {_BASH_MAX_TIMEOUT_MS}ms / {_BASH_MAX_TIMEOUT_MS // 60000} minutes). By default, your command will timeout after {_BASH_DEFAULT_TIMEOUT_MS}ms ({_BASH_DEFAULT_TIMEOUT_MS // 60000} minutes).
 - You can use the `run_in_background` parameter to run the command in the background. Only use this if you don't need the result immediately and are OK being notified when the command completes later. You do not need to check the output right away - you'll be notified when it finishes. You do not need to use '&' at the end of the command when using this parameter.
 - When issuing multiple commands:
  - If the commands are independent and can run in parallel, make multiple {BASH_TOOL_NAME} tool calls in a single message. Example: if you need to run "git status" and "git diff", send a single message with two {BASH_TOOL_NAME} tool calls in parallel.
  - If the commands depend on each other and must run sequentially, use a single {BASH_TOOL_NAME} call with '&&' to chain them together.
  - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail.
  - DO NOT use newlines to separate commands (newlines are ok in quoted strings).
 - For git commands:
  - Prefer to create a new commit rather than amending an existing commit.
  - Before running destructive operations (e.g., git reset --hard, git push --force, git checkout --), consider whether there is a safer alternative that achieves the same goal. Only use destructive operations when they are truly the best approach.
  - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign, -c commit.gpgsign=false) unless the user has explicitly asked for it. If a hook fails, investigate and fix the underlying issue.
 - Avoid unnecessary `sleep` commands:
  - Do not sleep between commands that can run immediately — just run them.
  - If your command is long running and you would like to be notified when it finishes — use `run_in_background`. No sleep needed.
  - Do not retry failing commands in a sleep loop — diagnose the root cause.
  - If waiting for a background task you started with `run_in_background`, you will be notified when it completes — do not poll.
  - If you must poll an external process, use a check command (e.g. `gh run view`) rather than sleeping first.
  - If you must sleep, keep the duration short (1-5 seconds) to avoid blocking the user.

# Committing changes with git

Only create commits when requested by the user. If unclear, ask first. When the user asks you to create a new git commit, follow these steps carefully:

You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. The numbered steps below indicate which commands should be batched in parallel.

Git Safety Protocol:
- NEVER update the git config
- NEVER run destructive git commands (push --force, reset --hard, checkout ., restore ., clean -f, branch -D) unless the user explicitly requests these actions. Taking unauthorized destructive actions is unhelpful and can result in lost work, so it's best to ONLY run these commands when given direct instructions
- NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless the user explicitly requests it
- NEVER run force push to main/master, warn the user if they request it
- CRITICAL: Always create NEW commits rather than amending, unless the user explicitly requests a git amend. When a pre-commit hook fails, the commit did NOT happen — so --amend would modify the PREVIOUS commit, which may result in destroying work or losing previous changes. Instead, after hook failure, fix the issue, re-stage, and create a NEW commit
- When staging files, prefer adding specific files by name rather than using "git add -A" or "git add .", which can accidentally include sensitive files (.env, credentials) or large binaries
- NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive

1. Run the following bash commands in parallel, each using the {BASH_TOOL_NAME} tool:
  - Run a git status command to see all untracked files. IMPORTANT: Never use the -uall flag as it can cause memory issues on large repos.
  - Run a git diff command to see both staged and unstaged changes that will be committed.
  - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.
2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:
  - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. "add" means a wholly new feature, "update" means an enhancement to an existing feature, "fix" means a bug fix, etc.).
  - Do not commit files that likely contain secrets (.env, credentials.json, etc). Warn the user if they specifically request to commit those files
  - Draft a concise (1-2 sentences) commit message that focuses on the "why" rather than the "what"
  - Ensure it accurately reflects the changes and their purpose
3. Run the following commands in parallel:
   - Add relevant untracked files to the staging area.
   - Create the commit with a message ending with:
   {_COMMIT_ATTRIBUTION}
   - Run git status after the commit completes to verify success.
   Note: git status depends on the commit completing, so run it sequentially after the commit.
4. If the commit fails due to pre-commit hook: fix the issue and create a NEW commit

Important notes:
- NEVER run additional commands to read or explore code, besides git bash commands
- NEVER use the {TODO_WRITE_TOOL_NAME} or {AGENT_TOOL_NAME} tools
- DO NOT push to the remote repository unless the user explicitly asks you to do so
- IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.
- IMPORTANT: Do not use --no-edit with git rebase commands, as the --no-edit flag is not a valid option for git rebase.
- If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit
- In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:
<example>
git commit -m "$(cat <<'EOF'
   Commit message here.

   {_COMMIT_ATTRIBUTION}
   EOF
   )"
</example>

# Creating pull requests
Use the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases. If given a Github URL use the gh command to get the information needed.

IMPORTANT: When the user asks you to create a pull request, follow these steps carefully:

1. Run the following bash commands in parallel using the {BASH_TOOL_NAME} tool, in order to understand the current state of the branch since it diverged from the main branch:
   - Run a git status command to see all untracked files (never use -uall flag)
   - Run a git diff command to see both staged and unstaged changes that will be committed
   - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote
   - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch)
2. Analyze all changes that will be included in the pull request, making sure to look at all relevant commits (NOT just the latest commit, but ALL commits that will be included in the pull request!!!), and draft a pull request title and summary:
   - Keep the PR title short (under 70 characters)
   - Use the description/body for details, not the title
3. Run the following commands in parallel:
   - Create new branch if needed
   - Push to remote with -u flag if needed
   - Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting.
<example>
gh pr create --title "the pr title" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points>

## Test plan
[Bulleted markdown checklist of TODOs for testing the pull request...]

{_PR_ATTRIBUTION}
EOF
)"
</example>

Important:
- DO NOT use the {TODO_WRITE_TOOL_NAME} or {AGENT_TOOL_NAME} tools
- Return the PR URL when you're done, so the user can see it

# Other common operations
- View comments on a Github PR: gh api repos/foo/bar/pulls/123/comments"""


# === Agent (AgentTool/prompt.ts getPrompt) ===
# Adapted to codemason's actual Agent: a single-`prompt`, foreground, stateless
# subprocess. The latest prompt advertises subagent_type / background /
# worktree / SendMessage / description params that codemason's Agent does not
# have yet — those are deliberately omitted so a local model isn't told to call
# parameters that don't exist. One deliberate, non-canonical addition is kept:
# the "Write the agent prompt in the user's language" note — the global
# "# Language" section makes the model respond in the user's language, but local
# models still tend to write the *agent prompt parameter* (an internal task spec)
# in English, so this one line nudges it. Everything else stays faithful.

# AgentTool supplies its own wire description (tools/built_in_agents.py
# build_agent_description, reached through AgentTool.DESCRIPTION), so this map
# never resolves Agent and a copy here could only drift out of it — it had,
# by two usage notes and the wording of the language rule.
# Map tool name -> ported description, for the API request builder.
TOOL_DESCRIPTIONS = {
    FILE_READ_TOOL_NAME: READ_DESCRIPTION,
    FILE_WRITE_TOOL_NAME: WRITE_DESCRIPTION,
    FILE_EDIT_TOOL_NAME: EDIT_DESCRIPTION,
    GLOB_TOOL_NAME: GLOB_DESCRIPTION,
    GREP_TOOL_NAME: GREP_DESCRIPTION,
    BASH_TOOL_NAME: BASH_DESCRIPTION,
}
