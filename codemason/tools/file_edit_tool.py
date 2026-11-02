"""
File Edit Tool for CodeMason
Based on src/tools/FileEditTool/ from Claude Code

The tool name is 'Edit' (FILE_EDIT_TOOL_NAME). The input schema, DESCRIPTION,
prompt text, validation behavior (old_string uniqueness, replace_all,
read-before-edit, curly-quote normalization) and the apply/patch logic are ported
verbatim from the implementation. File I/O uses codemason's local
substrate; read-before-edit uses ToolUseContext.read_file_state (a
path -> ReadFileState map, populated by FileRead). Line-ending and encoding preservation
are reproduced as far as the local substrate allows.
"""

import difflib
import os
import re
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from pydantic import Field

from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
    ValidationResultFailure,
)
from ..utils.formatting import Colors, colorize, format_edit_diff_preview
from ..utils.state import get_cwd
from ..types.query import ReadFileState


# === Constants (constants.ts) ===

FILE_EDIT_TOOL_NAME = 'Edit'

FILE_UNEXPECTEDLY_MODIFIED_ERROR = (
    'File has been unexpectedly modified. Read it again before attempting to write it.'
)

# NotebookEditTool/constants.ts
NOTEBOOK_EDIT_TOOL_NAME = 'NotebookEdit'

# FileReadTool/prompt.ts
FILE_READ_TOOL_NAME = 'Read'

# V8/Bun string length limit guard. 1 GiB (stat bytes).
MAX_EDIT_FILE_SIZE = 1024 * 1024 * 1024


# === Description / prompt (prompt.ts, verbatim) ===

# FileEditTool.ts description() returns this string.
DESCRIPTION = 'A tool for editing files'


def _format_file_size(num_bytes: int) -> str:
    """Local approximation of utils/format.formatFileSize."""
    size = float(num_bytes)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024.0 or unit == 'TB':
            if unit == 'B':
                return f'{int(size)}{unit}'
            return f'{size:.1f}{unit}'
        size /= 1024.0
    return f'{num_bytes}B'


def _get_pre_read_instruction() -> str:
    # prompt.ts getPreReadInstruction()
    return (
        f"\n- You must use your `{FILE_READ_TOOL_NAME}` tool at least once in the "
        "conversation before editing. This tool will error if you attempt an edit "
        "without reading the file. "
    )


def get_edit_tool_description() -> str:
    """prompt.ts getEditToolDescription() -> getDefaultEditDescription()."""
    # isCompactLinePrefixEnabled() is environment-gated; the default substrate
    # uses the non-compact prefix format.
    prefix_format = 'spaces + line number + arrow'
    # USER_TYPE === 'ant' adds a minimal-uniqueness hint; off by default.
    minimal_uniqueness_hint = (
        "\n- Use the smallest old_string that's clearly unique — usually 2-4 "
        "adjacent lines is sufficient. Avoid including 10+ lines of context when "
        "less uniquely identifies the target."
        if os.environ.get('USER_TYPE') == 'ant'
        else ''
    )
    return (
        "Performs exact string replacements in files.\n"
        "\n"
        f"Usage:{_get_pre_read_instruction()}\n"
        "- When editing text from Read tool output, ensure you preserve the exact "
        "indentation (tabs/spaces) as it appears AFTER the line number prefix. The "
        f"line number prefix format is: {prefix_format}. Everything after that is "
        "the actual file content to match. Never include any part of the line "
        "number prefix in the old_string or new_string.\n"
        "- ALWAYS prefer editing existing files in the codebase. NEVER write new "
        "files unless explicitly required.\n"
        "- Only use emojis if the user explicitly requests it. Avoid adding emojis "
        "to files unless asked.\n"
        "- The edit will FAIL if `old_string` is not unique in the file. Either "
        "provide a larger string with more surrounding context to make it unique or "
        f"use `replace_all` to change every instance of `old_string`.{minimal_uniqueness_hint}\n"
        "- Use `replace_all` for replacing and renaming strings across the file. "
        "This parameter is useful if you want to rename a variable for instance."
    )



# === Input schema (types.ts, verbatim) ===

class FileEditInput(ToolInputSchema):
    file_path: str = Field(..., description="The absolute path to the file to modify")
    old_string: str = Field(..., description="The text to replace")
    new_string: str = Field(
        ...,
        description="The text to replace it with (must be different from old_string)",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default false)",
    )


# === Quote normalization (utils.ts) ===

LEFT_SINGLE_CURLY_QUOTE = '‘'
RIGHT_SINGLE_CURLY_QUOTE = '’'
LEFT_DOUBLE_CURLY_QUOTE = '“'
RIGHT_DOUBLE_CURLY_QUOTE = '”'


def normalize_quotes(s: str) -> str:
    """utils.ts normalizeQuotes()."""
    return (
        s.replace(LEFT_SINGLE_CURLY_QUOTE, "'")
        .replace(RIGHT_SINGLE_CURLY_QUOTE, "'")
        .replace(LEFT_DOUBLE_CURLY_QUOTE, '"')
        .replace(RIGHT_DOUBLE_CURLY_QUOTE, '"')
    )


# === De-sanitization / trailing-whitespace normalization (utils.ts) ===

# utils.ts DESANITIZATIONS — replacements to de-sanitize strings from Claude.
# Since Claude can't see any of these strings (they are sanitized out in the
# API), it outputs the sanitized versions in the edit response. Copied verbatim
# from the source; insertion order is significant (Python dicts preserve
# it, matching the JS Object.entries iteration order).
DESANITIZATIONS: Dict[str, str] = {
    '<fnr>': '<function_results>',
    '<n>': '<name>',
    '</n>': '</name>',
    '<o>': '<output>',
    '</o>': '</output>',
    '<e>': '<error>',
    '</e>': '</error>',
    '<s>': '<system>',
    '</s>': '</system>',
    '<r>': '<result>',
    '</r>': '</result>',
    '< META_START >': '<META_START>',
    '< META_END >': '<META_END>',
    '< EOT >': '<EOT>',
    '< META >': '<META>',
    '< SOS >': '<SOS>',
    '\n\nH:': '\n\nHuman:',
    '\n\nA:': '\n\nAssistant:',
}


def strip_trailing_whitespace(s: str) -> str:
    """
    utils.ts stripTrailingWhitespace() — strips trailing whitespace from each
    line while preserving line endings (CRLF, LF, CR).
    """
    # Split capturing the line endings, mirroring str.split(/(\r\n|\n|\r)/).
    parts = re.split(r'(\r\n|\n|\r)', s)
    result: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Even indices are line content.
            result.append(re.sub(r'\s+$', '', part))
        else:
            # Odd indices are line endings.
            result.append(part)
    return ''.join(result)


def desanitize_match_string(match_string: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    utils.ts desanitizeMatchString() — applies the DESANITIZATIONS replacements,
    returning the rewritten string and the list of (from, to) pairs that
    actually changed something.
    """
    result = match_string
    applied_replacements: List[Tuple[str, str]] = []
    for frm, to in DESANITIZATIONS.items():
        before_replace = result
        result = result.replace(frm, to)
        if before_replace != result:
            applied_replacements.append((frm, to))
    return result, applied_replacements


def normalize_file_edit_input(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    file_content: Optional[str],
) -> Tuple[str, str, bool]:
    """
    utils.ts normalizeFileEditInput() — pure-string normalization applied before
    findActualString.

    - strips trailing whitespace from new_string, EXCEPT for .md/.mdx files where
      two trailing spaces are a hard line break (original utils.ts:597).
    - if old_string is not an exact substring of the file, retries with the
      de-sanitized version; when that matches, the same applied replacements are
      mirrored onto new_string.

    file_content is None when the file could not be read (e.g. it doesn't exist
    yet); in that case only the trailing-whitespace normalization applies, which
    matches the original's catch-on-ENOENT-returns-original-input behavior for
    old_string while still normalizing new_string.
    """
    # Markdown uses two trailing spaces as a hard line break — stripping would
    # silently change semantics. Skip stripTrailingWhitespace for .md/.mdx.
    is_markdown = bool(re.search(r'\.(md|mdx)$', file_path, re.IGNORECASE))

    normalized_new_string = new_string if is_markdown else strip_trailing_whitespace(new_string)

    # If the file couldn't be read, return original old_string (the source's
    # catch block returns the original input unchanged) with normalized new.
    if file_content is None:
        return old_string, normalized_new_string, replace_all

    # If exact string match works, keep it as is.
    if old_string in file_content:
        return old_string, normalized_new_string, replace_all

    # Try de-sanitize string if exact match fails.
    desanitized_old_string, applied_replacements = desanitize_match_string(old_string)

    if desanitized_old_string in file_content:
        # Apply the same exact replacements to new_string.
        desanitized_new_string = normalized_new_string
        for frm, to in applied_replacements:
            desanitized_new_string = desanitized_new_string.replace(frm, to)
        return desanitized_old_string, desanitized_new_string, replace_all

    return old_string, normalized_new_string, replace_all


def find_actual_string(file_content: str, search_string: str) -> Optional[str]:
    """utils.ts findActualString() — handles quote normalization."""
    # First try exact match
    if search_string in file_content:
        return search_string

    # Try with normalized quotes
    normalized_search = normalize_quotes(search_string)
    normalized_file = normalize_quotes(file_content)

    search_index = normalized_file.find(normalized_search)
    if search_index != -1:
        return file_content[search_index:search_index + len(search_string)]

    return None


def _is_opening_context(chars: List[str], index: int) -> bool:
    """utils.ts isOpeningContext()."""
    if index == 0:
        return True
    prev = chars[index - 1]
    return prev in (
        ' ', '\t', '\n', '\r', '(', '[', '{', '—', '–',
    )


def _apply_curly_double_quotes(s: str) -> str:
    chars = list(s)
    result: List[str] = []
    for i, ch in enumerate(chars):
        if ch == '"':
            result.append(
                LEFT_DOUBLE_CURLY_QUOTE
                if _is_opening_context(chars, i)
                else RIGHT_DOUBLE_CURLY_QUOTE
            )
        else:
            result.append(ch)
    return ''.join(result)


def _apply_curly_single_quotes(s: str) -> str:
    chars = list(s)
    result: List[str] = []
    for i, ch in enumerate(chars):
        if ch == "'":
            prev = chars[i - 1] if i > 0 else None
            nxt = chars[i + 1] if i < len(chars) - 1 else None
            prev_is_letter = prev is not None and _is_letter(prev)
            next_is_letter = nxt is not None and _is_letter(nxt)
            if prev_is_letter and next_is_letter:
                # Apostrophe in a contraction — use right single curly quote
                result.append(RIGHT_SINGLE_CURLY_QUOTE)
            else:
                result.append(
                    LEFT_SINGLE_CURLY_QUOTE
                    if _is_opening_context(chars, i)
                    else RIGHT_SINGLE_CURLY_QUOTE
                )
        else:
            result.append(ch)
    return ''.join(result)


def _is_letter(ch: str) -> bool:
    # Equivalent of /\p{L}/u.test(ch)
    return bool(re.match(r'[^\W\d_]', ch, re.UNICODE))


def preserve_quote_style(
    old_string: str, actual_old_string: str, new_string: str
) -> str:
    """utils.ts preserveQuoteStyle()."""
    if old_string == actual_old_string:
        return new_string

    has_double_quotes = (
        LEFT_DOUBLE_CURLY_QUOTE in actual_old_string
        or RIGHT_DOUBLE_CURLY_QUOTE in actual_old_string
    )
    has_single_quotes = (
        LEFT_SINGLE_CURLY_QUOTE in actual_old_string
        or RIGHT_SINGLE_CURLY_QUOTE in actual_old_string
    )

    if not has_double_quotes and not has_single_quotes:
        return new_string

    result = new_string
    if has_double_quotes:
        result = _apply_curly_double_quotes(result)
    if has_single_quotes:
        result = _apply_curly_single_quotes(result)
    return result


# === Edit application (utils.ts) ===

def apply_edit_to_file(
    original_content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """utils.ts applyEditToFile()."""
    def f(content: str, search: str, replace: str) -> str:
        if replace_all:
            return content.replace(search, replace)
        return content.replace(search, replace, 1)

    if new_string != '':
        return f(original_content, old_string, new_string)

    strip_trailing_newline = (
        not old_string.endswith('\n') and (old_string + '\n') in original_content
    )

    if strip_trailing_newline:
        return f(original_content, old_string + '\n', new_string)
    return f(original_content, old_string, new_string)


def get_updated_file_for_edit(
    file_contents: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """
    utils.ts getPatchForEdit()/getPatchForEdits() — returns the updated file
    contents. (The display patch is not reproduced; the local substrate does not
    render structured diffs.)
    """
    # Special case for empty files.
    if not file_contents and old_string == '' and new_string == '':
        return ''

    if old_string == '':
        updated_file = new_string
    else:
        updated_file = apply_edit_to_file(
            file_contents, old_string, new_string, replace_all
        )

    if updated_file == file_contents:
        raise ValueError('String not found in file. Failed to apply edit.')

    return updated_file


# === Path / line-ending / encoding helpers ===

def _expand_path(file_path: str) -> str:
    """utils/path.expandPath + cwd resolution for relative paths."""
    expanded = os.path.expanduser(file_path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(get_cwd(), expanded)
    return expanded


def _read_file_for_edit(absolute_file_path: str) -> Tuple[str, bool, str, str]:
    """
    Returns (content, file_exists, encoding, line_endings).

    Content is normalized to LF for matching (mirrors readFileForEdit + the
    validateInput byte read that does .replaceAll('\\r\\n', '\\n')). The detected
    encoding and line-ending style are returned so they can be restored on write.
    """
    try:
        with open(absolute_file_path, 'rb') as fh:
            raw = fh.read()
    except FileNotFoundError:
        return '', False, 'utf-8', 'LF'

    # Use the shared detector rather than a local utf-16-le-BOM-or-utf-8 guess.
    # The old guess sent every non-UTF-8 file through decode(errors='replace'),
    # and _write_text_content re-encodes and writes back what was decoded — so
    # editing one ASCII line of a CP949/EUC-KR source (Korean Windows' default)
    # replaced every Hangul byte in the file with U+FFFD, permanently. The
    # detector's latin-1 fallback maps bytes 1:1, so an undecodable file now
    # round-trips byte-for-byte instead. This is also what file_write_tool
    # already does, and what its comment claims Edit was doing.
    from ..utils.file import (
        detect_file_encoding, is_legacy_label, verified_legacy_encoding_exact,
    )
    encoding = detect_file_encoding(absolute_file_path)

    # A legacy label was proved against the detector's 4096-byte HEADER. Edit
    # rewrites the WHOLE file, so it has to hold for the whole file: re-verify
    # against `raw` and drop to byte-preserving latin-1 if it does not. The Rust
    # port's round-trip test caught exactly this gap — without it a CP949 source
    # was read as U+FFFD and written back that way, which is the original
    # data-loss bug.
    if is_legacy_label(encoding):
        encoding = verified_legacy_encoding_exact(raw) or 'latin-1'

    try:
        content = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        # The detector only samples the first 4096 bytes, so a file that turns
        # invalid later still lands here. latin-1 cannot fail and preserves the
        # bytes, which keeps the write-back lossless.
        encoding = 'latin-1'
        content = raw.decode(encoding)
    line_endings = 'CRLF' if '\r\n' in content else 'LF'
    # Normalize to LF for matching.
    content = content.replace('\r\n', '\n')
    return content, True, encoding, line_endings


def _write_text_content(
    absolute_file_path: str, content: str, encoding: str, line_endings: str
) -> None:
    """utils/file.writeTextContent — restore line endings + encoding on write."""
    out = content
    if line_endings == 'CRLF':
        # Normalize first to avoid doubling, then apply CRLF.
        out = out.replace('\r\n', '\n').replace('\n', '\r\n')
    from ..utils.file import atomic_write_bytes
    atomic_write_bytes(absolute_file_path, out.encode(encoding))


# === FileEdit Tool ===

class FileEditTool(Tool):
    """A tool for editing files (exact string replacement)."""

    name = FILE_EDIT_TOOL_NAME
    input_schema = FileEditInput
    DESCRIPTION = get_edit_tool_description()   # full prompt.ts text (was short blurb; original API sends tool.prompt())

    @classmethod
    async def description(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return get_edit_tool_description()

    @classmethod
    def is_read_only(cls) -> bool:
        return False

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def needs_permissions(cls) -> bool:
        return True

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return 'Update'

    @classmethod
    async def validate_input(
        cls,
        input_data: Dict[str, Any],
        context: Any,
    ) -> ValidationResult:
        """Faithful port of FileEditTool.ts validateInput()."""
        file_path = input_data.get('file_path', '')
        old_string = input_data.get('old_string', '')
        new_string = input_data.get('new_string', '')
        replace_all = input_data.get('replace_all', False)

        full_file_path = _expand_path(file_path)

        # old_string === new_string -> no changes (errorCode 1)
        if old_string == new_string:
            return ValidationResultFailure(
                result=False,
                message='No changes to make: old_string and new_string are exactly the same.',
            )

        # SECURITY: Skip filesystem operations for UNC paths.
        if full_file_path.startswith('\\\\') or full_file_path.startswith('//'):
            return ValidationResultSuccess()

        # Prevent OOM on multi-GB files (errorCode 10)
        try:
            size = os.path.getsize(full_file_path)
            if size > MAX_EDIT_FILE_SIZE:
                return ValidationResultFailure(
                    result=False,
                    message=(
                        f'File is too large to edit ({_format_file_size(size)}). '
                        f'Maximum editable file size is {_format_file_size(MAX_EDIT_FILE_SIZE)}.'
                    ),
                )
        except OSError:
            pass

        # Read the file (bytes, encoding-detected, LF-normalized).
        file_content, file_exists, _enc, _le = _read_file_for_edit(full_file_path)

        # File doesn't exist
        if not file_exists:
            # Empty old_string on nonexistent file means new file creation — valid
            if old_string == '':
                return ValidationResultSuccess()
            from ..utils.file import find_similar_file
            _sim = find_similar_file(full_file_path)
            message = (
                f'File does not exist. Current working directory: {get_cwd()}.'
                + (f' Did you mean {_sim}?' if _sim else '')
            )
            return ValidationResultFailure(
                result=False,
                message=message,
                meta={'isFilePathAbsolute': str(os.path.isabs(file_path))},
            )

        # File exists with empty old_string — only valid if file is empty
        if old_string == '':
            if file_content.strip() != '':
                return ValidationResultFailure(
                    result=False,
                    message='Cannot create new file - file already exists.',
                )
            return ValidationResultSuccess()

        # Jupyter notebooks must use NotebookEdit
        if full_file_path.endswith('.ipynb'):
            return ValidationResultFailure(
                result=False,
                message=(
                    f'File is a Jupyter Notebook. Use the {NOTEBOOK_EDIT_TOOL_NAME} '
                    'to edit this file.'
                ),
            )

        # read-before-edit (errorCode 6) + partial-view rejection. presence in
        # readFileState means the file was read; a PARTIAL view (offset/limit
        # set) does not count — the editor never saw the whole file
        # (FileEditTool.ts:275 `!readTimestamp || readTimestamp.isPartialView`).
        states = getattr(context, 'read_file_state', None)
        read_state = states.get(full_file_path) if states is not None else None
        if not read_state or read_state.is_partial_view:
            return ValidationResultFailure(
                result=False,
                message='File has not been read yet. Read it first before writing to it.',
                meta={'isFilePathAbsolute': str(os.path.isabs(file_path))},
            )

        # Modified-since-read check (errorCode 7) with the full-read content
        # equality fallback: mtime can change without the content changing
        # (cloud sync / AV / linter touch), so for a full read we treat matching
        # content as unchanged (FileEditTool.ts:296-309).
        try:
            last_write_time = os.path.getmtime(full_file_path)
            if last_write_time > read_state.timestamp:
                content_unchanged = read_state.is_full_read and file_content == read_state.content
                if not content_unchanged:
                    return ValidationResultFailure(
                        result=False,
                        message=(
                            'File has been modified since read, either by the user or by '
                            'a linter. Read it again before attempting to write it.'
                        ),
                    )
        except OSError:
            pass

        # normalizeFileEditInput: de-sanitize API-stripped tokens in
        # (old_string, new_string) and strip trailing whitespace on new_string
        # (except .md/.mdx) BEFORE findActualString. Mirrors the input
        # normalization the original applies in utils/api.ts before
        # validateInput/call run.
        old_string, new_string, replace_all = normalize_file_edit_input(
            file_path=file_path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            file_content=file_content,
        )

        # findActualString to handle quote normalization (errorCode 8)
        actual_old_string = find_actual_string(file_content, old_string)
        if not actual_old_string:
            return ValidationResultFailure(
                result=False,
                message=f'String to replace not found in file.\nString: {old_string}',
                meta={'isFilePathAbsolute': str(os.path.isabs(file_path))},
            )

        matches = file_content.count(actual_old_string)

        # Multiple matches but replace_all is false (errorCode 9)
        if matches > 1 and not replace_all:
            return ValidationResultFailure(
                result=False,
                message=(
                    f'Found {matches} matches of the string to replace, but '
                    'replace_all is false. To replace all occurrences, set replace_all '
                    'to true. To replace only one occurrence, please provide more '
                    'context to uniquely identify the instance.\n'
                    f'String: {old_string}'
                ),
                meta={
                    'isFilePathAbsolute': str(os.path.isabs(file_path)),
                    'actualOldString': actual_old_string,
                },
            )

        return ValidationResultSuccess()

    @classmethod
    def get_tool_use_summary(cls, input_data):
        """The display path (UI.tsx getToolUseSummary)."""
        from ..utils.file import get_display_path
        if not isinstance(input_data, dict):
            return None
        value = input_data.get('file_path')
        if not value:
            return None
        return get_display_path(str(value))

    @classmethod
    def get_activity_description(cls, input_data):
        """"Editing <path>" (getActivityDescription)."""
        summary = cls.get_tool_use_summary(input_data)
        return f'Editing {summary}' if summary else 'Editing file'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """The edit IS the diff — show it, not the path (FileEditTool/UI.tsx)."""
        file_path = input_data.get('file_path', '') if isinstance(input_data, dict) else ''
        old_string = input_data.get('old_string', '') if isinstance(input_data, dict) else ''
        new_string = input_data.get('new_string', '') if isinstance(input_data, dict) else ''
        if not file_path or old_string is None or new_string is None:
            return ''
        # format_edit_diff_preview already indents its own lines.
        return format_edit_diff_preview(file_path, old_string, new_string).lstrip('\n')

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """Post-edit summary; the diff was already shown at tool-use time."""
        if not isinstance(data, dict) or not data.get('filePath'):
            return None
        diff = difflib.unified_diff(
            (data.get('oldString') or '').split('\n'),
            (data.get('newString') or '').split('\n'),
            n=0, lineterm='',
        )
        adds = rems = 0
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                adds += 1
            elif line.startswith('-') and not line.startswith('---'):
                rems += 1
        parts = []
        if adds:
            parts.append(f"{adds} addition{'s' if adds != 1 else ''}")
        if rems:
            parts.append(f"{rems} removal{'s' if rems != 1 else ''}")
        summary = ' and '.join(parts) if parts else 'no changes'
        name = os.path.basename(data['filePath'])
        return f"Updated {colorize(name, Colors.BOLD)} with {summary}"

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        """Faithful port of FileEditTool.ts call()."""
        file_path = input_data.get('file_path', '')
        old_string = input_data.get('old_string', '')
        new_string = input_data.get('new_string', '')
        replace_all = input_data.get('replace_all', False)

        absolute_file_path = _expand_path(file_path)

        try:
            # Ensure parent directory exists (matches fs.mkdir(dirname)).
            parent = os.path.dirname(absolute_file_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)

            # 2. Load current state and confirm no changes since last read.
            (
                original_file_contents,
                file_exists,
                encoding,
                line_endings,
            ) = _read_file_for_edit(absolute_file_path)

            if file_exists:
                states = getattr(context, 'read_file_state', None)
                last_read = (
                    states.get(absolute_file_path)
                    if states is not None
                    else None
                )
                try:
                    last_write_time = os.path.getmtime(absolute_file_path)
                except OSError:
                    last_write_time = 0
                if not last_read or last_write_time > last_read.timestamp:
                    # Content-equality fallback for full reads (FileEditTool.ts:451-468).
                    content_unchanged = (
                        last_read is not None
                        and last_read.is_full_read
                        and original_file_contents == last_read.content
                    )
                    if not content_unchanged:
                        raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)

            # normalizeFileEditInput: de-sanitize API-stripped tokens in
            # (old_string, new_string) and strip trailing whitespace on
            # new_string (except .md/.mdx) BEFORE findActualString. When the
            # file doesn't exist, file_content is None so only new_string is
            # normalized (matches the original's ENOENT-returns-original path).
            old_string, new_string, replace_all = normalize_file_edit_input(
                file_path=file_path,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
                file_content=original_file_contents if file_exists else None,
            )

            # 3. findActualString to handle quote normalization.
            actual_old_string = (
                find_actual_string(original_file_contents, old_string) or old_string
            )

            # Preserve curly quotes in new_string when the file uses them.
            actual_new_string = preserve_quote_style(
                old_string, actual_old_string, new_string
            )

            # 4. Generate updated file.
            updated_file = get_updated_file_for_edit(
                file_contents=original_file_contents,
                old_string=actual_old_string,
                new_string=actual_new_string,
                replace_all=replace_all,
            )

            # 5. Write to disk (preserving encoding + line endings).
            _write_text_content(
                absolute_file_path, updated_file, encoding, line_endings
            )

            # 6. Refresh readFileState so a subsequent edit sees the new content
            # as "already read" (full read: store the updated text for the
            # equality fallback). Mirrors the original re-recording post-write.
            states = getattr(context, 'read_file_state', None)
            if states is not None:
                try:
                    states[absolute_file_path] = ReadFileState(
                        timestamp=os.path.getmtime(absolute_file_path),
                        content=updated_file,
                    )
                except OSError:
                    pass

            # mapToolResultToToolResultBlockParam
            if replace_all:
                result_text = (
                    f'The file {file_path} has been updated. '
                    'All occurrences were successfully replaced.'
                )
            else:
                result_text = f'The file {file_path} has been updated successfully.'

            yield ToolResultSuccess(
                type='result',
                result_for_assistant=result_text,
                data={
                    'filePath': file_path,
                    'oldString': actual_old_string,
                    'newString': new_string,
                    'originalFile': original_file_contents,
                    'userModified': False,
                    'replaceAll': replace_all,
                },
            )

        except Exception as e:
            # Faithful: edit failures (FILE_UNEXPECTEDLY_MODIFIED_ERROR, "String
            # not found", I/O errors) PROPAGATE as a tool error — the original
            # throws. The query loop turns a raised exception into an is_error
            # tool_result; yielding a success result here would report a failure
            # to the model as success.
            from ..utils.log import log_error
            log_error(e)
            raise
