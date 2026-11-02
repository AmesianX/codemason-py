"""
FileRead Tool for CodeMason
Based on src/tools/FileReadTool/FileReadTool.ts (name = 'Read') from Claude Code.

Ported faithfully:
- Input schema: file_path (absolute), offset, limit, pages (PDF page range) with
  the source's Field descriptions.
- DESCRIPTION: the resolved renderPromptTemplate text, already ported verbatim as
  READ_DESCRIPTION in constants/tool_prompts.py — imported and reused here (not
  duplicated). prompt() returns it.
- Text read: cat -n line format (line number + tab — the compact-prefix path of
  addLineNumbers()), MAX_LINES default 2000, offset/limit slicing, and the
  empty / short-offset <system-reminder> warning behavior from
  mapToolResultToToolResultBlockParam.
- Jupyter notebook (.ipynb): real notebook reading — parses JSON and emits each
  cell's source plus its outputs, faithful to utils/notebook.ts
  (processCell / cellContentToToolResult / cellOutputToToolResult).

SUBSTRATE BOUNDARY (local model may not be multimodal):
- Images: when the server takes images (services/vision.py) the result carries
  real image blocks, prepared by utils/image_prep.py and backed up by
  utils/image_store.py. Otherwise, a TEXT metadata line ("[Image file: <path>
  ...]") so the tool stays functional without a multimodal channel.
- PDFs: always the TEXT metadata line ("[PDF file: <path>, N pages]").

KEPT from the prior simplified port: population of
`context.read_file_state[file_path] = ReadFileState(...)` after resolving the path —
other tools' read-before-edit enforcement (Edit/Write/NotebookEdit) depends on it.
"""

import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

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
from ..constants.tool_prompts import READ_DESCRIPTION
from ..utils.state import get_cwd
from ..types.query import ReadFileState
from ..utils.formatting import Colors, colorize


FILE_READ_TOOL_NAME = 'Read'

# limits.ts / prompt.ts
MAX_LINES_TO_READ = 2000
# 0.25 MB — utils/file.ts MAX_OUTPUT_SIZE (the maxSizeBytes cap on a full read).
MAX_OUTPUT_SIZE = int(0.25 * 1024 * 1024)

# limits.ts DEFAULT_MAX_OUTPUT_TOKENS + the env override (precedence: env > default).
DEFAULT_MAX_OUTPUT_TOKENS = 25000

# prompt.ts FILE_UNCHANGED_STUB (verbatim) — returned by the same-range dedup.
FILE_UNCHANGED_STUB = (
    'File unchanged since last read. The content from the earlier Read '
    'tool_result in this conversation is still current — refer to that instead '
    'of re-reading.'
)

# services/tokenEstimation.ts bytesPerTokenForFileType: JSON family packs ~2
# bytes/token, everything else ~4.
_JSON_EXTS = {'json', 'jsonl', 'jsonc'}
# utils/pdfUtils.ts — PDF_MAX_PAGES_PER_READ
PDF_MAX_PAGES_PER_READ = 20

# utils/notebook.ts LARGE_OUTPUT_THRESHOLD
LARGE_OUTPUT_THRESHOLD = 10000

IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
# [ours] An image file larger than this is not read into memory to be attached.
# Phone photos are a few MB and the phone channel stops at 20 MB.
MAX_IMAGE_FILE_BYTES = 50 * 1024 * 1024

# constants/files.ts hasBinaryExtension — binary extensions this tool refuses
# (images / PDF excluded; they render via the substrate metadata path below).
BINARY_EXTENSIONS = {
    'mp3', 'mp4', 'avi', 'mkv', 'mov', 'wav', 'flac', 'ogg', 'm4a', 'wmv', 'flv',
    'zip', 'tar', 'gz', 'bz2', 'xz', '7z', 'rar',
    'exe', 'dll', 'so', 'dylib', 'o', 'a', 'pyc', 'pyo', 'class', 'jar', 'wasm',
    'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
    'ttf', 'otf', 'woff', 'woff2', 'eot',
    'bin', 'dat', 'db', 'sqlite', 'sqlite3',
}


# === Input Schema ===

class FileReadInput(ToolInputSchema):
    """Input schema for FileRead tool (mirrors FileReadTool.ts inputSchema)."""

    file_path: str = Field(
        ...,
        description="The absolute path to the file to read",
    )
    offset: Optional[int] = Field(
        default=None,
        description="The line number to start reading from. Only provide if the "
        "file is too large to read at once",
    )
    limit: Optional[int] = Field(
        default=None,
        description="The number of lines to read. Only provide if the file is too "
        "large to read at once.",
    )
    pages: Optional[str] = Field(
        default=None,
        description=(
            'Page range for PDF files (e.g., "1-5", "3", "10-20"). Only applicable '
            f'to PDF files. Maximum {PDF_MAX_PAGES_PER_READ} pages per request.'
        ),
    )


# === Helpers ===

def _resolve(file_path: str) -> str:
    """expandPath equivalent: make absolute relative to cwd."""
    return file_path if os.path.isabs(file_path) else os.path.join(get_cwd(), file_path)


# FileReadTool.ts isBlockedDevicePath — path-only (no I/O). Reading these hangs
# the process or spins forever (infinite/blocking devices); /dev/null etc. are
# intentionally NOT blocked. Missing in the port -> `open().read()` on /dev/zero
# would OOM and /dev/stdin would block.
_BLOCKED_DEVICE_PATHS = {
    '/dev/zero', '/dev/random', '/dev/urandom', '/dev/full',
    '/dev/stdin', '/dev/tty', '/dev/console',
    '/dev/stdout', '/dev/stderr',
    '/dev/fd/0', '/dev/fd/1', '/dev/fd/2',
}


def _is_blocked_device_path(path: str) -> bool:
    if path in _BLOCKED_DEVICE_PATHS:
        return True
    # /proc/self/fd/{0,1,2} and /proc/<pid>/fd/{0,1,2} are stdio aliases.
    return path.startswith('/proc/') and path.endswith(('/fd/0', '/fd/1', '/fd/2'))


def _parse_pdf_page_range(pages: str) -> Optional[Dict[str, float]]:
    """parsePDFPageRange — supports "N", "N-M", "N-" (open-ended). 1-indexed."""
    pages = pages.strip()
    if not pages:
        return None
    if '-' in pages:
        first_s, _, last_s = pages.partition('-')
        first_s, last_s = first_s.strip(), last_s.strip()
        if not first_s.isdigit():
            return None
        first = int(first_s)
        if last_s == '':
            last = float('inf')
        elif last_s.isdigit():
            last = int(last_s)
        else:
            return None
        if first < 1 or (last != float('inf') and last < first):
            return None
        return {'firstPage': first, 'lastPage': last}
    if not pages.isdigit():
        return None
    n = int(pages)
    if n < 1:
        return None
    return {'firstPage': n, 'lastPage': n}


def _get_max_output_tokens() -> int:
    """limits.ts getDefaultFileReadingLimits().maxTokens (env > default)."""
    override = os.environ.get('CODEMASON_FILE_READ_MAX_OUTPUT_TOKENS')
    if override:
        try:
            v = int(override)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_MAX_OUTPUT_TOKENS


def _fmt_size(num_bytes: int) -> str:
    """utils/format.formatFileSize (KB/MB/GB)."""
    size = float(num_bytes)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024.0 or unit == 'TB':
            return f'{int(size)}{unit}' if unit == 'B' else f'{size:.1f}{unit}'
        size /= 1024.0
    return f'{num_bytes}B'


def _validate_content_tokens(content: str, ext: str) -> Optional[str]:
    """validateContentTokens (FileReadTool.ts:755). Fast-path on a rough
    bytes/token estimate; only past maxTokens/4 do we pay for a real token
    count. Returns the MaxFileReadTokenExceededError message when the content
    exceeds the cap, else None. (The original throws; codemason surfaces it as
    an error tool_result — same effect for the model.)"""
    max_tokens = _get_max_output_tokens()
    bytes_per_token = 2 if ext in _JSON_EXTS else 4
    estimate = round(len(content) / bytes_per_token)
    if not estimate or estimate <= max_tokens / 4:
        return None
    # countTokensWithAPI substrate: an exact local count — the served model's own
    # tokenizer when available (matches the server), else the CJK heuristic.
    try:
        from ..utils.tokens import count_text_tokens
        count = count_text_tokens(content)
    except Exception:
        count = estimate
    if count > max_tokens:
        return (
            f'File content ({count} tokens) exceeds maximum allowed tokens '
            f'({max_tokens}). Use offset and limit parameters to read specific '
            'portions of the file, or search for specific content instead of '
            'reading the whole file.'
        )
    return None


def _truncation_notice(offset: int, num_lines: int, total_lines: int,
                       ended_with_newline: bool) -> Optional[str]:
    """[ours — not in upstream] The notice a short read owes the model.

    A full read stops at MAX_LINES_TO_READ and the original says nothing about
    it, so the model takes the first 2000 lines of a longer file for the whole
    file and reasons from a view it does not know is partial. The tool already
    holds the real count; this hands it over. It costs nothing in the fixed
    prefix — it rides on the tool result, only when a read actually stopped
    short — which is why the fix belongs here and not in the Read description.

    total_lines follows str.split('\n'), so a file ending in a newline carries
    one phantom empty element; it is dropped before comparing, or every such
    file would report a missing line.
    """
    file_lines = total_lines - 1 if ended_with_newline else total_lines
    last_line = offset - 1 + num_lines
    if last_line >= file_lines:
        return None
    return (
        f'\n\n<system-reminder>Showed lines {offset}-{last_line} of {file_lines}. '
        f'{file_lines - last_line} more lines were not returned — call Read again '
        f'with offset={last_line + 1} to continue.</system-reminder>'
    )


def _add_line_numbers(content: str, start_line: int) -> str:
    """addLineNumbers (compact-prefix path): "<lineno>\\t<line>" — cat -n format."""
    if not content:
        return ''
    out: List[str] = []
    for i, line in enumerate(content.split('\n')):
        out.append(f"{i + start_line}\t{line}")
    return '\n'.join(out)


def _detect_text_encoding(path: str) -> str:
    """The SHARED detector, so Read and Edit agree about a file's encoding.

    This used to do its own BOM sniff, documented as "matches
    _read_file_for_edit's detection" — and that stopped being true the moment
    Edit moved to the shared detector. The result was that Edit could read a
    CP949 file correctly while Read showed the same file as replacement
    characters, so the model could edit a file it could not read. Measured on
    Windows against a real CP949 file before this was fixed.
    """
    from ..utils.file import detect_file_encoding
    try:
        return detect_file_encoding(path)
    except OSError:
        return 'utf-8'


def _read_line_window(path: str, enc: str, line_offset: int, end_line: int):
    """Stream `path`, returning (selected_lines, total_lines, ended_with_newline)
    with ONLY the
    [line_offset, end_line) window held in memory — the rest is counted, not
    retained. This is the ranged-read path (limit set): the byte-size cap is
    intentionally skipped there, so without streaming a `Read(hugefile, limit=N)`
    would f.read() the whole (multi-GB) file into RAM → OOM. Line semantics
    reproduce str.split('\\n') EXACTLY, including the trailing empty element a
    final newline produces, so total_lines matches the full-read path."""
    selected: List[str] = []
    total = 0
    ended_with_newline = False
    first = True
    with open(path, 'r', encoding=enc, errors='replace') as f:
        for line in f:  # universal newlines: each line ends with '\n' except maybe the last
            ended_with_newline = line.endswith('\n')
            if first and line[:1] == '﻿':
                line = line[1:]
            first = False
            if line_offset <= total < end_line:
                s = line[:-1] if line.endswith('\n') else line
                if s.endswith('\r'):
                    s = s[:-1]
                selected.append(s)
            total += 1
    # Reproduce str.split('\n'): '' -> [''] (1), and a trailing '\n' yields one
    # extra empty element beyond the counted lines.
    if total == 0:
        total = 1
        if line_offset <= 0 < end_line:
            selected.append('')
    elif ended_with_newline:
        if line_offset <= total < end_line:
            selected.append('')
        total += 1
    return selected, total, ended_with_newline


# --- Notebook (faithful port of utils/notebook.ts) ---

def _process_output_text(text: Any) -> str:
    if not text:
        return ''
    return ''.join(text) if isinstance(text, list) else str(text)


def _extract_image(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if isinstance(data.get('image/png'), str):
        return {'media_type': 'image/png'}
    if isinstance(data.get('image/jpeg'), str):
        return {'media_type': 'image/jpeg'}
    return None


def _process_output(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ot = output.get('output_type')
    if ot == 'stream':
        return {'output_type': ot, 'text': _process_output_text(output.get('text'))}
    if ot in ('execute_result', 'display_data'):
        data = output.get('data') or {}
        return {
            'output_type': ot,
            'text': _process_output_text(data.get('text/plain')),
            'image': _extract_image(data),
        }
    if ot == 'error':
        traceback = output.get('traceback') or []
        text = f"{output.get('ename')}: {output.get('evalue')}\n" + '\n'.join(traceback)
        return {'output_type': ot, 'text': _process_output_text(text)}
    return None


def _is_large_outputs(outputs: List[Optional[Dict[str, Any]]]) -> bool:
    size = 0
    for o in outputs:
        if not o:
            continue
        size += len(o.get('text') or '')
        if size > LARGE_OUTPUT_THRESHOLD:
            return True
    return False


def _process_cell(cell: Dict[str, Any], index: int, code_language: str) -> Dict[str, Any]:
    cell_type = cell.get('cell_type')
    source = cell.get('source', '')
    if isinstance(source, list):
        source = ''.join(source)
    cell_data: Dict[str, Any] = {
        'cellType': cell_type,
        'source': source,
        'cell_id': cell.get('id') or f'cell-{index}',
    }
    if cell_type == 'code':
        cell_data['language'] = code_language
        outs = cell.get('outputs') or []
        if outs:
            processed = [_process_output(o) for o in outs]
            if _is_large_outputs(processed):
                cell_data['outputs'] = [{
                    'output_type': 'stream',
                    'text': "Outputs are too large to include. Use Bash with: "
                            f"cat <notebook_path> | jq '.cells[{index}].outputs'",
                }]
            else:
                cell_data['outputs'] = [p for p in processed if p]
    return cell_data


def _render_notebook_cells(cells: List[Dict[str, Any]]) -> str:
    """Faithful text rendering of cellContentToToolResult + cellOutputToToolResult.

    The original emits a sequence of text/image blocks; here we flatten to text
    (image blocks become a metadata placeholder, per the substrate boundary)."""
    parts: List[str] = []
    for cell in cells:
        cell_id = cell['cell_id']
        metadata = ''
        if cell['cellType'] != 'code':
            metadata += f"<cell_type>{cell['cellType']}</cell_type>"
        if cell.get('language') and cell['language'] != 'python' and cell['cellType'] == 'code':
            metadata += f"<language>{cell['language']}</language>"
        block = f'<cell id="{cell_id}">{metadata}{cell["source"]}</cell id="{cell_id}">'
        for output in cell.get('outputs') or []:
            if output.get('text'):
                block += '\n' + '\n' + output['text']
            if output.get('image'):
                block += f"\n[Image output: {output['image']['media_type']}]"
        parts.append(block)
    return '\n'.join(parts)


# === FileRead Tool ===

def _agent_output_task_id(file_path: str):
    """The task id, when `file_path` is an agent's output file
    (UI.tsx getAgentOutputTaskId).

    Agent output files are `<task output dir>/<taskId>.output`. The id is
    validated rather than just sliced out, so an unrelated file that happens to
    sit in that directory is not reported as a task.
    """
    from ..utils.task.disk_output import get_task_output_dir
    import os as _os
    import re as _re
    prefix = get_task_output_dir() + _os.sep
    suffix = '.output'
    if file_path.startswith(prefix) and file_path.endswith(suffix):
        task_id = file_path[len(prefix):-len(suffix)]
        if 0 < len(task_id) <= 20 and _re.fullmatch(r'[a-zA-Z0-9_-]+', task_id):
            return task_id
    return None


class FileReadTool(Tool):
    """Read a file from the local filesystem (text, image meta, PDF meta, notebook)."""

    name = FILE_READ_TOOL_NAME
    input_schema = FileReadInput
    # Infinity = never persist (FileReadTool.maxResultSizeChars). Read self-bounds
    # via its own limits, and persisting its output to a file the model reads back
    # with Read is circular.
    max_result_size_chars = float('inf')
    DESCRIPTION = READ_DESCRIPTION

    @classmethod
    async def description(cls, **kwargs) -> str:
        return READ_DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return READ_DESCRIPTION

    @classmethod
    def is_read_only(cls) -> bool:
        return True

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def needs_permissions(cls) -> bool:
        return True

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return 'Read'

    @classmethod
    def get_tool_use_summary(cls, input_data):
        """The display path (UI.tsx getToolUseSummary)."""
        from ..utils.file import get_display_path
        if not isinstance(input_data, dict):
            return None
        value = input_data.get('file_path')
        if not value:
            return None
        # An agent's output file is named by its task id and lives under a temp
        # directory nobody wants to read: the source shows the id alone, which
        # is also the thing that ties the line back to the agent.
        task_id = _agent_output_task_id(str(value))
        if task_id:
            return task_id
        return get_display_path(str(value))

    @classmethod
    def get_activity_description(cls, input_data):
        """"Reading <path>" (getActivityDescription)."""
        summary = cls.get_tool_use_summary(input_data)
        return f'Reading {summary}' if summary else 'Reading file'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """Path, plus the range when one was asked for (FileReadTool/UI.tsx).

        The verb lives in the header ([Read]) and the spinner, so the body is the
        argument alone."""
        if not isinstance(input_data, dict):
            return ''
        file_path = input_data.get('file_path', '')
        if not file_path:
            return ''
        parts = [str(file_path)]
        offset, limit, pages = (input_data.get('offset'), input_data.get('limit'),
                                input_data.get('pages'))
        if pages:
            parts.append(f"pages {pages}")
        elif offset or limit:
            start = offset or 1
            parts.append(f"lines {start}+{limit}" if limit else f"from line {start}")
        return colorize(' · '.join(parts), Colors.DIM)

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        """One line — what was read, not the file's first few lines (FileReadTool/UI.tsx).

        Dumping the head of the file is noise: the model got the content, the user
        only needs to know the read happened and how big it was. Ctrl+O still
        expands to the full text."""
        if not isinstance(data, dict) or data.get('error'):
            return None
        kind = data.get('type')
        if kind == 'text':
            n = data.get('num_lines', 0)
            return f"Read {n} line{'' if n == 1 else 's'}"
        if kind == 'file_unchanged':
            return colorize('Unchanged since last read', Colors.DIM)
        if kind == 'notebook':
            n = data.get('num_cells', 0)
            return f"Read {n} cell{'' if n == 1 else 's'}"
        if kind == 'image':
            return f"Read image ({_fmt_size(data.get('original_size', 0))})"
        if kind == 'pdf':
            n = data.get('page_count', 0)
            return (f"Read PDF ({_fmt_size(data.get('original_size', 0))}, "
                    f"{n} page{'' if n == 1 else 's'})")
        return None

    @classmethod
    async def validate_input(
        cls,
        input_data: Dict[str, Any],
        context: Any,
    ) -> ValidationResult:
        """Faithful subset of FileReadTool.validateInput (pages parse + binary check)."""
        file_path = input_data.get('file_path', '')
        pages = input_data.get('pages')

        if not file_path:
            return ValidationResultFailure(result=False, message='file_path is required')

        # pages parameter (pure string parsing, no I/O).
        if pages is not None:
            parsed = _parse_pdf_page_range(pages)
            if not parsed:
                return ValidationResultFailure(
                    result=False,
                    message=f'Invalid pages parameter: "{pages}". Use formats like '
                            '"1-5", "3", or "10-20". Pages are 1-indexed.',
                )
            if parsed['lastPage'] == float('inf'):
                range_size = PDF_MAX_PAGES_PER_READ + 1
            else:
                range_size = parsed['lastPage'] - parsed['firstPage'] + 1
            if range_size > PDF_MAX_PAGES_PER_READ:
                return ValidationResultFailure(
                    result=False,
                    message=f'Page range "{pages}" exceeds maximum of '
                            f'{PDF_MAX_PAGES_PER_READ} pages per request. Please use a '
                            'smaller range.',
                )

        full_path = _resolve(file_path)

        # Block infinite/blocking device files (would hang or OOM the process).
        if _is_blocked_device_path(full_path):
            return ValidationResultFailure(
                result=False,
                message=f'Cannot read device file: {full_path}',
            )

        ext = os.path.splitext(full_path)[1].lower().lstrip('.')

        # Binary extension check (PDF + images excluded — rendered via metadata).
        if ext in BINARY_EXTENSIONS:
            return ValidationResultFailure(
                result=False,
                message='This tool cannot read binary files. The file appears to be a '
                        f'binary .{ext} file. Please use appropriate tools for binary '
                        'file analysis.',
            )

        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        """Read the file. Dispatches: notebook / image / pdf / text."""
        file_path = input_data.get('file_path', '')
        offset = input_data.get('offset')
        if offset is None:
            offset = 1
        limit = input_data.get('limit')
        pages = input_data.get('pages')

        full_path = _resolve(file_path)
        ext = os.path.splitext(full_path)[1].lower().lstrip('.')

        if not os.path.exists(full_path):
            from ..utils.file import find_similar_file
            _sim = find_similar_file(full_path)
            _hint = f" Did you mean {_sim}?" if _sim else ""
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=f"File does not exist. Current working directory "
                                     f"is {get_cwd()}.{_hint}",
                data={'error': 'ENOENT', 'path': full_path},
            )
            return

        if os.path.isdir(full_path):
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=f"EISDIR: illegal operation on a directory, read "
                                     f"'{full_path}'. Use an ls command via Bash to read "
                                     "a directory.",
                data={'error': 'EISDIR', 'path': full_path},
            )
            return

        # KEEP: populate the readFileState substrate so Edit/Write/NotebookEdit
        # read-before-edit + modified-since-read checks work. offset/limit are
        # recorded from the RAW input (None = full read) so the editors can
        # reject partial views; content is filled in by the text path below for
        # full reads (the equality fallback). image/pdf/notebook reads leave
        # content '' — no text fallback applies there.
        raw_offset = input_data.get('offset')
        raw_limit = input_data.get('limit')
        # The original Read stores the DEFAULTED offset (=1 for a full read; call
        # destructures `{ offset = 1 }`), so a Read entry always has a defined
        # offset >= 1 and is therefore never `is_full_read` — that flag (offset &
        # limit both None) is the marker Edit/Write set on their post-write
        # entries, which is what the content-equality fallback keys on. Keeping
        # this faithful means the dedup below can also distinguish Read entries
        # (offset set) from Edit/Write entries (offset None).
        eff_offset = raw_offset if raw_offset is not None else 1

        # Read dedup (FileReadTool.ts:540-572): if the SAME range of this file was
        # already Read and its mtime is unchanged, return FILE_UNCHANGED_STUB
        # instead of a second full copy (the earlier tool_result is still in
        # context). Only text/notebook reads dedup; images/PDFs aren't cached.
        # Exclude Edit/Write entries (offset None) and partial-view entries.
        existing = getattr(context, 'read_file_state', {}).get(full_path) \
            if hasattr(context, 'read_file_state') else None
        if (
            existing is not None
            and not existing.is_partial_view
            and existing.offset is not None
            and existing.offset == eff_offset
            and existing.limit == raw_limit
            and ext not in IMAGE_EXTENSIONS
            and ext != 'pdf'
        ):
            try:
                if os.path.getmtime(full_path) == existing.timestamp:
                    yield ToolResultSuccess(
                        type='result',
                        result_for_assistant=FILE_UNCHANGED_STUB,
                        data={'type': 'file_unchanged', 'path': full_path},
                    )
                    return
            except OSError:
                pass  # stat failed — fall through to a full read

        try:
            if hasattr(context, 'read_file_state'):
                context.read_file_state[full_path] = ReadFileState(
                    timestamp=os.path.getmtime(full_path),
                    offset=eff_offset,
                    limit=raw_limit,
                )
        except OSError:
            pass

        try:
            # --- Notebook ---
            if ext == 'ipynb':
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    notebook = json.loads(f.read())
                language = ((notebook.get('metadata') or {}).get('language_info') or {}).get('name') or 'python'
                raw_cells = notebook.get('cells') or []
                cells = [_process_cell(c, i, language) for i, c in enumerate(raw_cells)]
                result = _render_notebook_cells(cells)
                yield ToolResultSuccess(
                    type='result',
                    result_for_assistant=result,
                    data={'type': 'notebook', 'path': full_path,
                          'cells': cells, 'num_cells': len(cells)},
                )
                return

            # --- Image: real image blocks when the server can see, else metadata ---
            if ext in IMAGE_EXTENSIONS:
                size = os.path.getsize(full_path)
                from ..services.vision import enabled_limit
                limit = enabled_limit()
                err = ''
                if limit > 0 and size > MAX_IMAGE_FILE_BYTES:
                    err = (f'larger than {MAX_IMAGE_FILE_BYTES // (1024 * 1024)} MB; '
                           'shrink or crop it first')
                elif limit > 0:
                    import asyncio
                    import base64
                    from ..utils.image_prep import prepare_image
                    with open(full_path, 'rb') as f:
                        raw = f.read()
                    # At most half the server's limit from one Read: that is the
                    # batch the oldest images are dropped in, so two Reads in one
                    # step still fit, and a long page cut into tiles cannot push
                    # out images the model has not looked at yet.
                    slots = max(1, limit // 2)
                    try:
                        # Decode and Lanczos take a while on a big photo; not on
                        # the event loop, which also draws the spinner.
                        prepared = await asyncio.to_thread(prepare_image, raw, slots)
                    except Exception as e:  # not decodable as an image
                        prepared = []
                        err = str(e) or type(e).__name__
                    if prepared:
                        from ..utils.image_store import save_original
                        original_copy = save_original(raw, ext, source=Path(full_path))
                        p0 = prepared[0]
                        if len(prepared) == 1:
                            note = f"{p0.width}x{p0.height}"
                        else:
                            note = (f"cut into {len(prepared)} tiles of about "
                                    f"{p0.width}x{p0.height}, top to bottom / left to right")
                        if p0.animated:
                            note += ", first frame of an animation"
                        blocks = [{'type': 'text', 'text':
                                   f"[Image file: {full_path} ({size} bytes) — shown below, {note}]"}]
                        for p in prepared:
                            blocks.append({'type': 'image', 'source': {
                                'type': 'base64', 'media_type': p.media_type,
                                'data': base64.b64encode(p.data).decode('ascii')}})
                        yield ToolResultSuccess(
                            type='result',
                            result_for_assistant=blocks,
                            data={'type': 'image', 'path': full_path,
                                  'media_type': prepared[0].media_type, 'original_size': size,
                                  'images': len(prepared),
                                  'tokens': sum(p.tokens for p in prepared),
                                  'original_copy': str(original_copy)},
                        )
                        return
                    err = err or 'no image in the file'
                if limit > 0:
                    result = f"[Image file: {full_path} ({size} bytes) could not be attached: {err}]"
                else:
                    result = (f"[Image file: {full_path} ({size} bytes, image/{ext}). "
                              "Multimodal image rendering is not available on this "
                              "substrate; metadata only.]")
                yield ToolResultSuccess(
                    type='result',
                    result_for_assistant=result,
                    data={'type': 'image', 'path': full_path,
                          'media_type': f'image/{ext}', 'original_size': size},
                )
                return

            # --- PDF (substrate: metadata text, not a document block) ---
            if ext == 'pdf':
                size = os.path.getsize(full_path)
                page_count = cls._pdf_page_count(full_path)
                pages_note = f", {page_count} pages" if page_count is not None else ""
                range_note = f", pages {pages}" if pages else ""
                result = (f"[PDF file: {full_path} ({size} bytes{pages_note}{range_note}). "
                          "PDF document rendering is not available on this substrate; "
                          "metadata only.]")
                yield ToolResultSuccess(
                    type='result',
                    result_for_assistant=result,
                    data={'type': 'pdf', 'path': full_path,
                          'original_size': size, 'page_count': page_count},
                )
                return

            # --- Text file (cat -n) ---
            # maxSizeBytes (FileReadTool.ts:1026 + readFileInRange FileTooLargeError):
            # a NO-LIMIT read of a file whose TOTAL size exceeds the cap returns
            # an error rather than truncating — truncation was tested and
            # REVERTED upstream (#21841: the ~100-byte error beats ~25K tokens of
            # clipped content). Limited reads are exempt (limit set => no cap).
            if limit is None:
                try:
                    fsize = os.path.getsize(full_path)
                except OSError:
                    fsize = 0
                if fsize > MAX_OUTPUT_SIZE:
                    yield ToolResultSuccess(
                        type='result',
                        result_for_assistant=(
                            f'File content ({_fmt_size(fsize)}) exceeds maximum allowed '
                            f'size ({_fmt_size(MAX_OUTPUT_SIZE)}). Use offset and limit '
                            'parameters to read specific portions of the file, or search '
                            'for specific content instead of reading the whole file.'
                        ),
                        data={'error': 'FileTooLarge', 'path': full_path,
                              'size': fsize, 'max_size': MAX_OUTPUT_SIZE},
                    )
                    return
            enc = _detect_text_encoding(full_path)
            line_offset = 0 if offset == 0 else offset - 1
            end_line = line_offset + (limit if limit is not None else MAX_LINES_TO_READ)

            if limit is None:
                # Full read — already size-capped above. Materialize so the full
                # content can be recorded for the modified-since-read equality
                # fallback, then window it.
                with open(full_path, 'r', encoding=enc, errors='replace') as f:
                    raw = f.read()
                # Strip BOM (readFileInRange does this; utf-8-sig/utf-16 already do).
                if raw and raw[0] == '﻿':
                    raw = raw[1:]
                if raw_offset is None and raw_limit is None and hasattr(context, 'read_file_state'):
                    _st = context.read_file_state.get(full_path)
                    if _st is not None:
                        _st.content = raw
                # Split lines, strip \r (CRLF -> LF), select [offset, offset+limit).
                all_lines = raw.split('\n')
                total_lines = len(all_lines)
                _ends_nl = bool(all_lines) and all_lines[-1] == ''
                selected = [ln[:-1] if ln.endswith('\r') else ln
                            for ln in all_lines[line_offset:end_line]]
            else:
                # Ranged read: stream only the requested window (see
                # _read_line_window) so a huge file is never fully loaded — the
                # byte cap above does not apply to ranged reads.
                selected, total_lines, _ends_nl = _read_line_window(
                    full_path, enc, line_offset, end_line)
            content = '\n'.join(selected)

            if not content:
                # mapToolResultToToolResultBlockParam empty / short-offset warning.
                if total_lines == 0 or (total_lines == 1 and all_lines == ['']):
                    warning = ('<system-reminder>Warning: the file exists but the '
                               'contents are empty.</system-reminder>')
                else:
                    warning = ('<system-reminder>Warning: the file exists but is shorter '
                               f'than the provided offset ({offset}). The file has '
                               f'{total_lines} lines.</system-reminder>')
                yield ToolResultSuccess(
                    type='result',
                    result_for_assistant=warning,
                    data={'type': 'text', 'path': full_path,
                          'total_lines': total_lines, 'num_lines': 0,
                          'start_line': offset},
                )
                return

            # maxTokens (FileReadTool.ts:1030 validateContentTokens): a read whose
            # token count exceeds the 25k cap returns an error telling the model
            # to use offset/limit. Validated on the selected content, like the
            # original (the byte cap above already bounds full reads).
            token_error = _validate_content_tokens(content, ext)
            if token_error is not None:
                yield ToolResultSuccess(
                    type='result',
                    result_for_assistant=token_error,
                    data={'error': 'MaxFileReadTokenExceeded', 'path': full_path},
                )
                return

            result = _add_line_numbers(content, start_line=offset)

            note = _truncation_notice(offset, len(selected), total_lines, _ends_nl)
            if note:
                result += note

            yield ToolResultSuccess(
                type='result',
                result_for_assistant=result,
                data={'type': 'text', 'path': full_path,
                      'total_lines': total_lines, 'num_lines': len(selected),
                      'start_line': offset},
            )

        except Exception as e:
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=f"Error reading file: {str(e)}",
                data={'error': str(e)},
            )

    @staticmethod
    def _pdf_page_count(path: str) -> Optional[int]:
        """Best-effort PDF page count without external deps (count /Type /Page).

        Scans at most the first 64 MiB rather than loading the whole PDF into RAM
        just for a page count (a large PDF would otherwise be fully materialized).
        Best-effort by design, so under-counting a PDF whose page objects extend
        past the scan window is acceptable."""
        try:
            import re as _re
            with open(path, 'rb') as f:
                data = f.read(64 * 1024 * 1024)
            n = len(_re.findall(rb'/Type\s*/Page[^s]', data))
            return n or None
        except Exception:
            return None
