"""
NotebookEdit Tool for CodeMason
Based on src/tools/NotebookEditTool/ from Claude Code

Schema, description, prompt, validation (abs path, .ipynb, edit modes, read-
before-edit, cell-id resolution) and the cell replace/insert/delete logic are
ported verbatim. File I/O uses codemason's local substrate; read-before-edit
uses ToolUseContext.read_file_state (populated by Read).
"""

import json
import os
import re
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional

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
from ..utils.formatting import Colors, colorize
from ..utils.state import get_cwd
from ..types.query import ReadFileState


NOTEBOOK_EDIT_TOOL_NAME = 'NotebookEdit'

# Lines of the new cell source echoed at tool-use time (verbose: all).
_PREVIEW_LINES = 10

DESCRIPTION = 'Replace the contents of a specific cell in a Jupyter notebook.'

PROMPT = (
    "Completely replaces the contents of a specific cell in a Jupyter notebook "
    "(.ipynb file) with new source. Jupyter notebooks are interactive documents "
    "that combine code, text, and visualizations, commonly used for data analysis "
    "and scientific computing. The notebook_path parameter must be an absolute "
    "path, not a relative path. The cell_number is 0-indexed. Use edit_mode=insert "
    "to add a new cell at the index specified by cell_number. Use edit_mode=delete "
    "to delete the cell at the index specified by cell_number."
)


class NotebookEditInput(ToolInputSchema):
    notebook_path: str = Field(
        ...,
        description="The absolute path to the Jupyter notebook file to edit (must be "
        "absolute, not relative)",
    )
    cell_id: Optional[str] = Field(
        default=None,
        description="The ID of the cell to edit. When inserting a new cell, the new "
        "cell will be inserted after the cell with this ID, or at the beginning if "
        "not specified.",
    )
    new_source: str = Field(..., description="The new source for the cell")
    cell_type: Optional[Literal['code', 'markdown']] = Field(
        default=None,
        description="The type of the cell (code or markdown). If not specified, it "
        "defaults to the current cell type. If using edit_mode=insert, this is "
        "required.",
    )
    edit_mode: Optional[Literal['replace', 'insert', 'delete']] = Field(
        default='replace',
        description="The type of edit to make (replace, insert, delete). Defaults to "
        "replace.",
    )


def _resolve(notebook_path: str) -> str:
    return notebook_path if os.path.isabs(notebook_path) else os.path.join(get_cwd(), notebook_path)


def _detect_line_endings(content: str) -> str:
    """Detect 'CRLF' vs 'LF' by majority (utils/fileRead.detectLineEndingsForString)."""
    crlf_count = 0
    lf_count = 0
    for i, ch in enumerate(content):
        if ch == '\n':
            if i > 0 and content[i - 1] == '\r':
                crlf_count += 1
            else:
                lf_count += 1
    return 'CRLF' if crlf_count > lf_count else 'LF'


def _parse_cell_id(cell_id: str) -> Optional[int]:
    """Parse a 'cell-N' style id to its numeric index (utils/notebook.parseCellId)."""
    m = re.fullmatch(r'cell-(\d+)', cell_id)
    return int(m.group(1)) if m else None


class NotebookEditTool(Tool):
    """Edit Jupyter notebook cells (.ipynb)."""

    name = NOTEBOOK_EDIT_TOOL_NAME
    input_schema = NotebookEditInput
    DESCRIPTION = PROMPT   # full notebook prompt (was 1-line blurb; original API sends tool.prompt())

    @classmethod
    async def description(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return PROMPT

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
        return 'Edit Notebook'

    @classmethod
    def get_tool_use_summary(cls, input_data):
        """The display path (UI.tsx getToolUseSummary)."""
        from ..utils.file import get_display_path
        if not isinstance(input_data, dict):
            return None
        value = input_data.get('notebook_path')
        if not value:
            return None
        return get_display_path(str(value))

    @classmethod
    def get_activity_description(cls, input_data):
        """"Editing notebook <path>" (getActivityDescription)."""
        summary = cls.get_tool_use_summary(input_data)
        return f'Editing notebook {summary}' if summary else 'Editing notebook'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """Cell location + the new source (NotebookEditTool/UI.tsx)."""
        if not isinstance(input_data, dict):
            return ''
        path = input_data.get('notebook_path', '')
        if not path:
            return ''
        location = str(path)
        cell_id = input_data.get('cell_id')
        if cell_id:
            location += f" · cell {cell_id}"
        edit_mode = input_data.get('edit_mode') or 'replace'
        if edit_mode != 'replace':
            location += f" · {edit_mode}"
        cell_type = input_data.get('cell_type')
        if cell_type:
            location += f" · {cell_type}"

        lines = [colorize(location, Colors.DIM)]
        source = str(input_data.get('new_source', '') or '').split('\n')
        shown = source if verbose else source[:_PREVIEW_LINES]
        lines += [line[:100] for line in shown]
        if len(source) > len(shown):
            lines.append(colorize(f'... ({len(source) - len(shown)} more lines)', Colors.DIM))
        return '\n'.join(lines)

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        notebook_path = input_data.get('notebook_path', '')
        cell_type = input_data.get('cell_type')
        cell_id = input_data.get('cell_id')
        edit_mode = input_data.get('edit_mode') or 'replace'
        full_path = _resolve(notebook_path)

        if full_path.startswith('\\\\') or full_path.startswith('//'):
            return ValidationResultSuccess()

        if os.path.splitext(full_path)[1] != '.ipynb':
            return ValidationResultFailure(
                result=False,
                message='File must be a Jupyter notebook (.ipynb file). For editing '
                'other file types, use the FileEdit tool.',
            )

        if edit_mode not in ('replace', 'insert', 'delete'):
            return ValidationResultFailure(result=False, message='Edit mode must be replace, insert, or delete.')

        if edit_mode == 'insert' and not cell_type:
            return ValidationResultFailure(result=False, message='Cell type is required when using edit_mode=insert.')

        # Read-before-edit + partial-view rejection (matches FileEditTool/
        # FileWriteTool). Notebooks are JSON, so no text content-equality
        # fallback applies — the mtime check is authoritative.
        states = getattr(context, 'read_file_state', None)
        if states is not None:
            read_state = states.get(full_path)
            if not read_state or read_state.is_partial_view:
                return ValidationResultFailure(
                    result=False,
                    message='File has not been read yet. Read it first before writing to it.',
                )
            try:
                if os.path.getmtime(full_path) > read_state.timestamp:
                    return ValidationResultFailure(
                        result=False,
                        message='File has been modified since read, either by the user '
                        'or by a linter. Read it again before attempting to write it.',
                    )
            except OSError:
                pass

        try:
            # STRICT, not errors='replace'. The edit path re-serializes whatever
            # it decoded and writes it back, so a replaced byte is not a display
            # glitch — it is permanent loss of the original file's content, the
            # same way the Edit tool used to destroy CP949 sources. A notebook is
            # JSON and JSON is UTF-8, so anything else is a malformed file we
            # must refuse rather than silently rewrite.
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return ValidationResultFailure(result=False, message='Notebook file does not exist.')
        except UnicodeDecodeError:
            return ValidationResultFailure(
                result=False,
                message='Notebook is not valid UTF-8 (JSON must be UTF-8); editing it would replace the undecodable bytes and corrupt the file.',
            )

        try:
            notebook = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return ValidationResultFailure(result=False, message='Notebook is not valid JSON.')

        cells = notebook.get('cells', [])
        if not cell_id:
            if edit_mode != 'insert':
                return ValidationResultFailure(result=False, message='Cell ID must be specified when not inserting a new cell.')
        else:
            idx = next((i for i, c in enumerate(cells) if c.get('id') == cell_id), -1)
            if idx == -1:
                parsed = _parse_cell_id(cell_id)
                if parsed is not None:
                    if parsed < 0 or parsed >= len(cells):
                        return ValidationResultFailure(result=False, message=f'Cell with index {parsed} does not exist in notebook.')
                else:
                    return ValidationResultFailure(result=False, message=f'Cell with ID "{cell_id}" not found in notebook.')

        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        notebook_path = input_data.get('notebook_path', '')
        new_source = input_data.get('new_source', '')
        cell_id = input_data.get('cell_id')
        cell_type = input_data.get('cell_type')
        edit_mode = input_data.get('edit_mode') or 'replace'
        full_path = _resolve(notebook_path)

        try:
            # Read raw to detect the original line-ending style (CRLF vs LF)
            # before parsing, so an edit can write it back unchanged instead
            # of silently normalizing CRLF notebooks to LF. Mirrors the
            # original readFileSyncWithMetadata + writeTextContent(encoding,
            # lineEndings). A notebook is JSON, so it IS UTF-8 — anything else
            # was refused in validate_input (ensure_ascii=False below).
            # Strict as well — validate_input already refused a non-UTF-8
            # notebook, and if that ever stops being true this must fail loudly
            # rather than write replacement characters over the user's file.
            with open(full_path, 'r', encoding='utf-8', newline='') as f:
                raw = f.read()
            line_endings = _detect_line_endings(raw[:4096])
            content = raw.replace('\r\n', '\n')
            notebook = json.loads(content)
            cells = notebook.setdefault('cells', [])

            # Resolve target index
            if not cell_id:
                cell_index = 0
            else:
                cell_index = next((i for i, c in enumerate(cells) if c.get('id') == cell_id), -1)
                if cell_index == -1:
                    parsed = _parse_cell_id(cell_id)
                    if parsed is not None:
                        cell_index = parsed
                if edit_mode == 'insert':
                    cell_index += 1

            # Replace one-past-the-end becomes insert
            if edit_mode == 'replace' and cell_index == len(cells):
                edit_mode = 'insert'
                if not cell_type:
                    cell_type = 'code'

            nbformat = notebook.get('nbformat', 4)
            nbformat_minor = notebook.get('nbformat_minor', 0)
            new_cell_id = None
            if nbformat > 4 or (nbformat == 4 and nbformat_minor >= 5):
                if edit_mode == 'insert':
                    new_cell_id = uuid.uuid4().hex[:13]
                elif cell_id is not None:
                    new_cell_id = cell_id

            if edit_mode == 'delete':
                if 0 <= cell_index < len(cells):
                    cells.pop(cell_index)
            elif edit_mode == 'insert':
                if cell_type == 'markdown':
                    new_cell = {'cell_type': 'markdown', 'source': new_source, 'metadata': {}}
                else:
                    new_cell = {'cell_type': 'code', 'source': new_source,
                                'metadata': {}, 'execution_count': None, 'outputs': []}
                # nbformat < 4.5 cells carry NO `id` field; only add it when the
                # format supports ids (writing "id": null produced an off-spec cell).
                if new_cell_id is not None:
                    new_cell['id'] = new_cell_id
                cells.insert(cell_index, new_cell)
            else:
                # Guard against an unresolved cell_id (cell_index == -1): without
                # this, cells[-1] would silently edit the LAST cell. validate_input
                # rejects such ids today, but call() must not depend on that.
                if not (0 <= cell_index < len(cells)):
                    raise ValueError(f"cell_id {cell_id!r} did not resolve to a valid cell to replace")
                target = cells[cell_index]
                target['source'] = new_source
                if target.get('cell_type') == 'code':
                    target['execution_count'] = None
                    target['outputs'] = []
                if cell_type and cell_type != target.get('cell_type'):
                    target['cell_type'] = cell_type

            updated = json.dumps(notebook, indent=1, ensure_ascii=False)
            from ..utils.file import atomic_write_bytes
            if line_endings == 'CRLF':
                # Normalize any CRLF first so the join doesn't produce \r\r\n,
                # then restore CRLF (matches writeTextContent).
                updated = updated.replace('\r\n', '\n').replace('\n', '\r\n')
            atomic_write_bytes(full_path, updated.encode('utf-8'))

            # Refresh readFileState post-write (matches the original).
            states = getattr(context, 'read_file_state', None)
            if states is not None:
                try:
                    states[full_path] = ReadFileState(timestamp=os.path.getmtime(full_path))
                except OSError:
                    pass

            disp_id = new_cell_id if new_cell_id else cell_id
            if edit_mode == 'delete':
                result_text = f"Deleted cell {disp_id}"
            elif edit_mode == 'insert':
                result_text = f"Inserted cell {disp_id} with {new_source}"
            else:
                result_text = f"Updated cell {disp_id} with {new_source}"

            yield ToolResultSuccess(type='result', result_for_assistant=result_text,
                                    data={'notebook_path': full_path, 'edit_mode': edit_mode, 'cell_id': disp_id})

        except Exception as e:
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=f"Error editing notebook: {str(e)}",
                data={'error': str(e)},
            )
