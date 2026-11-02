"""
FileWrite Tool for CodeMason
Based on src/tools/FileWriteTool/ from Claude Code

Tool name is 'Write' (FILE_WRITE_TOOL_NAME). Schema, description, and prompt are
ported verbatim from FileWriteTool.ts / prompt.ts. Behavior reproduced:
overwrite/create file, read-before-edit requirement for EXISTING files, parent
directory creation, and the modified-since-read staleness check. File I/O uses
codemason's local substrate; read-before-edit uses
ToolUseContext.read_file_state (populated by Read), and the read state
is updated after writing.
"""

import math
import os
from typing import Any, AsyncGenerator, Callable, Dict, Optional

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
from ..utils.file import detect_file_encoding

# Lines of new content echoed under the path at tool-use time (verbose: all).
_PREVIEW_LINES = 10
from ..types.query import ReadFileState


FILE_WRITE_TOOL_NAME = 'Write'

# Ported verbatim from prompt.ts (DESCRIPTION) and FileWriteTool.ts description().
DESCRIPTION = 'Write a file to the local filesystem.'

# Ported verbatim from prompt.ts getWriteToolDescription(). FILE_READ_TOOL_NAME
# is 'Read'.
PROMPT = (
    "Writes a file to the local filesystem.\n"
    "\n"
    "Usage:\n"
    "- This tool will overwrite the existing file if there is one at the provided path.\n"
    "- If this is an existing file, you MUST use the Read tool first to read the "
    "file's contents. This tool will fail if you did not read the file first.\n"
    "- Prefer the Edit tool for modifying existing files — it only sends the diff. "
    "Only use this tool to create new files or for complete rewrites.\n"
    "- NEVER create documentation files (*.md) or README files unless explicitly "
    "requested by the User.\n"
    "- Only use emojis if the user explicitly requests it. Avoid writing emojis to "
    "files unless asked."
)


# === Input Schema (ported verbatim from FileWriteTool.ts inputSchema) ===

class FileWriteInput(ToolInputSchema):
    file_path: str = Field(
        ...,
        description="The absolute path to the file to write (must be absolute, not relative)",
    )
    content: str = Field(..., description="The content to write to the file")


def _resolve(file_path: str) -> str:
    return file_path if os.path.isabs(file_path) else os.path.join(get_cwd(), file_path)


class FileWriteTool(Tool):
    """Write a file to the local filesystem (create or overwrite)."""

    name = FILE_WRITE_TOOL_NAME
    input_schema = FileWriteInput
    DESCRIPTION = PROMPT   # full write prompt (was 1-line blurb; original API sends tool.prompt())

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
        return 'Write'

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        file_path = input_data.get('file_path', '')
        full_file_path = _resolve(file_path)

        # SECURITY: Skip filesystem operations for UNC paths to prevent NTLM
        # credential leaks. Let the permission check handle UNC paths.
        if full_file_path.startswith('\\\\') or full_file_path.startswith('//'):
            return ValidationResultSuccess()

        # Stat the file. ENOENT (new file) => no read-before-edit requirement.
        try:
            file_mtime_ms = os.stat(full_file_path).st_mtime * 1000
        except FileNotFoundError:
            return ValidationResultSuccess()

        # Read-before-edit for EXISTING files + partial-view rejection
        # (FileWriteTool.ts:198-206: `!readTimestamp || readTimestamp.isPartialView`).
        states = getattr(context, 'read_file_state', None)
        read_state = states.get(full_file_path) if states is not None else None
        if not read_state or read_state.is_partial_view:
            return ValidationResultFailure(
                result=False,
                message='File has not been read yet. Read it first before writing to it.',
            )

        # Modified-since-read: mtime bump. Apply the SAME content-equality
        # fallback Edit.validate_input uses (and this tool's own call() applies) —
        # otherwise validate_input rejected on mtime alone and returned before
        # call() could run its fallback, so Write spuriously blocked on a
        # linter/formatter/sync touch that left the content identical. timestamp
        # is stored in seconds, so compare in ms.
        last_write_time = math.floor(file_mtime_ms)
        if last_write_time > math.floor(read_state.timestamp * 1000):
            content_unchanged = False
            if read_state.is_full_read and read_state.content is not None:
                try:
                    with open(full_file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content_unchanged = (f.read() == read_state.content)
                except OSError:
                    content_unchanged = False
            if not content_unchanged:
                return ValidationResultFailure(
                    result=False,
                    message='File has been modified since read, either by the user or by a '
                    'linter. Read it again before attempting to write it.',
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
        """"Writing <path>" (getActivityDescription)."""
        summary = cls.get_tool_use_summary(input_data)
        return f'Writing {summary}' if summary else 'Writing file'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """Path + a preview of what is about to be written (FileWriteTool/UI.tsx)."""
        if not isinstance(input_data, dict):
            return ''
        file_path = input_data.get('file_path', '') or ''
        content = input_data.get('content', '') or ''
        if not file_path:
            return ''
        lines = [colorize(file_path, Colors.DIM)]
        content_lines = content.split('\n')
        shown = content_lines if verbose else content_lines[:_PREVIEW_LINES]
        # Mark a cut line. Only the LINE-count overflow was annotated below, so
        # a line trimmed at 100 chars looked complete: 99 benign characters
        # followed by anything at all was shown as clean, and this rendering is
        # what the user approves.
        lines += [(line[:100] + ' …') if len(line) > 100 else line for line in shown]
        if len(content_lines) > len(shown):
            hidden = len(content_lines) - len(shown)
            lines.append(colorize(f'... ({hidden} more lines)', Colors.DIM))
        return '\n'.join(lines)

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        file_path = input_data.get('file_path', '')
        content = input_data.get('content', '')
        full_file_path = _resolve(file_path)
        dir_name = os.path.dirname(full_file_path)

        try:
            # Ensure parent directory exists before the read-modify-write section.
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # Load current state. ENOENT => new file (old_content is None).
            try:
                with open(full_file_path, 'r', encoding='utf-8', errors='replace') as f:
                    old_content = f.read()
            except FileNotFoundError:
                old_content = None

            # Staleness check: confirm no changes since last read, with the
            # full-read content-equality fallback (FileWriteTool.ts:279-294).
            if old_content is not None:
                states = getattr(context, 'read_file_state', None)
                last_read = states.get(full_file_path) if states is not None else None
                try:
                    last_write_time = os.path.getmtime(full_file_path)
                except OSError:
                    last_write_time = None
                if last_read is None or (last_write_time is not None and last_write_time > last_read.timestamp):
                    content_unchanged = (
                        last_read is not None
                        and last_read.is_full_read
                        and old_content == last_read.content
                    )
                    if not content_unchanged:
                        raise RuntimeError(
                            'File has been modified since read, either by the user or by a '
                            'linter. Read it again before attempting to write it.'
                        )

            # Write is a full content replacement. The model sent explicit line
            # endings in `content` and meant them; do not rewrite them. But the
            # existing file's ENCODING is preserved on overwrite (e.g. UTF-16LE),
            # matching Edit — plain utf-8 would silently transcode the file.
            write_encoding = (detect_file_encoding(full_file_path)
                              if os.path.exists(full_file_path) else 'utf-8')
            # Atomic (temp + rename) so a crash mid-write can't leave a truncated
            # file. newline='' semantics preserved: we encode the content verbatim.
            # Re-attach the UTF-16 BOM the model never saw. The read path KEEPS
            # a file's U+FEFF in the string, which is what lets Edit round-trip
            # it — but Read strips it before showing the text, so a full
            # replacement arrives here with none and the utf-16-le/-be codecs add
            # none of their own. The file came out BOM-less and was then detected
            # as utf-8 forever after (NULs are valid UTF-8): it read back as
            # NUL-laced garbage and Grep skipped it as binary. Measured end to
            # end. utf-8-sig already prepends its own, so only the pair needs it.
            if write_encoding in ('utf-16-le', 'utf-16-be') and not content.startswith('\ufeff'):
                content = '\ufeff' + content
            from ..utils.file import atomic_write_bytes
            atomic_write_bytes(full_file_path, content.encode(write_encoding))

            # Refresh readFileState post-write (full read: store the written
            # content for the equality fallback), to invalidate stale writes.
            states = getattr(context, 'read_file_state', None)
            if states is not None:
                try:
                    states[full_file_path] = ReadFileState(
                        timestamp=os.path.getmtime(full_file_path),
                        content=content,
                    )
                except OSError:
                    pass

            if old_content is not None:
                data = {
                    'type': 'update',
                    'filePath': file_path,
                    'content': content,
                    'originalFile': old_content,
                }
                result_msg = f"The file {file_path} has been updated successfully."
            else:
                data = {
                    'type': 'create',
                    'filePath': file_path,
                    'content': content,
                    'originalFile': None,
                }
                result_msg = f"File created successfully at: {file_path}"

            yield ToolResultSuccess(
                type='result',
                result_for_assistant=result_msg,
                data=data,
            )

        except Exception as e:
            # Faithful: write failures (modified-since-read, I/O errors)
            # PROPAGATE as a tool error — the original throws. The query loop
            # turns a raised exception into an is_error tool_result; yielding a
            # success result here would report a failure to the model as success.
            from ..utils.log import log_error
            log_error(e)
            raise
