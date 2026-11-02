"""Persist large tool results to disk — adapted from utils/toolResultStorage.ts
(the per-tool persistence layer).

When a single tool result exceeds its effective threshold, the full content is
written to a session file and the model receives a `<persisted-output>` preview
+ file path instead of the full text — so a huge result (a big grep, a verbose
build log) doesn't blow the context. Also injects a marker for empty results.

Effective threshold = getPersistenceThreshold(toolName, declared): a declared
maxResultSizeChars of Infinity opts out entirely (Read — persisting output the
model reads back with Read is circular), otherwise it is clamped by the
system-wide DEFAULT_MAX_RESULT_SIZE_CHARS (50k). So a tool declaring 100k still
persists at 50k; Bash (30k) / Grep (20k) persist sooner.

Substrate notes (what is NOT ported):
- The per-MESSAGE aggregate budget (enforceToolResultBudget / ContentReplacement
  State) is OFF by default and exists for prompt-
  cache prefix stability across turns/resume; codemason (llama.cpp) has no such
  cache contract, so only the per-tool layer is ported.
- The GrowthBook per-tool override (tengu_satin_quoll) has no analogue — the
  threshold is just min(declared, default).
- Session/project dir: codemason has no getSessionId/getProjectDir, so files
  live under a fixed temp dir keyed by the unique tool_use_id (no cross-session
  collision since each process is one session).
"""

import math
import os
import tempfile
from typing import Optional
from .platform import chmod_posix

# System-wide cap regardless of what a tool declares (DEFAULT_MAX_RESULT_SIZE_CHARS).
DEFAULT_MAX_RESULT_SIZE_CHARS = 50_000

TOOL_RESULTS_SUBDIR = "tool-results"
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
# Preview size in bytes for the reference message.
PREVIEW_SIZE_BYTES = 2000


def _fixed1_drop_zero(value: float) -> str:
    """`value.toFixed(1).replace(/\\.0$/, '')` — one decimal, strip a trailing .0."""
    s = f"{value:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _format_file_size(size_in_bytes: int) -> str:
    """Mirror of utils/format.formatFileSize (KB/MB/GB, drop trailing .0)."""
    kb = size_in_bytes / 1024
    if kb < 1:
        return f"{size_in_bytes} bytes"
    if kb < 1024:
        return f"{_fixed1_drop_zero(kb)}KB"
    mb = kb / 1024
    if mb < 1024:
        return f"{_fixed1_drop_zero(mb)}MB"
    gb = mb / 1024
    return f"{_fixed1_drop_zero(gb)}GB"


# Per-session/process token so concurrent sessions don't share a namespace and
# a stale tool_use_id from a prior run can't collide with our exclusive-create
# ('x') writes (the old fixed /tmp/codemason/tool-results was shared by every
# process, contradicting this module's own "one process = one session" claim).
# Original keyed results under projectDir/sessionId.
_SESSION_TOKEN = os.environ.get('CODEMASON_SESSION_ID') or ('p%d-%s' % (os.getpid(), os.urandom(4).hex()))


def _session_root() -> str:
    return os.path.join(tempfile.gettempdir(), "codemason", _SESSION_TOKEN)


def get_tool_results_dir() -> str:
    return os.path.join(_session_root(), TOOL_RESULTS_SUBDIR)


def get_tool_result_path(id_: str, is_json: bool = False) -> str:
    ext = "json" if is_json else "txt"
    return os.path.join(get_tool_results_dir(), f"{id_}.{ext}")


def ensure_tool_results_dir() -> None:
    d = get_tool_results_dir()
    os.makedirs(d, exist_ok=True)
    # Owner-only: tool outputs may hold sensitive data and /tmp is world-readable.
    try:
        chmod_posix(d, 0o700)
        chmod_posix(_session_root(), 0o700)
    except OSError:
        pass


import atexit as _atexit
import shutil as _shutil


@_atexit.register
def _cleanup_tool_results_session_dir() -> None:
    """Remove this session's results tree on exit (was never cleaned up)."""
    _shutil.rmtree(_session_root(), ignore_errors=True)


def get_persistence_threshold(tool_name: str, declared_max_result_size_chars: float) -> float:
    """Resolve the effective persistence threshold (getPersistenceThreshold).

    Infinity is a hard opt-out (checked first). Otherwise clamp the declared cap
    by the system-wide default. (The GrowthBook override is substrate-absent.)
    """
    if not math.isfinite(declared_max_result_size_chars):
        return declared_max_result_size_chars
    return min(declared_max_result_size_chars, DEFAULT_MAX_RESULT_SIZE_CHARS)


def generate_preview(content: str, max_bytes: int) -> tuple:
    """Truncate at a newline boundary when possible (generatePreview).

    Returns (preview, has_more)."""
    if len(content) <= max_bytes:
        return content, False
    truncated = content[:max_bytes]
    last_newline = truncated.rfind("\n")
    cut_point = last_newline if last_newline > max_bytes * 0.5 else max_bytes
    return content[:cut_point], True


def is_tool_result_content_empty(content) -> bool:
    """True when content is empty or whitespace-only (isToolResultContentEmpty).

    codemason tool results reach this as strings; non-string falls through to
    non-empty."""
    if not content:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    return False


def persist_tool_result(content_str: str, tool_use_id: str, is_json: bool = False) -> dict:
    """Write the content to disk and return persistence info (persistToolResult).

    Uses exclusive create ('x') so a result already persisted on a prior turn
    (same deterministic tool_use_id) is not rewritten. Returns {'error': ...} on
    failure so the caller can fall back to the original content.
    """
    ensure_tool_results_dir()
    filepath = get_tool_result_path(tool_use_id, is_json)
    try:
        with open(filepath, "x", encoding="utf-8") as f:
            f.write(content_str)
    except FileExistsError:
        # Already persisted on a prior turn — fall through to preview.
        pass
    except OSError as e:
        return {"error": str(e)}

    preview, has_more = generate_preview(content_str, PREVIEW_SIZE_BYTES)
    return {
        "filepath": filepath,
        "originalSize": len(content_str),
        "isJson": is_json,
        "preview": preview,
        "hasMore": has_more,
    }


def build_large_tool_result_message(result: dict) -> str:
    """Build the <persisted-output> preview message (buildLargeToolResultMessage)."""
    message = f"{PERSISTED_OUTPUT_TAG}\n"
    message += (
        f"Output too large ({_format_file_size(result['originalSize'])}). "
        f"Full output saved to: {result['filepath']}\n\n"
    )
    message += f"Preview (first {_format_file_size(PREVIEW_SIZE_BYTES)}):\n"
    message += result["preview"]
    message += "\n...\n" if result["hasMore"] else "\n"
    message += PERSISTED_OUTPUT_CLOSING_TAG
    return message


def process_tool_result_content(
    content: str,
    tool_name: str,
    tool_use_id: str,
    max_result_size_chars: float,
) -> str:
    """Map a tool's result string for the API, persisting it when too large.

    codemason equivalent of processToolResultBlock → maybePersistLargeToolResult
    (the result is already the mapped string here, so we operate on it directly):
      - empty content → a short marker so the model always has something to react
        to (inc-4586: a bare empty tool_result tail makes some models emit a stop
        sequence and end the turn with no output)
      - content over the effective threshold → persisted to disk, replaced with
        the <persisted-output> preview message
      - otherwise → unchanged
    On persist failure the original content is returned (never silently lost).
    """
    if is_tool_result_content_empty(content):
        return f"({tool_name} completed with no output)"

    threshold = get_persistence_threshold(tool_name, max_result_size_chars)
    if len(content) <= threshold:
        return content

    result = persist_tool_result(content, tool_use_id)
    if "error" in result:
        # Persistence failed — return original content unchanged.
        return content
    return build_large_tool_result_message(result)
