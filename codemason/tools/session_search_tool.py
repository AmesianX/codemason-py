"""SessionSearch tool — full-text recall over this project's past sessions.

The local counterpart of claude.ai's conversation_search: transcripts are JSONL
files on disk here, so past-conversation recall is a search over them
(utils/session_index.py owns the index). The tool is read-only and scoped to
the current project directory's sessions; the current session is never in the
results.
"""

import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from pydantic import Field

from ..utils.formatting import Colors, colorize
from ..utils.i18n import _
from ..utils.session_index import search
from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultFailure,
    ValidationResultSuccess,
)

SESSION_SEARCH_TOOL_NAME = "SessionSearch"

_MAX_LIMIT = 20


class SessionSearchInput(ToolInputSchema):
    query: str = Field(
        ...,
        description=(
            "Words to search for (ANDed, prefix-matched). \"Quote a word or "
            "phrase\" for exact matching."
        ),
    )
    limit: int = Field(
        default=8, ge=1, le=_MAX_LIMIT,
        description="Maximum number of sessions to return",
    )


def _short_id(session_id: str) -> str:
    return session_id[:8] if len(session_id) > 8 else session_id


def _day(ts: str) -> str:
    # Timestamps are ISO-8601 Z; the date part is enough to place a hit in time.
    return ts[:10] if ts else ''


def _format_for_assistant(groups: List[Dict[str, Any]], query: str) -> str:
    if not groups:
        return (
            f'No past-session messages match "{query}" in this project. '
            'Loosen the query (fewer words, or a * prefix wildcard) before '
            'concluding the topic was never discussed here.'
        )
    hit_count = sum(len(g['hits']) for g in groups)
    lines = [
        f'Found {hit_count} matching message(s) in {len(groups)} past session(s) '
        f'for "{query}". Matched terms are [bracketed]; excerpts are from this '
        'project\'s transcripts, best match first.'
    ]
    for i, g in enumerate(groups, 1):
        title = g.get('first_prompt') or '(no prompt)'
        newest = max((h.get('ts') or '' for h in g['hits']), default='')
        head = f'{i}. session {_short_id(g["session_id"])} — "{title}"'
        if newest:
            head += f' ({_day(newest)})'
        lines.append('')
        lines.append(head)
        for h in g['hits']:
            day = _day(h.get('ts') or '')
            stamp = f' {day}' if day else ''
            lines.append(f'   - [{h.get("role")}{stamp}] {h.get("snippet")}')
    lines.append('')
    lines.append('The user can reopen any of these sessions with /resume.')
    return '\n'.join(lines)


class SessionSearchTool(Tool):
    name = SESSION_SEARCH_TOOL_NAME
    input_schema = SessionSearchInput

    @classmethod
    async def description(cls, **kwargs) -> str:
        return "Search past sessions of this project"

    @classmethod
    def is_read_only(cls) -> bool:
        return True

    @classmethod
    async def prompt(cls) -> str:
        return (
            "Searches the transcripts of this project's PAST sessions (earlier "
            "conversations in this working directory) by keyword and returns "
            "matching excerpts of user and assistant messages, grouped by "
            "session.\n\n"
            "Usage:\n"
            "- Use this when the user refers to earlier work you have no context "
            "for: \"as we discussed\", \"last time\", \"that bug we fixed\", "
            "\"continue where we left off\". Search BEFORE answering that you "
            "don't know or asking the user to re-explain.\n"
            "- Also use it to recover your own past findings in this project "
            "(a diagnosis, a decision, a command that worked) instead of "
            "re-deriving them.\n"
            "- query: words are ANDed and match by prefix (auth matches "
            "authentication; a Korean word matches itself plus attached "
            "particles). \"Quote a word or phrase\" for exact matching.\n"
            "- Matched terms come back [bracketed]. The current session is never "
            "included, and only this project's sessions are searched.\n"
            "- MEMORY.md holds durable saved facts; this tool recovers specifics "
            "that were never saved as memories. If a recovered fact is worth "
            "keeping, save it as a memory.\n"
            "- Results are excerpts, not full turns. To pick a session back up, "
            "tell the user which one and suggest /resume — you cannot load it "
            "yourself."
        )

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        if not isinstance(input_data, dict):
            return ''
        return colorize(str(input_data.get('query') or ''), Colors.DIM)

    @classmethod
    def render_tool_result_message(cls, data: Any, verbose: bool = False) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        groups = data.get('groups') or []
        if not groups:
            return colorize(_('No matches in past sessions'), Colors.DIM)
        hits = sum(len(g.get('hits') or []) for g in groups)
        return colorize(
            _('{hits} match(es) in {sessions} past session(s)').format(
                hits=hits, sessions=len(groups)),
            Colors.DIM)

    @classmethod
    async def validate_input(cls, input_data, context) -> ValidationResult:
        if not str(input_data.get('query') or '').strip():
            return ValidationResultFailure(message='query is required',
                                           meta={'errorCode': 1})
        return ValidationResultSuccess()

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        query = str(input_data.get('query') or '').strip()
        limit = int(input_data.get('limit') or 8)
        # First search after new sessions accumulate re-indexes them — file IO
        # and sqlite work that must not stall the event loop.
        try:
            groups = await asyncio.to_thread(search, query, limit)
        except ValueError as e:
            data = {'query': query, 'groups': []}
            yield ToolResultSuccess(
                result_for_assistant=f'Invalid query: {e}', data=data)
            return
        data = {'query': query, 'groups': groups}
        yield ToolResultSuccess(
            result_for_assistant=_format_for_assistant(groups, query), data=data)
