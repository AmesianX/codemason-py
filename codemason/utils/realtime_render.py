"""DSML-safe incremental stream filter for realtime (-rt/--realtime) rendering.

The normal path buffers the whole assistant message and renders it once at the
end (so clean_special_tokens / tool-call recovery can run on the complete text).
Realtime mode instead prints content deltas AS THEY ARRIVE — but the raw stream
contains markup that must never reach the screen:

  * DSML tool-call markup (full-width OR ascii pipe):  <｜DSML｜invoke name="..."> …
    Once this starts, the rest of the message is a tool call → SUPPRESS to end
    (the structured tool_calls channel / end-of-stream recovery handles it).
  * <think> … </think> leaked reasoning blocks → drop the WHOLE block (matches
    clean_special_tokens' THINKING_MODE behaviour when reasoning isn't separated).
  * <|special|> control tokens (e.g. <|end|>, <|channel|>) → strip inline.

The filter holds back a trailing fragment that could still grow into one of the
above markers (so a marker split across two chunks is never leaked), emitting
only bytes that are provably safe. `feed(delta)` returns text safe to print now;
`flush()` returns any held-back text at end of stream (markers never completed →
they were real text after all).
"""

import re

_PIPE = r'[|｜]'
# Triggers that, once seen, suppress the REST of the message (tool call body).
# BOTH polarities: the server's tool-call parser can terminate a call early and
# spill the remainder of the argument into `content`, so the opening tags are
# consumed upstream and the message arrives carrying only the ORPHAN closing
# tags (</｜DSML｜parameter> …). Matching '<｜DSML｜' alone let those fall through
# every branch below and get printed verbatim, one '<' at a time.
_DSML_START = re.compile(r'</?' + _PIPE + r'DSML' + _PIPE)
# Inline strippable control token: <|word|> (ascii or full-width pipe).
_SPECIAL_TOKEN = re.compile(r'<' + _PIPE + r'[^|｜<>]*' + _PIPE + r'>')
# Bare-XML tool-call markup, the form GLM emits: <tool_call>, <arg_key>,
# <arg_value> and their closers, with no pipes anywhere.
#
# Everything else stripped here is pipe-delimited (<|DSML|>, <|word|>), and this
# family matched none of it. Observed 2026-09-07: a turn whose visible answer
# ended "dessin</arg_value></tool_call>" — the call itself had been parsed
# correctly on the structured channel and its commands ran, so only the rendered
# text was wrong, but it puts unexplained markup in the middle of the model's
# prose. The module already noted that a message can arrive carrying ORPHAN
# closing tags; it only ever handled the DSML spelling.
_XML_TOOL_TAGS = ('tool_call', 'tool_response', 'arg_key', 'arg_value', 'args')
_XML_TOOL_TAG = re.compile(r'</?(?:' + '|'.join(_XML_TOOL_TAGS) + r')\s*>', re.IGNORECASE)

_THINK_OPEN = re.compile(r'<think>', re.IGNORECASE)
_THINK_CLOSE = re.compile(r'</think>', re.IGNORECASE)


# Longest marker this filter can ever hold for, in bytes.
#
# The XML family is a fixed list (longest '</tool_response>', 16), but the
# special/DSML family is '<|' + arbitrary text + '|>' — the grammar puts no
# bound on the middle, so a '<|' whose closing pipe never arrives would hold the
# ENTIRE rest of the stream in _pending, and every later delta re-lowercases all
# of it. That is O(n^2) on a buffer that never stops growing: a live hang, not
# just slow rendering.
#
# Measured: across both bundled tokenizers (DSV4-Flash, glm-5.3-flash) the
# longest special token is 29 bytes ('<|place_holder_mm_span_0435|>'). 64 leaves
# better than 2x headroom, so no real marker is ever cut short, while anything
# longer is treated as what it is — prose that happens to start '<'.
_MAX_MARKER_LEN = 64


def _could_be_marker_prefix(s: str) -> bool:
    """True if s (starting with '<') might still grow into a suppressible/strip
    marker, so we should hold it instead of emitting."""
    # Bound the hold BEFORE any whole-string work (the .lower() below).
    if len(s) > _MAX_MARKER_LEN:
        return False
    # Prefix of a special/DSML token: '<' (optionally '/') + pipe + (letters,
    # no closing yet). The '/' branch keeps a closing tag split across chunks
    # ('</｜DS' + 'ML｜parameter>') held instead of leaked.
    if re.fullmatch(r'</?' + _PIPE + r'[^|｜<>]*', s):
        return True
    if s in ('<',):
        return True
    # Prefix of <think> / </think> or of a bare-XML tool tag. Held rather than
    # emitted, so a tag split across two stream chunks is not printed in halves
    # before it can be recognised.
    _tags = ['<think>', '</think>']
    for _t in _XML_TOOL_TAGS:
        _tags += ['<%s>' % _t, '</%s>' % _t]
    for tag in _tags:
        if tag.startswith(s.lower()) and len(s) < len(tag):
            return True
    if s in ('</',):
        return True
    return False


class RealtimeStreamFilter:
    def __init__(self):
        self._pending = ''          # unclassified tail (may hold a partial marker)
        self._suppress_all = False  # DSML tool call started → drop everything after
        self._in_think = False      # inside <think> … </think> → drop inner

    def feed(self, delta: str) -> str:
        if self._suppress_all or not delta:
            return ''
        self._pending += delta
        out = []
        while self._pending:
            if self._in_think:
                m = _THINK_CLOSE.search(self._pending)
                if not m:
                    # keep a tail that could be a partial '</think>'
                    keep = _tail_partial(self._pending, '</think>')
                    self._pending = self._pending[len(self._pending) - keep:] if keep else ''
                    return ''.join(out)
                self._pending = self._pending[m.end():]
                self._in_think = False
                continue

            lt = self._pending.find('<')
            if lt == -1:
                out.append(self._pending)
                self._pending = ''
                break
            # emit safe text before the '<'
            if lt > 0:
                out.append(self._pending[:lt])
                self._pending = self._pending[lt:]

            # now self._pending starts with '<'
            if _DSML_START.match(self._pending):
                self._suppress_all = True
                self._pending = ''
                break
            m = _THINK_OPEN.match(self._pending)
            if m:
                self._pending = self._pending[m.end():]
                self._in_think = True
                continue
            # Bare-XML tool markup: strip the tag, keep going. Unlike DSML this
            # does NOT suppress the rest — the leak seen in practice was orphan
            # CLOSING tags after real prose, and swallowing everything behind
            # them would eat the answer.
            m = _XML_TOOL_TAG.match(self._pending)
            if m:
                self._pending = self._pending[m.end():]
                continue

            m = _SPECIAL_TOKEN.match(self._pending)
            if m:
                self._pending = self._pending[m.end():]   # strip <|...|>
                continue
            # incomplete? hold and wait for more. (_could_be_marker_prefix
            # already covers every DSML-start prefix via its <PIPE… fullmatch.)
            if _could_be_marker_prefix(self._pending):
                return ''.join(out)
            # definitely not a marker — the '<' is real text; emit it and move on
            out.append('<')
            self._pending = self._pending[1:]
        return ''.join(out)

    def flush(self) -> str:
        """End of stream: any held-back tail never completed a marker, so it was
        real text (unless we're mid-DSML/think, in which case it's dropped)."""
        if self._suppress_all or self._in_think:
            self._pending = ''
            return ''
        # a lone trailing '<x' that looked like a maybe-marker but never resolved
        rest = self._pending
        self._pending = ''
        # drop only if it's a bare incomplete control-token opener; otherwise
        # emit. Same bound as _could_be_marker_prefix: past it this is prose.
        if len(rest) > _MAX_MARKER_LEN:
            return rest
        if re.fullmatch(r'</?' + _PIPE + r'[^|｜<>]*', rest):
            return ''
        # A complete tool tag held to the very end is markup, not text.
        if _XML_TOOL_TAG.fullmatch(rest):
            return ''
        return rest


class MarkdownBlockBuffer:
    """Groups streamed safe text into complete MARKDOWN BLOCKS so each can be
    rendered (render_markdown) and committed as a whole — instead of raw
    per-char writes that (a) can't be markdown-formatted and (b) leave dangling
    partial lines that a prompt repaint wipes. A block is a blank-line-delimited
    paragraph, or a whole ``` fenced code block (blank lines inside a fence are
    NOT boundaries, so code/tables/lists stay intact). feed() returns the blocks
    that are now complete; flush() returns the trailing partial block at stream
    end.
    """

    def __init__(self):
        self._pending = ''      # bytes not yet split into complete lines
        self._cur = []          # complete lines of the block being built
        self._in_fence = False
        self._fence = ''        # the opening fence marker (``` or ~~~)

    def feed(self, text: str) -> list:
        self._pending += text
        blocks = []
        while '\n' in self._pending:
            line, self._pending = self._pending.split('\n', 1)
            stripped = line.lstrip()
            if stripped.startswith('```') or stripped.startswith('~~~'):
                if not self._in_fence:
                    self._in_fence = True
                    self._fence = stripped[:3]
                    self._cur.append(line)
                elif stripped.startswith(self._fence):
                    self._in_fence = False
                    self._cur.append(line)
                    blocks.append('\n'.join(self._cur))   # complete code block
                    self._cur = []
                else:
                    self._cur.append(line)
                continue
            if not self._in_fence and line.strip() == '':
                body = '\n'.join(self._cur).strip('\n')
                if body.strip():
                    blocks.append(body)                    # complete paragraph
                self._cur = []
                continue
            self._cur.append(line)
        return blocks

    def flush(self) -> str:
        if self._pending:
            self._cur.append(self._pending)
            self._pending = ''
        body = '\n'.join(self._cur).strip('\n')
        self._cur = []
        return body


def _tail_partial(s: str, marker: str) -> int:
    """Length of the longest suffix of s that is a proper prefix of marker."""
    maxlen = min(len(s), len(marker) - 1)
    for k in range(maxlen, 0, -1):
        if marker.lower().startswith(s[-k:].lower()):
            return k
    return 0
