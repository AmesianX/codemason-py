"""
WebSearch Tool for CodeMason
Reconstruction of src/tools/WebSearchTool/ from Claude Code on the local substrate.

The original WebSearchTool is NOT a single literal search: its call() spins up a
sub-LLM that drives Anthropic's server-side web_search_20250305 tool (max_uses: 8).
That sub-LLM expands the user's intent into several queries, reads page *content*
(not just SERP snippets), and emits synthesized text + citations.

codemason has no server-side web_search, but it already owns the two halves that
tool provided locally:
  - a model loop            -> services.backend.query_model (routes to vllm.py / llamacpp.py)
  - a fetch+extract substrate -> web_fetch_tool.get_url_markdown_content (HTML->md)
The only piece genuinely missing is the SERP engine itself, which `ddgs` supplies.

So rather than collapsing the tool into one literal DuckDuckGo call (which drops
multi-search, page-content reading, and breaks domain filtering), we reconstruct
the original's structure locally:

  1. query expansion   - one small LLM call rewrites the intent into N focused
                         queries (recovers the original's multi-search / max_uses
                         loop). Falls back to the literal query if no model is
                         reachable, so the tool still works headless / in tests.
  2. parallel SERP     - every query hits ddgs with a *merged* backend list
                         (duckduckgo+brave+mojeek+google); ddgs aggregates and
                         ranks across whichever engines respond. Single-engine
                         reliability swings wildly run-to-run, so the merge is the
                         robustness mechanism, not a nicety. Queries run concurrently.
  3. top-K page fetch  - the highest-ranked results are fetched and reduced to a
                         short extract via the existing web_fetch substrate
                         (recovers the original's page-content reading).

Domain filtering is two-layer: `site:` / `-site:` operators injected into the
query bias the engines toward the right pages (recall), and a hostname filter on
the merged results enforces the contract (precision). The prior port did ONLY the
filter over a fixed generic result set, so allowed_domains kept whatever ranked
organically — verified to return 2/0/0 across repeats. The operators alone aren't
enough either: not every backend honors `site:` (verified ~3/8 leaks), so both
layers are required.

Schema (query / allowed_domains / blocked_domains), validation (incl. the
allowed/blocked mutual-exclusion, WebSearchTool.ts:244-251), the getWebSearchPrompt
description, and the trailing REMINDER line (WebSearchTool.ts:426-427) are ported
faithfully.
"""

import asyncio
import hashlib
import json
import os
import random
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import Field

from ..utils.formatting import Colors, colorize
from .base import (
    Tool,
    ToolInputSchema,
    ToolResult,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultSuccess,
    ValidationResultFailure,
    PermissionResult,
)

WEB_SEARCH_TOOL_NAME = 'WebSearch'

_MAX_RESULTS_PER_QUERY = 8
# Internal caps (not user-tunable): final merged result cap, how many top results
# get a page fetch, and per-page extract length.
_MAX_MERGED = 12
# 3 x 600 chars gave the model ~1.8k of page body for an entire search, and the
# extract is taken from the top of the page — on most sites that budget is spent
# entirely on nav chrome before any real content. Deep technical questions (the
# ones worth searching for) were answered from titles alone.
_FETCH_TOP_K = 5
_EXTRACT_CHARS = 4000


# All user-tunable WebSearch settings live in codemason's normal config system
# (config.py: config.json field + CODEMASON_* env override + missing-key backfill),
# NOT module-level env reads — so they honour ~/.codemason/config.json like every
# other setting and materialize into the file. Read at call time via _ws_config()
# so config.json edits take effect without a reimport.
#
#   ddgs backends      -> websearch_backends  (merged engine list; ddgs ranks
#                         across whichever respond — no single engine is reliable)
#   rate-limit posture -> websearch_max_expansions / _throttle_min / _throttle_max
#                         / _retry_attempts / _backoff_base / _backoff_cap. Each
#                         keeps one WebSearch from becoming a request BURST that
#                         trips per-IP anti-scraping (the upstream original
#                         delegated to Anthropic's server-side web_search and never
#                         hit this; codemason scrapes from the client IP, so it does).
#   result cache       -> websearch_cache_ttl / websearch_cache_dir
#   SearXNG (primary)  -> searxng_url / searxng_timeout / searxng_lang


def _ws_config():
    """codemason Config for WebSearch settings, read at call time (config.json +
    env override). Falls back to a defaults instance if config is unavailable
    (headless / tests) so the tool still works."""
    from ..utils.config import Config, get_config
    try:
        return get_config()
    except Exception:
        return Config()


def _searxng_conf() -> Tuple[str, float, str, str]:
    """(url, timeout, lang, api_key) for the SearXNG primary backend."""
    c = _ws_config()
    return (
        (c.searxng_url or '').strip(),
        float(c.searxng_timeout),
        (c.searxng_lang or 'auto'),
        (c.searxng_api_key or '').strip(),
    )


def _cache_dir() -> Path:
    """Cache root, under codemason's config home by default (relocatable via
    config.websearch_cache_dir / CODEMASON_WEBSEARCH_CACHE_DIR / CODEMASON_CONFIG_DIR)."""
    cache_dir = _ws_config().websearch_cache_dir
    if cache_dir:
        return Path(cache_dir)
    try:
        from ..utils.config import CONFIG_DIR
        return Path(CONFIG_DIR) / 'cache' / 'websearch'
    except Exception:
        return Path.home() / '.codemason' / 'cache' / 'websearch'


def _cache_key(query: str, allowed: List[str], blocked: List[str]) -> str:
    payload = json.dumps(
        {'q': query, 'a': sorted(d.lower() for d in allowed), 'b': sorted(d.lower() for d in blocked)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]


def _cache_get(key: str) -> Optional[List[Dict[str, str]]]:
    """Return cached results if a fresh entry exists, else None. Any I/O or parse
    error degrades to a miss (never raises into the search path)."""
    ttl = _ws_config().websearch_cache_ttl
    if ttl <= 0:
        return None
    path = _cache_dir() / f'{key}.json'
    try:
        raw = path.read_text(encoding='utf-8')
        obj = json.loads(raw)
        if (time.time() - float(obj.get('ts', 0))) > ttl:
            return None
        results = obj.get('results')
        return results if isinstance(results, list) and results else None
    except Exception:
        return None


def _cache_put(key: str, results: List[Dict[str, str]]) -> None:
    """Persist a non-empty result set. Best-effort: failures are swallowed."""
    if _ws_config().websearch_cache_ttl <= 0 or not results:
        return
    try:
        d = _cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        # Per-process temp name. The cache is shared by every codemason process
        # on this machine, so two of them missing on the SAME query wrote the
        # same `<key>.json.tmp` at the same time: each truncated the other's
        # bytes and whichever renamed first published a half-written entry, which
        # then failed to parse for the rest of its TTL — a cache that quietly
        # stopped being a cache. The rename is still the atomic publish; it is
        # only the staging file that has to be private.
        tmp = d / f'{key}.json.{os.getpid()}.tmp'
        try:
            tmp.write_text(json.dumps({'ts': time.time(), 'results': results}), encoding='utf-8')
            tmp.replace(d / f'{key}.json')
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
    except Exception:
        pass


def _get_description() -> str:
    # Ported from upstream getWebSearchPrompt (prompt.ts) — EXCEPT the trailing
    # "Search in the language that fits the topic" block, which is a codemason
    # addition. Upstream has no language guidance (and its server-side search is
    # US-region-locked), so Korea-specific queries tend to get anglicized and lose
    # their best results; this steers the model to keep Korean and to run
    # Korean+English in parallel. Paired with the same guidance in _expand_queries.
    current_month_year = date.today().strftime('%B %Y')
    return f"""
- Allows the assistant to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search result information formatted as search result blocks, including links as markdown hyperlinks
- Use this tool for accessing information beyond the model's knowledge cutoff
- Runs a web search and fetches the most relevant pages to ground the answer

CRITICAL REQUIREMENT - You MUST follow this:
  - After answering the user's question, you MUST include a "Sources:" section at the end of your response
  - In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)
  - This is MANDATORY - never skip including sources in your response
  - Example format:

    [Your answer here]

    Sources:
    - [Source Title 1](https://example.com/1)
    - [Source Title 2](https://example.com/2)

Usage notes:
  - Domain filtering is supported to include or block specific websites

IMPORTANT - Use the correct year in search queries:
  - The current month is {current_month_year}. You MUST use this year when searching for recent information, documentation, or current events.
  - Example: If the user asks for "latest React docs", search for "React documentation" with the current year, NOT last year

IMPORTANT - Search in the language that fits the topic:
  - Search in the language most likely to surface the best sources. A topic rooted in one country or language — its people, media, news, products, local services, regulations — is documented in that language first, and the best sources may never be translated. Search in it.
  - Transliterating a name into the Latin alphabet usually loses those sources: the local spelling is what the local pages actually use, so search the name as it is written at home.
  - When two languages could both help, issue BOTH queries so coverage runs in parallel instead of committing to one.
  - For globally English or technical topics (programming, standards, international products), English is usually best — do NOT force a local language where English clearly wins.
"""


class WebSearchInput(ToolInputSchema):
    query: str = Field(..., min_length=2, description="The search query to use")
    allowed_domains: Optional[List[str]] = Field(
        default=None,
        description="Only include search results from these domains",
    )
    blocked_domains: Optional[List[str]] = Field(
        default=None,
        description="Never include search results from these domains",
    )


def _build_query(query: str, allowed: List[str], blocked: List[str]) -> str:
    """Translate domain filters into SERP operators the engines understand.

    allowed_domains -> `(site:a.com OR site:b.com) <query>` (provider does the
    site-restricted search), blocked_domains -> `<query> -site:a.com ...`.
    allowed/blocked are mutually exclusive (rejected at validation), but applying
    both independently here is harmless if that ever changes."""
    q = query
    if allowed:
        sites = ' OR '.join(f'site:{d.lower().lstrip(".")}' for d in allowed if d)
        if sites:
            q = f'({sites}) {q}'
    if blocked:
        q = q + ''.join(f' -site:{d.lower().lstrip(".")}' for d in blocked if d)
    return q


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or '').lower()
    except Exception:
        return ''


def _domain_match(host: str, domains: List[str]) -> bool:
    """host == domain or a subdomain of it (suffix match on a label boundary)."""
    host = host.lower()
    for d in domains:
        d = d.lower().lstrip('.')
        if d and (host == d or host.endswith('.' + d)):
            return True
    return False


def _searxng_search(
    query: str, max_results: int = _MAX_RESULTS_PER_QUERY
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """Query a self-hosted SearXNG instance's JSON API.

    Returns (results, error):
      - (list, None)  : instance reached and answered (list may be empty)
      - (None, error) : instance unavailable — unreachable, non-200, or non-JSON
                        (e.g. the instance has the `json` format disabled). A None
                        first element is the caller's signal to fall back to ddgs.
    Uses only stdlib urllib so it adds no dependency and stays portable to the
    machine codemason is copied onto."""
    url_base, timeout, lang, api_key = _searxng_conf()
    if not url_base:
        return None, 'searxng disabled'
    import urllib.error
    import urllib.parse
    import urllib.request

    base = url_base.rstrip('/')
    params = urllib.parse.urlencode(
        {'q': query, 'format': 'json', 'safesearch': '0', 'language': lang}
    )
    headers = {'User-Agent': 'codemason-websearch/1.0', 'Accept': 'application/json'}
    # API-key auth: the reverse proxy in front of a public instance gates on this
    # shared-secret header and 403s anything without it. Only sent when configured.
    if api_key:
        headers['X-API-Key'] = api_key
    req = urllib.request.Request(f'{base}/search?{params}', headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, 'status', 200) != 200:
                return None, f'searxng HTTP {resp.status}'
            raw = resp.read().decode('utf-8', 'replace')
        # A JSON-format-disabled instance answers 200 with an HTML page; json.loads
        # then raises and we correctly report unavailable -> ddgs fallback.
        data = json.loads(raw)
    except Exception as e:
        return None, f'searxng unavailable: {e}'

    out: List[Dict[str, str]] = []
    seen = set()
    for h in (data.get('results') or []):
        url = h.get('url') or ''
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({'title': h.get('title', ''), 'url': url, 'body': h.get('content', '')})
        if len(out) >= max_results:
            break
    return out, None


def _search_sync(query: str, max_results: int = _MAX_RESULTS_PER_QUERY) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Blocking ddgs search for one query across the merged backend list.
    Returns (results, error). ddgs raises DDGSException('No results found.') when
    every engine comes back empty (genuine-empty and all-rate-limited are
    indistinguishable at this layer) — surfaced as the error string so the caller
    can tell an all-failed sweep from a hit."""
    from ddgs import DDGS
    c = _ws_config()
    backends = c.websearch_backends
    attempts = max(1, c.websearch_retry_attempts)
    backoff_base, backoff_cap = c.websearch_backoff_base, c.websearch_backoff_cap
    err: Optional[str] = None
    # Say which attempt is running. Worst case is the per-engine timeout times
    # `attempts`, plus backoff between them — 51 seconds on the shipped
    # defaults, all of it previously behind an unchanging spinner.
    from ..utils.tokens import set_wait_note
    for attempt in range(attempts):  # exponential backoff on rate-limit flakiness
        set_wait_note('searching (attempt %d/%d)' % (attempt + 1, attempts))
        try:
            # Context manager so the HTTP session/sockets close (avoids the
            # ResourceWarning: unclosed SSLSocket leak).
            with DDGS() as ddg:
                hits = ddg.text(query, max_results=max_results, backend=backends)
            out: List[Dict[str, str]] = []
            seen = set()
            for h in (hits or []):
                url = h.get('href') or h.get('url') or ''
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append({'title': h.get('title', ''), 'url': url, 'body': h.get('body', '')})
            set_wait_note(None)
            return out, None
        except Exception as e:
            err = f'{e}'
            # Back off before the next attempt (never after the last). Immediate
            # retries just re-hit the same rate limit; spacing them out is what
            # lets a transient block clear.
            if attempt < attempts - 1:
                delay = min(backoff_cap, backoff_base * (2 ** attempt))
                time.sleep(delay + random.uniform(0, delay))
    set_wait_note(None)
    return [], err


def _message_text(msg: Any) -> str:
    """Extract concatenated text blocks from a query_model AssistantMessage."""
    content = getattr(getattr(msg, 'message', None), 'content', None) or []
    parts: List[str] = []
    for b in content:
        btype = getattr(b, 'type', None) if not isinstance(b, dict) else b.get('type')
        if btype == 'text':
            text = getattr(b, 'text', None) if not isinstance(b, dict) else b.get('text', '')
            if text:
                parts.append(text)
    return '\n'.join(parts).strip()


def _parse_query_list(text: str) -> List[str]:
    """Pull a JSON array of query strings out of the planner model's output."""
    if not text:
        return []
    m = re.search(r'\[.*\]', text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    return [x.strip() for x in arr if isinstance(x, str) and x.strip()]


async def _expand_queries(query: str, context: Any) -> List[str]:
    """Rewrite the user's intent into several focused queries via one small LLM
    call — the local stand-in for the original sub-LLM issuing up to max_uses
    searches. Always returns at least [query]; degrades to exactly that whenever
    no model is reachable (no context, server down, unparseable output), so the
    tool stays functional headless and in tests."""
    max_exp = _ws_config().websearch_max_expansions
    expansions: List[str] = []
    if context is not None and max_exp > 0:
        try:
            from ..utils.messages import create_user_message
            from ..services.backend import query_model

            today = date.today().strftime('%B %Y')
            prompt = (
                f'Today is {today}. A user wants to web-search for: "{query}".\n'
                f'Produce up to {max_exp} focused web-search queries that together '
                f'cover this intent (synonyms, sub-questions, recency-scoped variants). '
                f'Use the current year where recency matters. '
                # Two-language coverage. The anchor (verbatim) query is searched
                # as-is, so an expansion in the SAME language just paraphrases what
                # is already covered; the value is in spanning the other one. Which
                # two languages those are depends on the query, not on where this
                # was written — a topic is documented in the language it happened
                # in, and the interface language is usually English.
                f'LANGUAGE RULE (important): the anchor query is searched verbatim, so '
                f'at least one expansion MUST be a natural TRANSLATION into another '
                f'language rather than a same-language rephrase. '
                f'If the query is not in English, translate it into English. '
                f'If the query is in English but the topic belongs to a particular '
                f'country or language community (a person, outlet, product, place or '
                f'regulation documented there), translate it into that language, '
                f'written the way it is written locally. '
                f'This spans both audiences instead of committing to one. '
                f'For purely global or technical English topics, keep the expansion in English. '
                f'Respond with ONLY a JSON array of query strings and nothing else.'
            )
            signal = getattr(context, 'abort_controller', None)
            msg = await query_model(
                messages=[create_user_message(prompt)],
                system_prompt=['You are a search-query planner for a web search tool.'],
                tools=[],
                signal=signal,
                # 256 truncated Korean expansions to an EMPTY response: Korean
                # tokenizes longer and DSV4-Flash emits reasoning (thinking is a
                # server default, inert to disable client-side), so the JSON array
                # never fit. 768 leaves headroom; the planner still stops early on
                # its short array, so the ceiling adds no latency in the common case.
                options={'max_tokens': 768},
            )
            expansions = _parse_query_list(_message_text(msg))
        except Exception:
            expansions = []

    # Verbatim original first (anchor), then expansions; dedup preserving order.
    ordered: List[str] = []
    for q in [query] + expansions:
        q = (q or '').strip()
        if q and q not in ordered:
            ordered.append(q)
        if len(ordered) >= max_exp + 1:
            break
    return ordered


_BOT_CHALLENGE_MARKERS = (
    'establishing a secure connection',
    'challenge loading',
    'bunny-shield',
    'just a moment...',
    'checking your browser',
    'cf-browser-verification',
    'enable javascript and cookies to continue',
    'verifying you are human',
    'ddos-guard',
)


def _is_bot_challenge(flat_text: str) -> bool:
    """True if the page is an anti-bot interstitial rather than real content.
    Only applied to short pages — a long article legitimately discussing
    Cloudflare must not be discarded."""
    if len(flat_text) > 1500:
        return False
    low = flat_text.lower()
    return any(m in low for m in _BOT_CHALLENGE_MARKERS)


async def _fetch_extract(
    url: str, signal: Any, max_chars: int = _EXTRACT_CHARS
) -> Tuple[str, Optional[str]]:
    """Fetch a result page and reduce it to a short whitespace-collapsed extract,
    reusing the web_fetch substrate.

    Returns (extract, error). Failures used to collapse to '' — a blocked host, a
    timeout and a genuinely empty page were indistinguishable, so a search whose
    pages all failed looked exactly like a search that found nothing worth
    quoting. The reason is now surfaced to the model (and to whoever is debugging
    why a search came back thin)."""
    try:
        from .web_fetch_tool import get_url_markdown_content, RedirectInfo
        res = await get_url_markdown_content(url, signal)
        if isinstance(res, RedirectInfo):
            return '', 'redirected off-origin'
        content = (res.get('content') or '').strip()
        if not content:
            return '', 'empty page'
        flat = ' '.join(content.split())
        if _is_bot_challenge(flat):
            # Cloudflare / Bunny Shield / DDoS-Guard interstitials return HTTP 200
            # with a few hundred characters of "Establishing a secure connection".
            # Passed through as content it reads like a real (if useless) page —
            # xdaforums, a primary source for Android work, looked "found but
            # empty" on every search.
            return '', 'bot challenge — site requires a JS browser'
        return flat[:max_chars], None
    except Exception as e:
        return '', f'{type(e).__name__}: {e}'


def _format_results(query: str, merged: List[Dict[str, str]], searched: Optional[List[str]]) -> str:
    """Render the merged results as the assistant-facing text. Shared by the live
    search path and the cache-hit path so both emit byte-identical output (minus
    the '(searched: ...)' line, which only the live multi-query path has)."""
    lines = [f'Search results for "{query}":', '']
    if searched and len(searched) > 1:
        lines.append(f'(searched: {"; ".join(searched)})')
        lines.append('')
    for r in merged:
        lines.append(f"- [{r['title']}]({r['url']})")
        snippet = r.get('extract') or r.get('body')
        if snippet:
            lines.append(f"  {snippet}")
        if r.get('extract_error'):
            # Say so rather than silently showing a title with no body — otherwise
            # "the page could not be read" is indistinguishable from "the page had
            # nothing useful", and neither the model nor the user can tell.
            lines.append(f"  (page body not retrieved — {r['extract_error']})")
    # Verbatim reminder the original adds in
    # mapToolResultToToolResultBlockParam (WebSearchTool.ts:426-427).
    lines.append(
        '\nREMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.'
    )
    return '\n'.join(lines)


class WebSearchTool(Tool):
    """Search the web (ddgs, merged backends) with intent expansion + page fetch."""

    name = WEB_SEARCH_TOOL_NAME
    input_schema = WebSearchInput
    DESCRIPTION = _get_description()

    @classmethod
    async def description(cls, **kwargs) -> str:
        return _get_description()

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return _get_description()

    @classmethod
    def is_read_only(cls) -> bool:
        return True

    @classmethod
    async def check_permissions(cls, input_data, context=None) -> PermissionResult:
        """Ask, despite being read-only — same reasoning as WebFetch: the query
        is model-controlled and leaves this machine, so it is an egress channel
        however small. Never reached under dangerously-skip-permissions.
        """
        return PermissionResult(behavior='ask')

    @classmethod
    async def is_enabled(cls) -> bool:
        return True

    @classmethod
    def needs_permissions(cls) -> bool:
        return True  # network access

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return 'WebSearch'

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        q = input_data.get('query', '')
        if not q or len(q) < 2:
            return ValidationResultFailure(result=False, message='query must be at least 2 characters')
        # WebSearchTool.ts:244-251 (errorCode 2): allowed and blocked domains are
        # mutually exclusive.
        if (input_data.get('allowed_domains') or []) and (input_data.get('blocked_domains') or []):
            return ValidationResultFailure(
                result=False,
                message='Error: Cannot specify both allowed_domains and blocked_domains in the same request',
            )
        return ValidationResultSuccess()

    @classmethod
    def get_tool_use_summary(cls, input_data):
        """The query, truncated for display (UI.tsx getToolUseSummary)."""
        from ..utils.formatting import TOOL_SUMMARY_MAX_LENGTH, truncate
        if not isinstance(input_data, dict):
            return None
        value = input_data.get('query')
        if not value:
            return None
        return truncate(str(value), TOOL_SUMMARY_MAX_LENGTH)

    @classmethod
    def get_activity_description(cls, input_data):
        """"Searching for <query>" (getActivityDescription)."""
        summary = cls.get_tool_use_summary(input_data)
        return f'Searching for {summary}' if summary else 'Searching the web'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        query = input_data.get('query', '') if isinstance(input_data, dict) else ''
        return colorize(str(query), Colors.DIM) if query else ''

    @classmethod
    async def call(
        cls, input_data: Dict[str, Any], context: Any, can_use_tool: Optional[Callable] = None
    ) -> AsyncGenerator[ToolResult, None]:
        query = input_data.get('query', '')
        allowed = input_data.get('allowed_domains') or []
        blocked = input_data.get('blocked_domains') or []
        signal = getattr(context, 'abort_controller', None) if context is not None else None

        # 0. cache lookup. A coding agent re-searches near-identical queries
        #    constantly; a fresh hit skips every upstream request (SERP + page
        #    fetch), which is the single biggest cut to rate-limit exposure.
        cache_key = _cache_key(query, allowed, blocked)
        cached = _cache_get(cache_key)
        if cached is not None:
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=_format_results(query, cached, None),
                data={'query': query, 'cached': True,
                      'content': [{'title': r['title'], 'url': r['url']} for r in cached]},
            )
            return

        # 1. expand intent -> multiple queries (recovers the multi-search loop).
        queries = await _expand_queries(query, context)

        # 2. SERIALIZED SERP, SearXNG-first. For each query: try the SearXNG
        #    instance (0-priority); if it's unavailable OR returns nothing, fall
        #    back to the ddgs scrape. Once SearXNG is proven unreachable in this
        #    call, stop trying it for the remaining queries (no repeated timeouts).
        #    Domain filters become site:/-site: operators per query. ddgs calls run
        #    one-at-a-time with jitter (was a concurrent gather blast of every query
        #    × every engine at once) so they don't present as a rate-limit-tripping
        #    BURST; a SearXNG hit skips that path entirely.
        built = [_build_query(q, allowed, blocked) for q in queries]
        searched: List[Tuple[List[Dict[str, str]], Optional[str]]] = []
        sources: set = set()
        _wc = _ws_config()
        lo = min(_wc.websearch_throttle_min, _wc.websearch_throttle_max)
        hi = max(_wc.websearch_throttle_min, _wc.websearch_throttle_max)
        searxng_live = bool(_searxng_conf()[0])  # cleared on first unreachable result
        ddgs_called = False                # gate the inter-call throttle
        for b in built:
            results: List[Dict[str, str]] = []
            err: Optional[str] = None

            if searxng_live:
                sx, sx_err = await asyncio.to_thread(_searxng_search, b, _MAX_RESULTS_PER_QUERY)
                if sx is None:
                    # Host unreachable / JSON disabled — don't retry it this call.
                    searxng_live = False
                    err = sx_err
                elif sx:
                    results = sx
                    sources.add('searxng')

            if not results:  # SearXNG off / unavailable / empty -> ddgs fallback
                # Space out successive ddgs scrapes (burst avoidance); the first
                # ddgs call of the search doesn't wait.
                if ddgs_called and hi > 0:
                    await asyncio.sleep(random.uniform(lo, hi))
                ddgs_called = True
                try:
                    results, dd_err = await asyncio.to_thread(_search_sync, b)
                except Exception as e:
                    results, dd_err = [], f'{e}'
                if results:
                    sources.add('ddgs')
                err = None if results else (dd_err or err)

            searched.append((results, err))

        merged: List[Dict[str, str]] = []
        seen = set()
        errors: List[str] = []
        for results, err in searched:
            if err:
                errors.append(err)
            for r in results:
                if r['url'] in seen:
                    continue
                seen.add(r['url'])
                merged.append(r)

        # Enforce the domain contract. The site:/-site: operators bias the engines
        # toward the right pages (so we don't filter a generic set down to zero),
        # but not every backend honors them — verified leaks of ~3/8. This filter
        # is the precision guarantee on top of that recall bias.
        if allowed:
            merged = [r for r in merged if _domain_match(_host(r['url']), allowed)]
        if blocked:
            merged = [r for r in merged if not _domain_match(_host(r['url']), blocked)]

        if not merged:
            if errors and len(errors) == len(built):
                msg = (
                    f'Web search is temporarily unavailable — all backends failed '
                    f'({"; ".join(sorted(set(errors)))}). This is usually transient '
                    f'rate-limiting; try again in a moment.'
                )
            else:
                msg = f'No search results found for "{query}".'
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=msg,
                data={'query': query, 'content': []},
            )
            return

        merged = merged[:_MAX_MERGED]

        # 3. top-K page fetch -> short extract (recovers page-content reading).
        top = merged[:_FETCH_TOP_K]
        extracts = await asyncio.gather(*[_fetch_extract(r['url'], signal) for r in top])
        for r, (ex, err) in zip(top, extracts):
            if ex:
                r['extract'] = ex
            elif err:
                r['extract_error'] = err

        # Cache the completed, non-empty result set so the next near-identical
        # query is served without touching the network (_cache_put never stores an
        # empty set, so a transient block isn't pinned as "no results").
        _cache_put(cache_key, merged)

        yield ToolResultSuccess(
            type='result',
            result_for_assistant=_format_results(query, merged, queries),
            data={'query': query, 'source': sorted(sources),
                  'content': [{'title': r['title'], 'url': r['url']} for r in merged]},
        )
