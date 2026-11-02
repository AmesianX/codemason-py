"""
WebFetch Tool for CodeMason
Based on src/tools/WebFetchTool/ from Claude Code, adapted to the local substrate.

Flow (WebFetchTool.ts): fetch URL -> HTML->markdown -> run the caller's prompt on
the content via a small model -> return the model's response. The DESCRIPTION and
makeSecondaryModelPrompt are ported verbatim. The "small fast model" is just the
served model — there is no separate small one, and the backend it runs on is
whatever `backend` selects. The SDK rule-based permission engine is
substrate-specific and simplified to codemason's needs_permissions approval.

Cross-host redirect detection (utils.ts isPermittedRedirect / getWithPermittedRedirects),
the preapproved-host list (preapproved.ts), the 15-minute URL cache (utils.ts URL_CACHE),
and validateURL / content-length caps are ported faithfully. The substrate swap is
turndown -> markdownify; the domain-blocklist preflight (api.anthropic.com) is dropped
as it has no local equivalent.
"""

import html as _html
import re
import time
from collections import OrderedDict
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse, urlunparse

from pydantic import Field

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
from ..utils.formatting import Colors, colorize

WEB_FETCH_TOOL_NAME = 'WebFetch'

# ---------------------------------------------------------------------------
# Preapproved hosts — faithful port of preapproved.ts.
#
# WHAT THIS LIST ACTUALLY DOES HERE, because the upstream wording below invites
# the wrong reading and it was written into a commit message once: it is NOT a
# permission gate. check_permissions asks for EVERY WebFetch regardless of this
# list (and -dsp skips the question for every URL, equally regardless). The only
# thing being on it changes is that markdown under the size cap is returned RAW
# instead of being summarized through a model call. So an entry buys a cheaper,
# more faithful read of a site — not access to one.
#
# Upstream's rationale, kept for provenance: "For legal and security concerns,
# WebFetch typically only accesses domains the user has provided. We make an
# exception for a list of preapproved code-related domains." That held where the
# list did gate access; in this port it does not.
#
# Some entries are path-scoped (e.g. "vercel.com/docs"); those are matched with
# path-segment boundaries so "/docs" does not match "/docs-evil/malware".
# ---------------------------------------------------------------------------
PREAPPROVED_HOSTS: Set[str] = {
    # Open specs this program implements. Another product's own documentation
    # and repositories used to sit here too; they were removed on identity
    # grounds — this is not that product — NOT as a security measure. This list
    # is not a permission gate: WebFetch asks for every URL regardless of it,
    # and being on it only means markdown under the cap is returned RAW instead
    # of summarized through a model call.
    'modelcontextprotocol.io',
    'agentskills.io',

    # Top Programming Languages
    'docs.python.org',
    'en.cppreference.com',
    'docs.oracle.com',
    'learn.microsoft.com',
    'developer.mozilla.org',
    'go.dev',
    'pkg.go.dev',
    'www.php.net',
    'docs.swift.org',
    'kotlinlang.org',
    'ruby-doc.org',
    'doc.rust-lang.org',
    'www.typescriptlang.org',

    # Web & JavaScript Frameworks/Libraries
    'react.dev',
    'angular.io',
    'vuejs.org',
    'nextjs.org',
    'expressjs.com',
    'nodejs.org',
    'bun.sh',
    'jquery.com',
    'getbootstrap.com',
    'tailwindcss.com',
    'd3js.org',
    'threejs.org',
    'redux.js.org',
    'webpack.js.org',
    'jestjs.io',
    'reactrouter.com',

    # Python Frameworks & Libraries
    'docs.djangoproject.com',
    'flask.palletsprojects.com',
    'fastapi.tiangolo.com',
    'pandas.pydata.org',
    'numpy.org',
    'www.tensorflow.org',
    'pytorch.org',
    'scikit-learn.org',
    'matplotlib.org',
    'requests.readthedocs.io',
    'jupyter.org',

    # PHP Frameworks
    'laravel.com',
    'symfony.com',
    'wordpress.org',

    # Java Frameworks & Libraries
    'docs.spring.io',
    'hibernate.org',
    'tomcat.apache.org',
    'gradle.org',
    'maven.apache.org',

    # .NET & C# Frameworks
    'asp.net',
    'dotnet.microsoft.com',
    'nuget.org',
    'blazor.net',

    # Mobile Development
    'reactnative.dev',
    'docs.flutter.dev',
    'developer.apple.com',
    'developer.android.com',

    # Data Science & Machine Learning
    'keras.io',
    'spark.apache.org',
    'huggingface.co',
    'www.kaggle.com',

    # Databases
    'www.mongodb.com',
    'redis.io',
    'www.postgresql.org',
    'dev.mysql.com',
    'www.sqlite.org',
    'graphql.org',
    'prisma.io',

    # Cloud & DevOps
    'docs.aws.amazon.com',
    'cloud.google.com',
    'kubernetes.io',
    'www.docker.com',
    'www.terraform.io',
    'www.ansible.com',
    'vercel.com/docs',
    'docs.netlify.com',
    'devcenter.heroku.com',

    # Testing & Monitoring
    'cypress.io',
    'selenium.dev',

    # Game Development
    'docs.unity.com',
    'docs.unrealengine.com',

    # Other Essential Tools
    'git-scm.com',
    'nginx.org',
    'httpd.apache.org',
}


def _split_preapproved() -> Tuple[Set[str], Dict[str, List[str]]]:
    """Split entries into hostname-only (O(1) set membership) and path-scoped
    (host -> list of path prefixes), mirroring preapproved.ts's module-load split."""
    hosts: Set[str] = set()
    paths: Dict[str, List[str]] = {}
    for entry in PREAPPROVED_HOSTS:
        slash = entry.find('/')
        if slash == -1:
            hosts.add(entry)
        else:
            host = entry[:slash]
            path = entry[slash:]
            paths.setdefault(host, []).append(path)
    return hosts, paths


_HOSTNAME_ONLY, _PATH_PREFIXES = _split_preapproved()


def is_preapproved_host(hostname: str, pathname: str) -> bool:
    """Port of preapproved.ts isPreapprovedHost."""
    if hostname in _HOSTNAME_ONLY:
        return True
    prefixes = _PATH_PREFIXES.get(hostname)
    if prefixes:
        for p in prefixes:
            # Enforce path segment boundaries: "/docs" must not match
            # "/docs-evil/malware". Only exact match or "/" after the
            # prefix is allowed.
            if pathname == p or pathname.startswith(p + '/'):
                return True
    return False

DESCRIPTION = """
- Fetches content from a specified URL and processes it using an AI model
- Takes a URL and a prompt as input
- Fetches the URL content, converts HTML to markdown
- Processes the content with the prompt using a small, fast model
- Returns the model's response about the content
- Use this tool when you need to retrieve and analyze web content

Usage notes:
  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions.
  - The URL must be a fully-formed valid URL
  - HTTP URLs will be automatically upgraded to HTTPS
  - The prompt should describe what information you want to extract from the page
  - This tool is read-only and does not modify any files
  - Results may be summarized if the content is very large
  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL
  - When a URL redirects to a different host, the tool will inform you and provide the redirect URL in a special format. You should then make a new WebFetch request with the redirect URL to fetch the content.
  - For GitHub URLs, prefer using the gh CLI via Bash instead (e.g., gh pr view, gh issue view, gh api).
"""

# Truncate to not spend too many tokens (utils.ts MAX_MARKDOWN_LENGTH).
MAX_MARKDOWN_LENGTH = 100_000

# validateURL caps (utils.ts).
MAX_URL_LENGTH = 2000

# Cap the HTTP read at 10MB (utils.ts MAX_HTTP_CONTENT_LENGTH).
MAX_HTTP_CONTENT_LENGTH = 10 * 1024 * 1024

# Cap same-host redirect hops (utils.ts MAX_REDIRECTS).
MAX_REDIRECTS = 10

# utils.ts URL_CACHE: LRUCache with a 15-minute TTL and a 50MB byte-size cap
# (CACHE_TTL_MS / MAX_CACHE_SIZE_BYTES). Keyed by the ORIGINAL url (not the
# upgraded/redirected one). Ported as an OrderedDict with byte accounting +
# LRU eviction rather than the previous entry-count (>64) heuristic, which let
# a few large pages bloat memory unboundedly between evictions.
_CACHE_TTL_S = 15 * 60
_MAX_CACHE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

_URL_CACHE: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
_cache_bytes = 0


def _entry_size(value: Dict[str, Any]) -> int:
    """sizeCalculation: byte length of the cached markdown content."""
    c = value.get('content') if isinstance(value, dict) else None
    return len(c.encode('utf-8')) if isinstance(c, str) else 1


def _cache_evict(url: str) -> None:
    global _cache_bytes
    entry = _URL_CACHE.pop(url, None)
    if entry is not None:
        _cache_bytes -= _entry_size(entry[1])


def _cache_get(url: str) -> Optional[Dict[str, Any]]:
    entry = _URL_CACHE.get(url)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > _CACHE_TTL_S:
        _cache_evict(url)
        return None
    _URL_CACHE.move_to_end(url)  # LRU touch
    return value


def _cache_set(url: str, value: Dict[str, Any]) -> None:
    global _cache_bytes
    now = time.time()
    if url in _URL_CACHE:
        _cache_evict(url)
    # Drop expired entries first (TTL), then enforce the byte cap by evicting
    # least-recently-used entries until under MAX_CACHE_SIZE_BYTES.
    for k in [k for k, (ts, _) in list(_URL_CACHE.items()) if now - ts > _CACHE_TTL_S]:
        _cache_evict(k)
    _URL_CACHE[url] = (now, value)
    _cache_bytes += _entry_size(value)
    while _cache_bytes > _MAX_CACHE_SIZE_BYTES and len(_URL_CACHE) > 1:
        oldest = next(iter(_URL_CACHE))
        if oldest == url:
            break
        _cache_evict(oldest)


def make_secondary_model_prompt(
    markdown_content: str, prompt: str, is_preapproved_domain: bool = False
) -> str:
    """Port of makeSecondaryModelPrompt (prompt.ts:23-46)."""
    if is_preapproved_domain:
        guidelines = (
            "Provide a concise response based on the content above. "
            "Include relevant details, code examples, and documentation excerpts as needed."
        )
    else:
        guidelines = (
            "Provide a concise response based only on the content above. In your response:\n"
            " - Enforce a strict 125-character maximum for quotes from any source document. "
            "Open Source Software is ok as long as we respect the license.\n"
            " - Use quotation marks for exact language from articles; any language outside of "
            "the quotation should never be word-for-word the same.\n"
            " - You are not a lawyer and never comment on the legality of your own prompts and responses.\n"
            " - Never produce or reproduce exact song lyrics."
        )
    return f"\nWeb page content:\n---\n{markdown_content}\n---\n\n{prompt}\n\n{guidelines}\n"


def is_preapproved_url(url: str) -> bool:
    """Port of utils.ts isPreapprovedUrl."""
    try:
        parsed = urlparse(url)
        if not parsed.hostname:
            return False
        return is_preapproved_host(parsed.hostname, parsed.path or '/')
    except Exception:
        return False


def _is_internal_host(hostname: str) -> bool:
    """True if the host is loopback/private/link-local/reserved or an internal
    name. Original relied on per-domain USER APPROVAL as the SSRF boundary; this
    port auto-allows read-only tools (console adaptation), so that boundary is
    gone — a hard internal-address guard replaces it so the model can't reach
    127.0.0.1 / 192.168.x / 169.254.169.254 (cloud metadata) / etc."""
    import ipaddress
    import socket

    h = (hostname or '').strip('[]').lower()
    if not h:
        return True
    if h == 'localhost' or h.endswith(('.localhost', '.internal', '.local')):
        return True

    # NAT64 (RFC 6052 well-known prefix). On an IPv6-only access network with
    # 464XLAT — e.g. a KT mobile/CPE uplink — the resolver hands back
    # 64:ff9b::<v4> for ordinary public sites. Python classifies that whole
    # prefix as is_reserved, so github.com and twrp.me resolved to "internal"
    # and every fetch was rejected as an SSRF attempt. Unwrap to the embedded
    # IPv4 and judge THAT: a NAT64-mapped 192.168.x is still blocked, a
    # NAT64-mapped public address is not.
    _NAT64_WKP = ipaddress.ip_network('64:ff9b::/96')

    def _blocked(ip_str: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_str.split('%')[0])
        except ValueError:
            return False
        if ip.version == 6:
            if ip in _NAT64_WKP:
                ip = ipaddress.ip_address(int(ip) & 0xFFFFFFFF)
            elif ip.ipv4_mapped is not None:      # ::ffff:a.b.c.d
                ip = ip.ipv4_mapped
        return (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified)

    # Literal IP host.
    try:
        ipaddress.ip_address(h)
        return _blocked(h)
    except ValueError:
        pass

    # Resolve the name; block if ANY resolved address is internal. (DNS is only
    # queried once here; DNS-rebinding TOCTOU between this and the fetch is out
    # of scope for a local tool.)
    try:
        infos = socket.getaddrinfo(h, None)
    except Exception:
        return False  # unresolvable → the fetch will fail on its own
    return any(_blocked(info[4][0]) for info in infos)


def validate_url(url: str) -> bool:
    """Port of utils.ts validateURL, plus an internal-address SSRF guard (the
    original's per-domain approval gate is not reproduced in this console port)."""
    if len(url) > MAX_URL_LENGTH:
        return False
    try:
        parsed = urlparse(url if '://' in url else 'https://' + url)
    except Exception:
        return False

    # Only http(s) — the fetch upgrades http->https; block file:/ftp:/gopher: etc.
    if parsed.scheme not in ('http', 'https'):
        return False

    # Block URLs carrying username/password (userinfo) credentials.
    if parsed.username or parsed.password:
        return False

    # Require a publicly-resolvable, dotted hostname (>= 2 labels).
    hostname = parsed.hostname or ''
    if len(hostname.split('.')) < 2:
        return False

    # SSRF guard: reject loopback/private/link-local/reserved/metadata targets.
    if _is_internal_host(hostname):
        return False

    return True


def is_permitted_redirect(original_url: str, redirect_url: str) -> bool:
    """Port of utils.ts isPermittedRedirect. Allows only same-origin redirects
    and www-toggle redirects (add/remove leading 'www.')."""
    try:
        po = urlparse(original_url)
        pr = urlparse(redirect_url)

        if pr.scheme != po.scheme:
            return False
        if pr.port != po.port:
            return False
        if pr.username or pr.password:
            return False

        def strip_www(host: str) -> str:
            return re.sub(r'^www\.', '', host or '')

        return strip_www(po.hostname or '') == strip_www(pr.hostname or '')
    except Exception:
        return False


_CHARSET_TOKEN_RE = re.compile(r'[^A-Za-z0-9_-]')


def _charset_from_content_type(content_type: str) -> Optional[str]:
    """`charset=` out of a Content-Type header value, lowercased."""
    lower = (content_type or '').lower()
    idx = lower.find('charset=')
    if idx < 0:
        return None
    rest = lower[idx + len('charset='):].lstrip('"\'')
    m = _CHARSET_TOKEN_RE.search(rest)
    cs = rest[:m.start()] if m else rest
    return cs or None


def declared_charset(raw: bytes, content_type: str) -> Optional[str]:
    """The charset a response declares, lowercased, or None.

    Header first, then a `<meta>` in the document head — the order browsers use,
    and the order that matters when the two disagree. Only the head is scanned:
    a charset declaration is required to appear early, and a later "charset="
    inside body text is prose, not a declaration.
    """
    cs = _charset_from_content_type(content_type)
    if cs:
        return cs
    # ASCII-compatible enough to find the declaration in any of these encodings;
    # that is exactly why the standard requires it inside the first bytes.
    head = raw[:2048].decode('utf-8', errors='replace').lower()
    idx = head.find('charset')
    if idx < 0:
        return None
    rest = head[idx + len('charset'):].lstrip()
    if not rest.startswith('='):
        return None
    rest = rest[1:].lstrip().lstrip('"\'')
    m = _CHARSET_TOKEN_RE.search(rest)
    cs = rest[:m.start()] if m else rest
    return cs or None


def decode_body(raw: bytes, content_type: str) -> str:
    """Decode a fetched body using the charset it declares.

    Everything used to go through a lossy UTF-8 decode, so a page served as
    EUC-KR, Shift_JIS or GBK reached the model as a wall of replacement
    characters — it could not answer about the page, and nothing said why. The
    declaration is authoritative here (unlike a local file, which declares
    nothing and would need guessing), so this is a lookup, not a heuristic.

    An unrecognised or absent charset keeps the previous behaviour exactly:
    lossy UTF-8, which is also what the standard defaults to.
    """
    label = declared_charset(raw, content_type)
    if label:
        try:
            return raw.decode(label, errors='replace')
        except (LookupError, ValueError):
            pass
    return raw.decode('utf-8', errors='replace')


def _html_to_markdown(content: str, content_type: str) -> str:
    """HTML -> markdown via markdownify, the Python counterpart of the original's
    `turndown` (utils.ts getTurndownService). Non-HTML content is returned as-is.
    Falls back to a minimal tag-strip only if markdownify is unavailable."""
    if 'html' not in content_type.lower():
        return content
    try:
        from markdownify import markdownify
        md = markdownify(content)
    except Exception:
        md = re.sub(r'(?is)<(script|style|head|noscript)[^>]*>.*?</\1>', ' ', content)
        md = re.sub(r'<[^>]+>', '', md)
        md = _html.unescape(md)
    md = re.sub(r'\n\s*\n\s*\n+', '\n\n', md)
    return md.strip()


async def _get_with_permitted_redirects(
    url: str, signal: Any, depth: int = 0
) -> Union[Dict[str, Any], 'RedirectInfo']:
    """Port of utils.ts getWithPermittedRedirects. Does NOT blindly follow
    redirects: httpx follow_redirects=False; same-host / www-toggle redirects are
    followed manually (capped at MAX_REDIRECTS hops). A cross-host redirect is
    surfaced to the caller as a RedirectInfo."""
    import httpx

    if depth > MAX_REDIRECTS:
        raise RuntimeError(f'Too many redirects (exceeded {MAX_REDIRECTS})')

    # SSRF guard on EVERY hop, not just the initial URL. A permitted www-toggle
    # redirect (www.foo.com -> foo.com) could resolve to an internal/private IP;
    # re-validate the host before each fetch so a redirect can't reach a target
    # the initial validation would have rejected.
    from urllib.parse import urlparse as _urlparse
    _host = _urlparse(url).hostname or ''
    if _is_internal_host(_host):
        raise RuntimeError(f'Blocked fetch to internal host: {_host}')

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
        async with client.stream(
            'GET',
            url,
            headers={
                'Accept': 'text/markdown, text/html, */*',
                'User-Agent': 'CodeMason/0.1 (+local)',
            },
        ) as resp:
            if resp.status_code in (301, 302, 307, 308):
                location = resp.headers.get('location')
                if not location:
                    raise RuntimeError('Redirect missing Location header')
                redirect_url = urljoin(url, location)
                if is_permitted_redirect(url, redirect_url):
                    return await _get_with_permitted_redirects(
                        redirect_url, signal, depth + 1
                    )
                return RedirectInfo(
                    original_url=url,
                    redirect_url=redirect_url,
                    status_code=resp.status_code,
                )

            content_type = resp.headers.get('content-type', '')
            # Cap the read at MAX_HTTP_CONTENT_LENGTH.
            chunks = bytearray()
            async for chunk in resp.aiter_bytes():
                # Honor the user's abort mid-download — a large/slow fetch was
                # otherwise uncancellable (only the later model call checked it).
                if signal is not None and getattr(signal, 'is_set', lambda: False)():
                    raise RuntimeError('Fetch aborted')
                chunks.extend(chunk)
                if len(chunks) > MAX_HTTP_CONTENT_LENGTH:
                    raise RuntimeError(
                        f'Response exceeds maximum content length ({MAX_HTTP_CONTENT_LENGTH} bytes)'
                    )
            raw_bytes = bytes(chunks)

    return {
        'raw_bytes': raw_bytes,
        'content_type': content_type,
        'code': resp.status_code,
        'code_text': resp.reason_phrase or '',
    }


class RedirectInfo:
    """Mirror of utils.ts RedirectInfo."""

    def __init__(self, original_url: str, redirect_url: str, status_code: int):
        self.original_url = original_url
        self.redirect_url = redirect_url
        self.status_code = status_code


async def get_url_markdown_content(
    url: str, signal: Any
) -> Union[Dict[str, Any], RedirectInfo]:
    """Port of utils.ts getURLMarkdownContent. Returns either a FetchedContent
    dict or a RedirectInfo. Uses the 15-minute URL cache keyed by the original URL."""
    if not validate_url(url):
        raise RuntimeError('Invalid URL')

    cached = _cache_get(url)
    if cached is not None:
        return cached

    # HTTP -> HTTPS upgrade (the cache key stays the original URL).
    upgraded = url
    parsed = urlparse(url if '://' in url else 'https://' + url)
    if parsed.scheme == 'http':
        upgraded = urlunparse(parsed._replace(scheme='https'))
    elif '://' not in url:
        upgraded = urlunparse(parsed)

    # Say which host is being waited on. A slow one gets the full 60s the
    # client allows, and until now that was 60 seconds of unexplained spinner.
    from ..utils.tokens import set_wait_note
    set_wait_note('fetching %s' % (urlparse(upgraded).hostname or upgraded))
    try:
        response = await _get_with_permitted_redirects(upgraded, signal)
    finally:
        set_wait_note(None)
    if isinstance(response, RedirectInfo):
        return response

    raw_bytes = response['raw_bytes']
    content_type = response['content_type']
    html_content = decode_body(raw_bytes, content_type)
    markdown_content = _html_to_markdown(html_content, content_type)

    entry = {
        'content': markdown_content,
        'bytes': len(raw_bytes),
        'code': response['code'],
        'code_text': response['code_text'],
        'content_type': content_type,
    }
    _cache_set(url, entry)
    return entry


class WebFetchInput(ToolInputSchema):
    url: str = Field(..., description="The URL to fetch content from")
    prompt: str = Field(..., description="The prompt to run on the fetched content")


class WebFetchTool(Tool):
    """Fetch a URL and process its content with the model."""

    name = WEB_FETCH_TOOL_NAME
    input_schema = WebFetchInput
    DESCRIPTION = DESCRIPTION

    @classmethod
    async def description(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    async def prompt(cls, **kwargs) -> str:
        return DESCRIPTION

    @classmethod
    def is_read_only(cls) -> bool:
        return True

    @classmethod
    async def check_permissions(cls, input_data, context=None) -> PermissionResult:
        """Ask, despite being read-only.

        Nothing changes locally, but the URL is model-controlled and the request
        leaves this machine — so a Read of any file (Read has no path
        restriction) can be posted to an arbitrary host on the next call, with
        neither step prompting. The SSRF guard in this module is thorough about
        reaching INWARD and says nothing about egress, because the upstream
        boundary for that was per-domain user approval.

        Under dangerously-skip-permissions this is never reached:
        has_permissions_to_use_tool returns before the read-only branch, so
        skipping permissions still skips this.
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
        return 'WebFetch'

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any) -> ValidationResult:
        url = input_data.get('url', '')
        try:
            parsed = urlparse(url if '://' in url else 'https://' + url)
            if not parsed.netloc:
                raise ValueError('no host')
        except Exception:
            return ValidationResultFailure(
                result=False,
                message=f'Error: Invalid URL "{url}". The URL provided could not be parsed.',
            )
        return ValidationResultSuccess()

    @classmethod
    def get_tool_use_summary(cls, input_data):
        """The url, truncated for display (UI.tsx getToolUseSummary)."""
        from ..utils.formatting import TOOL_SUMMARY_MAX_LENGTH, truncate
        if not isinstance(input_data, dict):
            return None
        value = input_data.get('url')
        if not value:
            return None
        return truncate(str(value), TOOL_SUMMARY_MAX_LENGTH)

    @classmethod
    def get_activity_description(cls, input_data):
        """"Fetching <url>" (getActivityDescription)."""
        summary = cls.get_tool_use_summary(input_data)
        return f'Fetching {summary}' if summary else 'Fetching web page'

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        """url, then what we asked of the page (WebFetchTool/UI.tsx: "url: prompt")."""
        if not isinstance(input_data, dict):
            return ''
        url = input_data.get('url', '')
        if not url:
            return ''
        prompt = ' '.join(str(input_data.get('prompt', '') or '').split())
        if prompt and not verbose and len(prompt) > 100:
            prompt = prompt[:100] + '…'
        lines = [colorize(str(url), Colors.DIM)]
        if prompt:
            lines.append(colorize(prompt, Colors.DIM))
        return '\n'.join(lines)

    @classmethod
    async def call(
        cls,
        input_data: Dict[str, Any],
        context: Any,
        can_use_tool: Optional[Callable] = None,
    ) -> AsyncGenerator[ToolResult, None]:
        from ..services.backend import query_model
        from ..utils.messages import create_user_message

        url = input_data.get('url', '')
        user_prompt = input_data.get('prompt', '')
        start = time.time()
        signal = getattr(context, 'abort_controller', None)

        try:
            response = await get_url_markdown_content(url, signal)
        except Exception as e:
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=f"Error fetching {url}: {e}",
                data={'error': str(e), 'url': url},
            )
            return

        # Cross-host redirect: do not follow. Tell the model to re-issue WebFetch
        # with the redirect URL (WebFetchTool.ts:217-249).
        if isinstance(response, RedirectInfo):
            status_text = {
                301: 'Moved Permanently',
                308: 'Permanent Redirect',
                307: 'Temporary Redirect',
            }.get(response.status_code, 'Found')
            message = (
                f"REDIRECT DETECTED: The URL redirects to a different host.\n\n"
                f"Original URL: {response.original_url}\n"
                f"Redirect URL: {response.redirect_url}\n"
                f"Status: {response.status_code} {status_text}\n\n"
                f"To complete your request, I need to fetch content from the redirected URL. "
                f"Please use WebFetch again with these parameters:\n"
                f'- url: "{response.redirect_url}"\n'
                f'- prompt: "{user_prompt}"'
            )
            yield ToolResultSuccess(
                type='result',
                result_for_assistant=message,
                data={
                    'url': url,
                    'code': response.status_code,
                    'code_text': status_text,
                    'duration_ms': int((time.time() - start) * 1000),
                },
            )
            return

        content = response['content']
        content_type = response['content_type']
        is_preapproved = is_preapproved_url(url)

        # Preapproved + text/markdown + under cap: return raw content, no model
        # call (WebFetchTool.ts:261-278).
        if (
            is_preapproved
            and 'text/markdown' in content_type
            and len(content) < MAX_MARKDOWN_LENGTH
        ):
            result_text = content
        else:
            # Truncate content to avoid "Prompt is too long" errors (utils.ts
            # applyPromptToMarkdown).
            if len(content) > MAX_MARKDOWN_LENGTH:
                content = content[:MAX_MARKDOWN_LENGTH] + '\n\n[Content truncated due to length...]'
            try:
                model_response = await query_model(
                    messages=[create_user_message(
                        make_secondary_model_prompt(content, user_prompt, is_preapproved)
                    )],
                    system_prompt=["You are a helpful assistant that analyzes web page content and answers questions about it."],
                    tools=[],
                    signal=signal,
                    options={'dangerously_skip_permissions': False},
                )
                content_blocks = model_response.message.content
                result_text = ''
                if isinstance(content_blocks, list):
                    for b in content_blocks:
                        t = b.get('text') if isinstance(b, dict) else getattr(b, 'text', None)
                        if t:
                            result_text += t
                result_text = result_text or 'No response from model'
            except Exception as e:
                result_text = f"Fetched {len(content)} chars but failed to process: {e}"

        yield ToolResultSuccess(
            type='result',
            result_for_assistant=result_text,
            data={
                'url': url,
                'bytes': response['bytes'],
                'code': response['code'],
                'code_text': response['code_text'],
                'duration_ms': int((time.time() - start) * 1000),
            },
        )
