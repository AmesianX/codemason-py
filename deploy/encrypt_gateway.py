#!/usr/bin/env python3
"""Decrypt gateway that sits in front of a stock vLLM (or any OpenAI-compatible)
server, so codemason can talk to it with end-to-end encrypted request/response
BODIES. vLLM itself is never modified — run it on localhost, run this in front.

    codemason ──(AES-GCM body)──► [this gateway] ──(plaintext, localhost)──► vLLM
              ◄─(AES-GCM chunks)──┘                ◄──(plaintext SSE)────────┘

Only the request/response BODY is encrypted, with a pre-shared symmetric key that
never crosses the network (it is the AES key, not a bearer token). That defeats a
MITM even when the environment can present a trusted-CA certificate (e.g. a stolen
internal root CA doing TLS interception): the interceptor sees only ciphertext and
holds no key. Wire format mirrors codemason/utils/body_crypto.py exactly:

    base64( nonce[12] || AES-256-GCM ciphertext || tag[16] )

Run (same box as vLLM):
    pip install fastapi uvicorn httpx cryptography
    export LLAMACPP_AES_KEY=<same 32-byte base64/hex key as the client>
    export VLLM_UPSTREAM=http://127.0.0.1:8000      # stock vLLM, localhost only
    uvicorn encrypt_gateway:app --host 0.0.0.0 --port 8443

Then point codemason at it:
    LLAMACPP_BASE_URL=http://<gateway-host>:8443
    LLAMACPP_AES_KEY=<same key>
(HTTP is fine here — the body is already encrypted — but TLS on top is welcome.)
"""

import base64
import os

import httpx
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

NONCE_LEN = 12
UPSTREAM = os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
_HTTP_TIMEOUT = float(os.environ.get("GATEWAY_HTTP_TIMEOUT", "1800"))


def _load_key() -> bytes:
    s = (os.environ.get("LLAMACPP_AES_KEY") or "").strip()
    if not s:
        raise SystemExit("LLAMACPP_AES_KEY is required (32-byte base64 or hex)")
    if len(s) == 64 and all(c in "0123456789abcdefABCDEF" for c in s):
        key = bytes.fromhex(s)
    else:
        key = base64.b64decode(s + "=" * (-len(s) % 4))
    if len(key) != 32:
        raise SystemExit(f"LLAMACPP_AES_KEY must decode to 32 bytes, got {len(key)}")
    return key


KEY = _load_key()


def encrypt(plaintext) -> str:
    if isinstance(plaintext, str):
        plaintext = plaintext.encode("utf-8")
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(KEY).encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt(blob) -> bytes:
    if isinstance(blob, str):
        blob = blob.encode("ascii")
    raw = base64.b64decode(blob)
    if len(raw) < NONCE_LEN + 16:
        raise InvalidTag()
    return AESGCM(KEY).decrypt(raw[:NONCE_LEN], raw[NONCE_LEN:], None)


app = FastAPI()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    body = await request.body()
    encrypted = request.headers.get("X-Body-Encrypted", "").lower() == "aesgcm"

    # Forward non-encrypted requests untouched (e.g. GET /v1/models, /slots probes).
    if not encrypted:
        return await _passthrough(request, path, body)

    # Encrypted POST: decrypt body -> forward plaintext JSON to vLLM.
    try:
        plaintext = decrypt(body)
    except (InvalidTag, ValueError, Exception):  # noqa: BLE001
        return Response(status_code=400, content=b"bad ciphertext or wrong key")

    upstream_headers = {"Content-Type": "application/json"}
    auth = request.headers.get("Authorization")
    if auth:
        upstream_headers["Authorization"] = auth

    is_stream = b'"stream"' in plaintext and b'"stream": false' not in plaintext \
        and b'"stream":false' not in plaintext

    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    if is_stream:
        return StreamingResponse(
            _encrypt_stream(url, upstream_headers, plaintext),
            media_type="text/event-stream",
        )

    # Non-streaming: encrypt the whole response body.
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        r = await client.post(url, content=plaintext, headers=upstream_headers)
        if r.status_code != 200:
            return Response(status_code=r.status_code, content=r.content)
        return Response(content=encrypt(r.content).encode("ascii"),
                        media_type="application/octet-stream")


async def _encrypt_stream(url, headers, plaintext):
    """Stream vLLM's SSE, re-emitting each event with its body AES-GCM sealed."""
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        async with client.stream("POST", url, content=plaintext, headers=headers) as r:
            if r.status_code != 200:
                err = await r.aread()
                # errors go back in the clear so the client can surface the status
                yield f"data: {encrypt(err.decode('utf-8', 'replace'))}\n\n".encode()
                return
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    yield b"data: [DONE]\n\n"
                else:
                    yield f"data: {encrypt(payload)}\n\n".encode()


async def _passthrough(request: Request, path: str, body: bytes):
    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    fwd = {k: v for k, v in request.headers.items()
           if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        r = await client.request(request.method, url, content=body or None, headers=fwd)
    resp_headers = {k: v for k, v in r.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding", "connection")}
    return Response(content=r.content, status_code=r.status_code, headers=resp_headers)
