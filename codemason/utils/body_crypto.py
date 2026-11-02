"""Symmetric body encryption for the codemason ↔ vLLM channel.

Threat model (small self-hosted team, ≤3 users): defend against a MITM that can
present a certificate the client's trust store accepts — e.g. a stolen/abused
internal root CA doing TLS interception. Plain TLS and cert pinning both anchor
on the CA trust chain, so they can't. The defense that survives a compromised CA
is to never trust the transport for confidentiality: a pre-shared symmetric key
(the same shape as the existing API key) is used ONLY to AEAD-encrypt request and
response BODIES, and is never transmitted. An interceptor that fully terminates
TLS still sees only ciphertext, and possesses no key to forge or read it.

Wire format (one self-contained message, base64 of):

    nonce (12 bytes) || AES-256-GCM ciphertext || tag (16 bytes)

AES-256-GCM is authenticated: a wrong key or any tampering fails decryption
(the GCM tag), so possession of the key doubles as authentication — there is no
separate token to send. A fresh random nonce per message is the only invariant
GCM requires; it is prepended, not secret.

Streaming: each SSE event is an independent message (its own nonce), so the
gateway can encrypt token chunks as they arrive and the client decrypts each in
place. The same functions serve the client and the server-side gateway.
"""

import base64
import os
from typing import Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

__all__ = [
    "NONCE_LEN",
    "load_key",
    "encrypt",
    "decrypt",
    "BodyCryptoError",
]

NONCE_LEN = 12  # 96-bit nonce, the AES-GCM standard/recommended size


class BodyCryptoError(Exception):
    """Key is malformed, or a message failed to decrypt/authenticate."""


def load_key(raw: str) -> bytes:
    """Parse a configured key string into 32 raw bytes (AES-256).

    Accepts base64 (with or without padding) or hex — whichever the user pasted.
    Must decode to exactly 32 bytes; anything else is a configuration error, not
    something to silently pad or truncate.
    """
    s = (raw or "").strip()
    if not s:
        raise BodyCryptoError("empty AES key")
    # Try BOTH hex and base64 and take whichever yields exactly 32 bytes. A valid
    # 32-byte key is 64 hex chars XOR 44 base64 chars — the lengths never overlap,
    # so there is no ambiguity to resolve by guessing (the old length==64 all-hex
    # heuristic mis-classified a base64 string that happened to be 64 hex-only
    # chars). hex is preferred when both somehow decode, but that can't produce
    # two DIFFERENT 32-byte keys.
    candidates = []
    if len(s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in s):
        try:
            candidates.append(bytes.fromhex(s))
        except ValueError:
            pass
    try:
        candidates.append(base64.b64decode(s + "=" * (-len(s) % 4)))
    except Exception:  # noqa: BLE001 - a bad base64 just isn't a candidate
        pass

    for key in candidates:
        if len(key) == 32:
            return key
    if not candidates:
        raise BodyCryptoError("AES key is neither valid base64 nor hex")
    raise BodyCryptoError(
        f"AES key must be 32 bytes (256-bit); got {len(candidates[0])} after decoding"
    )


def encrypt(key: bytes, plaintext: Union[bytes, str]) -> str:
    """Encrypt one message. Returns base64(nonce || ciphertext || tag)."""
    if isinstance(plaintext, str):
        plaintext = plaintext.encode("utf-8")
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(key).encrypt(nonce, plaintext, None)  # ct includes the 16-byte tag
    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt(key: bytes, blob: Union[bytes, str]) -> bytes:
    """Reverse encrypt(). Raises BodyCryptoError on a wrong key or tampering."""
    if isinstance(blob, str):
        blob = blob.encode("ascii")
    try:
        raw = base64.b64decode(blob)
    except Exception as e:  # noqa: BLE001
        raise BodyCryptoError(f"ciphertext is not valid base64: {e}")
    if len(raw) < NONCE_LEN + 16:
        raise BodyCryptoError("ciphertext too short to contain nonce + tag")
    nonce, ct = raw[:NONCE_LEN], raw[NONCE_LEN:]
    try:
        return AESGCM(key).decrypt(nonce, ct, None)
    except InvalidTag:
        raise BodyCryptoError("decryption failed: wrong key or tampered message")
