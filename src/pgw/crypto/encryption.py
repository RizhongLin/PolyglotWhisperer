"""AES-256-GCM encryption for user credentials.

Derives a 256-bit key from ``PGW_SECRET_KEY`` via HKDF-SHA256,
then encrypts/decrypts with AES-GCM (authenticated, nonce-based).
"""

from __future__ import annotations

import os
import secrets
from base64 import urlsafe_b64decode, urlsafe_b64encode
from hashlib import sha256

_DEV_SECRET = secrets.token_urlsafe(32)


def _get_secret() -> bytes:
    key = os.environ.get("PGW_SECRET_KEY") or _DEV_SECRET
    return key.encode("utf-8")


def _derive_key() -> bytes:
    """HKDF-SHA256: derive a 256-bit AES key from PGW_SECRET_KEY."""
    import hmac

    secret = _get_secret()
    master = sha256(secret).digest()
    return hmac.digest(master, b"pgw:credential:encryption:v1", sha256)


def encrypt(plaintext: str) -> str:
    """Encrypt *plaintext* and return a base64-encoded ciphertext.

    Format: ``base64(nonce || ciphertext || tag)`` where nonce is
    12 bytes (AES-GCM standard) and tag is 16 bytes.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError("cryptography is required for credential encryption")

    key = _derive_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return urlsafe_b64encode(nonce + ciphertext).decode("ascii")


def decrypt(encoded: str) -> str:
    """Decrypt a ciphertext produced by :func:`encrypt`."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError("cryptography is required for credential encryption")

    key = _derive_key()
    aesgcm = AESGCM(key)
    raw = urlsafe_b64decode(encoded.encode("ascii"))
    nonce, ciphertext = raw[:12], raw[12:]
    return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
