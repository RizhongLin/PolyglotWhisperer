"""Argon2id password hashing.

Library: argon2-cffi. Defaults are the OWASP-recommended values for
2024+: time_cost=2, memory_cost=64 MiB, parallelism=1.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError, VerifyMismatchError

_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=64 * 1024,
    parallelism=1,
    hash_len=32,
    salt_len=16,
)


def hash_password(plaintext: str) -> str:
    """Return an argon2id encoded hash for ``plaintext``."""
    if not plaintext:
        raise ValueError("password must be non-empty")
    return _hasher.hash(plaintext)


def verify_password(plaintext: str, hashed: str) -> bool:
    """Constant-time compare; returns True on match, False otherwise."""
    try:
        _hasher.verify(hashed, plaintext)
        return True
    except (VerifyMismatchError, VerificationError, InvalidHash):
        return False


def needs_rehash(hashed: str) -> bool:
    """True if the stored hash uses parameters weaker than current."""
    try:
        return _hasher.check_needs_rehash(hashed)
    except InvalidHash:
        return True
