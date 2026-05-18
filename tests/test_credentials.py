"""Tests for credential encryption, endpoints, and DB sync."""

from __future__ import annotations

import json
import os

import pytest

# ── Encryption round-trip ─────────────────────────────────────────────


def test_encrypt_decrypt_round_trip():
    from pgw.crypto.encryption import decrypt, encrypt

    os.environ["PGW_SECRET_KEY"] = "round-trip-test-key"
    plain = "sk-test-api-key-12345"
    cipher = encrypt(plain)
    assert len(cipher) > 40
    assert decrypt(cipher) == plain


def test_encrypt_decrypt_fails_with_wrong_key():
    from pgw.crypto.encryption import decrypt, encrypt

    os.environ["PGW_SECRET_KEY"] = "key-A"
    cipher = encrypt("secret-1")
    os.environ["PGW_SECRET_KEY"] = "key-B"
    with pytest.raises(Exception):
        decrypt(cipher)


def test_encrypt_stable_with_same_key():
    from pgw.crypto.encryption import decrypt, encrypt

    os.environ["PGW_SECRET_KEY"] = "stable-key"
    cipher = encrypt("my-key")
    assert decrypt(cipher) == "my-key"
    cipher2 = encrypt("my-key")
    assert cipher2 != cipher
    assert decrypt(cipher2) == "my-key"


# ── _ok helper ────────────────────────────────────────────────────────


def test_ok_helper():
    from pgw.server.routes.credentials import _ok

    assert _ok() == {"ok": True}
    assert _ok(id=42) == {"ok": True, "id": 42}


# ── JSONText type ─────────────────────────────────────────────────────


class _FakeSQLiteDialect:
    name = "sqlite"


class _FakePostgresDialect:
    name = "postgresql"


def test_jsontext_type_sqlite():
    from pgw.db.types import JSONText

    jt = JSONText()
    dialect = _FakeSQLiteDialect()
    bound = jt.process_bind_param({"key": "value"}, dialect)
    assert isinstance(bound, str)
    assert json.loads(bound) == {"key": "value"}

    result = jt.process_result_value(bound, dialect)
    assert result == {"key": "value"}


def test_jsontext_type_null():
    from pgw.db.types import JSONText

    jt = JSONText()
    dialect = _FakeSQLiteDialect()
    assert jt.process_bind_param(None, dialect) is None
    assert jt.process_result_value(None, dialect) == {}


def test_jsontext_type_invalid_json():
    from pgw.db.types import JSONText

    jt = JSONText()
    dialect = _FakeSQLiteDialect()
    assert jt.process_result_value("not-json", dialect) == {}


# ── _load_user_env_overrides ──────────────────────────────────────────


def test_load_env_overrides_no_user():
    from pgw.server.jobs import _load_user_env_overrides

    assert _load_user_env_overrides(None) == {}
