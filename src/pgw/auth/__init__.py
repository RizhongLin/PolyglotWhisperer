"""Authentication primitives.

- ``passwords``: argon2id hash/verify
- ``sessions``: opaque-token DB-backed sessions
- ``csrf``: signed double-submit cookie
- ``deps``: FastAPI dependencies (``current_user``, ``verify_csrf``)
- ``bootstrap``: env-var fast path for first admin
"""

from __future__ import annotations
