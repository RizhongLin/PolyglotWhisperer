"""Out-of-band maintenance tasks invoked by the CLI or first-boot.

Currently:
- ``backfill``: import on-disk ``pgw_workspace/`` workspaces + vocab
  JSON into the DB, owned by the admin user. Idempotent.
"""

from __future__ import annotations
