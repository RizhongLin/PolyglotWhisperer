#!/bin/sh
# Docker entrypoint for pgw.
#
# 1. If PGW_DATABASE_URL is set (Postgres in production), run
#    `pgw maintenance migrate` so the schema is up-to-date before
#    serving. The migrate command is idempotent and safe to run on
#    every container start.
# 2. Forward whatever args were passed (e.g. `serve --no-open`) to
#    the pgw CLI.
#
# Without PGW_DATABASE_URL, pgw falls back to a SQLite file under the
# workspace dir and `_bootstrap_db()`'s create_all handles the schema.

set -e

if [ -n "$PGW_DATABASE_URL" ]; then
    echo "[entrypoint] running migrations against ${PGW_DATABASE_URL%%://*}://…"
    pgw maintenance migrate
fi

exec pgw "$@"
