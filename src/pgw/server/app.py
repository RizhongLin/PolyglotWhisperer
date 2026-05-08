"""FastAPI app factories for ``pgw serve``.

Two ASGI apps:

- :func:`create_library_app` — multi-workspace library + end-to-end
  pipeline launcher. Serves the React SPA (built into
  ``src/pgw/templates/dist`` by ``frontend/``), JSON APIs the SPA
  consumes, the jobs API, and raw workspace files at
  ``/ws/<slug>/<ts>/<file>`` for the player ``<video>`` / ``<track>``.
- :func:`create_workspace_app` — single-workspace mode (``pgw serve
  <path>``). Serves the same SPA but pinned to one workspace.

The SPA does its own client-side routing via TanStack Router; the
backend's only job is to:

  1. expose JSON over /api/...
  2. expose raw workspace blobs over /ws/<slug>/<ts>/...
  3. serve the static SPA bundle for everything else (catch-all so a
     deep-link refresh on /library/foo/bar still hits index.html)
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from importlib.resources import files
from pathlib import Path
from queue import Full
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from starlette.responses import FileResponse

from pgw.auth.deps import current_user_or_bootstrap
from pgw.server.jobs import JobManager, JobRequest
from pgw.server.templates import (
    _ICON_PNG,
    _LOGO_PNG,
    _SIBLING_PREFIX,
    _discover_tracks,
    _discover_workspaces,
    _find_sibling_workspaces,
    _load_metadata,
    _redownload_video_streaming,
)
from pgw.utils.paths import find_video

logger = logging.getLogger(__name__)

# ── Static SPA bundle ────────────────────────────────────────────────────
# ``frontend/`` builds to ``src/pgw/templates/dist/``. The dist directory
# may be missing in dev (before the first ``npm run build``); we tolerate
# that and serve a helpful message.
#
# Set ``PGW_SPA_DIR`` to an absolute path (e.g. a host-mounted volume) to
# override the built-in bundle — useful during Docker-based frontend dev
# where you rebuild on the host and the container picks it up instantly:
#
#   cd frontend && npm run build          # host
#   docker run … -v "$PWD/src/pgw/templates/dist:/spa" -e PGW_SPA_DIR=/spa
_OVERRIDE_DIR = os.environ.get("PGW_SPA_DIR")
if _OVERRIDE_DIR:
    _DIST_PATH = Path(_OVERRIDE_DIR)
    try:
        _SPA_INDEX = (_DIST_PATH / "index.html").read_text(encoding="utf-8")
        _SPA_ASSETS = _DIST_PATH / "assets"
        _SPA_AVAILABLE = _SPA_INDEX != ""
    except (FileNotFoundError, OSError):
        _DIST_PATH = None  # type: ignore[assignment]
        _SPA_INDEX = ""
        _SPA_ASSETS = None  # type: ignore[assignment]
        _SPA_AVAILABLE = False
else:
    _DIST_DIR = files("pgw.templates").joinpath("dist")
    try:
        _DIST_PATH = Path(str(_DIST_DIR))
        _SPA_INDEX = (_DIST_PATH / "index.html").read_text(encoding="utf-8")
        _SPA_ASSETS = _DIST_PATH / "assets"
        _SPA_AVAILABLE = _SPA_INDEX != ""
    except (FileNotFoundError, OSError):
        _DIST_PATH = None  # type: ignore[assignment]
        _SPA_INDEX = ""
        _SPA_ASSETS = None  # type: ignore[assignment]
        _SPA_AVAILABLE = False

_FALLBACK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>pgw — frontend not built</title>
<style>body{font:14px/1.6 system-ui;max-width:42rem;margin:4rem auto;padding:0 1rem}
code{background:#eee;padding:.1rem .3rem;border-radius:.25rem}</style></head>
<body><h1>Frontend bundle not found</h1>
<p>The React app at <code>src/pgw/templates/dist/</code> is missing.</p>
<p>Build it with:</p>
<pre><code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code></pre>
<p>The Docker image builds it automatically; this message only appears
when running <code>pgw serve</code> directly against an unbuilt source tree.</p>
</body></html>"""

_CACHE_MAX_AGE = 86400
_CSP = (
    "default-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "media-src 'self'; "
    "connect-src 'self'; "
    "font-src 'self' data:"
)
_CSP_ASSETS = "default-src 'self'"
_TS_RE = re.compile(r"^\d{8}_\d{6}$")
_SLUG_RE = re.compile(r"^[\w-]+$")
_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


# ── Helpers ──────────────────────────────────────────────────────────────


def _input_is_safe(value: str, base_dir: Path) -> bool:
    """URLs pass; local paths must resolve under ``base_dir``."""
    if not value:
        return False
    if value.startswith(("http://", "https://", "ftp://")):
        return True
    try:
        resolved = Path(value).resolve()
    except OSError:
        return False
    if not resolved.is_file():
        return False
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError:
        return False
    return True


def _safe_filename(name: str) -> str:
    base = Path(name).name
    sanitised = re.sub(r"[^\w.\-]", "_", base)[:128]
    if not sanitised or set(sanitised) == {"_"} or sanitised in (".", ".."):
        return "upload.bin"
    return sanitised


def _png(data: bytes) -> Response:
    return Response(
        data,
        media_type="image/png",
        headers={"Cache-Control": f"public, max-age={_CACHE_MAX_AGE}"},
    )


def _validate_ws(slug: str, timestamp: str, base_dir: Path) -> Path:
    if not _SLUG_RE.match(slug) or not _TS_RE.match(timestamp):
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    workspace = base_dir / slug / timestamp
    if not workspace.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return workspace


def _workspace_summary(ws: dict) -> dict:
    """Convert the dict from ``_discover_workspaces`` into JSON-safe shape."""
    return {
        "slug": ws["slug"],
        "timestamp": ws["timestamp"],
        "title": ws["title"],
        "language": ws.get("language") or None,
        "target_language": ws.get("target_language") or None,
        "lang_pairs": ws.get("lang_pairs", []),
        "duration": ws.get("duration"),
        "created_at": ws.get("created_at"),
        "upload_date": ws.get("upload_date") or None,
        "uploader": ws.get("uploader") or None,
        "thumbnail": ws.get("thumbnail") or None,
        "difficulty": ws.get("difficulty") or None,
        "has_video": bool(ws.get("has_video")),
    }


def _workspace_detail(workspace: Path, base_dir: Path) -> dict:
    """Full workspace blob: metadata + tracks + downloadable files + sibling list."""
    meta = _load_metadata(workspace)
    siblings = _find_sibling_workspaces(workspace, base_dir)
    tracks = _discover_tracks(workspace, sibling_paths=siblings)
    files_index = []
    for entry in sorted(workspace.iterdir()):
        if not entry.is_file() or entry.name.startswith("."):
            continue
        try:
            size = entry.stat().st_size
        except OSError:
            size = 0
        files_index.append(
            {
                "name": entry.name,
                "size": size,
                "suffix": entry.suffix.lstrip("."),
            }
        )
    video = find_video(workspace)
    return {
        "slug": workspace.parent.name,
        "timestamp": workspace.name,
        "metadata": meta,
        "tracks": tracks,
        "files": files_index,
        "video": video.name if video is not None else None,
        "siblings": [{"slug": sp.parent.name, "timestamp": sp.name} for sp in siblings],
    }


def _read_vocab(workspace: Path) -> dict | None:
    for f in sorted(workspace.glob("vocabulary.*.json")):
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _form_defaults() -> dict:
    """Pre-fill the Studio form with the user's pgw config."""
    try:
        from pgw.core.config import load_config

        cfg = load_config()
        return {
            "language": cfg.whisper.language or "fr",
            "translate": cfg.llm.target_language or "",
            "backend": cfg.whisper.backend or "local",
            "llm_backend": cfg.llm.backend or "local",
            "whisper_model": cfg.whisper.model or "",
            "llm_model": cfg.llm.model or "",
        }
    except Exception:  # noqa: BLE001 - best-effort
        return {
            "language": "fr",
            "translate": "",
            "backend": "api",
            "llm_backend": "api",
            "whisper_model": "",
            "llm_model": "",
        }


# ── Library app ──────────────────────────────────────────────────────────


def get_job_manager() -> JobManager | None:
    """Module-level shim for legacy callers.

    The canonical handle is now ``app.state.job_manager`` — set in the
    factory below. This helper looks the manager up via the most-recently
    created FastAPI app's state when callers can't reach a request scope
    (e.g. the worker WS handler before it has a chance to read
    ``ws.app.state``). Prefer ``ws.app.state.job_manager`` /
    ``request.app.state.job_manager`` over this.
    """
    return _last_app.state.job_manager if _last_app is not None else None


# Tracks the most-recently created app so ``get_job_manager()`` keeps
# working for code paths that already imported it. New code should pull
# from request/WebSocket scope instead.
_last_app: FastAPI | None = None


def _bootstrap_db() -> None:
    """Bring the schema up to date + provision admin from env if any.

    On Postgres we run ``alembic upgrade head`` so the
    ``alembic_version`` table tracks the schema and future migrations
    apply incrementally. On SQLite (tests + single-user dev) we keep
    the fast ``Base.metadata.create_all`` path so each test fixture
    doesn't pay migration cost.
    """
    import pgw.db.models  # noqa: F401  ensure ORM tables registered
    from pgw.auth.bootstrap import ensure_admin_from_env
    from pgw.db import Base, SessionLocal, get_engine

    engine = get_engine()
    if engine.dialect.name == "postgresql":
        _run_alembic_upgrade()
    else:
        Base.metadata.create_all(engine)

    with SessionLocal() as db:
        ensure_admin_from_env(db)


def _run_alembic_upgrade() -> None:
    """Programmatic ``alembic upgrade head`` against the configured engine.

    Falls back to ``Base.metadata.create_all`` if alembic.ini cannot be
    located on disk (e.g. running from a wheel without the migration
    tree). Stamps head when a legacy ``create_all`` schema is detected.
    """
    from sqlalchemy import inspect

    from pgw.db import Base, get_engine

    here = Path(__file__).resolve()
    cfg_path: Path | None = None
    for parent in [here.parent, *here.parents]:
        candidate = parent / "alembic.ini"
        if candidate.is_file():
            cfg_path = candidate
            break

    if cfg_path is None:
        logger.warning("alembic.ini not found — falling back to create_all")
        Base.metadata.create_all(get_engine())
        return

    try:
        from alembic import command
        from alembic.config import Config

        cfg = Config(str(cfg_path))
        cfg.set_main_option("script_location", str(cfg_path.parent / "alembic"))

        engine = get_engine()
        tables = set(inspect(engine).get_table_names())
        if "users" in tables and "alembic_version" not in tables:
            logger.warning("legacy create_all schema detected; stamping head")
            command.stamp(cfg, "head")
        else:
            command.upgrade(cfg, "head")
    except Exception:
        logger.exception("alembic upgrade failed; falling back to create_all")
        Base.metadata.create_all(get_engine())


def create_library_app(base_dir: Path, jobs: JobManager) -> FastAPI:
    _bootstrap_db()

    from pgw.server.worker_registry import GLOBAL_WORKERS

    GLOBAL_WORKERS.set_disconnect_callback(
        lambda _user_id, job_ids: jobs.mark_jobs_interrupted(job_ids)
    )

    app = FastAPI(
        title="pgw library",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    # Canonical handle for routes/WS handlers — read via
    # ``request.app.state.job_manager`` or ``ws.app.state.job_manager``.
    app.state.job_manager = jobs
    app.state.base_dir = base_dir
    global _last_app
    _last_app = app

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        """Liveness + readiness probe for docker-compose / k8s.

        Checks DB reachability and worker-registry sanity. Returns
        ``200 {"status": "ok", ...}`` on success. Any failure returns
        ``503`` with the failing component name so operators can
        triage quickly.
        """
        from sqlalchemy import text

        from pgw.db import get_engine
        from pgw.server.worker_registry import GLOBAL_WORKERS as _GW

        components: dict[str, str] = {}
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            components["db"] = "ok"
        except Exception as exc:  # noqa: BLE001
            components["db"] = f"error: {exc.__class__.__name__}"

        try:
            # Touching the lock counts as registry sanity — a held lock
            # would surface as a deadlock here, not silent corruption.
            with _GW._lock:  # noqa: SLF001
                workers_known = len(_GW._workers)  # noqa: SLF001
            components["worker_registry"] = "ok"
            components["workers_known"] = str(workers_known)
        except Exception as exc:  # noqa: BLE001
            components["worker_registry"] = f"error: {exc.__class__.__name__}"

        ok = all(v == "ok" for k, v in components.items() if k != "workers_known")
        if not ok:
            raise HTTPException(status_code=503, detail={"status": "degraded", **components})
        return {"status": "ok", **components}

    # Mount auth + setup endpoints first so they're discoverable
    # even before the rest of the app initialises.
    from pgw.server.routes.auth import router as auth_router
    from pgw.server.routes.workers import router as workers_router
    from pgw.server.routes.workers import ws_router as worker_ws_router

    app.include_router(auth_router)
    app.include_router(workers_router)
    app.include_router(worker_ws_router)

    # ── Static icons (referenced by both SPA and the OS-level favicon) ──

    @app.get("/icon.png", include_in_schema=False)
    def icon() -> Response:
        return _png(_ICON_PNG)

    @app.get("/logo.png", include_in_schema=False)
    def logo() -> Response:
        return _png(_LOGO_PNG)

    # ── JSON APIs the SPA consumes ──

    @app.get("/api/workspaces")
    def api_workspaces() -> dict:
        rows = _discover_workspaces(base_dir, backfill_metadata=False)
        return {"workspaces": [_workspace_summary(r) for r in rows]}

    @app.get("/api/workspaces/{slug}/{timestamp}")
    def api_workspace(slug: str, timestamp: str) -> dict:
        workspace = _validate_ws(slug, timestamp, base_dir)
        return _workspace_detail(workspace, base_dir)

    @app.get("/api/workspaces/{slug}/{timestamp}/vocab")
    def api_workspace_vocab(slug: str, timestamp: str) -> dict:
        workspace = _validate_ws(slug, timestamp, base_dir)
        vocab = _read_vocab(workspace)
        if vocab is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No vocabulary file found")
        return vocab

    @app.get("/api/config/defaults")
    def api_form_defaults() -> dict:
        return _form_defaults()

    @app.get("/api/languages")
    def api_languages() -> list[dict]:
        from pgw.core.languages import ALL_LANGUAGES

        return [
            {
                "code": li.code,
                "name": li.name,
                "has_spacy": li.has_spacy,
                "has_alignment": li.has_alignment,
            }
            for li in ALL_LANGUAGES
        ]

    # ── Jobs API ──

    @app.get("/jobs")
    def list_jobs() -> dict:
        return {"jobs": [r.public() for r in jobs.list()]}

    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        request: Request,
        user=Depends(current_user_or_bootstrap),
    ) -> dict:
        from pgw.errors import Err
        from pgw.errors import envelope as err_envelope
        from pgw.server.exceptions import WorkerNotConnectedError
        from pgw.server.worker_registry import GLOBAL_WORKERS

        try:
            body = await request.body()
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid JSON")
        try:
            req = JobRequest(**payload)
        except Exception as exc:  # noqa: BLE001 - surface validation errors
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid request: {exc}") from exc
        if not _input_is_safe(req.input, base_dir):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=err_envelope(Err.JOB_INPUT_REJECTED, "Input path is not allowed"),
            )

        # Server-side execution uses operator API keys — restrict to
        # admins. Bootstrap mode (SYSTEM_USER) is treated as admin so
        # solo + pre-setup deployments still work.
        worker_connected = user.id is not None and GLOBAL_WORKERS.is_connected(user.id)
        will_run_locally = req.executor == "server" or (
            req.executor == "auto" and not worker_connected
        )
        if will_run_locally and not user.is_admin:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail=err_envelope(
                    Err.AUTH_ADMIN_REQUIRED,
                    "server-side execution is admin-only — "
                    "connect a worker with `pgw worker connect`",
                ),
            )

        try:
            job_id = jobs.submit(req, user_id=user.id)
        except WorkerNotConnectedError as exc:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail=err_envelope(Err.WORKER_NO_CONNECTED, str(exc)),
            ) from exc

        return {"job_id": job_id}

    @app.post(
        "/api/jobs/{job_id}/artifacts",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def upload_artifact(
        job_id: str,
        request: Request,
        slug: str,
        timestamp: str,
        name: str,
    ) -> Response:
        """Worker pushes a single artifact (VTT, JSON, audio) to the
        server's filesystem under the workspace path.

        Auth: ``Authorization: Bearer <worker-token>`` header.
        """
        if not _JOB_ID_RE.match(job_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)

        auth = request.headers.get("Authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail="expected Authorization: Bearer <worker-token>",
            )
        raw_token = auth[len("bearer ") :].strip()
        from pgw.db.session import SessionLocal
        from pgw.worker.tokens import lookup as lookup_token

        with SessionLocal() as db:
            row = lookup_token(db, raw_token)
        if row is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail="invalid or revoked worker token",
            )

        workspace = _validate_ws(slug, timestamp, base_dir)
        safe_name = _safe_filename(name)
        body = await request.body()
        if not body:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty body")
        dest = workspace / safe_name
        dest.write_bytes(body)
        return Response(status_code=204)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        if not _JOB_ID_RE.match(job_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        snap = jobs.snapshot(job_id)
        if snap is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return snap.public()

    @app.delete("/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        if not _JOB_ID_RE.match(job_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return {"cancelled": jobs.cancel(job_id)}

    @app.get("/jobs/{job_id}/events")
    async def stream_job(job_id: str) -> StreamingResponse:
        if not _JOB_ID_RE.match(job_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        if jobs.snapshot(job_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return StreamingResponse(
            jobs.stream(job_id),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/uploads", status_code=status.HTTP_201_CREATED)
    async def upload(file: UploadFile) -> dict:
        if file.filename is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No filename")
        safe = _safe_filename(file.filename)
        target_dir = jobs.uploads_dir / uuid.uuid4().hex
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe
        size = 0
        try:
            with open(target_path, "wb") as out:
                while chunk := await file.read(1 << 20):
                    out.write(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, f"Save failed: {exc}"
            ) from exc
        return {"files": [{"path": str(target_path), "name": safe, "size": size}]}

    # ── Workspace file passthrough (raw blobs the player consumes) ──

    @app.get("/ws/{slug}/{timestamp}/{file_part:path}", include_in_schema=False)
    def workspace_file(slug: str, timestamp: str, file_part: str) -> Response:
        workspace = _validate_ws(slug, timestamp, base_dir)
        return _resolve_workspace_asset(workspace, file_part, base_dir)

    @app.post("/ws/{slug}/{timestamp}/redownload", include_in_schema=False)
    def workspace_redownload(slug: str, timestamp: str) -> StreamingResponse:
        workspace = _validate_ws(slug, timestamp, base_dir)
        return _stream_redownload_response(workspace)

    # ── SPA shell + assets ──

    @app.get("/assets/{file_path:path}", include_in_schema=False)
    def spa_asset(file_path: str) -> Response:
        if _SPA_ASSETS is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        safe = Path(file_path)
        if safe.is_absolute() or ".." in safe.parts:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        target = Path(str(_SPA_ASSETS)) / safe
        if not target.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return FileResponse(
            target,
            headers={
                "Cache-Control": f"public, max-age={_CACHE_MAX_AGE}, immutable",
                "Content-Security-Policy": _CSP_ASSETS,
            },
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_index(full_path: str) -> HTMLResponse:
        # Catch-all so any SPA deep link (/library/<slug>/<ts>) refreshes
        # cleanly. Any path that should NOT hit the SPA must be defined
        # above this route.
        del full_path
        if not _SPA_AVAILABLE:
            return HTMLResponse(
                _FALLBACK_HTML,
                status_code=503,
                headers={"Content-Security-Policy": _CSP},
            )
        return HTMLResponse(_SPA_INDEX, headers={"Content-Security-Policy": _CSP})

    return app


# ── Workspace app (`pgw serve <path>`) ───────────────────────────────────


def _ensure_workspace_app_db() -> None:
    """Workspace mode also needs the schema for auth (when enabled).

    Single-workspace mode is auth-optional in P9 via ``PGW_AUTH_OPTIONAL``;
    we still bootstrap the schema so a future user opt-in to auth works
    without a separate migration step. Postgres takes the alembic path,
    SQLite uses ``create_all``.
    """
    import pgw.db.models  # noqa: F401
    from pgw.db import Base, get_engine

    engine = get_engine()
    if engine.dialect.name == "postgresql":
        _run_alembic_upgrade()
    else:
        Base.metadata.create_all(engine)


def create_workspace_app(workspace: Path) -> FastAPI:
    _ensure_workspace_app_db()

    """Serve a single workspace via the same React SPA, pinned to one path.

    The SPA still sees a "library" of size 1 — simpler than maintaining a
    parallel UI for the single-workspace case.
    """
    app = FastAPI(
        title="pgw workspace",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    base_dir = workspace.parent.parent
    slug = workspace.parent.name
    ts = workspace.name

    @app.get("/icon.png", include_in_schema=False)
    def icon() -> Response:
        return _png(_ICON_PNG)

    @app.get("/logo.png", include_in_schema=False)
    def logo() -> Response:
        return _png(_LOGO_PNG)

    @app.get("/api/workspaces")
    def api_workspaces() -> dict:
        meta = _load_metadata(workspace)
        return {
            "workspaces": [
                _workspace_summary(
                    {
                        "slug": slug,
                        "timestamp": ts,
                        "title": meta.get("title", slug),
                        "language": meta.get("language", ""),
                        "target_language": meta.get("target_language", ""),
                        "duration": meta.get("source_duration"),
                        "created_at": meta.get("created_at", ""),
                        "has_video": find_video(workspace) is not None,
                        "upload_date": meta.get("upload_date", ""),
                        "uploader": meta.get("uploader", ""),
                        "thumbnail": meta.get("thumbnail", ""),
                        "lang_pairs": [(meta.get("language", ""), meta.get("target_language", ""))],
                    }
                )
            ]
        }

    @app.get("/api/workspaces/{request_slug}/{request_ts}")
    def api_workspace(request_slug: str, request_ts: str) -> dict:
        if request_slug != slug or request_ts != ts:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return _workspace_detail(workspace, base_dir)

    @app.get("/api/workspaces/{request_slug}/{request_ts}/vocab")
    def api_workspace_vocab(request_slug: str, request_ts: str) -> dict:
        if request_slug != slug or request_ts != ts:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        vocab = _read_vocab(workspace)
        if vocab is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No vocabulary file found")
        return vocab

    @app.get("/api/config/defaults")
    def api_form_defaults() -> dict:
        return _form_defaults()

    @app.get("/api/languages")
    def api_languages() -> list[dict]:
        from pgw.core.languages import ALL_LANGUAGES

        return [
            {
                "code": li.code,
                "name": li.name,
                "has_spacy": li.has_spacy,
                "has_alignment": li.has_alignment,
            }
            for li in ALL_LANGUAGES
        ]

    @app.post("/ws/{request_slug}/{request_ts}/redownload", include_in_schema=False)
    def redownload(request_slug: str, request_ts: str) -> StreamingResponse:
        if request_slug != slug or request_ts != ts:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return _stream_redownload_response(workspace)

    @app.get("/ws/{request_slug}/{request_ts}/{file_part:path}", include_in_schema=False)
    def workspace_file(request_slug: str, request_ts: str, file_part: str) -> Response:
        if request_slug != slug or request_ts != ts:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return _resolve_workspace_asset(workspace, file_part, base_dir)

    @app.get("/assets/{file_path:path}", include_in_schema=False)
    def spa_asset(file_path: str) -> Response:
        if _SPA_ASSETS is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        safe = Path(file_path)
        if safe.is_absolute() or ".." in safe.parts:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        target = Path(str(_SPA_ASSETS)) / safe
        if not target.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return FileResponse(
            target,
            headers={
                "Cache-Control": f"public, max-age={_CACHE_MAX_AGE}, immutable",
                "Content-Security-Policy": _CSP_ASSETS,
            },
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_index(full_path: str) -> HTMLResponse:
        del full_path
        if not _SPA_AVAILABLE:
            return HTMLResponse(
                _FALLBACK_HTML,
                status_code=503,
                headers={"Content-Security-Policy": _CSP},
            )
        return HTMLResponse(_SPA_INDEX, headers={"Content-Security-Policy": _CSP})

    return app


# ── shared internals ─────────────────────────────────────────────────────


def _resolve_workspace_asset(workspace: Path, file_part: str, base_dir: Path) -> Response:
    file_part = file_part.rstrip("/")
    if file_part.startswith(_SIBLING_PREFIX):
        return FileResponse(_resolve_sibling(workspace, file_part, base_dir))
    safe = Path(file_part).name
    file_path = workspace / safe
    if not file_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return FileResponse(file_path)


def _resolve_sibling(workspace: Path, file_part: str, base_dir: Path) -> Path:
    rest = file_part[len(_SIBLING_PREFIX) :]
    if "/" not in rest:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    sibling_ts, filename = rest.split("/", 1)
    if not _TS_RE.match(sibling_ts):
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    safe = Path(filename).name
    candidate = workspace.parent / sibling_ts
    if not candidate.is_dir():
        for sp in _find_sibling_workspaces(workspace, base_dir):
            if sp.name == sibling_ts:
                candidate = sp
                break
        else:
            raise HTTPException(status.HTTP_404_NOT_FOUND)
    file_path = candidate / safe
    if not file_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return file_path


def _stream_redownload_response(workspace: Path) -> StreamingResponse:
    _MAX_IDLE_HEARTBEATS = 60  # 15 min at 15s intervals

    async def iterator() -> AsyncIterator[bytes]:
        import asyncio
        import threading
        from queue import Empty, Queue

        q: Queue = Queue(maxsize=64)
        error_holder: dict[str, str] = {}

        def send(line: str) -> None:
            try:
                q.put(line, timeout=0.5)
            except Full:
                pass

        def worker() -> None:
            try:
                _redownload_video_streaming(workspace, send)
            except Exception as exc:  # noqa: BLE001 - surface to client
                error_holder["msg"] = str(exc) or repr(exc)
            finally:
                try:
                    q.put(None, timeout=1.0)
                except Full:
                    pass

        threading.Thread(target=worker, daemon=True).start()
        loop = asyncio.get_running_loop()
        idle_count = 0
        while True:
            try:
                item = await loop.run_in_executor(None, q.get, True, 15)
            except Empty:
                idle_count += 1
                if idle_count > _MAX_IDLE_HEARTBEATS:
                    yield (
                        json.dumps(
                            {"status": "error", "detail": "Download timed out — worker stalled"}
                        )
                        + "\n"
                    ).encode("utf-8")
                    break
                yield b"\n"
                continue
            if item is None:
                if error_holder:
                    yield (
                        json.dumps({"status": "error", "detail": error_holder["msg"]}) + "\n"
                    ).encode("utf-8")
                break
            idle_count = 0
            yield (item + "\n").encode("utf-8")

    return StreamingResponse(
        iterator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )
