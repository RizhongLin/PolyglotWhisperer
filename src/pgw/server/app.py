"""FastAPI app factories for ``pgw serve``.

Two ASGI apps are provided:

- :func:`create_library_app` — multi-workspace library browser plus the
  end-to-end pipeline UI (``POST /jobs``, ``GET /jobs/<id>/events``, …).
- :func:`create_workspace_app` — single-workspace player (the ``pgw serve
  <path>`` mode), plus the "re-download missing video" SSE stream.

Both apps share the underlying static assets (HTML, CSS, JS, icons)
loaded once at module import via ``importlib.resources``.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from queue import Full
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import (
    HTMLResponse,
    Response,
    StreamingResponse,
)
from starlette.responses import FileResponse

from pgw.server.jobs import JobManager, JobRequest
from pgw.server.templates import (
    _ICON_PNG,
    _LIBRARY_CSS,
    _LOGO_PNG,
    _PLAYER_CSS,
    _PLAYER_JS,
    _SIBLING_PREFIX,
    _build_html,
    _build_library_html,
    _discover_workspaces,
    _find_sibling_workspaces,
    _redownload_video_streaming,
    get_jobs_js,
)
from pgw.utils.paths import find_video

_CACHE_MAX_AGE = 86400
_CSP_LIBRARY = (
    "default-src 'self'; "
    "style-src 'self' https://cdn.jsdelivr.net; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https:; "
    "media-src 'self'; "
    "connect-src 'self' https://cdn.jsdelivr.net"
)
_CSP_WORKSPACE = _CSP_LIBRARY  # same policy for both
_TS_RE = re.compile(r"^\d{8}_\d{6}$")
_SLUG_RE = re.compile(r"^[\w-]+$")
_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_VALID_BACKENDS = {"local", "api"}


# ── shared helpers ───────────────────────────────────────────────────────


def _input_is_safe(value: str, base_dir: Path) -> bool:
    """Reject path traversal and arbitrary filesystem probing.

    URLs (``http(s)://`` / ``ftp://``) pass straight through. Local paths
    must resolve to an existing file under ``base_dir`` (uploads live in
    ``base_dir/.uploads/<uuid>/``). With ``PGW_SERVE_HOST=0.0.0.0`` (the
    Docker default), this stops network peers from probing arbitrary
    server-side paths or coercing the pipeline into reading e.g.
    ``/etc/passwd``.
    """
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
    """Strip directory components and unsafe chars from an upload filename.

    Falls back to ``upload.bin`` when the input would sanitise to an empty
    or all-underscore string (which carries no useful information for the
    user inspecting their .uploads/ directory later).
    """
    base = Path(name).name
    sanitised = re.sub(r"[^\w.\-]", "_", base)[:128]
    if not sanitised or set(sanitised) == {"_"}:
        return "upload.bin"
    return sanitised


def _html_response(content: str, csp: str = _CSP_LIBRARY) -> HTMLResponse:
    return HTMLResponse(
        content,
        headers={"Content-Security-Policy": csp},
    )


def _png_response(data: bytes) -> Response:
    return Response(
        data,
        media_type="image/png",
        headers={"Cache-Control": f"public, max-age={_CACHE_MAX_AGE}"},
    )


def _serve_workspace_file(workspace: Path, filename: str) -> FileResponse:
    safe = Path(filename).name
    file_path = workspace / safe
    if not file_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return FileResponse(file_path)


def _resolve_sibling(workspace: Path, file_part: str, base_dir: Path) -> Path:
    """Resolve ``sibling:<timestamp>/<file>`` paths relative to a workspace."""
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


# ── library app (no-arg `pgw serve`) ─────────────────────────────────────


def create_library_app(base_dir: Path, jobs: JobManager) -> FastAPI:
    """Build the library/jobs ASGI app.

    Args:
        base_dir: Workspace directory (the ``config.workspace_dir``).
        jobs:     Shared :class:`JobManager` — owns the worker pool, log
                  fan-out, and persistence under ``base_dir/.jobs``.
    """
    app = FastAPI(
        title="pgw library",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # ── Page + assets ───────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
    def library_index() -> HTMLResponse:
        workspaces = _discover_workspaces(base_dir, backfill_metadata=False)
        return _html_response(_build_library_html(workspaces))

    @app.get("/library.css", include_in_schema=False)
    def library_css() -> Response:
        return Response(_LIBRARY_CSS, media_type="text/css; charset=utf-8")

    @app.get("/jobs.js", include_in_schema=False)
    def jobs_js() -> Response:
        return Response(
            get_jobs_js(),
            media_type="application/javascript; charset=utf-8",
        )

    @app.get("/icon.png", include_in_schema=False)
    def icon() -> Response:
        return _png_response(_ICON_PNG)

    @app.get("/logo.png", include_in_schema=False)
    def logo() -> Response:
        return _png_response(_LOGO_PNG)

    # ── Jobs API ────────────────────────────────────────────────────

    @app.get("/jobs")
    def list_jobs() -> dict:
        return {"jobs": [r.public() for r in jobs.list()]}

    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(request: Request) -> dict:
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
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Input path is not allowed")
        return {"job_id": jobs.submit(req)}

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

    # ── Workspace passthrough (so /ws/<slug>/<ts>/ works in library mode) ──

    @app.get("/ws/{slug}/{timestamp}", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/ws/{slug}/{timestamp}/", response_class=HTMLResponse, include_in_schema=False)
    def workspace_index(slug: str, timestamp: str) -> HTMLResponse:
        workspace = _validate_ws(slug, timestamp, base_dir)
        video = find_video(workspace)
        siblings = _find_sibling_workspaces(workspace, base_dir)
        html = _build_html(
            workspace,
            video,
            url_prefix=f"/ws/{slug}/{timestamp}",
            sibling_paths=siblings,
            library_url="/",
        )
        return _html_response(html, csp=_CSP_WORKSPACE)

    @app.get("/ws/{slug}/{timestamp}/{file_part:path}", include_in_schema=False)
    def workspace_file(slug: str, timestamp: str, file_part: str) -> Response:
        workspace = _validate_ws(slug, timestamp, base_dir)
        return _resolve_workspace_asset(workspace, file_part, base_dir)

    @app.post("/ws/{slug}/{timestamp}/redownload", include_in_schema=False)
    def workspace_redownload(slug: str, timestamp: str) -> StreamingResponse:
        workspace = _validate_ws(slug, timestamp, base_dir)
        return _stream_redownload_response(workspace)

    return app


# ── workspace app (`pgw serve <path>`) ───────────────────────────────────


def create_workspace_app(workspace: Path) -> FastAPI:
    app = FastAPI(
        title="pgw workspace",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
    def workspace_index() -> HTMLResponse:
        video = find_video(workspace)
        return _html_response(_build_html(workspace, video), csp=_CSP_WORKSPACE)

    @app.get("/player.css", include_in_schema=False)
    def player_css() -> Response:
        return Response(_PLAYER_CSS, media_type="text/css; charset=utf-8")

    @app.get("/player.js", include_in_schema=False)
    def player_js() -> Response:
        return Response(
            _PLAYER_JS,
            media_type="application/javascript; charset=utf-8",
        )

    @app.get("/icon.png", include_in_schema=False)
    def icon() -> Response:
        return _png_response(_ICON_PNG)

    @app.get("/logo.png", include_in_schema=False)
    def logo() -> Response:
        return _png_response(_LOGO_PNG)

    @app.post("/redownload", include_in_schema=False)
    def redownload() -> StreamingResponse:
        return _stream_redownload_response(workspace)

    @app.get("/{filename:path}", include_in_schema=False)
    def workspace_asset(filename: str) -> Response:
        if filename in {"index.html", "player.css", "player.js", "icon.png", "logo.png"}:
            # Already routed above; keep this catch-all for normal files.
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        return _serve_workspace_file(workspace, filename)

    return app


# ── shared internals ─────────────────────────────────────────────────────


def _validate_ws(slug: str, timestamp: str, base_dir: Path) -> Path:
    if not _SLUG_RE.match(slug) or not _TS_RE.match(timestamp):
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    workspace = base_dir / slug / timestamp
    if not workspace.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return workspace


def _resolve_workspace_asset(workspace: Path, file_part: str, base_dir: Path) -> Response:
    file_part = file_part.rstrip("/")
    if file_part == "" or file_part == "index.html":
        slug = workspace.parent.name
        timestamp = workspace.name
        video = find_video(workspace)
        siblings = _find_sibling_workspaces(workspace, base_dir)
        html = _build_html(
            workspace,
            video,
            url_prefix=f"/ws/{slug}/{timestamp}",
            sibling_paths=siblings,
            library_url="/",
        )
        return _html_response(html, csp=_CSP_WORKSPACE)
    if file_part == "player.css":
        return Response(_PLAYER_CSS, media_type="text/css; charset=utf-8")
    if file_part == "player.js":
        return Response(_PLAYER_JS, media_type="application/javascript; charset=utf-8")
    if file_part == "icon.png":
        return _png_response(_ICON_PNG)
    if file_part == "logo.png":
        return _png_response(_LOGO_PNG)
    if file_part.startswith(_SIBLING_PREFIX):
        sibling_path = _resolve_sibling(workspace, file_part, base_dir)
        return FileResponse(sibling_path)
    return _serve_workspace_file(workspace, file_part)


def _stream_redownload_response(workspace: Path) -> StreamingResponse:
    """Bridge the existing sync ``send_event(str)`` callback to NDJSON.

    Worker exceptions are forwarded to the client as a final
    ``{"status":"error", "detail":...}`` line so the UI can surface
    re-download failures rather than seeing a clean EOS.
    """

    async def iterator() -> AsyncIterator[bytes]:
        import asyncio
        import threading
        from queue import Empty, Queue

        q: Queue = Queue(maxsize=64)
        error_holder: dict[str, str] = {}

        def send(line: str) -> None:
            try:
                q.put_nowait(line)
            except Full:
                pass

        def worker() -> None:
            try:
                _redownload_video_streaming(workspace, send)
            except Exception as exc:  # noqa: BLE001 - surface to client
                error_holder["msg"] = str(exc) or repr(exc)
            finally:
                q.put_nowait(None)

        threading.Thread(target=worker, daemon=True).start()
        loop = asyncio.get_running_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, q.get, True, 15)
            except Empty:
                yield b"\n"
                continue
            if item is None:
                if error_holder:
                    yield (
                        json.dumps({"status": "error", "detail": error_holder["msg"]}) + "\n"
                    ).encode("utf-8")
                break
            yield (item + "\n").encode("utf-8")

    return StreamingResponse(
        iterator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )
