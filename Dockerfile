# Build:  docker build -t pgw .
# Run pipeline (CLI):
#   docker run --rm -it -v "$PWD:/data" pgw run /data/video.mp4 --translate en
# Run web UI (end-to-end pipeline + library):
#   docker run --rm -it -p 8321:8321 -v "$PWD:/data" pgw serve --no-open
#
# The web UI is a TypeScript bundle — the js-builder stage compiles it so
# end users never need Node installed locally.

# ── Stage 1: React SPA build (Vite) ──────────────────────────────────────
FROM node:22-slim AS js-builder

WORKDIR /frontend
# Cache-friendly install: copy lockfile first so the npm layer is reused
# unless dependencies actually change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# Vite writes the bundle to ../src/pgw/templates/dist/. Pre-create the
# parent so the COPY target in the Python builder stage finds it.
RUN mkdir -p ../src/pgw/templates && npm run build


# ── Stage 2: Python build ────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    build-essential python3-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --all-extras --frozen --no-install-project

COPY src/ src/
COPY README.md ./
# Bring in the freshly built React SPA so the Python wheel's templates
# directory ships the latest dist/ tree.
COPY --from=js-builder /src/pgw/templates/dist src/pgw/templates/dist

RUN uv sync --all-extras --frozen


# ── Stage 3: Runtime ─────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    ffmpeg mpv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

ENV PATH="/app/.venv/bin:$PATH"
# Bind to 0.0.0.0 inside the container; host publishes via -p.
ENV PGW_SERVE_HOST=0.0.0.0
# Cap concurrent job workers at 1 by default so a local Whisper model stays
# warm across sequential jobs without VRAM/duplication. Override per run
# with: docker run -e PGW_SERVE_MAX_JOBS=2 ...
ENV PGW_SERVE_MAX_JOBS=1
# Persist .env, workspaces, and job logs across container restarts by
# mounting the working directory at /data.
WORKDIR /data
EXPOSE 8321

ENTRYPOINT ["pgw"]
CMD ["--help"]
