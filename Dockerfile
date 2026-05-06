# Build:  docker build -t pgw .
# Run:    docker run --rm -it -v "$PWD:/data" pgw run /data/video.mp4 --translate en

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    build-essential python3-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --all-extras --frozen --no-install-project

COPY src/ src/
COPY README.md ./
RUN uv sync --all-extras --frozen


FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    ffmpeg mpv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

ENV PATH="/app/.venv/bin:$PATH"
ENV PGW_SERVE_HOST=0.0.0.0
# load_dotenv() reads .env from CWD (/data), so mount with -v "$PWD:/data"
RUN python -m pip install --upgrade pip --quiet

WORKDIR /data

ENTRYPOINT ["pgw"]
CMD ["--help"]
