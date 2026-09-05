# App service image — the "app" layer of the compose stack.
#
# Build context is the REPO ROOT (see docker-compose.yml / .dockerignore): the
# image bakes in both the backend and the built frontend, so one container
# serves the UI and the API from the same origin — the UI's relative /api and
# /ws URLs need no proxy or CORS.
#
# Bundles Stockfish so a full deterministic game works with the LLM turned off:
# the engine is a calculation tool driven in-process via python-chess UCI, not a
# separate container. The llama-server "brain" *is* a separate container, reached
# over the network (LLAMACPP_BASE_URL) — so nothing here needs the model.

# --- frontend stage: build the static UI bundle --------------------------------
FROM node:24-slim AS frontend

WORKDIR /ui

# Lockfile first so the (slow) npm ci layer caches across source-only changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- build stage: compile the wheel into a self-contained venv -----------------
FROM python:3.12-slim AS build

WORKDIR /app

# Only the package sources are needed to build (see .dockerignore); copying them
# explicitly keeps the build cache tight and the layer independent of tests/docs.
COPY backend/pyproject.toml ./
COPY backend/src ./src

# A venv at a fixed path is relocatable to the same path in the runtime stage,
# which keeps the console-script shebangs (#!/opt/venv/bin/python) valid.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir .

# --- runtime stage -------------------------------------------------------------
FROM python:3.12-slim

# Stockfish ships as a Debian package; the binary lands at /usr/games/stockfish.
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=frontend /ui/dist /opt/frontend

# PATH puts the venv first (resolves `chessapp` + `python`). CHESSAPP_STOCKFISH
# attaches the engine by default; still optional — unset it to run engine-off.
# CHESSAPP_STATIC_DIR serves the baked-in UI at / (unset it for API-only).
# Host 0.0.0.0 so the app is reachable across the home network.
ENV PATH=/opt/venv/bin:$PATH \
    CHESSAPP_STOCKFISH=/usr/games/stockfish \
    CHESSAPP_STATIC_DIR=/opt/frontend \
    CHESSAPP_HOST=0.0.0.0 \
    CHESSAPP_PORT=8000

# Don't run the network service as root. Pre-create the saved-games dir owned
# by `app` so a named volume mounted there (compose sets CHESSAPP_SAVE_DIR to
# it) inherits app ownership on first creation and stays writable by uid 1000.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/saves \
    && chown -R app:app /data
USER app

EXPOSE 8000

# Self-contained probe (no curl in the slim image): the API is up once
# /api/state answers 200. `chessapp.healthcheck` rather than a `python -c`
# one-liner — it carries its own connect timeout, prints one line instead of a
# traceback, and can be tested (`tests/test_healthcheck.py`).
#
# Ten seconds and two retries, not thirty and three: the old numbers left up to
# two minutes of "healthy" over a dead API, which is what the 2026-09-04
# walkthrough saw. A restart is cheap here (a cold start is ~5s) and the app is
# the only thing in the container, so detecting death fast costs nothing and
# waiting costs the whole deploy window.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=2 \
    CMD ["python", "-m", "chessapp.healthcheck"]

CMD ["chessapp"]
