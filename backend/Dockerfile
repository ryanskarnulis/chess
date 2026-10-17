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

# --- maia stage: Maia-3 in a venv of its own, weights included (#510) ----------
# The tiers below `maximum` play Maia-3's human-like move (docs/maia-tiers.md).
# It runs as its own UCI process, like Stockfish, from its own venv: torch and
# the AGPL-licensed maia3 package stay out of the app's. CPU torch only (no
# CUDA wheels), and the 79M checkpoint baked in so it never fetches at runtime.
# Pinned: the code, torch and the weights the tier map was measured with.
FROM python:3.12-slim AS maia

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG MAIA3_COMMIT=1e13597c42d4858b7cfd7cfdae01e297263364b2
ARG MAIA3_WEIGHTS_REVISION=a107d6ceb7b298cb04ae1da4edffe2939858b894
RUN python -m venv /opt/maia \
    && /opt/maia/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && /opt/maia/bin/pip install --no-cache-dir \
        "maia3 @ git+https://github.com/CSSLab/maia3@${MAIA3_COMMIT}" \
    && /opt/maia/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('UofTCSSLab/Maia3-79M', 'maia3-79m.pt', \
revision='${MAIA3_WEIGHTS_REVISION}', local_dir='/opt/maia/weights')" \
    && rm -rf /opt/maia/weights/.cache

# --- runtime stage -------------------------------------------------------------
FROM python:3.12-slim

# Stockfish ships as a Debian package; the binary lands at /usr/games/stockfish.
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=maia /opt/maia /opt/maia
COPY --from=frontend /ui/dist /opt/frontend

# PATH puts the venv first (resolves `chessapp` + `python`). CHESSAPP_STOCKFISH
# attaches the engine by default; still optional — unset it to run engine-off.
# CHESSAPP_MAIA attaches Maia-3 for the tiers below `maximum`, with its baked-in
# weights; unset it and those tiers play their Stockfish settings.
# CHESSAPP_STATIC_DIR serves the baked-in UI at / (unset it for API-only).
# Host 0.0.0.0 so the app is reachable across the home network.
# The commit this image was built from (#317): the image carries no .git, so
# the serving manifest can only name its revision if the build is told it.
# deploy.yml passes `git rev-parse HEAD`; a bare `docker compose build` says
# "unknown", which the manifest records as exactly that.
ARG CHESSAPP_REVISION=unknown
ENV PATH=/opt/venv/bin:$PATH \
    CHESSAPP_REVISION=$CHESSAPP_REVISION \
    CHESSAPP_STOCKFISH=/usr/games/stockfish \
    CHESSAPP_MAIA=/opt/maia/bin/maia3-uci \
    CHESSAPP_MAIA_CHECKPOINT=/opt/maia/weights/maia3-79m.pt \
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
