# Backend service image — the "app" layer of the compose stack.
#
# Bundles Stockfish so a full deterministic game works with the LLM turned off:
# the engine is a calculation tool driven in-process via python-chess UCI, not a
# separate container. The llama-server "brain" *is* a separate container, reached
# over the network (CHESSAPP_LLAMA_URL) — so nothing here needs the model.

# --- build stage: compile the wheel into a self-contained venv -----------------
FROM python:3.12-slim AS build

WORKDIR /app

# Only the package sources are needed to build (see .dockerignore); copying them
# explicitly keeps the build cache tight and the layer independent of tests/docs.
COPY pyproject.toml ./
COPY src ./src

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

# PATH puts the venv first (resolves `chessapp` + `python`). CHESSAPP_STOCKFISH
# attaches the engine by default; still optional — unset it to run engine-off.
# Host 0.0.0.0 so the app is reachable across the home network.
ENV PATH=/opt/venv/bin:$PATH \
    CHESSAPP_STOCKFISH=/usr/games/stockfish \
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

# Self-contained probe (no curl in the slim image): the API is up once /api/state
# answers 200. The compose slice can lean on this for depends_on healthchecks.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/state').status==200 else 1)"

CMD ["chessapp"]
