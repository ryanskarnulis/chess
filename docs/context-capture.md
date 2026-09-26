# Context capture

The turn trace (`CHESSAPP_TRACE_PATH`) says what the agent decided and what
each call cost. It does not say what the model was *shown*, and that is the
open question when a turn goes wrong: was the context missing something, or
did it hold too much? Context capture answers it by keeping every model call
exactly as it went over the wire (#359).

It is a debugging tool. Turn it on for a session you are looking into and
turn it off after.

## Turning it on

```bash
cd backend
CHESSAPP_CONTEXT_PATH=/tmp/context.jsonl CHESSAPP_TRACE_PATH=/tmp/turns.jsonl chessapp
```

Then, in another terminal, watch it while you play:

```bash
cd backend
python scripts/watch_context.py /tmp/context.jsonl --trace /tmp/turns.jsonl
```

The watcher prints each call as it lands, grouped under its turn: the prompt
the chat template rendered, then the raw response. When the turn record is
written, the turn's decisions follow: tools and results, the engine's reply,
the guard, and the commentary. Options:

- `--phase planner` follows one phase (`planner`, `closer`, `reaction`,
  `rewrite`, `answer`).
- `--json` prints the request body instead of the rendered prompt.
- `--from-start` replays the files from the top instead of waiting at the end.

The captured text is printed exactly as captured. The headers and separators
are the only things the watcher adds, and the decisions block is the only part
it formats.

## The deployed app

CD deploys from a clean clone at `~/deploy/chess` and refuses to deploy over
local edits there ("deploy clone is dirty", `.github/workflows/deploy.yml`).
So don't add the variable to that clone's `docker-compose.yml`, where a
forgotten edit stops every later deploy. Put it in an override file outside
the clone and recreate the container with both files:

```bash
cat > /tmp/context-capture.override.yml <<'EOF'
services:
  app:
    environment:
      CHESSAPP_CONTEXT_PATH: /data/saves/context.jsonl
EOF
cd ~/deploy/chess
docker compose -f docker-compose.yml -f /tmp/context-capture.override.yml up -d --no-build app
```

The live game survives the restart (`live.json`,
`docs/persistence-and-identity.md`).

The capture lands on the `chess-saves` volume next to the trace, and the host
user can't read that volume. The watcher needs only the standard library, so
run it inside the container, fed from this checkout (from the repo root):

```bash
docker exec -i chess-app-1 python -u - /data/saves/context.jsonl --trace /data/saves/turns.jsonl < backend/scripts/watch_context.py
```

Options go after the paths. For example, add `--from-start` to replay a
session you have already played.

To turn it off, recreate the container without the override. The next CD
deploy does the same, since it runs `docker compose up` without it.

```bash
cd ~/deploy/chess && docker compose up -d --no-build app
```

The capture file stays on the volume until you delete it:
`docker exec chess-app-1 rm /data/saves/context.jsonl`.

## What a record holds

The file gets one JSONL line per model call (`kind: "model_call"`,
`schema: 1`), written when the call ends:

| field | what it is |
| --- | --- |
| `turn_id`, `correlation_id` | the interaction, the same ids the turn trace uses. `null` for a call made outside one (evals, probes) |
| `phase` | `planner`, `closer`, `reaction`, `rewrite`, `answer`, or `unknown` |
| `seq` | the call's place among the calls its interaction sent, counted when it was sent |
| `started_at`, `ended_at`, `ms` | wall clock |
| `url` | where the request went |
| `request` | the request body, byte for byte: messages, tools, and every sampling and generation parameter |
| `status_code`, `response` | the response body, byte for byte, including `reasoning_content` and raw tool-call argument strings. `null` when no response arrived |
| `error` | the typed provider error the call raised, `""` when it did not raise |
| `template` | `{"prompt": ...}`: the string the server's chat template rendered from the same request, which is the text the model tokenized. `{"error": ...}` when rendering failed. `null` when the server never answered |

Nothing is reformatted, trimmed or redacted.

## How it works

- **At the provider seam.** Every model call goes through
  `LlamaCppProvider._post`, so no phase can be missed. The request is built
  once and its `content` is both what is sent and what is recorded.
- **Ids and phase by `ContextVar`.** The provider does not know which turn or
  phase it is serving. The interaction's ids come from `progress` (the same
  variable progress events read, copied into worker threads). The phase comes
  from `context_capture.model_phase`, which the brain wraps around each call
  site.
- **The rendered prompt comes from the server.** After the call returns, the
  same request bytes are POSTed to
  `{root}/upstream/{model}/apply-template`. llama-swap proxies that to the
  model's own llama-server, which renders the messages and tools through its
  `--jinja` template and returns the string. Keeping a copy of the template
  in this repo would give a second version that could drift from the server's.
  llama-swap does not route `/v1/apply-template` or `/apply-template` (404).
  Only the `/upstream/<model>/` path works.
- **It never costs a turn.** A capture that fails to write, or a template
  request that fails, is logged or recorded and dropped. The call's own
  result, or its typed error, is always what the turn gets. With the variable
  unset, the provider sends one request per call, with the same bytes as
  before.
- **Thought blocks are only recorded.** History is still built from
  `ChatResult`, which never carries them.

## Size

A planner call is about 14 KB of request JSON (≈3k prompt tokens), and its
rendered prompt is about as large again. A two-planner-call turn with a closer
wrote about 70 KB, so a forty-move game comes to a few megabytes. That is why
capture is off by default.
