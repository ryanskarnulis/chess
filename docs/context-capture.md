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

`docker-compose.yml` does not set the variable. To capture from the deployed
container, add it next to `CHESSAPP_TRACE_PATH` for that session only.

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
