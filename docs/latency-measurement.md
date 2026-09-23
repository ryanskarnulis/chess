# Latency measurement

How to follow one interaction from the end of the player's speech to the end of
Glitch's reply, and how to report latency across many interactions without
misleading anyone (#317). Measure first: this note does not set latency
targets. A target gets proposed from a report like the one below, run on a
real session.

## What is recorded

Setting `CHESSAPP_TRACE_PATH` (on by default in the home deployment:
`/data/saves/turns.jsonl`) makes the app append one JSONL record per event.
Every record carries `schema` (currently 2) and `kind`:

| `kind` | Written by | When | Key fields |
| --- | --- | --- | --- |
| `serving` | server | At startup, and whenever what serves the app changes | `manifest_id`, `session`, `app.revision`, `client` (sampling, token limits, budgets), `server` (`model_path`, `model_ftype`, `build_info`, `n_ctx`, llama-swap `cmd`, `source`) |
| `turn` | server | Once per interaction that reached the pipeline | `interaction_id`, `correlation_id`, `route`, `calls[]` (phase, status, `ms`, `server_ms`, `cached_tokens`, `budget_ms`), `spans_ms`, `planning`, `serving.manifest_id` |
| `speech` | server | Once per STT or TTS round trip | `op` (`stt` or `tts`), `interaction_id`, `ms`, `status`, and the byte and character counts |
| `voice` | browser, via `POST /api/telemetry/voice` | Once the interaction settles, or is abandoned after 180 s | `interaction_id`, `correlation_id`, `origin`, `start`, `marks`, `outcome`, `censored` |

`speech` records never store audio or text. `voice` records carry the
browser's timing marks and nothing the player said.

### One interaction, joined by id

The browser mints `interaction_id` (12 hex characters) at the moment the
player's wait starts:
- **Voice:** when the VAD detects the end of speech. That is 1 s of silence
  after the last word (`END_OF_SPEECH_MS`), so every voice mark includes that
  second.
- **Push-to-talk:** at the stop tap.
- **Typed:** at submit.

The same id is sent on the transcription (`X-Interaction-Id`), in the
command body (`interaction_id`), and on the speech request. `/api/command`
answers with the turn's `correlation_id`, which the browser's report repeats.
So one voice interaction produces four joinable records: `speech/stt`,
`turn`, `speech/tts` and `voice`.

The browser records these marks, in order:

`stt_done`, `command_sent`, `first_board_update` (the first newer board the page
accepted), `engine_reply` (the first board where it is the player's move again
after one where it was not), `command_done`, `tts_requested`, `tts_ready`,
`playback_started` (the audio element's `playing` event, not the `play()`
call), `playback_ended`.

A mark the page never observed is left out, not recorded as zero. For
example, a turn whose intermediate board arrived only in the final response
has no `engine_reply`.

### Two clocks, never mixed

- **Browser marks** are `performance.now()` offsets from the start of the
  interaction.
- **Server durations** are `time.monotonic()` differences.
- The server stamps each record's `ts` when it writes it.

Nothing ever subtracts a time from one clock from a time on the other: the
clocks share no origin, and phones drift. The report computes segments within
one clock only (for example `playback_started − tts_requested`) and joins
records across clocks by id alone.

### Outcomes and censoring

- **`outcome`** says how the interaction ended for the player:
  - `ended`: the reply played to the end.
  - `error`, `interrupted`, `blocked` (autoplay refused): playback started but
    did not finish normally.
  - `no_audio`: TTS failed.
  - `timeout`: the speak deadline passed.
  - `silent`: the reply wasn't voiced.
  - `no_command`: nothing was transcribed, or no turn came back.
- **`censored`** means the page gave up at its 180 s ceiling. Its last mark is
  only a lower bound.
- **On the server,** a model call with `status: late` is also a lower bound.
  Its `ms` is how long the turn waited before giving up, and `budget_ms` is
  the limit it was held to.

## Measuring

Measure from the deployed container; it is already tracing:

```bash
docker exec chess-app-1 cat /data/saves/turns.jsonl > /tmp/turns.jsonl
cd backend
python scripts/latency_report.py /tmp/turns.jsonl --since 2026-09-24T00:00
python scripts/latency_report.py /tmp/turns.jsonl --json > /tmp/latency.json
```

For a local run, set `CHESSAPP_TRACE_PATH` and `CHESSAPP_EXPERIMENT` (the
experiment name lands on every turn and serving record), then filter with
`--experiment`. `--manifest` narrows the report to one serving configuration.

Before any run whose result is a latency, check that the GPU is idle
(`/running` on llama-swap, and `nvidia-smi`). Contention from another app
distorts the numbers unevenly, not by a constant.

### Reading the report

- **Denominators.** Every percentile is printed next to its `n`. Percentiles
  are nearest-rank, and one the sample cannot support prints as `–`: p95 needs
  at least 20 readings and p99 at least 100.
- **Failures and censored waits** are counted in their own columns and left
  out of the percentiles. A failed call has no duration worth ranking, and a
  late one is only a lower bound.
- **Conditions** are reported as separate rows:
  - `cold`: the call's wall clock exceeded the server's own prompt-plus-
    generation time by at least `--cold-gap-ms` (default 5 s). That time went
    to a model load or a queue.
  - `warm`: the server reported its timings and the gap was below that.
  - `unknown`: the server reported no timings.
  - `+contended`: the turn waited on the mutation lock, or overlapped another
    turn in time.
- **Overlap.** The observe beat runs while Stockfish computes, and speech
  happens after the turn. The report prints spans side by side and never adds
  them into a serial total.
- **Old records.** Records with a schema before 2 are counted and skipped.

## Proposing targets

After a real session, take the p95 of each player-facing segment from the
warm, uncontended rows: `speech_end→transcript`, `command→first_board`,
`command→engine_reply`, `start→first_audio`. Propose targets from those
numbers, recording the sample size and the `manifest_id` they came from. A
target set without the manifest that produced it can't be checked against
anything later.
