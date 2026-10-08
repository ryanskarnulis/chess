# Model profiles

What the app knows about a *model*, as opposed to what it knows about chess,
lives in one TOML file per model: `backend/src/chessapp/data/profiles/<model>.toml`,
named for the llama-swap model id (#375). Swapping in a better model means
writing its profile, not editing code.

## What a profile holds

| Key | Meaning |
|---|---|
| `[sampling]` `temperature`, `top_p`, `top_k` | Sent on every request. A key left out is not sent, so the server's default applies. |
| `thinking_kwarg` | The `chat_template_kwargs` key that toggles the thinking channel (`enable_thinking` for Gemma). `""` means the model has none, and no `chat_template_kwargs` is sent. |
| `[phases.planner\|narrator\|answer]` `temperature`, `max_tokens` | Per-phase knobs. A phase without a `temperature` samples at `sampling.temperature`. A phase left out keeps the default caps. |
| `[phases.planner]` `thinking` | `true` makes the planner reason before every call, not only after an analysis tool has answered (#298). Planner only: the narrator's thinking stays the brain's rule. Default `false`. |
| `crutches` | Guidance that exists only because this model needed it (below). |
| `quirks` | Measured standing facts about the model, as text, for the manifest and the reader. |

Loading is strict. An unknown key, phase or crutch fails to load, because a
typo would mean running a model on settings nobody chose.

## A model per phase

Each phase runs on `LLAMACPP_MODEL` unless told otherwise:

| Variable | Phase |
|---|---|
| `CHESSAPP_PLANNER_MODEL` | the planner (tool choice) |
| `CHESSAPP_NARRATOR_MODEL` | the narrator (Glitch's words) |
| `CHESSAPP_ANSWER_MODEL` | the yes/no reader in front of a destructive op |

`app.phase_models_from_env` resolves them, and the eval harness, the
frontier tier and `scripts/probe_planner.py` resolve through it too, so a
split arm such as #298's (planner on a tool-strong model, narrator on Gemma)
is a config change that the gate measures as shipped.

- Each phase uses its own model's profile for sampling, temperature and caps.
- With one model everywhere, the provider is the single `LlamaCppProvider`
  the app always built. With a split, a `PhasedProvider` routes each call by
  the phase the brain named around it (`context_capture.model_phase`).
- Both models must fit in VRAM together. If llama-swap swaps between phases,
  each turn pays a model load and the gain is gone.

The serving manifest records `client.phases` (model and profile per phase)
and every profile in full. It probes each model on its own, so its server
fields go to `server` for `LLAMACPP_MODEL` and to `servers.<model>` for any
other.

### The thinking-toggle check

The probe reads each model's `chat_template` from `/props` and records
`thinking_toggle`:

- `ok`: the template reads the profile's `thinking_kwarg`.
- `absent`: it does not, so the toggle the app sends does nothing.
- `null`: it can't tell.

Before running a new candidate, check it with:

```bash
python scripts/check_profile.py <model>          # leaves an unloaded model alone
python scripts/check_profile.py <model> --load   # loads it to look
```

## A model with no profile

It runs on the built-in default profile (`profiles.DEFAULT_PROFILE`), and a
`no_model_profile` warning is logged. The default profile has:

- no sampling opinion, so the server's own defaults apply
- the `enable_thinking` toggle
- gemma-4-12b's token caps (planner 2048, narrator 4096, answer 16)
- no crutches

Its manifest shows `profile.name: "default"`. This is the right starting
point for a trial. A model that is going to ship gets its own file with its
own `top_p`/`top_k`.

## gemma-4-12b

`gemma-4-12b.toml` is the behaviour the app shipped with before profiles
existed:

- Sampling is the fleet's, from `../agent-standard/model-profile.md`. It is
  copied, never edited here. `tests/test_profiles.py` checks for drift when
  the sibling repo is present.
- The planner runs at 0.3 (#286).
- The token caps are those of #191.

The file's comments hold the measurement behind every number.
`tests/test_profile_bytes.py` pins that the default config sends exactly the
bytes recorded from `main` before #375: every phase, with thinking both off
and on. Re-record that fixture only for a change that is meant to move the
bytes, and say so in its PR.

`CHESSAPP_PLANNER_TEMPERATURE` still overrides the planner's temperature, as
the experiment knob.

## #298 candidates

Two bake-off candidates kept their profiles, each named for its llama-swap
entry (the entry's comment in `../llama-swap/config.yaml` has the serving
sweep):

| Profile | Model | Speed on the 3060 | Thinking narrator turn (1 sample) |
|---|---|---|---|
| `gemma-4-26b-a4b` | Gemma 4 26B-A4B MoE, QAT Q4 + MTP | ~60 tok/s | 504 tokens, ~10 s |
| `qwen38-27b` | Qwen3.8 27B dense, IQ3_S | ~8 tok/s, prefill ~400 tok/s | 727 tokens, ~93 s |

`gemma-4-26b-a4b`'s planner thinks, with a 4,096-token cap (its file has
the measurement). Otherwise they are starting values, not results. Each
profile has:

- the vendor's published sampling (Qwen's non-thinking set, since one set
  serves both modes)
- the 12B's planner temperature of 0.3 and its token caps, carried over
- all five crutches, so the default arm is like-for-like with gemma-4-12b

`check_profile.py --load` reported `thinking toggle: ok` for both on
2026-10-03, and a request with the toggle off returned no reasoning. Replace
each carried value with what the bake-off measures, and say so in the file.

`qwen38-27b` serves `-c 32768`, which is enough for the 32k input budget,
but it is a chess-only entry. It runs one slot, so it serves one request at
a time.

### The planner screens (2026-10-03 and 2026-10-07)

`probe_planner.py` on its 26 dev items, every arm with all five crutches and
the planner at 0.3, gemma-4-12b as the control in the same batch:

| Planner | Thinking off | Thinking on |
|---|---|---|
| gemma-4-12b | 219/260 (84%), 0.6 s | not run |
| gemma-4-26b-a4b | 207/260 (80%), 0.9 s | 50/52, 8.5 s median |
| qwen38-27b | 152/260 (58%), 6.6 s | 51/52, 30 s median (74 s p90) |
| ornith-1.5-35b-a3b | 152/260 (58%), 2.0 s | 41/52, 11 s |
| gemma-4-12b-agentic-v2 | 168/260 (65%), 1.0 s | not run |
| granite-4.2-8b | 143/260 (55%), 2.6 s | 14/26 (one sample), 75 s |
| qwen36-35b-a3b | 133/260 (51%), 1.9 s | 39/52, 14 s |

In the same thinking-on batch, the 12B with thinking off scored 44/52.
Thinking fixed the pawn asks every thinking-off model failed. With it on, a
planner call can reach the 2048-token cap before calling a tool (1 to 4
times in 52 per model), so a thinking planner needs its own cap.

The four below the 12B were dropped on 2026-10-07. Their weights, llama-swap
entries and profiles are gone:

- **Qwen3.6, Ornith:** answer an ambiguous ask in prose instead of calling
  `ask_player`.
- **Agentic fine-tune:** sends `before_move` as the string `"7"`, which the
  schema refuses.
- **Granite:** fills optional `ask_player` fields with the string `"None"`.

## Crutches

These are the 12B-specific items from #375's 2026-09-29 comment. Each was
measured to fix a gemma-4-12b failure, so the gemma profile lists all five.
The planner's profile decides which crutches the brain gets
(`ToolContext.crutches`). A crutch the profile leaves out is cut from the
brain's copy of the offer: the description text goes, and `make_move`'s
`source` stays labelled but optional. The pick refusal drops its resubmit
script too.

The registry keeps every word, so the MCP server and the delegate wire see
no change. A bare `ToolContext` carries all five. The text crutches are
matched by exact wording (`tools._TEXT_CRUTCHES`), and `tests/test_crutches.py`
fails if a docstring is reworded so that one no longer matches. With the
gemma profile, the offer is pinned to the one recorded before the switch
existed.

| Name | Where | Why it exists (measured) |
|---|---|---|
| `move_source_required` | `make_move`'s required `source` (#351) | The 12B plays `legal_moves[0]` for "the first one" with nothing asked (20/20) |
| `undo_call_again` | `undo`'s "call this again for each further named move" | "undo X and undo Y" was read as one `undo(plies=2)` (`undo_twice_and_replace`) |
| `ask_player_examples` | The worked examples in `ask_player`'s description | Copied word for word from the #371 arm `parts_ask5` |
| `pick_resubmit_script` | The pick refusal's "resubmit … exactly as written there, still picked_by_position" | Fixes the frontier `undo_then_ambiguous_bishop` |
| `difficulty_constraint_rule` | `set_difficulty`'s "say so and ask, do not call this" | "go easy without changing the difficulty" still changed it |

The code checks behind them stay for every model: whether a list stands to
pick from (#351/#411), and the step arithmetic (#338).

For #298, run each candidate both with and without the crutch set. A model
that holds the gate without them ships without them. `CHESSAPP_CRUTCHES`
overrides the profile for one run, in the app, the gate, the frontier tier
and the planner probe alike:

```bash
CHESSAPP_CRUTCHES=none CHESSAPP_AGENT_EVALS=1 pytest tests/test_agent_evals.py
CHESSAPP_CRUTCHES=undo_call_again,move_source_required chessapp
```

`all` gives every crutch, and an unknown name refuses to start. The eval and
frontier headers record the `crutches` a run used.
