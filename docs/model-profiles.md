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

## Crutches

These are the 12B-specific items from #375's 2026-09-29 comment. Each was
measured to fix a gemma-4-12b failure, so the gemma profile lists all five.
The list is recorded in the profile and the serving manifest now. The tool
offer starts reading it in #375's crutch PR, and until then every model gets
all five.

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
that holds the gate without them ships without them.
