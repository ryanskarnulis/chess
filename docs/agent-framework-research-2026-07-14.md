# Open-source agent framework research — 2026-07-14

Companion to `agent-harness-audit-2026-07-14.md`. Question asked: *is there a
better way to build this app's agent — any open-source solutions worth trying?*

## What "better" has to mean here

Any candidate is graded against the constraints this app actually has, not a
generic feature list:

1. **Local-first** — gemma-4-12b on llama.cpp behind llama-swap, 12 GB VRAM,
   OpenAI-compatible wire. No cloud calls in the loop.
2. **Deterministic code owns truth** — the LLM must be structurally unable to
   corrupt game state. Anything that has the model execute code is disqualified.
3. **The fleet standard** — the loop/registry shape is shared with
   project-command-center and conductor (`../agent-standard/STANDARD.md`).
   A chess-only framework switch forks the house standard; that's a fleet
   decision, not a chess decision.
4. **The swappable-brain seam** — everything model-specific already lives
   behind `Brain.get_agent_response`. This cuts both ways: it makes any
   framework *pilot* cheap (implement it as another `Brain`), and it means the
   framework could only ever replace ~700 lines (`brain.py`, `llama_brain.py`,
   `provider.py`) — the most commodity code in the app.

## The finding that reframes the question

**The reliability layer agent frameworks sell is already in our stack.**
llama.cpp's server constrains tool-call output with grammars when running with
`--jinja` ([function-calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md),
[PR #9639](https://github.com/ggml-org/llama.cpp/pull/9639)) — native lazy
grammars for supported templates, generic JSON-schema-constrained output
otherwise. Our shared server runs `--jinja` (`../llama-swap/config.yaml`), which
is why the eval baseline records **zero schema corrections**: malformed tool
JSON is mechanically near-impossible at the sampler level.

So the failures we've actually had were never formatting failures — they were
*decision* failures (wrong tool, no tool, fabricated prose), and every fix that
worked moved a decision out of the model into code (`fastparse`, `_gate`, the
honesty guard, signature-derived defaults, fresh-fact injection). **No framework
below owns any of those for us.** That's the core of the recommendation.

## Candidates evaluated

| Framework | Verdict | One-line reason |
| --- | --- | --- |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | **Best general-purpose fit — pilot-worthy, not urgent** | Same philosophy we already follow, plus retries/evals/tracing for free |
| [DSPy](https://futureagi.com/blog/dspy-optimizers-explained/) (GEPA/MIPROv2) | **Genuinely novel value — recommend an experiment** | Doesn't replace the loop; *optimizes our prompt* against our own evals |
| [smolagents](https://github.com/huggingface/smolagents) | No | Flagship CodeAgent violates the core invariant (model writes/executes Python) |
| [LangGraph](https://alicelabs.ai/en/insights/best-ai-agent-frameworks-2026) | No | Graph orchestration for problems we don't have; heavy dependency |
| [CrewAI](https://gurusup.com/blog/best-multi-agent-frameworks-2026) / AG2 | No | Multi-agent role-play machinery; chess is one agent, four routes |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | No (marginal) | Lightweight and provider-agnostic via LiteLLM, but adds a LiteLLM layer to get what `OpenAIChatModel` already gives Pydantic AI; guardrails/handoffs solve problems we solved in code |
| [Outlines](https://pydantic.dev/docs/ai/models/overview/) / Instructor | Already have it | Structured generation is llama-server's `--jinja` grammar, already on |
| [llama-cpp-agent](https://llama-cpp-agent.readthedocs.io/en/latest/get-started/) | No | Niche, low-activity; nothing our provider doesn't do |
| Letta (MemGPT) | No | Server-owned memory agent; our state is deterministic and injected fresh — the opposite design, on purpose |

### Pydantic AI — the real candidate, examined honestly

It is the framework closest to what we already built, which is both the
argument for it and against it:

- **Same tool philosophy:** tools from typed function signatures + docstrings,
  Pydantic-validated, errors fed back to the model for retry — that is our
  registry and correction budget, maintained by someone else. Fleet standard §1
  literally uses the same FastMCP derivation.
- **What we'd gain:** `ModelRetry`/output validators (our correction loop),
  `UsageLimits` (our `max_iterations`), [pydantic-evals](https://pydantic.dev/docs/ai/overview/)
  (our harness is homegrown), Logfire tracing (our JSONL tracer), durable
  execution (don't need it), maintained model quirk-handling.
- **What we'd keep building ourselves regardless:** the fast path, the
  confirmation gate, the honesty guard, the transcript, `narrate`, the
  board-state injection, the route pipeline — i.e. everything that has ever
  fixed a real failure.
- **The wart that matters for us:** generic OpenAI-compatible endpoints are a
  second-class citizen — an [open issue asks for a llama.cpp provider](https://github.com/pydantic/pydantic-ai/issues/4878)
  precisely because `OpenAIChatModel` has odd behaviors against llama-server
  (e.g. thinking toggles inferred from the model *name*). Our 300-line
  `provider.py` handles Gemma's `reasoning_content` and
  `chat_template_kwargs.enable_thinking` exactly; Pydantic AI would need
  customization to match, in the least-supported corner of the framework.
- **Net:** we'd trade ~700 lines of well-tested code we fully understand for a
  large dependency whose weakest area is exactly our backend, to get features
  we already have. Worth a pilot only if loop maintenance starts costing real
  time — and the `Brain` seam means a `PydanticAIBrain` can be built and A/B'd
  through the existing eval harness in a day or two without touching the app.

### DSPy — the one that attacks a problem we actually have

DSPy doesn't replace the agent loop; it **compiles prompts**. Its optimizers
(MIPROv2, and GEPA — [ICLR 2026, reflective prompt evolution beating MIPROv2 by
~13% with 35× fewer rollouts](https://www.morphllm.com/gepa-prompt-optimization),
evaluated on models as small as Qwen3 8B) search for instructions that maximize
a metric *you define* — and we already own the two hard parts of that setup:

- **A metric:** the eval harness's pass-rate scenarios are executable
  pass/fail judgments (`my_mistake_is_mine`, `play_as_black`,
  `long_capture[poisoned]`, `hints_off_no_advice`…).
- **A trainset:** every traced misfire carries its own `fen_before` +
  `utterance` by design.

The harness audit found ~1,500 tokens of hand-written system prompt with known
dead weight and instruction competition. A bounded DSPy experiment —
optimize the tool-decision instruction against our own scenarios, on our own
GPU (compile time 1–4 h at local speeds) — could plausibly produce a *shorter
and better* prompt than hand-tuning, including possibly cracking the one
standing xfail (the hints leak) if we decide to keep any of it prompt-side.
Cost if it fails: a branch and a few GPU-hours. It slots in without any
architecture change: the compiled prompt is just a string handed to
`system_prompt_provider`.

### Why the orchestration frameworks all miss

LangGraph, CrewAI, AutoGen/AG2 (and Haystack/LlamaIndex agent stacks) solve
*coordination*: many agents, branching long-running graphs, replay, human
handoffs. This app is one agent with four routes, three of them deterministic,
over a single game session — the pipeline in `api._run_command` *is* the graph
and it's ~100 lines we can read. Adopting one imports a large dependency tree
into a local-first app to restate existing code in framework vocabulary.
smolagents deserves a specific note: its signature CodeAgent has the model
write Python that executes — on this architecture that is not a feature but the
exact thing the whole design exists to prevent; its ToolCallingAgent subset is
just a plainer version of the loop we have.

## Recommendation

1. **Don't replace the harness.** It is small, tested, fleet-standard, and the
   observed failures were never in the part a framework would replace. The
   audit's diet (prompt, schemas, narrate, window) is the highest-value work
   available on agent quality.
2. **Run the DSPy/GEPA prompt-compilation experiment** after the audit's
   observability fix lands (so results are measurable). It's the only candidate
   offering something we can't already do, it uses our eval harness as-is, and
   it's cheap to abandon.
3. **Keep Pydantic AI on the shelf, warm.** If loop maintenance ever becomes a
   cost, pilot it as a `PydanticAIBrain` behind the existing seam and let the
   eval baseline decide. Revisit if the llama.cpp provider issue gets first-class
   support upstream. As a fleet question, raise it against
   `agent-standard/STANDARD.md`, not here.

## Sources

- [llama.cpp function-calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) · [tool-call PR #9639 (lazy grammars)](https://github.com/ggml-org/llama.cpp/pull/9639) · [GBNF grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) · [docs overview](https://pydantic.dev/docs/ai/overview/) · [model providers](https://pydantic.dev/docs/ai/models/overview/) · [llama.cpp provider issue #4878](https://github.com/pydantic/pydantic-ai/issues/4878)
- [smolagents](https://github.com/huggingface/smolagents) · [docs](https://huggingface.co/docs/smolagents/index)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [DSPy optimizers explained (2026)](https://futureagi.com/blog/dspy-optimizers-explained/) · [GEPA overview](https://www.morphllm.com/gepa-prompt-optimization) · [MIPROv2 guide](https://deepeval.com/docs/prompt-optimization-miprov2)
- Framework landscape surveys: [firecrawl.dev](https://www.firecrawl.dev/blog/best-open-source-agent-frameworks) · [alicelabs.ai](https://alicelabs.ai/en/insights/best-ai-agent-frameworks-2026) · [gurusup.com](https://gurusup.com/blog/best-multi-agent-frameworks-2026) · [uvik.net](https://uvik.net/blog/python-ai-agent-frameworks/)
- [Instructor + llama-cpp-python structured outputs](https://python.useinstructor.com/integrations/llama-cpp-python/) · [llama-cpp-agent](https://llama-cpp-agent.readthedocs.io/en/latest/get-started/)
