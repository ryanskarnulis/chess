"""App-assembly entrypoint: where the pieces finally meet.

`create_app` (in `chessapp.api`) takes a ready `ToolContext` and `Brain`;
this module is what builds them and wires them together into a runnable app.
One shared `ToolContext` holds the session, settings, and optional engine, so
the tools the brain calls and the state the API serves are the same truth.

Live prompt settings land here: the brain is built with a
`system_prompt_provider` and a `planner_prompt_provider` that both read
`ctx.settings`, and `set_verbosity` mutates exactly those settings — so a
change takes effect on the very next command, no rebuild. Verbosity only
reaches the narrator (the planner has no words to lengthen); the planner's
contract is static, but the provider seam stays, because the next live-tuned
prompt input will want the same wire.

`main()` reads config from the environment and runs the app under uvicorn.
Basic gameplay works with the LLM off (no brain / no `/api/command`) and with
Stockfish off (no engine); both are optional here.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from chessapp.api import create_app
from chessapp.brain import Brain
from chessapp.coordinator import TurnCoordinator
from chessapp.engine import EnginePlayer
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.personality import PLANNER_PROMPT, system_prompt_for
from chessapp.progress import ProgressReporter
from chessapp.provider import ChatProvider
from chessapp.tools import ToolContext, brain_tool_exclusions, build_registry
from chessapp.trace import JsonlTracer, Tracer
from chessapp.voice import (
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    SpeechClient,
    create_speech_client,
)

DEFAULT_LLAMA_BASE_URL = "http://127.0.0.1:8200/v1"
DEFAULT_MODEL = "gemma-4-12b"


def build_app(
    *,
    llama_base_url: str = DEFAULT_LLAMA_BASE_URL,
    model: str = DEFAULT_MODEL,
    engine: EnginePlayer | None = None,
    save_dir: Path | None = None,
    brain: Brain | None = None,
    agent_enabled: bool = True,
    provider: ChatProvider | None = None,
    speech: SpeechClient | None = None,
    static_dir: Path | None = None,
    tracer: Tracer | None = None,
    planner_temperature: float | None = None,
) -> FastAPI:
    """Assemble the full app around one shared `ToolContext`.

    Pass a `brain` to inject one (tests, or an alternate backend); otherwise a
    `LlamaBrain` is built against `llama_base_url`, wired to resolve both of its
    prompts from `ctx.settings` on every command so `set_verbosity` takes
    effect live. `provider` injects a fake `ChatProvider` into that default
    brain without a real llama-server. `planner_temperature` samples the
    planner phase apart from the narrator (None: both on the provider's
    default).

    `agent_enabled=False` is **direct mode**: no brain is constructed at all, so
    `/api/command` 503s and the board plays the deterministic exchange. It needs
    its own parameter because `brain=None` already means "construct one" — the
    LLM-off invariant was otherwise unreachable from configuration, every branch
    behind it written and correct but dead. Injecting a brain *and* disabling
    the agent is incoherent and raises rather than picking a quiet winner.
    """
    if brain is not None and not agent_enabled:
        raise ValueError("agent_enabled=False cannot be combined with a brain")
    ctx = ToolContext(session=GameSession(), engine=engine, save_dir=save_dir)
    if engine is not None:
        # Stockfish's own default is full strength; make the engine play at
        # the settings default so strength and reported settings agree.
        if ctx.settings.tier is not None:
            engine.set_tier(ctx.settings.tier)
        elif ctx.settings.skill_level is not None:
            engine.set_skill_level(ctx.settings.skill_level)
        elif ctx.settings.elo is not None:
            engine.set_elo(ctx.settings.elo)
    # One turn coordinator for the whole app, shared by the tool path and the
    # board endpoints: the turn sequence (and the engine's reply inside it) is
    # one machine's, so a dragged move and a typed move cannot disagree about
    # which turn the game is on.
    coordinator = TurnCoordinator(ctx)
    # Live progress (audit item 19). Built here, before the things that report
    # through it, because the *brain* is one of them and only assembly can reach
    # it: the coordinator and the registry are wired by `create_app`, but a
    # brain is constructed complete. `on_narrating` is why this is not merely
    # decoration — the narrator turn *is* the coordinator's observation beat,
    # and the brain, holding no coordinator by design, can only report that it
    # started; marking the phase is this wire's job.
    progress = ProgressReporter(on_narrating=coordinator.mark_observation)
    # One registry for the whole app: the brain dispatches its tool calls
    # through it and the pipeline runs the fast path through it, so the tools
    # the agent is offered are exactly the tools that execute.
    # `atomic_exchange=False`: `make_move` applies the player's move and stops,
    # because the command pipeline owns the beats that follow — Glitch's reaction
    # to the verified move, then collecting the engine's reply. (The MCP server,
    # which has no pipeline, keeps the atomic default.)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)

    def offered_tools() -> list[dict[str, Any]]:
        """What the brain may call this command, resolved live off the shared
        context by `brain_tool_exclusions` — the pure reads always, plus
        whatever the app already knows the answer to (no claimable draw).
        Callers with no such injection (MCP, the delegate wire, /api/game/hint)
        keep the full registry.
        """
        return registry.definitions(exclude=brain_tool_exclusions(ctx))

    if brain is None and agent_enabled:
        brain = create_llama_brain(
            base_url=llama_base_url,
            model=model,
            dispatcher=registry,
            # The brain dispatches through the registry but is *offered* less
            # (`offered_tools` above), re-resolved per command so the offer
            # tracks the live board (a draw becoming claimable).
            tool_definitions=offered_tools,
            # The narrator's prompt (personality + verbosity), re-resolved per
            # command off the live settings; the planner's compact tool
            # contract is static but rides the same provider seam.
            system_prompt_provider=lambda: system_prompt_for(ctx.settings.verbosity),
            planner_prompt_provider=lambda: PLANNER_PROMPT,
            planner_temperature=planner_temperature,
            provider=provider,
            # The brain's own two phases, live (`progress.py`). Nothing else
            # can see inside `get_agent_response`, and the narrator half of it
            # is the observe beat.
            on_phase=progress.brain,
        )
    return create_app(
        ctx,
        brain=brain,
        speech=speech,
        static_dir=static_dir,
        registry=registry,
        tracer=tracer,
        coordinator=coordinator,
        progress=progress,
    )


def _engine_from_env() -> EnginePlayer | None:
    path = os.environ.get("CHESSAPP_STOCKFISH")
    return EnginePlayer(path=path) if path else None


def _speech_from_env() -> SpeechClient | None:
    """Voice is optional exactly like the brain: no SPEECH_BASE_URL means no
    speech client, and the voice endpoints answer 503. The env var names are
    the fleet voice contract's (../agent-standard/voice.md)."""
    url = os.environ.get("SPEECH_BASE_URL")
    if not url:
        return None
    return create_speech_client(
        base_url=url,
        # TTS on its own server (the custom-voice Kokoro container); unset
        # means the Speaches backend serves both, exactly as before.
        tts_base_url=os.environ.get("TTS_BASE_URL"),
        stt_model=os.environ.get("STT_MODEL", DEFAULT_STT_MODEL),
        tts_model=os.environ.get("TTS_MODEL", DEFAULT_TTS_MODEL),
        tts_voice=os.environ.get("TTS_VOICE", DEFAULT_TTS_VOICE),
    )


def _agent_enabled_from_env() -> bool:
    """`CHESSAPP_AGENT=off` runs the app in direct mode — Stockfish only.

    App-local, not the fleet's `LLAMACPP_BASE_URL`: that name is the workspace
    agent standard's and other apps parse it as a URL, so an unset variable has
    to keep meaning "use the default endpoint" rather than doubling as an off
    switch. Agent-on is the no-config default for the same reason — a missing
    variable must never silently change the operating mode.

    Strict on purpose. The bug this switch exists to fix is the app
    *advertising an agent it hasn't got*, and a permissive parse reproduces it
    from a typo (`of` reading as "on"), which is the one failure that must not
    be quiet. So an unrecognized value refuses to start, at assembly, where
    someone is watching.
    """
    value = os.environ.get("CHESSAPP_AGENT")
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in ("on", "off"):
        return normalized == "on"
    raise ValueError(f"CHESSAPP_AGENT must be 'on' or 'off', not {value!r}")


def _tracer_from_env() -> Tracer | None:
    """Turn tracing on by pointing `CHESSAPP_TRACE_PATH` at a JSONL file.

    Off by default: it is a review tool for a session you are debugging, not
    something a normal game should be paying for or quietly accumulating.
    """
    path = os.environ.get("CHESSAPP_TRACE_PATH")
    return JsonlTracer(Path(path)) if path else None


def _planner_temperature_from_env() -> float | None:
    """The planner phase's sampling temperature, if one is pinned.

    Unset (the default) means the provider's own temperature, so nothing
    changes until the number has been measured — the split's sampling
    experiment is run through this knob, not by editing a constant.
    """
    value = os.environ.get("CHESSAPP_PLANNER_TEMPERATURE")
    return float(value) if value else None


def build_app_from_env(engine: EnginePlayer | None = None) -> FastAPI:
    """`build_app` configured from environment variables (for `main`/ASGI).

    `engine` is for a caller that wants to *own* the engine's lifetime — which
    is `serve`, because the process cannot exit while one is open. Omitted, the
    engine is built from the environment as before and nothing closes it, which
    is right for an ASGI server that outlives no process of ours.
    """
    save_dir_env = os.environ.get("CHESSAPP_SAVE_DIR")
    static_dir_env = os.environ.get("CHESSAPP_STATIC_DIR")
    return build_app(
        llama_base_url=os.environ.get("LLAMACPP_BASE_URL", DEFAULT_LLAMA_BASE_URL),
        model=os.environ.get("LLAMACPP_MODEL", DEFAULT_MODEL),
        agent_enabled=_agent_enabled_from_env(),
        engine=engine if engine is not None else _engine_from_env(),
        save_dir=Path(save_dir_env) if save_dir_env else None,
        speech=_speech_from_env(),
        static_dir=Path(static_dir_env) if static_dir_env else None,
        tracer=_tracer_from_env(),
        planner_temperature=_planner_temperature_from_env(),
    )


def serve(runner: Callable[[FastAPI, str, int], None]) -> None:
    """Build the app from the environment, serve it, and close the engine.

    The close is the whole reason this is a function. Stockfish is driven over
    UCI from a *non-daemon* thread python-chess starts per engine
    (`chess.engine.run_in_background`), so an open engine keeps the interpreter
    alive after the server has stopped: `docker compose stop` logged
    "Application shutdown complete", then sat out the full ten-second grace
    period and died on SIGKILL (exit 137), every deploy (walkthrough #4). The
    app was gone for those ten seconds and the container was still there.

    Whoever builds the engine closes it, and that is this function — the
    lifespan in `create_app` deliberately does not, because an injected engine
    belongs to its caller (the eval harness shares one across every scenario
    and every app it builds).

    `runner` is the server, injected: uvicorn in `main`, a stub in the test
    that pins the close still happens when serving raises.
    """
    engine = _engine_from_env()
    app = build_app_from_env(engine=engine)
    try:
        runner(
            app,
            os.environ.get("CHESSAPP_HOST", "0.0.0.0"),
            int(os.environ.get("CHESSAPP_PORT", "8000")),
        )
    finally:
        if engine is not None:
            engine.close()


def main() -> None:  # pragma: no cover - thin runtime shim over uvicorn
    import uvicorn

    serve(lambda app, host, port: uvicorn.run(app, host=host, port=port))


if __name__ == "__main__":  # pragma: no cover
    main()
