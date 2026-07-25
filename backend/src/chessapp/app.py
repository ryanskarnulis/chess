"""App-assembly entrypoint: where the pieces finally meet.

`create_app` (in `chessapp.api`) takes a ready `ToolContext` and `Brain`;
this module is what builds them and wires them together into a runnable app.
One shared `ToolContext` holds the session, settings, and optional engine, so
the tools the brain calls and the state the API serves are the same truth.

Live prompt settings land here: the brain is built with a
`system_prompt_provider` and a `planner_prompt_provider` that both read
`ctx.settings`, and `set_verbosity` / `set_hints_mode` mutate exactly those
settings — so a change takes effect on the very next command, no rebuild.
Verbosity only reaches the narrator (the planner has no words to lengthen);
hints reach both, as tone on one side and tool permission on the other.

`main()` reads config from the environment and runs the app under uvicorn.
Basic gameplay works with the LLM off (no brain / no `/api/command`) and with
Stockfish off (no engine); both are optional here.
"""

import os
from pathlib import Path

from fastapi import FastAPI

from chessapp.api import create_app
from chessapp.brain import Brain
from chessapp.coordinator import TurnCoordinator
from chessapp.engine import EnginePlayer
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.personality import planner_prompt_for, system_prompt_for
from chessapp.provider import ChatProvider
from chessapp.tools import BOARD_STATE_TOOLS, ToolContext, build_registry
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
    provider: ChatProvider | None = None,
    speech: SpeechClient | None = None,
    static_dir: Path | None = None,
    tracer: Tracer | None = None,
    planner_temperature: float | None = None,
) -> FastAPI:
    """Assemble the full app around one shared `ToolContext`.

    Pass a `brain` to inject one (tests, or an alternate backend); otherwise a
    `LlamaBrain` is built against `llama_base_url`, wired to resolve both of its
    prompts from `ctx.settings` on every command so `set_verbosity` and
    `set_hints_mode` take effect live. `provider` injects a fake `ChatProvider`
    into that default brain without a real llama-server. `planner_temperature`
    samples the planner phase apart from the narrator (None: both on the
    provider's default).
    """
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
    # One registry for the whole app: the brain dispatches its tool calls
    # through it and the pipeline runs the fast path through it, so the tools
    # the agent is offered are exactly the tools that execute.
    # `atomic_exchange=False`: `make_move` applies the player's move and stops,
    # because the command pipeline owns the beats that follow — Glitch's reaction
    # to the verified move, then collecting the engine's reply. (The MCP server,
    # which has no pipeline, keeps the atomic default.)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    if brain is None:
        brain = create_llama_brain(
            base_url=llama_base_url,
            model=model,
            dispatcher=registry,
            # The brain dispatches through the whole registry but is *offered*
            # less: it gets the board state in its prompt every turn, so the
            # tools that only read that state back can teach it nothing and
            # would cost it a round trip out of four. Callers with no such
            # injection (MCP, the delegate wire) still see the full list.
            tool_definitions=registry.definitions(exclude=BOARD_STATE_TOOLS),
            # The narrator's prompt (personality + verbosity + hints tone) and
            # the planner's (the compact tool contract, plus the hints
            # permission), each re-resolved per command off the live settings.
            system_prompt_provider=lambda: system_prompt_for(
                ctx.settings.verbosity,
                ctx.settings.hints_mode,
            ),
            planner_prompt_provider=lambda: planner_prompt_for(ctx.settings.hints_mode),
            planner_temperature=planner_temperature,
            provider=provider,
        )
    return create_app(
        ctx,
        brain=brain,
        speech=speech,
        static_dir=static_dir,
        registry=registry,
        tracer=tracer,
        coordinator=coordinator,
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


def build_app_from_env() -> FastAPI:
    """`build_app` configured from environment variables (for `main`/ASGI)."""
    save_dir_env = os.environ.get("CHESSAPP_SAVE_DIR")
    static_dir_env = os.environ.get("CHESSAPP_STATIC_DIR")
    return build_app(
        llama_base_url=os.environ.get("LLAMACPP_BASE_URL", DEFAULT_LLAMA_BASE_URL),
        model=os.environ.get("LLAMACPP_MODEL", DEFAULT_MODEL),
        engine=_engine_from_env(),
        save_dir=Path(save_dir_env) if save_dir_env else None,
        speech=_speech_from_env(),
        static_dir=Path(static_dir_env) if static_dir_env else None,
        tracer=_tracer_from_env(),
        planner_temperature=_planner_temperature_from_env(),
    )


def main() -> None:  # pragma: no cover - thin runtime shim over uvicorn
    import uvicorn

    uvicorn.run(
        build_app_from_env(),
        host=os.environ.get("CHESSAPP_HOST", "0.0.0.0"),
        port=int(os.environ.get("CHESSAPP_PORT", "8000")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
