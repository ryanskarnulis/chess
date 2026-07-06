"""App-assembly entrypoint: where the pieces finally meet.

`create_app` (in `chessapp.api`) takes a ready `ToolContext` and `Brain`;
this module is what builds them and wires them together into a runnable app.
One shared `ToolContext` holds the session, settings, and optional engine, so
the tools the brain calls and the state the API serves are the same truth.

Live personality switching lands here: the brain is built with a
`system_prompt_provider` that reads `ctx.settings.personality`, and
`set_personality` mutates exactly that setting — so a personality change takes
effect on the very next command, no rebuild.

`main()` reads config from the environment and runs the app under uvicorn.
Basic gameplay works with the LLM off (no brain / no `/api/command`) and with
Stockfish off (no engine); both are optional here.
"""

import os
from pathlib import Path

from fastapi import FastAPI

from chessapp.api import create_app
from chessapp.brain import Brain
from chessapp.engine import EnginePlayer
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.personality import system_prompt_for
from chessapp.tools import ToolContext, build_registry

DEFAULT_LLAMA_BASE_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "gemma"


def build_app(
    *,
    llama_base_url: str = DEFAULT_LLAMA_BASE_URL,
    model: str = DEFAULT_MODEL,
    engine: EnginePlayer | None = None,
    save_dir: Path | None = None,
    brain: Brain | None = None,
    openai_client: object | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Assemble the full app around one shared `ToolContext`.

    Pass a `brain` to inject one (tests, or an alternate backend); otherwise a
    `LlamaBrain` is built against `llama_base_url`, wired to resolve its system
    prompt from `ctx.settings.personality` on every command so
    `set_personality` switches personality live. `openai_client` injects a fake
    OpenAI client into that default brain without a real llama-server.
    """
    ctx = ToolContext(session=GameSession(), engine=engine, save_dir=save_dir)
    if brain is None:
        brain = create_llama_brain(
            base_url=llama_base_url,
            model=model,
            tool_definitions=build_registry(ctx).definitions(),
            system_prompt_provider=lambda: system_prompt_for(ctx.settings.personality),
            client=openai_client,
        )
    return create_app(ctx, brain=brain, static_dir=static_dir)


def _engine_from_env() -> EnginePlayer | None:
    path = os.environ.get("CHESSAPP_STOCKFISH")
    return EnginePlayer(path=path) if path else None


def build_app_from_env() -> FastAPI:
    """`build_app` configured from environment variables (for `main`/ASGI)."""
    save_dir_env = os.environ.get("CHESSAPP_SAVE_DIR")
    static_dir_env = os.environ.get("CHESSAPP_STATIC_DIR")
    return build_app(
        llama_base_url=os.environ.get("CHESSAPP_LLAMA_URL", DEFAULT_LLAMA_BASE_URL),
        model=os.environ.get("CHESSAPP_MODEL", DEFAULT_MODEL),
        engine=_engine_from_env(),
        save_dir=Path(save_dir_env) if save_dir_env else None,
        static_dir=Path(static_dir_env) if static_dir_env else None,
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
