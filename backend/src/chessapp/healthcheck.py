"""The container's health probe: is the API actually answering?

`python -m chessapp.healthcheck`, which is what the image's `HEALTHCHECK`
runs. A module rather than the `python -c` one-liner it replaces, for three
reasons the walkthrough turned up (#4): the one-liner had no timeout of its
own, so a server that accepted the connection and then said nothing was
Docker's problem to time out rather than the probe's; it printed a twenty-line
traceback into the health log for the ordinary case of "nothing is listening";
and being a string inside a Dockerfile it could not be tested.

Answering, specifically. The probe asks for `/api/state`, which is served by
the app itself and touches board truth, so a process that is up but no longer
serving fails it. It deliberately does not check the brain, the engine or the
voice stack: all three are optional by design, the app plays a full game with
every one of them down, and a health check that fails when they are would
restart a container that is working exactly as intended.
"""

import os
import sys
import urllib.error
import urllib.request

# Its own ceiling, under Docker's `--timeout`, so the probe fails as a probe —
# with a reason on stderr — rather than being killed mid-connect.
TIMEOUT_S = 2.0


def probe(url: str, timeout: float = TIMEOUT_S) -> str | None:
    """`None` when the API answered 200, else why it did not."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return f"{url} answered {response.status}"
    except urllib.error.HTTPError as exc:
        return f"{url} answered {exc.code}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"{url} unreachable: {exc}"
    return None


def health_url() -> str:
    """The port the app was told to listen on — the one the probe must ask."""
    return f"http://localhost:{os.environ.get('CHESSAPP_PORT', '8000')}/api/state"


def main() -> int:
    failure = probe(health_url())
    if failure is None:
        return 0
    print(failure, file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - the container's entry point
    sys.exit(main())
