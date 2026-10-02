"""Check a model's profile against the server that runs it (#375).

Prints the profile the app would use for each model, then what the server
says: whether the model's chat template reads the profile's thinking key (a
key it ignores means the toggle the app sends does nothing), and the server's
own sampling beside the profile's. Run it before a bake-off arm (#298):

    python scripts/check_profile.py gemma-4-12b
    python scripts/check_profile.py qwen36-35b-a3b --load

A model llama-swap has not loaded is reported as such and left alone: asking
for its props would load it, a ~100 s cold start on the shared card. `--load`
asks anyway. Exit status 1 when a template ignores the thinking key.
"""

import argparse
import json
import sys

import httpx

from chessapp.profiles import load_profile
from chessapp.serving import THINKING_ABSENT, read_props, thinking_toggle

DEFAULT_BASE_URL = "http://127.0.0.1:8200/v1"


def check(client: httpx.Client, root: str, model: str, load: bool) -> bool:
    """Print one model's report; False when its template ignores the key."""
    profile = load_profile(model)
    print(f"== {model}: profile {profile.name} ({profile.source})")
    print(json.dumps(profile.describe(), indent=2))
    running = client.get(f"{root}/running")
    props_url = f"{root}/upstream/{model}/props"
    if running.status_code == 404:
        props_url = f"{root}/props"
    elif running.status_code == 200:
        states = {
            entry.get("model"): entry.get("state")
            for entry in running.json().get("running", [])
            if isinstance(entry, dict)
        }
        if states.get(model) != "ready" and not load:
            print(f"server: {model} is not loaded; pass --load to ask anyway")
            return True
    props = client.get(props_url, timeout=300 if load else 10)
    if props.status_code != 200:
        print(f"server: props answered HTTP {props.status_code}")
        return True
    body = props.json()
    toggle = thinking_toggle(body, profile.thinking_kwarg)
    print(f"thinking toggle {profile.thinking_kwarg!r} in template: {toggle}")
    print(f"server sampling: {read_props(body)['sampling']}")
    print(f"profile sampling: {dict(profile.sampling) or 'server default'}")
    return toggle != THINKING_ABSENT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args(argv)
    root = args.base_url.rstrip("/")
    root = root[: -len("/v1")] if root.endswith("/v1") else root
    with httpx.Client(timeout=10) as client:
        ok = [check(client, root, model, args.load) for model in args.models]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
