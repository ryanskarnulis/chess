"""The container's health probe (walkthrough #4).

The old probe was a `python -c` string in the Dockerfile, which is why nothing
here existed. What it must do is narrow: say yes when `/api/state` answers 200
over the port the app was told to use, and no — quickly, and with a reason —
for every other outcome, so a container whose API has died stops being
reported as healthy.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from chessapp.healthcheck import health_url, main, probe


class _Handler(BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        self.send_response(self.status)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args: object) -> None:
        """Silence the default stderr access log."""


@pytest.fixture
def server():
    """A one-endpoint HTTP server on a free port, torn down after the test."""
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _url(httpd: HTTPServer) -> str:
    return f"http://127.0.0.1:{httpd.server_port}/api/state"


def test_a_live_api_is_healthy(server):
    assert probe(_url(server)) is None


def test_a_refused_connection_is_not_healthy(server):
    """The shape the walkthrough hit: uvicorn has shut down, the process has
    not, and the port is closed."""
    url = _url(server)
    server.shutdown()
    server.server_close()
    failure = probe(url)
    assert failure is not None
    assert "unreachable" in failure


def test_an_error_status_is_not_healthy(server, monkeypatch):
    """Up, listening, and broken is not healthy either."""
    monkeypatch.setattr(_Handler, "status", 503)
    assert probe(_url(server)) is not None


def test_the_probe_has_its_own_timeout(server, monkeypatch):
    """A server that accepts and then says nothing must fail the probe rather
    than hang until Docker kills it — the probe owns its own ceiling."""
    started = threading.Event()

    def never_answer(self: _Handler) -> None:
        started.set()
        threading.Event().wait(30)

    monkeypatch.setattr(_Handler, "do_GET", never_answer)
    failure = probe(_url(server), timeout=0.2)
    assert started.is_set()
    assert failure is not None


def test_the_probe_follows_the_configured_port(monkeypatch):
    """The image sets `CHESSAPP_PORT`; a probe hard-wired to 8000 would pass or
    fail for the wrong reason the moment that changes."""
    monkeypatch.setenv("CHESSAPP_PORT", "9123")
    assert health_url() == "http://localhost:9123/api/state"


def test_main_answers_a_live_api_with_a_zero_exit(server, monkeypatch):
    monkeypatch.setenv("CHESSAPP_PORT", str(server.server_port))
    assert main() == 0


def test_main_reports_failure_as_a_nonzero_exit(server, monkeypatch, capsys):
    """Docker reads the exit code; a human reads the health log — so the reason
    goes to stderr as one line, not the twenty-line traceback the old inline
    probe left there."""
    monkeypatch.setenv("CHESSAPP_PORT", str(server.server_port))
    server.shutdown()
    server.server_close()

    assert main() == 1
    assert capsys.readouterr().err.count("\n") == 1
