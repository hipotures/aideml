from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import urlparse

from .state import WebDashboardState

DEFAULT_REFRESH_SECONDS = 2.0
STATIC_DIR = Path(__file__).with_name("static")
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


def clamp_refresh_seconds(value: Any) -> float:
    try:
        refresh = float(value)
    except (TypeError, ValueError):
        return DEFAULT_REFRESH_SECONDS
    if not math.isfinite(refresh):
        return DEFAULT_REFRESH_SECONDS
    return min(30.0, max(0.5, refresh))


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class AideWebServer:
    def __init__(
        self,
        state: WebDashboardState,
        *,
        host: str = "0.0.0.0",
        port: int = 8766,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        static_dir: Path | str = STATIC_DIR,
    ) -> None:
        self.state = state
        self.host = host
        self.port = port
        self.refresh_seconds = clamp_refresh_seconds(refresh_seconds)
        self.static_dir = Path(static_dir)
        self._server: _ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return

        state = self.state
        refresh_seconds = self.refresh_seconds
        static_dir = self.static_dir

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                path = urlparse(self.path).path
                if path == "/api/snapshot":
                    payload = state.get_snapshot().to_dict()
                    payload["refresh_seconds"] = refresh_seconds
                    body = json.dumps(payload, ensure_ascii=False, indent=2).encode(
                        "utf-8"
                    )
                    self._send_bytes(200, "application/json; charset=utf-8", body)
                    return

                route = STATIC_ROUTES.get(path)
                if route is not None:
                    filename, content_type = route
                    try:
                        body = (static_dir / filename).read_bytes()
                    except OSError:
                        self._send_bytes(
                            404,
                            "text/plain; charset=utf-8",
                            b"not found",
                        )
                        return
                    self._send_bytes(200, content_type, body)
                    return

                self._send_bytes(404, "text/plain; charset=utf-8", b"not found")

            def _send_bytes(
                self,
                status: int,
                content_type: str,
                body: bytes,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = _ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="aide-web-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None
