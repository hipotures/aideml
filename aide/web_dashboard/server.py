from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import urlparse

from .state import WebDashboardState

DEFAULT_REFRESH_SECONDS = 2.0


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


def _html(refresh_seconds: float) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="color-scheme" content="dark">
  <title>AIDE live</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; margin: 0; overflow: hidden; background: #1b1d21; color: #e7eaf0; }}
    html {{ -webkit-text-size-adjust: none; text-size-adjust: none; }}
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
    .app {{ height: 100dvh; display: grid; grid-template-rows: 48px 1fr; }}
    .tabs {{ display: grid; grid-template-columns: repeat(3, 1fr); border-bottom: 1px solid #3b414b; background: #111923; }}
    .tab {{ appearance: none; border: 0; border-right: 1px solid #2b313a; margin: 0; padding: 0 10px; background: transparent; color: #9aa7b7; font: 700 14px/48px inherit; text-align: center; }}
    .tab.active {{ color: #34d399; background: #102234; box-shadow: inset 0 -3px 0 #1d9bf0; }}
    .panel {{ min-height: 0; overflow: auto; -webkit-overflow-scrolling: touch; overscroll-behavior: contain; padding: 12px 10px 24px; display: none; }}
    .panel.active {{ display: block; }}
    .headline {{ color: #111827; background: #1d9bf0; display: inline-block; padding: 0 4px; margin: 0 0 10px; font-weight: 700; font-size: 13px; line-height: 17px; }}
    .tree {{ font-size: 12px; line-height: 1.18; font-weight: 700; letter-spacing: 0; white-space: nowrap; }}
    .tree-line {{ height: 15px; display: flex; align-items: center; }}
    .prefix {{ white-space: pre; color: #d9dce2; }}
    .dot {{ width: 0.74em; height: 0.74em; flex: 0 0 auto; border-radius: 999px; margin: 0 0.28em 0 0; background: #34d399; }}
    .label {{ color: #34d399; }}
    .tree-line.best .dot {{ background: #e6c700; }}
    .tree-line.best .label {{ color: #e6c700; }}
    .tree-line.bug .dot {{ background: #ff4d63; }}
    .tree-line.bug .label {{ color: #ff4d63; }}
    .tree-line.generated .dot {{ background: #22d3ee; }}
    .tree-line.generated .label {{ color: #22d3ee; }}
    .tree-line.active .dot {{ background: #1d9bf0; }}
    .tree-line.active .label {{ color: #1d9bf0; }}
    .run-list {{ display: grid; gap: 9px; max-width: 900px; }}
    .datum {{ display: grid; grid-template-columns: minmax(88px, 30%) 1fr; gap: 10px; border-bottom: 1px solid #2b313a; padding-bottom: 8px; }}
    .datum-label {{ color: #22d3ee; font-weight: 700; }}
    .datum-value {{ color: #d6b900; overflow-wrap: anywhere; }}
    .logs {{ margin: 0; font: 700 12px/1.35 inherit; color: #a9adb6; white-space: pre-wrap; }}
    .empty {{ color: #7d8794; }}
    @media (min-width: 1000px) {{
      .tree {{ font-size: 13px; }}
      .tree-line {{ height: 16px; }}
      .panel {{ padding: 14px 12px 28px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <nav class="tabs" aria-label="Dashboard panels">
      <button class="tab active" data-tab="tree" type="button">AIDE/ Tree</button>
      <button class="tab" data-tab="run" type="button">Run data</button>
      <button class="tab" data-tab="logs" type="button">Logs</button>
    </nav>
    <main>
      <section class="panel active" data-panel="tree">
        <div class="headline" id="tree-title">Solution tree</div>
        <div class="tree" id="tree"></div>
      </section>
      <section class="panel" data-panel="run">
        <div class="run-list" id="run-data"></div>
      </section>
      <section class="panel" data-panel="logs">
        <pre class="logs" id="logs"></pre>
      </section>
    </main>
  </div>
  <script>
    let refreshMs = {int(clamp_refresh_seconds(refresh_seconds) * 1000)};
    let activeTab = "tree";

    function switchTab(name) {{
      activeTab = name;
      document.querySelectorAll(".tab").forEach((tab) => {{
        tab.classList.toggle("active", tab.dataset.tab === name);
      }});
      document.querySelectorAll(".panel").forEach((panel) => {{
        panel.classList.toggle("active", panel.dataset.panel === name);
      }});
    }}

    document.querySelectorAll(".tab").forEach((tab) => {{
      tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    }});

    function text(value) {{
      return value == null ? "" : String(value);
    }}

    function renderTree(snapshot) {{
      document.getElementById("tree-title").textContent = snapshot.tree_title || "Solution tree";
      const tree = document.getElementById("tree");
      tree.replaceChildren();
      const lines = snapshot.tree_lines || [];
      if (!lines.length) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "waiting for tree";
        tree.appendChild(empty);
        return;
      }}
      for (const line of lines) {{
        const row = document.createElement("div");
        row.className = `tree-line ${{line.kind || "ok"}}`;
        const prefix = document.createElement("span");
        prefix.className = "prefix";
        prefix.textContent = text(line.prefix);
        const dot = document.createElement("span");
        dot.className = "dot";
        const label = document.createElement("span");
        label.className = "label";
        label.textContent = text(line.label);
        row.append(prefix, dot, label);
        tree.appendChild(row);
      }}
    }}

    function renderRunData(snapshot) {{
      const list = document.getElementById("run-data");
      list.replaceChildren();
      const items = snapshot.run_data || [];
      if (!items.length) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "waiting for run data";
        list.appendChild(empty);
        return;
      }}
      for (const item of items) {{
        const row = document.createElement("div");
        row.className = "datum";
        const label = document.createElement("div");
        label.className = "datum-label";
        label.textContent = text(item.label);
        const value = document.createElement("div");
        value.className = "datum-value";
        value.textContent = text(item.value);
        row.append(label, value);
        list.appendChild(row);
      }}
    }}

    function renderLogs(snapshot) {{
      const logs = document.getElementById("logs");
      logs.textContent = (snapshot.log_lines || []).join("\\n") || "waiting for process log";
    }}

    async function refresh() {{
      try {{
        const response = await fetch("/api/snapshot", {{cache: "no-store"}});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const snapshot = await response.json();
        refreshMs = Math.max(500, Math.min(30000, Number(snapshot.refresh_seconds || refreshMs / 1000) * 1000));
        renderTree(snapshot);
        renderRunData(snapshot);
        renderLogs(snapshot);
      }} catch (error) {{
        document.getElementById("logs").textContent = `dashboard refresh failed: ${{error}}`;
      }} finally {{
        setTimeout(refresh, refreshMs);
      }}
    }}

    refresh();
  </script>
</body>
</html>
"""


class AideWebServer:
    def __init__(
        self,
        state: WebDashboardState,
        *,
        host: str = "0.0.0.0",
        port: int = 8766,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    ) -> None:
        self.state = state
        self.host = host
        self.port = port
        self.refresh_seconds = clamp_refresh_seconds(refresh_seconds)
        self._server: _ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return

        state = self.state
        refresh_seconds = self.refresh_seconds

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                path = urlparse(self.path).path
                if path in {"", "/"}:
                    self._send(200, "text/html; charset=utf-8", _html(refresh_seconds))
                    return
                if path == "/api/snapshot":
                    payload = state.get_snapshot().to_dict()
                    payload["refresh_seconds"] = refresh_seconds
                    body = json.dumps(payload, ensure_ascii=False, indent=2)
                    self._send(200, "application/json; charset=utf-8", body)
                    return
                self._send(404, "text/plain; charset=utf-8", "not found")

            def _send(self, status: int, content_type: str, body: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

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
