from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from aide.journal import Journal, Node
from aide.utils.metric import MetricValue
from aide.web_dashboard.server import AideWebServer, clamp_refresh_seconds
from aide.web_dashboard.state import (
    WebDashboardSnapshot,
    WebDashboardState,
    WebRunDatum,
    WebTreeLine,
)
from aide.web_dashboard.tree import build_web_tree_lines


def _scored_node(score: float, *, parent: Node | None = None) -> Node:
    node = Node(code="print('ok')", plan="ok", parent=parent)
    node.status = "ok"
    node.is_buggy = False
    node.metric = MetricValue(score, maximize=True)
    return node


def test_web_tree_lines_use_compact_text_prefixes_without_horizontal_rule():
    journal = Journal()
    root_a = _scored_node(0.91)
    root_b = _scored_node(0.92)
    child_a = _scored_node(0.93, parent=root_b)
    child_b = _scored_node(0.94, parent=root_b)
    root_c = _scored_node(0.95)
    for node in [root_a, root_b, child_a, child_b, root_c]:
        journal.append(node)

    lines = build_web_tree_lines(journal)

    assert [(line.prefix, line.label) for line in lines] == [
        ("├", "0.91000·0"),
        ("├", "0.92000·1"),
        ("│├", "0.93000·2"),
        ("│└", "0.94000·3"),
        ("└", "0.95000·4"),
    ]
    assert all("─" not in line.prefix for line in lines)


def test_web_server_serves_html_snapshot_and_404():
    state = WebDashboardState()
    state.update(
        WebDashboardSnapshot(
            run_id="2-delicate-cherubic-crane",
            refresh_seconds=1.5,
            tree_title="Solution tree",
            tree_lines=[WebTreeLine(prefix="├", label="0.96104·0", kind="ok")],
            run_data=[WebRunDatum(label="Progress", value="39/1500")],
            log_lines=["Fold 1 OOF balanced accuracy: 0.965367"],
        )
    )
    server = AideWebServer(state, host="127.0.0.1", port=0, refresh_seconds=1.5)
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/", timeout=2) as response:
            html = response.read().decode("utf-8")
        assert "AIDE/ Tree" in html
        assert "data-tab=\"tree\"" in html

        with urlopen(
            f"http://127.0.0.1:{server.port}/api/snapshot",
            timeout=2,
        ) as response:
            payload = response.read().decode("utf-8")
        assert "\"run_id\": \"2-delicate-cherubic-crane\"" in payload
        assert "\"prefix\": \"├\"" in payload

        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"http://127.0.0.1:{server.port}/missing", timeout=2)
        assert exc_info.value.code == 404
    finally:
        server.stop()


def test_refresh_seconds_are_clamped_for_browser_polling():
    assert clamp_refresh_seconds(0.1) == 0.5
    assert clamp_refresh_seconds(2.0) == 2.0
    assert clamp_refresh_seconds(60.0) == 30.0
    assert clamp_refresh_seconds("bad") == 2.0
