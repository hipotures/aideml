from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from aide.journal import Journal, Node
from aide.utils.metric import MetricValue
from aide.web_dashboard.server import AideWebServer, STATIC_DIR, clamp_refresh_seconds
from aide.web_dashboard.state import (
    WebDashboardSnapshot,
    WebDashboardState,
    WebRunDatum,
    WebRunSection,
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
            run_sections=[
                WebRunSection(
                    title="Agent",
                    items=[WebRunDatum(label="mode", value="legacy")],
                )
            ],
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
        assert "\"title\": \"Agent\"" in payload

        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"http://127.0.0.1:{server.port}/missing", timeout=2)
        assert exc_info.value.code == 404
    finally:
        server.stop()


def test_web_server_reads_static_html_from_disk_on_each_request(tmp_path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("first version", encoding="utf-8")

    server = AideWebServer(
        WebDashboardState(),
        host="127.0.0.1",
        port=0,
        static_dir=static_dir,
    )
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/", timeout=2) as response:
            first = response.read().decode("utf-8")
        (static_dir / "index.html").write_text("second version", encoding="utf-8")
        with urlopen(f"http://127.0.0.1:{server.port}/", timeout=2) as response:
            second = response.read().decode("utf-8")
    finally:
        server.stop()

    assert first == "first version"
    assert second == "second version"


def test_refresh_seconds_are_clamped_for_browser_polling():
    assert clamp_refresh_seconds(0.1) == 0.5
    assert clamp_refresh_seconds(2.0) == 2.0
    assert clamp_refresh_seconds(60.0) == 30.0
    assert clamp_refresh_seconds("bad") == 2.0


def test_html_constrains_active_panel_as_touch_scroll_container():
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert "main {\n  min-height: 0;\n  overflow: hidden;\n}" in css
    assert ".panel {\n  height: 100%;" in css
    assert "  overflow: auto;" in css


def test_static_js_renders_sectioned_run_data_with_legacy_fallback():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "snapshot.run_sections" in js
    assert "section-title" in js
    assert "legacyItems" in js
    assert "legacyRunSections" in js
    assert "buckets.Agent" in js


def test_tree_toolbar_controls_best_active_and_follow():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert 'class="tree-toolbar"' in html
    assert 'data-tree-target="best"' in html
    assert 'data-tree-target="active"' in html
    assert 'id="tree-follow"' in html
    assert "disabled" in html
    assert "treeTarget = null" in js
    assert "treeFollow = false" in js
    assert ".tree-line.best" in js
    assert ".tree-line.active" in js
    assert "scrollIntoView" in js
    assert ".tree-toolbar" in css
    assert "position: sticky;" in css
    assert "z-index: 5;" in css


def test_run_data_labels_do_not_wrap():
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert "grid-template-columns: 12ch 1fr;" in css
    assert "white-space: nowrap;" in css


def test_active_tree_line_blinks_slowly():
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert ".tree-line.active .dot" in css
    assert ".tree-line.active .label" in css
    assert "animation: active-node-fade 1.8s ease-in-out infinite;" in css
    assert "@keyframes active-node-fade" in css
    assert "opacity: 0.45;" in css
