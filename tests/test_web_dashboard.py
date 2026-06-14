from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from aide.journal import Journal, Node
from aide.utils.metric import MetricValue, WorstMetricValue
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


def _timeout_node(*, parent: Node | None = None) -> Node:
    node = Node(code="while True: pass", plan="timeout", parent=parent)
    node.status = "bug"
    node.is_buggy = True
    node.metric = WorstMetricValue()
    node.exc_type = "TimeoutError"
    node.analysis = "Execution exceeded the configured timeout."
    node._term_out = ["TimeoutError: Execution exceeded the time limit"]
    return node


def test_web_tree_lines_include_compact_and_desktop_text_prefixes():
    journal = Journal()
    root_a = _scored_node(0.91)
    root_b = _scored_node(0.92)
    child_a = _scored_node(0.93, parent=root_b)
    child_b = _scored_node(0.94, parent=root_b)
    root_c = _scored_node(0.95)
    root_a.exec_time = 0.0
    root_b.exec_time = 12 * 60
    child_a.exec_time = 28 * 60 + 20
    child_b.exec_time = 47 * 60
    root_c.exec_time = 64 * 60
    for node in [root_a, root_b, child_a, child_b, root_c]:
        journal.append(node)

    lines = build_web_tree_lines(journal)

    assert [(line.prefix, line.label) for line in lines] == [
        ("├", "0.91000·0"),
        ("├", "0.92000·1·12m"),
        ("│├", "0.93000·2·28m"),
        ("│└", "0.94000·3·47m"),
        ("└", "0.95000·4·64m"),
    ]
    assert all("─" not in line.prefix for line in lines)
    assert [line.desktop_prefix for line in lines] == [
        "├── ",
        "├── ",
        "│   ├── ",
        "│   └── ",
        "└── ",
    ]


def test_web_tree_lines_mark_timeout_as_blocked():
    journal = Journal()
    root = _scored_node(0.91)
    timeout = _timeout_node(parent=root)
    for node in [root, timeout]:
        journal.append(node)

    lines = build_web_tree_lines(journal)

    assert lines[1].label.startswith("timeout")
    assert lines[1].kind == "blocked"


def test_web_tree_lines_include_virtual_root_hypotheses():
    journal = Journal()
    root = _scored_node(0.0)
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000011"]
    root.status = "generated"
    root.metric = None
    journal.append(root)

    lines = build_web_tree_lines(
        journal,
        virtual_root_rows=[
            {"hypothesis_id": "000011", "status": "generated"},
            {"hypothesis_id": "000012", "status": "hypothesis"},
        ],
    )

    assert [(line.prefix, line.label, line.kind) for line in lines] == [
        ("├", "generated·000011", "generated"),
        ("└", "hypothesis·000012", "hypothesis"),
    ]


def test_web_tree_lines_keep_tiny_improving_child_unblocked():
    journal = Journal()
    parent = _scored_node(0.967787)
    child = _scored_node(0.967792, parent=parent)
    journal.append(parent)
    journal.append(child)

    lines = build_web_tree_lines(journal, plateau_block_epsilon=0.00001)

    assert lines[1].label == "0.96779·1"
    assert lines[1].kind == "best"


def test_web_tree_lines_mark_non_improving_plateau_child_as_blocked():
    journal = Journal()
    parent = _scored_node(0.967787)
    child = _scored_node(0.967782, parent=parent)
    journal.append(parent)
    journal.append(child)

    lines = build_web_tree_lines(journal, plateau_block_epsilon=0.00001)

    assert lines[1].label == "0.96778·1"
    assert lines[1].kind == "blocked"


def test_web_tree_lines_keep_public_marker_on_blocked_submitted_node():
    journal = Journal()
    parent = _scored_node(0.967787)
    child = _scored_node(0.967782, parent=parent)
    journal.append(parent)
    journal.append(child)

    lines = build_web_tree_lines(
        journal,
        plateau_block_epsilon=0.00001,
        public_scores_by_node_id={child.id: 0.96869},
    )

    assert lines[1].label == "0.96778·1"
    assert lines[1].kind == "blocked public"


def test_web_tree_lines_accept_active_step_placeholder():
    journal = Journal()
    root = _scored_node(0.91)
    journal.append(root)

    lines = build_web_tree_lines(
        journal,
        active_parent_node=root,
        active_stage="generating",
        active_step=48,
    )

    assert lines[1].label == "generating·48"
    assert lines[1].kind == "active"


def test_web_tree_lines_do_not_duplicate_existing_active_generated_node():
    journal = Journal()
    root = _scored_node(0.91)
    generated = Node(code="print('pending')", plan="pending", parent=root)
    generated.status = "generated"
    generated.is_buggy = False
    generated.metric = None
    journal.append(root)
    journal.append(generated)

    lines = build_web_tree_lines(
        journal,
        active_node=generated,
        active_parent_node=root,
        active_stage="executing",
        active_step=generated.step,
    )

    assert [line.label for line in lines] == [
        "0.91000·0",
        "generated·1",
    ]
    assert all(line.kind != "active" for line in lines)


def test_web_tree_lines_keep_child_outside_plateau_epsilon_unblocked():
    journal = Journal()
    parent = _scored_node(0.965327)
    child = _scored_node(0.965385, parent=parent)
    journal.append(parent)
    journal.append(child)

    lines = build_web_tree_lines(journal, plateau_block_epsilon=0.00001)

    assert lines[1].label == "0.96539·1"
    assert lines[1].kind == "best"


def test_web_tree_lines_mark_public_bonus_node_but_keep_cv_label():
    journal = Journal()
    cv_best = _scored_node(0.967931)
    public_best = _scored_node(0.967889)
    public_worse = _scored_node(0.967728)
    journal.append(cv_best)
    journal.append(public_best)
    journal.append(public_worse)

    lines = build_web_tree_lines(
        journal,
        public_scores_by_node_id={
            cv_best.id: 0.96800,
            public_best.id: 0.96830,
            public_worse.id: 0.96753,
        },
        public_score_bonus_weight=0.5,
        public_score_bonus_cap=0.0005,
    )

    assert lines[0].label == "0.96793·0"
    assert lines[0].kind == "best public public-bonus"
    assert lines[1].label == "0.96789·1"
    assert lines[1].kind == "public public-best"
    assert lines[2].label == "0.96773·2"
    assert lines[2].kind == "public public-worse"
    assert "0.96830" not in lines[1].label


def test_web_tree_lines_mark_raw_public_best_not_public_adjusted_best():
    journal = Journal()
    cv_best = _scored_node(0.96815)
    public_best = _scored_node(0.96805)
    journal.append(cv_best)
    journal.append(public_best)

    lines = build_web_tree_lines(
        journal,
        public_scores_by_node_id={
            cv_best.id: 0.96877,
            public_best.id: 0.96892,
        },
        public_score_bonus_weight=0.5,
        public_score_bonus_cap=0.0005,
    )

    assert lines[0].label == "0.96815·0"
    assert lines[0].kind == "best public public-bonus"
    assert lines[1].label == "0.96805·1"
    assert lines[1].kind == "public public-best"


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
        assert "\"desktop_prefix\": \"\"" in payload
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
    assert ".tree-line.public-best .dot" in css
    assert ".tree-toolbar" in css
    assert "position: sticky;" in css
    assert "z-index: 5;" in css


def test_run_data_labels_do_not_wrap():
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert "grid-template-columns: 12ch 1fr;" in css
    assert "white-space: nowrap;" in css


def test_blocked_public_tree_marker_uses_dim_diamond():
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert ".tree-line.blocked.public .dot" in css
    assert ".tree-line.blocked.public .dot::before" in css
    assert 'content: "◆";' in css
    assert "color: #94a3b8;" in css


def test_active_tree_line_blinks_slowly():
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert ".tree-line.active .dot" in css
    assert ".tree-line.active .label" in css
    assert "animation: active-node-fade 1.8s ease-in-out infinite;" in css
    assert "@keyframes active-node-fade" in css
    assert "opacity: 0.45;" in css


def test_static_js_uses_desktop_tree_prefix_above_mobile_width():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'window.matchMedia("(max-width: 767px)")' in js
    assert "line.desktop_prefix || line.prefix" in js
    assert "if (lastSnapshot) renderTree(lastSnapshot);" in js


def test_web_dashboard_marks_logs_tab_when_snapshot_refresh_fails():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert 'id="server-status"' not in html
    assert 'data-tab="logs"' in html
    assert "grid-template-columns: repeat(3, 1fr);" in css
    assert "setLogsConnectionError(false);" in js
    assert "setLogsConnectionError(true);" in js
    assert "dashboard refresh failed:" in js
    assert ".tab.connection-error::after" in css
    assert "@keyframes log-error-pulse" in css
