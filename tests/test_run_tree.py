from rich.console import Console

from aide.journal import Journal, Node
from aide.run import (
    build_path_summary,
    journal_to_rich_tree,
    run_with_live_refresh,
    stage_status_message,
)
from aide.utils.metric import MetricValue


def _good_node(score: float, parent: Node | None = None) -> Node:
    node = Node(code="print('ok')", plan="ok", parent=parent)
    node.metric = MetricValue(score, maximize=True)
    node.is_buggy = False
    return node


def _bug_node(parent: Node | None = None) -> Node:
    node = Node(code="raise RuntimeError('bug')", plan="bug", parent=parent)
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = True
    return node


def _render_text(tree) -> str:
    console = Console(record=True, width=100, color_system=None)
    console.print(tree)
    return console.export_text()


def _render_ansi(tree) -> str:
    console = Console(record=True, width=100, color_system="standard")
    console.print(tree)
    return console.export_text(styles=True)


def test_journal_tree_renders_blinking_active_child_under_selected_parent():
    journal = Journal()
    parent = _good_node(0.945)
    journal.append(parent)

    tree = journal_to_rich_tree(
        journal,
        active_parent_node=parent,
        active_stage="executing",
        blink_on=True,
    )

    output = _render_text(tree)

    assert "● 0.94500 (best)" in output
    assert "[*]" in output
    assert "executing" not in output


def test_journal_tree_colors_active_placeholder_by_stage():
    expected_ansi = {
        "generating": "\x1b[1;37m[*]",
        "executing": "\x1b[1;33m[*]",
        "reviewing": "\x1b[1;34m[*]",
    }

    for stage, ansi in expected_ansi.items():
        journal = Journal()
        tree = journal_to_rich_tree(journal, active_stage=stage, blink_on=True)

        output = _render_ansi(tree)

        assert ansi in output


def test_journal_tree_replaces_active_placeholder_with_final_bug_result():
    journal = Journal()
    parent = _good_node(0.945)
    child = _bug_node(parent=parent)
    journal.append(parent)
    journal.append(child)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert "[*]" not in output
    assert "[ ]" not in output
    assert "◍ bug" in output


def test_journal_tree_renders_children_in_step_order():
    journal = Journal()
    parent = _good_node(0.945)
    later_child = _good_node(0.943, parent=parent)
    earlier_child = _good_node(0.941, parent=parent)
    journal.append(parent)
    journal.append(earlier_child)
    journal.append(later_child)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert output.index("● 0.94100") < output.index("● 0.94300")


def test_journal_tree_treats_appended_node_without_metric_as_bug():
    journal = Journal()
    node = Node(code="print('pending')", plan="pending")
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = False
    journal.append(node)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert "n/a" not in output
    assert "◍ bug" in output


def test_journal_tree_ignores_unappended_active_child_node():
    journal = Journal()
    parent = _good_node(0.945)
    journal.append(parent)
    child = Node(code="print('not saved yet')", plan="not saved yet", parent=parent)
    child.metric = MetricValue(None, maximize=True)
    child.is_buggy = False

    tree = journal_to_rich_tree(
        journal,
        active_parent_node=parent,
        active_stage="generating",
        blink_on=True,
    )

    output = _render_text(tree)

    assert "● 0.94500 (best)" in output
    assert "n/a" not in output
    assert "[*]" in output


def test_path_summary_shows_shared_base_once_and_relative_paths(tmp_path):
    log_dir = tmp_path / "logs" / "2-example-run"
    workspace_dir = tmp_path / "workspaces" / "2-example-run"

    output = _render_text(build_path_summary(log_dir, workspace_dir))

    assert "Base path" in output
    assert f"▶ {tmp_path}/" in output
    assert "Result visualization" not in output
    assert "tree_plot.html" not in output
    assert "▶ workspaces/2-example-run" in output
    assert "▶ logs/2-example-run" in output


def test_stage_status_message_names_review_stage():
    assert stage_status_message("generating") == "[green]Generating code..."
    assert stage_status_message("executing") == "[magenta]Executing code..."
    assert stage_status_message("reviewing") == "[cyan]Reviewing result..."
    assert (
        stage_status_message("generating", 65) == "[green]Generating code... (1m 05s)"
    )


def test_run_with_live_refresh_updates_while_worker_is_running():
    import time

    class FakeLive:
        def __init__(self):
            self.updates = []

        def update(self, renderable, *, refresh=False):
            self.updates.append((renderable, refresh))

    live = FakeLive()

    def slow_work():
        time.sleep(0.35)
        return "done"

    result = run_with_live_refresh(live, lambda: "rendered", slow_work)

    assert result == "done"
    assert len(live.updates) >= 2
    assert all(refresh for _, refresh in live.updates)
