from rich.console import Console

from aide.journal import Journal, Node
from aide.run import build_path_summary, journal_to_rich_tree
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

    assert "● 0.945 (best)" in output
    assert "[*]" in output
    assert "executing" not in output


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


def test_path_summary_shows_shared_base_once_and_relative_paths(tmp_path):
    log_dir = tmp_path / "logs" / "2-example-run"
    workspace_dir = tmp_path / "workspaces" / "2-example-run"

    output = _render_text(build_path_summary(log_dir, workspace_dir))

    assert "Base path" in output
    assert f"▶ {tmp_path}/" in output
    assert "▶ logs/2-example-run/tree_plot.html" in output
    assert "▶ workspaces/2-example-run" in output
    assert "▶ logs/2-example-run" in output
    assert f"{log_dir}/tree_plot.html" not in output
