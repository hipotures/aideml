from rich.console import Console

from aide.journal import Journal, Node
from aide.run import (
    build_path_summary,
    build_run_data,
    journal_to_rich_tree,
    last_error_lines,
    run_with_live_refresh,
    stage_status_message,
)
from aide.synthesis import SYNTHESIS_PLAN_PREFIX
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
    node._term_out = [
        '  File "runfile.py", line 10, in <module>\n'
        "    raise ValueError('bad feature')\n"
        "ValueError: bad feature\n"
        "Execution time: 1 second seconds (time limit is 20 minutes)."
    ]
    node.analysis = "bug analysis"
    node.exc_type = "ValueError"
    return node


def _submission_bug_node(parent: Node | None = None) -> Node:
    node = _bug_node(parent=parent)
    node.exc_type = "SubmissionValidationError"
    node.exc_info = {"args": ["row count 10 != expected 2"]}
    node.submission_validation = {
        "status": "error",
        "error": "row count 10 != expected 2",
    }
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


def test_journal_tree_hides_invalid_submission_branch_by_default():
    journal = Journal()
    root = _submission_bug_node()
    child = _good_node(0.99, parent=root)
    journal.append(root)
    journal.append(child)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert "◍ bug" not in output
    assert "0.99000" not in output


def test_journal_tree_best_marker_ignores_hidden_invalid_submission_branch():
    journal = Journal()
    invalid = _submission_bug_node()
    hidden_best = _good_node(0.99, parent=invalid)
    visible_best = _good_node(0.90)
    journal.append(invalid)
    journal.append(hidden_best)
    journal.append(visible_best)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert "0.99000" not in output
    assert "● 0.90000 (best)" in output


def test_journal_tree_can_show_invalid_submission_branch():
    journal = Journal()
    root = _submission_bug_node()
    child = _good_node(0.99, parent=root)
    journal.append(root)
    journal.append(child)

    tree = journal_to_rich_tree(journal, show_invalid_submission_branches=True)

    output = _render_text(tree)

    assert "◍ bug" in output
    assert "0.99000" in output


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


def test_journal_tree_marks_synthesis_root_blue_but_children_normal():
    journal = Journal()
    root = Node(
        code="print('synth')",
        plan=f"{SYNTHESIS_PLAN_PREFIX} 000010",
    )
    root.metric = MetricValue(0.946, maximize=True)
    root.is_buggy = False
    child = _good_node(0.947, parent=root)
    journal.append(root)
    journal.append(child)

    output = _render_text(journal_to_rich_tree(journal))
    ansi = _render_ansi(journal_to_rich_tree(journal))

    assert "◆ 0.94600" in output
    assert "synthesis)" not in output
    assert "● 0.94700 (best)" in output
    assert "\x1b[34m◆ 0.94600" in ansi


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


def test_run_data_shows_research_status_when_enabled(tmp_path):
    log_dir = tmp_path / "logs" / "2-example-run"
    workspace_dir = tmp_path / "workspaces" / "2-example-run"

    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Generating code...",
            research_status="[cyan]Research: ▶ 000010",
            synthesis_status=None,
            journal=Journal(),
            log_dir=log_dir,
            workspace_dir=workspace_dir,
        )
    )

    assert "Research: ▶ 000010" in output
    assert "Agent workspace directory" in output


def test_run_data_hides_research_status_when_disabled(tmp_path):
    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Generating code...",
            research_status=None,
            synthesis_status=None,
            journal=Journal(),
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
        )
    )

    assert "Research:" not in output


def test_run_data_shows_synthesis_status_when_enabled(tmp_path):
    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Generating code...",
            research_status=None,
            synthesis_status="[green]Synthesis: ✓ 000015",
            journal=Journal(),
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
        )
    )

    assert "Synthesis: ✓ 000015" in output
    assert "Agent workspace directory" in output


def test_last_error_lines_uses_latest_bug_and_skips_execution_time():
    journal = Journal()
    first = _bug_node()
    first._term_out = ["RuntimeError: old bug\nExecution time: 1 second"]
    journal.append(first)
    journal.append(_good_node(0.9))
    latest = _bug_node()
    latest._term_out = [
        '  File "runfile.py", line 12, in <module>\n'
        "    model.fit(X, y)\n"
        "TypeError: bad categorical value\n"
        "Execution time: 2 seconds"
    ]
    journal.append(latest)

    lines = last_error_lines(journal)

    assert lines == ["model.fit(X, y)", "TypeError: bad categorical value"]


def test_last_error_lines_prefers_exception_over_successful_terminal_output():
    journal = Journal()
    node = Node(code="print('ok')", plan="bad submission")
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = True
    node._term_out = ["CV AUC: 0.9528\nSubmission saved successfully.\n"]
    node.analysis = "Submission validation failed: row count 4668287 != expected 188165"
    node.exc_type = "SubmissionValidationError"
    node.exc_info = {"args": ["row count 4668287 != expected 188165"]}
    journal.append(node)

    lines = last_error_lines(journal)

    assert lines == [
        "SubmissionValidationError: row count 4668287 != expected 188165"
    ]


def test_last_error_lines_uses_analysis_when_no_exception_or_error_output():
    journal = Journal()
    node = Node(code="print('ok')", plan="bad review")
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = True
    node._term_out = ["CV AUC: 0.9528\nSubmission saved successfully.\n"]
    node.analysis = "Reviewer marked this node as buggy."
    node.exc_type = None
    journal.append(node)

    lines = last_error_lines(journal)

    assert lines == ["Reviewer marked this node as buggy."]


def test_run_data_shows_last_error_below_separator(tmp_path):
    journal = Journal()
    journal.append(_bug_node())

    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Generating code...",
            research_status=None,
            synthesis_status=None,
            journal=journal,
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
        )
    )

    assert "Experiment log directory" in output
    assert "Last Error" in output
    assert output.index("Experiment log directory") < output.index("Last Error")
    assert "ValueError: bad feature" in output
    assert "Execution time:" not in output


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
