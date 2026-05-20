import datetime as dt
import json

import pytest
from rich.console import Console

from aide.interpreter import ExecutionInterrupted
from aide.journal import Journal, Node
from aide.run import (
    _sparkline,
    active_run_log_path,
    build_all_hypotheses_view,
    build_best_branch_view,
    build_path_summary,
    build_model_summary,
    build_resource_summary,
    build_run_log_summary,
    build_run_data,
    build_search_decision_debug_view,
    build_hypothesis_phase_status,
    build_operator_notice_summary,
    build_root_hypotheses_view,
    model_settings_for_run,
    ResourceSnapshot,
    ArrowKeyReader,
    _overlay_top,
    active_tree_item_id,
    best_tree_item_id,
    build_tree_view,
    center_tree_viewport,
    clamp_tree_viewport,
    journal_to_rich_tree,
    last_error_lines,
    move_tree_focus,
    _mark_node_execution_crash,
    next_left_panel_view,
    recover_tree_focus_by_index,
    render_tree_view,
    run_with_live_refresh,
    stage_status_message,
    synthesis_injected_node_ids,
)
from aide.synthesis import SYNTHESIS_PLAN_PREFIX
from aide.autogluon_preprocess import BASELINE_PLAN_PREFIX
from aide.utils.artifact_manifest import SEEDED_BASE_PLAN_PREFIX
from aide.utils.config import _load_cfg, prep_cfg
from aide.utils.resource_monitor import ResourceHistory
from aide.utils.metric import MetricValue


def test_resource_sparkline_uses_latest_samples_without_rebinning_history():
    assert _sparkline([100.0, 0.0, 0.0, 0.0], width=3, ceiling=100.0) == "▁▁▁"


def test_search_decision_debug_view_explains_best_node_rejection():
    view = build_search_decision_debug_view(
        {
            "step": 143,
            "mode": "hypothesis",
            "reason": "highest_policy_score_after_filters",
            "forced_hypothesis_root": "000365",
            "counts": {
                "good_nodes": 91,
                "after_hypothesis_child_candidates": 64,
                "after_branch_candidate": 52,
            },
            "selected": {
                "hypothesis_id": "000011",
                "metric": 0.95193,
                "step": 11,
                "child_count": 7,
            },
            "best_node": {
                "hypothesis_id": "000002",
                "metric": 0.95239,
                "step": 142,
                "selected": False,
                "rejected_at": "branch_candidate",
                "reason": "parent_metric_missing",
                "parent_metric": None,
                "parent_is_buggy": True,
            },
            "top_candidates": [
                {
                    "rank": 1,
                    "hypothesis_id": "000011",
                    "metric": 0.95193,
                    "policy_score": 0.873,
                    "step": 11,
                }
            ],
            "policy_diagnostics": {
                "candidate_count": 64,
                "exploration_weight": 0.05,
                "metric_min": 0.94939,
                "metric_max": 0.95239,
                "metric_span": 0.003,
                "selected_minus_best_policy_score": 0.018,
                "selected_minus_best_metric": -0.00046,
                "selected": {
                    "policy_score": 1.106,
                    "normalized_metric": 0.847,
                    "exploration_bonus": 0.259,
                },
                "best": {
                    "policy_score": 1.088,
                    "normalized_metric": 1.0,
                    "exploration_bonus": 0.088,
                },
                "fresh_child_metric_threshold": {
                    "child_count": 0,
                    "direction": ">=",
                    "metric": 0.95182,
                },
                "selection_override": {
                    "reason": "best_score_min_children_before_exploration",
                    "best_child_count": 1,
                    "min_children": 3,
                },
            },
        }
    )

    output = _render_text(view)

    assert "SEARCH DECISION step=143 mode=hypothesis" in output
    assert "forced_root    000365" in output
    assert "SELECTED        0.95193*000011" in output
    assert "BEST SCORE NODE 0.95239*000002" in output
    assert "not selected:   branch_candidate / parent_metric_missing" in output
    assert "POLICY SCORE" in output
    assert "best filtered   policy=1.08800" in output
    assert "override        best children 1/3 before exploration" in output
    assert "children=0 beats best if score >= 0.95182" in output


def test_overlay_top_centers_with_console_edge_margin():
    assert _overlay_top(console_height=40, overlay_height=10, edge_margin=3) == 15
    assert _overlay_top(console_height=20, overlay_height=10, edge_margin=3) == 5
    assert _overlay_top(console_height=12, overlay_height=10, edge_margin=3) == 3


def _good_node(
    score: float,
    parent: Node | None = None,
    ctime: float | None = None,
) -> Node:
    kwargs = {"ctime": ctime} if ctime is not None else {}
    node = Node(code="print('ok')", plan="ok", parent=parent, **kwargs)
    node.metric = MetricValue(score, maximize=True)
    node.is_buggy = False
    return node


def _hypothesis_node(node: Node, hypothesis_id: str) -> Node:
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = [hypothesis_id]
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


def _failed_node(parent: Node | None = None) -> Node:
    node = Node(code="# Failed checkpoint did not produce code.\n", plan="failed", parent=parent)
    node.status = "failed"
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = True
    node._term_out = ["Failed: checkpoint failed"]
    node.analysis = "checkpoint failed"
    node.exc_type = "Failed"
    return node


def _oom_bug_node(parent: Node | None = None) -> Node:
    node = _bug_node(parent=parent)
    node.status = "bug"
    node.analysis = (
        "REPL child process died unexpectedly\n\n"
        "CatBoost GPU ran out of memory while the REPL child process was executing."
    )
    node._term_out = [
        "RuntimeError: REPL child process died unexpectedly\n"
        "CatBoost GPU ran out of memory while the REPL child process was executing."
    ]
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

    assert "* 0.94500" in output
    assert "[*]" in output
    assert "executing" not in output


def test_journal_tree_renders_active_hypothesis_id_on_placeholder():
    journal = Journal()
    parent = _hypothesis_node(_good_node(0.945), "000111")
    journal.append(parent)

    tree = journal_to_rich_tree(
        journal,
        active_parent_node=parent,
        active_stage="generating",
        active_hypothesis_id="000348",
        blink_on=True,
    )

    output = _render_text(tree)

    assert "[*]·000348" in output


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
    assert "● bug" in output


def test_journal_tree_hides_failed_nodes_by_default():
    journal = Journal()
    failed = _failed_node()
    good = _good_node(0.9)
    journal.append(failed)
    journal.append(good)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert "failed" not in output.lower()
    assert "0.90000" in output


def test_journal_tree_hides_catboost_gpu_oom_nodes_by_default():
    journal = Journal()
    oom = _oom_bug_node()
    good = _good_node(0.9)
    journal.append(oom)
    journal.append(good)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert "bug" not in output.lower()
    assert "0.90000" in output


def test_journal_tree_hides_invalid_submission_branch_by_default():
    journal = Journal()
    root = _submission_bug_node()
    child = _good_node(0.99, parent=root)
    journal.append(root)
    journal.append(child)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert "● bug" not in output
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
    assert "* 0.90000" in output


def test_journal_tree_can_show_invalid_submission_branch():
    journal = Journal()
    root = _submission_bug_node()
    child = _good_node(0.99, parent=root)
    journal.append(root)
    journal.append(child)

    tree = journal_to_rich_tree(journal, show_invalid_submission_branches=True)

    output = _render_text(tree)

    assert "● bug" in output
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
    assert "* 0.94700" in output
    assert "\x1b[34m◆\x1b[0m \x1b[32m0.94600" in ansi


def test_journal_tree_marks_status_recorded_synthesis_root_blue():
    journal = Journal()
    root = Node(
        code="print('synth')",
        plan="This synthesis keeps the strongest feature families.",
    )
    root.metric = MetricValue(0.946, maximize=True)
    root.is_buggy = False
    best = _good_node(0.947)
    journal.append(root)
    journal.append(best)

    tree = journal_to_rich_tree(journal, synthesis_node_ids={root.id})
    output = _render_text(tree)
    ansi = _render_ansi(tree)

    assert "◆ 0.94600" in output
    assert "* 0.94700" in output
    assert "\x1b[34m◆\x1b[0m \x1b[32m0.94600" in ansi


def test_journal_tree_marks_baseline_with_star_until_new_best_then_bullseye():
    journal = Journal()
    baseline = Node(
        code="print('baseline')",
        plan=f"{BASELINE_PLAN_PREFIX}: raw features",
    )
    baseline.metric = MetricValue(0.950, maximize=True)
    baseline.is_buggy = False
    journal.append(baseline)

    output = _render_text(journal_to_rich_tree(journal))
    ansi = _render_ansi(journal_to_rich_tree(journal))

    assert "* 0.95000" in output
    assert "◎" not in output
    assert "\x1b[1;33m*" in ansi

    improved = _good_node(0.951)
    journal.append(improved)

    output = _render_text(journal_to_rich_tree(journal))
    ansi = _render_ansi(journal_to_rich_tree(journal))

    assert "◎ 0.95000" in output
    assert "* 0.95100" in output
    assert "\x1b[95m◎" in ansi


def test_journal_tree_treats_appended_node_without_metric_as_bug():
    journal = Journal()
    node = Node(code="print('pending')", plan="pending")
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = False
    journal.append(node)

    tree = journal_to_rich_tree(journal)

    output = _render_text(tree)

    assert "n/a" not in output
    assert "● bug" in output


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

    assert "* 0.94500" in output
    assert "n/a" not in output
    assert "[*]" in output


def test_tree_view_starts_with_focusable_header_and_node_rows():
    journal = Journal()
    root = _good_node(0.90)
    child = _good_node(0.91, parent=root)
    journal.append(root)
    journal.append(child)

    view = build_tree_view(journal)

    assert view.items[0].item_id == "header"
    assert view.items[0].parent_id is None
    assert [item.item_id for item in view.items[1:]] == [root.id, child.id]
    assert view.children_by_id["header"] == [root.id]
    assert view.children_by_id[root.id] == [child.id]


def test_root_hypotheses_view_sorts_scored_roots_by_score():
    journal = Journal()
    baseline = Node(code="base", plan=f"{BASELINE_PLAN_PREFIX}: raw")
    baseline.metric = MetricValue(0.99, maximize=True)
    baseline.is_buggy = False
    journal.append(baseline)
    weak = _hypothesis_node(
        _good_node(0.95000, ctime=dt.datetime(2026, 5, 18, 3, 12).timestamp()),
        "000111",
    )
    weak.step = 7
    strong = _hypothesis_node(
        _good_node(0.95200, ctime=dt.datetime(2026, 5, 18, 5, 59).timestamp()),
        "000222",
    )
    strong.step = 145
    child = _hypothesis_node(_good_node(0.95300, parent=weak), "000333")
    journal.append(weak)
    journal.append(strong)
    journal.append(child)

    view = build_root_hypotheses_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "Root hypotheses" in output
    assert "#    score    hypothesis  time" in output
    assert "002  0.95200  000222      05-18 05:59" in output
    assert "001  0.95000  000111      05-18 03:12" in output
    assert output.index("000222") < output.index("000111")
    assert "000333" not in output
    assert "0.99000" not in output


def test_all_hypotheses_view_aggregates_successful_usage_by_hypothesis():
    journal = Journal()
    root = _hypothesis_node(_good_node(0.95100), "000111")
    child = _hypothesis_node(_good_node(0.95200, parent=root), "000222")
    repeated = _hypothesis_node(_good_node(0.95300, parent=child), "000111")
    bug = _hypothesis_node(_bug_node(parent=root), "000333")
    journal.append(root)
    journal.append(child)
    journal.append(repeated)
    journal.append(bug)

    view = build_all_hypotheses_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "All hypotheses" in output
    assert output.index("000111") < output.index("000222")
    assert "0.95300" in output
    assert "2 (root 1, branch 1)" in output
    assert "000333" not in output


def test_best_branch_view_renders_top_path_root_to_leaf():
    journal = Journal()
    root = _hypothesis_node(_good_node(0.95100), "000111")
    child = _hypothesis_node(_good_node(0.95200, parent=root), "000222")
    top = _hypothesis_node(_good_node(0.95300, parent=child), "000333")
    other = _hypothesis_node(_good_node(0.95000), "000444")
    journal.append(root)
    journal.append(child)
    journal.append(top)
    journal.append(other)

    view = build_best_branch_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "Best branch" in output
    assert "000111 0.95100 -> 000222 0.95200 -> 000333 0.95300" in output
    assert "000444" not in output


def test_best_branch_view_skips_unscored_ancestors():
    journal = Journal()
    root = _hypothesis_node(Node(code="pending", plan="pending"), "000111")
    root.metric = MetricValue(None, maximize=True)
    root.is_buggy = False
    top = _hypothesis_node(_good_node(0.95300, parent=root), "000222")
    journal.append(root)
    journal.append(top)

    view = build_best_branch_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "Best branch" in output
    assert "000111" not in output
    assert "n/a" not in output
    assert "000222 0.95300" in output


def test_best_branch_view_skips_failed_retry_with_same_hypothesis_id():
    journal = Journal()
    failed_retry = _hypothesis_node(_bug_node(), "000002")
    top = _hypothesis_node(_good_node(0.95300, parent=failed_retry), "000002")
    journal.append(failed_retry)
    journal.append(top)

    view = build_best_branch_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "Best branch" in output
    assert "n/a" not in output
    assert "000002 0.95300" in output
    assert output.count("000002") == 1


def test_tree_focus_moves_by_siblings_parent_and_child():
    journal = Journal()
    root = _good_node(0.90)
    child = _good_node(0.91, parent=root)
    second_child = _good_node(0.92, parent=root)
    sibling = _good_node(0.89)
    journal.append(root)
    journal.append(child)
    journal.append(second_child)
    journal.append(sibling)
    view = build_tree_view(journal)

    assert move_tree_focus(view, "header", "down") == root.id
    assert move_tree_focus(view, root.id, "down") == sibling.id
    assert move_tree_focus(view, sibling.id, "down") == sibling.id
    assert move_tree_focus(view, root.id, "up") == "header"
    assert move_tree_focus(view, sibling.id, "up") == root.id
    assert move_tree_focus(view, child.id, "down") == second_child.id
    assert move_tree_focus(view, second_child.id, "up") == child.id
    assert move_tree_focus(view, child.id, "left") == root.id
    assert move_tree_focus(view, root.id, "left") == "header"
    assert move_tree_focus(view, "header", "right") == root.id
    assert move_tree_focus(view, root.id, "right") == child.id
    assert move_tree_focus(view, child.id, "right") == child.id


def test_tree_focus_recovers_by_previous_index_when_focused_item_disappears():
    journal = Journal()
    root = _good_node(0.90)
    old_child = _good_node(0.91, parent=root)
    journal.append(root)
    journal.append(old_child)
    old_view = build_tree_view(
        journal,
        active_parent_node=root,
        active_stage="generating",
    )
    previous_index = old_view.index_by_id["active"]

    new_child = _good_node(0.92, parent=root)
    journal.append(new_child)
    new_view = build_tree_view(journal)

    assert recover_tree_focus_by_index(
        new_view,
        fallback_index=previous_index,
    ) == new_child.id


def test_keyboard_reader_maps_f_to_follow_toggle():
    assert ArrowKeyReader.CHAR_KEY_MAP[b"f"] == "follow"
    assert ArrowKeyReader.CHAR_KEY_MAP[b"F"] == "follow"


def test_keyboard_reader_maps_v_to_view_toggle():
    assert ArrowKeyReader.CHAR_KEY_MAP[b"v"] == "view"
    assert ArrowKeyReader.CHAR_KEY_MAP[b"V"] == "view"


def test_next_left_panel_view_cycles_in_declared_order():
    assert next_left_panel_view("tree") == "root"
    assert next_left_panel_view("root") == "all"
    assert next_left_panel_view("all") == "branch"
    assert next_left_panel_view("branch") == "tree"
    assert next_left_panel_view("unknown") == "tree"


def test_tree_viewport_keeps_focus_visible_without_empty_bottom():
    current_scroll = 0

    assert (
        clamp_tree_viewport(
            total_lines=20,
            viewport_height=5,
            focus_index=3,
            current_scroll=current_scroll,
        )
        == 0
    )
    assert (
        clamp_tree_viewport(
            total_lines=20,
            viewport_height=5,
            focus_index=6,
            current_scroll=0,
        )
        == 2
    )
    assert (
        clamp_tree_viewport(
            total_lines=20,
            viewport_height=5,
            focus_index=1,
            current_scroll=6,
        )
        == 1
    )
    assert (
        clamp_tree_viewport(
            total_lines=20,
            viewport_height=5,
            focus_index=19,
            current_scroll=18,
        )
        == 15
    )


def test_center_tree_viewport_places_focus_near_middle_and_clamps_edges():
    assert (
        center_tree_viewport(
            total_lines=20,
            viewport_height=5,
            focus_index=10,
        )
        == 8
    )
    assert (
        center_tree_viewport(
            total_lines=20,
            viewport_height=5,
            focus_index=1,
        )
        == 0
    )
    assert (
        center_tree_viewport(
            total_lines=20,
            viewport_height=5,
            focus_index=19,
        )
        == 15
    )


def test_tree_best_and_active_focus_targets():
    journal = Journal()
    root = _good_node(0.90)
    child = _good_node(0.95, parent=root)
    journal.append(root)
    journal.append(child)
    view = build_tree_view(
        journal,
        active_parent_node=root,
        active_stage="generating",
    )

    assert (
        best_tree_item_id(
            view,
            journal,
            show_invalid_submission_branches=False,
        )
        == child.id
    )
    assert active_tree_item_id(view) == "active"


def test_render_tree_view_highlights_focused_line_and_slices_viewport():
    journal = Journal()
    nodes = [_good_node(0.90 + idx / 100) for idx in range(4)]
    for node in nodes:
        journal.append(node)
    view = build_tree_view(journal)

    rendered = render_tree_view(
        view,
        focused_item_id=nodes[2].id,
        scroll_top=1,
        viewport_height=3,
    )
    output = _render_ansi(rendered)

    assert "Solution tree" not in output
    assert "0.90000" in output
    assert "0.91000" in output
    assert "0.92000" in output
    assert "0.93000" not in output
    assert "\x1b[7;" in output


def test_render_tree_view_highlights_node_marker_and_score_not_tree_guides():
    journal = Journal()
    root = _good_node(0.90)
    child = _good_node(0.91, parent=root)
    journal.append(root)
    journal.append(child)
    view = build_tree_view(journal)

    output = _render_ansi(render_tree_view(
        view,
        focused_item_id=child.id,
        scroll_top=0,
        viewport_height=10,
    ))

    assert "\x1b[7m└──" not in output
    assert "\x1b[1;7;33m*" in output


def test_tree_view_renders_active_placeholder_as_tree_child():
    journal = Journal()
    root = _good_node(0.90)
    child = _good_node(0.91, parent=root)
    journal.append(root)
    journal.append(child)

    view = build_tree_view(
        journal,
        active_parent_node=root,
        active_stage="executing",
        blink_on=True,
    )
    output = _render_text(render_tree_view(
        view,
        focused_item_id="header",
        scroll_top=0,
        viewport_height=10,
    ))

    assert "├── * 0.91000" in output
    assert "└── [*]" in output


def test_tree_view_renders_active_hypothesis_id_on_placeholder():
    journal = Journal()
    root = _hypothesis_node(_good_node(0.90), "000111")
    journal.append(root)

    view = build_tree_view(
        journal,
        active_parent_node=root,
        active_stage="generating",
        active_hypothesis_id="000348",
        blink_on=True,
    )
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "└── [*]·000348" in output


def test_tree_view_marks_baseline_bullseye_when_not_best():
    journal = Journal()
    baseline = Node(code="print('baseline')", plan=f"{BASELINE_PLAN_PREFIX}: raw")
    baseline.metric = MetricValue(0.950, maximize=True)
    baseline.is_buggy = False
    best = _good_node(0.951)
    journal.append(baseline)
    journal.append(best)

    view = build_tree_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "◎ 0.95000" in output
    assert "* 0.95100" in output


def test_tree_view_marks_seeded_base_bullseye_when_not_best():
    journal = Journal()
    seeded = Node(code="print('seed')", plan=f"{SEEDED_BASE_PLAN_PREFIX}: source")
    seeded.metric = MetricValue(0.950, maximize=True)
    seeded.is_buggy = False
    best = _good_node(0.951)
    journal.append(seeded)
    journal.append(best)

    view = build_tree_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "◎ 0.95000" in output
    assert "* 0.95100" in output


def test_tree_view_keeps_oom_saturated_parent_active_by_default():
    journal = Journal()
    parent = _good_node(0.951)
    good_child = _good_node(0.950, parent=parent)
    journal.append(parent)
    journal.append(good_child)
    for _ in range(3):
        journal.append(_oom_bug_node(parent=parent))

    tree = render_tree_view(
        build_tree_view(journal),
        focused_item_id="header",
        scroll_top=0,
        viewport_height=10,
    )
    output = _render_text(tree)

    assert "* 0.95100" in output
    assert "✕ 0.95100" not in output
    assert "failed" not in output.lower()
    assert "0.95000" in output


def test_tree_view_marks_oom_blocked_parent_gray_when_enabled():
    journal = Journal()
    parent = _good_node(0.951)
    good_child = _good_node(0.950, parent=parent)
    journal.append(parent)
    journal.append(good_child)
    for _ in range(3):
        journal.append(_oom_bug_node(parent=parent))

    tree = render_tree_view(
        build_tree_view(journal, disable_oom_saturated_parents=True),
        focused_item_id="header",
        scroll_top=0,
        viewport_height=10,
    )
    output = _render_text(tree)
    ansi = _render_ansi(tree)

    assert "✕ 0.95100" in output
    assert "failed" not in output.lower()
    assert "0.95000" in output
    assert "\x1b[90m✕ 0.95100" in ansi


def test_tree_view_colors_best_metric_yellow_like_star():
    journal = Journal()
    best = _good_node(0.951)
    journal.append(_good_node(0.950))
    journal.append(best)

    tree = render_tree_view(
        build_tree_view(journal),
        focused_item_id="header",
        scroll_top=0,
        viewport_height=10,
    )
    output = _render_text(tree)
    ansi = _render_ansi(tree)

    assert "* 0.95100" in output
    assert "\x1b[1;33m* " in ansi
    assert "\x1b[1;33m0.95100" in ansi


def test_tree_view_marks_status_recorded_synthesis_root_blue():
    journal = Journal()
    root = Node(
        code="print('synth')",
        plan="This synthesis keeps the strongest feature families.",
    )
    root.metric = MetricValue(0.946, maximize=True)
    root.is_buggy = False
    best = _good_node(0.947)
    journal.append(root)
    journal.append(best)

    view = build_tree_view(journal, synthesis_node_ids={root.id})
    rendered = render_tree_view(
        view,
        focused_item_id="header",
        scroll_top=0,
        viewport_height=10,
    )
    output = _render_text(rendered)
    ansi = _render_ansi(rendered)

    assert "◆ 0.94600" in output
    assert "* 0.94700" in output
    assert "\x1b[34m◆\x1b[0m \x1b[32m0.94600" in ansi


def test_synthesis_injected_node_ids_reads_checkpoint_status(tmp_path):
    checkpoint = tmp_path / "synthesis" / "checkpoint-000003"
    checkpoint.mkdir(parents=True)
    (checkpoint / "status.json").write_text(
        json.dumps({"injected_node_id": "node-123"}),
        encoding="utf-8",
    )

    assert synthesis_injected_node_ids(tmp_path) == {"node-123"}


def test_tree_view_renders_root_active_placeholder_as_tree_child():
    journal = Journal()
    root = _good_node(0.90)
    journal.append(root)

    view = build_tree_view(
        journal,
        active_parent_node=None,
        active_stage="generating",
        blink_on=False,
    )
    output = _render_text(render_tree_view(
        view,
        focused_item_id="header",
        scroll_top=0,
        viewport_height=10,
    ))

    assert "├── * 0.90000" in output
    assert "└── [ ]" in output


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

    assert "◆ Research   010 ▶" in output
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

    assert "◆ Synthesis  015 ✓" in output
    assert "Agent workspace directory" in output


def test_run_data_shows_checkpoint_and_best_score_statuses_with_times(tmp_path):
    journal = Journal()
    old_best = _good_node(0.950, ctime=dt.datetime(2026, 5, 8, 2, 10, 0).timestamp())
    new_best = _good_node(0.95108, ctime=dt.datetime(2026, 5, 8, 2, 11, 25).timestamp())
    journal.append(old_best)
    journal.append(new_best)
    log_dir = tmp_path / "logs" / "2-example-run"
    research_checkpoint = log_dir / "research" / "checkpoint-000098"
    research_checkpoint.mkdir(parents=True)
    (research_checkpoint / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_step": 98,
                "completed_at": "2026-05-08T02:08:34",
            }
        ),
        encoding="utf-8",
    )
    synthesis_checkpoint = log_dir / "synthesis" / "checkpoint-000098"
    synthesis_checkpoint.mkdir(parents=True)
    (synthesis_checkpoint / "status.json").write_text(
        json.dumps(
            {
                "status": "injected",
                "checkpoint_step": 98,
                "injected_at": "2026-05-08T02:09:15",
            }
        ),
        encoding="utf-8",
    )

    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Generating code...",
            research_status="[green]Research: ✓ 000098",
            synthesis_status="[green]Synthesis: ✓ 000098",
            journal=journal,
            log_dir=log_dir,
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
            model_settings=[("code", "gpt-5.5", "medium")],
        )
    )

    assert "◆ Research   098 @ 02:08:34 ✓" in output
    assert "◆ Synthesis  098 @ 02:09:15 ✓" in output
    assert "★ Best Score 001 @ 05-08 02:11 0.95108" in output
    assert output.index("◆ Research") < output.index("◆ Synthesis")
    assert output.index("◆ Synthesis") < output.index("★ Best Score")
    assert output.index("★ Best Score") < output.index("Models")

    ansi = _render_ansi(
        build_run_data(
            progress="Progress: 1/20",
            status="Generating code...",
            research_status="[green]Research: ✓ 000098",
            synthesis_status="[green]Synthesis: ✓ 000098",
            journal=journal,
            log_dir=log_dir,
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
        )
    )
    assert "\x1b[1;36m★ Best Score" in ansi
    assert "\x1b[1;33m★ Best Score" not in ansi


def test_run_data_shows_hypothesis_status_and_best_score_hypothesis(tmp_path):
    journal = Journal()
    best = _hypothesis_node(
        _good_node(0.95115, ctime=dt.datetime(2026, 5, 8, 19, 13, 0).timestamp()),
        "000122",
    )
    journal.append(best)

    output = _render_text(
        build_run_data(
            progress="Progress: 30/1500",
            status="Generating code...",
            research_status="[green]Research: ✓ 000030 @ 000122",
            synthesis_status=None,
            journal=journal,
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
        )
    )

    assert "◆ Research   030 @ 000122 ✓" in output
    assert "★ Best Score 000 @ 05-08 19:13 0.95115 · 000122" in output


def test_run_data_shows_operator_notice_before_last_error(tmp_path):
    output = _render_text(
        build_run_data(
            progress="Progress: 30/1500",
            status="Training AutoGluon...",
            research_status=None,
            synthesis_status=None,
            journal=Journal(),
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
            operator_notice=(
                "Ctrl+C received. Waiting for current code to finish. "
                "The node will be reviewed and saved."
            ),
        )
    )

    assert "Training AutoGluon..." in output
    assert "Operator Notice" in output
    assert "Ctrl+C received. Waiting for current code to finish." in output
    assert output.index("Operator Notice") < output.index("Last Error")


def test_operator_notice_summary_uses_notice_color():
    ansi = _render_ansi(
        build_operator_notice_summary("Ctrl+C received. Waiting for current code.")
    )

    assert "\x1b[33mCtrl+C received" in ansi


def test_hypothesis_phase_status_shows_both_counters_and_active_color(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_limit = 3
    cfg.agent.steps = 10
    cfg = prep_cfg(cfg)
    journal = Journal()
    for hypothesis_id in ["000001", "000002"]:
        journal.append(_hypothesis_node(_good_node(0.95), hypothesis_id))
    child = _hypothesis_node(_good_node(0.951, parent=journal.nodes[0]), "000003")
    journal.append(child)

    output = _render_text(build_hypothesis_phase_status(cfg, journal))
    ansi = _render_ansi(build_hypothesis_phase_status(cfg, journal))

    assert "⬢ Phase      exploration 2/3 · exploitation 1/7" in output
    assert "\x1b[32mexploration 2/3" in ansi
    assert "exploitation 1/7" in ansi


def test_hypothesis_phase_status_ignores_lower_resume_limit(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_limit = 1
    cfg.agent.steps = 10
    cfg = prep_cfg(cfg)
    journal = Journal()
    for hypothesis_id in ["000001", "000002", "000003"]:
        journal.append(_hypothesis_node(_good_node(0.95), hypothesis_id))
    journal.append(_hypothesis_node(_good_node(0.951, parent=journal.nodes[0]), "000004"))

    output = _render_text(build_hypothesis_phase_status(cfg, journal))

    assert "⬢ Phase      exploration 3/3 · exploitation 1/7" in output


def test_tree_view_appends_hypothesis_id_to_metric_and_bug_labels():
    journal = Journal()
    root = _hypothesis_node(_good_node(0.95104), "000122")
    bug = _hypothesis_node(_bug_node(parent=root), "000205")
    journal.append(root)
    journal.append(bug)

    view = build_tree_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "0.95104·000122" in output
    assert "bug·000205" in output


def test_tree_view_shows_timeout_label_with_hypothesis_id():
    journal = Journal()
    root = _hypothesis_node(_good_node(0.95104), "000122")
    timeout = _hypothesis_node(_bug_node(parent=root), "000447")
    timeout.exc_type = "TimeoutError"
    journal.append(root)
    journal.append(timeout)

    view = build_tree_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "timeout·000447" in output
    assert "bug·000447" not in output


def test_tree_view_shows_preprocess_timeout_label_with_hypothesis_id():
    journal = Journal()
    root = _hypothesis_node(_good_node(0.95104), "000122")
    timeout = _hypothesis_node(_bug_node(parent=root), "000447")
    timeout.exc_type = "PreprocessTimeoutError"
    journal.append(root)
    journal.append(timeout)

    view = build_tree_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "timeout·000447" in output
    assert "bug·000447" not in output


def test_tree_view_shows_hypothesis_protocol_failures_with_id():
    journal = Journal()
    root = _hypothesis_node(_good_node(0.95104), "000122")
    failed = _hypothesis_node(_failed_node(parent=root), "000311")
    journal.append(root)
    journal.append(failed)

    view = build_tree_view(journal)
    output = _render_text(
        render_tree_view(
            view,
            focused_item_id="header",
            scroll_top=0,
            viewport_height=10,
        )
    )

    assert "● failed·000311" in output


def test_run_data_shows_current_artifact_directory_under_log_dir(tmp_path):
    active_node = Node(
        code="print('running')",
        plan="active",
        ctime=dt.datetime(2026, 5, 5, 21, 18, 50).timestamp(),
    )

    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Generating code...",
            research_status=None,
            synthesis_status=None,
            journal=Journal(),
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
            active_artifact_dir=(
                tmp_path / "logs" / "2-example-run" / "artifacts" / "20260505T211850"
            ),
        )
    )

    assert "Experiment log directory" in output
    assert "Current artifact directory" in output
    assert "▶ logs/2-example-run/artifacts/20260505T211850" in output
    assert output.index("Experiment log directory") < output.index(
        "Current artifact directory"
    )
    assert active_node.ctime


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


def test_run_data_shows_last_error_step_and_time(tmp_path):
    journal = Journal()
    node = _bug_node()
    node.ctime = dt.datetime(2026, 5, 7, 12, 23, 44).timestamp()
    journal.append(node)

    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Executing code...",
            research_status=None,
            synthesis_status=None,
            journal=journal,
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
        )
    )

    assert "Last Error · 0@12:23:44" in output


def test_run_data_shows_resources_below_last_error(tmp_path):
    history = ResourceHistory(window_seconds=30 * 60, interval_seconds=1)
    history.add(
        ResourceSnapshot(
            cpu_percent=320.0,
            ram_bytes=int(10.0 * 1024**3),
            peak_ram_bytes=int(10.0 * 1024**3),
            process_count=4,
            gpu_percent=25.0,
            gpu_memory_used_bytes=int(6.0 * 1024**3),
            gpu_memory_total_bytes=int(24.0 * 1024**3),
            gpu_power_draw_watts=120.0,
            gpu_power_limit_watts=450.0,
            gpu_temperature_celsius=52.0,
        )
    )
    history.add(
        ResourceSnapshot(
            cpu_percent=640.0,
            ram_bytes=int(18.4 * 1024**3),
            peak_ram_bytes=int(22.1 * 1024**3),
            process_count=9,
            gpu_percent=91.0,
            gpu_memory_used_bytes=int(15.5 * 1024**3),
            gpu_memory_total_bytes=int(24.0 * 1024**3),
            gpu_power_draw_watts=321.0,
            gpu_power_limit_watts=450.0,
            gpu_temperature_celsius=68.0,
        )
    )

    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Executing code...",
            research_status=None,
            synthesis_status=None,
            journal=Journal(),
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
            resource_history=history,
            resource_active=True,
        )
    )

    assert "Last Error" in output
    assert "Resources" in output
    assert output.index("Last Error") < output.index("Resources")
    assert "CPU" in output and "640%" in output
    assert "RAM" in output and "18.4G" in output
    assert "peak" in output and "22.1G" in output
    assert "GPU" in output and "91%" in output
    assert "VRAM" in output and "15.5G" in output
    assert "PWR" in output and "321W" in output
    assert "TEMP" in output and "68C" in output
    assert "proc" not in output
    assert "█" in output
    assert "▁" in output or "▂" in output or "▃" in output

    resource_lines = [
        line for line in output.splitlines() if line.startswith("▶ ") and "█" in line
    ]
    assert len(resource_lines) == 7
    bar_columns = [line.index("█") for line in resource_lines]
    assert len(set(bar_columns)) == 1


def test_run_data_uses_configured_resource_graph_width(tmp_path):
    history = ResourceHistory(window_seconds=30 * 60, interval_seconds=1)
    for index in range(50):
        history.add(
            ResourceSnapshot(
                cpu_percent=float(index),
                ram_bytes=int((1.0 + index / 100) * 1024**3),
                peak_ram_bytes=int((1.5 + index / 100) * 1024**3),
                process_count=1,
                gpu_percent=float(index % 100),
                gpu_memory_used_bytes=int((index / 10) * 1024**3),
                gpu_memory_total_bytes=int(24.0 * 1024**3),
                gpu_power_draw_watts=float(index),
                gpu_power_limit_watts=450.0,
                gpu_temperature_celsius=40.0 + float(index % 10),
            )
        )

    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Executing code...",
            research_status=None,
            synthesis_status=None,
            journal=Journal(),
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
            resource_history=history,
            resource_active=True,
            resource_graph_width=40,
        )
    )

    cpu_line = next(line for line in output.splitlines() if line.startswith("▶ CPU"))
    assert len(cpu_line.rsplit(" ", 1)[-1]) == 40


def test_resource_summary_marks_only_busy_gpu_percent_red():
    history = ResourceHistory(window_seconds=30 * 60, interval_seconds=1)
    history.add(
        ResourceSnapshot(
            cpu_percent=120.0,
            ram_bytes=int(4.0 * 1024**3),
            peak_ram_bytes=int(5.2 * 1024**3),
            process_count=2,
            gpu_percent=78.0,
            gpu_memory_used_bytes=int(18.9 * 1024**3),
            gpu_memory_total_bytes=int(24.0 * 1024**3),
            gpu_power_draw_watts=186.0,
            gpu_power_limit_watts=450.0,
            gpu_temperature_celsius=46.0,
        )
    )

    output = _render_ansi(build_resource_summary(history, graph_width=0))

    assert "GPU" in output
    assert "\x1b[31m    78%\x1b[0m" in output
    assert "\x1b[31mGPU\x1b[0m" not in output


def test_resource_summary_keeps_idle_gpu_percent_yellow():
    history = ResourceHistory(window_seconds=30 * 60, interval_seconds=1)
    history.add(
        ResourceSnapshot(
            cpu_percent=120.0,
            ram_bytes=int(4.0 * 1024**3),
            peak_ram_bytes=int(5.2 * 1024**3),
            process_count=2,
            gpu_percent=10.0,
            gpu_memory_used_bytes=int(0.5 * 1024**3),
            gpu_memory_total_bytes=int(24.0 * 1024**3),
            gpu_power_draw_watts=91.0,
            gpu_power_limit_watts=450.0,
            gpu_temperature_celsius=40.0,
        )
    )

    output = _render_ansi(build_resource_summary(history, graph_width=0))

    assert "10%" in output
    assert "\x1b[31m    10%\x1b[0m" not in output


def test_run_data_shows_resolved_model_settings(tmp_path):
    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Generating code...",
            research_status=None,
            synthesis_status="[green]Synthesis: ✓ 000015",
            journal=Journal(),
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
            model_settings=[
                ("code", "gemma-4-31B", None),
                ("feedback", "gemma-4-31B", None),
                ("report", "gemma-4-31B", None),
                ("synthesis", "gpt-5.5", "low"),
            ],
        )
    )

    assert "Models" in output
    assert "code" in output and "gemma-4-31B" in output and " - " in output
    assert "synthesis" in output and "gpt-5.5" in output and "low" in output
    assert output.index("Synthesis") < output.index("Models")
    assert output.index("Models") < output.index("Base path")


def test_model_settings_hide_research_model_in_hypothesis_mode(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "model-settings-test"
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.research.model = "unused-research-model"
    cfg = prep_cfg(cfg)

    output = _render_text(build_model_summary(model_settings_for_run(cfg)))

    assert "code" in output
    assert "feedback" in output
    assert "report" in output
    assert "research" not in output
    assert "unused-research-model" not in output


def test_run_data_hides_resources_when_code_is_not_executing(tmp_path):
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

    assert "Resources" not in output
    assert "waiting for code execution sample" not in output


def test_run_data_shows_waiting_resources_during_execution_before_first_sample(
    tmp_path,
):
    output = _render_text(
        build_run_data(
            progress="Progress: 1/20",
            status="Executing code...",
            research_status=None,
            synthesis_status=None,
            journal=Journal(),
            log_dir=tmp_path / "logs" / "2-example-run",
            workspace_dir=tmp_path / "workspaces" / "2-example-run",
            resource_active=True,
        )
    )

    assert "Resources" in output
    assert "▶ waiting for code execution sample" in output


def test_active_run_log_path_prefers_legacy_process_stdout(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    process_log = artifact_dir / "process_stdout.log"
    process_log.write_text("legacy log\n", encoding="utf-8")
    (artifact_dir / "autogluon_stdout.log").write_text("ag log\n", encoding="utf-8")

    assert active_run_log_path(artifact_dir) == process_log


def test_run_log_summary_uses_latest_lines_and_clips_width(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "autogluon_stdout.log").write_text(
        "\n".join(
            [
                "old line",
                "Fold 1 ROC AUC - XGB: 0.948123, CatBoost: 0.948953, best local blend w_xgb=0.35: 0.949320",
                "Fold 2 ROC AUC - XGB: 0.949261, CatBoost: 0.949760, best local blend w_xgb=0.40: 0.950258",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = _render_text(
        build_run_log_summary(artifact_dir, max_lines=2, max_width=48)
    )

    assert "old line" not in output
    assert "Fold 1 ROC AUC - XGB: 0.948123, CatBoost: 0.948…" in output
    assert "Fold 2 ROC AUC - XGB: 0.949261, CatBoost: 0.949…" in output


def test_run_log_summary_shows_active_hypothesis_when_process_log_is_missing(
    tmp_path,
):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    output = _render_text(
        build_run_log_summary(
            artifact_dir,
            max_lines=3,
            max_width=48,
            missing_log_hint=(
                "Hypothesis 000122\n"
                "Title: Rival-relative pit-wave features\n"
                "Try: Add current-lap rival aggregate features."
            ),
        )
    )

    assert "Hypothesis 000122" in output
    assert "Title: Rival-relative pit-wave features" in output
    assert "Try: Add current-lap rival aggregate features." in output
    assert "waiting for process log" not in output

    styled_output = _render_ansi(
        build_run_log_summary(
            artifact_dir,
            max_lines=3,
            max_width=48,
            missing_log_hint=(
                "Hypothesis 000122\n"
                "Title: Rival-relative pit-wave features\n"
                "Try: Add current-lap rival aggregate features."
            ),
        )
    )
    assert "\x1b[1;36mHypothesis \x1b[0m\x1b[2m000122\x1b[0m" in styled_output
    assert (
        "\x1b[1;36mTitle: \x1b[0m"
        "\x1b[2mRival-relative pit-wave features\x1b[0m"
    ) in styled_output


def test_run_log_summary_wraps_active_hypothesis_to_available_height(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    output = _render_text(
        build_run_log_summary(
            artifact_dir,
            max_lines=5,
            max_width=40,
            missing_log_hint=(
                "Hypothesis 000327\n"
                "Summary: Official F1 strategy breakdowns frame "
                "undercut/overcut as an economic trade: current tyre fade, "
                "fresh-tyre warm-up, clean air, striking range, and pit-loss "
                "delta.\n"
                "Try: Create features for pit_now_edge versus "
                "stay_out_one_more_lap."
            ),
        )
    )

    lines = output.splitlines()
    assert len(lines) == 5
    assert "…" not in output
    assert "current tyre fade" in output
    assert "fresh-tyre" in output
    assert "warm-up" in output


def test_run_log_summary_wraps_literal_json_newlines_as_spaces(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    output = _render_text(
        build_run_log_summary(
            artifact_dir,
            max_lines=4,
            max_width=80,
            missing_log_hint=(
                "Hypothesis 000515\n"
                "Try: Use prior stops only.\\nAdd template distance."
            ),
        )
    )

    assert "\\n" not in output
    assert "Try: Use prior stops only. Add template distance." in output


def test_run_log_summary_does_not_style_continuation_colon_as_key(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    styled_output = _render_ansi(
        build_run_log_summary(
            artifact_dir,
            max_lines=6,
            max_width=96,
            missing_log_hint=(
                "Hypothesis 000447\n"
                "Try: Within each sorted sequence, add trailing features for "
                "`LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, "
                "`Position_Change`: last value, EWMA, slope, acceleration."
            ),
        )
    )

    assert "\x1b[1;36mTry: \x1b[0m" in styled_output
    assert "\x1b[1;36m`LapTime" not in styled_output
    assert "\x1b[1;36m`Position_Change`: \x1b[0m" not in styled_output


def test_run_log_summary_prefers_process_log_over_active_hypothesis_hint(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "process_stdout.log").write_text("real execution log\n")

    output = _render_text(
        build_run_log_summary(
            artifact_dir,
            max_lines=3,
            max_width=48,
            missing_log_hint="Hypothesis 000122\nTitle: hidden",
        )
    )

    assert "real execution log" in output
    assert "Hypothesis 000122" not in output


def test_stage_status_message_names_review_stage():
    assert stage_status_message("generating") == "[green]Generating code..."
    assert stage_status_message("executing") == "[magenta]Executing code..."
    assert stage_status_message("reviewing") == "[cyan]Reviewing result..."
    assert (
        stage_status_message("generating", 65) == "[green]Generating code... (1m 05s)"
    )


def test_stage_status_message_names_autogluon_preprocess_before_fit_log(tmp_path):
    assert (
        stage_status_message(
            "executing",
            19,
            agent_mode="autogluon_preprocess",
            active_artifact_dir=tmp_path,
        )
        == "[magenta]Preprocessing features... (19s)"
    )


def test_stage_status_message_names_autogluon_training_after_fit_log(tmp_path):
    (tmp_path / "autogluon_stdout.log").write_text(
        "AIDE AutoGluon: starting fit\n",
        encoding="utf-8",
    )

    assert (
        stage_status_message(
            "executing",
            74,
            agent_mode="autogluon_preprocess",
            active_artifact_dir=tmp_path,
        )
        == "[magenta]Training AutoGluon... (1m 14s)"
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


def test_run_with_live_refresh_can_continue_after_first_keyboard_interrupt():
    import time

    class FakeLive:
        def __init__(self):
            self.updates = []
            self.interrupted = False

        def update(self, renderable, *, refresh=False):
            self.updates.append((renderable, refresh))
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt

    live = FakeLive()
    interrupts = []

    def slow_work():
        time.sleep(0.05)
        return "done"

    result = run_with_live_refresh(
        live,
        lambda: "rendered",
        slow_work,
        on_keyboard_interrupt=lambda: interrupts.append("interrupt") or "continue",
    )

    assert result == "done"
    assert interrupts == ["interrupt"]
    assert len(live.updates) >= 2


def test_run_with_live_refresh_aborts_after_keyboard_interrupt_request():
    import time

    class FakeLive:
        def update(self, renderable, *, refresh=False):
            raise KeyboardInterrupt

    def slow_work():
        time.sleep(0.2)
        return "done"

    with pytest.raises(ExecutionInterrupted):
        run_with_live_refresh(
            FakeLive(),
            lambda: "rendered",
            slow_work,
            on_keyboard_interrupt=lambda: "abort",
        )


def test_mark_node_execution_crash_records_bug_result():
    node = Node(code="print('crash')", plan="crash")

    _mark_node_execution_crash(
        node,
        RuntimeError("REPL child process died unexpectedly"),
    )

    assert node.is_buggy is True
    assert node.metric.is_worst
    assert node.status == "bug"
    assert node.exc_type == "RuntimeError"
    assert node.exc_info == {"args": ["REPL child process died unexpectedly"]}
    assert "REPL child process died unexpectedly" in node.term_out
    assert "REPL child process died unexpectedly" in node.analysis


def test_mark_node_execution_crash_reports_autogluon_gpu_oom(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "autogluon_stdout.log").write_text(
        "Fitting model: CatBoost ...\n"
        "catboost/cuda/cuda_lib/cuda_base.h:177: CUDA error 2: out of memory\n",
        encoding="utf-8",
    )
    node = Node(code="print('crash')", plan="crash")

    _mark_node_execution_crash(
        node,
        RuntimeError("REPL child process died unexpectedly"),
        artifact_dir=artifact_dir,
    )

    assert "CatBoost GPU ran out of memory" in node.analysis
    assert "CUDA error 2: out of memory" in node.analysis
    assert "CatBoost GPU ran out of memory" in node.term_out
    assert node.status == "failed"
    assert node.is_terminal_failure is True
