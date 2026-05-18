# TUI View Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cyclic `v` view switching for the left TUI panel and persist AutoGluon run statistics for new nodes without displaying those statistics in the new tables yet.

**Architecture:** Keep the live left panel backed by in-memory `journal` data. Reuse the existing flat `TreeView`/`TreeViewItem` renderer for both the current tree and the new table/path views so scrolling, focus highlighting, and terminal slicing stay consistent. Persist AutoGluon runtime/model statistics through the existing `AIDE_RESULT_JSON` marker into `Node.run_stats`, then into each node's `aide_result.json` manifest.

**Tech Stack:** Python dataclasses, Rich `Text`/`Group` rendering, existing AIDE `Journal`/`Node`, AutoGluon `TabularPredictor.leaderboard(silent=True)`, pytest.

---

## File Structure

- Modify `aide/journal.py`
  - Add `Node.run_stats: dict | None`.
- Modify `aide/agent.py`
  - Preserve `run_stats` from parsed result markers into the node.
- Modify `aide/autogluon_preprocess.py`
  - Measure preprocess and fit times.
  - Extract `feature_count`.
  - Serialize `predictor.leaderboard(silent=True)` into JSON-safe `run_stats.models`.
  - Include `run_stats` in `AIDE_RESULT_JSON`.
- Modify `aide/utils/artifact_manifest.py`
  - Include `node.run_stats` in `aide_result.json`.
  - Restore `run_stats` when reconstructing a journal from manifests.
- Modify `aide/run.py`
  - Add `v` keyboard mapping.
  - Add left-panel view cycling.
  - Add builders for root table, all-hypotheses table, and best-branch path view.
  - Render selected left-panel view in `generate_live()`.
- Modify `tests/test_autogluon_preprocess.py`
  - Cover `run_stats` marker parsing and wrapper source shape.
- Modify `tests/test_save_run_artifacts.py`
  - Cover manifest persistence of `run_stats`.
- Modify `tests/test_run_tree.py`
  - Cover keyboard mapping, view cycling, table/path builders, and subtitle.

---

### Task 1: Persist `run_stats` Through Node And Manifest

**Files:**
- Modify: `aide/journal.py`
- Modify: `aide/agent.py`
- Modify: `aide/utils/artifact_manifest.py`
- Test: `tests/test_autogluon_preprocess.py`
- Test: `tests/test_save_run_artifacts.py`

- [ ] **Step 1: Write failing tests for marker-to-node and manifest persistence**

Add to `tests/test_autogluon_preprocess.py` after `test_parse_result_marker_short_circuits_feedback_review`:

```python
def test_parse_result_marker_preserves_run_stats(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    exec_result = ExecutionResult(
        term_out=[
            'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ag ok", '
            '"metric": 0.91, "lower_is_better": false, '
            '"run_stats": {"feature_count": 42, "preprocess_time": 1.2, '
            '"training_time": 3.4, "models": [{"model": "XGBoost", '
            '"score_val": 0.91}]}}\n'
        ],
        exec_time=5.0,
        exc_type=None,
    )

    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: pytest.fail("feedback LLM should not be called"),
    )

    agent.parse_exec_result(node, exec_result)

    assert node.run_stats == {
        "feature_count": 42,
        "preprocess_time": 1.2,
        "training_time": 3.4,
        "models": [{"model": "XGBoost", "score_val": 0.91}],
    }
```

Add to `tests/test_save_run_artifacts.py` after `test_save_run_archives_current_node_code_and_submission_with_same_timestamp`:

```python
def test_save_run_writes_node_run_stats_to_manifest(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    working_dir = workspace_dir / "working"
    working_dir.mkdir(parents=True)
    (working_dir / "submission.csv").write_text("id,PitNextLap\n1,0.7\n")

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(
        code="print('current node')",
        plan="current node plan",
        ctime=1777750547.0057797,
    )
    node.metric = MetricValue(0.9473, maximize=True)
    node.is_buggy = False
    node._term_out = ["CV ROC AUC: 0.9473\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"
    node.run_stats = {
        "feature_count": 42,
        "preprocess_time": 1.2,
        "training_time": 3.4,
        "total_exec_time": 5.0,
        "models": [{"model": "WeightedEnsemble_L2", "score_val": 0.95}],
    }
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    manifest = json.loads(
        (log_dir / "artifacts" / "20260502T213547" / "aide_result.json").read_text()
    )

    expected = dict(node.run_stats)
    expected["total_exec_time"] = 1.0
    assert manifest["run_stats"] == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_parse_result_marker_preserves_run_stats tests/test_save_run_artifacts.py::test_save_run_writes_node_run_stats_to_manifest
```

Expected: FAIL because `Node` has no `run_stats` field and the manifest does not write it yet.

- [ ] **Step 3: Add `Node.run_stats`**

In `aide/journal.py`, add the field near the existing execution/evaluation metadata:

```python
    run_stats: dict | None = field(default=None, kw_only=True)
```

Place it after `exec_stack` or after `submission_validation`. The important part is that `dataclasses_json` serializes it in `journal.json`.

- [ ] **Step 4: Preserve marker `run_stats` in `Agent.parse_exec_result`**

In `aide/agent.py`, inside the `marker_response is not None` branch, after `node.validity_warning = ...`, add:

```python
            run_stats = marker_response.get("run_stats")
            node.run_stats = run_stats if isinstance(run_stats, dict) else None
```

Do not parse `run_stats` from the LLM review response path. Legacy/manual code may later emit it through a marker, but this first implementation only guarantees wrapper-generated marker preservation.

- [ ] **Step 5: Write `run_stats` into manifests**

In `aide/utils/artifact_manifest.py`, add this helper near `metric_payload(...)`:

```python
def run_stats_payload(node: Node) -> dict[str, Any] | None:
    if not isinstance(node.run_stats, dict):
        return None
    payload = dict(node.run_stats)
    payload["total_exec_time"] = node.exec_time
    return payload
```

Then inside `build_node_artifact_manifest(...)`, add a top-level key near `"execution"`:

```python
        "run_stats": run_stats_payload(node),
```

In `_node_from_manifest(...)`, after execution fields are restored, add:

```python
    run_stats = manifest.get("run_stats")
    node.run_stats = run_stats if isinstance(run_stats, dict) else None
```

- [ ] **Step 6: Run tests to verify Task 1 passes**

Run:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_parse_result_marker_preserves_run_stats tests/test_save_run_artifacts.py::test_save_run_writes_node_run_stats_to_manifest
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add aide/journal.py aide/agent.py aide/utils/artifact_manifest.py tests/test_autogluon_preprocess.py tests/test_save_run_artifacts.py
git commit -m "Persist node run stats in manifests"
```

---

### Task 2: Collect AutoGluon `run_stats` In The Wrapper

**Files:**
- Modify: `aide/autogluon_preprocess.py`
- Test: `tests/test_autogluon_preprocess.py`

- [ ] **Step 1: Write failing source-shape test for wrapper stats**

Add to `tests/test_autogluon_preprocess.py` near `test_build_autogluon_wrapper_compiles_and_preserves_preprocess`:

```python
def test_build_autogluon_wrapper_emits_run_stats_collection(tmp_path):
    cfg = _cfg(tmp_path)
    code = build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg)

    assert "import time" in code
    assert "preprocess_started_at = time.time()" in code
    assert "preprocess_time = time.time() - preprocess_started_at" in code
    assert "feature_count = int(len(preprocessed.columns))" in code
    assert "training_started_at = time.time()" in code
    assert "training_time = time.time() - training_started_at" in code
    assert "leaderboard = predictor.leaderboard(silent=True)" in code
    assert "\"run_stats\": run_stats" in code
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_build_autogluon_wrapper_emits_run_stats_collection
```

Expected: FAIL because wrapper code does not yet collect or emit `run_stats`.

- [ ] **Step 3: Add JSON-safe leaderboard helpers to generated wrapper**

In `aide/autogluon_preprocess.py`, inside the generated wrapper source, add `import time` next to the other generated imports:

```python
import time
```

Add these generated helper functions before `main()`:

```python
def _json_safe_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _leaderboard_records(predictor: TabularPredictor) -> list[dict]:
    keep_columns = [
        "model",
        "score_val",
        "score_test",
        "eval_metric",
        "fit_time",
        "fit_time_marginal",
        "pred_time_val",
        "pred_time_val_marginal",
        "stack_level",
        "can_infer",
        "fit_order",
    ]
    try:
        leaderboard = predictor.leaderboard(silent=True)
    except Exception as exc:
        return [{"error": f"leaderboard unavailable: {exc}"}]
    records = []
    for row in leaderboard.to_dict(orient="records"):
        records.append(
            {
                column: _json_safe_scalar(row.get(column))
                for column in keep_columns
                if column in row
            }
        )
    return records
```

- [ ] **Step 4: Measure preprocess time and feature count**

In generated `main()`, replace:

```python
    print("AIDE AutoGluon: starting preprocess", flush=True)
    with _preprocess_timeout(int(AIDE_AG_CONFIG.get("preprocess_timeout", 180))):
        preprocessed = preprocess(combined.copy())
    print(
        f"AIDE AutoGluon: finished preprocess rows={len(preprocessed)} cols={len(preprocessed.columns)}",
        flush=True,
    )
```

with:

```python
    print("AIDE AutoGluon: starting preprocess", flush=True)
    preprocess_started_at = time.time()
    with _preprocess_timeout(int(AIDE_AG_CONFIG.get("preprocess_timeout", 180))):
        preprocessed = preprocess(combined.copy())
    preprocess_time = time.time() - preprocess_started_at
    feature_count = int(len(preprocessed.columns))
    print(
        f"AIDE AutoGluon: finished preprocess rows={len(preprocessed)} cols={feature_count}",
        flush=True,
    )
```

- [ ] **Step 5: Measure training time and collect leaderboard records**

In generated `main()`, wrap `predictor.fit(**fit_kwargs)`:

```python
    training_started_at = time.time()
    with _quiet_model_output(working_dir):
        print("AIDE AutoGluon: starting fit", flush=True)
        predictor = TabularPredictor(
            label=target_col,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=str(model_dir),
            verbosity=2,
        )
        predictor.fit(**fit_kwargs)
        print("AIDE AutoGluon: finished fit", flush=True)
    training_time = time.time() - training_started_at
    model_records = _leaderboard_records(predictor)
```

Keep validation/prediction behavior unchanged.

- [ ] **Step 6: Emit `run_stats` in the result marker**

Before the final `print(RESULT_MARKER + ...)`, construct:

```python
    run_stats = {
        "feature_count": feature_count,
        "preprocess_time": float(preprocess_time),
        "training_time": float(training_time),
        "models": model_records,
    }
```

Then add it to the JSON payload:

```python
        "run_stats": run_stats,
```

Do not add `total_exec_time` inside the wrapper; the interpreter only knows total execution time outside the generated code. Task 1 manifest persistence adds `total_exec_time` to `aide_result.json` from `node.exec_time`.

- [ ] **Step 7: Run Task 2 tests**

Run:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_build_autogluon_wrapper_emits_run_stats_collection tests/test_autogluon_preprocess.py::test_build_autogluon_wrapper_compiles_and_preserves_preprocess
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add aide/autogluon_preprocess.py tests/test_autogluon_preprocess.py
git commit -m "Collect AutoGluon run stats"
```

---

### Task 3: Add Table/Path Builders For Hypothesis Views

**Files:**
- Modify: `aide/run.py`
- Test: `tests/test_run_tree.py`

- [ ] **Step 1: Write failing tests for root/all/branch builders**

Add these imports to `tests/test_run_tree.py`:

```python
    build_all_hypotheses_view,
    build_best_branch_view,
    build_root_hypotheses_view,
```

Add tests near the existing tree view tests:

```python
def test_root_hypotheses_view_sorts_scored_roots_by_score():
    journal = Journal()
    baseline = Node(code="base", plan=f"{BASELINE_PLAN_PREFIX}: raw")
    baseline.metric = MetricValue(0.99, maximize=True)
    baseline.is_buggy = False
    journal.append(baseline)
    weak = _hypothesis_node(_good_node(0.95000), "000111")
    strong = _hypothesis_node(_good_node(0.95200), "000222")
    child = _hypothesis_node(_good_node(0.95300, parent=weak), "000333")
    journal.append(weak)
    journal.append(strong)
    journal.append(child)

    view = build_root_hypotheses_view(journal)
    output = _render_text(
        render_tree_view(view, focused_item_id="header", scroll_top=0, viewport_height=10)
    )

    assert "Root hypotheses" in output
    assert "0.95200" in output
    assert "000222" in output
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
        render_tree_view(view, focused_item_id="header", scroll_top=0, viewport_height=10)
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
        render_tree_view(view, focused_item_id="header", scroll_top=0, viewport_height=10)
    )

    assert "Best branch" in output
    assert "000111 0.95100 -> 000222 0.95200 -> 000333 0.95300" in output
    assert "000444" not in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_run_tree.py::test_root_hypotheses_view_sorts_scored_roots_by_score tests/test_run_tree.py::test_all_hypotheses_view_aggregates_successful_usage_by_hypothesis tests/test_run_tree.py::test_best_branch_view_renders_top_path_root_to_leaf
```

Expected: FAIL because the builders do not exist.

- [ ] **Step 3: Add small formatting helpers**

In `aide/run.py`, after `recover_tree_focus_by_index(...)`, add:

```python
def _is_scored_hypothesis_node(node: Node) -> bool:
    return (
        not node.is_buggy
        and not node.is_terminal_failure
        and node.metric is not None
        and node.metric.value is not None
        and hypothesis_id_for_node(node) is not None
    )


def _metric_sort_key(node: Node) -> float:
    assert node.metric is not None
    assert node.metric.value is not None
    value = float(node.metric.value)
    return value if node.metric.maximize is not False else -value


def _score_text(node: Node) -> str:
    assert node.metric is not None
    assert node.metric.value is not None
    return f"{node.metric.value:.5f}"


def _plain_line_view(title: str, lines: list[Text]) -> TreeView:
    items = [TreeViewItem("header", None, Text(title, style="bold blue"), focus_start=0)]
    children_by_id: dict[str, list[str]] = {"header": []}
    parent_by_id: dict[str, str | None] = {"header": None}
    for index, line in enumerate(lines):
        item_id = f"row:{index}"
        items.append(TreeViewItem(item_id, "header", line, focus_start=0))
        children_by_id["header"].append(item_id)
        children_by_id[item_id] = []
        parent_by_id[item_id] = "header"
    return TreeView(
        items=items,
        index_by_id={item.item_id: index for index, item in enumerate(items)},
        parent_by_id=parent_by_id,
        children_by_id=children_by_id,
    )
```

- [ ] **Step 4: Implement root table builder**

Add below the helpers:

```python
def build_root_hypotheses_view(journal: Journal) -> TreeView:
    roots = [
        node
        for node in journal.nodes
        if node.parent is None
        and not _is_base_root(node)
        and _is_scored_hypothesis_node(node)
    ]
    roots.sort(key=_metric_sort_key, reverse=True)
    lines = [Text("score     hypothesis", style=TUI_ROW_LABEL_STYLE)]
    for node in roots:
        hypothesis_id = hypothesis_id_for_node(node) or "n/a"
        line = Text()
        line.append(f"{_score_text(node):<9}", style=TUI_METRIC_VALUE_STYLE)
        line.append(hypothesis_id, style=TUI_NEUTRAL_VALUE_STYLE)
        lines.append(line)
    if not roots:
        lines.append(Text("n/a", style=TUI_INACTIVE_VALUE_STYLE))
    return _plain_line_view("Root hypotheses", lines)
```

- [ ] **Step 5: Implement all-hypotheses aggregation builder**

Add below `build_root_hypotheses_view`:

```python
def build_all_hypotheses_view(journal: Journal) -> TreeView:
    rows: dict[str, dict[str, object]] = {}
    for node in journal.nodes:
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is None or not _is_scored_hypothesis_node(node):
            continue
        row = rows.setdefault(
            hypothesis_id,
            {
                "hypothesis_id": hypothesis_id,
                "best_node": node,
                "uses_total": 0,
                "root_uses": 0,
                "branch_uses": 0,
            },
        )
        row["uses_total"] = int(row["uses_total"]) + 1
        if node.parent is None:
            row["root_uses"] = int(row["root_uses"]) + 1
        else:
            row["branch_uses"] = int(row["branch_uses"]) + 1
        best_node = row["best_node"]
        assert isinstance(best_node, Node)
        if _metric_sort_key(node) > _metric_sort_key(best_node):
            row["best_node"] = node

    sorted_rows = sorted(
        rows.values(),
        key=lambda row: _metric_sort_key(cast(Node, row["best_node"])),
        reverse=True,
    )
    lines = [Text("best      hypothesis  uses", style=TUI_ROW_LABEL_STYLE)]
    for row in sorted_rows:
        best_node = cast(Node, row["best_node"])
        line = Text()
        line.append(f"{_score_text(best_node):<10}", style=TUI_METRIC_VALUE_STYLE)
        line.append(f"{row['hypothesis_id']:<12}", style=TUI_NEUTRAL_VALUE_STYLE)
        line.append(
            f"{row['uses_total']} (root {row['root_uses']}, branch {row['branch_uses']})",
            style=TUI_INACTIVE_VALUE_STYLE,
        )
        lines.append(line)
    if not sorted_rows:
        lines.append(Text("n/a", style=TUI_INACTIVE_VALUE_STYLE))
    return _plain_line_view("All hypotheses", lines)
```

`cast` is already imported in `aide/run.py`; use it here.

- [ ] **Step 6: Implement best branch path builder**

Add below `build_all_hypotheses_view`:

```python
def build_best_branch_view(journal: Journal) -> TreeView:
    best_node = _best_scored_node(journal)
    if best_node is None:
        return _plain_line_view(
            "Best branch",
            [Text("n/a", style=TUI_INACTIVE_VALUE_STYLE)],
        )

    path: list[Node] = []
    node: Node | None = best_node
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()

    line = Text()
    for index, path_node in enumerate(path):
        if index:
            line.append(" -> ", style=TUI_SEPARATOR_STYLE)
        line.append(hypothesis_id_for_node(path_node) or "n/a", style=TUI_NEUTRAL_VALUE_STYLE)
        line.append(f" {_score_text(path_node)}", style=TUI_METRIC_VALUE_STYLE)
    return _plain_line_view("Best branch", [line])
```

- [ ] **Step 7: Run Task 3 tests**

Run:

```bash
uv run pytest tests/test_run_tree.py::test_root_hypotheses_view_sorts_scored_roots_by_score tests/test_run_tree.py::test_all_hypotheses_view_aggregates_successful_usage_by_hypothesis tests/test_run_tree.py::test_best_branch_view_renders_top_path_root_to_leaf
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add aide/run.py tests/test_run_tree.py
git commit -m "Add hypothesis table view builders"
```

---

### Task 4: Wire `v` View Cycling Into The Live TUI

**Files:**
- Modify: `aide/run.py`
- Test: `tests/test_run_tree.py`

- [ ] **Step 1: Write failing tests for key mapping and view cycling**

In `tests/test_run_tree.py`, add `next_left_panel_view` to the import list from `aide.run`.

Add tests near `test_keyboard_reader_maps_f_to_follow_toggle`:

```python
def test_keyboard_reader_maps_v_to_view_toggle():
    assert ArrowKeyReader.CHAR_KEY_MAP[b"v"] == "view"
    assert ArrowKeyReader.CHAR_KEY_MAP[b"V"] == "view"


def test_next_left_panel_view_cycles_in_declared_order():
    assert next_left_panel_view("tree") == "root"
    assert next_left_panel_view("root") == "all"
    assert next_left_panel_view("all") == "branch"
    assert next_left_panel_view("branch") == "tree"
    assert next_left_panel_view("unknown") == "tree"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_run_tree.py::test_keyboard_reader_maps_v_to_view_toggle tests/test_run_tree.py::test_next_left_panel_view_cycles_in_declared_order
```

Expected: FAIL because `v` and `next_left_panel_view` do not exist.

- [ ] **Step 3: Add view-mode constants and cycle helper**

In `aide/run.py`, near TUI constants, add:

```python
LeftPanelView = Literal["tree", "root", "all", "branch"]
LEFT_PANEL_VIEW_ORDER: tuple[LeftPanelView, ...] = ("tree", "root", "all", "branch")


def next_left_panel_view(current: str) -> LeftPanelView:
    if current not in LEFT_PANEL_VIEW_ORDER:
        return "tree"
    index = LEFT_PANEL_VIEW_ORDER.index(cast(LeftPanelView, current))
    return LEFT_PANEL_VIEW_ORDER[(index + 1) % len(LEFT_PANEL_VIEW_ORDER)]
```

This uses existing `Literal` and `cast` imports.

- [ ] **Step 4: Add keyboard mapping**

In `ArrowKeyReader.CHAR_KEY_MAP`, add:

```python
        b"v": "view",
        b"V": "view",
```

- [ ] **Step 5: Add TUI state for left view and table focus**

In `run()` near existing state:

```python
    focused_tree_item_id = "header"
    focused_tree_item_index = 0
    tree_scroll_top = 0
    tree_follow_mode = "off"
```

add:

```python
    left_panel_view: LeftPanelView = "tree"
    focused_table_item_id = "header"
    focused_table_item_index = 0
    table_scroll_top = 0
```

- [ ] **Step 6: Add current view builder**

Inside `run()`, near `current_tree_view`, add:

```python
    def current_left_panel_view(*, blink_on: bool) -> TreeView:
        if left_panel_view == "tree":
            return current_tree_view(blink_on=blink_on)
        if left_panel_view == "root":
            return build_root_hypotheses_view(journal)
        if left_panel_view == "all":
            return build_all_hypotheses_view(journal)
        return build_best_branch_view(journal)
```

- [ ] **Step 7: Split navigation into tree and table behavior**

Rename `drain_tree_navigation(view: TreeView)` to `drain_left_panel_navigation(view: TreeView)`.

Inside it, declare:

```python
        nonlocal focused_tree_item_id, focused_tree_item_index, tree_scroll_top
        nonlocal focused_table_item_id, focused_table_item_index, table_scroll_top
        nonlocal tree_follow_mode, left_panel_view
```

At the top of the function, branch on `left_panel_view`.

For table modes, use this behavior:

```python
        if left_panel_view != "tree":
            if focused_table_item_id not in view.index_by_id:
                focused_table_item_id = recover_tree_focus_by_index(
                    view,
                    fallback_index=focused_table_item_index,
                )
            while key_reader is not None:
                key = key_reader.read_key()
                if key is None:
                    break
                if key == "view":
                    left_panel_view = next_left_panel_view(left_panel_view)
                    focused_table_item_id = "header"
                    focused_table_item_index = 0
                    table_scroll_top = 0
                    return
                if key in {"up", "down"}:
                    focused_table_item_id = move_tree_focus(view, focused_table_item_id, key)
            focus_index = view.index_by_id.get(focused_table_item_id, 0)
            table_scroll_top = clamp_tree_viewport(
                total_lines=len(view.items),
                viewport_height=tree_viewport_height(),
                focus_index=focus_index,
                current_scroll=table_scroll_top,
            )
            focused_table_item_index = view.index_by_id.get(focused_table_item_id, 0)
            return
```

For tree mode, keep existing behavior, but add this keyboard branch before `follow`:

```python
            if key == "view":
                left_panel_view = next_left_panel_view(left_panel_view)
                focused_table_item_id = "header"
                focused_table_item_index = 0
                table_scroll_top = 0
                continue
```

Keep `follow`, `best`, `active`, and arrow behavior tree-only.

- [ ] **Step 8: Render selected left-panel view**

In `generate_live()`, replace:

```python
        tree_view = current_tree_view(blink_on=blink_on)
        drain_tree_navigation(tree_view)
        tree = render_tree_view(
            tree_view,
            focused_item_id=focused_tree_item_id,
            scroll_top=tree_scroll_top,
            viewport_height=tree_viewport_height(),
        )
```

with:

```python
        left_view = current_left_panel_view(blink_on=blink_on)
        drain_left_panel_navigation(left_view)
        if left_panel_view == "tree":
            focused_item_id = focused_tree_item_id
            scroll_top = tree_scroll_top
        else:
            focused_item_id = focused_table_item_id
            scroll_top = table_scroll_top
        left_panel_content = render_tree_view(
            left_view,
            focused_item_id=focused_item_id,
            scroll_top=scroll_top,
            viewport_height=tree_viewport_height(),
        )
```

Then change the panel body:

```python
            Padding(left_panel_content, (0, 1, 0, 1)),
```

Keep the panel name `tree` in the layout for now to avoid unrelated layout churn.

- [ ] **Step 9: Update subtitle**

Replace the left-panel subtitle with:

```python
            subtitle=(
                "↑/↓ move  ← parent  → child  b best  a active  "
                f"f follow:{tree_follow_mode}  v view:{left_panel_view}  Ctrl+C stop"
            ),
```

This displays tree-only controls in every mode for consistency. Table modes ignore tree-only controls.

- [ ] **Step 10: Update tick callback calls**

Replace calls like:

```python
tick=lambda: drain_tree_navigation(current_tree_view(blink_on=True))
```

with:

```python
tick=lambda: drain_left_panel_navigation(current_left_panel_view(blink_on=True))
```

There are several occurrences in `run()` around generation, execution, review, synthesis, and checkpoint paths. Update all of them.

- [ ] **Step 11: Run focused TUI tests**

Run:

```bash
uv run pytest tests/test_run_tree.py::test_keyboard_reader_maps_v_to_view_toggle tests/test_run_tree.py::test_next_left_panel_view_cycles_in_declared_order tests/test_run_tree.py::test_render_tree_view_highlights_focused_line_and_slices_viewport
```

Expected: PASS.

- [ ] **Step 12: Run all run-tree tests**

Run:

```bash
uv run pytest tests/test_run_tree.py
```

Expected: PASS.

- [ ] **Step 13: Commit Task 4**

Run:

```bash
git add aide/run.py tests/test_run_tree.py
git commit -m "Add TUI view cycling"
```

---

### Task 5: Final Verification

**Files:**
- Verify only; no planned edits.

- [ ] **Step 1: Run focused test modules**

Run:

```bash
uv run pytest tests/test_run_tree.py tests/test_autogluon_preprocess.py tests/test_save_run_artifacts.py
```

Expected: PASS.

- [ ] **Step 2: Run lint on touched files**

Run:

```bash
uv run ruff check aide/run.py aide/journal.py aide/agent.py aide/autogluon_preprocess.py aide/utils/artifact_manifest.py tests/test_run_tree.py tests/test_autogluon_preprocess.py tests/test_save_run_artifacts.py
```

Expected: PASS.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: clean working tree after the task commits.

- [ ] **Step 4: Push if requested**

Only if the user asks for push, run:

```bash
git push
```

Expected: branch `main` pushed successfully.

---

## Self-Review

Spec coverage:

- `v` key and cyclic view order: Task 4.
- Default `tree` view: Task 4.
- Root table: Task 3.
- All hypotheses table: Task 3.
- Best branch path view: Task 3.
- No active placeholder in tables: Task 3 by building from completed `journal.nodes` only.
- New AutoGluon stats persisted but not displayed, including manifest-level `total_exec_time`: Tasks 1 and 2.
- No live log/model-dir scanning: Tasks 3 and 4 use only `journal`; Tasks 1 and 2 write data at execution time.

Placeholder scan:

- No `TBD`, `TODO`, or open placeholders.

Type consistency:

- `Node.run_stats` is a `dict | None`.
- `LeftPanelView` is a `Literal["tree", "root", "all", "branch"]`.
- New table builders return existing `TreeView`, so `render_tree_view` and viewport helpers remain reusable.
