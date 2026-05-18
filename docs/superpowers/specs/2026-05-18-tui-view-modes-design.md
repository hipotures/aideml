# TUI View Modes And AutoGluon Run Stats Design

## Goal

Add a cyclic `v` key to switch the left TUI panel between the current solution tree and three hypothesis-focused summary views. Also persist new AutoGluon execution statistics for future analysis, but do not display those runtime/model details in the new TUI tables yet.

The design keeps the live TUI responsive by deriving visible tables from in-memory `journal` data and by reading at most one manifest file per node when manifest-backed data is needed later. It must not scan logs, model directories, or artifact subtrees on every refresh.

## Non-Goals

- Do not replace the existing tree renderer.
- Do not add model leaderboard columns to the TUI tables in this iteration.
- Do not infer missing old-run runtime statistics from logs.
- Do not scan `autogluon_stdout.log`, `metadata.json`, `version.txt`, or AutoGluon model directories from the live TUI.
- Do not add a new experiment-selection algorithm.

## Left Panel View Modes

The left panel gets a new state named `left_panel_view`, defaulting to `tree`.

Pressing `v` cycles views in this fixed order:

1. `tree`
2. `root`
3. `all`
4. `branch`
5. back to `tree`

The left-panel subtitle should include the current mode:

```text
↑/↓ move  ← parent  → child  b best  a active  f follow:off  v view:tree  Ctrl+C stop
```

The current tree view preserves existing behavior, focus, viewport, best-node jump, active-node jump, and follow mode. Table views may support `↑/↓` row movement, but `b`, `a`, and `f` are tree-only controls in the first implementation. In table modes they should not change tree focus or view state.

## Root Table

The `root` view ranks root hypotheses by their root-node score.

Data source:

- `journal.nodes`

Rows:

- Include scored nodes with `parent is None`.
- Include only nodes that have a real `metric.value`.
- Exclude buggy and terminal failure nodes.
- Exclude baseline and seeded-base roots because they are not hypothesis root tests.
- Include only nodes with a hypothesis id from `research_hypotheses_offered`.

Sort:

- Best score first.
- Use `metric.maximize` to decide direction. In the current Kaggle ROC AUC use case, this means descending.

Columns:

- `score`
- `hypothesis`

Runtime and model statistics are intentionally not shown here yet.

## All Hypotheses Table

The `all` view ranks hypotheses across root and branch usage.

Data source:

- `journal.nodes`

Rows:

- Aggregate by hypothesis id from `research_hypotheses_offered`.
- Count only nodes with exactly one hypothesis id, matching current `research.mode=hypothesis` behavior.
- Count only completed nodes already present in the journal; do not show the active placeholder because it has no score yet.
- Ignore buggy/failed nodes for `best_score`, but keep the usage count scoped to successful scored nodes in this iteration. Bug/failure counts can be added later if needed.

Per-hypothesis fields:

- `best_score`
- `uses_total`
- `root_uses`
- `branch_uses`

Sort:

- Best score first using the same maximize/minimize logic as the root table.

Columns:

- `best score`
- `hypothesis`
- `uses`, formatted like `12 (root 1, branch 11)`

## Best Branch Path View

The `branch` view shows the ancestor path that leads to the current best scored node.

Data source:

- `journal.nodes`
- parent links already reconstructed in the journal

Selection:

- Find the best non-buggy scored node.
- Walk its parents back to the root.
- Reverse the sequence to render root-to-leaf order.

Rendering:

- Render horizontal path segments, not a tree.
- Each segment shows hypothesis id and score.
- If the terminal width is too narrow, wrap onto the next line while preserving path order.

Example:

```text
000002 0.95000 -> 000459 0.95217 -> 000443 0.95217
```

If a path node has no hypothesis id, show `n/a` for the hypothesis part. If no scored node exists, show a short empty-state line.

## AutoGluon Run Stats Persistence

New AutoGluon preprocess-wrapper runs should persist runtime/model data into the node result manifest for later analysis. The TUI table views do not display these fields in this iteration.

Storage:

- Add a JSON-safe `run_stats` object to `aide_result.json`.
- Also preserve the data through the result marker path when possible so the manifest can be written without parsing logs.

Suggested shape:

```json
{
  "run_stats": {
    "feature_count": 50,
    "preprocess_time": 1.23,
    "training_time": 49.61,
    "total_exec_time": 62.8,
    "models": [
      {
        "model": "WeightedEnsemble_L2",
        "score_val": 0.951,
        "fit_time": 0.4,
        "pred_time_val": 0.02,
        "stack_level": 2
      }
    ]
  }
}
```

Field definitions:

- `feature_count`: number of columns after `preprocess(df)` and validation.
- `preprocess_time`: wall time around the `preprocess(combined.copy())` call.
- `training_time`: wall time around `predictor.fit(...)`.
- `total_exec_time`: total interpreter execution time already available on the node, copied for convenience when writing the manifest.
- `models`: rows from `predictor.leaderboard(silent=True)` after fit, reduced to JSON-safe scalar fields.

Model leaderboard columns to keep when present:

- `model`
- `score_val`
- `score_test`
- `eval_metric`
- `fit_time`
- `fit_time_marginal`
- `pred_time_val`
- `pred_time_val_marginal`
- `stack_level`
- `can_infer`
- `fit_order`

If AutoGluon trains an ensemble, for example `WeightedEnsemble_L2`, it should appear naturally in the `models` list. If no ensemble is trained, no ensemble row is invented.

Missing data behavior:

- New code should write `null` for fields that were attempted but unavailable.
- Old manifests may have no `run_stats`.
- TUI and future analysis code must treat missing `run_stats` as `n/a`.

## Data Access Rules

Live TUI refresh must not scan artifact directories beyond the single manifest file associated with a node. For this iteration, the new table views should not need manifests at all because they use only `journal` fields.

Allowed live sources:

- in-memory `journal`
- active node state already tracked by the agent
- one `aide_result.json` per node only when future runtime/model views need it

Disallowed live sources:

- recursive artifact scans
- log parsing
- AutoGluon model directory reads
- `metadata.json`
- `version.txt`

## Testing Plan

Unit tests should cover:

- `v` maps to a view-switch action.
- View cycling order is `tree -> root -> all -> branch -> tree`.
- The default view is `tree`.
- Root table includes only scored hypothesis roots and sorts by best score.
- All table aggregates usage counts and best score by hypothesis id.
- Branch view renders the best-node path in root-to-leaf order.
- Active placeholders are not shown in table views.
- AutoGluon result marker includes `run_stats` for new wrapper runs.
- Manifest writing preserves `run_stats`.
- Missing `run_stats` renders or serializes as absent/null without errors.

## Open Decisions

None. The first implementation deliberately stores AutoGluon runtime/model stats without displaying them in the new TUI tables.
