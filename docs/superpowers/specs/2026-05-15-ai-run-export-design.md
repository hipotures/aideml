# AI Run Export Design

## Goal

Create a dedicated exporter that turns one complete AIDE run into an attachment-friendly dataset for external AI review. The target consumer is a strong external model that should inspect the whole search history and suggest what to try next.

The exporter must preserve the full run tree. Low-scoring, buggy, duplicate, and near-duplicate nodes remain in the output because their position and failures are useful search evidence.

## Non-Goals

- Do not prune nodes from the export.
- Do not replace duplicate nodes with `code_ref` in the default output.
- Do not change AIDE search behavior, prompts, scoring, or submission logic.
- Do not build a visual tree exporter in this iteration.

## Output Shape

The command writes one export directory:

```text
exports/<run-id>-<export-timestamp>/
  run_export.meta.json
  run_export.nodes.jsonl
```

`run_export.nodes.jsonl` contains one JSON object per node, ordered by `step`. JSONL is preferred over one huge JSON document because it is easier to inspect, stream, and recover if a single record is malformed.

`run_export.meta.json` contains run-level context, summary statistics, best local/public records, and enough schema metadata for an AI reviewer to understand the node file.

## CLI

Add a dedicated script:

```bash
uv run python scripts/export_run_for_ai.py logs/2-enthusiastic-crane-of-completion
```

Initial options:

```text
--output-dir exports
--near-submission-rmse-threshold 1e-6
--prediction-similarity-sample-size 200
--prediction-similarity-min-common-sample-size 100
--no-near-duplicates
```

Defaults:

- include all nodes, including bugs and failures
- include full code in every node
- compute exact code duplicate groups
- compute exact submission duplicate groups
- compute near-submission duplicate hints unless disabled

## Node Record

Each node record should include:

```json
{
  "step": 61,
  "node_id": "...",
  "parent_id": "...",
  "children_ids": ["..."],
  "depth": 4,
  "status": "ok",
  "is_buggy": false,
  "is_terminal_failure": false,
  "origin": "source_node",
  "local_cv_score": 0.9511,
  "kaggle_public_score": 0.94896,
  "metric_maximize": true,
  "created_at": "2026-05-06T09:40:19+02:00",
  "exec_time": 80.4,
  "artifact_dir": "logs/.../artifacts/...",
  "code_sha256": "...",
  "submission_sha256": "...",
  "duplicate": {
    "exact_code_group": "...",
    "exact_code_role": "canonical",
    "exact_code_canonical_node_id": "...",
    "exact_submission_group": "...",
    "exact_submission_role": "duplicate",
    "exact_submission_canonical_node_id": "...",
    "near_submission_canonical_node_id": "...",
    "near_submission_rmse": 0.000003
  },
  "plan": "...",
  "analysis": "...",
  "validity_warning": null,
  "error": {
    "exc_type": null,
    "summary": null
  },
  "code": "..."
}
```

The exact field set can grow, but these fields are the stable initial contract.

## Metadata Record

`run_export.meta.json` should include:

```json
{
  "schema_version": 1,
  "run": "2-enthusiastic-crane-of-completion",
  "exported_at": "2026-05-15T12:00:00+02:00",
  "node_count": 632,
  "scored_node_count": 603,
  "best_local": {
    "step": 61,
    "node_id": "...",
    "local_cv_score": 0.9511
  },
  "best_public": {
    "step": 0,
    "node_id": "...",
    "kaggle_public_score": 0.95168,
    "submission_sha256": "..."
  },
  "config": {},
  "notes_for_ai": "This is a complete AIDE tree export. Nodes are ordered by step and connected by parent_id/children_ids. Duplicate hints are advisory; no node was pruned."
}
```

## Public Score Mapping

Public score should be read from `logs/submission_registry.json` and matched to nodes by:

1. `node_id`, when present
2. `run + step + timestamp`
3. `submission_sha256`, including 10+ character SHA prefixes from Kaggle descriptions

If multiple public scores match the same node, keep the best completed score for maximize metrics and include enough metadata to explain the source.

## Duplicate Hints

Duplicate handling is advisory only. It must never remove or reorder nodes.

Exact code duplicates:

- normalize line endings
- hash the full code text
- assign an `exact_code_group`
- choose the canonical node as the earliest node in the group, unless a later node has a better local score

Exact submission duplicates:

- group by `submission_sha256`
- choose the canonical node as the best local score in the group, falling back to earliest step

Near submission duplicates:

- reuse the existing sampled RMSE logic from `aide.utils.prediction_similarity`
- default sample size: 200
- default minimum common sample size: 100
- default threshold: `1e-6`
- write only hints: canonical node id and RMSE

Near code duplicates are intentionally deferred. Code-level near-duplicate heuristics are more likely to produce misleading matches than sampled submission RMSE.

## Error Handling

- Missing `journal.json`: fail with a clear error.
- Missing artifact directory: keep the node and set artifact fields to null.
- Missing `submission.csv`: keep the node and set submission hash/duplicate fields to null.
- Malformed registry: fail with a clear error unless a `--no-public-scores` option is added later.
- RMSE read failure for a pair: skip that near-duplicate hint and continue.

## Testing

Add focused tests for:

- exporting a small three-node tree with parent/child/depth preserved
- keeping buggy and low-scoring nodes by default
- mapping public score by `node_id`
- mapping public score by SHA prefix
- exact code duplicate annotations without pruning
- exact submission duplicate annotations without pruning
- near-submission duplicate annotation using a small synthetic submission pair
- graceful handling of missing submission artifacts

## Success Criteria

- A complete run can be exported into JSONL plus metadata with all nodes present.
- Parent-child relationships are reconstructable from the export.
- GPT Pro can inspect every node's code and see duplicate hints without losing tree context.
- Public scores are included when available and do not confuse profile/seed artifacts with ordinary tree nodes.
- Existing AIDE run behavior is unchanged.
