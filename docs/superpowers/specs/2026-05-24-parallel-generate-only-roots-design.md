# Parallel Generate-Only Hypothesis Root Generation

Date: 2026-05-24

## Goal

Add an opt-in way to generate multiple hypothesis ROOT code candidates concurrently in `--generate-only` runs.

The default remains serial generation with one worker, matching current behavior unless the user explicitly enables more workers.

## Configuration

Add:

```yaml
research:
  hypothesis_root_generate_workers: 1
```

Command-line usage:

```bash
uv run aide --generate-only ... research.hypothesis_root_generate_workers=4
```

Valid values are integers from `1` through `8`. Values below `1`, above `8`, or non-integers are configuration errors and should stop the run before generation starts.

The option only affects:

- `--generate-only`
- `research.mode=hypothesis`
- opening new ROOT hypothesis nodes

It must not parallelize execution, review, debugging, synthesis, or child hypothesis generation.

## Recommended Architecture

Use a single-writer, parallel-worker design:

1. The main thread decides that the run should open ROOT hypothesis nodes.
2. The main thread reserves up to `hypothesis_root_generate_workers` ROOT hypotheses serially.
3. Each reserved hypothesis is assigned a stable future step offset, such as `len(journal.nodes) + batch_index`.
4. Worker agents generate code concurrently for their assigned hypothesis only.
5. The main thread collects worker results and writes journal entries, hypothesis code manifests, and UI state changes serially.

This keeps the expensive LLM calls parallel while preserving deterministic selection and file writes.

When a worker finishes, the main thread should refill the pool immediately if more ROOT hypotheses are available and the root limit has not been reached. With `hypothesis_root_generate_workers=8`, the run tries to keep eight active ROOT generations in flight. This is a rolling pool, not fixed batches of eight.

## Hypothesis Selection

Workers must not call the normal selector independently.

The reservation step must preserve existing ordering semantics:

- `research.hypothesis_root_order=default` keeps the existing seeded randomized order with usage de-duplication.
- `research.hypothesis_root_order=manifest_score` keeps the existing score-prioritized order.
- `research.hypothesis_root_score_mode` continues to decide which manifest mode supplies scores.
- `research.ignore_hypothesis_agent_modes` continues to control compatibility filtering.
- `research.hypothesis_root_limit` continues to cap total generated ROOT hypotheses.

Reservation records offer usage before worker launch so interrupted or in-flight candidates are not reselected by the same batch or by a resumed run.

## Writes And Concurrency

Only the main thread may write:

- `journal.json`
- generated-only journal nodes
- `research_hypotheses/.../code_manifest.json`
- newly saved `legacy-NNN.py` or `autogluon-NNN.py`
- `logs/<run>/research_hypotheses/usage.json`
- `logs/<run>/research_hypotheses/offers.jsonl`
- search decision records

Workers may write only their own LLM request/response logs under their assigned artifact directory.

Each worker gets a unique artifact directory before generation starts.

Current artifact paths are derived from `node.ctime` with second precision:

```text
logs/<run>/artifacts/YYYYMMDDTHHMMSS
```

Parallel workers launched in the same wall-clock second would otherwise collide. Do not solve this by changing `ctime`, because `ctime` should remain the node creation time, not an artifact-directory allocator.

Use explicit artifact directory names for new generated nodes. The preferred format is:

```text
logs/<run>/artifacts/YYYYMMDDTHHMMSS-<8hex>
```

The timestamp prefix keeps the directory sortable and human-readable. The random hex suffix prevents collisions for parallel workers. A full UUID is acceptable internally, but the short suffix is enough if generated from a UUID or cryptographically strong random value.

Every new node should receive an explicit artifact directory name, not only nodes created by this parallel generate-only feature. Add the field to the serialized `Node`, for example:

```python
artifact_dir_name: str | None = None
```

The field stores only the directory basename, not an absolute path. This keeps run directories movable and keeps existing log path configuration authoritative.

Artifact lookup then becomes:

1. if the node has an explicit artifact directory name, use it;
2. otherwise fall back to the legacy `ctime -> YYYYMMDDTHHMMSS` path.

This keeps all existing runs readable and lets new runs mix old timestamp artifacts with new unique artifact directories during resume.

## Agent State

The existing `Agent` instance has single-node mutable state such as `active_node`, `active_parent_node`, `active_stage`, and active hypothesis display fields. It must not be shared by concurrent workers.

Each worker should use an isolated generation context or a separate agent instance with read-only access to the shared config and a journal snapshot. Workers receive a preselected hypothesis, not a request to run search policy.

For `hypothesis_root_generate_workers=1`, the implementation should keep the current serial path or a behaviorally identical path.

## TUI Behavior

The solution tree should render every in-flight ROOT generation as a root-level placeholder until it becomes a normal generated node.

Example while a batch is running:

```text
Solution tree
├── generated·000528
├── [ ]·000405
├── [ ]·000941
└── [*]·001104
```

`follow:active` should track the most recently launched in-flight hypothesis. If workers 1 through 8 start together, it follows worker 8. If five workers finish and a new worker is launched to refill the pool to eight, follow moves to that newly launched hypothesis.

When an in-flight placeholder is committed to the journal, the placeholder is replaced by the normal generated node and must not appear twice.

The run data panel should make parallel generation visible, for example:

```text
Agent
▶ mode      legacy
▶ run       generate-only
▶ workers   4
```

## Failure Handling

A failed worker must not corrupt the batch or block successful workers from being saved.

On worker failure:

- retry the same hypothesis up to three total attempts;
- sleep 5 seconds before retrying;
- keep the same reserved hypothesis id, but use a fresh artifact directory for each retry attempt;
- log the hypothesis id, attempt number, exception type, and artifact directory for each failed attempt;
- do not write a journal node for attempts that fail before producing code;
- keep successful generated nodes saving serially.

If the same hypothesis fails three times, stop launching new workers and let the current in-flight workers finish or fail. This prevents a broad network, disk, or Codex outage from turning a run into hundreds of failed hypothesis attempts.

Failed reservations must be retried first on the next run or resume before selecting fresh hypotheses. Store enough run-local state to know which reserved ROOT hypotheses failed generation and how many attempts were made. This retry queue is for generation failures, not model execution failures, because `--generate-only` does not execute code.

Keyboard interrupts should stop launching new workers. Already completed worker results should be saved if possible; in-flight workers should be cancelled or allowed to finish according to the existing interruption model for LLM calls.

## Tests

Add focused coverage for:

- default config value is `1`;
- invalid worker values fail fast for `0`, negative values, non-integers, and values above `8`;
- `workers=1` preserves current generate-only behavior;
- batch reservation selects unique hypotheses and respects `default` ordering;
- batch reservation respects `manifest_score` ordering;
- rolling refill starts a new hypothesis when a worker finishes and capacity remains;
- journal and manifest writes happen in deterministic reservation order, not worker completion order;
- new nodes use explicit artifact directory names, while old timestamp-only nodes still resolve through the legacy fallback;
- multiple in-flight ROOT placeholders render without duplicates;
- `follow:active` targets the most recently launched in-flight hypothesis;
- completed placeholders are replaced by generated nodes;
- worker failure retries the same hypothesis three times with 5 second sleeps;
- after three failures for one hypothesis, refill stops and the failed hypothesis is retried first on the next run.

## Out Of Scope

This feature does not parallelize:

- normal execution mode;
- generated-only node execution after resuming without `--generate-only`;
- child hypothesis generation;
- debug/fix generation;
- synthesis;
- model training.
