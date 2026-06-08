# TODO / Future Ideas

## Split terminal UI into state producer and read-only viewers

This is a future architecture idea, not a current implementation task. The local
fork has already modernized the upstream project in several places, so this is
mainly useful as a reference if rebuilding the terminal UI from scratch.

The current AIDE terminal UI renders the solution tree and run data as one Rich
layout inside the main process. A cleaner design would keep the main AIDE process
as the single owner of run state, then publish an atomic JSON snapshot once per
second. Separate tmux panes or standalone viewer commands could read that snapshot
and render independent views, such as the solution tree, run metadata, last error,
and resource telemetry.

Suggested split:

- `run_state.json`: written by the main process with journal-derived tree state,
  progress, active stage, active artifact directory, research/synthesis status,
  last error, and the current executor process target.
- Tree viewer: read-only renderer for the tree snapshot, with optional navigation
  state kept local to the viewer.
- Run-data viewer: read-only renderer for run metadata and errors.
- Resource providers: viewer-side samplers for generic system/GPU telemetry, with
  process-specific CPU/RAM sampling enabled only when AIDE publishes a root PID or
  process group ID for the executor.

The boundary should stay conservative: viewers may sample telemetry and render,
but they should not mutate AIDE state. A generic "JSON describes every UI widget"
engine is probably too broad for a first version; a typed AIDE run-state snapshot
plus small reusable telemetry/render modules would be more practical and easier to
reuse in other terminal tools later.

## Generate parent-child diff descriptions for run steps

Future idea: add a cached `diff_description` for each non-root node so submit
tables and research tooling can show what actually changed in a step, not only
the generated plan or validation score.

Use `journal.json` as the source of truth: resolve `node2parent`, diff the
parent and child `code` fields with a unified diff, then send a compact diff plus
the parent/child plans and metrics to an LLM. The LLM should return one or two
plain-text sentences describing the real code change, focused on modeling,
features, validation, ensembling, calibration, data usage, or output artifacts.
It should not infer beyond the diff, and should explicitly say when the change is
mostly formatting or refactoring.

This should be implemented as an offline/cacheable command, not during submit,
for example `scripts/generate_diff_descriptions.py --run RUN --missing-only`.
The generated descriptions can be stored in `aide_result.json`, in
`submission_index.json`, or in a small central cache keyed by `(run, node_id)`.
`kaggle_submission_lab` should only read the cached field and optionally show it
in full view.
