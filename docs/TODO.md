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
