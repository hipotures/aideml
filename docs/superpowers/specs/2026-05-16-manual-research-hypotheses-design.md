# Manual Research Hypotheses Design

## Goal

Add a manual research mode for AIDE runs. Instead of asking an LLM research
advisor to generate fresh hypotheses during the run, AIDE can sample from a
small prewritten hypothesis library for the current task.

The design must make two things visible:

- which manual hypotheses were offered to each experiment;
- which hypotheses the model/reviewer later claimed were actually used.

The manual hypothesis text must not be copied wholesale into every prompt.
Only a bounded sample is rendered for each checkpoint.

## Current Behavior

The existing research advisor is LLM-driven:

- `research.enabled=true` enables the advisor.
- Every `research.every_steps` scored working nodes, it creates a research
  checkpoint under `logs/<run>/research/checkpoint-XXXXXX`.
- The LLM prompt currently asks for exactly 5 hypotheses.
- Agent prompts load the latest completed research checkpoint and render its
  hypotheses as external hints.

There is no manual hypothesis source, no per-node research selection record,
and no clear way to see whether the agent used a specific research idea.

## Non-Goals

- Do not add a separate import CLI.
- Do not search the filesystem for hypothesis libraries.
- Do not copy every hypothesis markdown file into each run log.
- Do not change the LLM research prompt count or renderer limit as part of the
  first manual-mode implementation.
- Do not force the agent to try every hypothesis before it can continue.

## Hypothesis Library

Manual hypotheses live once in the repository:

```text
research_hypotheses/
  playground-series-s6e5/
    source.md
    hypotheses/
      hypothesis-000001.md
      hypothesis-000002.md
      ...
      hypothesis-000009.md
```

The initial library can be created directly from `/tmp/research01.md` during
implementation. That temporary file is an import source only. Runtime must not
depend on it.

The source file currently contains 9 sections with source headings numbered
`## 2.` through `## 10.`. The library should renumber them locally from
`000001` through `000009`.

Runtime must not rely on a hand-maintained list of hypothesis ids. The source
of truth is the `hypotheses/hypothesis-*.md` directory. At run startup, manual
mode indexes all matching files in sorted filename order, derives the id from
the filename, and builds the sampling pool from that runtime index. If a new
file such as `hypothesis-000010.md` is added before a later run, that later run
must pick it up automatically.

Each hypothesis file should be self-describing:

```markdown
# Replace single random holdout selection with race/year-aware repeated validation

Summary: Use race/year-aware repeated validation to reduce CV/public mismatch.

Source heading: ## 2. Replace single random holdout selection with race/year-aware repeated validation

...
```

`title` comes from the first markdown heading in the file. `summary` is a short
runtime and reporting summary stored inside the same file. It can be derived
manually from the hypothesis text for the initial library; no LLM is required.

## Library Location

The library location is deterministic:

```text
<repo_root>/research_hypotheses/<task_slug>/
```

For:

```bash
data_dir="aide/example_tasks/playground-series-s6e5"
```

`task_slug` is the basename of `data_dir`:

```text
playground-series-s6e5
```

So the library path is:

```text
/home/xai/DEV/aideml/research_hypotheses/playground-series-s6e5/
```

There is no disk search. If the directory is missing, manual research mode
fails with a clear error.

## Configuration

Use short mode names:

```yaml
research:
  enabled: true
  mode: llm
  manual_sample_size: 3
  manual_seed: 42
```

Modes:

- `llm`: current LLM-generated research behavior.
- `manual`: sample from the task's manual hypothesis library.

For the target run:

```bash
research.enabled=true \
research.mode=manual \
research.manual_sample_size=3
```

No `manual_source_file` and no `manual_dir` are needed for the default path.
A future `research.manual_slug` override can be added if one task must reuse a
different library, but it is not part of this design.

## Sampling

Manual mode samples `research.manual_sample_size` hypotheses at each research
checkpoint. The default is 3.

Sampling should prefer under-selected hypotheses within the current run:

1. Load the per-run `selected_count` values.
2. Pick from the least-selected hypotheses first.
3. Use deterministic random tie-breaking based on `manual_seed`, run id, and
   checkpoint step.
4. Record the selection before rendering it to prompts.

This gives broad coverage without forcing a strict queue. With 9 hypotheses
and sample size 3, all hypotheses should normally be presented after about
three manual research checkpoints.

## Prompt Rendering

Manual mode should feed the selected hypotheses through the same conceptual
surface as current external research hints:

- summary text;
- numbered hypothesis items;
- title;
- implementation hint or relevant excerpt;
- risk/expected effect when available.

Only the selected hypotheses are rendered. The full library is never placed in
the agent prompt.

## Per-Run Telemetry

Run logs store references and usage statistics only:

```text
logs/<run>/research_hypotheses/
  source_ref.json
  selections.jsonl
  usage.json
```

`source_ref.json` records the library identity:

```json
{
  "source_dir": "/home/xai/DEV/aideml/research_hypotheses/playground-series-s6e5",
  "source_hash": "...",
  "indexed_hypothesis_count": 9,
  "indexed_at": "..."
}
```

The hash covers all indexed `hypothesis-*.md` files and their relative paths.
It lets later analysis detect whether a run used the current library or an
older version. `source_ref.json` must not be the source of the sampling pool;
it is only an audit record written after startup indexing.

`selections.jsonl` is append-only:

```json
{"step": 44, "selected": ["000002", "000005", "000009"], "node_id": "..."}
```

`usage.json` aggregates per-run statistics:

```json
{
  "000002": {
    "selected_count": 4,
    "declared_used_count": 1,
    "selected_node_ids": ["..."],
    "declared_used_node_ids": ["..."]
  }
}
```

Do not duplicate the hypothesis markdown files in the run log.

## Selected Versus Used

Manual mode must distinguish two states:

- `selected`: AIDE offered a hypothesis to the agent prompt.
- `declared_used`: the model/reviewer later claimed that a node used that
  hypothesis.

`selected` is a hard system fact. `declared_used` is model-reported evidence
and must be treated as weaker, but it is still useful for visibility and later
analysis.

The node metadata should include at least:

```json
{
  "research_mode": "manual",
  "research_hypotheses_selected": ["000002", "000005", "000009"],
  "research_source_hash": "..."
}
```

The review result should be extended with a bounded field such as:

```json
{
  "research_hypotheses_used": ["000005"],
  "research_usage_note": "Used the stop-debt hypothesis in the feature block."
}
```

If no hypothesis was used, the list is empty and the note should say that the
research hints were not used.

## Later Analysis

The design should support a future report that scans runs and joins:

- hypothesis id;
- selected node ids;
- declared-used node ids;
- local validation metrics;
- Kaggle public scores from the submission registry when available.

This allows questions such as:

- which hypotheses were shown most often;
- which were declared used;
- which correlated with metric improvements;
- which produced public leaderboard gains or regressions.

The global library provides stable hypothesis ids. Per-run telemetry provides
experiment linkage.

## Error Handling

Manual mode should fail early with clear errors for:

- missing library directory;
- missing `hypotheses/` directory;
- no `hypothesis-*.md` files;
- duplicate or invalid hypothesis ids derived from filenames;
- `manual_sample_size <= 0`;
- `manual_sample_size` larger than the available hypothesis count;
- missing title or summary metadata in a hypothesis file.

If `research.enabled=false`, none of this behavior runs.

## Testing

Focused tests should cover:

- task slug derivation from `data_dir`;
- startup indexing from sorted `hypothesis-*.md` files without filesystem search;
- adding `hypothesis-000010.md` changes the next run's sampling pool without
  editing any central id list;
- missing-library error message;
- deterministic under-selected sampling;
- prompt rendering includes only sampled hypotheses;
- per-run `selections.jsonl` and `usage.json` updates;
- agent prompt includes manual research hints;
- node/review metadata can record selected and declared-used hypothesis ids;
- `mode=llm` preserves current LLM research behavior.
