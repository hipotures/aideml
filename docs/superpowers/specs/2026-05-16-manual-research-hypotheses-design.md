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
- Do not copy every hypothesis JSON file into each run log.
- Do not change the LLM research prompt count or renderer limit as part of the
  first manual-mode implementation.
- Do not force the agent to try every hypothesis before it can continue.

## Hypothesis Library

Manual hypotheses live once in the repository:

```text
research_hypotheses/
  playground-series-s6e5/
    hypotheses/
      hypothesis-000001.json
      hypothesis-000002.json
      ...
      hypothesis-000009.json
```

Runtime must not rely on a hand-maintained list of hypothesis ids. The source
of truth is the `hypotheses/hypothesis-*.json` directory. At run startup, manual
mode indexes all matching files in sorted filename order, derives the id from
the filename, and builds the sampling pool from that runtime index. If a new
file such as `hypothesis-000010.json` is added before a later run, that later run
must pick it up automatically.

Each hypothesis file is structured JSON. The id is not stored inside the file;
it comes only from the filename.

```json
{
  "enabled": true,
  "agent_modes": ["legacy", "autogluon"],
  "title": "Replace single random holdout selection with race/year-aware repeated validation",
  "summary": "Use race/year-aware repeated validation to reduce CV/public mismatch.",
  "body": "Full hypothesis text, including evidence, rationale, implementation notes, risks, expected impact, and any prompt snippet."
}
```

`enabled` controls whether a hypothesis participates in sampling; disabled
files stay in the library for audit/history but are not offered to the agent.
`agent_modes` controls compatibility with the current coding agent mode. Values
are short external keys: `legacy` for the full-script agent and `autogluon` for
the AutoGluon preprocess agent. A hypothesis can list both modes. `title` and
`summary` are explicit fields for status, analysis, and concise prompt
rendering. `body` contains the full hypothesis text. Runtime should validate
these fields but should not parse headings out of free-form text.

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
checkpoint. The default is 3. Only hypotheses with `enabled: true` are eligible
for sampling, and they must also include the current agent compatibility key in
`agent_modes`. For example, `agent.mode=autogluon` is normalized internally to
`autogluon_preprocess`, but manual hypothesis filtering uses the external key
`autogluon`.

Sampling should prefer under-offered hypotheses within the current run:

1. Load the per-run `offered_count` values.
2. Pick from the least-offered hypotheses first.
3. Use deterministic random tie-breaking based on `manual_seed`, run id, and
   checkpoint step.
4. Record the offer event before rendering it to prompts.

This gives broad coverage without forcing a strict queue. With 9 hypotheses
and sample size 3, all hypotheses should normally be presented after about
three manual research checkpoints.

## Prompt Rendering

Manual mode should feed the offered hypotheses through the same conceptual
surface as current external research hints:

- summary text;
- numbered hypothesis items;
- title;
- relevant excerpt from `body`;
- risk/expected effect when available.

Only the offered hypotheses are rendered. The full library is never placed in
the agent prompt.

The prompt must explicitly tell the coding agent that hypothesis ids are
tracked:

```text
You were offered manual research hypotheses with ids. If your solution
intentionally uses any of them, mention the ids in your plan/rationale. If none
are relevant, say that no manual research hypothesis was used.
```

This instruction does not make usage certain. It creates model-reported
evidence that can be tracked separately from the hard fact that a hypothesis
was offered.

## Per-Run Telemetry

Run logs store references and usage statistics only:

```text
logs/<run>/research_hypotheses/
  source_ref.json
  offers.jsonl
  usage.json
```

`source_ref.json` records the library identity:

```json
{
  "source_dir": "/home/xai/DEV/aideml/research_hypotheses/playground-series-s6e5",
  "source_hash": "...",
  "indexed_hypothesis_count": 9,
  "enabled_hypothesis_count": 8,
  "agent_mode": "autogluon",
  "compatible_hypothesis_count": 6,
  "indexed_at": "..."
}
```

The hash covers all indexed `hypothesis-*.json` files and their relative paths.
It lets later analysis detect whether a run used the current library or an
older version. `source_ref.json` must not be the source of the sampling pool;
it is only an audit record written after startup indexing.

`offers.jsonl` is append-only:

```json
{"checkpoint_step": 44, "offered": ["000002", "000005", "000009"], "source_hash": "sha256:...", "created_at": "2026-05-16T13:21:00"}
```

`usage.json` aggregates per-run statistics:

```json
{
  "000002": {
    "offered_count": 4,
    "llm_claimed_used_count": 1,
    "offered_checkpoint_steps": [44, 55, 66, 77],
    "prompt_node_ids": ["node-a", "node-b"],
    "llm_claimed_used_node_ids": ["node-b"],
    "last_offered_at": "2026-05-16T13:21:00",
    "last_llm_claimed_used_at": "2026-05-16T13:44:00"
  }
}
```

`offered_count` is a hard system fact. `llm_claimed_used_count` is a model or
reviewer claim and must not be treated as guaranteed proof that the code
actually used the idea.

Do not duplicate the hypothesis JSON files in the run log.

## Offered Versus Claimed Used

Manual mode must distinguish two states:

- `offered`: AIDE offered a hypothesis to the agent prompt.
- `llm_claimed_used`: the model/reviewer later claimed that a node used that
  hypothesis.

`offered` is a hard system fact. `llm_claimed_used` is model-reported evidence
and must be treated as weaker. The names should stay explicit so later reports
do not imply false certainty about model intent.

The node metadata should include at least:

```json
{
  "research_mode": "manual",
  "research_hypotheses_offered": ["000002", "000005", "000009"],
  "research_source_hash": "..."
}
```

The review result should be extended with a bounded field such as:

```json
{
  "research_hypotheses_llm_claimed_used": ["000005"],
  "research_usage_note": "The review claims the node used 000005 for the stop-debt feature block."
}
```

If no hypothesis is claimed as used, the list is empty and the note should say
that no offered manual research hypothesis was used.

## Later Analysis

The design should support a future report that scans runs and joins:

- hypothesis id;
- offered checkpoint steps and prompt node ids;
- LLM-claimed-used node ids;
- local validation metrics;
- Kaggle public scores from the submission registry when available.

This allows questions such as:

- which hypotheses were offered most often;
- which were claimed as used;
- which correlated with metric improvements;
- which produced public leaderboard gains or regressions.

The global library provides stable hypothesis ids. Per-run telemetry provides
experiment linkage.

## Error Handling

Manual mode should fail early with clear errors for:

- missing library directory;
- missing `hypotheses/` directory;
- no `hypothesis-*.json` files;
- duplicate or invalid hypothesis ids derived from filenames;
- invalid hypothesis JSON;
- missing or non-boolean `enabled` in a hypothesis file;
- missing, empty, or invalid `agent_modes` in a hypothesis file;
- `manual_sample_size <= 0`;
- `manual_sample_size` larger than the compatible enabled hypothesis count;
- missing `title`, `summary`, or `body` in a hypothesis file.

If `research.enabled=false`, none of this behavior runs.

## Testing

Focused tests should cover:

- task slug derivation from `data_dir`;
- startup indexing from sorted `hypothesis-*.json` files without filesystem search;
- adding `hypothesis-000010.json` changes the next run's sampling pool without
  editing any central id list;
- disabled hypotheses are indexed but not offered;
- hypotheses incompatible with the current `agent.mode` are indexed but not offered;
- missing-library error message;
- deterministic under-offered sampling;
- prompt rendering includes only offered hypotheses;
- prompt instructions ask the agent to report any intentionally used hypothesis ids;
- per-run `offers.jsonl` and `usage.json` updates;
- agent prompt includes manual research hints;
- node/review metadata can record offered and LLM-claimed-used hypothesis ids;
- `mode=llm` preserves current LLM research behavior.
