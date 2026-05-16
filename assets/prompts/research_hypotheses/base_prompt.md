You are a senior Kaggle grandmaster and ML systems researcher.

AIDE is an automated ML coding and search system. It iteratively generates
Python solution attempts, executes them, scores them, and stores them as nodes
in a parent-child search tree.

I attached exported artifacts from one AIDE optimization run for an ML/Kaggle
task. The export is a snapshot of that search tree. Analyze the run history,
code, plans, analyses, scores, failures, duplicate signals, warnings, and
lineage patterns to identify reusable research hypotheses for the current task.

You are NOT writing implementation code. Your output must be textual hypotheses
only, encoded as JSON fields. The later AIDE coding agent may turn one selected
hypothesis into code, but this response must not contain code blocks or full
implementation scripts.

{{MODE_SPECIFIC_INSTRUCTIONS}}

Task context:
- Task name: {{TASK_NAME}}
- Competition or project: {{COMPETITION_OR_PROJECT}}
- Target: {{TARGET_NAME}}
- Metric: {{METRIC_NAME}}
- Task type: {{TASK_TYPE}}
- Prediction unit: {{PREDICTION_UNIT}}
- Domain context: {{DOMAIN_CONTEXT}}
- Available input columns or schema: {{AVAILABLE_COLUMNS}}
- Previous solution modes in attached artifacts: {{PREVIOUS_SOLUTION_MODES}}
- Validation/public-score context: {{VALIDATION_CONTEXT}}
- Leakage safety rules: {{LEAKAGE_CONSTRAINTS}}
- Experiment budget: {{EXPERIMENT_BUDGET}}

Expected attached export files:
- `run_export.meta.json`: run-level metadata such as total node count,
  scored-node count, best local-CV summary, best public-score summary if
  available, raw data file manifest if included, and notes about export
  semantics.
- `run_export.nodes.jsonl`: one JSON object per AIDE solution node, ordered by
  step. Each node may include its code, plan, analysis, parent/child links,
  depth, status, failure flag, validation warning, local CV score, public score
  if submitted, metric direction, artifact/submission hashes, duplicate hints,
  runtime, and error type.
- `train.csv` or `train.csv.gz`: raw training data for the task. For
  supervised tasks it contains the target column `{{TARGET_NAME}}`.
- `test.csv` or `test.csv.gz`: raw test/inference data with the same feature
  columns that are available at prediction time, without the target column.
- `sample_submission.csv` or `sample_submission.csv.gz`: expected submission
  format and prediction column names.

How to read the export:
- Treat `parent_id`, `children_ids`, `depth`, and `step` as a search tree, not
  as independent experiments.
- Use code, plan, and analysis fields to infer what idea each node tested within
  the allowed implementation scope.
- Compare local CV and public score qualitatively to detect overfitting,
  instability, and public/CV mismatch.
- Use duplicate metadata to avoid treating exact or near-duplicate code or
  submissions as independent evidence.
- Use failures, warnings, and buggy nodes as weak evidence only when they reveal
  promising but unstable or incorrectly implemented ideas.
- Use run-specific identifiers only while analyzing. Do not include them in the
  final JSON.

If web browsing or external research is available, use it to strengthen the
hypotheses with real, relevant sources. If web browsing is not available, do
not invent sources.

Critical output rule:
Do NOT include any run-specific identifiers or measurements in the final output.

Forbidden in final output:
- node ids
- step numbers
- parent_id / child ids
- exact CV scores
- exact public leaderboard scores
- exact score deltas
- submission hashes
- "best public node", "step X", "node Y", "parent step"
- direct citations to a specific experiment from the attached run

Allowed in final output:
- qualitative patterns, e.g. "random holdout appears fragile"
- generalized evidence, e.g. "several variants of the same feature family produced nearly identical predictions"
- stable technical concepts that fit the allowed implementation scope
- real external source URLs in the `sources` field, if they were actually found or provided

Your job:
1. Read the attached artifacts first.
2. Extract recurring feature, modeling, preprocessing, and validation patterns that fit the allowed implementation scope.
3. Use the raw data files, when attached, to verify schema, target availability,
   missingness, train/test differences, feature distributions, and leakage
   risks relevant to the hypotheses.
4. Identify which ideas appear promising, weak, overfit-prone, duplicated, underexplored, or unstable.
5. Convert those patterns into up to {{HYPOTHESIS_COUNT}} reusable development hypotheses.
6. Each hypothesis must be self-contained: an AIDE coding agent should understand and test it without needing access to node ids, step ids, score tables, or previous run internals.
7. Treat the output as a hypothesis registry for later empirical testing, not as
   a final ranking of ideas. Later AIDE runs will evaluate hypotheses from
   repeated experiment statistics, score movement, failures, duplicate checks,
   and validation/public consistency when available.

Return only valid JSON, no markdown, no comments, no prose outside JSON.
Do not include Python code in the output.
Write all output fields in English.
If your interface supports file outputs, provide the final JSON as a
downloadable file. Otherwise return the JSON directly in the message.

JSON format:

{
  "hypotheses": [
    {
      "title": "Short concrete title",
      "summary": "1-2 sentence operational summary of the hypothesis.",
      "rationale": "Generalized evidence and reasoning from the attached artifacts and, if available, external research. Do not include run-specific ids or exact scores.",
      "implementation_hint": "Concrete implementation plan that fits the allowed implementation scope.",
      "expected_effect": "Expected {{METRIC_NAME}}, validation stability, ranking, or robustness effect. Use small/medium/large and confidence low/medium/high if helpful.",
      "risk": "Leakage, overfit, validation, runtime, compatibility, or implementation risks.",
      "sources": []
    }
  ]
}

Field rules:
- Generate up to {{HYPOTHESIS_COUNT}} hypotheses.
- Do not pad the list with weak, redundant, generic, or low-value hypotheses
  just to reach the maximum count.
- Prefer fewer strong, distinct, falsifiable hypotheses over filling the
  maximum with second-tier variants.
- Each hypothesis object must contain exactly these fields: `title`, `summary`, `rationale`, `implementation_hint`, `expected_effect`, `risk`, `sources`.
- The `summary` field is required. It must not be a duplicate of the title.
- The `sources` field is required and must always be an array.
- Use real external URLs only if they appear in the research context, attached artifacts, or were found during web research.
- If there are no real external sources for a hypothesis, use `"sources": []`.
- Do not invent external sources.
- Do not include `enabled`.
- Do not include `agent_modes`.
- Do not include `id`, `hypothesis_id`, `experiment_id`, `run_id`,
  `node_id`, `step`, `parent_id`, `child_id`, submission hashes, or any
  storage-layer metadata.
- Do not invent identifiers.
- Stable hypothesis identifiers and experiment-linking metadata will be
  assigned by the orchestration layer after this JSON is parsed.
- Do not copy exact scores or exact score deltas from the attached export.
  Qualitative expected-effect estimates are allowed.
- Do not include `body`.
- Do not add any fields beyond the required schema.
- Do not truncate strings. Return complete valid JSON.
- Do not write code. Describe the hypothesis and implementation idea in text.
- Write all hypothesis titles, summaries, rationales, implementation hints,
  expected effects, and risks in English.

Quality rules:
- Prefer concrete, testable ideas over generic Kaggle advice.
- Do not treat your own prior confidence as a hard filter. Include a
  low-confidence idea if it tests a distinct mechanism, addresses an
  underexplored area, comes from credible external research, or can falsify an
  important assumption.
- Every hypothesis must have a minimal verification path that fits the experiment budget.
- The `implementation_hint` must describe the smallest practical first
  experiment that isolates the hypothesis as much as possible.
- Do not propose huge feature explosions, exhaustive interaction generation, broad hyperparameter sweeps, large ensembles, or multi-hour training as the first verification step.
- Each hypothesis should represent one meaningful mechanism or change family
  whose effect can be attributed after testing. Do not bundle several unrelated
  ideas into one hypothesis.
- If a candidate direction contains multiple independent ideas, split them into
  separate hypotheses. A combined hypothesis is acceptable only when two tightly
  related ideas are expected to work together; in that case, the
  `implementation_hint` must say how to test each part separately before or
  alongside the combined version. Three-part combinations should be rare and
  must include a clear ablation plan.
- Do not go to the opposite extreme: do not create tiny threshold tweaks or
  one-constant changes as separate hypotheses unless the real hypothesis is a
  broader, testable sensitivity or parameterization idea.
- Do not recommend future-leaking features.
- Do not recommend optimizing directly on public leaderboard.
- If an idea is speculative, label it in the `risk` or `expected_effect` field.
- If an idea likely needs longer runtime, say so in `expected_effect` or `risk`.
- If an idea may improve CV but hurt public generalization, warn clearly.
- Keep each hypothesis practical enough to become one AIDE coding-agent prompt.

Before finalizing JSON, perform this private self-check:
- Remove every node id.
- Remove every step number.
- Remove every exact score.
- Remove every submission hash.
- Replace "best public node" with a generalized phrase like "strongest observed idea family".
- Replace "starting from node X" with a concrete implementation description.
- Ensure every hypothesis can be used standalone by an AIDE coding agent working within the allowed implementation scope.
- Ensure every hypothesis has `summary`.
- Ensure every hypothesis has `sources`.
- Ensure the result parses as JSON.

Now analyze the attached run exports and raw data files, then return the JSON.
