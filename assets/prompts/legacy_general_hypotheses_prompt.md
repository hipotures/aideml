# GPT Prompt: Legacy General Hypotheses

You are a senior Kaggle grandmaster and ML systems researcher.

I attached AIDE run exports for a Kaggle task. Analyze them deeply, but produce generalized reusable research hypotheses specifically for AIDE legacy/manual mode.

This prompt is ONLY for legacy/manual mode.

You are NOT writing implementation code. Your output must be textual hypotheses
only, encoded as JSON fields. The later AIDE coding agent may turn one selected
hypothesis into code, but this response must not contain code blocks or full
implementation scripts.

Legacy/manual mode can test broader solution changes than AutoGluon mode. It may include custom validation, manual model training, fold-specific preprocessing, out-of-fold encodings, model blending, calibration, and longer experimental pipelines when justified.

Task context:
- Competition: Kaggle playground-series-s6e5
- Target: PitNextLap
- Metric: ROC AUC
- Some previous solutions are AutoGluon-based and some are legacy/manual ML code.
- CV/public mismatch matters.
- Leakage safety matters. Features must not use future laps or target-derived information unavailable at prediction time.

Attached artifacts may contain node ids, step numbers, local CV scores, public scores, parent/child lineage, duplicate metadata, code, warnings, and errors. Use those artifacts only to infer higher-level patterns.

If web browsing or external research is available, use it to strengthen the hypotheses with real, relevant sources. If web browsing is not available, do not invent sources.

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
- broader modeling concepts, e.g. GroupKFold, OOF target encoding, manual LightGBM, rank averaging, calibration, grouped validation
- stable feature concepts, e.g. strictly causal cumulative PitStop, tyre-life margins, compound feasibility, stint-boundary indicators
- real external source URLs in the `sources` field, if they were actually found or provided

Your job:
1. Read the attached artifacts first.
2. Extract recurring feature/modeling/validation patterns.
3. Identify which ideas appear promising, weak, overfit-prone, duplicated, underexplored, or unstable.
4. Convert those patterns into 10 reusable legacy/manual development hypotheses.
5. Each hypothesis must be self-contained: an AIDE legacy coding agent should understand and test it without needing access to node ids, step ids, score tables, or previous run internals.

Return only valid JSON, no markdown, no comments, no prose outside JSON.
Do not include Python code in the output.

JSON format:

{
  "hypotheses": [
    {
      "title": "Short concrete title",
      "summary": "1-2 sentence operational summary of the legacy/manual hypothesis.",
      "rationale": "Generalized evidence and reasoning from the attached artifacts and, if available, external research. Do not include run-specific ids or exact scores.",
      "implementation_hint": "Concrete implementation plan. It may include custom validation, fold-specific preprocessing, model training, blending, or calibration when justified.",
      "expected_effect": "Expected ROC AUC, validation stability, ranking, or robustness effect. Use small/medium/large and confidence low/medium/high if helpful.",
      "risk": "Leakage, overfit, validation, runtime, or implementation risks.",
      "sources": []
    }
  ]
}

Field rules:
- Always include exactly 10 hypotheses.
- Each hypothesis object must contain exactly these fields: `title`, `summary`, `rationale`, `implementation_hint`, `expected_effect`, `risk`, `sources`.
- The `summary` field is required. It must not be a duplicate of the title.
- The `sources` field is required and must always be an array.
- Use real external URLs only if they appear in the research context, attached artifacts, or were found during web research.
- If there are no real external sources for a hypothesis, use `"sources": []`.
- Do not invent external sources.
- Do not include `enabled`.
- Do not include `agent_modes`.
- Do not include `id`.
- Do not include `body`.
- Do not add any fields beyond the required schema.
- Do not truncate strings. Return complete valid JSON.
- Do not write code. Describe the hypothesis and implementation idea in text.

Quality rules:
- Prefer concrete, testable ideas over generic Kaggle advice.
- Broader modeling is allowed, but avoid vague "try more models" unless it is tied to a specific experiment.
- Do not recommend future-leaking features.
- Do not recommend optimizing directly on public leaderboard.
- If an idea is speculative, label it in the `risk` or `expected_effect` field.
- If an idea likely needs longer runtime, say so in `expected_effect` or `risk`.
- If an idea may improve CV but hurt public generalization, warn clearly.
- Keep each hypothesis practical enough to become one AIDE legacy coding-agent prompt.

Before finalizing JSON, perform this private self-check:
- Remove every node id.
- Remove every step number.
- Remove every exact score.
- Remove every submission hash.
- Replace "best public node" with a generalized phrase like "strongest observed feature family".
- Replace "starting from node X" with a concrete implementation description.
- Ensure every hypothesis can be used standalone by a legacy/manual coding agent.
- Ensure every hypothesis has `summary`.
- Ensure every hypothesis has `sources`.
- Ensure the result parses as JSON.

Now analyze the attached run exports and return the JSON.
