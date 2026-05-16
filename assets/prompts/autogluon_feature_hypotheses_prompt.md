# GPT Prompt: AutoGluon Feature Hypotheses

You are a senior Kaggle grandmaster and ML systems researcher.

I attached AIDE run exports for a Kaggle task. Analyze them deeply, but produce generalized reusable research hypotheses specifically for AIDE AutoGluon mode.

This prompt is ONLY for AutoGluon mode.

You are NOT writing implementation code. You are NOT writing `preprocess(df)`.
Your output must be textual hypotheses only, encoded as JSON fields. The short
code example below exists only to explain what kind of later implementation the
hypotheses must be compatible with.

AutoGluon mode constraint:
The AIDE coding agent will later implement a Python feature procedure named `preprocess(df)`. AIDE will wrap that procedure in its own AutoGluon training script. The agent will not write a full manual training pipeline, custom cross-validation loop, manual LightGBM/CatBoost/XGBoost training code, or custom multi-model ensemble.

The generated hypotheses must therefore be feature/preprocessing hypotheses that can realistically be tested inside a single `preprocess(df: pd.DataFrame) -> pd.DataFrame` procedure and then handed to AutoGluon.

Do not propose AutoGluon optimizer/configuration hypotheses. This prompt is not
for tuning AutoGluon itself. Exclude ideas about presets, time_limit,
included_model_types, hyperparameters, validation_strategy, bagging, stacking,
weighted ensembles, refit_full, final full-train refits, or changing AutoGluon
fit arguments. Those are out of scope even if they might improve score.

Example of the kind of implementation AutoGluon mode can build later:

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    compound = df["Compound"].astype("string").str.strip().str.upper()
    tyre_life = pd.to_numeric(df["TyreLife"], errors="coerce").astype(float)
    lap_number = pd.to_numeric(df["LapNumber"], errors="coerce").astype(float)
    race_progress = pd.to_numeric(df["RaceProgress"], errors="coerce").astype(float)

    expected_life = compound.map({
        "SOFT": 18.0,
        "MEDIUM": 25.0,
        "HARD": 35.0,
        "INTERMEDIATE": 20.0,
        "WET": 16.0,
    }).fillna(24.0)

    estimated_total_laps = (lap_number / (race_progress + eps)).clip(
        lower=lap_number,
        upper=250.0,
    )
    estimated_laps_remaining = (estimated_total_laps - lap_number).clip(lower=0.0)

    df["TyreLife_ExpectedLife_Ratio"] = tyre_life / (expected_life + eps)
    df["Expected_Life_Remaining"] = expected_life - tyre_life
    df["CurrentTyre_CanReachFinish"] = (
        estimated_laps_remaining <= (expected_life - tyre_life)
    ).astype("int8")
    return df
```

Good AutoGluon hypotheses:
- target-free tabular feature blocks
- causal cumulative features using current and past rows only
- safe categorical/frequency/statistical encodings that do not use the target
- compound, tyre-life, stint, race-progress, driver/race context features
- robust transformations, interactions, clipping, missingness flags
- features that AutoGluon can consume directly after preprocessing

Bad AutoGluon hypotheses:
- manual LightGBM/CatBoost/XGBoost training
- custom cross-validation loops
- out-of-fold target encoding
- target-derived encodings
- custom rank averaging of multiple trained predictors
- calibration pipelines
- AutoGluon preset, model-list, hyperparameter, time-limit, or fit-argument tuning
- final refit/bagging/stacking experiments that require changing AutoGluon fit logic
- instructions like "start from node X" or "copy step Y"

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
- stable feature concepts, e.g. strictly causal cumulative PitStop, tyre-life margins, compound feasibility, stint-boundary indicators
- real external source URLs in the `sources` field, if they were actually found or provided

Your job:
1. Read the attached artifacts first.
2. Extract recurring feature/preprocessing patterns.
3. Identify which feature ideas appear promising, weak, overfit-prone, duplicated, underexplored, or unstable.
4. Convert those patterns into 10 reusable AutoGluon-compatible feature hypotheses.
5. Each hypothesis must be self-contained: an AIDE AutoGluon coding agent should understand and test it without needing access to node ids, step ids, score tables, or previous run internals.

Return only valid JSON, no markdown, no comments, no prose outside JSON.
Do not include Python code in the output.

JSON format:

{
  "hypotheses": [
    {
      "title": "Short concrete title",
      "summary": "1-2 sentence operational summary of the AutoGluon feature hypothesis.",
      "rationale": "Generalized evidence and reasoning from the attached artifacts and, if available, external research. Do not include run-specific ids or exact scores.",
      "implementation_hint": "Concrete preprocess(df) feature plan. It must be realistic for AutoGluon mode.",
      "expected_effect": "Expected ROC AUC, validation stability, ranking, or robustness effect. Use small/medium/large and confidence low/medium/high if helpful.",
      "risk": "Leakage, overfit, validation, runtime, or feature compatibility risks.",
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
- Prefer concrete, testable feature hypotheses over generic Kaggle advice.
- Every `implementation_hint` should read like something that can become a `preprocess(df)` feature block.
- Do not recommend manual model training or custom validation as an AutoGluon hypothesis.
- Do not recommend tuning AutoGluon itself. Only propose feature/preprocess hypotheses.
- Do not recommend future-leaking features.
- Do not recommend optimizing directly on public leaderboard.
- If an idea is speculative, label it in the `risk` or `expected_effect` field.
- If an idea may improve random CV but hurt public generalization, warn clearly.

Before finalizing JSON, perform this private self-check:
- Remove every node id.
- Remove every step number.
- Remove every exact score.
- Remove every submission hash.
- Replace "best public node" with a generalized phrase like "strongest observed feature family".
- Replace "starting from node X" with a concrete feature/preprocess implementation description.
- Remove any hypothesis requiring custom CV, manual model training, OOF target encoding, or custom ensembling.
- Remove any hypothesis about AutoGluon presets, fit settings, bagging, stacking, final refit, model lists, or hyperparameter optimization.
- Ensure every hypothesis can be used standalone by an AutoGluon feature/preprocess agent.
- Ensure every hypothesis has `summary`.
- Ensure every hypothesis has `sources`.
- Ensure the result parses as JSON.

Now analyze the attached run exports and return the JSON.
