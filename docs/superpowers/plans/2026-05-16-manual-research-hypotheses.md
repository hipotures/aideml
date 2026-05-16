# Manual Research Hypotheses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `research.mode=manual`, where AIDE samples task-local `hypothesis-*.json` files, injects only offered hypotheses into prompts, and records offered versus LLM-claimed-used telemetry.

**Architecture:** Keep existing LLM research behavior intact. Add focused manual-hypothesis helpers in `aide/research.py`, minimal metadata fields on `Node`, and small Agent hooks for prompt injection and review usage tracking. Store global hypothesis JSON files under `research_hypotheses/<task_slug>/hypotheses/`; store only per-run audit and counters under `logs/<run>/research_hypotheses/`.

**Tech Stack:** Python dataclasses/dataclasses-json, OmegaConf config, pytest, existing AIDE research/agent modules.

---

### Task 1: Manual Hypothesis Library Loading

**Files:**
- Modify: `aide/research.py`
- Test: `tests/test_research_advisor.py`

- [ ] **Step 1: Write failing tests**

Add tests that create `research_hypotheses/playground-series-s6e5/hypotheses/hypothesis-000001.json` and `hypothesis-000002.json`, then assert:

```python
library = load_manual_hypothesis_library(cfg)
assert [h.id for h in library.hypotheses] == ["000001", "000002"]
assert library.hypotheses[0].title == "Grouped validation"
```

Also assert missing `hypotheses/`, invalid JSON, duplicate/invalid ids, and missing `title`/`summary`/`body` raise `ValueError` with concrete messages.

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_research_advisor.py -k manual_hypothesis_library`

- [ ] **Step 3: Implement library loading**

Add a small `ManualHypothesis` dataclass, task slug derivation from `Path(cfg.data_dir).name`, deterministic root path `<repo_root>/research_hypotheses/<task_slug>`, sorted `hypothesis-*.json` loading, `enabled`/field validation, and a source hash over relative paths plus file bytes.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_research_advisor.py -k manual_hypothesis_library`

### Task 2: Manual Sampling And Prompt Formatting

**Files:**
- Modify: `aide/research.py`
- Test: `tests/test_research_advisor.py`

- [ ] **Step 1: Write failing tests**

Add tests for deterministic under-offered sampling and prompt formatting:

```python
selection = select_manual_hypotheses(cfg=cfg, completed_steps=10)
assert [h.id for h in selection.hypotheses] == ["000001", "000002", "000003"]
assert "Manual research hypotheses offered" in format_manual_research_hints_for_prompt(selection)
assert "If your solution intentionally uses any of them" in rendered
```

Preload `usage.json` so `000001` has a higher `offered_count`; assert the next selection prefers less-offered ids.

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_research_advisor.py -k "manual_sampling or manual_prompt"`

- [ ] **Step 3: Implement sampling and formatting**

Add `select_manual_hypotheses()`, per-run `source_ref.json`, `offers.jsonl`, and `usage.json` helpers. Use deterministic tie-breaking with `manual_seed`, `cfg.exp_name`, and checkpoint step. Render only offered hypotheses with id, title, summary, and compact `body` excerpt.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_research_advisor.py -k "manual_sampling or manual_prompt"`

### Task 3: Agent Prompt And Review Metadata

**Files:**
- Modify: `aide/journal.py`
- Modify: `aide/agent.py`
- Modify: `aide/research.py`
- Test: `tests/test_research_advisor.py`
- Test: `tests/test_agent_review.py`

- [ ] **Step 1: Write failing tests**

Add tests that in manual mode:

```python
agent._draft()
assert "research_hypotheses_offered" on the generated node
assert "000001" in captured["prompt"]["External research hints"]
```

Add review-schema tests asserting `research_hypotheses_llm_claimed_used` and `research_usage_note` are accepted and copied to the node, while absent fields default to `[]` and `None`.

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_research_advisor.py tests/test_agent_review.py -k "manual or research_schema"`

- [ ] **Step 3: Implement Agent hooks**

Extend `Node` with `research_mode`, `research_hypotheses_offered`, `research_source_hash`, `research_hypotheses_llm_claimed_used`, and `research_usage_note`. Make `_add_research_hints()` return manual selection metadata in manual mode and attach it to the newly created node. Extend review prompt/schema with LLM-claimed-used fields and update per-run usage after review.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_research_advisor.py tests/test_agent_review.py -k "manual or research_schema"`

### Task 4: Config And Runtime Mode Wiring

**Files:**
- Modify: `aide/utils/config.py`
- Modify: `aide/utils/config.yaml`
- Modify: `aide/research.py`
- Test: `tests/test_model_reasoning_config.py`
- Test: `tests/test_resume_run.py`
- Test: `tests/test_research_advisor.py`

- [ ] **Step 1: Write failing tests**

Assert defaults:

```python
assert cfg.research.mode == "llm"
assert cfg.research.manual_sample_size == 3
assert cfg.research.manual_seed == 42
```

Assert `ResearchAdvisor.maybe_start()` keeps existing LLM checkpoint behavior for `mode=llm`, and does not spawn Codex subprocesses for `mode=manual`.

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_model_reasoning_config.py tests/test_resume_run.py tests/test_research_advisor.py -k "research"`

- [ ] **Step 3: Implement config wiring**

Add `mode`, `manual_sample_size`, and `manual_seed` to `ResearchConfig` and `config.yaml`. Keep current LLM behavior when `mode == "llm"`. In `mode == "manual"`, checkpoint timing remains `research.every_steps`, but the work is local sampling/telemetry instead of spawning Codex.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_model_reasoning_config.py tests/test_resume_run.py tests/test_research_advisor.py -k "research"`

### Task 5: Initial Playground Hypothesis JSON Files

**Files:**
- Create: `research_hypotheses/playground-series-s6e5/hypotheses/hypothesis-000001.json`
- Create through: `research_hypotheses/playground-series-s6e5/hypotheses/hypothesis-000010.json`
- Source: `aide_hypotheses_playground_s6e5.json`

- [ ] **Step 1: Create JSON files**

Split `aide_hypotheses_playground_s6e5.json` into `hypothesis-000001.json` through `hypothesis-000010.json`. Preserve the full text for each section in `body`; write concise `title` and `summary` fields; add `enabled` to every hypothesis and set `hypothesis-000001.json` disabled.

- [ ] **Step 2: Validate with loader test**

Run: `uv run pytest tests/test_research_advisor.py -k manual_hypothesis_library`

### Task 6: Final Verification

**Files:**
- All touched implementation and test files.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_research_advisor.py tests/test_agent_review.py tests/test_model_reasoning_config.py tests/test_resume_run.py`

- [ ] **Step 2: Run lint**

Run: `uv run ruff check aide/research.py aide/agent.py aide/journal.py aide/utils/config.py tests/test_research_advisor.py tests/test_agent_review.py tests/test_model_reasoning_config.py tests/test_resume_run.py`

- [ ] **Step 3: Review diff and commit**

Run: `git status --short` and `git diff --stat`. Commit only the manual research implementation, tests, plan/spec updates, and new hypothesis JSON files.
