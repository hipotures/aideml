# Hypothesis Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement `research.mode=hypothesis` as a hard one-hypothesis-per-node verification mode.

**Architecture:** Keep existing `llm` and `manual` behavior intact. Add hypothesis-mode selection helpers in `aide/research.py`, call them from `aide/agent.py` during generation, validate the claimed hypothesis during review, and append compact hypothesis IDs in the TUI.

**Tech Stack:** Python dataclasses, existing AIDE `Journal`/`Node`, Rich TUI, pytest.

---

### Task 1: Hypothesis Selection Helpers

**Files:**
- Modify: `aide/research.py`
- Test: `tests/test_research_advisor.py`

- [x] Add failing tests for root selection, root exhaustion, child uniqueness, sibling uniqueness, and debug inheritance.
- [x] Implement helpers that load enabled compatible hypotheses, derive used IDs from the journal, select exactly one hypothesis deterministically, and record the offer in the existing `research_hypotheses` usage files.
- [x] Run `uv run pytest tests/test_research_advisor.py -v`.

### Task 2: Agent Prompt And Contract Validation

**Files:**
- Modify: `aide/agent.py`
- Test: `tests/test_research_advisor.py`
- Test: `tests/test_agent_review.py`
- Test: `tests/test_agent_search_policy.py`

- [x] Add failing tests that `research.mode=hypothesis` puts one hard hypothesis in the prompt and records metadata on the node.
- [x] Add failing tests that missing or wrong claimed hypothesis IDs mark a node as `status="failed"` instead of `bug`.
- [x] Add failing tests that root cadence uses `research.every_steps` and skips root creation when the root pool is exhausted.
- [x] Implement agent integration without changing `manual` mode.
- [x] Run `uv run pytest tests/test_research_advisor.py tests/test_agent_review.py tests/test_agent_search_policy.py -v`.

### Task 3: TUI Hypothesis Labels

**Files:**
- Modify: `aide/run.py`
- Test: `tests/test_run_tree.py`

- [x] Add failing tests for `Research · step @ hypothesis ✓`, `Best Score · ... @ hypothesis`, and compact tree labels like `0.95104@000122`.
- [x] Implement compact label helpers that activate only when a node has exactly one hypothesis ID.
- [x] Keep non-hypothesis terminal failures hidden while showing hypothesis protocol failures as `failed@id`.
- [x] Run `uv run pytest tests/test_run_tree.py -v`.

### Task 4: Verification And Commit

**Files:**
- Modify: implementation files above
- Test: all targeted tests

- [x] Run targeted pytest files.
- [x] Run `uv run ruff check aide tests`.
- [x] Review `git diff`.
- [x] Commit the implementation.
