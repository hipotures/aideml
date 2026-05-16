# Hypothesis Branch Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace global `Memory` with branch-local context only for `research.mode=hypothesis` prompts.

**Architecture:** Add a branch-context renderer to `Journal`, because it already owns node summary formatting. Add one `Agent` prompt helper that inserts either existing global `Memory` or the new `Branch context` based on research mode and parent presence. Keep hypothesis selection, execution, review, usage stats, and TUI unchanged.

**Tech Stack:** Python, dataclasses, existing `Node`/`Journal` structures, pytest, ruff.

---

### Task 1: Add Branch Context Rendering To Journal

**Files:**
- Modify: `aide/journal.py`
- Test: `tests/test_agent_review.py`

- [ ] **Step 1: Write the failing branch-context test**

Add this test near the existing `journal.generate_summary()` tests in `tests/test_agent_review.py`:

```python
def test_journal_generates_branch_context_from_root_to_parent_only():
    journal = Journal()
    root = Node(code="root", plan="Root hypothesis plan")
    root.metric = MetricValue(0.91, maximize=True)
    root.is_buggy = False
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000101"]
    journal.append(root)

    child = Node(code="child", plan="Child hypothesis plan", parent=root)
    child.metric = MetricValue(0.92, maximize=True)
    child.is_buggy = False
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000202"]
    journal.append(child)

    unrelated = Node(code="other", plan="Unrelated root plan")
    unrelated.metric = MetricValue(0.99, maximize=True)
    unrelated.is_buggy = False
    unrelated.research_mode = "hypothesis"
    unrelated.research_hypotheses_offered = ["000999"]
    journal.append(unrelated)

    grandchild = Node(code="grandchild", plan="Grandchild plan", parent=child)
    grandchild.metric = MetricValue(0.93, maximize=True)
    grandchild.is_buggy = False
    grandchild.research_mode = "hypothesis"
    grandchild.research_hypotheses_offered = ["000303"]

    context = journal.generate_branch_context(child)

    assert "ancestor nodes of this parent, ordered from root to direct parent" in context
    assert "Branch path:\n000101 -> 000202" in context
    assert "Ancestor 1 / root:" in context
    assert "Hypothesis ID: 000101" in context
    assert "Design: Root hypothesis plan" in context
    assert "Validation Metric: 0.91000" in context
    assert "Ancestor 2 / direct parent:" in context
    assert "Hypothesis ID: 000202" in context
    assert "Design: Child hypothesis plan" in context
    assert "Validation Metric: 0.92000" in context
    assert "000999" not in context
    assert "Unrelated root plan" not in context
    assert "000303" not in context
    assert "Grandchild plan" not in context
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_agent_review.py::test_journal_generates_branch_context_from_root_to_parent_only
```

Expected: fail with `AttributeError: 'Journal' object has no attribute 'generate_branch_context'`.

- [ ] **Step 3: Implement the journal renderer**

Add these helpers/methods in `aide/journal.py` inside `Journal`, near `generate_summary`:

```python
    def _ancestor_path(self, node: Node) -> list[Node]:
        path: list[Node] = []
        current: Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def generate_branch_context(self, parent_node: Node) -> str:
        """Generate hypothesis-mode context for the selected parent branch."""
        ancestors = self._ancestor_path(parent_node)
        hypothesis_path = [
            n.research_hypotheses_offered[0]
            for n in ancestors
            if len(n.research_hypotheses_offered) == 1
        ]
        lines = [
            "The previous code is the current parent code. The entries below are the ancestor",
            "nodes of this parent, ordered from root to direct parent. They describe earlier",
            "hypotheses already incorporated into the previous code.",
            "",
        ]
        if hypothesis_path:
            lines.extend(["Branch path:", " -> ".join(hypothesis_path), ""])

        for idx, ancestor in enumerate(ancestors, start=1):
            labels = []
            if idx == 1:
                labels.append("root")
            if idx == len(ancestors):
                labels.append("direct parent")
            label_suffix = f" / {' / '.join(labels)}" if labels else ""
            lines.append(f"Ancestor {idx}{label_suffix}:")
            if len(ancestor.research_hypotheses_offered) == 1:
                lines.append(f"Hypothesis ID: {ancestor.research_hypotheses_offered[0]}")
            lines.append(f"Design: {_summary_plan_text(ancestor.plan)}")
            if ancestor.metric is not None and ancestor.metric.value is not None:
                lines.append(f"Validation Metric: {ancestor.metric.value:.5f}")
            if idx != len(ancestors):
                lines.extend(["", "-------------------------------"])
            lines.append("")

        lines.extend(
            [
                "Instruction: Use the previous code as the authoritative current state.",
                "This branch context only describes earlier changes already present in that code.",
                "Preserve these earlier branch changes unless they directly conflict with the assigned hypothesis.",
            ]
        )
        return "\n".join(lines).strip()
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_agent_review.py::test_journal_generates_branch_context_from_root_to_parent_only
```

Expected: pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add aide/journal.py tests/test_agent_review.py
git commit -m "Add branch context renderer"
```

---

### Task 2: Use Branch Context In Hypothesis Prompts

**Files:**
- Modify: `aide/agent.py`
- Test: `tests/test_research_advisor.py`

- [ ] **Step 1: Write the failing root prompt test**

Add this test near `test_agent_includes_hard_hypothesis_contract_in_draft_prompt` in `tests/test_research_advisor.py`:

```python
def test_hypothesis_root_prompt_omits_global_memory_and_branch_context(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    previous = _node(
        0.99,
        code="print('unrelated')",
        plan="Unrelated global winner",
    )
    journal = Journal()
    journal.append(previous)
    selection = research.ManualHypothesisSelection(
        completed_steps=1,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000122",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Rival-relative pit-wave features",
                summary="Use current-lap peer pit context.",
                rationale="Pit decisions often react to nearby rivals.",
                implementation_hint="Add current-lap rival aggregate features.",
                expected_effect="Improves reactive stop timing signal.",
                risk="Avoid future laps and target-derived aggregates.",
                sources=[],
                path=tmp_path / "hypothesis-000122.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "I will verify hypothesis 000122.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    assert "Memory" not in captured["prompt"]
    assert "Branch context" not in captured["prompt"]
    assert "Hypothesis under verification" in captured["prompt"]
```

- [ ] **Step 2: Write the failing child prompt test**

Add this test in the same file:

```python
def test_hypothesis_child_prompt_uses_branch_context_not_global_memory(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    root = _node(0.91, code="root code", plan="Root branch plan")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000101"]
    child = _node(0.92, code="child code", plan="Child branch plan")
    child.parent = root
    root.children.add(child)
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000202"]
    unrelated = _node(0.99, code="other code", plan="Unrelated global winner")
    unrelated.research_mode = "hypothesis"
    unrelated.research_hypotheses_offered = ["000999"]
    journal = Journal()
    journal.append(root)
    journal.append(child)
    journal.append(unrelated)
    selection = research.ManualHypothesisSelection(
        completed_steps=3,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000303",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Assigned child hypothesis",
                summary="Add one assigned child change.",
                rationale="This validates the selected hypothesis.",
                implementation_hint="Add the assigned feature block.",
                expected_effect="May improve ROC AUC.",
                risk="Keep it leakage-safe.",
                sources=[],
                path=tmp_path / "hypothesis-000303.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "I will verify hypothesis 000303.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(child)

    assert "Memory" not in captured["prompt"]
    assert "Branch context" in captured["prompt"]
    branch_context = captured["prompt"]["Branch context"]
    assert "Branch path:\n000101 -> 000202" in branch_context
    assert "Ancestor 1 / root:" in branch_context
    assert "Ancestor 2 / direct parent:" in branch_context
    assert "Unrelated global winner" not in branch_context
    assert "Hypothesis under verification" in captured["prompt"]
    assert "Hypothesis ID: 000303" in captured["prompt"]["Hypothesis under verification"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_research_advisor.py::test_hypothesis_root_prompt_omits_global_memory_and_branch_context tests/test_research_advisor.py::test_hypothesis_child_prompt_uses_branch_context_not_global_memory
```

Expected: both fail because prompts still use global `Memory`.

- [ ] **Step 4: Implement the Agent prompt helper**

In `aide/agent.py`, add this method near `_add_research_hints`:

```python
    def _add_memory_or_branch_context(
        self,
        prompt: dict[str, Any],
        *,
        parent_node: Node | None,
    ) -> None:
        if self._is_hypothesis_mode():
            if parent_node is not None:
                prompt["Branch context"] = self.journal.generate_branch_context(
                    parent_node
                )
            return
        prompt["Memory"] = self.journal.generate_summary()
```

Then replace direct `"Memory": self.journal.generate_summary()` entries in `_draft`, `_draft_autogluon_preprocess`, `_improve`, and `_improve_autogluon_preprocess` with calls to this helper after prompt creation:

```python
self._add_memory_or_branch_context(prompt, parent_node=None)
```

or:

```python
self._add_memory_or_branch_context(prompt, parent_node=parent_node)
```

For `_debug` and `_debug_autogluon_preprocess`, add the helper call with the buggy parent:

```python
self._add_memory_or_branch_context(prompt, parent_node=parent_node)
```

Do not remove `Previous solution` or `Previous preprocess function`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_research_advisor.py::test_hypothesis_root_prompt_omits_global_memory_and_branch_context tests/test_research_advisor.py::test_hypothesis_child_prompt_uses_branch_context_not_global_memory
```

Expected: pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add aide/agent.py tests/test_research_advisor.py
git commit -m "Use branch context for hypothesis prompts"
```

---

### Task 3: Preserve Existing Non-Hypothesis Prompt Behavior

**Files:**
- Modify: `tests/test_research_advisor.py`

- [ ] **Step 1: Write the regression test for normal draft memory**

Add this test to `tests/test_research_advisor.py`:

```python
def test_non_hypothesis_draft_prompt_keeps_global_memory(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "llm"
    cfg.agent.data_preview = False
    previous = _node(
        0.95,
        code="print('previous')",
        plan="Previous global design",
    )
    journal = Journal()
    journal.append(previous)
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    assert "Memory" in captured["prompt"]
    assert "Previous global design" in captured["prompt"]["Memory"]
    assert "Branch context" not in captured["prompt"]
```

- [ ] **Step 2: Run the regression test**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_research_advisor.py::test_non_hypothesis_draft_prompt_keeps_global_memory
```

Expected: pass after Task 2 implementation. If it fails, fix `_add_memory_or_branch_context` so only `hypothesis` mode suppresses global `Memory`.

- [ ] **Step 3: Run focused prompt tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_research_advisor.py tests/test_agent_review.py
```

Expected: all tests pass.

- [ ] **Step 4: Commit Task 3**

Run:

```bash
git add tests/test_research_advisor.py
git commit -m "Cover memory behavior outside hypothesis mode"
```

---

### Task 4: Final Verification

**Files:**
- No planned code edits.

- [ ] **Step 1: Run lint**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check aide tests
```

Expected: `All checks passed!`

- [ ] **Step 2: Run full test suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Expected: all tests pass. Existing multiprocessing deprecation warnings are acceptable.

- [ ] **Step 3: Check whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Report final commits and verification**

Report:

```text
Implemented hypothesis branch context prompt behavior.
Verification:
- ruff check aide tests: passed
- pytest: passed
- git diff --check: clean
```

