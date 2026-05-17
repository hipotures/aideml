# Ranked Hypothesis Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `research.mode=hypothesis` branch with deterministic child-hypothesis ordering based on root scores while preventing one root branch from consuming the whole run.

**Architecture:** Keep the current two-phase system: root exploration is still controlled by `research.hypothesis_root_limit`, and parent/node selection still uses the existing search policy. Change only the child hypothesis ordering and add a parent saturation filter: after `N` scored direct children fail to improve a parent, that parent is no longer selected for more sibling attempts. Child candidates still come from the full compatible hypothesis library; hypotheses that have root scores are ranked first by empirical root score, and untested hypotheses remain available through deterministic seeded fallback ordering.

**Tech Stack:** Python dataclasses and OmegaConf config in `aide/utils/config.py` and `aide/utils/config.yaml`; tree search in `aide/agent.py`; hypothesis loading/selection in `aide/research.py`; tests in `tests/test_research_advisor.py` and `tests/test_agent_search_policy.py`.

---

### Task 1: Add Search Config Knobs

**Files:**
- Modify: `aide/utils/config.py`
- Modify: `aide/utils/config.yaml`
- Test: existing config loading through downstream tests

- [ ] **Step 1: Add fields to the search config dataclass**

In `aide/utils/config.py`, add these fields to the existing `SearchConfig` dataclass:

```python
    hypothesis_child_order: str = "root_score"
    hypothesis_max_non_improving_children_per_parent: int = 3
```

Semantics:
- `hypothesis_child_order="root_score"` enables deterministic child hypothesis ordering for `research.mode=hypothesis`.
- `hypothesis_max_non_improving_children_per_parent=3` means a parent can receive at most three scored, non-improving direct child attempts before it is treated as locally saturated.
- A value `<= 0` disables the saturation filter and preserves old unlimited branching behavior.

- [ ] **Step 2: Add defaults to YAML config**

In `aide/utils/config.yaml`, under `agent.search`, add:

```yaml
    hypothesis_child_order: root_score
    hypothesis_max_non_improving_children_per_parent: 3
```

- [ ] **Step 3: Run config-sensitive tests**

Run:

```bash
uv run pytest tests/test_agent_search_policy.py::test_search_policy_zero_exploration_keeps_greedy_selection
```

Expected: pass. This verifies the new fields do not break config loading for existing search-policy tests.

- [ ] **Step 4: Commit**

```bash
git add aide/utils/config.py aide/utils/config.yaml
git commit -m "Add hypothesis branching search config"
```

---

### Task 2: Rank Child Hypotheses by Root Score

**Files:**
- Modify: `aide/research.py`
- Test: `tests/test_research_advisor.py`

- [ ] **Step 1: Write the failing child-order test**

Add this test near `test_select_hypothesis_for_child_excludes_ancestors_and_siblings` in `tests/test_research_advisor.py`:

```python
def test_select_hypothesis_for_child_prefers_root_score_ranking(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.search.hypothesis_child_order = "root_score"
    for idx in range(1, 6):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )

    journal = Journal()
    root = _node(0.91, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000001"]

    # Root-tested empirical ranking is 000003, 000002, 000004.
    root_rank_2 = _node(0.92, code="print('h2')", plan="h2")
    root_rank_2.research_mode = "hypothesis"
    root_rank_2.research_hypotheses_offered = ["000002"]
    root_rank_1 = _node(0.94, code="print('h3')", plan="h3")
    root_rank_1.research_mode = "hypothesis"
    root_rank_1.research_hypotheses_offered = ["000003"]
    root_rank_3 = _node(0.90, code="print('h4')", plan="h4")
    root_rank_3.research_mode = "hypothesis"
    root_rank_3.research_hypotheses_offered = ["000004"]

    journal.append(root)
    journal.append(root_rank_2)
    journal.append(root_rank_1)
    journal.append(root_rank_3)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=root,
        completed_steps=4,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000003"]
```

Expected current behavior: likely fail, because selection currently uses global attempt counts plus seeded tie-break, not root-score ranking.

- [ ] **Step 2: Write the fallback test for untested hypotheses**

Add:

```python
def test_select_hypothesis_for_child_keeps_untested_hypotheses_available(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.search.hypothesis_child_order = "root_score"
    for idx in range(1, 5):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )

    journal = Journal()
    root = _node(0.91, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000001"]

    scored_root = _node(0.92, code="print('h2')", plan="h2")
    scored_root.research_mode = "hypothesis"
    scored_root.research_hypotheses_offered = ["000002"]
    used_child = _node(0.90, code="print('child')", plan="child")
    used_child.parent = root
    root.children.add(used_child)
    used_child.research_mode = "hypothesis"
    used_child.research_hypotheses_offered = ["000002"]

    journal.append(root)
    journal.append(scored_root)
    journal.append(used_child)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=root,
        completed_steps=3,
        repo_root=tmp_path,
    )

    assert selection.hypotheses[0].id in {"000003", "000004"}
```

This test protects the rule that child branching is not limited to the root-sweep subset. Once all scored-root candidates are blocked by ancestors or siblings, untested compatible hypotheses must still be selectable.

- [ ] **Step 3: Implement root-score ranking helpers**

In `aide/research.py`, add helpers near `_hypothesis_sort_key`:

```python
def _root_hypothesis_score_ranks(journal: Journal) -> dict[str, tuple[float, str]]:
    ranks: dict[str, tuple[float, str]] = {}
    for node in journal.nodes:
        if node.parent is not None or node.is_buggy:
            continue
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is None or node.metric is None or node.metric.value is None:
            continue
        score = _metric_for_hypothesis_ranking(node)
        previous = ranks.get(hypothesis_id)
        if previous is None or score > previous[0]:
            ranks[hypothesis_id] = (score, node.id)
    return ranks


def _metric_for_hypothesis_ranking(node: Node) -> float:
    assert node.metric is not None and node.metric.value is not None
    value = float(node.metric.value)
    return -value if node.metric.maximize is False else value
```

If `_metric_for_search` from `aide.agent` cannot be imported without circular imports, keep this local helper in `aide/research.py`.

- [ ] **Step 4: Replace child ordering when enabled**

Update `select_hypothesis_for_node` so `parent_node is not None`, `not parent_node.is_buggy`, and `cfg.agent.search.hypothesis_child_order == "root_score"` uses a new sort key:

```python
def _hypothesis_child_root_score_sort_key(
    *,
    hypothesis: ManualHypothesis,
    root_scores: dict[str, tuple[float, str]],
    attempts: dict[str, int],
    seed_text: str,
) -> tuple[int, float, int, float, str]:
    root_score = root_scores.get(hypothesis.id)
    tie_break = random.Random(f"{seed_text}:{hypothesis.id}").random()
    if root_score is not None:
        score, root_node_id = root_score
        return (0, -score, attempts.get(hypothesis.id, 0), tie_break, root_node_id)
    return (1, 0.0, attempts.get(hypothesis.id, 0), tie_break, hypothesis.id)
```

Then select:

```python
if (
    parent_node is not None
    and not parent_node.is_buggy
    and getattr(cfg.agent.search, "hypothesis_child_order", "root_score") == "root_score"
):
    root_scores = _root_hypothesis_score_ranks(journal)
    selected = sorted(
        candidates,
        key=lambda hypothesis: _hypothesis_child_root_score_sort_key(
            hypothesis=hypothesis,
            root_scores=root_scores,
            attempts=attempts,
            seed_text=seed_text,
        ),
    )[0]
else:
    selected = sorted(
        candidates,
        key=lambda hypothesis: _hypothesis_sort_key(
            hypothesis=hypothesis,
            attempts=attempts,
            seed_text=seed_text,
        ),
    )[0]
```

Important behavior:
- Root selection remains the existing seeded/attempt-balanced order.
- Debugging a buggy parent still inherits the buggy parent hypothesis.
- Child selection excludes ancestors and direct-child sibling hypotheses exactly as before.
- Untested hypotheses remain available after tested-root candidates are exhausted.

- [ ] **Step 5: Run focused hypothesis selection tests**

Run:

```bash
uv run pytest tests/test_research_advisor.py::test_select_hypothesis_for_child_prefers_root_score_ranking tests/test_research_advisor.py::test_select_hypothesis_for_child_keeps_untested_hypotheses_available tests/test_research_advisor.py::test_select_hypothesis_for_child_excludes_ancestors_and_siblings
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add aide/research.py tests/test_research_advisor.py
git commit -m "Rank child hypotheses by root score"
```

---

### Task 3: Add Parent Saturation Filter

**Files:**
- Modify: `aide/agent.py`
- Test: `tests/test_agent_search_policy.py`

- [ ] **Step 1: Write test for saturated non-improving parent**

Add this test near the existing exploration tests in `tests/test_agent_search_policy.py`:

```python
def test_hypothesis_search_skips_parent_after_non_improving_child_limit(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 2

    journal = Journal()
    saturated = _good_node(0.9510)
    fallback = _good_node(0.9500)
    worse_a = _good_node(0.9501, parent=saturated)
    worse_b = _good_node(0.9502, parent=saturated)
    for node, hypothesis_id in [
        (saturated, "000001"),
        (fallback, "000002"),
        (worse_a, "000003"),
        (worse_b, "000004"),
    ]:
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        journal.append(node)

    monkeypatch.setattr("aide.agent.hypothesis_root_pool_exhausted", lambda cfg, journal: True)
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda cfg, journal, parent_nodes: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is fallback
```

- [ ] **Step 2: Write test that improved child prevents parent saturation**

Add:

```python
def test_hypothesis_search_keeps_parent_with_improving_child_available(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 2

    journal = Journal()
    parent = _good_node(0.9510)
    fallback = _good_node(0.9500)
    worse = _good_node(0.9501, parent=parent)
    better = _good_node(0.9512, parent=parent)
    for node, hypothesis_id in [
        (parent, "000001"),
        (fallback, "000002"),
        (worse, "000003"),
        (better, "000004"),
    ]:
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        journal.append(node)

    monkeypatch.setattr("aide.agent.hypothesis_root_pool_exhausted", lambda cfg, journal: True)
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda cfg, journal, parent_nodes: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is better
```

This verifies that if a child improves the parent, the search can descend deeper into the improved child instead of treating the branch as dead.

- [ ] **Step 3: Write test that bugs do not count as non-improving hypothesis evidence**

Add:

```python
def test_hypothesis_non_improving_limit_ignores_bug_children(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 1

    journal = Journal()
    parent = _good_node(0.9510)
    fallback = _good_node(0.9500)
    bug = _bug_node(parent=parent)
    worse = _good_node(0.9501, parent=parent)
    for node, hypothesis_id in [
        (parent, "000001"),
        (fallback, "000002"),
        (bug, "000003"),
        (worse, "000004"),
    ]:
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        journal.append(node)

    monkeypatch.setattr("aide.agent.hypothesis_root_pool_exhausted", lambda cfg, journal: True)
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda cfg, journal, parent_nodes: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is parent
```

Expected current behavior: the first two new tests fail because no saturation filter exists.

- [ ] **Step 4: Implement metric comparison helpers**

In `aide/agent.py`, add helpers near `_metric_for_search`:

```python
def _node_improves_parent(child: Node, parent: Node) -> bool:
    if child.metric is None or child.metric.value is None:
        return False
    if parent.metric is None or parent.metric.value is None:
        return False
    child_value = float(child.metric.value)
    parent_value = float(parent.metric.value)
    if child.metric.maximize is False:
        return child_value < parent_value
    return child_value > parent_value


def _is_scored_non_improving_child(child: Node, parent: Node) -> bool:
    if child.is_buggy or child.is_terminal_failure:
        return False
    if child.metric is None or child.metric.value is None:
        return False
    return not _node_improves_parent(child, parent)


def _has_improving_child(node: Node) -> bool:
    return any(_node_improves_parent(child, node) for child in node.children)


def _non_improving_child_count(node: Node) -> int:
    return sum(
        1
        for child in node.children
        if _is_scored_non_improving_child(child, node)
    )


def _is_hypothesis_parent_saturated(node: Node, *, limit: int) -> bool:
    if limit <= 0:
        return False
    if _has_improving_child(node):
        return False
    return _non_improving_child_count(node) >= limit
```

- [ ] **Step 5: Apply saturation filter only in hypothesis exploitation phase**

In `Agent.search_policy`, after `good_nodes = filter_hypothesis_candidate_parents(...)`, add:

```python
            limit = int(
                getattr(
                    search_cfg,
                    "hypothesis_max_non_improving_children_per_parent",
                    3,
                )
            )
            if limit > 0:
                good_nodes = [
                    node
                    for node in good_nodes
                    if not _is_hypothesis_parent_saturated(node, limit=limit)
                ]
```

Important: apply this after root phase is exhausted. Root opening still uses `research.hypothesis_root_limit`. The saturation filter only affects selecting existing good nodes for child branching.

- [ ] **Step 6: Run focused search policy tests**

Run:

```bash
uv run pytest tests/test_agent_search_policy.py::test_hypothesis_search_skips_parent_after_non_improving_child_limit tests/test_agent_search_policy.py::test_hypothesis_search_keeps_parent_with_improving_child_available tests/test_agent_search_policy.py::test_hypothesis_non_improving_limit_ignores_bug_children tests/test_agent_search_policy.py::test_search_policy_zero_exploration_keeps_greedy_selection
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add aide/agent.py tests/test_agent_search_policy.py
git commit -m "Limit non-improving hypothesis branches"
```

---

### Task 4: Verify Resume and Root-Limit Extension Behavior

**Files:**
- Test: `tests/test_research_advisor.py`
- Test: `tests/test_agent_search_policy.py`

- [ ] **Step 1: Add test for extended root limit preserving child access to new root scores**

Add this to `tests/test_research_advisor.py`:

```python
def test_child_ranking_uses_new_root_scores_after_root_limit_extension(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.search.hypothesis_child_order = "root_score"
    for idx in range(1, 5):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )

    journal = Journal()
    parent = _node(0.91, code="print('parent')", plan="parent")
    parent.research_mode = "hypothesis"
    parent.research_hypotheses_offered = ["000001"]
    old_root = _node(0.92, code="print('old')", plan="old root")
    old_root.research_mode = "hypothesis"
    old_root.research_hypotheses_offered = ["000002"]
    new_root = _node(0.95, code="print('new')", plan="new root")
    new_root.research_mode = "hypothesis"
    new_root.research_hypotheses_offered = ["000003"]
    journal.append(parent)
    journal.append(old_root)
    journal.append(new_root)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=parent,
        completed_steps=3,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000003"]
```

This models resume after increasing `research.hypothesis_root_limit`: newly scored roots become part of the child ranking automatically because the ranking is derived from the journal, not from a cached list.

- [ ] **Step 2: Run resume-relevant tests**

Run:

```bash
uv run pytest tests/test_research_advisor.py::test_child_ranking_uses_new_root_scores_after_root_limit_extension tests/test_research_advisor.py::test_hypothesis_root_pool_respects_configured_root_limit
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_research_advisor.py
git commit -m "Cover resumed hypothesis root ranking"
```

---

### Task 5: Final Verification

**Files:**
- Verify changed code and tests

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest tests/test_research_advisor.py tests/test_agent_search_policy.py
```

Expected: pass.

- [ ] **Step 2: Run lint on changed files**

Run:

```bash
uv run ruff check aide/research.py aide/agent.py aide/utils/config.py tests/test_research_advisor.py tests/test_agent_search_policy.py
```

Expected: pass.

- [ ] **Step 3: Inspect final diff**

Run:

```bash
git diff --stat
```

Expected: changes limited to config, search policy, hypothesis selection, and focused tests.

Run:

```bash
git diff
```

Expected:
- no unrelated TUI changes,
- no changes to AutoGluon runner behavior,
- no changes to root exploration limit semantics,
- no code that treats bugs or failed nodes as evidence against a hypothesis.

- [ ] **Step 4: Final commit if any verification-only edits were needed**

If Task 5 required fixes after Task 4 commit, commit them:

```bash
git add aide/research.py aide/agent.py aide/utils/config.py tests/test_research_advisor.py tests/test_agent_search_policy.py
git commit -m "Verify ranked hypothesis branching"
```

---

## Expected Runtime Behavior

With:

```bash
research.mode=hypothesis
research.hypothesis_root_limit=100
agent.search.hypothesis_child_order=root_score
agent.search.hypothesis_max_non_improving_children_per_parent=3
```

the run behaves as:

```text
Phase 1: root exploration
  open new root hypotheses until 100 compatible roots have been attempted

Phase 2: exploitation
  select a parent using the existing score/exploration search policy
  choose the child hypothesis deterministically:
    first: root-tested hypotheses ordered by best root score
    then: untested compatible hypotheses in deterministic seeded order
  exclude:
    ancestors already in this branch
    direct-child sibling hypotheses already tried for that parent
  if a parent has 3 scored direct children and none improves the parent:
    stop selecting that parent for more sibling attempts
  bugs, timeouts, and failed nodes do not count as non-improving evidence
```

If a resumed run increases:

```bash
research.hypothesis_root_limit=200
```

then root exploration resumes until 200 compatible root hypotheses have been attempted. After that, child ranking uses all root-scored hypotheses visible in the journal, including the newly added root hypotheses.

## Self-Review

- Spec coverage: covers deterministic child ordering, full-library fallback, root-limit extension on resume, and local parent saturation.
- Placeholder scan: no unfinished placeholder markers.
- Type consistency: new config fields live under `agent.search`; search-policy saturation stays in `aide/agent.py`; hypothesis ordering stays in `aide/research.py`.
- Scope check: no TUI, AutoGluon, prompt, execution timeout, import/export, or report-generation changes are included.
