# Parallel Generate-Only Roots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in parallel generation of hypothesis ROOT code in `--generate-only` mode, with safe journal writes, rolling refill, retries, and correct TUI follow behavior.

**Architecture:** Keep hypothesis selection and all persistent writes in the main thread. Add explicit artifact directory names for all new nodes, then build a rolling `ThreadPoolExecutor` runner that generates only pre-reserved ROOT hypotheses in isolated agent instances. Existing serial behavior remains the default through `research.hypothesis_root_generate_workers=1`.

**Tech Stack:** Python dataclasses/dataclasses-json, OmegaConf config, `concurrent.futures.ThreadPoolExecutor`, Rich TUI rendering in `aide/run.py`, hypothesis library logic in `aide/research.py`, tests in `tests/test_resume_run.py`, `tests/test_run_tree.py`, and `tests/test_research_advisor.py`.

---

### Task 1: Add Explicit Artifact Directory Names

**Files:**
- Modify: `aide/journal.py`
- Create: `aide/utils/node_artifacts.py`
- Modify: `aide/run.py`
- Modify: `aide/agent.py`
- Modify: `aide/utils/config.py`
- Test: `tests/test_resume_run.py`
- Test: `tests/test_save_run_artifacts.py`

- [ ] **Step 1: Write failing tests for artifact directory lookup**

Add to `tests/test_resume_run.py`:

```python
from aide.utils.node_artifacts import node_artifact_dir, node_artifact_submission_path


def test_node_artifact_dir_uses_explicit_name(tmp_path):
    node = Node(
        code="print('x')",
        plan="x",
        ctime=1_779_492_701.0,
        artifact_dir_name="20260523T220603-a1b2c3d4",
    )

    assert node_artifact_dir(tmp_path, node) == (
        tmp_path / "artifacts" / "20260523T220603-a1b2c3d4"
    )
    assert node_artifact_submission_path(tmp_path, node) == (
        tmp_path / "artifacts" / "20260523T220603-a1b2c3d4" / "submission.csv"
    )


def test_node_artifact_dir_falls_back_to_legacy_ctime_timestamp(tmp_path):
    node = Node(code="print('x')", plan="x", ctime=1_779_492_701.0)

    assert node_artifact_dir(tmp_path, node).name == "20260523T003141"
```

Add to `tests/test_save_run_artifacts.py`:

```python
def test_save_run_uses_explicit_artifact_dir_name(tmp_path):
    cfg = _cfg(tmp_path)
    node = Node(
        code="print('current node')",
        plan="plan",
        ctime=1_779_492_701.0,
        artifact_dir_name="20260523T220603-a1b2c3d4",
    )
    node.is_buggy = False
    journal = Journal()
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    artifact_dir = cfg.log_dir / "artifacts" / "20260523T220603-a1b2c3d4"
    assert (artifact_dir / "solution.py").read_text() == "print('current node')"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_resume_run.py::test_node_artifact_dir_uses_explicit_name tests/test_resume_run.py::test_node_artifact_dir_falls_back_to_legacy_ctime_timestamp tests/test_save_run_artifacts.py::test_save_run_uses_explicit_artifact_dir_name -q
```

Expected: fail because `Node.artifact_dir_name` and `aide.utils.node_artifacts` do not exist yet.

- [ ] **Step 3: Add `artifact_dir_name` to `Node`**

In `aide/journal.py`, add this field after `ctime`:

```python
    artifact_dir_name: str | None = field(default=None, kw_only=True)
```

- [ ] **Step 4: Create artifact helper module**

Create `aide/utils/node_artifacts.py`:

```python
import datetime as dt
import uuid
from pathlib import Path
from typing import Any


def legacy_node_artifact_dir_name(node: Any) -> str:
    return dt.datetime.fromtimestamp(node.ctime).strftime("%Y%m%dT%H%M%S")


def new_artifact_dir_name(*, ctime: float | None = None) -> str:
    timestamp = dt.datetime.fromtimestamp(
        ctime if ctime is not None else dt.datetime.now().timestamp()
    ).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def node_artifact_dir_name(node: Any) -> str:
    explicit = getattr(node, "artifact_dir_name", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return legacy_node_artifact_dir_name(node)


def node_artifact_dir(log_dir: Path | str, node: Any) -> Path:
    return Path(log_dir) / "artifacts" / node_artifact_dir_name(node)


def node_artifact_submission_path(log_dir: Path | str, node: Any) -> Path:
    return node_artifact_dir(log_dir, node) / "submission.csv"
```

- [ ] **Step 5: Route existing artifact lookups through helper**

In `aide/run.py`, import:

```python
from aide.utils.node_artifacts import (
    node_artifact_dir as artifact_dir_for_node,
    node_artifact_submission_path as artifact_submission_path_for_node,
)
```

Replace local implementations:

```python
def _node_artifact_submission_path(cfg, node: Node) -> Path:
    return artifact_submission_path_for_node(cfg.log_dir, node)


def _node_artifact_dir(cfg, node: Node) -> Path:
    return artifact_dir_for_node(cfg.log_dir, node)
```

Keep `_node_artifact_dir_from_ctime()` unchanged for the moment; it is still used for pre-node pending logs and will be removed or bypassed in the parallel task.

In `aide/utils/config.py`, import:

```python
from .node_artifacts import node_artifact_dir as artifact_dir_for_node
```

Then change `_save_node_artifacts()`:

```python
def _save_node_artifacts(cfg: Config, node) -> None:
    artifact_dir = artifact_dir_for_node(cfg.log_dir, node)
    artifact_dir.mkdir(parents=True, exist_ok=True)
```

After replacing the only `_save_node_artifacts()` use of `_node_artifact_timestamp()`, remove `_node_artifact_timestamp()` from `aide/utils/config.py`.

In `aide/agent.py`, import:

```python
from aide.utils.node_artifacts import node_artifact_dir as artifact_dir_for_node
```

Then change `Agent._node_artifact_dir()`:

```python
    def _node_artifact_dir(self, node: Node) -> Path:
        return artifact_dir_for_node(self.cfg.log_dir, node)
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/test_resume_run.py::test_node_artifact_dir_uses_explicit_name tests/test_resume_run.py::test_node_artifact_dir_falls_back_to_legacy_ctime_timestamp tests/test_save_run_artifacts.py::test_save_run_uses_explicit_artifact_dir_name -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add aide/journal.py aide/utils/node_artifacts.py aide/run.py aide/agent.py aide/utils/config.py tests/test_resume_run.py tests/test_save_run_artifacts.py
git commit -m "Add explicit node artifact directories"
```

---

### Task 2: Add And Validate Worker Config

**Files:**
- Modify: `aide/utils/config.py`
- Modify: `aide/utils/config.yaml`
- Modify: `aide/run.py`
- Test: `tests/test_model_reasoning_config.py`
- Test: `tests/test_resume_run.py`

- [ ] **Step 1: Write failing config tests**

Add to `tests/test_model_reasoning_config.py`:

```python
def test_default_hypothesis_root_generate_workers_is_one():
    cfg = load_cfg()

    assert cfg.research.hypothesis_root_generate_workers == 1
```

Add to `tests/test_resume_run.py`:

```python
import pytest


@pytest.mark.parametrize("workers", [0, -1, 9, "four"])
def test_validate_hypothesis_root_generate_workers_rejects_invalid_values(workers):
    cfg = load_cfg()
    cfg.research.hypothesis_root_generate_workers = workers

    with pytest.raises(ValueError, match="hypothesis_root_generate_workers"):
        validate_hypothesis_root_generate_workers(cfg)
```

Add the imports in `tests/test_resume_run.py`:

```python
from aide.run import validate_hypothesis_root_generate_workers
from aide.utils.config import load_cfg
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_model_reasoning_config.py::test_default_hypothesis_root_generate_workers_is_one tests/test_resume_run.py::test_validate_hypothesis_root_generate_workers_rejects_invalid_values -q
```

Expected: fail because the config field and validator do not exist.

- [ ] **Step 3: Add config field**

In `aide/utils/config.py`, add to `ResearchConfig`:

```python
    hypothesis_root_generate_workers: int = 1
```

In `aide/utils/config.yaml`, under `research`, add:

```yaml
  # Number of concurrent hypothesis ROOT code generations in --generate-only.
  # Valid range is 1..8. Default 1 preserves serial behavior.
  hypothesis_root_generate_workers: 1
```

- [ ] **Step 4: Add validation helper and call it before run starts**

In `aide/run.py`, add:

```python
def validate_hypothesis_root_generate_workers(cfg: Config) -> int:
    raw = getattr(cfg.research, "hypothesis_root_generate_workers", 1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            "research.hypothesis_root_generate_workers must be an integer from 1 to 8."
        )
    if raw < 1 or raw > 8:
        raise ValueError(
            "research.hypothesis_root_generate_workers must be an integer from 1 to 8."
        )
    return raw
```

Call it once in the main run path after config loading and resume override merge, before creating the live UI:

```python
    hypothesis_root_generate_workers = validate_hypothesis_root_generate_workers(cfg)
```

Pass the validated local value to downstream helpers instead of repeatedly reading raw config.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/test_model_reasoning_config.py::test_default_hypothesis_root_generate_workers_is_one tests/test_resume_run.py::test_validate_hypothesis_root_generate_workers_rejects_invalid_values -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add aide/utils/config.py aide/utils/config.yaml aide/run.py tests/test_model_reasoning_config.py tests/test_resume_run.py
git commit -m "Add generate-only root worker config"
```

---

### Task 3: Add Serial Root Reservation And Retry Queue

**Files:**
- Modify: `aide/research.py`
- Test: `tests/test_research_advisor.py`

- [ ] **Step 1: Write failing reservation tests**

Add near existing root selection tests in `tests/test_research_advisor.py`:

```python
def test_reserve_hypothesis_roots_returns_unique_manifest_score_order(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_order = "manifest_score"
    cfg.research.hypothesis_root_score_mode = "autogluon"
    for hypothesis_id, score in [("000001", 0.91), ("000002", 0.95), ("000003", 0.93)]:
        _write_manual_hypothesis(tmp_path, "playground-series-s6e5", hypothesis_id)
        hypothesis_dir = (
            tmp_path / "research_hypotheses" / "playground-series-s6e5" / hypothesis_id
        )
        (hypothesis_dir / "code_manifest.json").write_text(
            json.dumps(
                {
                    "versions": {
                        "autogluon-001.py": {
                            "score": score,
                            "buggy": False,
                        }
                    }
                }
            )
        )

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=Journal(),
        count=2,
        completed_steps=0,
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == [
        "000002",
        "000003",
    ]
    assert [reservation.completed_steps for reservation in reservations] == [0, 1]
```

Add retry-priority test:

```python
def test_reserve_hypothesis_roots_retries_failed_generation_first(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000002")
    research.record_hypothesis_root_generation_failure(
        cfg,
        hypothesis_id="000002",
        attempts=3,
        message="RuntimeError: network",
    )

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=Journal(),
        count=1,
        completed_steps=0,
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == ["000002"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_research_advisor.py::test_reserve_hypothesis_roots_returns_unique_manifest_score_order tests/test_research_advisor.py::test_reserve_hypothesis_roots_retries_failed_generation_first -q
```

Expected: fail because the reservation API does not exist.

- [ ] **Step 3: Add reservation dataclass**

In `aide/research.py`, add:

```python
@dataclass(frozen=True)
class HypothesisRootReservation:
    selection: ManualHypothesisSelection
    hypothesis_id: str
    completed_steps: int
    retry_attempts: int = 0
```

- [ ] **Step 4: Add retry queue helpers**

In `aide/research.py`, add:

```python
def _root_generation_failures_path(cfg: Config) -> Path:
    return _manual_run_dir(cfg) / "root_generation_failures.json"


def _load_root_generation_failures(cfg: Config) -> dict[str, Any]:
    path = _root_generation_failures_path(cfg)
    if not path.exists():
        return {}
    return _read_json(path)


def _write_root_generation_failures(cfg: Config, data: dict[str, Any]) -> None:
    _write_json(_root_generation_failures_path(cfg), data)


def record_hypothesis_root_generation_failure(
    cfg: Config,
    *,
    hypothesis_id: str,
    attempts: int,
    message: str,
) -> None:
    failures = _load_root_generation_failures(cfg)
    failures[hypothesis_id] = {
        "attempts": attempts,
        "message": message,
        "last_failed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    _write_root_generation_failures(cfg, failures)


def clear_hypothesis_root_generation_failure(cfg: Config, *, hypothesis_id: str) -> None:
    failures = _load_root_generation_failures(cfg)
    if hypothesis_id in failures:
        del failures[hypothesis_id]
        _write_root_generation_failures(cfg, failures)
```

- [ ] **Step 5: Add reservation API**

In `aide/research.py`, implement `reserve_hypothesis_roots()` by calling `select_hypothesis_for_node()` serially with a temporary journal view that includes already reserved hypothesis ids as generated root placeholders:

```python
def reserve_hypothesis_roots(
    cfg: Config,
    *,
    journal: Journal,
    count: int,
    completed_steps: int,
    repo_root: Path = REPO_ROOT,
) -> list[HypothesisRootReservation]:
    reservations: list[HypothesisRootReservation] = []
    working_journal = Journal(nodes=list(journal.nodes))
    failures = _load_root_generation_failures(cfg)
    retry_ids = sorted(failures)

    for retry_id in retry_ids:
        if len(reservations) >= count:
            break
        library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
        retry = next(
            (hypothesis for hypothesis in library.hypotheses if hypothesis.id == retry_id),
            None,
        )
        if retry is None:
            continue
        step = completed_steps + len(reservations)
        created_at = dt.datetime.now().isoformat(timespec="seconds")
        _write_manual_source_ref(cfg=cfg, library=library, created_at=created_at)
        _append_manual_offer(
            cfg=cfg,
            completed_steps=step,
            offered_ids=[retry.id],
            source_hash=library.source_hash,
            created_at=created_at,
        )
        _record_manual_offer_usage(
            cfg=cfg,
            offered_ids=[retry.id],
            completed_steps=step,
            created_at=created_at,
        )
        selection = ManualHypothesisSelection(
            completed_steps=step,
            source_hash=library.source_hash,
            source_dir=library.source_dir,
            hypotheses=[retry],
        )
        reservations.append(
            HypothesisRootReservation(
                selection=selection,
                hypothesis_id=retry.id,
                completed_steps=step,
                retry_attempts=int(failures.get(retry.id, {}).get("attempts", 0)),
            )
        )
        placeholder = Node(code="", plan="reserved")
        placeholder.research_mode = "hypothesis"
        placeholder.research_hypotheses_offered = [retry.id]
        working_journal.append(placeholder)

    while len(reservations) < count:
        selection = select_hypothesis_for_node(
            cfg,
            journal=working_journal,
            parent_node=None,
            completed_steps=completed_steps + len(reservations),
            repo_root=repo_root,
        )
        if len(selection.hypotheses) != 1:
            raise ValueError("Hypothesis mode requires exactly one selected root.")
        hypothesis_id = selection.hypotheses[0].id
        reservations.append(
            HypothesisRootReservation(
                selection=selection,
                hypothesis_id=hypothesis_id,
                completed_steps=selection.completed_steps,
            )
        )
        placeholder = Node(code="", plan="reserved")
        placeholder.research_mode = "hypothesis"
        placeholder.research_hypotheses_offered = [hypothesis_id]
        working_journal.append(placeholder)

    return reservations
```

If `select_hypothesis_for_node()` raises the existing no-candidate `ValueError`, let it propagate for now; the parallel runner task will translate pool exhaustion into no more launches.

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/test_research_advisor.py::test_reserve_hypothesis_roots_returns_unique_manifest_score_order tests/test_research_advisor.py::test_reserve_hypothesis_roots_retries_failed_generation_first -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add aide/research.py tests/test_research_advisor.py
git commit -m "Reserve hypothesis roots serially"
```

---

### Task 4: Generate A Preselected Root In An Isolated Agent

**Files:**
- Modify: `aide/agent.py`
- Test: `tests/test_research_advisor.py`

- [ ] **Step 1: Write failing preselected-generation test**

Add to `tests/test_research_advisor.py`:

```python
def test_agent_generates_preselected_hypothesis_root_without_selector(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    library = research.load_manual_hypothesis_library(cfg, repo_root=tmp_path)
    selection = research.ManualHypothesisSelection(
        completed_steps=0,
        source_hash=library.source_hash,
        source_dir=library.source_dir,
        hypotheses=[library.hypotheses[0]],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("selector called")),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        return "I will implement hypothesis 000001.", "print('root')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent.generate_preselected_hypothesis_root(
        selection,
        node_ctime=1_779_492_701.0,
        llm_log_dir=tmp_path / "artifact",
        artifact_dir_name="20260523T220603-a1b2c3d4",
    )

    assert node.code == "print('root')"
    assert node.ctime == 1_779_492_701.0
    assert node.artifact_dir_name == "20260523T220603-a1b2c3d4"
    assert node.research_hypotheses_offered == ["000001"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest tests/test_research_advisor.py::test_agent_generates_preselected_hypothesis_root_without_selector -q
```

Expected: fail because `generate_preselected_hypothesis_root()` does not exist.

- [ ] **Step 3: Refactor `_draft_hypothesis_root` to accept a selection**

In `aide/agent.py`, change `_draft_hypothesis_root()` into:

```python
    def _draft_hypothesis_root(
        self,
        selection: ManualHypothesisSelection | None = None,
    ) -> Node:
        if selection is None:
            selection = select_hypothesis_for_node(
                self.cfg,
                journal=self.journal,
                parent_node=None,
                completed_steps=len(self.journal.nodes),
            )
        if len(selection.hypotheses) != 1:
            raise ValueError("Hypothesis mode requires exactly one selected root.")
        hypothesis_id = selection.hypotheses[0].id
        self.active_research_hypothesis_id = hypothesis_id
        self.active_research_hypothesis_log_hint = format_hypothesis_for_log_panel(
            selection
        )
        root_code = load_hypothesis_root_code(self.cfg, hypothesis_id)
        if root_code is not None:
            metadata = {
                "research_mode": "hypothesis",
                "research_hypotheses_offered": [hypothesis_id],
                "research_source_hash": selection.source_hash,
            }
            plan = (
                f"Loaded library {root_code.agent_mode} root code for "
                f"hypothesis {hypothesis_id} from {root_code.path.name}."
            )
            return self._apply_research_metadata(
                self._new_node(plan=plan, code=root_code.code),
                metadata,
            )
        return self._draft(hypothesis_selection=selection)
```

Add this import with the existing `aide.research` imports:

```python
from aide.research import ManualHypothesisSelection
```

- [ ] **Step 4: Add preselected public method**

In `aide/agent.py`, add below `generate_node()`:

```python
    def generate_preselected_hypothesis_root(
        self,
        selection: ManualHypothesisSelection,
        *,
        node_ctime: float,
        llm_log_dir: Path,
        artifact_dir_name: str,
    ) -> Node:
        self.set_active_stage("generating")
        self.active_parent_node = None
        self.active_research_hypothesis_id = selection.hypotheses[0].id
        self.active_research_hypothesis_log_hint = format_hypothesis_for_log_panel(
            selection
        )
        previous_ctime = self._pending_node_ctime
        previous_log_dir = self._pending_llm_log_dir
        self._pending_node_ctime = node_ctime
        self._pending_llm_log_dir = llm_log_dir
        try:
            node = self._draft_hypothesis_root(selection)
            node.artifact_dir_name = artifact_dir_name
            return node
        finally:
            self._pending_node_ctime = previous_ctime
            self._pending_llm_log_dir = previous_log_dir
```

- [ ] **Step 5: Keep artifact assignment in the run loop**

Do not set all node artifact names in `Agent._new_node()`. Task 5 assigns names from the run loop so artifact directories and node ctimes remain coordinated.

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/test_research_advisor.py::test_agent_generates_preselected_hypothesis_root_without_selector tests/test_research_advisor.py::test_agent_loads_library_hypothesis_root_without_llm -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add aide/agent.py tests/test_research_advisor.py
git commit -m "Generate preselected hypothesis roots"
```

---

### Task 5: Assign Explicit Artifact Names To All New Nodes

**Files:**
- Modify: `aide/run.py`
- Modify: `aide/utils/node_artifacts.py`
- Test: `tests/test_resume_run.py`

- [ ] **Step 1: Write failing test for artifact allocation**

Add to `tests/test_resume_run.py`:

```python
def test_allocate_node_artifact_slot_sets_unique_explicit_name(tmp_path):
    first_ctime, first_dir_name, first_dir = allocate_node_artifact_slot(tmp_path)
    second_ctime, second_dir_name, second_dir = allocate_node_artifact_slot(tmp_path)

    assert first_ctime <= second_ctime
    assert first_dir_name != second_dir_name
    assert first_dir.name == first_dir_name
    assert second_dir.name == second_dir_name
    assert first_dir.exists()
    assert second_dir.exists()
    assert len(first_dir_name.split("-")[-1]) == 8
```

Import:

```python
from aide.run import allocate_node_artifact_slot
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest tests/test_resume_run.py::test_allocate_node_artifact_slot_sets_unique_explicit_name -q
```

Expected: fail because `allocate_node_artifact_slot()` does not exist.

- [ ] **Step 3: Add allocator helper**

In `aide/run.py`, import:

```python
from aide.utils.node_artifacts import new_artifact_dir_name
```

Add near `_node_artifact_dir_from_ctime()`:

```python
def allocate_node_artifact_slot(log_dir: Path | str) -> tuple[float, str, Path]:
    ctime = time.time()
    artifacts_dir = Path(log_dir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    while True:
        dir_name = new_artifact_dir_name(ctime=ctime)
        artifact_dir = artifacts_dir / dir_name
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return ctime, dir_name, artifact_dir
```

- [ ] **Step 4: Use allocator in the serial run loop**

In the existing generation block in `aide/run.py`, replace:

```python
node_ctime = time.time()
pending_artifact_dir = _node_artifact_dir_from_ctime(cfg, node_ctime)
pending_artifact_dir.mkdir(parents=True, exist_ok=True)
```

with:

```python
node_ctime, artifact_dir_name, pending_artifact_dir = allocate_node_artifact_slot(
    cfg.log_dir
)
```

After `result_node = run_with_live_refresh(...)` and before `pending_artifact_dir = None`, assign:

```python
result_node.artifact_dir_name = artifact_dir_name
```

If `agent.generate_node()` already returned a node with an explicit `artifact_dir_name`, preserve it:

```python
if result_node.artifact_dir_name is None:
    result_node.artifact_dir_name = artifact_dir_name
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/test_resume_run.py::test_allocate_node_artifact_slot_sets_unique_explicit_name tests/test_resume_run.py::test_record_generated_only_node_marks_saves_and_appends -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add aide/run.py aide/utils/node_artifacts.py tests/test_resume_run.py
git commit -m "Allocate explicit artifact slots for generated nodes"
```

---

### Task 6: Render Multiple In-Flight ROOT Generations

**Files:**
- Modify: `aide/run.py`
- Test: `tests/test_run_tree.py`

- [ ] **Step 1: Write failing tree tests**

Add to `tests/test_run_tree.py`:

```python
def test_tree_renders_multiple_active_root_generations():
    journal = Journal()
    view = build_tree_view(
        journal,
        active_root_generations=[
            ActiveRootGeneration("000405", launched_index=1),
            ActiveRootGeneration("000941", launched_index=2),
        ],
        active_stage="generating",
    )
    output = _render_ansi(render_tree_view(view, focused_item_id="active:000941", viewport_height=10))

    assert "[ ]·000405" in output
    assert "[*]·000941" in output
    assert active_tree_item_id(view) == "active:000941"
```

Add refill follow test:

```python
def test_tree_follow_active_uses_most_recent_launched_generation():
    journal = Journal()
    view = build_tree_view(
        journal,
        active_root_generations=[
            ActiveRootGeneration("000941", launched_index=5),
            ActiveRootGeneration("000405", launched_index=6),
        ],
        active_stage="generating",
    )

    assert active_tree_item_id(view) == "active:000405"
```

Import:

```python
from aide.run import ActiveRootGeneration
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_run_tree.py::test_tree_renders_multiple_active_root_generations tests/test_run_tree.py::test_tree_follow_active_uses_most_recent_launched_generation -q
```

Expected: fail because `ActiveRootGeneration` and multi-active rendering do not exist.

- [ ] **Step 3: Add active generation dataclass**

In `aide/run.py`, add:

```python
@dataclass(frozen=True)
class ActiveRootGeneration:
    hypothesis_id: str
    launched_index: int
```

- [ ] **Step 4: Extend `build_tree_view()`**

Add parameter:

```python
    active_root_generations: list[ActiveRootGeneration] | None = None,
```

Compute:

```python
    active_root_generations = active_root_generations or []
    latest_active_root = (
        max(active_root_generations, key=lambda item: item.launched_index)
        if active_root_generations
        else None
    )
```

Add helper near `append_active()`:

```python
    def append_active_root_generations(parent_id: str) -> None:
        nonlocal active_item_id
        for generation in active_root_generations:
            is_latest = (
                latest_active_root is not None
                and generation.hypothesis_id == latest_active_root.hypothesis_id
            )
            item_id = f"active:{generation.hypothesis_id}"
            prefix = "└── " if is_latest else "├── "
            line = Text(prefix)
            line.append_text(
                _tree_active_placeholder_line(
                    active_stage=active_stage,
                    active_hypothesis_id=generation.hypothesis_id,
                    blink_on=blink_on if is_latest else False,
                )
            )
            append_item(TreeViewItem(item_id, parent_id, line, focus_start=len(prefix)))
            if is_latest:
                active_item_id = item_id
```

At root rendering, suppress the old single active root placeholder when `active_root_generations` is non-empty and call `append_active_root_generations("header")` after normal roots.

- [ ] **Step 5: Leave production call sites unchanged in this task**

For this task, pass no active roots in production call sites. Task 7 wires runtime state into `current_tree_view()`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/test_run_tree.py::test_tree_renders_multiple_active_root_generations tests/test_run_tree.py::test_tree_follow_active_uses_most_recent_launched_generation tests/test_run_tree.py::test_tree_best_and_active_focus_targets -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add aide/run.py tests/test_run_tree.py
git commit -m "Render parallel active root generations"
```

---

### Task 7: Implement Parallel Generate-Only ROOT Runner

**Files:**
- Modify: `aide/run.py`
- Modify: `aide/research.py`
- Test: `tests/test_resume_run.py`
- Test: `tests/test_research_advisor.py`

- [ ] **Step 1: Write failing retry/refill unit tests**

Add to `tests/test_resume_run.py`:

```python
def test_generation_retry_policy_stops_refill_after_three_failures():
    state = ParallelRootFailureState()

    assert state.record_failure("000405", RuntimeError("network")) is False
    assert state.record_failure("000405", RuntimeError("network")) is False
    assert state.record_failure("000405", RuntimeError("network")) is True
    assert state.stop_refill is True
```

Add import:

```python
from aide.run import ParallelRootFailureState
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest tests/test_resume_run.py::test_generation_retry_policy_stops_refill_after_three_failures -q
```

Expected: fail because `ParallelRootFailureState` does not exist.

- [ ] **Step 3: Add failure state helper**

In `aide/run.py`, add:

```python
@dataclass
class ParallelRootFailureState:
    attempts_by_hypothesis: dict[str, int] = field(default_factory=dict)
    stop_refill: bool = False

    def record_failure(self, hypothesis_id: str, exc: BaseException) -> bool:
        attempts = self.attempts_by_hypothesis.get(hypothesis_id, 0) + 1
        self.attempts_by_hypothesis[hypothesis_id] = attempts
        if attempts >= 3:
            self.stop_refill = True
            return True
        return False
```

Update the existing `dataclasses` import to include `field`:

```python
from dataclasses import dataclass, field, replace
```

- [ ] **Step 4: Add worker result dataclasses**

In `aide/run.py`, add:

```python
@dataclass(frozen=True)
class ParallelRootJob:
    reservation: HypothesisRootReservation
    node_ctime: float
    artifact_dir_name: str
    artifact_dir: Path
    launched_index: int


@dataclass(frozen=True)
class ParallelRootResult:
    job: ParallelRootJob
    node: Node
```

Import `HypothesisRootReservation`, `reserve_hypothesis_roots`, `record_hypothesis_root_generation_failure`, and `clear_hypothesis_root_generation_failure` from `aide.research`.

- [ ] **Step 5: Add worker generation function**

In `aide/run.py`, add:

```python
def generate_reserved_hypothesis_root(
    *,
    base_agent: Agent,
    journal: Journal,
    job: ParallelRootJob,
) -> ParallelRootResult:
    worker_agent = Agent(
        task_desc=base_agent.task_desc,
        cfg=base_agent.cfg,
        journal=Journal(nodes=list(journal.nodes)),
    )
    worker_agent.data_preview = base_agent.data_preview
    node = worker_agent.generate_preselected_hypothesis_root(
        job.reservation.selection,
        node_ctime=job.node_ctime,
        llm_log_dir=job.artifact_dir,
        artifact_dir_name=job.artifact_dir_name,
    )
    return ParallelRootResult(job=job, node=node)
```

`Agent.task_desc` is assigned in `Agent.__init__`, so this helper can construct the worker agent directly from `base_agent.task_desc`, `base_agent.cfg`, and a journal snapshot.

- [ ] **Step 6: Add launch helper**

In `aide/run.py`, add:

```python
def make_parallel_root_job(
    *,
    cfg: Config,
    reservation: HypothesisRootReservation,
    launched_index: int,
) -> ParallelRootJob:
    node_ctime, artifact_dir_name, artifact_dir = allocate_node_artifact_slot(cfg.log_dir)
    return ParallelRootJob(
        reservation=reservation,
        node_ctime=node_ctime,
        artifact_dir_name=artifact_dir_name,
        artifact_dir=artifact_dir,
        launched_index=launched_index,
    )
```

- [ ] **Step 7: Wire rolling pool into main loop**

In `aide/run.py`, in the live loop, before the existing serial generation branch, detect:

```python
parallel_root_workers = (
    hypothesis_root_generate_workers
    if runtime_options.skip_execution
    and cfg.research.mode == "hypothesis"
    and hypothesis_root_generate_workers > 1
    else 1
)
```

When `parallel_root_workers > 1`, replace the serial `prepare_step()/generate_node()` path for ROOT generation with:

```python
with ThreadPoolExecutor(max_workers=parallel_root_workers) as executor:
    futures: dict[Future[ParallelRootResult], ParallelRootJob] = {}
    failure_state = ParallelRootFailureState()
    launched_counter = 0

    def launch_until_full() -> None:
        nonlocal launched_counter, operator_notice
        if failure_state.stop_refill:
            return
        slots = parallel_root_workers - len(futures)
        if slots <= 0:
            return
        try:
            reservations = reserve_hypothesis_roots(
                cfg,
                journal=journal,
                count=slots,
                completed_steps=len(journal) + len(futures),
            )
        except ValueError:
            return
        for reservation in reservations:
            launched_counter += 1
            job = make_parallel_root_job(
                cfg=cfg,
                reservation=reservation,
                launched_index=launched_counter,
            )
            futures[
                executor.submit(
                    generate_reserved_hypothesis_root,
                    base_agent=agent,
                    journal=journal,
                    job=job,
                )
            ] = job
```

Poll futures with `wait(..., timeout=1, return_when=FIRST_COMPLETED)` so `live.update()` and `drain_left_panel_navigation()` keep working. On success, call `record_generated_only_node()` in the main thread and `clear_hypothesis_root_generation_failure()`. On failure, call `failure_state.record_failure()`, `record_hypothesis_root_generation_failure()`, sleep 5 seconds before relaunching that same reservation unless attempts reached 3.

- [ ] **Step 8: Wire active placeholders**

Track in-flight jobs:

```python
active_root_generations = [
    ActiveRootGeneration(
        job.reservation.hypothesis_id,
        launched_index=job.launched_index,
    )
    for job in futures.values()
]
```

Pass this list to `build_tree_view()` from `current_tree_view()`.

- [ ] **Step 9: Keep serial path unchanged for `workers=1`**

Ensure the existing code path still runs when `hypothesis_root_generate_workers == 1`. No parallel executor should be created for the default setting.

- [ ] **Step 10: Run focused tests**

Run:

```bash
uv run pytest tests/test_resume_run.py::test_generation_retry_policy_stops_refill_after_three_failures tests/test_run_tree.py::test_tree_renders_multiple_active_root_generations tests/test_research_advisor.py::test_reserve_hypothesis_roots_returns_unique_manifest_score_order -q
```

Expected: pass.

- [ ] **Step 11: Commit**

```bash
git add aide/run.py aide/research.py tests/test_resume_run.py tests/test_research_advisor.py
git commit -m "Generate hypothesis roots in parallel"
```

---

### Task 8: Show Worker Count In Run Data

**Files:**
- Modify: `aide/run.py`
- Test: `tests/test_run_tree.py`

- [ ] **Step 1: Write failing run-data test**

Add to `tests/test_run_tree.py`:

```python
def test_run_data_shows_generate_only_worker_count():
    cfg = _cfg()
    cfg.agent.mode = "legacy"
    cfg.research.hypothesis_root_generate_workers = 4

    summary = build_agent_mode_summary(
        cfg,
        skip_execution=True,
        hypothesis_root_generate_workers=4,
    )
    output = _render_ansi(summary)

    assert "workers" in output
    assert "4" in output
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest tests/test_run_tree.py::test_run_data_shows_generate_only_worker_count -q
```

Expected: fail because `build_agent_mode_summary()` does not accept the worker count yet.

- [ ] **Step 3: Extend summary builder**

In `aide/run.py`, change signature:

```python
def build_agent_mode_summary(
    cfg: Config | None,
    *,
    skip_execution: bool = False,
    hypothesis_root_generate_workers: int = 1,
) -> Group | None:
```

Append workers line only when `skip_execution` is true:

```python
    lines = [Text("Agent", style=TUI_ROW_LABEL_STYLE), mode_line, run_line]
    if skip_execution:
        workers_line = Text()
        workers_line.append("▶ workers   ", style=TUI_ROW_LABEL_STYLE)
        workers_line.append(str(hypothesis_root_generate_workers), style=TUI_NEUTRAL_VALUE_STYLE)
        lines.append(workers_line)
    return Group(*lines)
```

Update call site:

```python
agent_mode_summary = build_agent_mode_summary(
    cfg,
    skip_execution=skip_execution,
    hypothesis_root_generate_workers=hypothesis_root_generate_workers,
)
```

- [ ] **Step 4: Run focused test**

Run:

```bash
uv run pytest tests/test_run_tree.py::test_run_data_shows_generate_only_worker_count tests/test_run_tree.py::test_run_data_shows_agent_mode_and_run_mode -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add aide/run.py tests/test_run_tree.py
git commit -m "Show generate-only root worker count"
```

---

### Task 9: Final Regression Verification

**Files:**
- No code edits expected

- [ ] **Step 1: Run resume/generate-only tests**

Run:

```bash
uv run pytest tests/test_resume_run.py -q
```

Expected: pass.

- [ ] **Step 2: Run tree tests**

Run:

```bash
uv run pytest tests/test_run_tree.py -q
```

Expected: pass.

- [ ] **Step 3: Run research advisor tests**

Run:

```bash
uv run pytest tests/test_research_advisor.py -q
```

Expected: pass.

- [ ] **Step 4: Run lint**

Run:

```bash
uv run ruff check aide/run.py aide/agent.py aide/research.py aide/journal.py aide/utils/config.py aide/utils/node_artifacts.py tests/test_resume_run.py tests/test_run_tree.py tests/test_research_advisor.py tests/test_save_run_artifacts.py tests/test_model_reasoning_config.py
```

Expected: `All checks passed!`

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: clean after the final commit.
