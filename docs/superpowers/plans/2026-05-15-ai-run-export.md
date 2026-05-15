# AI Run Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete AIDE run exporter that writes full node code and tree metadata for external AI review.

**Architecture:** Add a focused utility module under `aide/utils/ai_run_export.py` for loading a run, mapping public scores, computing duplicate hints, and writing JSONL/meta outputs. Add a thin CLI wrapper in `scripts/export_run_for_ai.py`. Keep the exporter read-only with respect to existing logs and leave AIDE runtime behavior unchanged.

**Tech Stack:** Python 3.12, existing AIDE `Journal`/`Node` dataclasses, `aide.utils.serialize`, `aide.utils.artifact_manifest`, `aide.utils.prediction_similarity`, pytest, ruff.

---

## File Structure

- Create `aide/utils/ai_run_export.py`
  - Owns export data extraction, public score matching, duplicate annotation, metadata generation, and file writing.
  - Does not parse CLI arguments.
- Create `scripts/export_run_for_ai.py`
  - Thin command-line entrypoint around `aide.utils.ai_run_export.export_run_for_ai`.
  - Handles paths, numeric options, and process exit codes.
- Create `tests/test_ai_run_export.py`
  - Unit tests for tree preservation, score mapping, duplicate hints, and missing artifact behavior.
- Modify no runtime search files.

---

### Task 1: Core Export Module Skeleton

**Files:**
- Create: `aide/utils/ai_run_export.py`
- Test: `tests/test_ai_run_export.py`

- [ ] **Step 1: Write the failing test for exporting a full tree**

Create `tests/test_ai_run_export.py` with this initial content:

```python
import json
from pathlib import Path

from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.ai_run_export import export_run_for_ai
from aide.utils.metric import MetricValue


def _write_run(tmp_path: Path) -> Path:
    log_dir = tmp_path / "logs" / "run-a"
    log_dir.mkdir(parents=True)
    root = Node(
        code="print('root')\n",
        plan="root plan",
        id="node-root",
        ctime=1770000000.0,
        metric=MetricValue(0.9, maximize=True),
        is_buggy=False,
        analysis="root analysis",
    )
    child = Node(
        code="print('child')\n",
        plan="child plan",
        id="node-child",
        ctime=1770000060.0,
        parent=root,
        metric=MetricValue(0.91, maximize=True),
        is_buggy=False,
        analysis="child analysis",
    )
    bug = Node(
        code="raise RuntimeError('bad')\n",
        plan="bug plan",
        id="node-bug",
        ctime=1770000120.0,
        parent=root,
        status="bug",
        is_buggy=True,
        analysis="bug analysis",
        exc_type="RuntimeError",
    )
    journal = Journal()
    journal.append(root)
    journal.append(child)
    journal.append(bug)
    serialize.dump_json(journal, log_dir / "journal.json")
    return log_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_export_preserves_tree_and_full_code(tmp_path):
    log_dir = _write_run(tmp_path)

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")

    meta = json.loads(result.meta_path.read_text())
    nodes = _read_jsonl(result.nodes_path)

    assert meta["run"] == "run-a"
    assert meta["node_count"] == 3
    assert meta["scored_node_count"] == 2
    assert [node["step"] for node in nodes] == [0, 1, 2]
    assert nodes[0]["node_id"] == "node-root"
    assert nodes[0]["parent_id"] is None
    assert nodes[0]["children_ids"] == ["node-child", "node-bug"]
    assert nodes[1]["parent_id"] == "node-root"
    assert nodes[1]["depth"] == 1
    assert nodes[2]["is_buggy"] is True
    assert nodes[2]["error"]["exc_type"] == "RuntimeError"
    assert nodes[0]["code"] == "print('root')\n"
    assert nodes[1]["local_cv_score"] == 0.91
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_preserves_tree_and_full_code -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `aide.utils.ai_run_export`.

- [ ] **Step 3: Implement the minimal exporter skeleton**

Create `aide/utils/ai_run_export.py` with:

```python
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.artifact_manifest import artifact_timestamp_from_ctime


@dataclass(frozen=True)
class ExportResult:
    export_dir: Path
    meta_path: Path
    nodes_path: Path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metric_value(node: Node) -> float | None:
    return None if node.metric is None else float(node.metric.value)


def _metric_maximize(node: Node) -> bool | None:
    return None if node.metric is None else bool(node.metric.maximize)


def _created_at(node: Node) -> str:
    return dt.datetime.fromtimestamp(node.ctime).astimezone().isoformat()


def _node_depth(node: Node) -> int:
    depth = 0
    parent = node.parent
    seen = {node.id}
    while parent is not None and parent.id not in seen:
        depth += 1
        seen.add(parent.id)
        parent = parent.parent
    return depth


def _artifact_dir(log_dir: Path, node: Node) -> Path:
    return log_dir / "artifacts" / artifact_timestamp_from_ctime(node.ctime)


def _node_record(log_dir: Path, node: Node) -> dict[str, Any]:
    artifact_dir = _artifact_dir(log_dir, node)
    children = sorted(node.children, key=lambda child: child.step)
    return {
        "step": node.step,
        "node_id": node.id,
        "parent_id": node.parent.id if node.parent is not None else None,
        "children_ids": [child.id for child in children],
        "depth": _node_depth(node),
        "status": node.status,
        "is_buggy": bool(node.is_buggy),
        "is_terminal_failure": bool(node.is_terminal_failure),
        "origin": "source_node",
        "local_cv_score": _metric_value(node),
        "kaggle_public_score": None,
        "metric_maximize": _metric_maximize(node),
        "created_at": _created_at(node),
        "exec_time": node.exec_time,
        "artifact_dir": str(artifact_dir) if artifact_dir.exists() else None,
        "code_sha256": _sha256_text(node.code or ""),
        "submission_sha256": None,
        "duplicate": {},
        "plan": node.plan,
        "analysis": node.analysis,
        "validity_warning": node.validity_warning,
        "error": {
            "exc_type": node.exc_type,
            "summary": node.exc_type,
        },
        "code": node.code,
    }


def _meta_record(log_dir: Path, journal: Journal, export_dir: Path) -> dict[str, Any]:
    scored = [node for node in journal.nodes if _metric_value(node) is not None]
    best = max(scored, key=lambda node: _metric_value(node) or float("-inf"), default=None)
    return {
        "schema_version": 1,
        "run": log_dir.name,
        "exported_at": dt.datetime.now().astimezone().isoformat(),
        "node_count": len(journal.nodes),
        "scored_node_count": len(scored),
        "best_local": None
        if best is None
        else {
            "step": best.step,
            "node_id": best.id,
            "local_cv_score": _metric_value(best),
        },
        "best_public": None,
        "config": {},
        "notes_for_ai": (
            "This is a complete AIDE tree export. Nodes are ordered by step and "
            "connected by parent_id/children_ids. Duplicate hints are advisory; "
            "no node was pruned."
        ),
    }


def export_run_for_ai(
    log_dir: Path,
    *,
    output_dir: Path = Path("exports"),
    near_duplicates: bool = True,
    near_submission_rmse_threshold: float = 1e-6,
    prediction_similarity_sample_size: int = 200,
    prediction_similarity_min_common_sample_size: int = 100,
) -> ExportResult:
    if not (log_dir / "journal.json").exists():
        raise FileNotFoundError(f"Missing journal.json in {log_dir}")

    journal = serialize.load_json(log_dir / "journal.json", Journal)
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    export_dir = output_dir / f"{log_dir.name}-{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=False)
    meta_path = export_dir / "run_export.meta.json"
    nodes_path = export_dir / "run_export.nodes.jsonl"

    nodes = [_node_record(log_dir, node) for node in sorted(journal.nodes, key=lambda n: n.step)]
    with nodes_path.open("w", encoding="utf-8") as f:
        for record in nodes:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    meta_path.write_text(
        json.dumps(_meta_record(log_dir, journal, export_dir), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return ExportResult(export_dir=export_dir, meta_path=meta_path, nodes_path=nodes_path)
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_preserves_tree_and_full_code -v
```

Expected: PASS.

- [ ] **Step 5: Run lint for new files**

Run:

```bash
uv run ruff check aide/utils/ai_run_export.py tests/test_ai_run_export.py
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add aide/utils/ai_run_export.py tests/test_ai_run_export.py
git commit -m "Add AI run export skeleton"
```

---

### Task 2: Artifact Hashes and Public Score Mapping

**Files:**
- Modify: `aide/utils/ai_run_export.py`
- Test: `tests/test_ai_run_export.py`

- [ ] **Step 1: Add failing tests for submission hash and public score mapping**

Append these tests to `tests/test_ai_run_export.py`:

```python
def test_export_includes_submission_hash_and_public_score_by_node_id(tmp_path):
    log_dir = _write_run(tmp_path)
    artifact_dir = log_dir / "artifacts" / "20260223T182100"
    artifact_dir.mkdir(parents=True)
    submission = artifact_dir / "submission.csv"
    submission.write_text("id,PitNextLap\n1,0.8\n")
    registry = log_dir.parent / "submission_registry.json"
    registry.write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "competition": "playground-series-s6e5",
                        "run": "run-a",
                        "step": 0,
                        "node_id": "node-root",
                        "timestamp": "20260223T182100",
                        "sha256": "placeholder",
                        "remote_status": "COMPLETE",
                        "public_score": "0.91234",
                    }
                ]
            }
        )
    )

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")
    nodes = _read_jsonl(result.nodes_path)
    meta = json.loads(result.meta_path.read_text())

    assert nodes[0]["submission_sha256"] is not None
    assert nodes[0]["kaggle_public_score"] == 0.91234
    assert meta["best_public"]["node_id"] == "node-root"
    assert meta["best_public"]["kaggle_public_score"] == 0.91234


def test_export_maps_public_score_by_sha_prefix(tmp_path):
    log_dir = _write_run(tmp_path)
    artifact_dir = log_dir / "artifacts" / "20260223T182100"
    artifact_dir.mkdir(parents=True)
    submission = artifact_dir / "submission.csv"
    submission.write_text("id,PitNextLap\n1,0.8\n")
    from aide.utils.ai_run_export import _sha256_file

    full_sha = _sha256_file(submission)
    registry = log_dir.parent / "submission_registry.json"
    registry.write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "competition": "playground-series-s6e5",
                        "run": "other-seeded-run",
                        "step": 0,
                        "timestamp": "20260510T021544",
                        "sha256": full_sha[:10],
                        "remote_status": "COMPLETE",
                        "public_score": "0.92345",
                    }
                ]
            }
        )
    )

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")
    nodes = _read_jsonl(result.nodes_path)

    assert nodes[0]["submission_sha256"] == full_sha
    assert nodes[0]["kaggle_public_score"] == 0.92345
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_includes_submission_hash_and_public_score_by_node_id tests/test_ai_run_export.py::test_export_maps_public_score_by_sha_prefix -v
```

Expected: FAIL because submission hash and registry mapping are not implemented.

- [ ] **Step 3: Implement artifact and public score helpers**

In `aide/utils/ai_run_export.py`, add imports and helpers:

```python
from scripts.smart_kaggle_submit import _parse_public_score, _sha256_matches


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(log_dir: Path) -> list[dict[str, Any]]:
    registry_path = log_dir.parent / "submission_registry.json"
    if not registry_path.exists():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries = data.get("submissions", [])
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError(f"Malformed submission registry: {registry_path}")
    return [entry for entry in entries if isinstance(entry, dict)]


def _timestamp_from_node(node: Node) -> str:
    return artifact_timestamp_from_ctime(node.ctime)


def _public_score_for_node(
    *,
    registry_entries: list[dict[str, Any]],
    run_id: str,
    node: Node,
    submission_sha256: str | None,
) -> float | None:
    timestamp = _timestamp_from_node(node)
    scores: list[float] = []
    for entry in registry_entries:
        if str(entry.get("remote_status", "")).upper() != "COMPLETE":
            continue
        public_score = _parse_public_score(entry.get("public_score"))
        if public_score is None:
            continue
        if entry.get("node_id") == node.id:
            scores.append(public_score)
            continue
        if (
            entry.get("run") == run_id
            and str(entry.get("step")) == str(node.step)
            and entry.get("timestamp") == timestamp
        ):
            scores.append(public_score)
            continue
        entry_sha = entry.get("sha256")
        if submission_sha256 and entry_sha and _sha256_matches(entry_sha, submission_sha256):
            scores.append(public_score)
    return max(scores) if scores else None
```

Then update `_node_record` signature and implementation by replacing the current
function with this complete version:

```python
def _node_record(
    log_dir: Path,
    node: Node,
    *,
    registry_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_dir = _artifact_dir(log_dir, node)
    submission_path = artifact_dir / "submission.csv"
    submission_sha256 = _sha256_file(submission_path) if submission_path.exists() else None
    public_score = _public_score_for_node(
        registry_entries=registry_entries,
        run_id=log_dir.name,
        node=node,
        submission_sha256=submission_sha256,
    )
    children = sorted(node.children, key=lambda child: child.step)
    return {
        "step": node.step,
        "node_id": node.id,
        "parent_id": node.parent.id if node.parent is not None else None,
        "children_ids": [child.id for child in children],
        "depth": _node_depth(node),
        "status": node.status,
        "is_buggy": bool(node.is_buggy),
        "is_terminal_failure": bool(node.is_terminal_failure),
        "origin": "source_node",
        "local_cv_score": _metric_value(node),
        "kaggle_public_score": public_score,
        "metric_maximize": _metric_maximize(node),
        "created_at": _created_at(node),
        "exec_time": node.exec_time,
        "artifact_dir": str(artifact_dir) if artifact_dir.exists() else None,
        "code_sha256": _sha256_text(node.code or ""),
        "submission_sha256": submission_sha256,
        "duplicate": {},
        "plan": node.plan,
        "analysis": node.analysis,
        "validity_warning": node.validity_warning,
        "error": {
            "exc_type": node.exc_type,
            "summary": node.exc_type,
        },
        "code": node.code,
    }
```

Update `export_run_for_ai` to call `_load_registry(log_dir)` once and pass the entries to `_node_record`.

Update `_meta_record` to accept `node_records: list[dict[str, Any]]` and compute:

```python
public_records = [
    record for record in node_records if record["kaggle_public_score"] is not None
]
best_public = max(
    public_records,
    key=lambda record: record["kaggle_public_score"],
    default=None,
)
```

Set `best_public` to `None` or:

```python
{
    "step": best_public["step"],
    "node_id": best_public["node_id"],
    "kaggle_public_score": best_public["kaggle_public_score"],
    "submission_sha256": best_public["submission_sha256"],
}
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_includes_submission_hash_and_public_score_by_node_id tests/test_ai_run_export.py::test_export_maps_public_score_by_sha_prefix -v
```

Expected: PASS.

- [ ] **Step 5: Run all exporter tests**

Run:

```bash
uv run pytest tests/test_ai_run_export.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aide/utils/ai_run_export.py tests/test_ai_run_export.py
git commit -m "Map public scores in AI run export"
```

---

### Task 3: Exact Duplicate Annotations

**Files:**
- Modify: `aide/utils/ai_run_export.py`
- Test: `tests/test_ai_run_export.py`

- [ ] **Step 1: Add failing tests for exact duplicates**

Append:

```python
def test_export_marks_exact_code_and_submission_duplicates_without_pruning(tmp_path):
    log_dir = _write_run(tmp_path)
    first_artifact = log_dir / "artifacts" / "20260223T182100"
    second_artifact = log_dir / "artifacts" / "20260223T182200"
    first_artifact.mkdir(parents=True)
    second_artifact.mkdir(parents=True)
    body = "id,PitNextLap\n1,0.8\n2,0.2\n"
    (first_artifact / "submission.csv").write_text(body)
    (second_artifact / "submission.csv").write_text(body)

    result = export_run_for_ai(log_dir, output_dir=tmp_path / "exports")
    nodes = _read_jsonl(result.nodes_path)

    assert len(nodes) == 3
    assert nodes[0]["duplicate"]["exact_code_role"] == "canonical"
    assert nodes[1]["duplicate"]["exact_code_role"] == "canonical"
    assert nodes[0]["duplicate"]["exact_submission_role"] == "duplicate"
    assert nodes[0]["duplicate"]["exact_submission_canonical_node_id"] == "node-child"
    assert nodes[1]["duplicate"]["exact_submission_role"] == "canonical"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_marks_exact_code_and_submission_duplicates_without_pruning -v
```

Expected: FAIL because duplicate fields are not populated.

- [ ] **Step 3: Implement exact duplicate annotation**

Add helper functions:

```python
def _score_sort_value(record: dict[str, Any]) -> tuple[float, int]:
    score = record.get("local_cv_score")
    normalized = float(score) if score is not None else float("-inf")
    step = int(record["step"]) if record.get("step") is not None else 10**12
    return normalized, -step


def _canonical_by_best_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=_score_sort_value)


def _group_records(
    node_records: list[dict[str, Any]],
    *,
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in node_records:
        value = record.get(key)
        if value:
            groups.setdefault(str(value), []).append(record)
    return groups


def annotate_exact_duplicates(node_records: list[dict[str, Any]]) -> None:
    for record in node_records:
        record["duplicate"] = {
            "exact_code_group": f"code:{record['code_sha256']}",
            "exact_code_role": "canonical",
            "exact_code_canonical_node_id": record["node_id"],
            "exact_submission_group": None,
            "exact_submission_role": None,
            "exact_submission_canonical_node_id": None,
            "near_submission_canonical_node_id": None,
            "near_submission_rmse": None,
        }

    for code_hash, group in _group_records(node_records, key="code_sha256").items():
        canonical = _canonical_by_best_score(group)
        for record in group:
            record["duplicate"]["exact_code_group"] = f"code:{code_hash}"
            record["duplicate"]["exact_code_role"] = (
                "canonical" if record is canonical else "duplicate"
            )
            record["duplicate"]["exact_code_canonical_node_id"] = canonical["node_id"]

    for submission_hash, group in _group_records(
        node_records,
        key="submission_sha256",
    ).items():
        canonical = _canonical_by_best_score(group)
        for record in group:
            record["duplicate"]["exact_submission_group"] = (
                f"submission:{submission_hash}"
            )
            record["duplicate"]["exact_submission_role"] = (
                "canonical" if record is canonical else "duplicate"
            )
            record["duplicate"]["exact_submission_canonical_node_id"] = canonical[
                "node_id"
            ]
```

Call `annotate_exact_duplicates(nodes)` before writing JSONL.

- [ ] **Step 4: Run duplicate test**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_marks_exact_code_and_submission_duplicates_without_pruning -v
```

Expected: PASS.

- [ ] **Step 5: Run all exporter tests and lint**

Run:

```bash
uv run pytest tests/test_ai_run_export.py -v
uv run ruff check aide/utils/ai_run_export.py tests/test_ai_run_export.py
```

Expected: PASS and `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add aide/utils/ai_run_export.py tests/test_ai_run_export.py
git commit -m "Annotate exact duplicates in AI run export"
```

---

### Task 4: Near Submission Duplicate Hints

**Files:**
- Modify: `aide/utils/ai_run_export.py`
- Test: `tests/test_ai_run_export.py`

- [ ] **Step 1: Add failing test for near submission hints**

Append:

```python
def test_export_marks_near_submission_duplicate_hint(tmp_path):
    log_dir = _write_run(tmp_path)
    first_artifact = log_dir / "artifacts" / "20260223T182100"
    second_artifact = log_dir / "artifacts" / "20260223T182200"
    first_artifact.mkdir(parents=True)
    second_artifact.mkdir(parents=True)
    (first_artifact / "submission.csv").write_text(
        "id,PitNextLap\n1,0.80000\n2,0.20000\n"
    )
    (second_artifact / "submission.csv").write_text(
        "id,PitNextLap\n1,0.80001\n2,0.20001\n"
    )

    result = export_run_for_ai(
        log_dir,
        output_dir=tmp_path / "exports",
        near_submission_rmse_threshold=0.0001,
        prediction_similarity_sample_size=2,
        prediction_similarity_min_common_sample_size=2,
    )
    nodes = _read_jsonl(result.nodes_path)

    assert nodes[0]["duplicate"]["near_submission_canonical_node_id"] == "node-child"
    assert nodes[0]["duplicate"]["near_submission_rmse"] is not None
    assert nodes[1]["duplicate"]["near_submission_canonical_node_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_marks_near_submission_duplicate_hint -v
```

Expected: FAIL because near duplicate hints are not implemented.

- [ ] **Step 3: Implement near duplicate annotation**

Add import:

```python
from aide.utils.prediction_similarity import submission_prediction_rmse
```

Add helper:

```python
def annotate_near_submission_duplicates(
    node_records: list[dict[str, Any]],
    *,
    threshold: float,
    sample_size: int,
    min_common_sample_size: int,
) -> None:
    candidates = [
        record for record in node_records if record.get("submission_path") is not None
    ]
    canonicals: list[dict[str, Any]] = []
    for record in sorted(candidates, key=_score_sort_value, reverse=True):
        if record["duplicate"].get("exact_submission_role") == "duplicate":
            continue
        matched = None
        matched_rmse = None
        for canonical in canonicals:
            rmse = submission_prediction_rmse(
                Path(record["submission_path"]),
                Path(canonical["submission_path"]),
                sample_size=sample_size,
                min_common_sample_size=min_common_sample_size,
            )
            if rmse is not None and rmse <= threshold:
                matched = canonical
                matched_rmse = rmse
                break
        if matched is None:
            canonicals.append(record)
            continue
        record["duplicate"]["near_submission_canonical_node_id"] = matched["node_id"]
        record["duplicate"]["near_submission_rmse"] = matched_rmse
```

Add `"submission_path"` to node records as an internal/export field:

```python
"submission_path": str(submission_path) if submission_path.exists() else None,
```

Call this after `annotate_exact_duplicates(nodes)`:

```python
if near_duplicates:
    annotate_near_submission_duplicates(
        nodes,
        threshold=near_submission_rmse_threshold,
        sample_size=prediction_similarity_sample_size,
        min_common_sample_size=prediction_similarity_min_common_sample_size,
    )
```

- [ ] **Step 4: Run focused test**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_marks_near_submission_duplicate_hint -v
```

Expected: PASS.

- [ ] **Step 5: Run exporter tests and lint**

Run:

```bash
uv run pytest tests/test_ai_run_export.py -v
uv run ruff check aide/utils/ai_run_export.py tests/test_ai_run_export.py
```

Expected: PASS and `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add aide/utils/ai_run_export.py tests/test_ai_run_export.py
git commit -m "Annotate near submission duplicates in AI run export"
```

---

### Task 5: CLI Wrapper

**Files:**
- Create: `scripts/export_run_for_ai.py`
- Test: `tests/test_ai_run_export.py`

- [ ] **Step 1: Add failing CLI smoke test**

Append:

```python
def test_export_run_for_ai_cli_writes_export(tmp_path):
    log_dir = _write_run(tmp_path)
    output_dir = tmp_path / "exports"

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_run_for_ai.py",
            str(log_dir),
            "--output-dir",
            str(output_dir),
            "--no-near-duplicates",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "run_export.nodes.jsonl" in result.stdout
    export_dirs = list(output_dir.iterdir())
    assert len(export_dirs) == 1
    assert (export_dirs[0] / "run_export.meta.json").exists()
    assert (export_dirs[0] / "run_export.nodes.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_run_for_ai_cli_writes_export -v
```

Expected: FAIL because `scripts/export_run_for_ai.py` does not exist.

- [ ] **Step 3: Create CLI script**

Create `scripts/export_run_for_ai.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from aide.utils.ai_run_export import export_run_for_ai


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a complete AIDE run tree for external AI review.",
    )
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("exports"))
    parser.add_argument(
        "--near-submission-rmse-threshold",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--prediction-similarity-sample-size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--prediction-similarity-min-common-sample-size",
        type=int,
        default=100,
    )
    parser.add_argument("--no-near-duplicates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = export_run_for_ai(
            args.log_dir,
            output_dir=args.output_dir,
            near_duplicates=not args.no_near_duplicates,
            near_submission_rmse_threshold=args.near_submission_rmse_threshold,
            prediction_similarity_sample_size=args.prediction_similarity_sample_size,
            prediction_similarity_min_common_sample_size=(
                args.prediction_similarity_min_common_sample_size
            ),
        )
    except Exception as exc:
        print(f"Export failed: {exc}")
        return 1
    print(f"Export directory: {result.export_dir}")
    print(f"Metadata: {result.meta_path}")
    print(f"Nodes: {result.nodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI smoke test**

Run:

```bash
uv run pytest tests/test_ai_run_export.py::test_export_run_for_ai_cli_writes_export -v
```

Expected: PASS.

- [ ] **Step 5: Run script against a real run without near duplicates**

Run:

```bash
uv run python scripts/export_run_for_ai.py logs/2-enthusiastic-crane-of-completion --output-dir /tmp/aideml-ai-export --no-near-duplicates
```

Expected: exit code 0 and printed paths to `run_export.meta.json` and `run_export.nodes.jsonl`.

- [ ] **Step 6: Commit**

```bash
git add scripts/export_run_for_ai.py tests/test_ai_run_export.py
git commit -m "Add AI run export CLI"
```

---

### Task 6: Final Verification and Docs Check

**Files:**
- Modify only if tests reveal a bug: `aide/utils/ai_run_export.py`, `scripts/export_run_for_ai.py`, `tests/test_ai_run_export.py`

- [ ] **Step 1: Run all related tests**

Run:

```bash
uv run pytest tests/test_ai_run_export.py tests/test_smart_kaggle_submit.py tests/test_kaggle_submission_lab.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run lint**

Run:

```bash
uv run ruff check aide/utils/ai_run_export.py scripts/export_run_for_ai.py tests/test_ai_run_export.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Inspect a real export summary**

Run:

```bash
uv run python -c "import json, pathlib; p=next(pathlib.Path('/tmp/aideml-ai-export').iterdir()); meta=json.loads((p/'run_export.meta.json').read_text()); print(meta['run'], meta['node_count'], meta['scored_node_count'], meta['best_local'])"
```

Expected: prints run id, nonzero node counts, and a `best_local` object.

- [ ] **Step 4: Inspect top nodes from JSONL**

Run:

```bash
uv run python -c "import json, pathlib; p=next(pathlib.Path('/tmp/aideml-ai-export').iterdir())/'run_export.nodes.jsonl'; rows=[json.loads(line) for line in p.read_text().splitlines()]; rows=sorted([r for r in rows if r['local_cv_score'] is not None], key=lambda r: r['local_cv_score'], reverse=True); print(rows[0]['step'], rows[0]['local_cv_score'], rows[0]['node_id'])"
```

Expected: prints the top local node from the exported file.

- [ ] **Step 5: Commit any final fixes**

If Step 1-4 required edits, commit them:

```bash
git add aide/utils/ai_run_export.py scripts/export_run_for_ai.py tests/test_ai_run_export.py
git commit -m "Stabilize AI run export"
```

If no edits were required, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Full run export with full code: Task 1.
- Parent/child/depth tree reconstruction: Task 1.
- Public score mapping including SHA prefixes: Task 2.
- Exact code and submission duplicate hints without pruning: Task 3.
- Near submission duplicate hints using sampled RMSE: Task 4.
- Dedicated CLI: Task 5.
- Real-run verification: Task 6.

No near-code duplicate task is included because the accepted design explicitly deferred it.
