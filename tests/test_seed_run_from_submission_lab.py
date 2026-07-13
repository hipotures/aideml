import contextlib
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from aide.autogluon_preprocess import extract_preprocess_source
from aide.journal import Journal
from aide.legacy_import_template import TEMPLATE_NAME
from aide.utils import serialize
from aide.utils.artifact_manifest import RESULT_MANIFEST_NAME
from scripts.seed_run_from_submission_lab import (
    rewrite_generated_submission_lab_run,
    seed_run_from_submission_lab,
)


def _write_source_artifact(
    root: Path,
    *,
    run: str,
    timestamp: str,
    step: int,
    code: str,
    autogluon: bool,
) -> Path:
    artifact_dir = root / "logs" / run / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True)
    solution_path = artifact_dir / "solution.py"
    solution_path.write_text(code, encoding="utf-8")
    solution_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 3,
        "kind": "source_node",
        "run": run,
        "timestamp": timestamp,
        "artifact_dir": str(artifact_dir),
        "status": "ok",
        "local_score": 0.9,
        "metric_maximize": True,
        "is_buggy": False,
        "profile": "source-ag-profile" if autogluon else None,
        "files": {
            "solution": {
                "path": "solution.py",
                "sha256": solution_sha,
                "size": len(code.encode("utf-8")),
            }
        },
        "node": {
            "id": f"node-{step}",
            "step": step,
            "status": "ok",
            "plan": f"source plan {step}",
            "is_buggy": False,
            "metric": {"value": 0.9, "maximize": True},
        },
    }
    (artifact_dir / RESULT_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return solution_path


def _ag_source(feature_name: str) -> str:
    return f'''AIDE_AG_CONFIG = {{"profile": "old"}}
RESULT_MARKER = "AIDE_RESULT_JSON:"

def preprocess(df):
    out = df.copy()
    out["{feature_name}"] = 1
    return out
'''


def test_seed_run_imports_unique_public_codes_into_resumable_legacy_roots(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "AIDE_PROJECT_NAME=test-competition\n"
        "AIDE_PROJECT_METRIC=balanced_accuracy\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    first_ag = _write_source_artifact(
        tmp_path,
        run="2-source-one",
        timestamp="20260701T010000-a1111111-1",
        step=1,
        code=_ag_source("feature_a"),
        autogluon=True,
    )
    second_ag = _write_source_artifact(
        tmp_path,
        run="2-source-two",
        timestamp="20260701T020000-b2222222-2",
        step=2,
        code=_ag_source("feature_b"),
        autogluon=True,
    )
    legacy_code = '''from pathlib import Path

Path("./working/submission.csv").write_text("id,target\\n1,0\\n")
print("Validation balanced_accuracy: 0.5")
'''
    legacy = _write_source_artifact(
        tmp_path,
        run="2-source-three",
        timestamp="20260701T030000-c3333333-3",
        step=3,
        code=legacy_code,
        autogluon=False,
    )

    rows = [
        {
            "public_score": "0.95019",
            "sha256": "submission-a",
            "artifact_dir": str(first_ag.parent),
        },
        {
            "public_score": "0.95018",
            "sha256": "submission-a-rerun",
            "artifact_dir": str(tmp_path / "rerun"),
            "source_solution_path": str(first_ag),
        },
        {
            "public_score": "0.95017",
            "sha256": "submission-b",
            "artifact_dir": str(second_ag.parent),
        },
        {
            "public_score": "0.95016",
            "sha256": "submission-legacy",
            "artifact_dir": str(legacy.parent),
        },
    ]
    result = seed_run_from_submission_lab(
        rows=rows,
        limit=3,
        run_id="2-public-legacy-test",
        prepare_workspace=False,
        cli_overrides=[
            f"data_dir={data_dir}",
            "goal=test",
            f"log_dir={tmp_path / 'logs'}",
            f"workspace_dir={tmp_path / 'workspaces'}",
            "generate_report=False",
            "agent.aux=false",
        ],
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)
    cfg = OmegaConf.load(result.log_dir / "config.yaml")
    assert [candidate.public_rank for candidate in result.candidates] == [1, 3, 4]
    assert [node.step for node in journal.nodes] == [0, 1, 2]
    assert [node.status for node in journal.nodes] == ["generated"] * 3
    assert all(node.parent is None for node in journal.nodes)
    assert cfg.agent.mode == "legacy"
    assert cfg.agent.gpu is True
    assert cfg.agent.k_fold_validation == 5
    assert cfg.agent.search.num_drafts == 3
    assert cfg.agent.legacy_starter.autogluon_profile is None

    assert 'out["feature_a"] = 1' in extract_preprocess_source(journal.nodes[0].code)
    assert 'out["feature_b"] = 1' in extract_preprocess_source(journal.nodes[1].code)
    assert journal.nodes[2].code == legacy_code
    for node in journal.nodes:
        compile(node.code, "<imported-root>", "exec")

    imported_ag_code = journal.nodes[0].code
    assert "autogluon" not in imported_ag_code.lower()
    assert "TabularPredictor" not in imported_ag_code
    assert "XGBClassifier" in imported_ag_code
    assert "LGBMClassifier" in imported_ag_code
    assert "CatBoostClassifier" in imported_ag_code
    assert "StratifiedKFold" in imported_ag_code
    assert "LogisticRegression" in imported_ag_code
    assert 'class_weight="balanced"' in imported_ag_code
    assert 'device="cuda"' in imported_ag_code
    assert 'device_type="gpu"' in imported_ag_code
    assert 'task_type="GPU"' in imported_ag_code
    assert 'write_submission(submission)' in imported_ag_code

    class FakeClassifier:
        def __init__(self, **_kwargs):
            self.class_count = 0

        def fit(self, _features, target, **_kwargs):
            self.class_count = len(np.unique(target))
            return self

        def predict_proba(self, features):
            probabilities = np.arange(1, self.class_count + 1, dtype=np.float64)
            probabilities /= probabilities.sum()
            return np.tile(probabilities, (len(features), 1))

    helper_module = types.ModuleType("aide_solution_helpers")
    helper_module.aide_stage = lambda _name: contextlib.nullcontext()
    helper_module.load_competition_data = lambda: None
    helper_module.log_stage = lambda _message: None
    helper_module.write_oof_predictions = lambda _frame: None
    helper_module.write_submission = lambda _frame: None
    helper_module.write_test_predictions = lambda _frame: None
    monkeypatch.setitem(sys.modules, "aide_solution_helpers", helper_module)
    namespace = {}
    definition_code = imported_ag_code.rsplit("\nmain()\n", 1)[0]
    exec(compile(definition_code, "<legacy-template-smoke>", "exec"), namespace)
    namespace["XGBClassifier"] = FakeClassifier
    namespace["LGBMClassifier"] = FakeClassifier
    namespace["CatBoostClassifier"] = FakeClassifier
    namespace["LogisticRegression"] = FakeClassifier
    smoke_train = pd.DataFrame(
        {
            "id": np.arange(15),
            "value": np.linspace(0.0, 1.0, 15),
            "target": ["a", "b", "c"] * 5,
        }
    )
    smoke_test = pd.DataFrame(
        {"id": np.arange(100, 106), "value": np.linspace(0.1, 0.9, 6)}
    )
    smoke_sample = pd.DataFrame({"id": smoke_test["id"], "target": "a"})
    written = {}
    namespace["load_competition_data"] = lambda: (
        smoke_train.copy(),
        smoke_test.copy(),
        smoke_sample.copy(),
    )
    namespace["aide_stage"] = lambda _name: contextlib.nullcontext()
    namespace["log_stage"] = lambda _message: None
    namespace["write_submission"] = lambda frame: written.setdefault(
        "submission", frame.copy()
    )
    namespace["write_oof_predictions"] = lambda frame: written.setdefault(
        "oof", frame.copy()
    )
    namespace["write_test_predictions"] = lambda frame: written.setdefault(
        "test", frame.copy()
    )
    namespace["main"]()
    assert written["submission"].columns.tolist() == ["id", "target"]
    assert len(written["submission"]) == len(smoke_test)
    assert len(written["oof"]) == len(smoke_train)
    assert len(written["test"]) == len(smoke_test)

    imported = json.loads((result.log_dir / "imported_roots.json").read_text())
    assert [root["public_rank"] for root in imported["roots"]] == [1, 3, 4]
    assert [root["source_kind"] for root in imported["roots"]] == [
        "autogluon",
        "autogluon",
        "legacy",
    ]
    assert imported["legacy_import_template"] == TEMPLATE_NAME

    rewritten = rewrite_generated_submission_lab_run(
        run_id=result.run_id,
        rows=rows,
        logs_dir=tmp_path / "logs",
        limit=3,
    )
    assert len(rewritten.seeded) == 3
    rewritten_journal = serialize.load_json(result.log_dir / "journal.json", Journal)
    assert "autogluon" not in rewritten_journal.nodes[0].code.lower()
    assert rewritten_journal.nodes[2].code == legacy_code

    for artifact in result.log_dir.glob("artifacts/*"):
        assert (artifact / "solution.py").exists()
        assert not (artifact / "submission.csv").exists()
