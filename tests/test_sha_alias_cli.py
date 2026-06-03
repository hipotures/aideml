import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script(module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_blend_submissions_treats_sha_as_sha256_alias():
    blend_submissions = _load_script("blend_submissions")

    parser = blend_submissions.build_arg_parser()
    args = parser.parse_args(["--sha", "abc123", "--sha256", "def456"])

    assert args.sha256 == ["abc123", "def456"]
    sha_action = next(action for action in parser._actions if action.dest == "sha256")
    assert "--sha" in sha_action.option_strings


def test_lazypredict_top_preview_treats_sha_as_sha256_alias(capsys):
    lazypredict_top_preview = _load_script("lazypredict_top_preview")

    args = lazypredict_top_preview.parse_args(
        ["--sha", "abc123", "--sha256", "def456"]
    )

    assert args.sha256 == ["abc123", "def456"]
    with pytest.raises(SystemExit):
        lazypredict_top_preview.parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert "--sha " in help_text


def test_lazypredict_top_preview_uses_project_env_defaults(tmp_path, monkeypatch):
    lazypredict_top_preview = _load_script("lazypredict_top_preview")
    monkeypatch.chdir(tmp_path)
    Path(".env").write_text(
        "AIDE_PROJECT_NAME=demo-project\n"
        "AIDE_PROJECT_METRIC=balanced_accuracy\n"
        "AIDE_PROJECT_DATA_DIR=demo-data\n"
    )

    args = lazypredict_top_preview.parse_args([])

    assert args.project == "demo-project"
    assert args.data_dir == Path("demo-data")
    assert args.metric == "balanced_accuracy"
    assert args.tune_metric == "balanced_accuracy"
    assert args.run is None


def test_lazypredict_top_preview_passes_aux_to_preprocess(tmp_path):
    lazypredict_top_preview = _load_script("lazypredict_top_preview")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("id,feature,target\n1,10,0\n2,20,1\n")
    (data_dir / "test.csv").write_text("id,feature\n3,30\n")
    (data_dir / "sample_submission.csv").write_text("id,target\n3,0\n")
    (data_dir / "aux.csv").write_text("aux_feature\n100\n200\n300\n")
    solution_path = tmp_path / "solution.py"
    solution_path.write_text(
        "AIDE_AG_CONFIG = {'aux_file': 'aux.csv'}\n"
        "RESULT_MARKER = 'AIDE_RESULT_JSON:'\n"
        "def preprocess(df: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:\n"
        "    out = df.copy()\n"
        "    out['aux_rows'] = len(aux)\n"
        "    return out\n"
    )

    train_fe, test_fe, _, _, metadata = lazypredict_top_preview.run_artifact_preprocess(
        {"solution_path": str(solution_path)},
        data_dir=data_dir,
        preprocess_time_limit=0,
    )

    assert train_fe["aux_rows"].tolist() == [3, 3]
    assert test_fe["aux_rows"].tolist() == [3]
    assert metadata["preprocess_accepts_aux"] is True
    assert metadata["aux_rows"] == 3


def test_lazypredict_top_preview_summarizes_string_targets():
    lazypredict_top_preview = _load_script("lazypredict_top_preview")

    metadata = lazypredict_top_preview.target_sample_metadata(
        lazypredict_top_preview.pd.Series(["STAR", "GALAXY", "STAR"]),
        lazypredict_top_preview.pd.Series(["QSO", "STAR"]),
    )

    assert metadata["target_mean_train"] is None
    assert metadata["target_counts_train"] == {"STAR": 2, "GALAXY": 1}
    assert metadata["target_counts_valid"] == {"QSO": 1, "STAR": 1}
