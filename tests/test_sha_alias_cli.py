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
