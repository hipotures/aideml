import json
from pathlib import Path

from scripts.import_archived_research_hypotheses import (
    import_archived_research_hypotheses,
)


def _write_existing_hypothesis(root: Path, task: str, hypothesis_id: str) -> None:
    path = (
        root
        / "research_hypotheses"
        / task
        / "hypotheses"
        / f"hypothesis-{hypothesis_id}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "agent_modes": ["legacy", "autogluon"],
                "title": "Race-lap pit-wave context",
                "summary": "Existing summary.",
                "rationale": "Existing rationale.",
                "implementation_hint": "Existing implementation.",
                "expected_effect": "Existing effect.",
                "risk": "Existing risk.",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )


def _write_run_config(logs_dir: Path, run_id: str, agent_mode: str) -> None:
    run_dir = logs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data_dir: !!python/object/apply:pathlib.PosixPath",
                "- aide",
                "- example_tasks",
                "- playground-series-s6e5",
                "agent:",
                "  gpu: true",
                f"  mode: {agent_mode}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_research_response(
    logs_dir: Path,
    run_id: str,
    step: int,
    hypotheses: list[dict[str, object]],
) -> None:
    checkpoint_dir = logs_dir / run_id / "research" / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "response.json").write_text(
        json.dumps(
            {
                "parsed_response": {
                    "summary": "Archived research summary.",
                    "hypotheses": hypotheses,
                }
            }
        ),
        encoding="utf-8",
    )


def _hypothesis(title: str, rationale: str | None = None) -> dict[str, object]:
    rationale = rationale or f"{title} rationale. Additional detail."
    return {
        "title": title,
        "rationale": rationale,
        "implementation_hint": f"{title} implementation.",
        "expected_effect": f"{title} effect.",
        "risk": f"{title} risk.",
        "sources": ["[paper](https://example.com/paper)", ""],
    }


def test_import_archived_research_uses_run_config_mode_and_deduplicates(tmp_path):
    task = "playground-series-s6e5"
    logs_dir = tmp_path / "logs"
    _write_existing_hypothesis(tmp_path, task, "000002")
    _write_run_config(logs_dir, "run-auto", "autogluon_preprocess")
    _write_run_config(logs_dir, "run-legacy", "legacy")
    _write_research_response(
        logs_dir,
        "run-auto",
        7,
        [
            _hypothesis(
                "Latent tyre filter",
                "Build a compact causal tyre-state filter. Keep only past rows.",
            ),
            _hypothesis("Race-lap pit-wave context"),
        ],
    )
    _write_research_response(
        logs_dir,
        "run-legacy",
        11,
        [_hypothesis("Ranked blending of independent model families")],
    )

    result = import_archived_research_hypotheses(
        logs_dir=logs_dir,
        task=task,
        repo_root=tmp_path,
    )

    assert result.created_count == 2
    assert result.duplicate_count == 1
    assert result.duplicate_records[0]["candidate_title"] == "Race-lap pit-wave context"
    assert result.duplicate_records[0]["matched_label"] == "hypothesis-000002.json"
    assert [path.name for path in result.created_paths] == [
        "hypothesis-000003.json",
        "hypothesis-000004.json",
    ]
    auto_payload = json.loads(result.created_paths[0].read_text(encoding="utf-8"))
    legacy_payload = json.loads(result.created_paths[1].read_text(encoding="utf-8"))
    assert auto_payload["agent_modes"] == ["legacy", "autogluon"]
    assert auto_payload["summary"] == "Build a compact causal tyre-state filter."
    assert auto_payload["sources"] == ["https://example.com/paper"]
    assert legacy_payload["agent_modes"] == ["legacy"]


def test_import_archived_research_skips_runs_without_config(tmp_path):
    task = "playground-series-s6e5"
    logs_dir = tmp_path / "logs"
    _write_research_response(
        logs_dir,
        "run-without-config",
        7,
        [_hypothesis("Configless hypothesis")],
    )

    result = import_archived_research_hypotheses(
        logs_dir=logs_dir,
        task=task,
        repo_root=tmp_path,
    )

    assert result.created_count == 0
    assert result.skipped_missing_config == 1
