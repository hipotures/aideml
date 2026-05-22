import json
from pathlib import Path

import pytest

from scripts.import_research_hypotheses import import_research_hypotheses


def _write_existing_hypothesis(root: Path, task: str, hypothesis_id: str) -> None:
    path = (
        root
        / "research_hypotheses"
        / task
        / hypothesis_id
        / f"hypothesis-{hypothesis_id}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "agent_modes": ["legacy"],
                "title": "Existing hypothesis",
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


def _incoming_hypothesis(title: str) -> dict[str, object]:
    return {
        "title": title,
        "summary": f"{title} summary.",
        "rationale": f"{title} rationale.",
        "implementation_hint": f"{title} implementation.",
        "expected_effect": f"{title} effect.",
        "risk": f"{title} risk.",
        "sources": [
            "[https://example.com/a](https://example.com/a)",
            " https://example.com/b ",
            "",
        ],
    }


def test_import_appends_autogluon_hypotheses_as_legacy_and_autogluon(tmp_path):
    task = "playground-series-s6e5"
    _write_existing_hypothesis(tmp_path, task, "000010")
    input_path = tmp_path / "from-gpt.json"
    input_path.write_text(
        json.dumps(
            {
                "hypotheses": [
                    _incoming_hypothesis("First imported"),
                    _incoming_hypothesis("Second imported"),
                ]
            }
        ),
        encoding="utf-8",
    )

    result = import_research_hypotheses(
        input_path,
        task=task,
        mode="autogluon",
        repo_root=tmp_path,
    )

    assert [path.name for path in result.created_paths] == [
        "hypothesis-000011.json",
        "hypothesis-000012.json",
    ]
    created = json.loads(result.created_paths[0].read_text(encoding="utf-8"))
    assert created == {
        "enabled": True,
        "agent_modes": ["legacy", "autogluon"],
        "title": "First imported",
        "summary": "First imported summary.",
        "rationale": "First imported rationale.",
        "implementation_hint": "First imported implementation.",
        "expected_effect": "First imported effect.",
        "risk": "First imported risk.",
        "sources": ["https://example.com/a", "https://example.com/b"],
    }


def test_import_legacy_hypotheses_only_for_legacy_mode(tmp_path):
    task = "playground-series-s6e5"
    input_path = tmp_path / "from-gpt.json"
    input_path.write_text(
        json.dumps([_incoming_hypothesis("Legacy imported")]),
        encoding="utf-8",
    )

    result = import_research_hypotheses(
        input_path,
        task=task,
        mode="legacy",
        repo_root=tmp_path,
    )

    created = json.loads(result.created_paths[0].read_text(encoding="utf-8"))
    assert result.created_count == 1
    assert created["agent_modes"] == ["legacy"]


def test_import_rejects_missing_required_fields(tmp_path):
    input_path = tmp_path / "from-gpt.json"
    input_path.write_text(
        json.dumps({"hypotheses": [{"title": "Only title"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field"):
        import_research_hypotheses(
            input_path,
            task="playground-series-s6e5",
            mode="autogluon",
            repo_root=tmp_path,
        )
