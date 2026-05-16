import json
from pathlib import Path

import pytest

from scripts.render_research_prompt import (
    MODE_TEMPLATE_BY_MODE,
    compose_template,
    default_values_path,
    default_output_path,
    load_values,
    render_prompt,
    repo_root,
    write_prompt,
)


def test_render_prompt_replaces_all_placeholders():
    rendered = render_prompt("Task {{TASK_NAME}} optimizes {{METRIC_NAME}}.", {
        "TASK_NAME": "demo-task",
        "METRIC_NAME": "AUC",
    })

    assert rendered == "Task demo-task optimizes AUC."


def test_render_prompt_reports_missing_values():
    with pytest.raises(ValueError, match="missing placeholder values: METRIC_NAME"):
        render_prompt("Metric {{METRIC_NAME}}", {})


def test_load_values_accepts_nested_placeholders_object(tmp_path: Path):
    path = tmp_path / "values.json"
    path.write_text(json.dumps({"placeholders": {"A": "x", "B": ["y"]}}))

    assert load_values(path) == {"A": "x", "B": "[\n  \"y\"\n]"}


def test_default_output_path_uses_mode_and_competition_slug():
    path = default_output_path(
        "legacy",
        {"COMPETITION_OR_PROJECT": "Kaggle Playground S6E5"},
    )

    assert path == Path("/tmp/prompt-legacy-Kaggle-Playground-S6E5.md")


def test_write_prompt_renders_to_explicit_output_path(tmp_path: Path):
    template_path = tmp_path / "template.md"
    values_path = tmp_path / "values.json"
    output_path = tmp_path / "rendered.md"
    template_path.write_text("Use {{TASK_NAME}} with {{METRIC_NAME}}.")
    values_path.write_text(
        json.dumps(
            {
                "TASK_NAME": "demo",
                "METRIC_NAME": "ROC AUC",
                "COMPETITION_OR_PROJECT": "demo",
            }
        )
    )

    result = write_prompt(
        mode="autogluon",
        values_path=values_path,
        template_path=template_path,
        out_path=output_path,
    )

    assert result == output_path
    assert output_path.read_text() == "Use demo with ROC AUC."


def test_write_prompt_applies_value_overrides(tmp_path: Path):
    template_path = tmp_path / "template.md"
    values_path = tmp_path / "values.json"
    output_path = tmp_path / "rendered.md"
    template_path.write_text("Return exactly {{HYPOTHESIS_COUNT}} hypotheses.")
    values_path.write_text(json.dumps({"HYPOTHESIS_COUNT": "10"}))

    write_prompt(
        mode="legacy",
        values_path=values_path,
        template_path=template_path,
        value_overrides={"HYPOTHESIS_COUNT": 7},
        out_path=output_path,
    )

    assert output_path.read_text() == "Return exactly 7 hypotheses."


def test_compose_template_inserts_mode_block():
    composed = compose_template(
        "Common\n{{MODE_SPECIFIC_INSTRUCTIONS}}\nEnd",
        "Mode block\n",
    )

    assert composed == "Common\nMode block\nEnd"


@pytest.mark.parametrize("mode", ["autogluon", "legacy"])
def test_playground_templates_render_without_placeholder_documentation(
    mode: str,
    tmp_path: Path,
):
    output_path = tmp_path / f"{mode}.md"

    write_prompt(
        mode=mode,
        values_path=default_values_path("playground-series-s6e5"),
        template_path=repo_root() / "assets/prompts/research_hypotheses/base_prompt.md",
        mode_template_path=repo_root() / MODE_TEMPLATE_BY_MODE[mode],
        out_path=output_path,
    )

    rendered = output_path.read_text()
    assert not rendered.startswith("# GPT Prompt")
    assert "MODE_SPECIFIC_INSTRUCTIONS" not in rendered
    assert "{{" not in rendered
    assert "}}" not in rendered
    assert "Template rule:" not in rendered
    assert "Required placeholders:" not in rendered
    assert "short task name or slug" not in rendered
    assert "This prompt is ONLY for" not in rendered
    assert "Always include exactly 10 hypotheses." in rendered
    assert "Convert those patterns into 10 reusable" in rendered
    assert "downloadable file" in rendered
    assert "Otherwise return the JSON directly" in rendered
    assert "under 1 hour" in rendered
    assert "200 GB RAM" in rendered
    assert "RTX 4090" in rendered
    assert "millions of generated features" in rendered
    assert "one meaningful mechanism or change family" in rendered
    assert "several unrelated" in rendered
    assert "ideas into one hypothesis" in rendered
    assert "clear ablation plan" in rendered
    assert "tiny threshold tweaks" in rendered
    assert "Write all output fields in English." in rendered
    assert "Write all hypothesis titles" in rendered
    if mode == "legacy":
        assert "AutoGluon" not in rendered
        assert "legacy/manual" not in rendered
        assert "Legacy/manual" not in rendered
