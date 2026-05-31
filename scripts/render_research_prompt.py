from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
MODE_SPECIFIC_PLACEHOLDER = "{{MODE_SPECIFIC_INSTRUCTIONS}}"

BASE_TEMPLATE_PATH = Path("assets/prompts/research_hypotheses/base_prompt.md")
ALLOWED_PACKAGES_PATH = Path("assets/prompts/research_hypotheses/allowed_packages.json")
MODE_TEMPLATE_BY_MODE = {
    "autogluon": Path("assets/prompts/research_hypotheses/modes/autogluon.md"),
    "legacy": Path("assets/prompts/research_hypotheses/modes/full_python.md"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a research-hypothesis prompt from a template and task values.",
    )
    parser.add_argument("mode", choices=sorted(MODE_TEMPLATE_BY_MODE))
    parser.add_argument(
        "--task",
        default="playground-series-s6e6",
        help=(
            "Task slug used to find research_hypotheses/<task>/prompt_values.json "
            "when --values is not provided."
        ),
    )
    parser.add_argument(
        "--values",
        type=Path,
        help="JSON file with placeholder values. Defaults to the selected task values.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Base prompt template path. Defaults to the shared research template.",
    )
    parser.add_argument(
        "--mode-template",
        type=Path,
        help="Mode-specific prompt block path. Defaults to the block for the selected mode.",
    )
    parser.add_argument(
        "--allowed-packages",
        type=Path,
        help=(
            "JSON file with allowed package lists for prompt modes. Defaults to "
            "assets/prompts/research_hypotheses/allowed_packages.json."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Output path. Defaults to "
            "/tmp/prompt-<mode>-<COMPETITION_OR_PROJECT>.md."
        ),
    )
    parser.add_argument(
        "--hypothesis-count",
        type=positive_int,
        default=10,
        help="Maximum number of hypotheses the rendered prompt should request.",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def load_values(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"values file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"values file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"values file must contain a JSON object: {path}")
    values = raw.get("placeholders", raw)
    if not isinstance(values, dict):
        raise ValueError(
            f"values file 'placeholders' field must be a JSON object: {path}"
        )
    return {str(key): _stringify_value(value) for key, value in values.items()}


def load_allowed_package_values(path: Path, mode: str) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"allowed packages file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"allowed packages file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"allowed packages file must contain a JSON object: {path}")

    mode_config = raw.get(mode)
    if mode_config is None:
        return {}
    if not isinstance(mode_config, dict):
        raise ValueError(f"allowed packages entry for mode {mode} must be an object")
    packages = mode_config.get("allowed_packages")
    if not isinstance(packages, list) or not all(
        isinstance(package, str) and package.strip() for package in packages
    ):
        raise ValueError(
            f"allowed packages entry for mode {mode} must contain "
            "allowed_packages as a non-empty string array"
        )
    return {
        "ALLOWED_PACKAGES": ", ".join(f"`{package.strip()}`" for package in packages)
    }


def render_prompt(template_text: str, values: dict[str, str]) -> str:
    required = sorted(set(PLACEHOLDER_RE.findall(template_text)))
    missing = [key for key in required if key not in values or values[key] == ""]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"missing placeholder values: {missing_list}")

    def replace(match: re.Match[str]) -> str:
        return values[match.group(1)]

    return PLACEHOLDER_RE.sub(replace, template_text)


def compose_template(base_text: str, mode_text: str) -> str:
    if MODE_SPECIFIC_PLACEHOLDER not in base_text:
        return base_text
    return base_text.replace(MODE_SPECIFIC_PLACEHOLDER, mode_text.strip())


def default_values_path(task: str) -> Path:
    return repo_root() / "research_hypotheses" / task / "prompt_values.json"


def default_template_path(mode: str) -> Path:
    return repo_root() / BASE_TEMPLATE_PATH


def default_mode_template_path(mode: str) -> Path:
    return repo_root() / MODE_TEMPLATE_BY_MODE[mode]


def default_allowed_packages_path() -> Path:
    return repo_root() / ALLOWED_PACKAGES_PATH


def default_output_path(mode: str, values: dict[str, str]) -> Path:
    name = values.get("COMPETITION_OR_PROJECT") or values.get("TASK_NAME") or "task"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "task"
    return Path("/tmp") / f"prompt-{mode}-{slug}.md"


def write_prompt(
    *,
    mode: str,
    values_path: Path,
    template_path: Path,
    mode_template_path: Path | None = None,
    allowed_packages_path: Path | None = None,
    value_overrides: dict[str, Any] | None = None,
    out_path: Path | None = None,
) -> Path:
    values = load_values(values_path)
    values.update(
        load_allowed_package_values(
            allowed_packages_path or default_allowed_packages_path(),
            mode,
        )
    )
    if value_overrides:
        values.update(
            {str(key): _stringify_value(value) for key, value in value_overrides.items()}
        )
    base_text = template_path.read_text()
    mode_text = (
        mode_template_path or default_mode_template_path(mode)
    ).read_text()
    template_text = compose_template(base_text, mode_text)
    rendered = render_prompt(template_text, values)
    output_path = out_path or default_output_path(mode, values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    return output_path


def _stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "unknown"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    values_path = args.values or default_values_path(args.task)
    template_path = args.template or default_template_path(args.mode)
    mode_template_path = args.mode_template or default_mode_template_path(args.mode)
    allowed_packages_path = args.allowed_packages or default_allowed_packages_path()
    try:
        output_path = write_prompt(
            mode=args.mode,
            values_path=values_path,
            template_path=template_path,
            mode_template_path=mode_template_path,
            allowed_packages_path=allowed_packages_path,
            value_overrides={"HYPOTHESIS_COUNT": args.hypothesis_count},
            out_path=args.out,
        )
    except Exception as exc:
        print(f"Prompt render failed: {exc}", file=sys.stderr)
        return 1
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
