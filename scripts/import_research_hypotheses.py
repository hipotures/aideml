from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_HYPOTHESIS_FIELDS = (
    "title",
    "summary",
    "rationale",
    "implementation_hint",
    "expected_effect",
    "risk",
    "sources",
)
HYPOTHESIS_FILENAME_RE = re.compile(r"^hypothesis-(\d{6})\.json$")
MARKDOWN_URL_RE = re.compile(r"^\[[^\]]+\]\((https?://[^)]+)\)$")
AGENT_MODES_BY_IMPORT_MODE = {
    "autogluon": ["legacy", "autogluon"],
    "legacy": ["legacy"],
}


@dataclass(frozen=True)
class ImportResult:
    task: str
    mode: str
    target_dir: Path
    created_paths: tuple[Path, ...]
    dry_run: bool = False

    @property
    def created_count(self) -> int:
        return len(self.created_paths)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def import_research_hypotheses(
    input_path: Path,
    *,
    task: str,
    mode: str,
    repo_root: Path = repo_root(),
    enabled: bool = True,
    dry_run: bool = False,
) -> ImportResult:
    if mode not in AGENT_MODES_BY_IMPORT_MODE:
        choices = ", ".join(sorted(AGENT_MODES_BY_IMPORT_MODE))
        raise ValueError(f"unsupported import mode: {mode}. Expected one of: {choices}")

    hypotheses = _load_hypotheses(input_path)
    target_dir = Path(repo_root) / "research_hypotheses" / task
    next_id = _next_hypothesis_number(target_dir)
    created_paths: list[Path] = []

    for offset, raw_hypothesis in enumerate(hypotheses):
        hypothesis_id = next_id + offset
        hypothesis_id_text = f"{hypothesis_id:06d}"
        output_path = (
            target_dir
            / hypothesis_id_text
            / f"hypothesis-{hypothesis_id_text}.json"
        )
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing hypothesis: {output_path}")
        payload = _normalize_hypothesis(
            raw_hypothesis,
            enabled=enabled,
            agent_modes=AGENT_MODES_BY_IMPORT_MODE[mode],
        )
        created_paths.append(output_path)
        if dry_run:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return ImportResult(
        task=task,
        mode=mode,
        target_dir=target_dir,
        created_paths=tuple(created_paths),
        dry_run=dry_run,
    )


def _load_hypotheses(input_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"input file does not exist: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"input file is not valid JSON: {input_path}: {exc}") from exc

    if isinstance(payload, dict):
        hypotheses = payload.get("hypotheses")
    else:
        hypotheses = payload
    if not isinstance(hypotheses, list):
        raise ValueError("input JSON must be an object with a hypotheses array or a list")
    if not hypotheses:
        raise ValueError("input JSON does not contain any hypotheses")
    if not all(isinstance(hypothesis, dict) for hypothesis in hypotheses):
        raise ValueError("each imported hypothesis must be a JSON object")
    return hypotheses


def _next_hypothesis_number(target_dir: Path) -> int:
    max_id = 0
    if not target_dir.exists():
        return 1
    for path in target_dir.glob("*/hypothesis-*.json"):
        if not path.parent.name.isdigit():
            continue
        match = HYPOTHESIS_FILENAME_RE.match(path.name)
        if match is None or match.group(1) != path.parent.name:
            continue
        max_id = max(max_id, int(match.group(1)))
    for path in (target_dir / "hypotheses").glob("hypothesis-*.json"):
        match = HYPOTHESIS_FILENAME_RE.match(path.name)
        if match is None:
            continue
        max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def _normalize_hypothesis(
    raw: dict[str, Any],
    *,
    enabled: bool,
    agent_modes: list[str],
) -> dict[str, Any]:
    missing = []
    for field in REQUIRED_HYPOTHESIS_FIELDS:
        if field == "sources":
            if field not in raw or not isinstance(raw.get(field), list):
                missing.append(field)
            continue
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            missing.append(field)
    if missing:
        raise ValueError(
            "imported hypothesis missing required field(s): " + ", ".join(missing)
        )

    sources = []
    for source in raw["sources"]:
        if not isinstance(source, str):
            raise ValueError("hypothesis sources must be strings")
        normalized_source = _normalize_source(source)
        if normalized_source:
            sources.append(normalized_source)

    return {
        "enabled": enabled,
        "agent_modes": list(agent_modes),
        "title": raw["title"].strip(),
        "summary": raw["summary"].strip(),
        "rationale": raw["rationale"].strip(),
        "implementation_hint": raw["implementation_hint"].strip(),
        "expected_effect": raw["expected_effect"].strip(),
        "risk": raw["risk"].strip(),
        "sources": sources,
    }


def _normalize_source(source: str) -> str:
    source = source.strip()
    if not source:
        return ""
    match = MARKDOWN_URL_RE.match(source)
    if match is not None:
        return match.group(1).strip()
    return source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import GPT-generated research hypotheses into an AIDE task library.",
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument(
        "--task",
        default="playground-series-s6e6",
        help="Task slug under research_hypotheses/<task>.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(AGENT_MODES_BY_IMPORT_MODE),
        required=True,
        help=(
            "Import compatibility. autogluon creates hypotheses compatible with "
            "both legacy and autogluon; legacy creates legacy-only hypotheses."
        ),
    )
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="Import hypotheses with enabled=false.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be created without writing them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = import_research_hypotheses(
            args.input_path,
            task=args.task,
            mode=args.mode,
            enabled=not args.disabled,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1

    action = "Would create" if result.dry_run else "Created"
    print(
        f"{action} {result.created_count} hypotheses in "
        f"{result.target_dir.relative_to(repo_root())}"
    )
    for path in result.created_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
