#!/usr/bin/env python3
"""Create a legacy AIDE run from the best unique public-score solutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from aide.autogluon_preprocess import extract_preprocess_source
from aide.journal import Journal
from aide.legacy_import_template import TEMPLATE_NAME, build_legacy_stacking_template
from aide.utils import serialize
from aide.utils.artifact_manifest import RESULT_MANIFEST_NAME, SEEDED_BASE_PLAN_PREFIX
from aide.utils.config import _load_cfg, prep_agent_workspace, prep_cfg, save_run
from aide.utils.path_portability import sanitize_persisted_payload, to_portable_path
from aide.utils.seed_artifact import (
    SeedArtifactSource,
    SeededArtifactNode,
    _rewrite_manifest,
    seed_artifact_source_from_manifest,
    seed_journal_from_artifacts,
    source_is_autogluon,
)
from scripts import kaggle_submission_lab as lab
from scripts import smart_kaggle_submit as smart
from scripts.seed_run_from_nodes import _validate_run_id


@dataclass(frozen=True)
class RankedImportCandidate:
    public_rank: int
    public_score: float
    registry_row: dict[str, Any]
    source: SeedArtifactSource
    source_solution_path: Path
    canonical_code_sha256: str
    preprocess_source: str | None
    source_code: str

    @property
    def is_autogluon(self) -> bool:
        return self.preprocess_source is not None


@dataclass(frozen=True)
class SubmissionLabSeedResult:
    run_id: str
    log_dir: Path
    workspace_dir: Path
    candidates: tuple[RankedImportCandidate, ...]
    seeded: tuple[SeededArtifactNode, ...]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _public_score(row: dict[str, Any]) -> float | None:
    try:
        score = float(row.get("public_score"))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _public_sort_score(row: dict[str, Any]) -> float:
    score = _public_score(row)
    return float("-inf") if score is None else score


def _source_solution_path(row: dict[str, Any]) -> Path:
    source_path = row.get("source_solution_path")
    if source_path:
        return lab._record_path(source_path)
    artifact_dir = lab._record_path(row.get("artifact_dir"))
    return artifact_dir / "solution.py"


def _candidate_from_row(
    row: dict[str, Any],
    *,
    public_rank: int,
) -> RankedImportCandidate:
    score = _public_score(row)
    if score is None:
        raise ValueError(f"Public rank {public_rank} has no numeric public score.")

    solution_path = _source_solution_path(row)
    if not solution_path.exists():
        raise FileNotFoundError(
            f"Public rank {public_rank} has no local source solution: {solution_path}"
        )
    manifest_path = solution_path.parent / RESULT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Public rank {public_rank} has no source artifact manifest: {manifest_path}"
        )

    source = seed_artifact_source_from_manifest(manifest_path, matched_kind="solution")
    source_code = solution_path.read_text(encoding="utf-8")
    preprocess_source = None
    canonical_code = source_code
    if source_is_autogluon(source):
        preprocess_source = extract_preprocess_source(source_code)
        canonical_code = preprocess_source

    return RankedImportCandidate(
        public_rank=public_rank,
        public_score=score,
        registry_row=dict(row),
        source=source,
        source_solution_path=solution_path,
        canonical_code_sha256=_sha256_text(canonical_code),
        preprocess_source=preprocess_source,
        source_code=source_code,
    )


def select_unique_public_candidates(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[RankedImportCandidate]:
    if limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    selected: list[RankedImportCandidate] = []
    seen_code: set[str] = set()
    ranked_rows = sorted(
        rows,
        key=_public_sort_score,
        reverse=True,
    )
    for public_rank, row in enumerate(ranked_rows, start=1):
        if _public_score(row) is None:
            continue
        candidate = _candidate_from_row(row, public_rank=public_rank)
        if candidate.canonical_code_sha256 in seen_code:
            continue
        seen_code.add(candidate.canonical_code_sha256)
        selected.append(candidate)
        if len(selected) == limit:
            return selected

    raise ValueError(
        f"Requested {limit} unique public-score codes, but only {len(selected)} "
        "were available in the ranked input. Increase --scan-limit or provide a larger "
        "--ranking-json payload."
    )


def load_ranked_rows(
    *,
    ranking_json: Path | None,
    index_path: Path,
    registry_path: Path,
    competition: str,
    scan_limit: int,
) -> list[dict[str, Any]]:
    if scan_limit <= 0:
        raise ValueError("--scan-limit must be greater than zero.")
    if ranking_json is not None:
        payload = json.loads(ranking_json.read_text(encoding="utf-8"))
        rows = payload.get("registry")
        if not isinstance(rows, list):
            raise ValueError("--ranking-json must contain the submission lab `registry` list.")
        return [dict(row) for row in rows[:scan_limit] if isinstance(row, dict)]

    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = list(index.get("records") or [])
    registry = smart.SubmissionRegistry.load(registry_path)
    return lab.registry_display_rows(
        registry,
        remote_submissions=None,
        records=records,
        limit=scan_limit,
        competition=competition,
    )


def _import_plan(candidate: RankedImportCandidate) -> str:
    row = candidate.registry_row
    transform = (
        f"AutoGluon preprocess(df) imported into legacy template {TEMPLATE_NAME}"
        if candidate.is_autogluon
        else "legacy solution imported unchanged"
    )
    source_plan = str(candidate.source.node_payload.get("plan") or "").strip()
    plan = (
        f"{SEEDED_BASE_PLAN_PREFIX}: public_rank={candidate.public_rank} "
        f"public_score={candidate.public_score:.5f} "
        f"source_run={candidate.source.run_id} "
        f"source_step={candidate.source.source_step if candidate.source.source_step is not None else '?'} "
        f"source_timestamp={candidate.source.timestamp} "
        f"submission_sha256={row.get('sha256') or '?'} "
        f"canonical_code_sha256={candidate.canonical_code_sha256}; {transform}."
    )
    if source_plan:
        plan += f" Original plan: {source_plan}"
    return plan


def _imported_code(candidate: RankedImportCandidate, *, cfg: Any) -> str:
    if candidate.preprocess_source is None:
        return candidate.source_code
    return build_legacy_stacking_template(candidate.preprocess_source, cfg)


def _write_import_manifest(
    result: SubmissionLabSeedResult,
) -> None:
    roots = []
    for candidate, seeded in zip(result.candidates, result.seeded, strict=True):
        row = candidate.registry_row
        roots.append(
            {
                "root_step": seeded.node.step,
                "root_node_id": seeded.node.id,
                "public_rank": candidate.public_rank,
                "public_score": candidate.public_score,
                "source_kind": "autogluon" if candidate.is_autogluon else "legacy",
                "source_run": candidate.source.run_id,
                "source_step": candidate.source.source_step,
                "source_timestamp": candidate.source.timestamp,
                "source_solution_path": to_portable_path(candidate.source_solution_path),
                "source_submission_sha256": row.get("sha256"),
                "canonical_code_sha256": candidate.canonical_code_sha256,
                "legacy_import_template": TEMPLATE_NAME if candidate.is_autogluon else None,
            }
        )
    payload = {
        "schema_version": 2,
        "selection": "submission_lab_public_desc_unique_canonical_code",
        "agent_mode": "legacy",
        "legacy_import_template": TEMPLATE_NAME,
        "roots": roots,
    }
    (result.log_dir / "imported_roots.json").write_text(
        json.dumps(sanitize_persisted_payload(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def seed_run_from_submission_lab(
    *,
    rows: list[dict[str, Any]],
    cli_overrides: list[str],
    limit: int = 10,
    run_id: str | None = None,
    prepare_workspace: bool = True,
) -> SubmissionLabSeedResult:
    _validate_run_id(run_id)
    if any(arg == "--resume" or arg.startswith("--resume=") for arg in cli_overrides):
        raise ValueError("Do not pass --resume; resume the created run afterwards.")

    candidates = select_unique_public_candidates(rows, limit=limit)
    cfg = prep_cfg(_load_cfg(cli_args=cli_overrides))

    logs_dir = Path(cfg.log_dir).parent
    workspaces_dir = Path(cfg.workspace_dir).parent
    if run_id is not None:
        cfg.exp_name = run_id
        cfg.log_dir = logs_dir / run_id
        cfg.workspace_dir = workspaces_dir / run_id

    cfg.agent.mode = "legacy"
    cfg.agent.gpu = True
    cfg.agent.k_fold_validation = 5
    cfg.agent.search.num_drafts = len(candidates)
    cfg.agent.legacy_starter.autogluon_profile = None

    log_dir = Path(cfg.log_dir)
    workspace_dir = Path(cfg.workspace_dir)
    if log_dir.exists():
        raise FileExistsError(f"Run log directory already exists: {log_dir}")
    if workspace_dir.exists():
        raise FileExistsError(f"Run workspace directory already exists: {workspace_dir}")

    if prepare_workspace:
        prep_agent_workspace(cfg)
    else:
        (workspace_dir / "input").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    imported_codes = [
        _imported_code(candidate, cfg=cfg) for candidate in candidates
    ]
    plans = [_import_plan(candidate) for candidate in candidates]
    journal, seeded = seed_journal_from_artifacts(
        cfg,
        [candidate.source for candidate in candidates],
        code_only=True,
        code_overrides=imported_codes,
        plan_overrides=plans,
    )
    save_run(cfg, journal)

    result = SubmissionLabSeedResult(
        run_id=str(cfg.exp_name),
        log_dir=log_dir,
        workspace_dir=workspace_dir,
        candidates=tuple(candidates),
        seeded=tuple(seeded),
    )
    _write_import_manifest(result)
    return result


def rewrite_generated_submission_lab_run(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    logs_dir: Path,
    limit: int = 10,
) -> SubmissionLabSeedResult:
    """Replace only generated roots in an existing imported run."""
    _validate_run_id(run_id)
    log_dir = (logs_dir / run_id).resolve()
    config_path = log_dir / "config.yaml"
    journal_path = log_dir / "journal.json"
    if not config_path.exists() or not journal_path.exists():
        raise FileNotFoundError(f"Existing run is incomplete: {log_dir}")

    cfg = OmegaConf.load(config_path)
    cfg.log_dir = log_dir
    workspace_dir = Path(cfg.workspace_dir)
    if not workspace_dir.is_absolute():
        workspace_dir = (Path.cwd() / workspace_dir).resolve()
    cfg.workspace_dir = workspace_dir
    if cfg.agent.mode != "legacy":
        raise ValueError(f"Refusing to rewrite non-legacy run: {run_id}")

    journal = serialize.load_json(journal_path, Journal)
    candidates = select_unique_public_candidates(rows, limit=limit)
    if len(journal.nodes) != len(candidates) or any(
        node.status != "generated" or node.parent is not None for node in journal.nodes
    ):
        raise ValueError(
            "Existing run may be rewritten only while all imported roots are unexecuted."
        )

    previous_manifest = json.loads(
        (log_dir / "imported_roots.json").read_text(encoding="utf-8")
    )
    previous_roots = list(previous_manifest.get("roots") or [])
    previous_identity = [
        (root.get("public_rank"), root.get("canonical_code_sha256"))
        for root in previous_roots
    ]
    current_identity = [
        (candidate.public_rank, candidate.canonical_code_sha256)
        for candidate in candidates
    ]
    if previous_identity != current_identity:
        raise ValueError("Ranked source identities differ from the existing imported run.")

    seeded: list[SeededArtifactNode] = []
    for node, candidate in zip(journal.nodes, candidates, strict=True):
        node.code = _imported_code(candidate, cfg=cfg)
        node.plan = _import_plan(candidate)
        node.run_stats = {
            "seeded_from_manifest": True,
            "code_only": True,
            "legacy_import_template": TEMPLATE_NAME,
        }
        artifact_dir = log_dir / "artifacts" / str(node.artifact_dir_name)
        (artifact_dir / "solution.py").write_text(node.code, encoding="utf-8")
        _rewrite_manifest(
            cfg=cfg,
            node=node,
            source=candidate.source,
            artifact_dir=artifact_dir,
            code_only=True,
        )
        seeded.append(
            SeededArtifactNode(
                source=candidate.source,
                node=node,
                artifact_dir=artifact_dir,
            )
        )

    save_run(cfg, journal)
    result = SubmissionLabSeedResult(
        run_id=run_id,
        log_dir=log_dir,
        workspace_dir=workspace_dir,
        candidates=tuple(candidates),
        seeded=tuple(seeded),
    )
    _write_import_manifest(result)
    return result


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Create a standard resumable legacy run from the top unique public-score "
            "codes in Kaggle submission lab. AutoGluon roots keep preprocess(df) and "
            "receive a fixed legacy-compatible AutoGluon runner."
        )
    )
    parser.add_argument("--ranking-json", type=Path)
    parser.add_argument("--index", type=Path, default=lab.DEFAULT_INDEX_PATH)
    parser.add_argument("--registry", type=Path, default=lab.DEFAULT_REGISTRY)
    parser.add_argument("--competition", default=lab.default_competition())
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--scan-limit", type=int, default=20)
    run_target = parser.add_mutually_exclusive_group()
    run_target.add_argument("--run-id")
    run_target.add_argument(
        "--rewrite-run",
        help="Rewrite an unexecuted imported run after template changes.",
    )
    parser.add_argument(
        "--no-prepare-workspace",
        action="store_true",
        help="Create empty input/working directories instead of preparing task data.",
    )
    return parser.parse_known_args(argv)


def _format_candidate(candidate: RankedImportCandidate, seeded: SeededArtifactNode) -> str:
    kind = "AG features" if candidate.is_autogluon else "legacy"
    return (
        f"  step {seeded.node.step}: public_rank={candidate.public_rank} "
        f"public={candidate.public_score:.5f} kind={kind} "
        f"source={candidate.source.run_id}:{candidate.source.source_step} "
        f"code={candidate.canonical_code_sha256[:12]}"
    )


def main(argv: list[str] | None = None) -> int:
    args, cli_overrides = parse_args(argv)
    rows = load_ranked_rows(
        ranking_json=args.ranking_json,
        index_path=args.index,
        registry_path=args.registry,
        competition=args.competition,
        scan_limit=args.scan_limit,
    )
    if args.rewrite_run:
        if cli_overrides:
            raise ValueError("AIDE config overrides are not accepted with --rewrite-run.")
        result = rewrite_generated_submission_lab_run(
            run_id=args.rewrite_run,
            rows=rows,
            logs_dir=args.index.parent,
            limit=args.limit,
        )
        print(f"Rewritten run: {result.run_id}")
    else:
        result = seed_run_from_submission_lab(
            rows=rows,
            cli_overrides=cli_overrides,
            limit=args.limit,
            run_id=args.run_id,
            prepare_workspace=not args.no_prepare_workspace,
        )
        print(f"Created run: {result.run_id}")
    print(f"Log dir: {result.log_dir}")
    print(f"Workspace dir: {result.workspace_dir}")
    print("Imported roots:")
    for candidate, seeded in zip(result.candidates, result.seeded, strict=True):
        print(_format_candidate(candidate, seeded))
    print(f"Resume with: uv run aide --resume {result.run_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Submission lab seed failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
