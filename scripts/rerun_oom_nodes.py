#!/usr/bin/env python3
"""Rerun journal nodes that failed because CatBoost GPU ran out of memory."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from aide.autogluon_preprocess import parse_result_marker
from aide.interpreter import ExecutionResult, Interpreter
from aide.journal import Journal, Node
from aide.run import enforce_submission_contract
from aide.utils import serialize
from aide.utils.config import Config, save_run
from aide.utils.metric import MetricValue, WorstMetricValue


RERUN_PLAN_PREFIX = "OOM recovery rerun"
LEDGER_RELATIVE_PATH = Path("reruns") / "oom_reruns.json"
OOM_TEXT_MARKERS = (
    "CatBoost GPU ran out of memory",
    "CUDA error 2: out of memory",
)


def load_saved_cfg(log_dir: Path) -> Config:
    cfg_path = log_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    return OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(cfg_path))  # type: ignore[return-value]


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Ledger must be a JSON list: {path}")
    return [record for record in data if isinstance(record, dict)]


def write_ledger(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def attempted_source_ids(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record["source_node_id"])
        for record in records
        if isinstance(record.get("source_node_id"), str)
    }


def is_recovery_rerun(node: Node) -> bool:
    return str(node.plan or "").startswith(RERUN_PLAN_PREFIX)


def is_oom_node(node: Node) -> bool:
    if is_recovery_rerun(node):
        return False
    text = f"{node.analysis or ''}\n{''.join(node._term_out or [])}"
    return any(marker in text for marker in OOM_TEXT_MARKERS)


def select_oom_nodes(
    journal: Journal,
    *,
    records: list[dict[str, Any]],
    steps: set[int] | None = None,
    include_attempted: bool = False,
) -> list[Node]:
    attempted = attempted_source_ids(records)
    selected: list[Node] = []
    for node in journal.nodes:
        if steps is not None and node.step not in steps:
            continue
        if not is_oom_node(node):
            continue
        if not include_attempted and node.id in attempted:
            continue
        selected.append(node)
    return selected


def _artifact_timestamp(ctime: float) -> str:
    return dt.datetime.fromtimestamp(ctime).strftime("%Y%m%dT%H%M%S")


def unique_node_ctime(log_dir: Path, *, start: float | None = None) -> float:
    ctime = start or time.time()
    while (log_dir / "artifacts" / _artifact_timestamp(ctime)).exists():
        ctime += 1.0
    return ctime


def node_artifact_dir(log_dir: Path, node: Node) -> Path:
    return log_dir / "artifacts" / _artifact_timestamp(node.ctime)


def make_rerun_node(source: Node, *, log_dir: Path) -> Node:
    ctime = unique_node_ctime(log_dir)
    source_plan = str(source.plan or "").strip()
    plan = f"{RERUN_PLAN_PREFIX} of step {source.step}"
    if source_plan:
        plan = f"{plan}\n\nOriginal plan:\n{source_plan}"
    return Node(
        code=source.code,
        plan=plan,
        parent=source.parent,
        ctime=ctime,
    )


def apply_execution_result_without_llm(node: Node, exec_result: ExecutionResult) -> None:
    node.absorb_exec_result(exec_result)
    marker_response = parse_result_marker(node.term_out)
    if marker_response is not None:
        metric = marker_response.get("metric")
        if not isinstance(metric, (float, int)) or isinstance(metric, bool):
            metric = None
        node.analysis = str(marker_response.get("summary", ""))
        node.is_buggy = (
            bool(marker_response.get("is_bug"))
            or node.exc_type is not None
            or metric is None
        )
        if node.is_buggy:
            node.metric = WorstMetricValue()
            node.status = "bug"
        else:
            node.metric = MetricValue(
                metric,
                maximize=not bool(marker_response.get("lower_is_better")),
            )
            node.status = "ok"
        return

    output = "".join(exec_result.term_out or []).strip()
    node.is_buggy = True
    node.metric = WorstMetricValue()
    node.status = "bug"
    node.analysis = (
        "Rerun did not produce AIDE_RESULT_JSON marker."
        if not output
        else f"Rerun did not produce AIDE_RESULT_JSON marker.\n\n{output}"
    )


def mark_runtime_crash(node: Node, exc: RuntimeError, *, artifact_dir: Path) -> None:
    message = str(exc) or exc.__class__.__name__
    for log_name in ("autogluon_stdout.log", "process_stdout.log"):
        log_path = artifact_dir / log_name
        if not log_path.exists():
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for marker in OOM_TEXT_MARKERS:
            if marker in log_text:
                message = f"{message}\n\nEvidence from {log_name}: {marker}"
                break
    node._term_out = [f"{exc.__class__.__name__}: {message}\n"]
    node.exec_time = 0.0
    node.exc_type = exc.__class__.__name__
    node.exc_info = {"args": [message]}
    node.exc_stack = None
    node.analysis = message
    node.metric = WorstMetricValue()
    node.is_buggy = True
    node.status = "failed"


def record_for_rerun(
    *,
    source: Node,
    rerun: Node,
    artifact_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    metric = None
    if isinstance(rerun.metric, MetricValue) and not rerun.metric.is_worst:
        metric = rerun.metric.value
    return {
        "source_node_id": source.id,
        "source_step": source.step,
        "rerun_node_id": rerun.id,
        "rerun_step": rerun.step,
        "status": rerun.status or ("bug" if rerun.is_buggy else "ok"),
        "metric": metric,
        "artifact_dir": str(artifact_dir.relative_to(log_dir)),
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def backup_journal(log_dir: Path) -> Path:
    src = log_dir / "journal.json"
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = log_dir / f"journal.json.bak-{timestamp}"
    shutil.copy2(src, dst)
    return dst


def rerun_node(
    *,
    cfg: Config,
    journal: Journal,
    source: Node,
    interpreter: Interpreter,
) -> tuple[Node, Path]:
    rerun = make_rerun_node(source, log_dir=Path(cfg.log_dir))
    artifact_dir = node_artifact_dir(Path(cfg.log_dir), rerun)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "solution.py").write_text(rerun.code, encoding="utf-8")
    (artifact_dir / "source_node.json").write_text(
        json.dumps(
            {
                "source_node_id": source.id,
                "source_step": source.step,
                "source_status": source.status,
                "source_metric": getattr(source.metric, "value", None),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    previous_artifact_dir = os.environ.get("AIDE_NODE_ARTIFACT_DIR")
    os.environ["AIDE_NODE_ARTIFACT_DIR"] = str(artifact_dir)
    try:
        try:
            exec_result = interpreter.run(rerun.code, reset_session=True)
        except RuntimeError as exc:
            mark_runtime_crash(rerun, exc, artifact_dir=artifact_dir)
        else:
            apply_execution_result_without_llm(rerun, exec_result)
            enforce_submission_contract(cfg, rerun)
    finally:
        if previous_artifact_dir is None:
            os.environ.pop("AIDE_NODE_ARTIFACT_DIR", None)
        else:
            os.environ["AIDE_NODE_ARTIFACT_DIR"] = previous_artifact_dir

    journal.append(rerun)
    return rerun, artifact_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append rerun nodes for journal entries that failed with CatBoost GPU OOM."
    )
    parser.add_argument("log_dir", type=Path, help="Run log directory.")
    parser.add_argument("--dry-run", action="store_true", help="Only list selected nodes.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum nodes to rerun.")
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        help="Only rerun this source step. Can be provided multiple times.",
    )
    parser.add_argument(
        "--include-attempted",
        action="store_true",
        help="Also rerun source nodes already present in the rerun ledger.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log_dir = args.log_dir.resolve()
    journal_path = log_dir / "journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"Missing journal: {journal_path}")

    cfg = load_saved_cfg(log_dir)
    journal = serialize.load_json(journal_path, Journal)
    ledger_path = log_dir / LEDGER_RELATIVE_PATH
    records = load_ledger(ledger_path)
    selected = select_oom_nodes(
        journal,
        records=records,
        steps=set(args.steps) if args.steps else None,
        include_attempted=args.include_attempted,
    )
    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]

    if args.dry_run:
        print(f"Selected {len(selected)} OOM node(s) for rerun.")
        for node in selected:
            print(f"step={node.step} id={node.id} status={node.status}")
        return 0

    if not selected:
        print("No OOM nodes selected for rerun.")
        return 0

    backup_path = backup_journal(log_dir)
    print(f"Backed up journal to {backup_path}")

    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),  # type: ignore[arg-type]
    )
    try:
        total = len(selected)
        for index, source in enumerate(selected, start=1):
            prefix = f"[{index}/{total}]"
            print(f"{prefix} Rerunning source step {source.step} ({source.id})")
            rerun, artifact_dir = rerun_node(
                cfg=cfg,
                journal=journal,
                source=source,
                interpreter=interpreter,
            )
            records.append(
                record_for_rerun(
                    source=source,
                    rerun=rerun,
                    artifact_dir=artifact_dir,
                    log_dir=log_dir,
                )
            )
            write_ledger(ledger_path, records)
            save_run(cfg, journal, current_node=rerun)
            status = rerun.status or ("bug" if rerun.is_buggy else "ok")
            metric = getattr(rerun.metric, "value", None)
            print(
                f"{prefix} Created rerun step {rerun.step}: "
                f"status={status} metric={metric}"
            )
    finally:
        interpreter.cleanup_session()

    print(f"Processed {len(selected)} OOM rerun(s). Ledger written to {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
