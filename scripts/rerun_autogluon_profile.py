from __future__ import annotations

import argparse
import datetime as dt
import difflib
import gzip
import json
import os
import resource
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from omegaconf import OmegaConf

from aide.autogluon_preprocess import (
    build_autogluon_wrapper,
    extract_preprocess_source,
    parse_result_marker,
    resolve_autogluon_settings,
    resolve_autogluon_included_model_types,
)
from aide.interpreter import ExecutionResult
from aide.utils.artifact_manifest import RESULT_MANIFEST_NAME
from aide.utils.config import _load_cfg
from aide.utils.path_portability import (
    resolve_portable_path,
    sanitize_persisted_payload,
    to_portable_path,
)
from aide.utils.submission_validation import validate_submission_file
from scripts import kaggle_submission_lab as lab


DEFAULT_PROFILE = "full_boost"

COMPETITION_AUTOGLUON_DEFAULTS: dict[str, dict[str, Any]] = {
    "playground-series-s6e6": {
        "eval_metric": "balanced_accuracy",
        "class_balance": "balanced",
    },
}
DEFAULT_PROCESS_TIMEOUT_MARGIN = 900


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_persisted_payload(payload), indent=2, sort_keys=True) + "\n"
    )


def _record_path(value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("/__aideml_missing_path__")
    return resolve_portable_path(text)


def _file_entry(path: Path, *, base_dir: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return {
        "path": path.relative_to(base_dir).as_posix(),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": lab.sha256_file(path),
    }


def _directory_file_entries(path: Path, *, base_dir: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        entry
        for child in sorted(path.glob("*.csv.gz"))
        if (entry := _file_entry(child, base_dir=base_dir)) is not None
    ]


def resolve_process_timeout(timeout: int | None, autogluon_time_limit: int) -> int:
    if timeout is not None:
        return int(timeout)
    return max(1200, int(autogluon_time_limit) + DEFAULT_PROCESS_TIMEOUT_MARGIN)


def _format_seconds(seconds: float) -> str:
    seconds_i = max(0, int(seconds))
    minutes, secs = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _copy_or_link_input(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if target.exists():
            continue
        os.symlink(child.resolve(), target, target_is_directory=child.is_dir())


def _copy_prediction_artifact_gz(source: Path, destination: Path) -> None:
    if source.suffix == ".gz":
        shutil.copy2(source, destination)
        return
    with source.open("rb") as src, gzip.open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst)


def _copy_prediction_dir(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.csv.gz"):
        shutil.copy2(source, destination_dir / source.name)


def source_input_dir(*, logs_dir: Path, run: str) -> Path:
    repo_root = Path(logs_dir).resolve().parent
    path = repo_root / "workspaces" / run / "input"
    if not path.exists():
        raise FileNotFoundError(f"Missing source input directory: {path}")
    return path


def available_autogluon_profiles() -> list[str]:
    cfg = _load_cfg(use_cli_args=False)
    return sorted(cfg.agent.autogluon.profiles)


def format_unknown_profile_error(profile: str, known_profiles: list[str]) -> str:
    lines = [f"Unknown AutoGluon profile: {profile}"]
    matches = difflib.get_close_matches(profile, known_profiles, n=1)
    if matches:
        lines.append(f"Did you mean: {matches[0]}")
    lines.append("Available profiles:")
    lines.extend(f"  - {known}" for known in known_profiles)
    return "\n".join(lines)


def validate_autogluon_profile(profile: str) -> None:
    known_profiles = available_autogluon_profiles()
    if profile not in known_profiles:
        raise ValueError(format_unknown_profile_error(profile, known_profiles))


def build_profile_config(
    *,
    source_record: dict[str, Any],
    profile: str,
    competition: str,
    presets: str | None,
    time_limit: int | None,
    fit_args: dict[str, Any] | None,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.agent.mode = "autogluon_preprocess"
    cfg.agent.autogluon.profile = profile
    cfg.agent.autogluon.included_model_types = None

    profiles = cfg.agent.autogluon.profiles
    if profile not in profiles:
        raise ValueError(format_unknown_profile_error(profile, sorted(profiles)))
    profile_settings = OmegaConf.to_container(profiles[profile], resolve=True)
    if isinstance(profile_settings, list):
        profile_settings = {"included_model_types": profile_settings}
    elif profile_settings is None:
        profile_settings = {}
    else:
        profile_settings = dict(profile_settings)

    for key, value in COMPETITION_AUTOGLUON_DEFAULTS.get(competition, {}).items():
        profile_settings.setdefault(key, value)

    if presets is not None:
        profile_settings["presets"] = presets
    if time_limit is not None:
        profile_settings["time_limit"] = int(time_limit)
    if fit_args is not None:
        profile_settings["fit_args"] = fit_args

    cfg.agent.autogluon.profiles[profile] = profile_settings
    return cfg


def build_profile_code(
    *,
    source_record: dict[str, Any],
    profile: str,
    competition: str,
    presets: str | None,
    time_limit: int | None,
    fit_args: dict[str, Any] | None,
) -> tuple[str, list[str], int, str]:
    solution_path = Path(source_record["solution_path"])
    preprocess_source = extract_preprocess_source(solution_path.read_text())
    cfg = build_profile_config(
        source_record=source_record,
        profile=profile,
        competition=competition,
        presets=presets,
        time_limit=time_limit,
        fit_args=fit_args,
    )
    code = build_autogluon_wrapper(preprocess_source, cfg)
    return (
        code,
        resolve_autogluon_included_model_types(cfg),
        int(resolve_autogluon_settings(cfg)["time_limit"]),
        str(resolve_autogluon_settings(cfg)["presets"]),
    )


def execute_code(
    code: str,
    *,
    workspace_dir: Path,
    artifact_dir: Path,
    timeout: int,
    memory_limit_gb: float | None,
    console: Console | None = None,
    progress_time_limit: int | None = None,
) -> ExecutionResult:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    runfile = workspace_dir / "solution.py"
    runfile.write_text(code)

    def limit_child_memory() -> None:
        if memory_limit_gb is None:
            return
        limit_bytes = int(float(memory_limit_gb) * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    env = os.environ.copy()
    env["AIDE_NODE_ARTIFACT_DIR"] = str(artifact_dir.resolve())
    start_time = time.time()

    proc = subprocess.Popen(
        [sys.executable, str(runfile.name)],
        cwd=str(workspace_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        preexec_fn=limit_child_memory,
    )
    try:
        if console is None or progress_time_limit is None:
            output_text, _ = proc.communicate(timeout=timeout)
        else:
            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.fields[elapsed_text]} / {task.fields[target_text]}"),
                console=console,
            )
            with progress:
                task_id = progress.add_task(
                    f"Running AutoGluon ({artifact_dir.name})",
                    total=max(1, int(progress_time_limit)),
                    elapsed_text="00:00",
                    target_text=_format_seconds(progress_time_limit),
                )
                while proc.poll() is None:
                    elapsed = time.time() - start_time
                    progress.update(
                        task_id,
                        completed=min(elapsed, float(progress_time_limit)),
                        elapsed_text=_format_seconds(elapsed),
                    )
                    if elapsed >= timeout:
                        raise subprocess.TimeoutExpired(proc.args, timeout)
                    time.sleep(1.0)
                output_text, _ = proc.communicate(timeout=5)
                progress.update(
                    task_id,
                    completed=min(time.time() - start_time, float(progress_time_limit)),
                    elapsed_text=_format_seconds(time.time() - start_time),
                )
        exec_time = time.time() - start_time
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            output_text, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output_text, _ = proc.communicate()
        exec_time = float(timeout)
        return ExecutionResult(
            [output_text, f"TimeoutError: Execution exceeded the time limit of {timeout} seconds"],
            exec_time,
            "TimeoutError",
            {},
            [],
        )

    output = [output_text]
    if proc.returncode == 0:
        output.append(
            f"Execution time: {exec_time:.0f} seconds (time limit is {timeout} seconds)."
        )
        return ExecutionResult(output, exec_time, None, None, None)

    output.append(
        f"Process exited with return code {proc.returncode} after {exec_time:.0f} seconds."
    )
    return ExecutionResult(output, exec_time, "RuntimeError", {"returncode": proc.returncode}, [])


def _error_text(exec_result: ExecutionResult, validation_error: str | None) -> str:
    sections = []
    if exec_result.exc_type:
        sections.append(f"Exception type:\n{exec_result.exc_type}")
    if exec_result.exc_info:
        sections.append("Exception info:\n" + json.dumps(exec_result.exc_info, indent=2))
    if exec_result.exc_stack:
        sections.append("Exception stack:\n" + json.dumps(exec_result.exc_stack, indent=2))
    if exec_result.term_out:
        sections.append("Terminal output:\n" + "".join(exec_result.term_out).rstrip())
    if validation_error:
        sections.append(f"Submission validation:\n{validation_error}")
    return "\n\n".join(sections).strip() or "Unknown error"


def _find_sample_submission(input_dir: Path) -> Path | None:
    for name in ("sample_submission.csv.gz", "sample_submission.csv"):
        path = input_dir / name
        if path.exists():
            return path
    return None


def run_profile_eval(
    source_record: dict[str, Any],
    *,
    logs_dir: Path,
    profile: str = DEFAULT_PROFILE,
    presets: str | None = None,
    time_limit: int | None = None,
    fit_args: dict[str, Any] | None = None,
    competition: str = lab.DEFAULT_COMPETITION,
    timeout: int | None = None,
    memory_limit_gb: float | None = 80.0,
    console: Console | None = None,
) -> dict[str, Any]:
    if source_record.get("kind") not in {"source_node", "profile_eval"}:
        raise ValueError("Source record must be a source_node or profile_eval.")
    if not source_record.get("sha256"):
        raise ValueError("Source record has no sha256.")

    run = str(source_record["run"])
    run_dir = Path(logs_dir) / run
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = timestamp_now()
        artifact_dir = artifacts_dir / timestamp
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            time.sleep(1.0)

    code, included_model_types, resolved_time_limit, autogluon_presets = build_profile_code(
        source_record=source_record,
        profile=profile,
        competition=competition,
        presets=presets,
        time_limit=time_limit,
        fit_args=fit_args,
    )
    effective_timeout = resolve_process_timeout(timeout, resolved_time_limit)
    (artifact_dir / "solution.py").write_text(code)
    if console is not None:
        console.print(f"Artifacts: {artifact_dir}")
        console.print(f"Log: {artifact_dir / 'autogluon_stdout.log'}")
        console.print(f"Submission: {artifact_dir / 'submission.csv'}")

    input_dir = source_input_dir(logs_dir=logs_dir, run=run)
    with tempfile.TemporaryDirectory(prefix=f"aide-profile-eval-{run}-") as tmp:
        workspace_dir = Path(tmp)
        _copy_or_link_input(input_dir, workspace_dir / "input")
        exec_result = execute_code(
            code,
            workspace_dir=workspace_dir,
            artifact_dir=artifact_dir,
            timeout=effective_timeout,
            memory_limit_gb=memory_limit_gb,
            **(
                {"console": console, "progress_time_limit": resolved_time_limit}
                if console is not None
                else {}
            ),
        )
        generated_submission = workspace_dir / "working" / "submission.csv"
        if generated_submission.exists():
            shutil.copy2(generated_submission, artifact_dir / "submission.csv")
        for name in (
            "oof_predictions.csv",
            "test_predictions.csv",
            "validation_predictions.csv",
        ):
            gzip_name = f"{name}.gz"
            for generated_prediction in (
                workspace_dir / "working" / gzip_name,
                workspace_dir / "working" / name,
            ):
                if generated_prediction.exists():
                    _copy_prediction_artifact_gz(generated_prediction, artifact_dir / gzip_name)
                    break
        _copy_prediction_dir(
            workspace_dir / "working" / "model_predictions",
            artifact_dir / "model_predictions",
        )
        generated_log = workspace_dir / "working" / "autogluon_stdout.log"
        if generated_log.exists() and not (artifact_dir / "autogluon_stdout.log").exists():
            shutil.copy2(generated_log, artifact_dir / "autogluon_stdout.log")

    marker = parse_result_marker("".join(exec_result.term_out))
    metric = marker.get("metric") if marker else None
    eval_metric = marker.get("eval_metric") if marker else None
    lower_is_better = bool(marker.get("lower_is_better")) if marker else False
    validation_error = None
    sample_path = _find_sample_submission(input_dir)
    submission_path = artifact_dir / "submission.csv"
    if sample_path is not None and submission_path.exists():
        validation_error = validate_submission_file(submission_path, sample_path)
    elif sample_path is not None:
        validation_error = "missing submission.csv while sample_submission exists"

    is_ok = (
        exec_result.exc_type is None
        and metric is not None
        and validation_error is None
        and submission_path.exists()
    )
    if not is_ok:
        (artifact_dir / "error.txt").write_text(
            _error_text(exec_result, validation_error) + "\n"
        )

    source_sha = source_record.get("sha256")
    metadata = {
        "kind": "profile_eval",
        "competition": competition,
        "status": "ok" if is_ok else "error",
        "profile": profile,
        "autogluon_presets": autogluon_presets,
        "included_model_types": included_model_types,
        "time_limit": resolved_time_limit,
        "process_timeout": effective_timeout,
        "local_score": float(metric) if metric is not None else None,
        "eval_metric": eval_metric,
        "metric_maximize": not lower_is_better,
        "exec_time": exec_result.exec_time,
        "source_run": source_record.get("run"),
        "source_node_id": source_record.get("node_id")
        or source_record.get("source_node_id"),
        "source_step": source_record.get("step") or source_record.get("source_step"),
        "source_timestamp": source_record.get("timestamp"),
        "source_sha256": source_sha,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_json(artifact_dir / "submission_eval.json", metadata)

    manifest = {
        "schema_version": 1,
        **metadata,
        "run": run,
        "timestamp": timestamp,
        "artifact_dir": to_portable_path(artifact_dir),
        "local_score": float(metric) if metric is not None else None,
        "metric_maximize": not lower_is_better,
        "is_buggy": not is_ok,
        "sha256": lab.sha256_file(submission_path) if submission_path.exists() else None,
        "files": {
            "solution": _file_entry(artifact_dir / "solution.py", base_dir=artifact_dir),
            "submission": _file_entry(submission_path, base_dir=artifact_dir),
            "oof_predictions": _file_entry(
                artifact_dir / "oof_predictions.csv.gz",
                base_dir=artifact_dir,
            ),
            "test_predictions": _file_entry(
                artifact_dir / "test_predictions.csv.gz",
                base_dir=artifact_dir,
            ),
            "validation_predictions": _file_entry(
                artifact_dir / "validation_predictions.csv.gz",
                base_dir=artifact_dir,
            ),
            "model_predictions": _directory_file_entries(
                artifact_dir / "model_predictions",
                base_dir=artifact_dir,
            ),
            "error": _file_entry(artifact_dir / "error.txt", base_dir=artifact_dir),
        },
        "node": {
            "id": None,
            "step": None,
            "ctime": dt.datetime.strptime(timestamp, "%Y%m%dT%H%M%S").timestamp(),
            "parent_id": None,
            "status": "ok" if is_ok else "bug",
            "origin": "profile_eval",
            "plan": f"Profile evaluation {profile}",
            "analysis": marker.get("summary") if marker else None,
            "is_buggy": not is_ok,
            "metric": {
                "value": float(metric) if metric is not None else None,
                "maximize": not lower_is_better,
                "name": eval_metric,
            },
            "submission_validation": (
                {"status": "ok"} if validation_error is None else {"status": "error", "error": validation_error}
            ),
        },
        "execution": {
            "exec_time": exec_result.exec_time,
            "exc_type": exec_result.exc_type,
            "exc_info": sanitize_persisted_payload(exec_result.exc_info),
            "exc_stack": sanitize_persisted_payload(exec_result.exc_stack),
        },
        "submission_validation": (
            {"status": "ok"} if validation_error is None else {"status": "error", "error": validation_error}
        ),
        "autogluon": {
            "profile": profile,
            "presets": autogluon_presets,
            "included_model_types": included_model_types,
            "time_limit": resolved_time_limit,
            "process_timeout": effective_timeout,
            "use_gpu": None,
            "eval_metric": eval_metric,
            "resolved_settings": {},
        },
        "source": {
            "source_run": source_record.get("run"),
            "source_node_id": source_record.get("node_id")
            or source_record.get("source_node_id"),
            "source_step": source_record.get("step") or source_record.get("source_step"),
            "source_timestamp": source_record.get("timestamp"),
            "source_sha256": source_sha,
        },
    }
    _write_json(artifact_dir / RESULT_MANIFEST_NAME, manifest)

    record = {
        **metadata,
        "run": run,
        "timestamp": timestamp,
        "artifact_dir": to_portable_path(artifact_dir),
        "solution_path": to_portable_path(artifact_dir / "solution.py"),
        "submission_path": to_portable_path(submission_path),
        "sha256": lab.sha256_file(submission_path) if submission_path.exists() else None,
        "step": None,
        "node_id": None,
        "parent_node_id": None,
        "is_buggy": not is_ok,
    }
    return record


def _find_existing_eval(
    records: list[dict[str, Any]],
    *,
    source_sha256: str,
    profile: str,
    presets: str | None,
    time_limit: int | None,
) -> dict[str, Any] | None:
    for record in records:
        if record.get("kind") != "profile_eval":
            continue
        if record.get("status") != "ok":
            continue
        if not record.get("sha256"):
            continue
        if not _record_path(record.get("submission_path")).exists():
            continue
        if record.get("source_sha256") != source_sha256:
            continue
        if record.get("profile") != profile:
            continue
        if presets is not None and record.get("autogluon_presets") != presets:
            continue
        if time_limit is not None and int(record.get("time_limit") or -1) != int(time_limit):
            continue
        return record
    return None


def _find_duplicate_profile_reruns(
    records: list[dict[str, Any]],
    planned: list[dict[str, Any]],
    *,
    profile: str,
    presets: str | None,
    time_limit: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    duplicates = []
    for record in planned:
        source_sha256 = record.get("sha256")
        if not source_sha256:
            continue
        existing = _find_existing_eval(
            records,
            source_sha256=source_sha256,
            profile=profile,
            presets=presets,
            time_limit=time_limit,
        )
        if existing is not None:
            duplicates.append((record, existing))
    return duplicates


def confirm_duplicate_profile_reruns(
    console: Console,
    duplicates: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    profile: str,
) -> bool:
    if not duplicates:
        return True

    if len(duplicates) == 1:
        warning = (
            "Warning: 1 selected source already has a successful profile evaluation "
            f"for profile {profile!r}."
        )
    else:
        warning = (
            f"Warning: {len(duplicates)} selected sources already have successful "
            f"profile evaluations for profile {profile!r}."
        )
    console.print(f"[yellow]{warning}[/yellow]")
    table = Table(title="Existing matching profile evaluations", expand=False)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("src_sha", no_wrap=True)
    table.add_column("existing_sha", no_wrap=True)
    table.add_column("run", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("artifacts", no_wrap=True)
    for rank, (source, existing) in enumerate(duplicates, start=1):
        table.add_row(
            str(rank),
            str(source.get("sha256") or "")[:10] or "-",
            str(existing.get("sha256") or "")[:10] or "-",
            _shorten_middle(source.get("run") or "-", 28),
            "" if source.get("step") is None else str(source.get("step")),
            str(existing.get("artifact_dir") or "-"),
        )
    console.print(table)

    if not sys.stdin.isatty():
        console.print(
            "[red]Refusing duplicate profile rerun in non-interactive mode. "
            "Use --force to run it anyway.[/red]"
        )
        return False

    return Confirm.ask(
        "Continue and rerun the same source/profile combination?",
        default=False,
        console=console,
    )


def parse_fit_args_json(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --fit-args-json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--fit-args-json must decode to a JSON object.")
    return parsed


def _format_score(value: Any) -> str:
    return "-" if value is None else f"{float(value):.5f}"


def _shorten_middle(value: Any, width: int) -> str:
    text = str(value or "-")
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    left = max(1, (width - 1) // 2)
    right = max(1, width - left - 1)
    return f"{text[:left]}…{text[-right:]}"


def render_profile_eval_results(console: Console, records: list[dict[str, Any]]) -> None:
    wide = console.width >= 150
    table = Table(title="Created profile evaluations", expand=False)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("cv", justify="right", no_wrap=True)
    table.add_column("profile", no_wrap=True)
    table.add_column("run", no_wrap=True)
    table.add_column("ts", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("src_sha", no_wrap=True)
    table.add_column("new_sha", no_wrap=True)
    if wide:
        table.add_column("artifacts", no_wrap=True)

    artifact_lines: list[str] = []
    for rank, record in enumerate(records, start=1):
        row = [
            str(rank),
            str(record.get("status") or "-"),
            _format_score(record.get("local_score")),
            str(record.get("profile") or "-"),
            _shorten_middle(record.get("run") or "-", 28 if wide else 20),
            str(record.get("timestamp") or "-"),
            "" if record.get("source_step") is None else str(record.get("source_step")),
            str(record.get("source_sha256") or "")[:10] or "-",
            str(record.get("sha256") or "")[:10] or "-",
        ]
        if wide:
            row.append(str(record.get("artifact_dir") or "-"))
        else:
            artifact_lines.append(f"{rank}. artifacts: {record.get('artifact_dir') or '-'}")
        table.add_row(*row)
    console.print(table)
    if artifact_lines:
        for line in artifact_lines:
            console.print(line)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun indexed AutoGluon candidates.")
    parser.add_argument("--competition", default=lab.DEFAULT_COMPETITION)
    parser.add_argument("--logs-dir", type=Path, default=lab.DEFAULT_LOGS_DIR)
    parser.add_argument("--index", type=Path, default=lab.DEFAULT_INDEX_PATH)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--presets")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--fit-args-json")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Process timeout in seconds. Defaults to profile time_limit plus 15 minutes.",
    )
    parser.add_argument("--memory-limit-gb", type=float, default=80.0)
    parser.add_argument(
        "--sha256", "--sha", dest="sha256", action="append", default=[], metavar="PREFIX"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-index", action="store_true")
    return parser.parse_args(argv)


def _build_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    try:
        sha_filters = lab.parse_sha256_filters(args.sha256)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    if not sha_filters:
        console.print("[red]At least one --sha256/--sha prefix is required.[/red]")
        return 2
    try:
        fit_args = parse_fit_args_json(args.fit_args_json)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    try:
        validate_autogluon_profile(args.profile)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    if args.refresh_index:
        with _build_progress(console) as progress:
            index = lab.refresh_index(
                logs_dir=args.logs_dir,
                index_path=args.index,
                competition=args.competition,
                progress=progress,
            )
    else:
        index = lab._load_json(args.index)
        if not index:
            console.print(
                f"[red]Missing submission index: {args.index}. "
                "Run with --refresh-index once.[/red]"
            )
            return 2
    try:
        selected = lab.filter_records_by_sha256(index.get("records", []), sha_filters)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    planned = selected

    if not args.execute:
        console.print(f"Would run {len(planned)} profile evaluation(s). Use --execute.")
        return 0

    if not args.force:
        duplicates = _find_duplicate_profile_reruns(
            index.get("records", []),
            planned,
            profile=args.profile,
            presets=args.presets,
            time_limit=args.time_limit,
        )
        if not confirm_duplicate_profile_reruns(
            console,
            duplicates,
            profile=args.profile,
        ):
            return 2

    results = []
    for record in planned:
        console.print(
            f"Running {args.profile} for {(record.get('sha256') or '')[:10]}..."
        )
        results.append(
            run_profile_eval(
                record,
                logs_dir=args.logs_dir,
                profile=args.profile,
                presets=args.presets,
                time_limit=args.time_limit,
                fit_args=fit_args,
                competition=args.competition,
                timeout=args.timeout,
                memory_limit_gb=args.memory_limit_gb,
                console=console,
            )
        )

    console.print(f"Created {len(results)} profile evaluation(s).")
    if results:
        render_profile_eval_results(console, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
