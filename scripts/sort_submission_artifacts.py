from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SortResult:
    path: Path
    changed: bool
    rows: int
    backup_path: Path | None = None


def _backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.name}.bak-{stamp}")


def sort_submission_file(
    path: Path,
    *,
    backup: bool = True,
    dry_run: bool = False,
) -> SortResult:
    submission = pd.read_csv(path)
    if "id" not in submission.columns:
        raise ValueError(f"{path} does not contain an id column")

    sorted_submission = submission.sort_values("id", kind="mergesort").reset_index(
        drop=True
    )
    changed = not sorted_submission.equals(submission.reset_index(drop=True))
    backup_path = None
    if changed and not dry_run:
        if backup:
            backup_path = _backup_path(path)
            shutil.copy2(path, backup_path)
        sorted_submission.to_csv(path, index=False)

    return SortResult(
        path=path,
        changed=changed,
        rows=len(submission),
        backup_path=backup_path,
    )


def sort_submission_tree(
    log_dir: Path,
    *,
    backup: bool = True,
    dry_run: bool = False,
) -> list[SortResult]:
    artifact_dir = log_dir / "artifacts"
    if not artifact_dir.exists():
        return []

    results: list[SortResult] = []
    for path in sorted(artifact_dir.glob("*/submission.csv")):
        results.append(sort_submission_file(path, backup=backup, dry_run=dry_run))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort saved AIDE artifact submissions by id."
    )
    parser.add_argument("log_dir", type=Path, help="Experiment log directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak files before rewriting submissions",
    )
    args = parser.parse_args()

    results = sort_submission_tree(
        args.log_dir,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )
    changed = [result for result in results if result.changed]
    for result in results:
        marker = "changed" if result.changed else "ok"
        backup = f" backup={result.backup_path}" if result.backup_path else ""
        print(f"{marker} rows={result.rows} {result.path}{backup}")
    print(f"Processed {len(results)} submissions; changed {len(changed)}.")


if __name__ == "__main__":
    main()
