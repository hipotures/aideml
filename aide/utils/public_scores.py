from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _registry_entries(payload: Any) -> list[dict[str, Any]]:
    entries = payload.get("submissions", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _parse_public_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def public_adjusted_oriented_score(
    *,
    local_score: float,
    public_score: float | None,
    maximize: bool,
    weight: float,
    cap: float,
) -> float:
    oriented_local = float(local_score) if maximize else -float(local_score)
    if public_score is None:
        return oriented_local
    weight = max(0.0, float(weight))
    cap = max(0.0, float(cap))
    if weight <= 0.0 or cap <= 0.0:
        return oriented_local
    oriented_public = float(public_score) if maximize else -float(public_score)
    uplift = min(max(oriented_public - oriented_local, 0.0), cap)
    return oriented_local + weight * uplift


def load_public_scores_by_node_id(log_dir: Path | str) -> dict[str, float]:
    run_dir = Path(log_dir)
    registry_path = run_dir.parent / "submission_registry.json"
    if not registry_path.exists():
        return {}
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    scores: dict[str, float] = {}
    for entry in _registry_entries(payload):
        if str(entry.get("run") or "") != run_dir.name:
            continue
        node_id = entry.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        public_score = _parse_public_score(entry.get("public_score"))
        if public_score is None:
            continue
        current = scores.get(node_id)
        if current is None or public_score > current:
            scores[node_id] = public_score
    return scores
