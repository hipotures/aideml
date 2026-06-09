# aide_refactor_runtime.py
#
# Fixed helper module for refactored AIDE/IDML solution scripts.
# The refactor LLM should import and use this module, not recreate it.

from __future__ import annotations

import atexit
import contextlib
import hashlib
import inspect
import json
import os
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union


def _now() -> float:
    return time.time()


def _perf() -> float:
    return time.perf_counter()


def _marker_value(value: Any) -> str:
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "/")


def _print_marker(prefix: str, **fields: Any) -> None:
    parts = [prefix]
    for key, value in fields.items():
        if value is not None:
            parts.append(f"{key}={_marker_value(value)}")
    print("|".join(parts), flush=True)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def stable_hash_json(value: Any, *, prefix: str = "") -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def hash_text(text: str, *, prefix: str = "") -> str:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def hash_sequence(values: Iterable[Any], *, prefix: str = "") -> str:
    return stable_hash_json(list(values), prefix=prefix)


def hash_index(values: Any, *, prefix: str = "") -> str:
    try:
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        if hasattr(values, "tolist"):
            values = values.tolist()
    except Exception:
        pass
    return stable_hash_json(values, prefix=prefix)


def hash_array_light(arr: Any, *, prefix: str = "", max_items: int = 2048) -> str:
    try:
        import numpy as np
        a = np.asarray(arr)
        flat = a.ravel()
        n = int(flat.size)
        if n <= max_items:
            sample = flat.tolist()
        else:
            idx = np.linspace(0, n - 1, num=max_items, dtype=np.int64)
            sample = flat[idx].tolist()
        return stable_hash_json(
            {"shape": list(a.shape), "dtype": str(a.dtype), "size": n, "sample": sample},
            prefix=prefix,
        )
    except Exception:
        return stable_hash_json({"type": str(type(arr)), "repr": repr(arr)[:4096]}, prefix=prefix)


def hash_function_source(fn: Callable[..., Any], *, prefix: str = "") -> str:
    try:
        source = inspect.getsource(fn)
    except Exception:
        source = repr(fn)
    return hash_text(source, prefix=prefix)


def safe_mkdir(path: Union[str, Path]) -> None:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def safe_write_json(path: Union[str, Path], data: Any) -> bool:
    try:
        path = Path(path)
        safe_mkdir(path.parent)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if "tmp" in locals():
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass
        return False


def append_jsonl(path: Union[str, Path], event: Mapping[str, Any]) -> bool:
    try:
        path = Path(path)
        safe_mkdir(path.parent)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(event), ensure_ascii=False, default=_json_default))
            f.write("\n")
        return True
    except Exception:
        return False


@dataclass
class AideContext:
    run_id: str
    node_id: str
    artifact_dir: Path
    shared_cache_dir: Path
    prediction_cache_dir: Path
    cache_mode: str
    stage_timings_path: Path
    stage_manifest_path: Path
    cache_events_path: Path
    cache_summary_path: Path
    refactor_notes_path: Path
    created_at_unix: float = field(default_factory=_now)


class AideRuntime:
    def __init__(self, working_dir: Optional[Union[str, Path]] = None):
        working_dir = Path(working_dir) if working_dir is not None else Path("./working")

        artifact_env = os.environ.get("AIDE_ARTIFACT_DIR", "").strip()
        artifact_dir = Path(artifact_env) if artifact_env else working_dir / "aide_artifacts"

        shared_cache_dir = Path(os.environ.get("AIDE_SHARED_CACHE_DIR", "logs/_shared_cache"))
        cache_mode = os.environ.get("AIDE_CACHE_MODE", "off").strip().lower()
        if cache_mode not in {"off", "write_only", "read_write"}:
            cache_mode = "off"

        self.context = AideContext(
            run_id=os.environ.get("AIDE_RUN_ID", "unknown_run"),
            node_id=os.environ.get("AIDE_NODE_ID", "unknown_node"),
            artifact_dir=artifact_dir,
            shared_cache_dir=shared_cache_dir,
            prediction_cache_dir=shared_cache_dir / "predictions",
            cache_mode=cache_mode,
            stage_timings_path=artifact_dir / "stage_timings.jsonl",
            stage_manifest_path=artifact_dir / "stage_manifest.json",
            cache_events_path=artifact_dir / "cache" / "cache_events.jsonl",
            cache_summary_path=artifact_dir / "cache" / "cache_summary.json",
            refactor_notes_path=artifact_dir / "refactor_notes.json",
        )

        self.stage_manifest: Dict[str, Dict[str, Any]] = {}
        self.cache_summary: Dict[str, Any] = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "load_failures": 0,
            "save_failures": 0,
            "saved_s_est": 0.0,
            "load_time_s": 0.0,
            "compute_time_s": 0.0,
            "by_model": {},
            "miss_reasons": {},
        }
        self.refactor_notes = []
        self._finalized = False

        safe_mkdir(self.context.artifact_dir)
        safe_mkdir(self.context.cache_events_path.parent)
        _print_marker("AIDE_CACHE", event="mode", mode=self.context.cache_mode)

    def register_stage(
        self,
        name: str,
        *,
        cache_candidate: bool = False,
        cache_type: Optional[str] = None,
        uses_target: Optional[bool] = None,
        fold_dependent: Optional[bool] = None,
    ) -> None:
        item = self.stage_manifest.setdefault(
            name,
            {
                "name": name,
                "cache_candidate": False,
                "cache_enabled_by_default": False,
                "uses_target": uses_target,
                "fold_dependent": fold_dependent,
            },
        )
        item["cache_candidate"] = bool(item.get("cache_candidate", False) or cache_candidate)
        if cache_type:
            item["cache_type"] = cache_type
        if uses_target is not None:
            item["uses_target"] = uses_target
        if fold_dependent is not None:
            item["fold_dependent"] = fold_dependent

    def log_stage_event(self, event: Dict[str, Any]) -> None:
        append_jsonl(self.context.stage_timings_path, event)

    def log_cache_event(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "run_id": self.context.run_id,
            "node_id": self.context.node_id,
            "ts_unix": _now(),
            **fields,
        }
        append_jsonl(self.context.cache_events_path, payload)

        marker = {k: v for k, v in payload.items() if k not in {"run_id", "node_id", "ts_unix", "shared_path"}}
        _print_marker("AIDE_CACHE", **marker)

        model = str(fields.get("model", fields.get("model_family", "unknown")))
        by_model = self.cache_summary.setdefault("by_model", {})
        m = by_model.setdefault(
            model,
            {"lookups": 0, "hits": 0, "misses": 0, "saves": 0, "load_failures": 0, "save_failures": 0, "saved_s_est": 0.0},
        )

        if event == "lookup":
            self.cache_summary["lookups"] += 1
            m["lookups"] += 1
        elif event == "hit":
            self.cache_summary["hits"] += 1
            m["hits"] += 1
            saved = float(fields.get("saved_s_est") or 0.0)
            load_s = float(fields.get("load_s") or 0.0)
            self.cache_summary["saved_s_est"] += saved
            self.cache_summary["load_time_s"] += load_s
            m["saved_s_est"] += saved
        elif event == "miss":
            self.cache_summary["misses"] += 1
            m["misses"] += 1
            reason = str(fields.get("reason", "unknown"))
            self.cache_summary.setdefault("miss_reasons", {})[reason] = self.cache_summary.setdefault("miss_reasons", {}).get(reason, 0) + 1
        elif event == "save":
            self.cache_summary["saves"] += 1
            m["saves"] += 1
            self.cache_summary["compute_time_s"] += float(fields.get("compute_s") or 0.0)
        elif event == "load_failed":
            self.cache_summary["load_failures"] += 1
            m["load_failures"] += 1
        elif event == "save_failed":
            self.cache_summary["save_failures"] += 1
            m["save_failures"] += 1

    def add_refactor_note(self, note_type: str, description: str, *, risk: str = "unknown", **extra: Any) -> None:
        self.refactor_notes.append({"type": note_type, "risk": risk, "description": description, **extra})

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        safe_write_json(
            self.context.stage_manifest_path,
            {
                "run_id": self.context.run_id,
                "node_id": self.context.node_id,
                "cache_mode": self.context.cache_mode,
                "artifact_dir": str(self.context.artifact_dir),
                "shared_cache_dir": str(self.context.shared_cache_dir),
                "prediction_cache_dir": str(self.context.prediction_cache_dir),
                "stages": list(self.stage_manifest.values()),
            },
        )
        safe_write_json(
            self.context.cache_summary_path,
            {
                "run_id": self.context.run_id,
                "node_id": self.context.node_id,
                "cache_mode": self.context.cache_mode,
                **self.cache_summary,
            },
        )
        if self.refactor_notes:
            safe_write_json(self.context.refactor_notes_path, self.refactor_notes)

        _print_marker(
            "AIDE_CACHE_SUMMARY",
            lookups=self.cache_summary.get("lookups", 0),
            hits=self.cache_summary.get("hits", 0),
            misses=self.cache_summary.get("misses", 0),
            saves=self.cache_summary.get("saves", 0),
            saved_s_est=f"{float(self.cache_summary.get('saved_s_est', 0.0)):.3f}",
        )


_RUNTIME: Optional[AideRuntime] = None


def get_aide_runtime(working_dir: Optional[Union[str, Path]] = None) -> AideRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = AideRuntime(working_dir=working_dir)
    return _RUNTIME


def get_aide_context(working_dir: Optional[Union[str, Path]] = None) -> AideContext:
    return get_aide_runtime(working_dir=working_dir).context


def finalize_aide_artifacts() -> None:
    try:
        if _RUNTIME is not None:
            _RUNTIME.finalize()
    except Exception:
        pass


atexit.register(finalize_aide_artifacts)


class aide_stage(contextlib.ContextDecorator):
    def __init__(
        self,
        name: str,
        *,
        cache_candidate: bool = False,
        cache_type: Optional[str] = None,
        uses_target: Optional[bool] = None,
        fold_dependent: Optional[bool] = None,
    ):
        self.name = name
        self.cache_candidate = cache_candidate
        self.cache_type = cache_type
        self.uses_target = uses_target
        self.fold_dependent = fold_dependent
        self.started = 0.0

    def __enter__(self):
        rt = get_aide_runtime()
        rt.register_stage(
            self.name,
            cache_candidate=self.cache_candidate,
            cache_type=self.cache_type,
            uses_target=self.uses_target,
            fold_dependent=self.fold_dependent,
        )
        self.started = _perf()
        _print_marker("AIDE_STAGE", event="start", stage=self.name)
        rt.log_stage_event({"event": "start", "stage": self.name, "ts_unix": _now(), "run_id": rt.context.run_id, "node_id": rt.context.node_id})
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = _perf() - self.started if self.started else 0.0
        rt = get_aide_runtime()
        if exc is None:
            _print_marker("AIDE_STAGE", event="end", stage=self.name, elapsed_s=f"{elapsed:.3f}")
            rt.log_stage_event({"event": "end", "stage": self.name, "elapsed_s": elapsed, "ts_unix": _now(), "run_id": rt.context.run_id, "node_id": rt.context.node_id})
            return False

        error_type = getattr(exc_type, "__name__", str(exc_type))
        _print_marker("AIDE_STAGE", event="failed", stage=self.name, elapsed_s=f"{elapsed:.3f}", error_type=error_type)
        rt.log_stage_event({
            "event": "failed",
            "stage": self.name,
            "elapsed_s": elapsed,
            "error_type": error_type,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb))[-8000:],
            "ts_unix": _now(),
            "run_id": rt.context.run_id,
            "node_id": rt.context.node_id,
        })
        return False


def log_cache_event(event: str, **fields: Any) -> None:
    try:
        get_aide_runtime().log_cache_event(event, **fields)
    except Exception:
        pass


def add_refactor_note(note_type: str, description: str, *, risk: str = "unknown", **extra: Any) -> None:
    try:
        get_aide_runtime().add_refactor_note(note_type, description, risk=risk, **extra)
    except Exception:
        pass


def _short_key(key: str) -> str:
    return key[:12]


def _cache_dir_for_key(key: str) -> Path:
    ctx = get_aide_context()
    return ctx.prediction_cache_dir / key[:2] / key[2:4] / key


def build_prediction_contract(
    *,
    model_family: str,
    variant_id: Union[str, int],
    fold_id: Union[str, int],
    class_order: Optional[Sequence[Any]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    train_features: Optional[Any] = None,
    valid_features: Optional[Any] = None,
    test_features: Optional[Any] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    fold_indices: Optional[Any] = None,
    target_values: Optional[Any] = None,
    sample_weight: Optional[Any] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    contract: Dict[str, Any] = {
        "cache_type": "fold_prediction",
        "model_family": str(model_family),
        "variant_id": str(variant_id),
        "fold_id": str(fold_id),
    }
    if class_order is not None:
        contract["class_order"] = [str(x) for x in class_order]
        contract["num_classes"] = len(class_order)
    if feature_cols is not None:
        contract["feature_cols_hash"] = hash_sequence(list(feature_cols), prefix="features_")
        contract["feature_cols_count"] = len(feature_cols)
    if train_features is not None:
        contract["train_features_hash"] = hash_array_light(train_features, prefix="train_features_")
    if valid_features is not None:
        contract["valid_features_hash"] = hash_array_light(valid_features, prefix="valid_features_")
    if test_features is not None:
        contract["test_features_hash"] = hash_array_light(test_features, prefix="test_features_")
    if model_params is not None:
        contract["model_params_hash"] = stable_hash_json(dict(model_params), prefix="params_")
    if fold_indices is not None:
        contract["fold_hash"] = hash_index(fold_indices, prefix="fold_")
    if target_values is not None:
        contract["target_hash"] = hash_index(target_values, prefix="target_")
    if sample_weight is not None:
        contract["sample_weight_hash"] = hash_array_light(sample_weight, prefix="sw_")
    if extra:
        contract.update(dict(extra))
    contract["prediction_key"] = stable_hash_json(contract)
    return contract


def _normalize_prediction_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, tuple) and len(result) in {3, 4}:
        out = {"valid_idx": result[0], "valid_proba": result[1], "test_proba": result[2]}
        if len(result) == 4:
            out["meta"] = result[3]
        return out
    raise TypeError("compute_fn must return dict or tuple(valid_idx, valid_proba, test_proba[, meta])")


def _validate_loaded(meta: Mapping[str, Any], valid_idx: Any, valid_proba: Any, test_proba: Any, contract: Mapping[str, Any]):
    try:
        import numpy as np
        vi = np.asarray(valid_idx)
        vp = np.asarray(valid_proba)
        tp = np.asarray(test_proba)
        if vi.ndim != 1:
            return False, "valid_idx_ndim"
        if vp.ndim != 2:
            return False, "valid_proba_ndim"
        if tp.ndim != 2:
            return False, "test_proba_ndim"
        if vp.shape[0] != vi.shape[0]:
            return False, "valid_rows_mismatch"
        if vp.shape[1] != tp.shape[1]:
            return False, "num_classes_mismatch"
        if contract.get("class_order") is not None and list(meta.get("class_order", [])) != list(contract.get("class_order", [])):
            return False, "class_order_mismatch"
        if contract.get("valid_rows") is not None and int(contract["valid_rows"]) != int(vp.shape[0]):
            return False, "expected_valid_rows_mismatch"
        if contract.get("test_rows") is not None and int(contract["test_rows"]) != int(tp.shape[0]):
            return False, "expected_test_rows_mismatch"
        if contract.get("num_classes") is not None and int(contract["num_classes"]) != int(vp.shape[1]):
            return False, "expected_num_classes_mismatch"
        return True, "ok"
    except Exception as ex:
        return False, f"validation_exception:{type(ex).__name__}"


def _load_prediction_cache(cache_dir: Path, contract: Mapping[str, Any]):
    started = _perf()
    try:
        meta_path = cache_dir / "meta.json"
        valid_idx_path = cache_dir / "valid_idx.npy"
        valid_proba_path = cache_dir / "valid_proba.npy"
        test_proba_path = cache_dir / "test_proba.npy"

        if not cache_dir.exists():
            return None, "not_found", 0.0
        for path, reason in [
            (meta_path, "missing_meta"),
            (valid_idx_path, "missing_valid_idx"),
            (valid_proba_path, "missing_valid_proba"),
            (test_proba_path, "missing_test_proba"),
        ]:
            if not path.exists():
                return None, reason, 0.0

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("prediction_key") != contract.get("prediction_key"):
            return None, "prediction_key_mismatch", 0.0

        import numpy as np
        valid_idx = np.load(valid_idx_path, allow_pickle=False)
        valid_proba = np.load(valid_proba_path, allow_pickle=False)
        test_proba = np.load(test_proba_path, allow_pickle=False)

        ok, reason = _validate_loaded(meta, valid_idx, valid_proba, test_proba, contract)
        if not ok:
            return None, reason, 0.0

        return {"valid_idx": valid_idx, "valid_proba": valid_proba, "test_proba": test_proba, "meta": meta, "cache_hit": True}, "ok", _perf() - started
    except Exception as ex:
        return None, f"load_exception:{type(ex).__name__}", 0.0


def _save_prediction_cache(cache_dir: Path, payload: Mapping[str, Any], contract: Mapping[str, Any], compute_s: float):
    try:
        import numpy as np
        safe_mkdir(cache_dir.parent)
        if cache_dir.exists():
            return True, "already_exists"

        tmp_dir = Path(tempfile.mkdtemp(prefix=f".tmp.{cache_dir.name}.", dir=str(cache_dir.parent)))
        try:
            valid_idx = np.asarray(payload["valid_idx"])
            valid_proba = np.asarray(payload["valid_proba"])
            test_proba = np.asarray(payload["test_proba"])
            extra_meta = dict(payload.get("meta") or {})
            meta = {
                **extra_meta,
                "cache_type": "fold_prediction",
                "prediction_key": contract.get("prediction_key"),
                "model_family": contract.get("model_family"),
                "variant_id": contract.get("variant_id"),
                "fold_id": contract.get("fold_id"),
                "run_id": get_aide_context().run_id,
                "node_id": get_aide_context().node_id,
                "class_order": list(contract.get("class_order", extra_meta.get("class_order", []))),
                "valid_rows": int(valid_proba.shape[0]),
                "test_rows": int(test_proba.shape[0]),
                "valid_proba_shape": list(valid_proba.shape),
                "test_proba_shape": list(test_proba.shape),
                "created_at_unix": _now(),
                "compute_s": float(compute_s),
                "contract": dict(contract),
            }
            np.save(tmp_dir / "valid_idx.npy", valid_idx)
            np.save(tmp_dir / "valid_proba.npy", valid_proba)
            np.save(tmp_dir / "test_proba.npy", test_proba)
            safe_write_json(tmp_dir / "meta.json", meta)
            os.replace(str(tmp_dir), str(cache_dir))
            return True, "saved"
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    except Exception as ex:
        return False, f"save_exception:{type(ex).__name__}"


def cached_fold_prediction(
    *,
    model_family: str,
    variant_id: Union[str, int],
    fold_id: Union[str, int],
    contract: Mapping[str, Any],
    compute_fn: Callable[[], Any],
) -> Dict[str, Any]:
    rt = get_aide_runtime()
    mode = rt.context.cache_mode

    full_contract = dict(contract)
    full_contract.setdefault("cache_type", "fold_prediction")
    full_contract.setdefault("model_family", str(model_family))
    full_contract.setdefault("variant_id", str(variant_id))
    full_contract.setdefault("fold_id", str(fold_id))
    full_contract.setdefault("prediction_key", stable_hash_json(full_contract))

    key = str(full_contract["prediction_key"])
    short = _short_key(key)
    cache_dir = _cache_dir_for_key(key)

    if mode == "read_write":
        rt.log_cache_event("lookup", type="fold_prediction", model=model_family, variant=variant_id, fold=fold_id, key=short, shared_path=str(cache_dir))
        loaded, reason, load_s = _load_prediction_cache(cache_dir, full_contract)
        if loaded is not None:
            compute_s = float((loaded.get("meta") or {}).get("compute_s") or 0.0)
            saved_s_est = max(0.0, compute_s - load_s)
            rt.log_cache_event("hit", type="fold_prediction", model=model_family, variant=variant_id, fold=fold_id, key=short, load_s=f"{load_s:.3f}", saved_s_est=f"{saved_s_est:.3f}", shared_path=str(cache_dir))
            return loaded
        if reason == "not_found":
            rt.log_cache_event("miss", type="fold_prediction", model=model_family, variant=variant_id, fold=fold_id, key=short, reason=reason, shared_path=str(cache_dir))
        else:
            rt.log_cache_event("load_failed", type="fold_prediction", model=model_family, variant=variant_id, fold=fold_id, key=short, reason=reason, action="recompute", shared_path=str(cache_dir))

    elif mode == "write_only":
        rt.log_cache_event("miss", type="fold_prediction", model=model_family, variant=variant_id, fold=fold_id, key=short, reason="write_only_no_read", shared_path=str(cache_dir))

    started = _perf()
    result = _normalize_prediction_result(compute_fn())
    compute_s = _perf() - started
    result["cache_hit"] = False

    if mode in {"write_only", "read_write"}:
        ok, reason = _save_prediction_cache(cache_dir, result, full_contract, compute_s)
        if ok:
            rt.log_cache_event("save", type="fold_prediction", model=model_family, variant=variant_id, fold=fold_id, key=short, compute_s=f"{compute_s:.3f}", shared_path=str(cache_dir))
        else:
            rt.log_cache_event("save_failed", type="fold_prediction", model=model_family, variant=variant_id, fold=fold_id, key=short, reason=reason, shared_path=str(cache_dir))

    return result
