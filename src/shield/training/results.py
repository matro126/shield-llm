from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value: Any) -> Any:
    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    payload = _finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


class ResultsWriter:
    def __init__(self, path: Path, payload: dict[str, Any]):
        self.path = Path(path)
        self.payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            **payload,
        }
        self.payload.setdefault("curves", {"train": [], "validation": []})
        self.payload.setdefault("best", None)
        self.flush()


    def set(self, **fields: Any) -> None:
        self.payload.update(fields)

    def set_in(self, section: str, **fields: Any) -> None:
        current = self.payload.setdefault(section, {})
        if isinstance(current, dict):
            current.update(fields)

    def set_curves(self, train: list[dict], validation: list[dict]) -> None:
        self.payload["curves"] = {"train": list(train), "validation": list(validation)}

    def set_best(self, best: dict[str, Any] | None) -> None:
        self.payload["best"] = best

    def flush(self) -> None:
        self.payload["updated_at"] = now_iso()
        write_json_atomic(self.path, self.payload)


def validation_row(
    epoch: float,
    step: int,
    val_loss: float,
    sectioned: dict[str, Any],
    eval_seconds: float,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "epoch": round(float(epoch), 4),
        "step": int(step),
        "val_loss": float(val_loss),
        "metrics": {k: float(v) for k, v in sectioned["mean"].items()},
        "sections": {
            name: ({k: float(v) for k, v in values.items()}
                   if isinstance(values, dict) else None)
            for name, values in (
                ("findings", sectioned.get("findings")),
                ("impression", sectioned.get("impression")),
                ("report", sectioned.get("report")),
                ("mesh", sectioned.get("mesh")),
            )
        },
        "eval_seconds": round(float(eval_seconds), 1),
        "elapsed_s": round(float(elapsed_s), 1),
    }


def flatten_validation_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "epoch": row["epoch"],
        "step": row["step"],
        "val_loss": row["val_loss"],
    }
    flat.update(row.get("metrics", {}))
    for section, values in (row.get("sections") or {}).items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
    flat["eval_seconds"] = row["eval_seconds"]
    flat["elapsed_s"] = row["elapsed_s"]
    return flat
