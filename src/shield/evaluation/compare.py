from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_metrics_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping) and isinstance(data.get("aggregate"), Mapping):
        return dict(data["aggregate"])
    return dict(data) if isinstance(data, Mapping) else {}


def baseline_metrics_path(
    project_root: str | Path, family: str, baseline_run: str
) -> Path:
    return (
        Path(project_root)
        / "outputs"
        / family
        / baseline_run
        / "evaluation"
        / "metrics.json"
    )


def baseline_predictions_path(
    project_root: str | Path, family: str, baseline_run: str
) -> Path:
    return (
        Path(project_root)
        / "outputs"
        / family
        / baseline_run
        / "evaluation"
        / "predictions.json"
    )


def compare_to_baseline(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in current.items():
        if not _is_number(value):
            continue
        base = baseline.get(key)
        if not _is_number(base):
            continue
        delta = value - base
        out[key] = {
            "current": value,
            "baseline": base,
            "delta": delta,
            "rel_pct": (delta / base * 100.0) if base else None,
        }
    return out


def beats_baseline(
    comparison: Mapping[str, Mapping[str, Any]], key: str = "bertscore_f1"
) -> bool | None:
    entry = comparison.get(key)
    if not entry:
        return None
    return entry["delta"] > 0
