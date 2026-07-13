from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

MetricFn = Callable[[list[str], list[str]], dict[str, float]]


def _factor_values(factors: Mapping[str, Any], key: str) -> list[str]:
    value = factors.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def disaggregate(
    records: list[Mapping[str, Any]],
    predictions: list[str],
    references: list[str],
    factor_keys: list[str],
    metric_fn: MetricFn,
    min_subgroup_size: int = 20,
    known_values: Mapping[str, list[str]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    known_values = known_values or {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for key in factor_keys:
        groups: dict[str, list[int]] = {
            str(value): [] for value in known_values.get(key, [])
        }
        for index, record in enumerate(records):
            factors = record.get("factors", {})
            factors = factors if isinstance(factors, Mapping) else {}
            for value in _factor_values(factors, key):
                groups.setdefault(value, []).append(index)

        key_out: dict[str, dict[str, Any]] = {}
        for value, indices in sorted(groups.items()):
            if len(indices) < min_subgroup_size:
                key_out[value] = {"n": len(indices), "status": "not_estimable"}
            else:
                sub_pred = [predictions[i] for i in indices]
                sub_ref = [references[i] for i in indices]
                metrics = metric_fn(sub_pred, sub_ref)
                metrics["n"] = len(indices)
                key_out[value] = metrics
        out[key] = key_out
    return out
