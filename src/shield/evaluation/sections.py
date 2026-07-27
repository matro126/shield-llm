from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..data.prompts import split_sections
from .metrics import compute_text_metrics


def _mean(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    keys = set(a) | set(b)
    out: dict[str, float] = {}
    for key in keys:
        if key == "num_examples":
            out[key] = a.get(key, b.get(key, 0.0))
        else:
            vals = [v for v in (a.get(key), b.get(key)) if v is not None]
            out[key] = sum(vals) / len(vals) if vals else 0.0
    return out


def sectioned_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    metric_names: Sequence[str],
    target: str,
    metric_fn: Callable[..., dict[str, float]] = compute_text_metrics,
    **metric_kwargs: Any,
) -> dict[str, Any]:
    pred_split = [split_sections(p) for p in predictions]
    ref_split = [split_sections(r) for r in references]

    pred_f = [f for f, _ in pred_split]
    ref_f = [f for f, _ in ref_split]
    findings = metric_fn(pred_f, ref_f, list(metric_names), **metric_kwargs)

    if target != "findings_impression":
        return {"findings": findings, "impression": None, "mean": findings}

    pred_i = [i or "" for _, i in pred_split]
    ref_i = [i or "" for _, i in ref_split]
    impression = metric_fn(pred_i, ref_i, list(metric_names), **metric_kwargs)

    return {
        "findings": findings,
        "impression": impression,
        "mean": _mean(findings, impression),
    }
