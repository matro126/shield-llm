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


WHOLE_REPORT_METRICS = {"chexbert": "chexbert_"}


def _join(sections: Sequence[tuple[str, str | None]]) -> list[str]:
    return [" ".join(part for part in (a, b) if part and part.strip()).strip()
            for a, b in sections]


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

    if target != "findings_impression":
        findings = metric_fn(pred_f, ref_f, list(metric_names),
                             chexbert_per_class=True, **metric_kwargs)
        return {"findings": findings, "impression": None, "report": None,
                "mean": findings}

    findings = metric_fn(pred_f, ref_f, list(metric_names), **metric_kwargs)

    pred_i = [i or "" for _, i in pred_split]
    ref_i = [i or "" for _, i in ref_split]
    impression = metric_fn(pred_i, ref_i, list(metric_names), **metric_kwargs)

    mean = _mean(findings, impression)
    report = None
    whole = [m for m in metric_names if m in WHOLE_REPORT_METRICS]
    if whole:
        report = metric_fn(_join(pred_split), _join(ref_split), whole,
                           chexbert_per_class=True, **metric_kwargs)
        prefixes = tuple(WHOLE_REPORT_METRICS[m] for m in whole)
        for key, value in report.items():
            if key.startswith(prefixes):
                mean[key] = value

    return {
        "findings": findings,
        "impression": impression,
        "report": report,
        "mean": mean,
    }
