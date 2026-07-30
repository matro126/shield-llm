from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..data.prompts import split_sections
from .metrics import compute_text_metrics


SECTIONS = ("findings", "impression")


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

    out: dict[str, Any] = {
        "findings": metric_fn(
            [f for f, _ in pred_split],
            [f for f, _ in ref_split],
            list(metric_names),
            **metric_kwargs,
        ),
        "impression": None,
    }
    if target == "findings_impression":
        out["impression"] = metric_fn(
            [i or "" for _, i in pred_split],
            [i or "" for _, i in ref_split],
            list(metric_names),
            **metric_kwargs,
        )
    return out
