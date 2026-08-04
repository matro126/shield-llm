from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from statistics import median
from typing import Any

CLINICAL_LABELS = (
    "Pneumothorax",
    "Pleural Effusion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Lung Lesion",
    "Lung Opacity",
    "Cardiomegaly",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Pleural Other",
    "Support Devices",
)
NO_FINDING = "No Finding"
OTHER = "Other"
UNLABELED = "Unlabeled"
ALLOWED_LABELS = frozenset((*CLINICAL_LABELS, NO_FINDING, OTHER, UNLABELED))
CLINICAL_SYSTEM_PROMPT = (
    "You are an expert radiologist. Classify the visible chest X-ray findings using "
    "only the requested clinical labels. Return only positive labels."
)
CLINICAL_USER_PROMPT = (
    "Classify the chest X-ray. Answer EXACTLY in this format:\n"
    "Clinical findings:\n"
    "<one positive label per line>\n"
    f"Allowed labels: {', '.join((*CLINICAL_LABELS, NO_FINDING))}.\n"
    "Use No Finding only when no listed abnormality is present."
)


def _labels(record: dict[str, Any]) -> list[str]:
    factors = record.get("factors")
    if not isinstance(factors, dict):
        raise ValueError(f"Record {record.get('id')!r} has invalid factors")
    labels = factors.get("diagnostic_category")
    if not isinstance(labels, list) or not labels:
        raise ValueError(
            f"Record {record.get('id')!r} has invalid diagnostic_category"
        )
    return labels


def validate_clinical_source(
    records: Sequence[dict[str, Any]], expected_images: int
) -> None:
    if expected_images < 1:
        raise ValueError("expected_images must be positive")
    for record in records:
        uid = record.get("id")
        images = record.get("images")
        if not isinstance(images, list) or len(images) != expected_images:
            raise ValueError(
                f"Record {uid!r} has {len(images) if isinstance(images, list) else 0} "
                f"images, expected {expected_images}"
            )
        labels = _labels(record)
        unknown = sorted(set(labels) - ALLOWED_LABELS)
        if unknown:
            raise ValueError(f"Record {uid!r} has unknown labels: {unknown}")
        if len(labels) != len(set(labels)):
            raise ValueError(f"Record {uid!r} has duplicate labels")
        if NO_FINDING in labels and len(labels) != 1:
            raise ValueError(f"Record {uid!r} mixes No Finding with pathology")
        if (OTHER in labels or UNLABELED in labels) and len(labels) != 1:
            raise ValueError(f"Record {uid!r} mixes fallback and clinical labels")
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"Record {uid!r} has invalid messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Record {uid!r} has no assistant target")


def _ordered_positive_labels(labels: Sequence[str]) -> list[str]:
    if labels == [NO_FINDING]:
        return [NO_FINDING]
    present = set(labels)
    return [label for label in CLINICAL_LABELS if label in present]


def _clinical_record(record: dict[str, Any]) -> dict[str, Any]:
    transformed = deepcopy(record)
    ordered = _ordered_positive_labels(_labels(record))
    transformed["messages"][0]["content"] = CLINICAL_SYSTEM_PROMPT
    content = transformed["messages"][1].get("content")
    if not isinstance(content, list):
        raise ValueError(f"Record {record.get('id')!r} has invalid user content")
    image_content = [item for item in content if item.get("type") == "image"]
    image_content.append({"type": "text", "text": CLINICAL_USER_PROMPT})
    transformed["messages"][1]["content"] = image_content
    transformed["messages"][-1]["content"] = "Clinical findings:\n" + "\n".join(
        ordered
    )
    transformed["factors"]["task_type"] = "clinical_classification"
    transformed["factors"]["diagnostic_category"] = ordered
    return transformed


def build_clinical_records(
    records: Sequence[dict[str, Any]], expected_images: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_clinical_source(records, expected_images)
    clinical: list[dict[str, Any]] = []
    excluded = Counter()
    frequencies = Counter()
    for record in records:
        labels = _labels(record)
        if labels == [OTHER]:
            excluded["other"] += 1
            continue
        if labels == [UNLABELED]:
            excluded["unlabeled"] += 1
            continue
        transformed = _clinical_record(record)
        clinical.append(transformed)
        frequencies.update(transformed["factors"]["diagnostic_category"])
    stats = {
        "source_records": len(records),
        "clinical_records": len(clinical),
        "excluded_other": excluded["other"],
        "excluded_unlabeled": excluded["unlabeled"],
        "label_frequencies": dict(sorted(frequencies.items())),
    }
    return clinical, stats


def _stratum(record: dict[str, Any]) -> str:
    labels = _labels(record)
    if labels == [NO_FINDING]:
        return "healthy"
    if any(label in CLINICAL_LABELS for label in labels):
        return "pathological"
    return "other"


def _allocated_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(ratios, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _label_weight_data(
    records: Sequence[dict[str, Any]], cap: float
) -> tuple[list[float], dict[str, Any]]:
    frequencies = Counter(
        label
        for record in records
        for label in _labels(record)
        if label in CLINICAL_LABELS
    )
    raw = {label: 1.0 / math.sqrt(count) for label, count in frequencies.items()}
    midpoint = median(raw.values()) if raw else 1.0
    effective = {
        label: min(cap, value / midpoint) for label, value in sorted(raw.items())
    }
    weights = [
        max((effective[label] for label in _labels(record) if label in effective), default=1.0)
        for record in records
    ]
    details: dict[str, Any] = dict(effective)
    details["median"] = 1.0
    details["max"] = max(effective.values(), default=1.0)
    details["cap"] = cap
    return weights, details


def _sample(
    rng: random.Random,
    records: Sequence[dict[str, Any]],
    count: int,
    weights: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if not records:
        raise ValueError("Cannot sample from an empty training stratum")
    return rng.choices(list(records), weights=weights, k=count)


def build_stage_two_records(
    report_records: Sequence[dict[str, Any]],
    clinical_records: Sequence[dict[str, Any]],
    cfg: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strategy = cfg.training_strategy
    if strategy == "standard":
        return report_records, {
            "strategy": strategy,
            "source_records": len(report_records),
            "effective_records": len(report_records),
            "task_counts": {"report_generation": len(report_records)},
        }
    rng = random.Random(cfg.seed)
    total = len(report_records)
    clinical_count = (
        round(total * cfg.clinical_rehearsal_ratio) if strategy == "clinical" else 0
    )
    report_count = total - clinical_count
    groups = {"healthy": [], "pathological": [], "other": []}
    for record in report_records:
        groups[_stratum(record)].append(record)
    ratios = {
        "healthy": cfg.healthy_ratio,
        "pathological": cfg.pathological_ratio,
        "other": cfg.other_ratio,
    }
    counts = _allocated_counts(report_count, ratios)
    pathology_weights, weight_details = _label_weight_data(
        groups["pathological"], cfg.rare_weight_cap
    )
    selected: list[dict[str, Any]] = []
    selected.extend(_sample(rng, groups["healthy"], counts["healthy"]))
    selected.extend(
        _sample(
            rng,
            groups["pathological"],
            counts["pathological"],
            pathology_weights,
        )
    )
    selected.extend(_sample(rng, groups["other"], counts["other"]))
    if clinical_count:
        clinical_weights, _ = _label_weight_data(
            [
                row
                for row in clinical_records
                if _labels(row) != [NO_FINDING]
            ],
            cfg.rare_weight_cap,
        )
        abnormal = [row for row in clinical_records if _labels(row) != [NO_FINDING]]
        source = abnormal if abnormal else clinical_records
        weights = clinical_weights if abnormal else None
        selected.extend(_sample(rng, source, clinical_count, weights))
    rng.shuffle(selected)
    task_counts = Counter(row["factors"]["task_type"] for row in selected)
    frequencies = Counter(
        label
        for row in selected
        if row["factors"]["task_type"] == "report_generation"
        for label in _labels(row)
    )
    stats = {
        "strategy": strategy,
        "seed": cfg.seed,
        "source_records": total,
        "effective_records": len(selected),
        "task_counts": dict(sorted(task_counts.items())),
        "source_strata_counts": {name: len(rows) for name, rows in groups.items()},
        "report_strata_counts": counts,
        "report_label_frequencies": dict(sorted(frequencies.items())),
        "pathology_weights": weight_details,
    }
    return selected, stats
