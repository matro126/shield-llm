from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def build_record(
    *,
    uid: str,
    categories: Sequence[str],
    mesh_raw: Sequence[str] | None,
    projections: Sequence[str],
    rel_images: Sequence[str],
    assistant_text: str,
    system_prompt: str,
    user_prompt: str,
    source_lang: str,
    target_lang: str,
    translation_method: str,
) -> dict[str, Any]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": rel} for rel in rel_images
    ]
    content.append({"type": "text", "text": user_prompt})
    record = {
        "id": uid,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant_text},
        ],
        "images": list(rel_images),
        "factors": {
            "diagnostic_category": list(categories),
            "projection": (
                "+".join(projections) if len(projections) > 1 else projections[0]
            ),
            "views": list(projections),
            "task_type": "report_generation",
        },
        "provenance": {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translation_method": translation_method,
        },
    }
    if mesh_raw is not None:
        record["mesh_raw"] = list(mesh_raw)
    return record


def compute_stats(
    records_by_split: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    category_counts: dict[str, int] = defaultdict(int)
    projection_counts: dict[str, int] = defaultdict(int)
    for records in records_by_split.values():
        for record in records:
            factors = record["factors"]
            for category in factors["diagnostic_category"]:
                category_counts[category] += 1
            projection_counts[factors["projection"]] += 1
    return {
        "n_examples": {
            split: len(records) for split, records in records_by_split.items()
        },
        "diagnostic_category": dict(category_counts),
        "projection": dict(projection_counts),
    }
