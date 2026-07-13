from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise TypeError(f"Formato messages non valido: {type(messages)!r}")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError(f"Messaggio chat non valido: {type(message)!r}")
        item = dict(message)
        if item.get("role") == "assistant" and isinstance(item.get("content"), list):
            item["content"] = "\n".join(str(value) for value in item["content"])
        normalized.append(item)
    return normalized


def extract_assistant_text(example: Mapping[str, Any]) -> str:
    for message in normalize_messages(example.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def extract_reference(example: Mapping[str, Any]) -> str:
    reference = example.get("r2gen_report")
    if isinstance(reference, str) and reference.strip():
        return reference
    return extract_assistant_text(example)


def extract_image_paths(example: Mapping[str, Any]) -> list[str]:
    images = example.get("images")
    if isinstance(images, str) and images:
        try:
            images = json.loads(images)
        except json.JSONDecodeError:
            images = [images]
    if isinstance(images, (list, tuple)) and images:
        return [str(value) for value in images if value]
    paths: list[str] = []
    for message in normalize_messages(example.get("messages", [])):
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image":
                    value = item.get("image")
                    if value:
                        paths.append(str(value))
    return paths


def extract_factors(example: Mapping[str, Any]) -> dict[str, Any]:
    factors = example.get("factors", {})
    return dict(factors) if isinstance(factors, Mapping) else {}


def resolve_images(
    record: Mapping[str, Any], images_root: str | Path
) -> dict[str, Any]:
    root = Path(images_root)

    def _abs(value: Any) -> Any:
        if isinstance(value, str) and value and not Path(value).is_absolute():
            return str(root / value)
        return value

    rec = dict(record)
    if isinstance(rec.get("images"), (list, tuple)):
        rec["images"] = [_abs(value) for value in rec["images"]]

    messages = normalize_messages(rec.get("messages", []))
    new_messages: list[dict[str, Any]] = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image":
                    item = dict(item)
                    item["image"] = _abs(item.get("image"))
                new_content.append(item)
            message["content"] = new_content
        new_messages.append(message)
    rec["messages"] = new_messages
    return rec


def load_records(
    dataset_root: str | Path,
    split: str,
    images_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    images = Path(images_root) if images_root is not None else root
    records = load_jsonl(root / f"{split}.jsonl")
    return [resolve_images(record, images) for record in records]


def to_hf_dataset(records: list[Mapping[str, Any]]):
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    for example in records:
        messages = normalize_messages(example.get("messages", []))
        rows.append(
            {
                "id": example.get("id"),
                "messages": json.dumps(messages, ensure_ascii=False),
                "images": json.dumps(extract_image_paths(example), ensure_ascii=False),
                "factors": json.dumps(example.get("factors", {}), ensure_ascii=False),
            }
        )
    return Dataset.from_list(rows)


def dataset_summary(
    train_records: list, val_records: list, test_records: list
) -> dict[str, int]:
    return {
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
    }
