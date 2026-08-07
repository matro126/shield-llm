from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


LABEL_TRANSLATIONS = {
    "Atelectasis": "Atelettasia",
    "Cardiomegaly": "Cardiomegalia",
    "Consolidation": "Consolidamento",
    "Edema": "Edema",
    "Enlarged Cardiomediastinum": "Allargamento cardiomediastinico",
    "Fracture": "Frattura",
    "Lung Lesion": "Lesione polmonare",
    "Lung Opacity": "Opacità polmonare",
    "Pleural Effusion": "Versamento pleurico",
    "Pleural Other": "Altra anomalia pleurica",
    "Pneumonia": "Polmonite",
    "Pneumothorax": "Pneumotorace",
    "Support Devices": "Dispositivi di supporto",
    "No Finding": "Nessun reperto",
    "Other": "Altro",
}


def load_annotation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotation root must be a JSON object")
    return payload


def translated_labels(sample_id: str, labels: Any) -> list[str]:
    if not isinstance(labels, list) or not labels:
        raise ValueError(
            f"{sample_id}: diagnostic_label must be a non-empty list"
        )
    if not all(isinstance(label, str) for label in labels):
        raise ValueError(
            f"{sample_id}: diagnostic_label entries must be strings"
        )
    translated: list[str] = []
    for label in labels:
        if label not in LABEL_TRANSLATIONS:
            raise ValueError(f"{sample_id}: unknown diagnostic label: {label}")
        translated.append(LABEL_TRANSLATIONS[label])
    return translated


def translate_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(annotation, Mapping):
        raise ValueError("annotation root must be a JSON object")
    result = copy.deepcopy(dict(annotation))
    for split, records in result.items():
        if not isinstance(records, list):
            raise ValueError(f"split {split!r} must be a list")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"non-object sample in split {split} at index {index}"
                )
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(
                    f"invalid sample id in split {split} at index {index}"
                )
            if "diagnostic_label" not in record:
                raise ValueError(f"{sample_id}: missing diagnostic_label")
            record["diagnostic_label"] = translated_labels(
                sample_id, record["diagnostic_label"]
            )
    return result


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_diagnostic_it{input_path.suffix}"
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traduci in italiano le diagnostic_label di un'annotation JSON"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> Path:
    input_path = args.input
    output_path = args.output or default_output_path(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")
    annotation = load_annotation(input_path)
    translated = translate_annotation(annotation)
    atomic_write_json(output_path, translated)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"errore: {exc}", file=sys.stderr)
        return 1
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
