from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .images import normalize_image
from .prompts import format_target, get_prompts
from .records import build_record, compute_stats

SPLITS = ("train", "val", "test")
VARIANTS = ("other", "no_other")
ANNOTATION_NAMES = (
    "annotation.json",
    "annotation_labeled.json",
    "annotation_complete_labeled.json",
    "annotation_complete.json",
)
LABELS = frozenset(
    {
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
        "No Finding",
        "Other",
    }
)


@dataclass(frozen=True)
class DiagnosticVersion:
    views: str
    target: str


VERSIONS = {
    "iu_xray_en_F-F": DiagnosticVersion("frontal", "findings"),
    "iu_xray_en_F-FI": DiagnosticVersion("frontal", "findings_impression"),
    "iu_xray_en_FL-F": DiagnosticVersion("frontal_lateral", "findings"),
    "iu_xray_en_FL-FI": DiagnosticVersion("frontal_lateral", "findings_impression"),
}

CODE_FILES = (
    "src/shield/data/build_diagnostic.py",
    "src/shield/data/images.py",
    "src/shield/data/prompts.py",
    "src/shield/data/records.py",
    "src/shield/tracking/__init__.py",
    "src/shield/tracking/core.py",
    "src/shield/tracking/provenance.py",
)


def find_annotation(source_dir: Path) -> Path:
    for name in ANNOTATION_NAMES:
        candidate = source_dir / name
        if candidate.is_file():
            return candidate
    expected = ", ".join(ANNOTATION_NAMES)
    raise FileNotFoundError(
        f"Annotation diagnostica assente in {source_dir}. Nomi supportati: {expected}"
    )


def image_names(entry: Mapping[str, Any]) -> tuple[str, str]:
    raw_paths = entry.get("image_path")
    if not isinstance(raw_paths, list) or len(raw_paths) != 2:
        raise ValueError("image_path deve contenere esattamente due immagini")
    names = [Path(str(path)).name for path in raw_paths]
    frontal = [name for name in names if name.startswith("frontal")]
    lateral = [name for name in names if name.startswith("lateral")]
    if len(frontal) != 1 or len(lateral) != 1 or frontal[0] == lateral[0]:
        raise ValueError("image_path deve contenere una frontale e una laterale")
    return frontal[0], lateral[0]


def diagnostic_labels(entry: Mapping[str, Any]) -> list[str]:
    sample_id = str(entry.get("id", "<senza id>"))
    labels = entry.get("diagnostic_label")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{sample_id}: diagnostic_label deve essere una lista non vuota")
    if any(not isinstance(label, str) for label in labels):
        raise ValueError(f"{sample_id}: diagnostic_label contiene valori non testuali")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"{sample_id}: diagnostic_label contiene duplicati {duplicates}")
    unknown = sorted(set(labels) - LABELS)
    if unknown:
        raise ValueError(f"{sample_id}: diagnostic_label sconosciute {unknown}")
    if "No Finding" in labels and len(labels) != 1:
        raise ValueError(f"{sample_id}: No Finding non può coesistere con altre etichette")
    return list(labels)


def validate_annotation(annotation: Mapping[str, Any], images_dir: Path) -> None:
    seen: set[str] = set()
    for split in SPLITS:
        records = annotation.get(split)
        if not isinstance(records, list):
            raise ValueError(f"Split {split!r} assente o non valido")
        for entry in records:
            if not isinstance(entry, Mapping):
                raise ValueError(f"Split {split!r}: sample non valido")
            sample_id = str(entry.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"Split {split!r}: id assente")
            if sample_id in seen:
                raise ValueError(f"ID duplicato tra gli split: {sample_id}")
            seen.add(sample_id)
            diagnostic_labels(entry)
            for field in ("report", "impression"):
                value = entry.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{sample_id}: {field} assente o vuoto")
            try:
                names = image_names(entry)
            except ValueError as exc:
                raise ValueError(f"{sample_id}: {exc}") from None
            missing = [name for name in names if not (images_dir / sample_id / name).is_file()]
            if missing:
                raise FileNotFoundError(f"{sample_id}: immagini assenti {missing}")


def selected_entries(
    annotation: Mapping[str, Any], split: str, variant: str
) -> list[Mapping[str, Any]]:
    records = annotation[split]
    if variant == "other":
        return list(records)
    return [entry for entry in records if "Other" not in diagnostic_labels(entry)]


def build_records(
    version: DiagnosticVersion,
    variant: str,
    annotation: Mapping[str, Any],
    images_dir: Path,
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    prompts = get_prompts(version.views, version.target, "en")
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for entry in selected_entries(annotation, split, variant):
            sample_id = str(entry["id"])
            frontal, lateral = image_names(entry)
            names = [frontal] if version.views == "frontal" else [frontal, lateral]
            projections = ["frontal"] if version.views == "frontal" else ["frontal", "lateral"]
            rel_images: list[str] = []
            for name in names:
                relative = f"images_normalized/{sample_id}/{name}"
                normalize_image(images_dir / sample_id / name, output_dir / relative)
                rel_images.append(relative)
            report = str(entry["report"]).strip()
            impression = str(entry["impression"]).strip()
            record = build_record(
                uid=sample_id,
                categories=diagnostic_labels(entry),
                mesh_raw=None,
                projections=projections,
                rel_images=rel_images,
                assistant_text=format_target(report, impression, version.target, "en"),
                system_prompt=prompts.system,
                user_prompt=prompts.user,
                source_lang="en",
                target_lang="en",
                translation_method="passthrough",
            )
            record["r2gen_report"] = report
            records_by_split[split].append(record)
    return records_by_split


def write_outputs(
    version_name: str,
    version: DiagnosticVersion,
    variant: str,
    records_by_split: Mapping[str, list[dict[str, Any]]],
    output_dir: Path,
    root: Path,
    source_dir: Path,
    annotation_path: Path,
) -> dict[str, Any]:
    import yaml

    from ..tracking import sha256_file, sha256_text, sha256_tree

    for split, records in records_by_split.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = compute_stats(records_by_split)
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prompts = get_prompts(version.views, version.target, "en")
    code_hashes = {
        relative: sha256_file(root / relative)
        for relative in CODE_FILES
        if (root / relative).is_file()
    }
    uv_lock = root / "uv.lock"
    manifest = {
        "dataset": "iu-xray",
        "version": version_name,
        "views": version.views,
        "target": version.target,
        "lang": "en",
        "label_policy": variant,
        "split_source": str(annotation_path.relative_to(root) if annotation_path.is_relative_to(root) else annotation_path),
        "text_source": "diagnostic annotation report and impression",
        "n_examples": stats["n_examples"],
        "prompts": {"system": prompts.system, "user": prompts.user},
        "integrity": {
            "inputs": {
                "annotation": sha256_file(annotation_path),
                "images": sha256_tree(source_dir / "images", "*.png"),
                "uv_lock": sha256_file(uv_lock) if uv_lock.is_file() else "absent",
            },
            "code": code_hashes,
            "prompts": {
                "system": sha256_text(prompts.system),
                "user": sha256_text(prompts.user),
            },
            "outputs": {
                **{
                    f"{split}.jsonl": sha256_file(output_dir / f"{split}.jsonl")
                    for split in SPLITS
                },
                "images_normalized": sha256_tree(output_dir / "images_normalized", "*.png"),
            },
        },
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return stats


def materialize(
    root: Path,
    source_dir: Path,
    out_root: Path,
    version_name: str,
    variant: str,
) -> dict[str, Any]:
    annotation_path = find_annotation(source_dir)
    loaded = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("La radice dell'annotation deve essere un oggetto JSON")
    images_dir = source_dir / "images"
    validate_annotation(loaded, images_dir)
    version = VERSIONS[version_name]
    destination = out_root / variant / version_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{version_name}.", dir=destination.parent))
    try:
        records = build_records(version, variant, loaded, images_dir, staging)
        stats = write_outputs(
            version_name,
            version,
            variant,
            records,
            staging,
            root,
            source_dir,
            annotation_path,
        )
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Genera dataset inglesi usando diagnostic_label come ground truth."
    )
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/iu-xray/iu_xray_r2gen_final_impression_diagnostic"),
    )
    parser.add_argument("--out", type=Path, default=Path("dataset/iu-xray/en"))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--version", required=True, choices=sorted(VERSIONS))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    source_dir = args.source if args.source.is_absolute() else root / args.source
    out_root = args.out if args.out.is_absolute() else root / args.out
    stats = materialize(root, source_dir, out_root, args.version, args.variant)
    counts = "  ".join(f"{split}={stats['n_examples'][split]}" for split in SPLITS)
    print(f"[build-diagnostic] {args.variant}/{args.version}  {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
