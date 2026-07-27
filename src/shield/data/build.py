from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .images import normalize_image
from .openi import map_mesh_to_chexpert
from .prompts import format_target, get_prompts
from .records import build_record, compute_stats
from .text_source import TextSource

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Version:
    views: str
    target: str
    lang: str


_LANG_DIR = {"en": "en", "it": "ita"}
_VIEWS_CODE = {"frontal_lateral": "FL", "frontal": "F"}
_TARGET_CODE = {"findings": "F", "findings_impression": "FI"}


def version_name(version: Version) -> str:
    return (
        f"iu_xray_{version.lang}_"
        f"{_VIEWS_CODE[version.views]}-{_TARGET_CODE[version.target]}"
    )


def version_relpath(version: Version) -> str:
    return f"{_LANG_DIR[version.lang]}/{version_name(version)}"


def _versions() -> dict[str, Version]:
    combos = [
        ("frontal_lateral", "findings"),
        ("frontal", "findings"),
        ("frontal_lateral", "findings_impression"),
        ("frontal", "findings_impression"),
    ]
    out: dict[str, Version] = {}
    for views, target in combos:
        for lang in ("en", "it"):
            version = Version(views, target, lang)
            out[version_name(version)] = version
    return out


VERSIONS: dict[str, Version] = _versions()


def _view_of(name: str) -> str:
    return "frontal" if name.startswith("frontal") else "lateral"


def pair_names(entry: Mapping[str, Any]) -> list[str]:
    return sorted(Path(path).name for path in entry["image_path"])


def select_images(names: Sequence[str], views: str) -> tuple[list[str], list[str]]:
    if views == "frontal":
        frontal = next(n for n in names if n.startswith("frontal"))
        return [frontal], ["frontal"]
    return list(names), [_view_of(n) for n in names]


def validate_annotation(
    annotation: Mapping[str, Any], text: TextSource, images_dir: Path
) -> list[str]:
    problems: list[str] = []
    for split in SPLITS:
        for entry in annotation.get(split, []):
            sample_id = str(entry["id"])
            names = pair_names(entry)
            folder = images_dir / sample_id

            missing = [n for n in names if not (folder / n).is_file()]
            if missing:
                problems.append(f"{sample_id} ({split}): immagini assenti su disco {missing}")
                continue
            views = sorted(_view_of(n) for n in names)
            if views != ["frontal", "lateral"]:
                problems.append(
                    f"{sample_id} ({split}): coppia non frontale+laterale {names}"
                )
            if not text.has_impression(sample_id, "en"):
                problems.append(f"{sample_id} ({split}): impression assente")
    return problems


def build_version(
    version: Version,
    annotation: Mapping[str, Any],
    text: TextSource,
    images_dir: Path,
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    prompts = get_prompts(version.views, version.target, version.lang)
    method = "passthrough" if version.lang == "en" else "csv_it"

    records_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        for entry in annotation.get(split, []):
            sample_id = str(entry["id"])
            folder = images_dir / sample_id
            names, projections = select_images(pair_names(entry), version.views)

            rel_images: list[str] = []
            for name in names:
                destination = f"images_normalized/{sample_id}/{name}"
                normalize_image(folder / name, output_dir / destination)
                rel_images.append(destination)

            findings, impression = text.report(sample_id, version.lang)
            majors = text.mesh_majors(sample_id)
            record = build_record(
                uid=sample_id,
                categories=map_mesh_to_chexpert(majors),
                mesh_raw=majors,
                projections=projections,
                rel_images=rel_images,
                assistant_text=format_target(
                    findings, impression, version.target, version.lang
                ),
                system_prompt=prompts.system,
                user_prompt=prompts.user,
                source_lang="en",
                target_lang=version.lang,
                translation_method=method,
            )
            record["r2gen_report"] = entry["report"]
            records_by_split[split].append(record)
    return records_by_split


CODE_FILES = (
    "src/shield/data/build.py",
    "src/shield/data/images.py",
    "src/shield/data/openi.py",
    "src/shield/data/prompts.py",
    "src/shield/data/records.py",
    "src/shield/data/text_source.py",
    "src/shield/tracking/__init__.py",
    "src/shield/tracking/core.py",
    "src/shield/tracking/provenance.py",
)


def integrity_section(
    version: Version,
    output_dir: Path,
    project_root: Path,
    annotation_path: Path,
    csv_path: Path,
    images_dir: Path,
) -> dict[str, Any]:
    from ..tracking import sha256_file, sha256_text, sha256_tree

    prompts = get_prompts(version.views, version.target, version.lang)
    return {
        "inputs": {
            "annotation": sha256_file(annotation_path),
            "translations_csv": sha256_file(csv_path),
            "labeled_images": sha256_tree(images_dir, "*.png"),
            "uv_lock": sha256_file(project_root / "uv.lock"),
        },
        "code": {rel: sha256_file(project_root / rel) for rel in CODE_FILES},
        "prompts": {
            "system": sha256_text(prompts.system),
            "user": sha256_text(prompts.user),
        },
        "outputs": {
            **{
                f"{split}.jsonl": sha256_file(output_dir / f"{split}.jsonl")
                for split in SPLITS
                if (output_dir / f"{split}.jsonl").is_file()
            },
            "images_normalized": sha256_tree(output_dir / "images_normalized", "*.png"),
        },
    }


def write_version(
    name: str,
    version: Version,
    records_by_split: Mapping[str, list[dict[str, Any]]],
    output_dir: Path,
    project_root: Path,
    annotation_path: Path,
    csv_path: Path,
    images_dir: Path,
) -> dict[str, Any]:
    import yaml

    from ..tracking import git_metadata

    for split, records in records_by_split.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = compute_stats(records_by_split)
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    prompts = get_prompts(version.views, version.target, version.lang)
    manifest = {
        "dataset": "iu-xray",
        "version": name,
        "views": version.views,
        "target": version.target,
        "lang": version.lang,
        "split_source": "R2Gen annotation.json (etichettato a mano, ripulito a monte)",
        "text_source": "iu_xray_translated.csv",
        "n_examples": stats["n_examples"],
        "prompts": {"system": prompts.system, "user": prompts.user},
        "provenance": git_metadata(project_root),
        "integrity": integrity_section(
            version, output_dir, project_root, annotation_path, csv_path, images_dir
        ),
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument("--version", action="append", choices=sorted(VERSIONS))
    parser.add_argument("--all", action="store_true", help="costruisci tutte le 8 versioni")
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("dataset/iu-xray/iu_xray_r2gen_final/annotation.json"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("dataset/iu-xray/iu_xray_r2gen_final/images"),
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("dataset/iu-xray/iu_xray_translated.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("dataset/iu-xray"))
    args = parser.parse_args(argv)

    root = args.root.resolve()
    wanted = sorted(VERSIONS) if args.all else (args.version or [])
    if not wanted:
        parser.error("indica --version oppure --all")

    annotation = json.loads((root / args.annotation).read_text(encoding="utf-8"))
    text = TextSource(root / args.csv)
    images_dir = root / args.images

    problems = validate_annotation(annotation, text, images_dir)
    if problems:
        print(
            f"[build] annotation non adatta: {len(problems)} studi da sistemare "
            f"prima di generare le versioni (nessun output prodotto)",
            file=sys.stderr,
        )
        for problem in problems[:30]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 30:
            print(f"  … e altri {len(problems) - 30}", file=sys.stderr)
        return 1

    n_studies = sum(len(annotation.get(split, [])) for split in SPLITS)
    counts = "  ".join(f"{s}={len(annotation.get(s, []))}" for s in SPLITS)
    print(f"[build] annotation: {n_studies} studi   {counts}", file=sys.stderr)

    for name in wanted:
        version = VERSIONS[name]
        output_dir = root / args.out / version_relpath(version)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[build] {name}  views={version.views}  target={version.target}  "
            f"lang={version.lang}",
            file=sys.stderr,
        )
        records_by_split = build_version(version, annotation, text, images_dir, output_dir)
        stats = write_version(
            name, version, records_by_split, output_dir, root,
            root / args.annotation, root / args.csv, images_dir,
        )
        written = "  ".join(f"{s}={n}" for s, n in stats["n_examples"].items())
        print(f"         {written}   → {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
