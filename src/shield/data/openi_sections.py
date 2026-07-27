from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .openi import iter_reports, report_paths

SPLITS = ("train", "val", "test")


def openi_study_uid(sample_id: str) -> str:
    return sample_id.split("_", 1)[0]


def extract_sections(reports_dir: Path) -> list[dict[str, str]]:
    total = len(report_paths(reports_dir))
    if not total:
        raise FileNotFoundError(f"Nessun XML in {reports_dir}")

    sections: list[dict[str, str]] = []
    for index, study in enumerate(iter_reports(reports_dir), start=1):
        sections.append(
            {
                "id": study.uid,
                "report_file": study.report_file,
                "findings": study.findings,
                "impression": study.impression,
            }
        )
        if index % 1000 == 0:
            print(f"  … {index}/{total} referti", file=sys.stderr)
    return sections


def annotation_uids(annotation: Mapping[str, Any]) -> set[str]:
    return {
        openi_study_uid(str(record["id"]))
        for split in SPLITS
        for record in annotation.get(split, [])
    }


def filter_to_annotation(
    sections: Sequence[Mapping[str, str]], annotation: Mapping[str, Any]
) -> list[dict[str, str]]:
    wanted = annotation_uids(annotation)
    return [dict(section) for section in sections if section["id"] in wanted]


def build_annotation_with_impression(
    annotation: Mapping[str, Any], sections: Sequence[Mapping[str, str]]
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    impression_by_uid = {
        section["id"]: section["impression"]
        for section in sections
        if section["impression"]
    }

    enriched: dict[str, Any] = {split: [] for split in SPLITS}
    dropped: dict[str, list[str]] = {split: [] for split in SPLITS}

    for split in SPLITS:
        for record in annotation.get(split, []):
            sample_id = str(record["id"])
            impression = impression_by_uid.get(openi_study_uid(sample_id))
            if not impression:
                dropped[split].append(sample_id)
                continue
            enriched[split].append({**record, "impression": impression})
    return enriched, dropped


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--reports", type=Path, default=Path("dataset/iu-xray/iu_xray_original/reports")
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("dataset/iu-xray/iu_xray_r2gen/annotation.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("dataset/iu-xray/impression"))
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_dir = root / args.out
    annotation = json.loads((root / args.annotation).read_text(encoding="utf-8"))

    print("[1/3] estrazione sezioni da OpenI…", file=sys.stderr)
    sections = extract_sections(root / args.reports)
    _write_json(out_dir / "openi_sections.json", sections)
    with_impression = sum(1 for s in sections if s["impression"])
    print(
        f"      {len(sections)} referti, di cui {with_impression} con impression",
        file=sys.stderr,
    )

    print("[2/3] filtro sugli id dello split R2Gen…", file=sys.stderr)
    kept = filter_to_annotation(sections, annotation)
    _write_json(out_dir / "openi_sections_r2gen.json", kept)
    print(f"      {len(sections)} → {len(kept)} referti", file=sys.stderr)

    print("[3/3] innesto dell'impression nell'annotation…", file=sys.stderr)
    enriched, dropped = build_annotation_with_impression(annotation, kept)
    _write_json(out_dir / "annotation_with_impression.json", enriched)

    n_before = sum(len(annotation.get(split, [])) for split in SPLITS)
    n_after = sum(len(enriched[split]) for split in SPLITS)
    n_dropped = sum(len(ids) for ids in dropped.values())
    _write_json(out_dir / "dropped_no_impression.json", dropped)

    print("", file=sys.stderr)
    print(f"  {'split':<8}{'prima':>8}{'dopo':>8}{'scartati':>10}", file=sys.stderr)
    for split in SPLITS:
        print(
            f"  {split:<8}{len(annotation.get(split, [])):>8}"
            f"{len(enriched[split]):>8}{len(dropped[split]):>10}",
            file=sys.stderr,
        )
    print(f"  {'TOTALE':<8}{n_before:>8}{n_after:>8}{n_dropped:>10}", file=sys.stderr)
    if n_dropped:
        print("", file=sys.stderr)
        print("  scartati (impression assente anche in OpenI):", file=sys.stderr)
        for split in SPLITS:
            for sample_id in dropped[split]:
                print(f"    {sample_id:<24}{split}", file=sys.stderr)
    print(f"\n  output: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
