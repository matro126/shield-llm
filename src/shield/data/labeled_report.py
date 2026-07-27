from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Study:
    sample_id: str
    split: str
    frontal: list[str]
    lateral: list[str]
    other: list[str]

    @property
    def n_images(self) -> int:
        return len(self.frontal) + len(self.lateral) + len(self.other)

    @property
    def has_frontal(self) -> bool:
        return bool(self.frontal)

    @property
    def has_lateral(self) -> bool:
        return bool(self.lateral)


def scan(images_dir: Path, splits: dict[str, str]) -> list[Study]:
    studies: list[Study] = []
    for folder in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        names = sorted(p.name for p in folder.glob("*.png"))
        frontal = [n for n in names if n.startswith("frontal")]
        lateral = [n for n in names if n.startswith("lateral")]
        other = [n for n in names if not n.startswith(("frontal", "lateral"))]
        studies.append(
            Study(folder.name, splits.get(folder.name, "-"), frontal, lateral, other)
        )
    return studies


def _table(title: str, rows: list[list[str]], headers: list[str]) -> list[str]:
    widths = [
        max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))
    ]
    line = "  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [title, "", line, "  " + "-" * (len(line) - 2)]
    for r in rows:
        out.append("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return out


def write_csv(path: Path, studies: list[Study]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "split", "n_images", "n_frontal", "n_lateral",
             "frontal_files", "lateral_files", "other_files"]
        )
        for s in studies:
            writer.writerow([
                s.sample_id, s.split, s.n_images, len(s.frontal), len(s.lateral),
                ";".join(s.frontal), ";".join(s.lateral), ";".join(s.other),
            ])


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--labeled",
        type=Path,
        default=Path("dataset/iu-xray/iu_xray_r2gen_labeled"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/labeled_report"))
    args = parser.parse_args(argv)

    labeled = args.root / args.labeled
    images_dir = labeled / "images"
    if not images_dir.is_dir():
        parser.error(f"cartella immagini non trovata: {images_dir}")

    annotation = json.loads((labeled / "annotation.json").read_text(encoding="utf-8"))
    splits = {r["id"]: s for s, rs in annotation.items() for r in rs}

    studies = scan(images_dir, splits)
    out_dir = args.root / args.out
    transcript: list[str] = []

    def emit(lines: list[str]) -> None:
        for line in lines:
            print(line)
            transcript.append(line)

    emit(["=" * 68, f"STUDI TOTALI: {len(studies)}", "=" * 68, ""])

    dist = Counter(s.n_images for s in studies)
    emit(_table(
        "distribuzione immagini per studio",
        [[n, dist[n]] for n in sorted(dist)],
        ["n_immagini", "studi"],
    ))

    multi = [s for s in studies if s.n_images > 2]
    emit(["", "=" * 68, f"1) STUDI CON PIU' DI 2 IMMAGINI: {len(multi)}", ""])
    emit(_table(
        "",
        [[s.sample_id, s.split, s.n_images, len(s.frontal), len(s.lateral)]
         for s in multi],
        ["sample_id", "split", "n_img", "frontali", "laterali"],
    ))

    one_view = [s for s in studies if not (s.has_frontal and s.has_lateral)]
    only_frontal = [s for s in one_view if s.has_frontal]
    only_lateral = [s for s in one_view if s.has_lateral]
    emit([
        "", "=" * 68,
        f"2) STUDI CON UNA SOLA VISTA: {len(one_view)}  "
        f"(solo frontale {len(only_frontal)}, solo laterale {len(only_lateral)})",
        "",
    ])
    emit(_table(
        "",
        [[s.sample_id, s.split, s.n_images,
          "solo frontale" if s.has_frontal else "solo laterale"]
         for s in one_view],
        ["sample_id", "split", "n_img", "vista"],
    ))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "all_studies.csv", studies)
    write_csv(out_dir / "more_than_2_images.csv", multi)
    write_csv(out_dir / "single_view.csv", one_view)
    (out_dir / "report.txt").write_text("\n".join(transcript) + "\n", encoding="utf-8")
    print(f"\noutput: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
