#!/usr/bin/env python3
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _table import emit  # noqa: E402

DEFAULT_DATASET = (Path(__file__).resolve().parents[2] / "dataset" / "iu-xray"
                   / "iu_xray_r2gen_labeled")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CATEGORIES = ("frontal", "lateral")


def categorize(filename: str):
    low = filename.lower()
    hits = [c for c in CATEGORIES if c in low]
    if len(hits) == 1:
        return hits[0], None
    if not hits:
        return None, "NESSUNA_CATEGORIA"
    return None, "AMBIGUO(frontal+lateral)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="dataset di input (deve contenere images/<ID>/...)")
    ap.add_argument("--out", type=Path,
                    help="cartella di output (default: <dataset>/labeling_check)")
    ap.add_argument("--dry-run", action="store_true",
                    help="non copia nulla, mostra solo il riepilogo")
    ap.add_argument("--overwrite", action="store_true",
                    help="sovrascrive i file gia' presenti nell'output")
    ap.add_argument("--show", type=int, default=10,
                    help="quanti file non classificati elencare")
    ap.add_argument("--table-out", type=Path,
                    default=None,
                    help="file Markdown col riepilogo (default: <out>/summary.md)")
    args = ap.parse_args()

    images_dir = args.dataset / "images"
    if not images_dir.is_dir():
        sys.exit(f"Cartella immagini non trovata: {images_dir}")
    out_dir = args.out or (args.dataset / "labeling_check")
    table_out = args.table_out or (out_dir / "summary.md")

    if not args.dry_run:
        for cat in CATEGORIES:
            (out_dir / cat).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    per_category = Counter()
    skipped, collisions = [], []
    studies = sorted(p for p in images_dir.iterdir() if p.is_dir())

    for study in studies:
        for img in sorted(study.iterdir()):
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            counts["immagini_totali"] += 1
            cat, problem = categorize(img.name)
            if cat is None:
                counts["non_classificate"] += 1
                skipped.append((f"{study.name}/{img.name}", problem))
                continue

            dest = out_dir / cat / f"{study.name}__{img.name}"
            per_category[cat] += 1
            if dest.exists() and not args.overwrite:
                collisions.append(str(dest.relative_to(out_dir)))
                counts["gia_presenti"] += 1
                continue
            if not args.dry_run:
                shutil.copy2(img, dest)
                counts["copiate"] += 1

    total = counts["immagini_totali"]
    overview = [
        ("Dataset", str(args.dataset)),
        ("Output", str(out_dir) + (" (dry-run)" if args.dry_run else "")),
        ("Studi (cartelle)", len(studies)),
        ("Immagini totali", total),
        ("Classificate", total - counts["non_classificate"]),
        ("Non classificate", counts["non_classificate"]),
        ("Copiate", counts["copiate"]),
        ("Gia' presenti (saltate)", counts["gia_presenti"]),
    ]
    cat_rows = [(c, per_category[c],
                 f"{100 * per_category[c] / total:.2f}%" if total else "-")
                for c in CATEGORIES]
    cat_rows.append(("NON CLASSIFICATE", counts["non_classificate"],
                     f"{100 * counts['non_classificate'] / total:.2f}%" if total else "-"))
    cat_rows.append(("TOTALE", total, "100.00%" if total else "-"))

    emit("CHECK LABELING: copia piatta frontal/lateral",
         [("Riepilogo", ["Metrica", "Valore"], overview, {1}),
          ("Per categoria", ["Categoria", "N", "%"], cat_rows, {1, 2})],
         None if args.dry_run else table_out)

    if skipped and args.show:
        print(f"\n--- file non classificati ({len(skipped)} totali) ---")
        for name, problem in skipped[: args.show]:
            print(f"  {name}  [{problem}]")
    if collisions and args.show:
        print(f"\n--- file gia' presenti, non sovrascritti ({len(collisions)}) ---")
        for name in collisions[: args.show]:
            print(f"  {name}")
        print("  (usa --overwrite per rigenerarli)")


if __name__ == "__main__":
    main()
