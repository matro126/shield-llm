#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _table import emit  # noqa: E402

DEFAULT_DATASET = (Path(__file__).resolve().parents[2] / "dataset" / "iu-xray"
                   / "iu_xray_r2gen_labeled")
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "same_view_samples"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def view_of(filename: str):
    low = Path(filename).name.lower()
    hits = [v for v in ("frontal", "lateral") if v in low]
    return hits[0] if len(hits) == 1 else None


def same_view(image_paths):
    if len(image_paths) != 2:
        return None
    a, b = (view_of(p) for p in image_paths)
    return a if a is not None and a == b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="dataset di input (annotation.json + images/)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"cartella di output (default: {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="non copia nulla, mostra solo cosa verrebbe estratto")
    ap.add_argument("--overwrite", action="store_true",
                    help="rimuove e ricrea le cartelle gia' presenti nell'output")
    ap.add_argument("--show", type=int, default=30,
                    help="quanti sample elencare a video (0 = tutti)")
    ap.add_argument("--table-out", type=Path, default=None,
                    help="file Markdown col riepilogo (default: <out>/summary.md)")
    args = ap.parse_args()

    images_dir = args.dataset / "images"
    ann_path = args.dataset / "annotation.json"
    for p in (images_dir, ann_path):
        if not p.exists():
            sys.exit(f"Percorso non trovato: {p}")
    table_out = args.table_out or (args.out / "summary.md")

    annotation = json.load(ann_path.open())
    records = [rec for split, recs in annotation.items() for rec in recs]

    counts = Counter()
    selected, only_two, missing_dir = [], [], []

    for rec in records:
        counts["sample_totali"] += 1
        view = same_view(rec["image_path"])
        if view is None:
            continue
        counts["stessa_vista"] += 1
        counts[f"stessa_vista_{view}"] += 1

        d = images_dir / rec["id"]
        if not d.is_dir():
            missing_dir.append(rec["id"])
            continue
        images = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) <= 2:
            only_two.append((rec["id"], view, len(images)))
            continue
        selected.append(dict(id=rec["id"], view=view, n_images=len(images),
                             src=d, images=[p.name for p in images],
                             annotation=[Path(p).name for p in rec["image_path"]]))

    copied = skipped = 0
    if not args.dry_run and selected:
        args.out.mkdir(parents=True, exist_ok=True)
    for s in selected:
        dest = args.out / s["id"]
        if dest.exists():
            if not args.overwrite:
                skipped += 1
                continue
            if not args.dry_run:
                shutil.rmtree(dest)
        if not args.dry_run:
            shutil.copytree(s["src"], dest)
            copied += 1

    overview = [
        ("Dataset", str(args.dataset)),
        ("Output", str(args.out) + (" (dry-run)" if args.dry_run else "")),
        ("Sample nell'annotation", counts["sample_totali"]),
        ("Con due immagini stessa vista", counts["stessa_vista"]),
        ("  frontal + frontal_2", counts["stessa_vista_frontal"]),
        ("  lateral + lateral_2", counts["stessa_vista_lateral"]),
        ("Esclusi (cartella con <= 2 immagini)", len(only_two)),
        ("ESTRATTI (cartella con > 2 immagini)", len(selected)),
        ("Cartelle copiate", copied),
        ("Gia' presenti, saltate", skipped),
    ]
    if missing_dir:
        overview.append(("Cartelle immagini mancanti", len(missing_dir)))

    dist = Counter(s["n_images"] for s in selected)
    dist_rows = [(n, dist[n]) for n in sorted(dist)]
    tables = [("Riepilogo", ["Metrica", "Valore"], overview, {1})]
    if dist_rows:
        tables.append(("Estratti per numero di immagini",
                       ["Immagini nella cartella", "Sample"], dist_rows, {0, 1}))

    emit("ESTRAZIONE SAMPLE CON DUE IMMAGINI DELLA STESSA VISTA",
         tables, None if args.dry_run else table_out)

    if selected:
        shown = selected if not args.show else selected[: args.show]
        print(f"\n--- sample estratti ({len(selected)}) ---")
        for s in shown:
            print(f"  {s['id']:<26} {s['view']:<8} annotation={'+'.join(s['annotation'])}"
                  f"  cartella({s['n_images']})={'|'.join(s['images'])}")
        if len(shown) < len(selected):
            print(f"  ... e altri {len(selected) - len(shown)} (--show 0 per tutti)")
    else:
        print("\nNessun sample da estrarre.")

    if only_two and args.show:
        print(f"\n--- esclusi: cartella con <= 2 immagini ({len(only_two)}) ---")
        for sid, view, n in only_two[: args.show]:
            print(f"  {sid:<26} {view:<8} {n} immagini")
    if missing_dir:
        print(f"\n--- cartella immagini mancante ({len(missing_dir)}) ---")
        for sid in missing_dir[: args.show or None]:
            print(f"  {sid}")


if __name__ == "__main__":
    main()
