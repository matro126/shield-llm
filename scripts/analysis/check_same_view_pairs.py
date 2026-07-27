#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from _table import emit

DEFAULT_DATASET = (Path(__file__).resolve().parents[2] / "dataset" / "iu-xray"
                   / "iu_xray_r2gen_labeled")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def view_of(filename: str):
    low = filename.lower()
    hits = [v for v in ("frontal", "lateral") if v in low]
    return hits[0] if len(hits) == 1 else None


def classify(images):
    if len(images) != 2:
        return "OTHER", []
    views = [view_of(p.name) for p in images]
    if None in views:
        return "OTHER", views
    if views[0] != views[1]:
        return "MIXED_PAIR", views
    return ("FRONTAL_PAIR" if views[0] == "frontal" else "LATERAL_PAIR"), views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="dataset di input (deve contenere annotation.json e images/)")
    ap.add_argument("--scan-dirs", action="store_true",
                    help="ignora l'annotation e analizza tutte le cartelle in images/")
    ap.add_argument("--csv-out", type=Path,
                    help="CSV con il dettaglio di TUTTI i sample analizzati")
    ap.add_argument("--list-out", type=Path,
                    help="file di testo con i soli id problematici (uno per riga)")
    ap.add_argument("--limit", type=int, help="analizza solo i primi N sample")
    ap.add_argument("--show", type=int, default=30,
                    help="quanti sample elencare a video (0 = tutti)")
    ap.add_argument("--table-out", type=Path,
                    default=Path(__file__).resolve().parent / "out"
                            / "same_view_pairs_summary.md",
                    help="file Markdown col riepilogo (default: out/same_view_pairs_summary.md)")
    args = ap.parse_args()

    images_dir = args.dataset / "images"
    if not images_dir.is_dir():
        sys.exit(f"Cartella immagini non trovata: {images_dir}")

    ann_path = args.dataset / "annotation.json"
    if args.scan_dirs or not ann_path.exists():
        samples = [(p.name, "") for p in sorted(images_dir.iterdir()) if p.is_dir()]
        source = "cartelle in images/"
    else:
        annotation = json.load(ann_path.open())
        samples = [(rec["id"], rec.get("split", split))
                   for split, recs in annotation.items() for rec in recs]
        samples.sort()
        source = str(ann_path)
    if args.limit:
        samples = samples[: args.limit]

    rows = []
    for sid, split in samples:
        d = images_dir / sid
        if not d.is_dir():
            rows.append(dict(id=sid, split=split, status="DIR_MISSING",
                             n_images=0, images="", views=""))
            continue
        images = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        status, views = classify(images)
        rows.append(dict(id=sid, split=split, status=status, n_images=len(images),
                         images="|".join(p.name for p in images),
                         views="|".join(v or "?" for v in views)))

    flagged = [r for r in rows if r["status"] in ("FRONTAL_PAIR", "LATERAL_PAIR")]
    total = len(rows)
    counts = Counter(r["status"] for r in rows)

    overview = [
        ("Dataset", str(args.dataset)),
        ("Sample presi da", source),
        ("Sample analizzati", total),
        ("Coppie stessa vista",
         f"{len(flagged)}/{total} ({100 * len(flagged) / total:.2f}%)"),
        ("  di cui frontal+frontal_2", counts["FRONTAL_PAIR"]),
        ("  di cui lateral+lateral_2", counts["LATERAL_PAIR"]),
    ]
    status_rows = [(s, n, f"{100 * n / total:.2f}%") for s, n in counts.most_common()]
    status_rows.append(("TOTALE", total, "100.00%"))

    tables = [("Riepilogo", ["Metrica", "Valore"], overview, {1}),
              ("Composizione cartelle", ["Status", "N", "%"], status_rows, {1, 2})]

    splits = sorted({r["split"] for r in rows if r["split"]})
    if len(splits) > 1:
        per_split = Counter((r["split"], r["status"] in ("FRONTAL_PAIR", "LATERAL_PAIR"))
                            for r in rows)
        split_rows = []
        for s in splits:
            tot_s = per_split[(s, True)] + per_split[(s, False)]
            bad_s = per_split[(s, True)]
            split_rows.append((s, bad_s, tot_s, f"{100 * bad_s / tot_s:.2f}%"))
        tables.append(("Coppie stessa vista per split",
                       ["Split", "Coppie", "Totale", "%"], split_rows, {1, 2, 3}))

    emit("SAMPLE CON DUE IMMAGINI DELLA STESSA VISTA", tables, args.table_out)

    if flagged:
        shown = flagged if not args.show else flagged[: args.show]
        print(f"\n--- sample con due immagini della stessa vista ({len(flagged)}) ---")
        for r in shown:
            print(f"  {r['id']:<26} {r['status']:<13} {r['images']}")
        if len(shown) < len(flagged):
            print(f"  ... e altri {len(flagged) - len(shown)} "
                  f"(--show 0 per vederli tutti, oppure --csv-out / --list-out)")
    else:
        print("\nNessun sample con due immagini della stessa vista.")

    if args.list_out:
        args.list_out.parent.mkdir(parents=True, exist_ok=True)
        args.list_out.write_text("\n".join(r["id"] for r in flagged) + "\n",
                                 encoding="utf-8")
        print(f"\nLista id scritta in: {args.list_out}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV dettagliato scritto in: {args.csv_out}")


if __name__ == "__main__":
    main()
