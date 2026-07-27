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


def view_of(image_path: str):
    low = Path(image_path).name.lower()
    hits = [v for v in ("frontal", "lateral") if v in low]
    return hits[0] if len(hits) == 1 else None


def classify(image_paths):
    views = [view_of(p) for p in image_paths]
    n_f = views.count("frontal")
    n_l = views.count("lateral")
    n_u = views.count(None)

    if n_u:
        return "UNCLASSIFIED", n_f, n_l, n_u
    if n_f == 1 and n_l == 1:
        return "OK", n_f, n_l, n_u
    if len(image_paths) != 2:
        return "WRONG_COUNT", n_f, n_l, n_u
    if n_f == 2 or n_l == 2:
        return "SAME_VIEW", n_f, n_l, n_u
    if n_f == 0:
        return "MISSING_FRONTAL", n_f, n_l, n_u
    if n_l == 0:
        return "MISSING_LATERAL", n_f, n_l, n_u
    return "DUPLICATE_VIEW", n_f, n_l, n_u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="dataset di input (deve contenere annotation.json)")
    ap.add_argument("--annotation", type=Path,
                    help="annotation.json da usare (default: <dataset>/annotation.json)")
    ap.add_argument("--csv-out", type=Path,
                    help="CSV col dettaglio di TUTTI i sample analizzati")
    ap.add_argument("--list-out", type=Path,
                    help="file di testo con i soli id non conformi (uno per riga)")
    ap.add_argument("--limit", type=int, help="analizza solo i primi N sample")
    ap.add_argument("--show", type=int, default=30,
                    help="quanti sample non conformi elencare a video (0 = tutti)")
    ap.add_argument("--table-out", type=Path,
                    default=Path(__file__).resolve().parent / "out"
                            / "annotation_views_summary.md",
                    help="file Markdown col riepilogo "
                         "(default: out/annotation_views_summary.md)")
    args = ap.parse_args()

    ann_path = args.annotation or (args.dataset / "annotation.json")
    if not ann_path.exists():
        sys.exit(f"Percorso non trovato: {ann_path}")

    annotation = json.load(ann_path.open())
    records = [(rec, rec.get("split", split))
               for split, recs in annotation.items() for rec in recs]
    if args.limit:
        records = records[: args.limit]

    rows = []
    for rec, split in records:
        paths = rec["image_path"]
        status, n_f, n_l, n_u = classify(paths)
        rows.append(dict(id=rec["id"], split=split, status=status,
                         n_images=len(paths), n_frontal=n_f, n_lateral=n_l,
                         n_unclassified=n_u,
                         images="|".join(Path(p).name for p in paths)))

    bad = [r for r in rows if r["status"] != "OK"]
    total = len(rows)
    counts = Counter(r["status"] for r in rows)

    overview = [
        ("Annotation", str(ann_path)),
        ("Sample analizzati", total),
        ("Conformi (1 frontal + 1 lateral)",
         f"{counts['OK']}/{total} ({100 * counts['OK'] / total:.2f}%)"),
        ("NON conformi", f"{len(bad)}/{total} ({100 * len(bad) / total:.2f}%)"),
    ]
    status_rows = [(s, n, f"{100 * n / total:.2f}%") for s, n in counts.most_common()]
    status_rows.append(("TOTALE", total, "100.00%"))

    tables = [("Riepilogo", ["Metrica", "Valore"], overview, {1}),
              ("Esiti", ["Status", "N", "%"], status_rows, {1, 2})]

    splits = sorted({r["split"] for r in rows if r["split"]})
    if len(splits) > 1:
        per_split = Counter((r["split"], r["status"] != "OK") for r in rows)
        split_rows = []
        for s in splits:
            tot_s = per_split[(s, True)] + per_split[(s, False)]
            bad_s = per_split[(s, True)]
            split_rows.append((s, bad_s, tot_s, f"{100 * bad_s / tot_s:.2f}%"))
        tables.append(("Non conformi per split",
                       ["Split", "Non conformi", "Totale", "%"], split_rows, {1, 2, 3}))

    emit("CHECK ANNOTATION: 1 immagine frontal + 1 lateral per sample",
         tables, args.table_out)

    if bad:
        shown = bad if not args.show else bad[: args.show]
        print(f"\n--- sample non conformi ({len(bad)}) ---")
        for r in shown:
            print(f"  {r['id']:<26} {r['status']:<16} "
                  f"F={r['n_frontal']} L={r['n_lateral']}  {r['images']}")
        if len(shown) < len(bad):
            print(f"  ... e altri {len(bad) - len(shown)} "
                  f"(--show 0 per tutti, oppure --csv-out / --list-out)")
    else:
        print("\nTutti i sample hanno esattamente 1 frontale e 1 laterale.")

    if args.list_out:
        args.list_out.parent.mkdir(parents=True, exist_ok=True)
        args.list_out.write_text("\n".join(r["id"] for r in bad) + "\n", encoding="utf-8")
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
