#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from _table import emit

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "iu-xray"


def load_annotation(dataset_dir: Path) -> dict:
    path = dataset_dir / "annotation.json"
    if not path.exists():
        sys.exit(f"annotation.json non trovato in {dataset_dir}")
    data = json.load(path.open())
    out = {}
    for split, records in data.items():
        for rec in records:
            out[rec["id"]] = {"image_path": rec["image_path"],
                              "split": rec.get("split", split),
                              "report": rec.get("report", "")}
    return out


def pixel_digest(path: Path):
    try:
        with Image.open(path) as im:
            im.load()
            arr = np.asarray(im)
            mode = im.mode
    except FileNotFoundError:
        return None, "FILE_MISSING", ""
    except Exception as exc:
        return None, f"UNREADABLE:{type(exc).__name__}", ""
    key = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
    return f"{mode}|{arr.shape}|{key}", "x".join(map(str, arr.shape)), mode


def diff_stats(pa: Path, pb: Path):
    try:
        with Image.open(pa) as ia, Image.open(pb) as ib:
            ia.load(); ib.load()
            a, b = np.asarray(ia), np.asarray(ib)
    except Exception:
        return "", ""
    if a.shape != b.shape:
        return "", ""
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    pct = (d.any(axis=-1) if d.ndim == 3 else d != 0).mean()
    return int(d.max()), round(100.0 * float(pct), 6)


def compare_sample(job):
    sid, split, paths_a, paths_b = job
    base = dict(id=sid, split=split, n_images_a=len(paths_a), n_images_b=len(paths_b),
                images_a="|".join(p.name for p in paths_a),
                images_b="|".join(p.name for p in paths_b),
                max_abs_diff="", pct_pixels_diff="", detail="")

    if len(paths_a) != len(paths_b):
        return dict(base, status="COUNT_MISMATCH")

    digests_a, digests_b, problems = [], [], []
    for tag, paths, sink in (("A", paths_a, digests_a), ("B", paths_b, digests_b)):
        for p in paths:
            dig, info, _ = pixel_digest(p)
            if dig is None:
                problems.append(f"{tag}:{p.name}:{info}")
            sink.append(dig)
    if problems:
        status = "FILE_MISSING" if all("FILE_MISSING" in x for x in problems) else "UNREADABLE"
        return dict(base, status=status, detail=";".join(problems))

    if digests_a == digests_b:
        return dict(base, status="IDENTICAL")

    if sorted(digests_a) == sorted(digests_b):
        order = [digests_a.index(d) for d in digests_b]
        return dict(base, status="PERMUTED",
                    detail="B[i] = A[" + ",".join(map(str, order)) + "]")

    bad = [i for i, (x, y) in enumerate(zip(digests_a, digests_b)) if x != y]
    mx, pct = diff_stats(paths_a[bad[0]], paths_b[bad[0]])
    return dict(base, status="CONTENT_MISMATCH", max_abs_diff=mx, pct_pixels_diff=pct,
                detail="posizioni diverse: " + ",".join(map(str, bad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=DEFAULT_ROOT / "iu_xray_r2gen",
                    help="dataset A (annotation.json + images/)")
    ap.add_argument("--b", type=Path, default=DEFAULT_ROOT / "iu_xray_r2gen_labeled",
                    help="dataset B (annotation.json + images/)")
    ap.add_argument("--csv-out", type=Path, help="dettaglio per-sample su CSV")
    ap.add_argument("--limit", type=int, help="analizza solo i primi N sample")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--show", type=int, default=5, help="esempi per categoria")
    ap.add_argument("--table-out", type=Path,
                    default=Path(__file__).resolve().parent / "out" / "annotation_images_r2gen_to_labeled_summary.md",
                    help="file Markdown con le tabelle di riepilogo "
                         "(default: out/annotation_images_r2gen_to_labeled_summary.md; '' per non scriverlo)")
    args = ap.parse_args()

    ann_a, ann_b = load_annotation(args.a), load_annotation(args.b)
    ids_a, ids_b = set(ann_a), set(ann_b)
    common = sorted(ids_a & ids_b)
    if args.limit:
        common = common[: args.limit]

    jobs = [(sid, ann_a[sid]["split"],
             [args.a / "images" / rel for rel in ann_a[sid]["image_path"]],
             [args.b / "images" / rel for rel in ann_b[sid]["image_path"]])
            for sid in common]

    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(compare_sample, jobs, chunksize=8))
    else:
        rows = [compare_sample(j) for j in jobs]

    for sid in sorted(ids_a - ids_b):
        rows.append(dict(id=sid, split=ann_a[sid]["split"], status="ID_MISSING_B",
                         n_images_a=len(ann_a[sid]["image_path"]), n_images_b=0,
                         images_a="|".join(ann_a[sid]["image_path"]), images_b="",
                         max_abs_diff="", pct_pixels_diff="", detail=""))
    for sid in sorted(ids_b - ids_a):
        rows.append(dict(id=sid, split=ann_b[sid]["split"], status="ID_MISSING_A",
                         n_images_a=0, n_images_b=len(ann_b[sid]["image_path"]),
                         images_a="", images_b="|".join(ann_b[sid]["image_path"]),
                         max_abs_diff="", pct_pixels_diff="", detail=""))

    total = len(rows)
    n_imgs = sum(r["n_images_a"] for r in rows)
    counts = Counter(r["status"] for r in rows)
    ok = counts["IDENTICAL"] + counts["PERMUTED"]
    diff_reports = [sid for sid in common
                    if ann_a[sid]["report"] != ann_b[sid]["report"]]

    overview = [
        ("Dataset A", str(args.a)),
        ("Dataset B", str(args.b)),
        ("Sample in A", len(ann_a)),
        ("Sample in B", len(ann_b)),
        ("Sample in comune", len(common)),
        ("Immagini confrontate", n_imgs),
        ("Sample pixel-identici", f"{ok}/{total} ({100 * ok / total:.2f}%)"),
        ("Sample con report diverso", len(diff_reports)),
    ]

    status_rows = [(s, n, f"{100 * n / total:.2f}%") for s, n in counts.most_common()]
    status_rows.append(("TOTALE", total, "100.00%"))

    split_rows = []
    per_split = Counter((r["split"], r["status"]) for r in rows)
    for s in sorted({r["split"] for r in rows}):
        tot_s = sum(v for (sp, _), v in per_split.items() if sp == s)
        ok_s = per_split[(s, "IDENTICAL")] + per_split[(s, "PERMUTED")]
        split_rows.append((s, ok_s, tot_s, f"{100 * ok_s / tot_s:.2f}%"))

    emit("CONFRONTO PIXEL PER PIXEL DELLE IMMAGINI REFERENZIATE DAI DUE ANNOTATION",
         [("Riepilogo", ["Metrica", "Valore"], overview, {1}),
          ("Esiti del confronto", ["Status", "N", "%"], status_rows, {1, 2}),
          ("Per split", ["Split", "Identici", "Totale", "%"], split_rows, {1, 2, 3})],
         args.table_out)

    if args.show:
        for status in [s for s in counts if s != "IDENTICAL"]:
            examples = [r for r in rows if r["status"] == status][: args.show]
            print(f"\n--- esempi {status} ({counts[status]} totali) ---")
            for r in examples:
                print(f"[{r['id']}] A={r['images_a']}  B={r['images_b']}  {r['detail']}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["id", "split", "status", "n_images_a", "n_images_b", "images_a",
                  "images_b", "max_abs_diff", "pct_pixels_diff", "detail"]
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV dettagliato scritto in: {args.csv_out}")


if __name__ == "__main__":
    main()
