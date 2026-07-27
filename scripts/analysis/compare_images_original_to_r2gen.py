#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from _table import emit

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "dataset" / "iu-xray"

ORIG_RE = re.compile(r"^(CXR.+)-([^-]+)\.png$", re.IGNORECASE)
PATIENT_RE = re.compile(r"^(CXR\d+(?:_\d+)?_IM-[A-Za-z0-9]+)", re.IGNORECASE)


def patient_id(name: str) -> str:
    m = PATIENT_RE.match(name)
    return m.group(1).upper() if m else name.upper()


def original_index(images_dir: Path):
    by_study = defaultdict(list)
    by_patient = defaultdict(list)
    for p in sorted(images_dir.glob("*.png")):
        m = ORIG_RE.match(p.name)
        if m:
            by_study[m.group(1)].append(p)
        by_patient[patient_id(p.name)].append(p)
    return dict(by_study), dict(by_patient)


def load_array(path: Path):
    with Image.open(path) as im:
        im.load()
        return np.asarray(im), im.mode


def digest(arr) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def diff_stats(a, b):
    a32 = a.astype(np.int32)
    b32 = b.astype(np.int32)
    d = np.abs(a32 - b32)
    return {
        "max_abs_diff": int(d.max()),
        "mean_abs_diff": round(float(d.mean()), 6),
        "pct_pixels_diff": round(100.0 * float((d.any(axis=-1) if d.ndim == 3
                                                else d != 0).mean()), 6),
    }


def compare_study(job):
    study_id, study_paths, extra_paths, r2_paths = job
    orig_paths = list(study_paths) + [p for p in extra_paths if p not in study_paths]
    study_set = set(study_paths)
    rows = []

    if not orig_paths:
        for rp in r2_paths:
            rows.append(dict(id=study_id, r2gen_image=rp.name, orig_image="",
                             status="ORIG_MISSING", r2gen_size="", orig_size="",
                             max_abs_diff="", mean_abs_diff="", pct_pixels_diff=""))
        return rows

    originals = []
    for op in orig_paths:
        try:
            arr, mode = load_array(op)
        except Exception as exc:
            rows.append(dict(id=study_id, r2gen_image="", orig_image=op.name,
                             status=f"ORIG_UNREADABLE:{type(exc).__name__}",
                             r2gen_size="", orig_size="", max_abs_diff="",
                             mean_abs_diff="", pct_pixels_diff=""))
            continue
        originals.append((op, arr, mode, digest(arr)))

    for rp in r2_paths:
        try:
            r_arr, r_mode = load_array(rp)
        except Exception as exc:
            rows.append(dict(id=study_id, r2gen_image=rp.name, orig_image="",
                             status=f"R2GEN_UNREADABLE:{type(exc).__name__}",
                             r2gen_size="", orig_size="", max_abs_diff="",
                             mean_abs_diff="", pct_pixels_diff=""))
            continue
        r_dig = digest(r_arr)
        r_size = "x".join(map(str, r_arr.shape))

        exact = next((o for o in originals
                      if o[1].shape == r_arr.shape and o[3] == r_dig), None)
        if exact is not None:
            rows.append(dict(id=study_id, r2gen_image=rp.name,
                             orig_image=exact[0].name,
                             status="EXACT" if exact[0] in study_set
                                    else "EXACT_PATIENT",
                             r2gen_size=r_size,
                             orig_size="x".join(map(str, exact[1].shape)),
                             max_abs_diff=0, mean_abs_diff=0.0,
                             pct_pixels_diff=0.0))
            continue

        same_shape = [o for o in originals if o[1].shape == r_arr.shape]
        if same_shape:
            best, best_stats = None, None
            for o in same_shape:
                st = diff_stats(r_arr, o[1])
                if best_stats is None or st["mean_abs_diff"] < best_stats["mean_abs_diff"]:
                    best, best_stats = o, st
            rows.append(dict(id=study_id, r2gen_image=rp.name,
                             orig_image=best[0].name, status="DIFF_PIXELS",
                             r2gen_size=r_size,
                             orig_size="x".join(map(str, best[1].shape)),
                             **best_stats))
        else:
            sizes = "|".join(sorted({"x".join(map(str, o[1].shape)) for o in originals}))
            rows.append(dict(id=study_id, r2gen_image=rp.name, orig_image="",
                             status="SHAPE_MISMATCH", r2gen_size=r_size,
                             orig_size=sizes, max_abs_diff="", mean_abs_diff="",
                             pct_pixels_diff=""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="cartella che contiene iu_xray_original/ e iu_xray_r2gen/")
    ap.add_argument("--csv-out", type=Path, help="scrive il dettaglio per-immagine su CSV")
    ap.add_argument("--limit", type=int, help="analizza solo i primi N studi")
    ap.add_argument("--workers", type=int, default=8, help="processi paralleli")
    ap.add_argument("--show", type=int, default=5, help="esempi per categoria non-EXACT")
    ap.add_argument("--table-out", type=Path,
                    default=Path(__file__).resolve().parent / "out" / "images_original_to_r2gen_summary.md",
                    help="file Markdown con le tabelle di riepilogo "
                         "(default: out/images_original_to_r2gen_summary.md; '' per non scriverlo)")
    args = ap.parse_args()

    orig_dir = args.dataset / "iu_xray_original" / "images"
    r2_dir = args.dataset / "iu_xray_r2gen" / "images"
    ann_path = args.dataset / "iu_xray_r2gen" / "annotation.json"
    for p in (orig_dir, r2_dir):
        if not p.exists():
            sys.exit(f"Percorso non trovato: {p}")

    orig_idx, orig_by_patient = original_index(orig_dir)

    if ann_path.exists():
        annotation = json.load(ann_path.open())
        study_ids = sorted({r["id"] for split in annotation.values() for r in split})
    else:
        study_ids = sorted(p.name for p in r2_dir.iterdir() if p.is_dir())
    if args.limit:
        study_ids = study_ids[: args.limit]

    jobs, missing_dirs = [], []
    for sid in study_ids:
        d = r2_dir / sid
        if not d.is_dir():
            missing_dirs.append(sid)
            continue
        jobs.append((sid, orig_idx.get(sid, []),
                     orig_by_patient.get(patient_id(sid), []),
                     sorted(d.glob("*.png"))))

    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(compare_study, jobs, chunksize=8):
                rows.extend(res)
    else:
        for job in jobs:
            rows.extend(compare_study(job))

    total = len(rows)
    counts = Counter(r["status"] for r in rows)
    ok = counts["EXACT"] + counts["EXACT_PATIENT"]

    per_study = defaultdict(list)
    for r in rows:
        per_study[r["id"]].append(r["status"])
    full_ok = sum(1 for v in per_study.values()
                  if all(s in ("EXACT", "EXACT_PATIENT") for s in v))

    used_ids = set(per_study)
    unused = sum(len(v) for k, v in orig_idx.items() if k not in used_ids)

    overview = [
        ("Studi confrontati", len(jobs)),
        ("Immagini R2Gen", total),
        ("Immagini originali (tot)", sum(len(v) for v in orig_idx.values())),
        ("ID originali distinti", len(orig_idx)),
        ("Immagini identiche", f"{ok}/{total} ({100 * ok / total:.2f}%)"),
        ("Studi 100% identici",
         f"{full_ok}/{len(per_study)} ({100 * full_ok / len(per_study):.2f}%)"),
        ("Originali non usati da R2Gen", unused),
    ]
    if missing_dirs:
        overview.append(("Cartelle R2Gen mancanti",
                         f"{len(missing_dirs)} (es. {', '.join(missing_dirs[:3])})"))

    status_rows = [(s, n, f"{100 * n / total:.2f}%") for s, n in counts.most_common()]
    status_rows.append(("TOTALE", total, "100.00%"))

    emit("MATCH IMMAGINI (pixel per pixel): iu_xray_r2gen vs iu_xray_original",
         [("Riepilogo", ["Metrica", "Valore"], overview, {1}),
          ("Esiti del confronto", ["Status", "N", "%"], status_rows, {1, 2})],
         args.table_out)

    if args.show:
        for status in [s for s in counts if s not in ("EXACT", "EXACT_PATIENT")]:
            examples = [r for r in rows if r["status"] == status][: args.show]
            print(f"\n--- esempi {status} ({counts[status]} totali) ---")
            for r in examples:
                print(f"[{r['id']}] {r['r2gen_image']} ({r['r2gen_size']}) vs "
                      f"{r['orig_image'] or '<nessuno>'} ({r['orig_size']}) "
                      f"mean_abs_diff={r['mean_abs_diff']} "
                      f"pct_diff={r['pct_pixels_diff']}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["id", "r2gen_image", "orig_image", "status", "r2gen_size",
                  "orig_size", "max_abs_diff", "mean_abs_diff", "pct_pixels_diff"]
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV dettagliato scritto in: {args.csv_out}")


if __name__ == "__main__":
    main()
