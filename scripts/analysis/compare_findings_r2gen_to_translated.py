#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from _table import emit

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "dataset" / "iu-xray"

ID_RE = re.compile(r"^CXR(\d+)_")


def extract_uid(annotation_id: str):
    m = ID_RE.match(annotation_id)
    return m.group(1) if m else None


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_stats(a: str, b: str):
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0, 1.0
    inter, union = ta & tb, ta | tb
    jaccard = len(inter) / len(union) if union else 0.0
    coverage = len(inter) / len(ta) if ta else 0.0  
    return jaccard, coverage


def classify(csv_findings: str, ann_report: str):
    if not csv_findings.strip():
        return "NO_FINDINGS_CSV", 0.0, 0.0
    nc, na = normalize(csv_findings), normalize(ann_report)
    if nc == na:
        return "EXACT", 1.0, 1.0
    jac, cov = token_stats(nc, na)
    if nc and nc in na:
        return "SUBSET", jac, cov
    if jac >= 0.9:
        return "NEAR", jac, cov
    if jac >= 0.5:
        return "PARTIAL", jac, cov
    return "MISMATCH", jac, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="cartella che contiene iu_xray_translated.csv e iu_xray_r2gen/")
    ap.add_argument("--csv-out", type=Path, help="scrive il dettaglio per-report su CSV")
    ap.add_argument("--limit", type=int, help="analizza solo i primi N record")
    ap.add_argument("--show", type=int, default=5,
                    help="quanti esempi stampare per ogni categoria non-EXACT")
    ap.add_argument("--table-out", type=Path,
                    default=Path(__file__).resolve().parent / "out" / "findings_r2gen_to_translated_summary.md",
                    help="file Markdown con le tabelle di riepilogo "
                         "(default: out/findings_r2gen_to_translated_summary.md; '' per non scriverlo)")
    args = ap.parse_args()

    csv_path = args.dataset / "iu_xray_translated.csv"
    ann_path = args.dataset / "iu_xray_r2gen" / "annotation.json"
    for p in (csv_path, ann_path):
        if not p.exists():
            sys.exit(f"Percorso non trovato: {p}")

    with csv_path.open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    by_uid = {row["uid"].strip(): row for row in csv_rows}

    annotation = json.load(ann_path.open())
    records = [r for split in annotation.values() for r in split]
    if args.limit:
        records = records[: args.limit]

    rows = []
    for rec in records:
        uid = extract_uid(rec["id"])
        base = dict(id=rec["id"], split=rec.get("split", ""), uid=uid or "",
                    status="", jaccard=0.0, coverage=0.0,
                    has_findings_it=False, csv_findings="",
                    annotation_report=rec["report"])
        if uid is None:
            base["status"] = "BAD_ID"
            rows.append(base)
            continue
        row = by_uid.get(uid)
        if row is None:
            base["status"] = "UID_MISSING_CSV"
            rows.append(base)
            continue

        findings = row.get("findings", "") or ""
        status, jac, cov = classify(findings, rec["report"])
        base.update(status=status, jaccard=round(jac, 4), coverage=round(cov, 4),
                    has_findings_it=bool((row.get("findings_it") or "").strip()),
                    csv_findings=findings)
        rows.append(base)

    total = len(rows)
    counts = Counter(r["status"] for r in rows)
    matched = counts["EXACT"] + counts["SUBSET"] + counts["NEAR"]
    with_it = sum(1 for r in rows if r["has_findings_it"])
    jacs = [r["jaccard"] for r in rows if r["status"] not in
            ("UID_MISSING_CSV", "BAD_ID", "NO_FINDINGS_CSV")]
    js = sorted(jacs)

    overview = [
        ("Record annotation", total),
        ("Righe nel CSV", len(csv_rows)),
        ("UID distinti nel CSV", len(by_uid)),
        ("UID distinti annotation", len({r["uid"] for r in rows if r["uid"]})),
        ("MATCH (EXACT+SUBSET+NEAR)", f"{matched}/{total} ({100 * matched / total:.2f}%)"),
        ("Con findings_it non vuoto", f"{with_it}/{total} ({100 * with_it / total:.2f}%)"),
    ]
    if jacs:
        overview += [
            ("Jaccard medio", f"{sum(jacs) / len(jacs):.4f}"),
            ("Jaccard mediana", f"{js[len(js) // 2]:.4f}"),
        ]

    status_rows = [(s, n, f"{100 * n / total:.2f}%") for s, n in counts.most_common()]
    status_rows.append(("TOTALE", total, "100.00%"))

    tables = [("Riepilogo", ["Metrica", "Valore"], overview, {1}),
              ("Esiti del confronto", ["Status", "N", "%"], status_rows, {1, 2})]

    per_split = Counter((r["split"], r["status"]) for r in rows)
    splits = sorted({r["split"] for r in rows})
    if len(splits) > 1:
        split_rows = []
        for s in splits:
            tot_s = sum(v for (sp, _), v in per_split.items() if sp == s)
            ok_s = sum(per_split[(s, k)] for k in ("EXACT", "SUBSET", "NEAR"))
            split_rows.append((s, ok_s, tot_s, f"{100 * ok_s / tot_s:.2f}%"))
        tables.append(("Per split", ["Split", "Match", "Totale", "%"],
                       split_rows, {1, 2, 3}))

    emit("MATCH FINDINGS: annotation.json vs iu_xray_translated.csv", tables,
         args.table_out)

    if args.show:
        for status in ("MISMATCH", "PARTIAL", "NEAR", "SUBSET",
                       "NO_FINDINGS_CSV", "UID_MISSING_CSV", "BAD_ID"):
            examples = [r for r in rows if r["status"] == status][: args.show]
            if not examples:
                continue
            print(f"\n--- esempi {status} ({counts[status]} totali) ---")
            for r in examples:
                print(f"[{r['id']}] uid={r['uid']} jaccard={r['jaccard']}")
                print(f"  CSV  : {r['csv_findings'][:200] or '<vuoto>'}")
                print(f"  ANNOT: {r['annotation_report'][:200]}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV dettagliato scritto in: {args.csv_out}")


if __name__ == "__main__":
    main()
