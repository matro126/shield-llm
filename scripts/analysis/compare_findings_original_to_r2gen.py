#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from _table import emit

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "dataset" / "iu-xray"

ID_RE = re.compile(r"^CXR(\d+)_")


def extract_uid(annotation_id: str):
    m = ID_RE.match(annotation_id)
    return m.group(1) if m else None


def parse_xml_report(path: Path) -> dict:
    root = ET.parse(path).getroot()
    out = {}
    for node in root.iter("AbstractText"):
        label = (node.get("Label") or "").strip().upper()
        text = (node.text or "").strip()
        out[label] = text
    return out


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_stats(a: str, b: str):
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0, 1.0
    inter = ta & tb
    union = ta | tb
    jaccard = len(inter) / len(union) if union else 0.0
    coverage = len(inter) / len(ta) if ta else 0.0  
    return jaccard, coverage


def classify(xml_findings: str, ann_report: str):
    if not xml_findings:
        return "NO_FINDINGS_XML", 0.0, 0.0
    nx, na = normalize(xml_findings), normalize(ann_report)
    if nx == na:
        return "EXACT", 1.0, 1.0
    jac, cov = token_stats(nx, na)
    if nx and nx in na:
        return "SUBSET", jac, cov          
    if jac >= 0.9:
        return "NEAR", jac, cov
    if jac >= 0.5:
        return "PARTIAL", jac, cov
    return "MISMATCH", jac, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="cartella che contiene iu_xray_original/ e iu_xray_r2gen/")
    ap.add_argument("--csv", type=Path, help="scrive il dettaglio per-report su CSV")
    ap.add_argument("--limit", type=int, help="analizza solo i primi N record")
    ap.add_argument("--show", type=int, default=5,
                    help="quanti esempi stampare per ogni categoria non-EXACT")
    ap.add_argument("--table-out", type=Path,
                    default=Path(__file__).resolve().parent / "out" / "findings_original_to_r2gen_summary.md",
                    help="file Markdown con le tabelle di riepilogo "
                         "(default: out/findings_original_to_r2gen_summary.md; '' per non scriverlo)")
    args = ap.parse_args()

    reports_dir = args.dataset / "iu_xray_original" / "reports"
    ann_path = args.dataset / "iu_xray_r2gen" / "annotation.json"
    for p in (reports_dir, ann_path):
        if not p.exists():
            sys.exit(f"Percorso non trovato: {p}")

    annotation = json.load(ann_path.open())
    records = [r for split in annotation.values() for r in split]
    if args.limit:
        records = records[: args.limit]

    xml_cache = {}
    rows = []
    for rec in records:
        uid = extract_uid(rec["id"])
        if uid is None:
            rows.append(dict(id=rec["id"], split=rec.get("split", ""), uid="",
                             xml_file="", status="BAD_ID", jaccard=0.0, coverage=0.0,
                             xml_findings="", annotation_report=rec["report"]))
            continue

        xml_file = reports_dir / f"{uid}.xml"
        if not xml_file.exists():
            rows.append(dict(id=rec["id"], split=rec.get("split", ""), uid=uid,
                             xml_file=xml_file.name, status="XML_MISSING",
                             jaccard=0.0, coverage=0.0, xml_findings="",
                             annotation_report=rec["report"]))
            continue

        if uid not in xml_cache:
            xml_cache[uid] = parse_xml_report(xml_file)
        sections = xml_cache[uid]
        findings = sections.get("FINDINGS", "")
        status, jac, cov = classify(findings, rec["report"])
        rows.append(dict(id=rec["id"], split=rec.get("split", ""), uid=uid,
                         xml_file=xml_file.name, status=status,
                         jaccard=round(jac, 4), coverage=round(cov, 4),
                         xml_findings=findings, annotation_report=rec["report"]))

    total = len(rows)
    counts = Counter(r["status"] for r in rows)
    matched = counts["EXACT"] + counts["SUBSET"] + counts["NEAR"]
    jacs = [r["jaccard"] for r in rows if r["status"] not in
            ("XML_MISSING", "BAD_ID", "NO_FINDINGS_XML")]
    jacs_sorted = sorted(jacs)

    overview = [
        ("Record analizzati", total),
        ("XML disponibili", len(list(reports_dir.glob("*.xml")))),
        ("UID distinti usati", len({r["uid"] for r in rows if r["uid"]})),
        ("MATCH (EXACT+SUBSET+NEAR)", f"{matched}/{total} ({100 * matched / total:.2f}%)"),
    ]
    if jacs:
        overview += [
            ("Jaccard medio", f"{sum(jacs) / len(jacs):.4f}"),
            ("Jaccard mediana", f"{jacs_sorted[len(jacs_sorted) // 2]:.4f}"),
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

    emit("MATCH FINDINGS: annotation.json vs reports XML originali", tables,
         args.table_out)

    if args.show:
        for status in ("MISMATCH", "PARTIAL", "NEAR", "NO_FINDINGS_XML",
                       "XML_MISSING", "BAD_ID"):
            examples = [r for r in rows if r["status"] == status][: args.show]
            if not examples:
                continue
            print(f"\n--- esempi {status} ({counts[status]} totali) ---")
            for r in examples:
                print(f"[{r['id']}] ({r['xml_file']}) jaccard={r['jaccard']}")
                print(f"  XML  : {r['xml_findings'][:200] or '<vuoto>'}")
                print(f"  ANNOT: {r['annotation_report'][:200]}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV dettagliato scritto in: {args.csv}")


if __name__ == "__main__":
    main()
