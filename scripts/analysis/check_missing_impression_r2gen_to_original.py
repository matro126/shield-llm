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
PLACEHOLDER_RE = re.compile(r"^(none|n/?a|xxxx|\.|-|_)+$")


def extract_uid(annotation_id: str):
    m = ID_RE.match(annotation_id)
    return m.group(1) if m else None


def parse_sections(path: Path) -> dict:
    root = ET.parse(path).getroot()
    return {(node.get("Label") or "").strip().upper(): (node.text or "").strip()
            for node in root.iter("AbstractText")}


def is_placeholder(text: str) -> bool:
    norm = re.sub(r"[^a-z0-9/]+", " ", text.lower()).strip()
    if not norm:
        return True
    return all(PLACEHOLDER_RE.match(tok) for tok in norm.split())


def classify(sections: dict):
    if "IMPRESSION" not in sections:
        return "MISSING_TAG", ""
    text = sections["IMPRESSION"]
    if not text:
        return "EMPTY", ""
    if is_placeholder(text):
        return "PLACEHOLDER", text
    return "OK", text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="cartella che contiene iu_xray_original/ e iu_xray_r2gen/")
    ap.add_argument("--annotation", type=Path,
                    help="annotation.json da usare (default: iu_xray_r2gen/annotation.json)")
    ap.add_argument("--csv-out", type=Path,
                    help="CSV con il dettaglio di TUTTI i sample analizzati")
    ap.add_argument("--list-out", type=Path,
                    help="file di testo con i soli id problematici (uno per riga)")
    ap.add_argument("--limit", type=int, help="analizza solo i primi N sample")
    ap.add_argument("--show", type=int, default=20,
                    help="quanti sample problematici elencare a video (0 = tutti)")
    ap.add_argument("--strict", action="store_true",
                    help="considera mancanti solo MISSING_TAG/EMPTY "
                         "(i placeholder tipo 'None.' contano come presenti)")
    ap.add_argument("--table-out", type=Path,
                    default=Path(__file__).resolve().parent / "out"
                            / "missing_impression_r2gen_to_original_summary.md",
                    help="file Markdown col riepilogo "
                         "(default: out/missing_impression_r2gen_to_original_summary.md)")
    args = ap.parse_args()

    reports_dir = args.dataset / "iu_xray_original" / "reports"
    ann_path = args.annotation or (args.dataset / "iu_xray_r2gen" / "annotation.json")
    for p in (reports_dir, ann_path):
        if not p.exists():
            sys.exit(f"Percorso non trovato: {p}")

    annotation = json.load(ann_path.open())
    records = [r for split in annotation.values() for r in split]
    if args.limit:
        records = records[: args.limit]

    sections_cache = {}
    rows = []
    for rec in records:
        uid = extract_uid(rec["id"])
        row = dict(id=rec["id"], split=rec.get("split", ""), uid=uid or "",
                   xml_file="", status="", impression="",
                   has_findings_xml="", report_annotation=rec["report"])

        if uid is None:
            rows.append(dict(row, status="BAD_ID"))
            continue

        xml_file = reports_dir / f"{uid}.xml"
        row["xml_file"] = xml_file.name
        if not xml_file.exists():
            rows.append(dict(row, status="XML_MISSING"))
            continue

        if uid not in sections_cache:
            sections_cache[uid] = parse_sections(xml_file)
        sections = sections_cache[uid]

        status, impression = classify(sections)
        rows.append(dict(row, status=status, impression=impression,
                         has_findings_xml=bool(sections.get("FINDINGS"))))

    missing_states = ({"MISSING_TAG", "EMPTY"} if args.strict
                      else {"MISSING_TAG", "EMPTY", "PLACEHOLDER"})
    unresolved = {"XML_MISSING", "BAD_ID"}
    missing = [r for r in rows if r["status"] in missing_states]
    problems = [r for r in rows if r["status"] in missing_states | unresolved]

    total = len(rows)
    counts = Counter(r["status"] for r in rows)

    overview = [
        ("Annotation", str(ann_path)),
        ("Report XML", str(reports_dir)),
        ("Sample analizzati", total),
        ("Con IMPRESSION valida",
         f"{counts['OK']}/{total} ({100 * counts['OK'] / total:.2f}%)"),
        ("SENZA IMPRESSION",
         f"{len(missing)}/{total} ({100 * len(missing) / total:.2f}%)"),
        ("Non verificabili (XML mancante / id non valido)",
         sum(counts[s] for s in unresolved)),
        ("Modalita'", "strict: placeholder = presente" if args.strict
                      else "placeholder = mancante"),
    ]

    status_rows = [(s, n, f"{100 * n / total:.2f}%") for s, n in counts.most_common()]
    status_rows.append(("TOTALE", total, "100.00%"))

    tables = [("Riepilogo", ["Metrica", "Valore"], overview, {1}),
              ("Esiti IMPRESSION", ["Status", "N", "%"], status_rows, {1, 2})]

    per_split = Counter((r["split"], r["status"] in missing_states) for r in rows)
    splits = sorted({r["split"] for r in rows})
    if len(splits) > 1:
        split_rows = []
        for s in splits:
            tot_s = per_split[(s, True)] + per_split[(s, False)]
            miss_s = per_split[(s, True)]
            split_rows.append((s, miss_s, tot_s, f"{100 * miss_s / tot_s:.2f}%"))
        tables.append(("Mancanti per split",
                       ["Split", "Senza IMPRESSION", "Totale", "%"],
                       split_rows, {1, 2, 3}))

    emit("SAMPLE R2GEN SENZA IMPRESSION NEL DATASET ORIGINALE", tables, args.table_out)

    if problems:
        shown = problems if not args.show else problems[: args.show]
        print(f"\n--- sample senza IMPRESSION ({len(problems)} totali) ---")
        for r in shown:
            extra = f' -> "{r["impression"]}"' if r["impression"] else ""
            print(f"  {r['id']:<24} uid={r['uid']:<6} {r['xml_file']:<10} "
                  f"{r['status']}{extra}")
        if len(shown) < len(problems):
            print(f"  ... e altri {len(problems) - len(shown)} "
                  f"(--show 0 per vederli tutti, oppure --csv-out / --list-out)")
    else:
        print("\nTutti i sample dell'annotation hanno l'IMPRESSION nel dataset originale.")

    if args.list_out:
        args.list_out.parent.mkdir(parents=True, exist_ok=True)
        args.list_out.write_text("\n".join(r["id"] for r in problems) + "\n",
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
