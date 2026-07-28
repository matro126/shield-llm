#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shield.data.prompts import split_sections  # noqa: E402

NEGATIVA = re.compile(
    r"(nessun|non si (osserv|rilev|eviden|identific)|assenza di|negativ|"
    r"nella norma|nei limiti|non vi (e|è|sono)|torace normale|"
    r"esame .{0,20}normale|non presenta)", re.I)

MASSA = re.compile(r"\bmass[ae]\b|\bnodul|\btumor|neoplas|carcinom", re.I)

TERMINI = {
    "Cardiomegaly": r"cardiomegal|cuore .{0,25}(ingrandi|aumentat|grande)|"
                    r"dimensioni cardiache .{0,25}(aumentat|ingrandi)|ingrandimento cardiaco",
    "Pleural Effusion": r"versament|effusion",
    "Pneumothorax": r"pneumotorac|pneumothorax",
    "Atelectasis": r"atelettas|atelectas",
    "Edema": r"edema|congestione vascolare|ipertensione venosa",
    "Consolidation": r"consolida|addensament",
    "Pneumonia": r"polmonit|pneumoni",
    "Lung Lesion": r"nodul|\bmass[ae]\b|lesion|granulom",
    "Lung Opacity": r"opacit|infiltrat",
    "Fracture": r"frattur",
    "Support Devices": r"catetere|sondino|tubo|pacemaker|sternotom|clip|picc|port|elettrod",
    "Enlarged Cardiomediastinum": r"mediastin.{0,30}(allarg|prominen|ingrandi)|"
                                  r"(allarg|prominen).{0,20}mediastin",
    "Pleural Other": r"pleuric|pleural|ispessiment",
}


def carica_split(dataset_root: Path, split: str) -> dict[str, dict]:
    out = {}
    path = dataset_root / f"{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"manca {path}")
    with path.open(encoding="utf-8") as fh:
        for riga in fh:
            if not riga.strip():
                continue
            r = json.loads(riga)
            testo = r["messages"][-1]["content"]
            f, i = split_sections(testo) if isinstance(testo, str) else (None, None)
            out[str(r["id"])] = {
                "categorie": (r.get("factors") or {}).get("diagnostic_category") or [],
                "riferimento_findings": f or "",
                "riferimento_impression": i or "",
            }
    return out


def scegli(record: dict[str, dict], n_sani: int, n_malati: int) -> list[str]:
    sani = sorted(k for k, v in record.items() if v["categorie"] == ["No Finding"])
    malati = defaultdict(list)
    for k in sorted(record):
        cats = [c for c in record[k]["categorie"] if c not in ("No Finding", "Unlabeled")]
        if cats:
            malati[sorted(cats)[0]].append(k)

    con_massa = [k for k in sorted(record)
                 if MASSA.search(record[k]["riferimento_findings"] + " "
                                 + record[k]["riferimento_impression"])
                 and record[k]["categorie"] != ["No Finding"]]

    scelti: list[str] = []
    if con_massa:
        scelti.append(con_massa[0])

    categorie = sorted(malati, key=lambda c: (-len(malati[c]), c))
    giro = 0
    while len(scelti) < n_malati and categorie:
        aggiunto = False
        for c in categorie:
            if giro < len(malati[c]):
                k = malati[c][giro]
                if k not in scelti:
                    scelti.append(k)
                    aggiunto = True
                if len(scelti) >= n_malati:
                    break
        if not aggiunto:
            break
        giro += 1
    return sani[:n_sani] + scelti[:n_malati]


def predizioni_val(results: Path) -> dict[str, dict]:
    path = results / "val_predictions_best.json"
    if not path.is_file():
        return {}
    dati = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for s in dati.get("samples", []):
        sez = s.get("prediction_sections") or {}
        out[str(s["id"])] = {"findings": sez.get("findings") or "",
                             "impression": sez.get("impression") or ""}
    return out


def predizioni_test(results: Path) -> dict[str, dict]:
    path = results / "test" / "predictions.csv"
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f, i = split_sections(r.get("prediction") or "")
            out[str(r["id"])] = {"findings": f or "", "impression": i or ""}
    return out


def esperimenti(training_root: Path, split: str) -> list[tuple[str, dict[str, dict]]]:
    out = []
    for d in sorted(training_root.glob("it/*/*/*/results")):
        if d.parent.parent.name == "baseline":
            continue
        nome = "_".join(["it", d.parents[2].name.replace("Qwen-3-VL-", "").replace("-Instruct", ""),
                         d.parents[1].name, d.parent.name.replace("iu_xray_r2gen_", "")])
        pred = predizioni_val(d) if split == "val" else predizioni_test(d)
        if pred:
            out.append((nome, pred))
    return out


def analizza(record: dict[str, dict], pred: dict[str, dict]) -> dict:
    sani_neg = sani_tot = mal_neg = mal_tot = trovate = trovabili = 0
    for sid, r in record.items():
        p = pred.get(sid)
        if not p:
            continue
        testo = (p["findings"] + " " + p["impression"]).strip()
        impr = p["impression"] or p["findings"]
        negativa = bool(NEGATIVA.search(impr))
        if r["categorie"] == ["No Finding"]:
            sani_tot += 1
            sani_neg += negativa
        else:
            mal_tot += 1
            mal_neg += negativa
            for c in r["categorie"]:
                if c in TERMINI:
                    trovabili += 1
                    trovate += bool(re.search(TERMINI[c], testo, re.I))
    return {
        "sani": sani_tot, "sani_negativa": sani_neg,
        "malati": mal_tot, "malati_negativa": mal_neg,
        "quota_sani_neg": sani_neg / sani_tot if sani_tot else 0.0,
        "falsi_negativi": mal_neg / mal_tot if mal_tot else 0.0,
        "patologie_citate": trovate / trovabili if trovabili else 0.0,
        "trovabili": trovabili,
    }


def patologia_citata(categorie: list[str], testo: str) -> str:
    trovate = [c for c in categorie if c in TERMINI
               and re.search(TERMINI[c], testo, re.I)]
    return ",".join(trovate)


def righe_csv(record: dict[str, dict], esp: list[tuple[str, dict[str, dict]]]) -> list[dict]:
    out = []
    for sid in sorted(record):
        r = record[sid]
        sano = r["categorie"] == ["No Finding"]
        for nome, pred in esp:
            p = pred.get(sid)
            if not p:
                continue
            impr = p["impression"] or p["findings"]
            out.append({
                "id": sid,
                "categorie": "|".join(r["categorie"]),
                "sano": "si" if sano else "no",
                "esperimento": nome,
                "rif_findings": r["riferimento_findings"],
                "rif_impression": r["riferimento_impression"],
                "gen_findings": p["findings"],
                "gen_impression": p["impression"],
                "impressione_negativa": "si" if NEGATIVA.search(impr) else "no",
                "patologie_citate": patologia_citata(r["categorie"],
                                                     p["findings"] + " " + p["impression"]),
            })
    return out


def relazione(split: str, record: dict[str, dict], mostrati: list[str],
              esp: list[tuple[str, dict[str, dict]]], out: Path) -> None:
    md = [f"# Confronto fra modelli — {split}", "",
          f"{len(mostrati)} studi mostrati per esteso su {len(record)} dello split. "
          f"{len(esp)} esperimenti.", "",
          "## Comportamento sull'INTERO split", "",
          "`falsi negativi` e' la quota di studi **patologici** per cui il modello produce "
          "comunque un'impressione negativa. `patologie citate` conta i termini della "
          "categoria presenti nel testo, **senza distinguere le negazioni**: e' un limite "
          "superiore, non la sensibilita' reale.", "",
          "| esperimento | sani | impressione negativa | patologici | falsi negativi | patologie citate |",
          "|---|---|---|---|---|---|"]
    for nome, pred in esp:
        a = analizza(record, pred)
        md.append(f"| {nome} | {a['sani']} | {a['quota_sani_neg']:.1%} | {a['malati']} | "
                  f"**{a['falsi_negativi']:.1%}** | {a['patologie_citate']:.1%} |")

    md += ["", "---", ""]
    for sid in mostrati:
        r = record[sid]
        stato = "SANO" if r["categorie"] == ["No Finding"] else "PATOLOGICO"
        md += [f"## `{sid}` — {stato} — {', '.join(r['categorie'])}", "",
               "### Riferimento", "",
               "**Reperti**", "", r["riferimento_findings"] or "_assenti_", ""]
        if r["riferimento_impression"]:
            md += ["**Impressione**", "", r["riferimento_impression"], ""]
        md += ["### Generati", ""]
        for nome, pred in esp:
            p = pred.get(sid)
            md += [f"**{nome}**", ""]
            if not p:
                md += ["_nessuna predizione_", ""]
                continue
            md += ["*Reperti* — " + (p["findings"] or "_assenti_"), ""]
            if p["impression"]:
                md += ["*Impressione* — " + p["impression"], ""]
        md += ["---", ""]
    out.write_text("\n".join(md), encoding="utf-8")


def stampa(split: str, record: dict[str, dict],
           esp: list[tuple[str, dict[str, dict]]]) -> None:
    print(f"\n═══ {split.upper()}  ({len(record)} studi, {len(esp)} esperimenti)")
    print(f"  {'esperimento':<24}{'sani neg.':>11}{'falsi neg.':>12}{'patol. citate':>15}")
    misure = {nome: analizza(record, pred) for nome, pred in esp}
    for nome, a in sorted(misure.items(), key=lambda kv: kv[1]["falsi_negativi"]):
        print(f"  {nome:<24}{a['quota_sani_neg']:>10.1%}{a['falsi_negativi']:>12.1%}"
              f"{a['patologie_citate']:>15.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "dataset" / "iu-xray" / "ita" / "iu_xray_it_FL-FI")
    parser.add_argument("--training-root", type=Path, default=ROOT / "training")
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "results" / "panel")
    parser.add_argument("--sani", type=int, default=15)
    parser.add_argument("--malati", type=int, default=15)
    parser.add_argument("--tutti", action="store_true",
                        help="mostra per esteso tutti gli studi dello split, non i 30 del pannello")
    parser.add_argument("--solo", action="append", default=[],
                        help="glob sul nome esperimento, ripetibile")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        record = carica_split(args.dataset_root, split)
        panel = scegli(record, args.sani, args.malati)
        esp = esperimenti(args.training_root, split)
        if not esp:
            print(f"  {split}: nessuna predizione disponibile, saltato")
            continue
        if args.solo:
            esp = [(n, p) for n, p in esp
                   if any(fnmatch.fnmatch(n, g) for g in args.solo)]
            if not esp:
                print(f"  {split}: nessun esperimento corrisponde a {args.solo}")
                continue
        mostrati = sorted(record) if args.tutti else panel
        conteggio = Counter(c for k in panel for c in record[k]["categorie"])
        print(f"\n  pannello {split}: {len(panel)} studi — " +
              ", ".join(f"{c} {n}" for c, n in conteggio.most_common()))
        if args.tutti:
            print(f"  --tutti: nel markdown finiscono tutti i {len(record)} studi")
        relazione(split, record, mostrati, esp, args.out / f"{split}.md")
        righe = righe_csv(record, esp)
        with (args.out / f"{split}_predizioni.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(righe[0]))
            w.writeheader(); w.writerows(righe)
        print(f"  csv: {len(righe)} righe (studio × esperimento)")
        stampa(split, record, esp)
    print(f"\nscritti in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
