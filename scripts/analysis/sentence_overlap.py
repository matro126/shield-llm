#!/usr/bin/env python3
"""Dizionari di frasi: quanto i modelli riusano il lessico del training.

Costruisce, per il training e per le predizioni, un dizionario `testo → quante
volte compare`, a due granularita':

    sezioni   il testo INTERO di reperti o impressione   → quanti referti sono uguali
    frasi     le singole frasi che lo compongono          → quali formule si ripetono

Due modi d'uso.

  UNO — il dettaglio di un esperimento: i quattro dizionari, la distribuzione
  delle frequenze e le frasi piu' amplificate.

      python scripts/analysis/sentence_overlap.py \\
          --train dataset/iu-xray/ita/iu_xray_it_FL-FI/train.jsonl \\
          --pred  training/it/.../results/val_predictions_best.json

  DUE — il confronto fra TUTTI gli esperimenti conclusi, che e' il modo per cui
  esiste: una riga per esperimento, raggruppate per dataset, sul miglior
  checkpoint di ciascuno.

      python scripts/analysis/sentence_overlap.py --all

I dataset con i soli reperti non hanno impressione: per loro le colonne della
sezione mancante restano `n/d`, non zero, e non entrano in nessuna media. Se un
modello addestrato sui soli reperti producesse comunque un'impressione, viene
segnalato: e' formato non richiesto.

Ogni misura e' accompagnata da quella sui RIFERIMENTI dello stesso file — testi
umani mai visti dal modello. Senza quel metro «l'80% delle frasi viene dal
training» non si sa se accusa il modello o il dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shield.data.prompts import split_sections  # noqa: E402
from shield.training.config import Identity, build_config  # noqa: E402

SEZIONI = ("findings", "impression")
DATASETS = ["F-F", "FL-F", "F-FI", "FL-FI"]
MODELLI = ["2B", "8B", "32B"]

ENUM = re.compile(r"^\s*(?:\d+\s*[.)]|[-•*])\s*")
BOUNDARY = re.compile(r"(?<!\d)[.;](?!\d)|\n+")


def normalizza(testo: str) -> str:
    return " ".join(testo.split()).strip().lower().strip(".,;: ")


def frasi(testo: str | None) -> list[str]:
    if not testo:
        return []
    out = []
    for pezzo in BOUNDARY.split(testo):
        pezzo = normalizza(ENUM.sub("", pezzo))
        if len(pezzo.split()) >= 3:
            out.append(pezzo)
    return out


# ─────────────────────────── dizionari ───────────────────────────


def vuoto() -> dict:
    return {s: {"sezioni": Counter(), "frasi": Counter()} for s in SEZIONI}


def aggiungi(diz: dict, findings: str | None, impression: str | None) -> None:
    for sezione, testo in (("findings", findings), ("impression", impression)):
        if not testo or not testo.strip():
            continue
        diz[sezione]["sezioni"][normalizza(testo)] += 1
        diz[sezione]["frasi"].update(frasi(testo))


_cache_train: dict[Path, tuple[dict, int]] = {}


def dal_training(paths: list[Path]) -> tuple[dict, int]:
    """I dizionari del training. In cache: i 24 esperimenti condividono 4 dataset."""
    chiave = tuple(sorted(paths))
    if chiave in _cache_train:
        return _cache_train[chiave]
    diz, n = vuoto(), 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                target = json.loads(line)["messages"][-1]["content"]
                if not isinstance(target, str):
                    continue
                head, tail = split_sections(target)
                aggiungi(diz, head, tail)
                n += 1
    _cache_train[chiave] = (diz, n)
    return diz, n


def dai_sample(samples: list[dict], key: str) -> dict:
    diz = vuoto()
    for sample in samples:
        sez = sample.get(key) or {}
        aggiungi(diz, sez.get("findings"), sez.get("impression"))
    return diz


def leggi_sample(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples") if isinstance(data, dict) else data
    if not samples:
        raise SystemExit(f"{path}: nessuna lista 'samples'")
    return samples


# ─────────────────────────── misure ───────────────────────────


def profilo(contatore: Counter) -> dict:
    occorrenze, distinte = sum(contatore.values()), len(contatore)
    if not occorrenze:
        return {"occorrenze": 0, "distinte": 0}
    fasce = Counter()
    for c in contatore.values():
        fasce["1" if c == 1 else "2-4" if c < 5 else "5-9" if c < 10
              else "10-49" if c < 50 else "50+"] += 1
    return {
        "occorrenze": occorrenze,
        "distinte": distinte,
        "ripetizione": occorrenze / distinte,
        "copertura_top10": sum(c for _, c in contatore.most_common(10)) / occorrenze,
        "hapax": fasce["1"] / distinte,
        "fasce": dict(fasce),
    }


def confronto(train: Counter, pred: Counter) -> dict:
    occ_p, occ_t = sum(pred.values()), sum(train.values())
    viste = {f: c for f, c in pred.items() if f in train}
    righe = []
    for frase, quante in sorted(pred.items(), key=lambda kv: -kv[1]):
        quota_p = quante / occ_p if occ_p else 0
        quota_t = train.get(frase, 0) / occ_t if occ_t else 0
        righe.append({"frase": frase, "train": train.get(frase, 0), "pred": quante,
                      "quota_train": quota_t, "quota_pred": quota_p,
                      "amplificazione": (quota_p / quota_t) if quota_t else None})
    # quanto il modello gonfia le formule che il training ha piu' spesso
    ampl = [r["amplificazione"] for r in righe
            if r["amplificazione"] is not None
            and r["frase"] in dict(train.most_common(10))]
    return {
        "occorrenze": occ_p,
        "quota_occorrenze": sum(viste.values()) / occ_p if occ_p else 0,
        "distinte": len(pred),
        "quota_distinte": len(viste) / len(pred) if pred else 0,
        "nuove": len(pred) - len(viste),
        "quota_nuove": (len(pred) - len(viste)) / len(pred) if pred else 0,
        "ampl_top10": sum(ampl) / len(ampl) if ampl else None,
        "righe": righe,
    }


def sintesi(diz_train: dict, diz_pred: dict, diz_rif: dict, sezione: str) -> dict | None:
    """I numeri di una sezione per un esperimento. None se la sezione non esiste."""
    pp = profilo(diz_pred[sezione]["frasi"])
    if not pp["occorrenze"]:
        return None
    pr = profilo(diz_rif[sezione]["frasi"])
    pt = profilo(diz_train[sezione]["frasi"])
    cm = confronto(diz_train[sezione]["frasi"], diz_pred[sezione]["frasi"])
    cr = confronto(diz_train[sezione]["frasi"], diz_rif[sezione]["frasi"])
    return {
        "distinte": pp["distinte"],
        "distinte_rif": pr["distinte"],
        "diversita": pp["distinte"] / pr["distinte"] if pr["distinte"] else None,
        "vocab_train": pp["distinte"] / pt["distinte"] if pt["distinte"] else None,
        "ripetizione": pp["ripetizione"],
        "top10": pp["copertura_top10"],
        "top10_rif": pr["copertura_top10"],
        "copia": cm["quota_occorrenze"],
        "copia_rif": cr["quota_occorrenze"],
        "delta_copia": cm["quota_occorrenze"] - cr["quota_occorrenze"],
        "nuove": cm["quota_nuove"],
        "ampl_top10": cm["ampl_top10"],
        "sezioni_uguali": profilo(diz_pred[sezione]["sezioni"])["ripetizione"],
    }


# ─────────────────────────── modalita' dettaglio ───────────────────────────


def dettaglio(diz_train, n_train, diz_pred, diz_rif, samples, top: int) -> None:
    print(f"training   : {n_train} referti")
    print(f"predizioni : {len(samples)} sample\n")

    print("═══ 1. I DIZIONARI")
    print(f"  {'':<34}{'occorr.':>8}{'distinte':>9}{'ripet.':>9}"
          f"{'top10':>9}{'uniche':>8}   1/2-4/5-9/10-49/50+")
    for sezione in SEZIONI:
        righe = 0
        for etichetta, diz in (("DATASET", diz_train), ("modello", diz_pred),
                               ("riferimenti", diz_rif)):
            for livello in ("sezioni", "frasi"):
                p = profilo(diz[sezione][livello])
                if not p["occorrenze"]:
                    continue
                righe += 1
                f = p["fasce"]
                print(f"  {sezione + ' · ' + etichetta + ' · ' + livello:<34}"
                      f"{p['occorrenze']:>8}{p['distinte']:>9}{p['ripetizione']:>8.1f}×"
                      f"{p['copertura_top10']:>9.1%}{p['hapax']:>8.1%}   "
                      f"{f.get('1',0)}/{f.get('2-4',0)}/{f.get('5-9',0)}/"
                      f"{f.get('10-49',0)}/{f.get('50+',0)}")
        if righe:
            print()

    print("═══ 2. LE FRASI PIU' PRODOTTE E LA LORO AMPLIFICAZIONE")
    for sezione in SEZIONI:
        c = confronto(diz_train[sezione]["frasi"], diz_pred[sezione]["frasi"])
        if not c["righe"]:
            continue
        print(f"\n── {sezione}   (amplificazione = quota predizioni / quota training)")
        print(f"  {'pred':>6}{'quota':>8}{'train':>7}{'quota':>8}{'ampl.':>8}   frase")
        for r in c["righe"][:top]:
            a = "NUOVA" if r["amplificazione"] is None else f"{r['amplificazione']:.1f}×"
            print(f"  {r['pred']:>6}{r['quota_pred']:>8.1%}{r['train']:>7}"
                  f"{r['quota_train']:>8.1%}{a:>8}   «{r['frase'][:64]}»")
        if c["nuove"]:
            print(f"  → {c['nuove']} frasi distinte non presenti nel training "
                  f"({c['quota_nuove']:.1%})")


# ─────────────────────────── modalita' confronto ───────────────────────────


def esperimenti(training_root: Path) -> list[tuple[str, Path, Path, str, str]]:
    """(nome, file predizioni, train.jsonl, codice dataset, target) per i conclusi."""
    out = []
    # training/<lang>/<modello>/<modalita>/<dataset>/results/
    for pred in sorted(training_root.glob("*/*/*/*/results/val_predictions_best.json")):
        rel = pred.parents[1].relative_to(ROOT)
        try:
            cfg = build_config(Identity.from_path(str(rel)), ROOT)
        except Exception as exc:                       # noqa: BLE001
            print(f"  ! {rel}: {exc}", file=sys.stderr)
            continue
        train = ROOT / cfg.dataset_root / "train.jsonl"
        if not train.is_file():
            print(f"  ! {cfg.experiment}: manca {train}", file=sys.stderr)
            continue
        out.append((cfg.experiment, pred, train, cfg.dataset_code, cfg.target))
    return out


def confronta_tutti(training_root: Path, out_csv: Path | None) -> int:
    trovati = esperimenti(training_root)
    if not trovati:
        print("Nessun esperimento con val_predictions_best.json.", file=sys.stderr)
        return 1

    righe: list[dict] = []
    anomalie: list[str] = []
    riferimenti: dict[tuple[str, str], dict] = {}

    for nome, pred_path, train_path, codice, target in trovati:
        diz_train, _ = dal_training([train_path])
        samples = leggi_sample(pred_path)
        diz_pred = dai_sample(samples, "prediction_sections")
        diz_rif = dai_sample(samples, "reference_sections")

        if target == "findings" and diz_pred["impression"]["frasi"]:
            anomalie.append(f"{nome}: ha prodotto un'impressione non richiesta "
                            f"({sum(diz_pred['impression']['sezioni'].values())} volte)")

        for sezione in SEZIONI:
            if target == "findings" and sezione == "impression":
                continue                    # sezione inesistente: non e' uno zero
            s = sintesi(diz_train, diz_pred, diz_rif, sezione)
            if s is None:
                anomalie.append(f"{nome}: nessuna {sezione} generata")
                continue
            righe.append({"esperimento": nome, "dataset": codice,
                          "sezione": sezione, "n_sample": len(samples), **s})
            riferimenti.setdefault((codice, sezione), {
                "distinte": s["distinte_rif"], "top10": s["top10_rif"],
                "copia": s["copia_rif"]})

    print(f"esperimenti analizzati: {len(trovati)}   righe: {len(righe)}\n")

    for sezione in SEZIONI:
        sel = [r for r in righe if r["sezione"] == sezione]
        if not sel:
            continue
        print(f"═══ {sezione.upper()}")
        print(f"  {'esperimento':<22}{'distinte':>9}{'div':>7}{'ripet':>8}"
              f"{'top10':>8}{'copia':>8}{'Δcopia':>8}{'ampl':>7}{'nuove':>7}")
        for codice in DATASETS:
            gruppo = [r for r in sel if r["dataset"] == codice]
            if not gruppo:
                continue
            rif = riferimenti.get((codice, sezione), {})
            print(f"  ── {codice}   riferimenti umani: {rif.get('distinte','—')} frasi "
                  f"distinte, top10 {rif.get('top10',0):.0%}, "
                  f"copia {rif.get('copia',0):.0%}")
            for r in sorted(gruppo, key=lambda r: (
                    MODELLI.index(r["esperimento"].split("_")[1])
                    if r["esperimento"].split("_")[1] in MODELLI else 9,
                    r["esperimento"])):
                div = f"{r['diversita']:.2f}" if r["diversita"] else "—"
                amp = f"{r['ampl_top10']:.1f}×" if r["ampl_top10"] else "—"
                print(f"  {r['esperimento']:<22}{r['distinte']:>9}{div:>7}"
                      f"{r['ripetizione']:>7.1f}×{r['top10']:>8.1%}"
                      f"{r['copia']:>8.1%}{r['delta_copia']:>+8.1%}"
                      f"{amp:>7}{r['nuove']:>7.1%}")
        print()

    print("═══ COME SI LEGGE")
    print("  distinte  frasi diverse che il modello produce; div = rispetto ai "
          "riferimenti umani")
    print("  ripet     quante volte in media riusa la stessa frase")
    print("  top10     quanto delle sue generazioni sta nelle sue 10 frasi piu' usate")
    print("  copia     quota di frasi generate gia' presenti nel training")
    print("  Δcopia    quanto copia PIU' dei riferimenti: e' la misura onesta")
    print("  ampl      di quanto gonfia le 10 formule piu' frequenti del training")
    print("  nuove     quota di frasi distinte mai viste nel training")

    if anomalie:
        print("\n═══ ANOMALIE")
        for a in anomalie:
            print(f"  ⚠ {a}")

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(righe[0]))
            w.writeheader()
            w.writerows(righe)
        print(f"\nscritto: {out_csv.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="confronta tutti gli esperimenti conclusi")
    parser.add_argument("--training-root", type=Path, default=ROOT / "training")
    parser.add_argument("--train", type=Path, action="append", help="train.jsonl")
    parser.add_argument("--pred", type=Path, help="val_predictions_best.json")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "training" / "results" / "sentence_overlap.csv")
    args = parser.parse_args(argv)

    if args.all or not args.pred:
        return confronta_tutti(args.training_root, args.csv)

    if not args.train:
        parser.error("con --pred serve anche --train")
    diz_train, n_train = dal_training(args.train)
    samples = leggi_sample(args.pred)
    dettaglio(diz_train, n_train, dai_sample(samples, "prediction_sections"),
              dai_sample(samples, "reference_sections"), samples, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
