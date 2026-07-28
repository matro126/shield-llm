#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONTROLLI = [
    ("training: archivia prima di ripartire",
     "src/shield/training/runner.py", r'archive_results\(results, "interrupted"'),
    ("training: 'test' fra gli artefatti",
     "src/shield/training/runner.py", r'RUN_ARTIFACTS = \([^)]*"test"'),
    ("baseline: archivia la precedente",
     "src/shield/training/baseline.py", r"archive_previous\("),
    ("valutazione: archivia la precedente",
     "training/evaluate_test.py", r"archive_previous\("),
    ("post-hoc: salta se gia' calcolato",
     "scripts/evaluate/metrics_posthoc.py", r"--overwrite"),
]

EFFETTI = [
    ("training", "archivia results.json, curve, val_predictions, best_adapter, test/"),
    ("baseline", "archivia metrics.json, predictions.csv, disaggregated.json"),
    ("evaluate_test.py", "archivia la cartella dello split prima di riscriverla"),
    ("metrics_posthoc.py", "non ricalcola senza --overwrite"),
    ("rq_report / baseline_report / collect", "sola lettura, scrivono in training/results/"),
    ("generate.py", "riscrive solo gli script .py, mai i risultati"),
]


def leggi(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def titolo(testo: str) -> None:
    print("\n" + "=" * 78)
    print(testo)
    print("=" * 78)


def main() -> int:
    runs = [(p, r) for p in sorted(ROOT.glob("training/it/*/*/*/results/results.json"))
            if (r := leggi(p))]
    baselines = [(p, m) for p in sorted(ROOT.glob("training/it/*/baseline/*/results/metrics.json"))
                 if (m := leggi(p))]

    titolo("A. INVENTARIO")
    adapter = sum(1 for p, _ in runs if (p.parent / "best_adapter").is_dir())
    valpred = sum(1 for p, _ in runs if (p.parent / "val_predictions").is_dir())
    test = sum(1 for p, _ in runs if (p.parent / "test" / "metrics.json").is_file())
    print(f"  esperimenti con results.json : {len(runs)}/24")
    print(f"  con best_adapter sul disco   : {adapter}")
    print(f"  con val_predictions          : {valpred}")
    print(f"  con valutazione test         : {test}")
    print(f"  baseline con metrics.json    : {len(baselines)}/12")

    titolo("B. LA RUN VIVA DI OGNI ESPERIMENTO")
    avvii = [r["timing"]["started_at"] for _, r in runs
             if (r.get("timing") or {}).get("started_at")]
    if avvii:
        print(f"  finestra: {min(avvii)[:19]}  →  {max(avvii)[:19]}")
    print(f"\n  {'esperimento':<24}{'avvio':<21}{'stato':<15}{'best':>8}{'archivi':>9}")
    sospetti: list[str] = []
    for p, r in sorted(runs, key=lambda x: x[1].get("experiment", "")):
        nome = r.get("experiment", "?")
        avvio = ((r.get("timing") or {}).get("started_at") or "")[:19]
        cartella = p.parent / "archive"
        archivi = len([d for d in cartella.glob("*") if d.is_dir()]) if cartella.is_dir() else 0
        best = (r.get("best") or {}).get("value")
        print(f"  {nome:<24}{avvio:<21}{r.get('status',''):<15}"
              f"{(f'{best:.4f}' if best else '—'):>8}{archivi:>9}")
        if r.get("status") not in ("completed", "early_stopped"):
            sospetti.append(f"{nome}: stato {r.get('status')}")
        if not best:
            sospetti.append(f"{nome}: nessun best checkpoint")

    titolo("C. ARCHIVI")
    tipi: dict[str, int] = {}
    for p, _ in runs:
        for d in (p.parent / "archive").glob("*"):
            if d.is_dir():
                k = re.sub(r"^\d{8}-\d{6}-?", "", d.name) or "?"
                tipi[k] = tipi.get(k, 0) + 1
    for k, v in sorted(tipi.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26}{v:>4} cartelle")
    if not tipi:
        print("  (nessun archivio)")

    titolo("D. PROTEZIONI ATTIVE NEL CODICE")
    attive = 0
    for nome, rel, pattern in CONTROLLI:
        try:
            ok = bool(re.search(pattern, (ROOT / rel).read_text(encoding="utf-8")))
        except OSError:
            ok = False
        attive += ok
        print(f"  [{'x' if ok else ' '}] {nome:<42} {rel}")

    titolo("E. COSA SUCCEDE SE RILANCI")
    for azione, effetto in EFFETTI:
        print(f"  {azione:<40}{effetto}")

    titolo("VERDETTO")
    ok = (len(runs) == 24 and len(baselines) == 12 and adapter == len(runs)
          and attive == len(CONTROLLI) and not sospetti)
    print(f"  {'PUOI PROCEDERE' if ok else 'DA GUARDARE'}")
    for s in dict.fromkeys(sospetti):
        print(f"    - {s}")
    if len(runs) != 24:
        print(f"    - {len(runs)} run invece di 24")
    if len(baselines) != 12:
        print(f"    - {len(baselines)} baseline invece di 12")
    if adapter != len(runs):
        print(f"    - {len(runs) - adapter} esperimenti senza best_adapter")
    if attive != len(CONTROLLI):
        print(f"    - {len(CONTROLLI) - attive} protezioni non attive")
    print("\n  Questo script non scrive nulla: solo letture.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
