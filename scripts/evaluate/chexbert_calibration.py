#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv as csv_module
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from translate import TranslationCache, translate_many  

from shield.training.results import write_json_atomic  

DEFAULT_CACHE = ROOT / "scripts" / "evaluate" / "out" / "translation_cache.jsonl"
DEFAULT_OUT = ROOT / "scripts" / "evaluate" / "out" / "chexbert_calibration.json"


def report_text(row: dict, lang: str) -> str:
    suffix = "" if lang == "en" else "_it"
    parts = [row.get(f"findings{suffix}", ""), row.get(f"impression{suffix}", "")]
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="studi da campionare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "dataset" / "iu-xray" / "iu_xray_translated.csv")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    with args.csv.open(encoding="utf-8") as handle:
        rows = [
            r for r in csv_module.DictReader(handle)
            if report_text(r, "en") and report_text(r, "it")
        ]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    print(f"studi campionati: {len(rows)}  (seed {args.seed})")

    english = [report_text(r, "en") for r in rows]
    italian = [report_text(r, "it") for r in rows]

    if args.dry_run:
        from translate import cache_stats

        print("preventivo traduzioni:", cache_stats(italian, TranslationCache(args.cache)))
        return 0

    print("traduco i riferimenti italiani in inglese…")
    cache = TranslationCache(args.cache)
    back, stats = translate_many(italian, cache, args.workers)
    print(f"  {stats}")

    keep = [i for i in range(len(rows)) if back[i].strip()]
    if not keep:
        print("Nessuna traduzione riuscita.", file=sys.stderr)
        return 1
    en = [english[i] for i in keep]
    bt = [back[i] for i in keep]
    it = [italian[i] for i in keep]
    print(f"  utilizzabili: {len(keep)}/{len(rows)}")

    from shield.evaluation import chexbert_f1

    print("\netichetto con CheXbert…")
    ceiling = chexbert_f1(bt, en)      
    raw_it = chexbert_f1(it, en)       

    print(f"\n  {'confronto':<44}{'f1_micro_top5':>16}{'accuracy':>12}")
    print(f"  {'traduzione IT→EN  vs  EN originale':<44}"
          f"{ceiling['chexbert_f1_micro_top5']:>16.4f}{ceiling['chexbert_accuracy']:>12.4f}"
          "   ← TETTO")
    print(f"  {'italiano GREZZO   vs  EN originale':<44}"
          f"{raw_it['chexbert_f1_micro_top5']:>16.4f}{raw_it['chexbert_accuracy']:>12.4f}"
          "   ← senza traduzione")

    verdict = (
        "utilizzabile" if ceiling["chexbert_f1_micro_top5"] >= 0.85
        else "utilizzabile con cautela: normalizzare sul tetto e dichiararlo"
        if ceiling["chexbert_f1_micro_top5"] >= 0.70
        else "NON utilizzabile: la traduzione perde troppe entita' cliniche"
    )
    print(f"\n  verdetto: {verdict}")

    write_json_atomic(args.out, {
        "schema_version": 1,
        "n_studies": len(keep),
        "seed": args.seed,
        "translation": {"model": "qwen/qwen3-235b-a22b-2507", "direction": "it->en"},
        "ceiling_translated_vs_english": {k: float(v) for k, v in ceiling.items()},
        "raw_italian_vs_english": {k: float(v) for k, v in raw_it.items()},
        "verdict": verdict,
        "translation_stats": stats,
    })
    print(f"  scritto: {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
