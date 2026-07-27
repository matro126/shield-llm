#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from translate import TranslationCache, cache_stats, translate_many  

from shield.data.prompts import split_sections  
from shield.data.text_source import TextSource  
from shield.training.config import TRAINING_MODES, Identity  
from shield.training.results import write_json_atomic  

DEFAULT_CACHE = ROOT / "scripts" / "evaluate" / "out" / "translation_cache.jsonl"
OUTPUT_NAME = "posthoc_metrics.json"
TRANSLATION_MODEL = "qwen/qwen3-235b-a22b-2507"

SECTIONED = ("bleu", "rougeL", "bertscore", "clinicalbert")   
TRANSLATED = ("chexbert",)                                    


def plain_text(target: str) -> str:
    findings, impression = split_sections(target)
    return " ".join(part for part in (findings, impression) if part).strip()


def english_reference(text: TextSource, sample_id: str, target: str) -> str:
    findings, impression = text.report(sample_id, "en")
    if target == "findings_impression":
        return " ".join(p for p in (findings, impression) if p).strip()
    return findings.strip()


def resolve_run_id(identity: Identity, source: str) -> str | None:
    results = ROOT / identity.relpath / "results"
    path = {
        "validation": results / "results.json",
        "test": results / "test" / "metrics.json",
        "baseline": results / "metrics.json",
    }[source]
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if source == "validation":
        return ((data.get("provenance") or {}).get("mlflow") or {}).get("run_id")
    return data.get("mlflow_run_id")


def log_to_mlflow(run_id: str, source: str, payload: dict[str, Any]) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    prefix = "val" if source == "validation" else source
    for group in payload["groups"]:
        step = int(group.get("step") or 0)
        for key, value in (group.get("sectioned_flat") or {}).items():
            if key != "num_examples":
                client.log_metric(run_id, f"{prefix}.{key}", float(value), step=step)
        for key, value in (group.get("chexbert_en") or {}).items():
            name = f"{prefix}.chexbert_en.{key.removeprefix('chexbert_')}"
            client.log_metric(run_id, name, float(value), step=step)

    client.set_tag(run_id, "posthoc.metrics", ",".join(payload["metrics"]))
    client.set_tag(run_id, "posthoc.computed_at", payload["created_at"])
    if payload.get("translation"):
        client.set_tag(run_id, "posthoc.translation_model", TRANSLATION_MODEL)


def read_predictions_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [
            {"id": r["id"], "prediction": r["prediction"], "reference": r["reference"]}
            for r in csv.DictReader(handle)
        ]


def collect_units(
    source: str, identities: list[Identity], patterns: list[str]
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []

    def selected(name: str) -> bool:
        return not patterns or any(fnmatch.fnmatch(name, p) for p in patterns)

    if source == "baseline":
        seen: dict[str, Identity] = {}
        for identity in identities:
            seen.setdefault(identity.baseline.name, identity.baseline)
        for identity in seen.values():
            if not selected(identity.name):
                continue
            results = ROOT / identity.relpath / "results"
            path = results / "predictions.csv"
            if path.is_file():
                units.append({
                    "identity": identity, "label": identity.name,
                    "out": results / OUTPUT_NAME,
                    "groups": [{"key": "baseline", "rows": read_predictions_csv(path)}],
                })
        return units

    for identity in identities:
        if not selected(identity.name):
            continue
        results = ROOT / identity.relpath / "results"
        if source == "test":
            path = results / "test" / "predictions.csv"
            if path.is_file():
                units.append({
                    "identity": identity, "label": identity.name,
                    "out": results / "test" / OUTPUT_NAME,
                    "groups": [{"key": "test", "rows": read_predictions_csv(path)}],
                })
        else:
            folder = results / "val_predictions"
            steps = sorted(folder.glob("step*.json")) if folder.is_dir() else []
            groups = []
            for step_file in steps:
                data = json.loads(step_file.read_text(encoding="utf-8"))
                groups.append({
                    "key": f"step{data['step']}",
                    "epoch": data.get("epoch"), "step": data.get("step"),
                    "rows": [
                        {"id": s["id"], "prediction": s["prediction"],
                         "reference": s["reference"]}
                        for s in data["samples"]
                    ],
                })
            if groups:
                units.append({
                    "identity": identity, "label": identity.name,
                    "out": folder / OUTPUT_NAME, "groups": groups,
                })
    return units


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("test", "baseline", "validation"), required=True)
    parser.add_argument("--metrics", nargs="+", default=["bertscore"],
                        choices=[*SECTIONED, *TRANSLATED])
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "dataset" / "iu-xray" / "iu_xray_translated.csv")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    wanted_sectioned = [m for m in args.metrics if m in SECTIONED]
    needs_translation = any(m in TRANSLATED for m in args.metrics)

    identities = [
        identity
        for p in sorted((ROOT / "training").glob("*/*/*/*"))
        if p.is_dir()
        for identity in [Identity.from_path(p)]
        if identity.mode in TRAINING_MODES
    ]
    units = collect_units(args.source, identities, args.only)
    if not units:
        print(f"Nessuna predizione per --source {args.source}.", file=sys.stderr)
        return 1

    if not args.overwrite:
        for unit in [u for u in units if u["out"].is_file()]:
            print(f"  – {unit['label']}: gia' calcolato (--overwrite per rifarlo)")
        units = [u for u in units if not u["out"].is_file()]
        if not units:
            print("Tutto gia' calcolato.")
            return 0

    if args.max_samples:
        for unit in units:
            for group in unit["groups"]:
                group["rows"] = group["rows"][: args.max_samples]

    n_sample = sum(len(g["rows"]) for u in units for g in u["groups"])
    print(f"sorgente: {args.source}   metriche: {', '.join(args.metrics)}")
    print(f"  unita': {len(units)}   sample: {n_sample}")

    cache = text = None
    if needs_translation:
        cache = TranslationCache(args.cache)
        text = TextSource(args.csv)
        everything = [
            plain_text(r["prediction"])
            for u in units for g in u["groups"] for r in g["rows"]
        ]
        preventivo = cache_stats(everything, cache)
        print(f"  traduzioni: {preventivo['unici']} testi unici, "
              f"{preventivo['gia_in_cache']} in cache, "
              f"{preventivo['da_tradurre']} DA TRADURRE")
    if args.dry_run:
        print("\n[dry-run] niente calcolato.")
        return 0

    from shield.evaluation import compute_text_metrics, sectioned_metrics
    from shield.evaluation.metrics import chexbert_f1
    from shield.training.evaluation import flatten_sectioned

    for index, unit in enumerate(units, start=1):
        identity: Identity = unit["identity"]
        print(f"\n[{index}/{len(units)}] {unit['label']}")

        payload: dict[str, Any] = {
            "schema_version": 1,
            "source": args.source,
            "experiment": unit["label"],
            "metrics": list(args.metrics),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "groups": [],
        }
        if needs_translation:
            payload["translation"] = {
                "model": TRANSLATION_MODEL,
                "direction": "it->en",
                "applied_to": "predictions only (reference is the original English)",
            }

        for group in unit["groups"]:
            rows = group["rows"]
            entry: dict[str, Any] = {"key": group["key"], "n_examples": len(rows)}
            for field in ("epoch", "step"):
                if group.get(field) is not None:
                    entry[field] = group[field]

            if wanted_sectioned:
                result = sectioned_metrics(
                    [r["prediction"] for r in rows],
                    [r["reference"] for r in rows],
                    wanted_sectioned,
                    identity.target,
                    metric_fn=compute_text_metrics,
                )
                entry["sectioned"] = result
                entry["sectioned_flat"] = flatten_sectioned(result)

            if needs_translation:
                sources = [plain_text(r["prediction"]) for r in rows]
                translated, stats = translate_many(sources, cache, args.workers)
                references = [
                    english_reference(text, r["id"], identity.target) for r in rows
                ]
                usable = [
                    (t, r) for t, r in zip(translated, references)
                    if t.strip() and r.strip()
                ]
                if usable:
                    entry["chexbert_en"] = {
                        k: float(v)
                        for k, v in chexbert_f1(
                            [t for t, _ in usable], [r for _, r in usable]
                        ).items()
                    }
                    entry["n_failed_translations"] = len(rows) - len(usable)
                    entry["translation_stats"] = stats

            payload["groups"].append(entry)

            summary = [
                f"{k}={entry['sectioned_flat'][k]:.4f}"
                for k in ("bertscore_f1", "rougeL", "bleu", "clinicalbert_f1")
                if k in entry.get("sectioned_flat", {})
            ]
            if "chexbert_en" in entry:
                summary.append(
                    f"chexbert_en={entry['chexbert_en']['chexbert_f1_micro_top5']:.4f}"
                )
            print(f"    {group['key']:<12} n={len(rows):<5} {'  '.join(summary)}")

        write_json_atomic(unit["out"], payload)
        print(f"    → {unit['out'].relative_to(ROOT)}")

        if args.no_mlflow:
            continue
        run_id = resolve_run_id(identity, args.source)
        if run_id is None:
            print("    · nessun run MLflow associato: metriche solo su disco")
            continue
        try:
            log_to_mlflow(run_id, args.source, payload)
            print(f"    · tracciate nel run MLflow {run_id[:12]}…")
        except Exception as exc:
            print(f"    · MLflow non raggiungibile ({type(exc).__name__}): "
                  "le metriche restano su disco, si possono riscrivere dopo")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
