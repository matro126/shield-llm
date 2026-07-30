#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shield.training.config import TRAINING_MODES, Identity  # noqa: E402
from shield.training.results import write_json_atomic  # noqa: E402

SCREEN_COLUMNS = (
    "experiment", "status", "epochs_completed",
    "best.rougeL", "best.val_loss",
    "baseline.rougeL", "test.rougeL", "delta.rougeL",
    "test.bertscore_f1", "baseline.chexbert_en.f1_micro_top5",
    "test.chexbert_en.f1_micro_top5", "delta.chexbert_en.f1_micro_top5",
    "wall_clock_s",
)


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), inner, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = "; ".join(str(v) for v in value)
    else:
        out[prefix] = value


def posthoc_group(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not payload:
        return {}
    for group in payload.get("groups", []):
        if group.get("key") == key:
            return group
    return {}


def posthoc_metrics(payload: dict[str, Any] | None, key: str) -> dict[str, float]:
    group = posthoc_group(payload, key)
    out = {
        k: float(v)
        for k, v in (group.get("sectioned_flat") or {}).items()
        if k != "num_examples"
    }
    for k, v in (group.get("chexbert_en") or {}).items():
        out[f"chexbert_en.{k.removeprefix('chexbert_')}"] = float(v)
    return out


def experiment_row(
    results: dict[str, Any],
    test: dict[str, Any] | None,
    baseline: dict[str, Any] | None = None,
    test_chexbert: dict[str, Any] | None = None,
    baseline_chexbert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": results.get("experiment"),
        "status": results.get("status"),
        "n_evaluations": results.get("n_evaluations"),
        "epochs_completed": results.get("epochs_completed"),
        "early_stopped": results.get("early_stopped"),
    }
    flatten("", results.get("identity") or {}, row)
    flatten("dataset", results.get("dataset") or {}, row)
    flatten("timing", results.get("timing") or {}, row)
    row["wall_clock_s"] = (results.get("timing") or {}).get("wall_clock_s")

    best = results.get("best")
    if best:
        row["best.metric"] = best.get("metric")
        row["best.value"] = best.get("value")
        row["best.epoch"] = best.get("epoch")
        row["best.step"] = best.get("step")
        row["best.val_loss"] = best.get("val_loss")
        row["best.adapter"] = best.get("adapter")
        for key, value in (best.get("metrics") or {}).items():
            row[f"best.{key}"] = value
        for section, values in (best.get("sections") or {}).items():
            if isinstance(values, dict):
                for key, value in values.items():
                    row[f"best.{section}.{key}"] = value

    train_curve = (results.get("curves") or {}).get("train") or []
    val_curve = (results.get("curves") or {}).get("validation") or []
    if train_curve:
        row["last.train_loss"] = train_curve[-1].get("loss")
        row["last.step"] = train_curve[-1].get("step")
    if val_curve:
        row["last.val_loss"] = val_curve[-1].get("val_loss")

    if test:
        row["test.n_examples"] = test.get("n_examples")
        row["test.split"] = test.get("split")
        for section in ("findings", "impression"):
            values = (test.get("by_section") or {}).get(section)
            if isinstance(values, dict):
                for key, value in values.items():
                    row[f"test.{section}.{key}"] = value

    if baseline:
        sezioni: dict[str, float] = {}
        for section in ("findings", "impression"):
            values = (baseline.get("by_section") or {}).get(section)
            if isinstance(values, dict):
                for key, value in values.items():
                    sezioni[f"{section}.{key}"] = value
        for key, value in sezioni.items():
            row[f"baseline.{key}"] = value
        row["baseline.name"] = baseline.get("experiment")
        row["baseline.n_examples"] = baseline.get("n_examples")
        formato = baseline.get("format_compliance") or {}
        row["baseline.format_compliance"] = formato.get("ratio")
        row["baseline.format_missing"] = formato.get("missing")
        for key, value in sezioni.items():
            fine_tuned = row.get(f"test.{key}")
            if isinstance(fine_tuned, (int, float)) and isinstance(value, (int, float)):
                row[f"delta.{key}"] = round(fine_tuned - value, 6)

    for key, value in posthoc_metrics(test_chexbert, "test").items():
        row.setdefault(f"test.{key}", value)
    for key, value in posthoc_metrics(baseline_chexbert, "baseline").items():
        row.setdefault(f"baseline.{key}", value)
        fine_tuned = row.get(f"test.{key}")
        if isinstance(fine_tuned, (int, float)):
            row[f"delta.{key}"] = round(fine_tuned - value, 6)

    flatten("config", results.get("config") or {}, row)
    flatten("provenance", results.get("provenance") or {}, row)
    return row


def curve_points(results: dict[str, Any]) -> list[dict[str, Any]]:
    identity = results.get("identity") or {}
    base = {
        "experiment": results.get("experiment"),
        "lang": identity.get("lang"),
        "model": identity.get("model_short") or identity.get("model"),
        "mode": identity.get("mode"),
        "dataset_code": identity.get("dataset_code"),
    }
    points: list[dict[str, Any]] = []
    curves = results.get("curves") or {}

    for row in curves.get("train") or []:
        for metric in ("loss", "learning_rate"):
            if isinstance(row.get(metric), (int, float)):
                points.append({**base, "curve": "train", "step": row.get("step"),
                               "epoch": row.get("epoch"), "metric": metric,
                               "value": float(row[metric]),
                               "elapsed_s": row.get("elapsed_s")})

    for row in curves.get("validation") or []:
        common = {**base, "curve": "validation", "step": row.get("step"),
                  "epoch": row.get("epoch"), "elapsed_s": row.get("elapsed_s")}
        if isinstance(row.get("val_loss"), (int, float)):
            points.append({**common, "metric": "val_loss", "value": float(row["val_loss"])})
        for metric, value in (row.get("metrics") or {}).items():
            if metric != "num_examples" and isinstance(value, (int, float)):
                points.append({**common, "metric": metric, "value": float(value)})
        for section, values in (row.get("sections") or {}).items():
            if not isinstance(values, dict):
                continue
            for metric, value in values.items():
                if metric != "num_examples" and isinstance(value, (int, float)):
                    points.append({**common, "metric": f"{section}.{metric}",
                                   "value": float(value)})
    return points


def posthoc_curve(
    results: dict[str, Any], payload: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not payload:
        return []
    identity = results.get("identity") or {}
    base = {
        "experiment": results.get("experiment"),
        "lang": identity.get("lang"),
        "model": identity.get("model_short") or identity.get("model"),
        "mode": identity.get("mode"),
        "dataset_code": identity.get("dataset_code"),
        "curve": "validation",
    }
    points = []
    for group in payload.get("groups", []):
        common = {**base, "step": group.get("step"), "epoch": group.get("epoch"),
                  "elapsed_s": None}
        for key, value in (group.get("sectioned_flat") or {}).items():
            if key != "num_examples" and isinstance(value, (int, float)):
                points.append({**common, "metric": key, "value": float(value)})
        for key, value in (group.get("chexbert_en") or {}).items():
            if isinstance(value, (int, float)):
                points.append({
                    **common,
                    "metric": f"chexbert_en.{key.removeprefix('chexbert_')}",
                    "value": float(value),
                })
    return points


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-root", type=Path, default=ROOT / "training")
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "results")
    parser.add_argument("--only", action="append", default=[],
                        help="glob sul nome esperimento (ripetibile)")
    parser.add_argument("--sort", default="best.value",
                        help="colonna di ordinamento della tabella a schermo")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args(argv)

    identities = [
        identity
        for p in sorted(args.training_root.glob("*/*/*/*"))
        if p.is_dir()
        for identity in [Identity.from_path(p)]
        if identity.mode in TRAINING_MODES
    ]
    if args.only:
        identities = [
            i for i in identities
            if any(fnmatch.fnmatch(i.name, p) for p in args.only)
        ]

    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    missing: list[str] = []

    for identity in identities:
        results_dir = ROOT / identity.relpath / "results"
        path = results_dir / "results.json"
        if not path.is_file():
            missing.append(identity.name)
            continue
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! {identity.name}: results.json illeggibile ({exc})", file=sys.stderr)
            continue

        def _read(path: Path) -> dict | None:
            if not path.is_file():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None

        test = _read(results_dir / "test" / "metrics.json")
        baseline_dir = ROOT / identity.baseline.relpath / "results"
        baseline = _read(baseline_dir / "metrics.json")
        test_chexbert = _read(results_dir / "test" / "posthoc_metrics.json")
        baseline_chexbert = _read(baseline_dir / "posthoc_metrics.json")

        rows.append(
            experiment_row(results, test, baseline, test_chexbert, baseline_chexbert)
        )
        points.extend(curve_points(results))
        points.extend(
            posthoc_curve(
                results, _read(results_dir / "val_predictions" / "posthoc_metrics.json")
            )
        )

    if not rows:
        print(f"Nessun results.json trovato ({len(missing)} esperimenti senza risultati).")
        print("esegui prima un training: python training/run_all.py --dry-run")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.out / "experiments.json",
        {"n_experiments": len(rows), "experiments": rows},
    )
    write_json_atomic(
        args.out / "curves.json",
        {
            "n_points": len(points),
            "schema": ["experiment", "lang", "model", "mode", "dataset_code",
                       "curve", "step", "epoch", "metric", "value", "elapsed_s"],
            "points": points,
        },
    )
    if not args.no_csv:
        dump_csv(args.out / "experiments.csv", rows)
        dump_csv(args.out / "curves.csv", points)

    columns = [c for c in SCREEN_COLUMNS if any(c in row for row in rows)]
    key = args.sort if any(args.sort in row for row in rows) else "experiment"
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get(key) is None, r.get(key)),
        reverse=key != "experiment",
    )

    widths = {c: max(len(c), 10) for c in columns}
    for row in rows_sorted:
        for c in columns:
            widths[c] = max(widths[c], len(_fmt(row.get(c))))
    print(f"\n{len(rows)} esperimenti con risultati"
          + (f", {len(missing)} senza" if missing else "") + f"   (ordinati per {key})\n")
    print("  " + "".join(f"{c:>{widths[c] + 2}}" for c in columns))
    for row in rows_sorted:
        print("  " + "".join(f"{_fmt(row.get(c)):>{widths[c] + 2}}" for c in columns))

    print(f"\nscritti in {args.out.relative_to(ROOT)}/:")
    print(f"  experiments.json   {len(rows)} righe (una per esperimento) → tabelle")
    print(f"  curves.json        {len(points)} punti (formato lungo)     → grafici")
    if not args.no_csv:
        print("  experiments.csv, curves.csv")
    if missing:
        print(f"\nsenza results.json: {', '.join(missing)}")
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.0f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
