#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CORE_CONFIG = (
    "base_model",
    "mode",
    "dataset_code",
    "views",
    "target",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "learning_rate",
    "vision_lr",
    "merger_lr",
    "optim",
    "lr_scheduler_type",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_epochs",
    "max_seq_length",
    "min_pixels",
    "max_pixels",
    "monitor_metric",
    "monitor_mode",
    "early_stopping_patience",
    "seed",
)

RUN_COLUMNS = (
    "run_idx",
    "archivio",
    "esperimento",
    "status",
    "mlflow_run_id",
    "mlflow_experiment_id",
    "started_at",
    "finished_at",
    "wall_clock_s",
    "epochs_completed",
    "early_stopped",
    "git_commit",
    "dvc_dataset_hash",
    "best_metric",
    "best_epoch",
    "best_step",
    "best_value",
)

EPOCH_COLUMNS = (
    "epoch",
    "step",
    "is_best",
    "train_loss",
    "val_loss",
    "eval_seconds",
    "elapsed_s",
)


def resolve_results(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for option in (candidate, candidate / "results"):
        if (option / "archive").is_dir() or (option / "results.json").is_file():
            return option
    raise SystemExit(f"Nessuna cartella results con archive sotto {path}")


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[salto] {path}: {exc}", file=sys.stderr)
        return None


def collect(results: Path, include_live: bool) -> list[tuple[str, dict[str, Any]]]:
    runs: list[tuple[str, dict[str, Any]]] = []
    for folder in sorted((results / "archive").glob("*")):
        payload = load(folder / "results.json") if folder.is_dir() else None
        if payload:
            runs.append((folder.name, payload))
    runs.sort(key=lambda item: item[1].get("timing", {}).get("started_at") or "")
    if include_live:
        payload = load(results / "results.json")
        if payload:
            seen = {run_id(p) for _, p in runs}
            if run_id(payload) not in seen:
                runs.append(("live", payload))
    return runs


def run_id(payload: dict[str, Any]) -> str:
    return (payload.get("provenance", {}).get("mlflow") or {}).get("run_id") or ""


def flatten(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def train_loss_by_epoch(payload: dict[str, Any]) -> dict[int, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for point in payload.get("curves", {}).get("train", []):
        epoch = point.get("epoch")
        loss = point.get("loss")
        if epoch is None or loss is None:
            continue
        buckets[max(1, math.ceil(float(epoch)))].append(float(loss))
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def all_config_keys(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    keys: list[str] = []
    for _, payload in runs:
        for key in payload.get("config", {}):
            if key not in keys:
                keys.append(key)
    return keys


def varying_keys(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [
        key
        for key in all_config_keys(runs)
        if len({flatten(p.get("config", {}).get(key)) for _, p in runs}) > 1
    ]


def config_columns(runs: list[tuple[str, dict[str, Any]]], everything: bool) -> list[str]:
    keys = all_config_keys(runs)
    if everything:
        return keys
    ordered = [k for k in CORE_CONFIG if k in keys]
    ordered += [k for k in varying_keys(runs) if k not in ordered]
    return ordered


def metric_columns(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    keys: list[str] = []
    for _, payload in runs:
        for entry in payload.get("curves", {}).get("validation", []):
            for key in entry.get("metrics", {}):
                if key not in keys:
                    keys.append(key)
    return keys


def run_fields(index: int, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    timing = payload.get("timing", {})
    provenance = payload.get("provenance", {})
    best = payload.get("best") or {}
    return {
        "run_idx": index,
        "archivio": name,
        "esperimento": payload.get("experiment", ""),
        "status": payload.get("status", ""),
        "mlflow_run_id": run_id(payload),
        "mlflow_experiment_id": (provenance.get("mlflow") or {}).get("experiment_id", ""),
        "started_at": timing.get("started_at", ""),
        "finished_at": timing.get("finished_at", ""),
        "wall_clock_s": timing.get("wall_clock_s", ""),
        "epochs_completed": payload.get("epochs_completed", ""),
        "early_stopped": payload.get("early_stopped", ""),
        "git_commit": (provenance.get("git") or {}).get("git.commit", ""),
        "dvc_dataset_hash": (provenance.get("dvc") or {}).get("dataset_hash", ""),
        "best_metric": best.get("metric", ""),
        "best_epoch": best.get("epoch", ""),
        "best_step": best.get("step", ""),
        "best_value": best.get("value", ""),
    }


def build_rows(
    runs: list[tuple[str, dict[str, Any]]],
    metrics: list[str],
    configs: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (name, payload) in enumerate(runs, 1):
        base = run_fields(index, name, payload)
        config = payload.get("config", {})
        for key in configs:
            base[key] = flatten(config.get(key, ""))
        losses = train_loss_by_epoch(payload)
        best_step = (payload.get("best") or {}).get("step")
        for entry in payload.get("curves", {}).get("validation", []):
            epoch = entry.get("epoch")
            row = dict(base)
            row["epoch"] = epoch
            row["step"] = entry.get("step", "")
            row["is_best"] = entry.get("step") == best_step
            row["train_loss"] = losses.get(
                max(1, math.ceil(float(epoch))) if epoch is not None else 0, ""
            )
            row["val_loss"] = entry.get("val_loss", "")
            row["eval_seconds"] = entry.get("eval_seconds", "")
            row["elapsed_s"] = entry.get("elapsed_s", "")
            values = entry.get("metrics", {})
            for key in metrics:
                row[key] = values.get(key, "")
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Storia per epoca di tutte le run archiviate di un esperimento."
    )
    parser.add_argument("path", type=Path, help="cartella esperimento o la sua results/")
    parser.add_argument("--out", type=Path, help="default: <results>/config_evolution.csv")
    parser.add_argument("--include-live", action="store_true",
                        help="aggiunge la run in results/ se non e' gia' archiviata")
    parser.add_argument("--all-config", action="store_true",
                        help="tutte le chiavi di config, non solo quelle che variano")
    args = parser.parse_args(argv)

    results = resolve_results(args.path)
    runs = collect(results, args.include_live)
    if not runs:
        raise SystemExit(f"Nessuna run archiviata in {results / 'archive'}")

    metrics = metric_columns(runs)
    configs = config_columns(runs, args.all_config)
    rows = build_rows(runs, metrics, configs)
    columns = list(RUN_COLUMNS) + list(EPOCH_COLUMNS) + metrics + configs

    destination = args.out or results / "config_evolution.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    try:
        shown = results.relative_to(ROOT)
    except ValueError:
        shown = results
    print(f"esperimento : {runs[0][1].get('experiment', '')}")
    print(f"risultati   : {shown}")
    print(f"run trovate : {len(runs)}   righe: {len(rows)}   colonne: {len(columns)}")
    for index, (name, payload) in enumerate(runs, 1):
        timing = payload.get("timing", {})
        best = payload.get("best") or {}
        print(
            f"  {index:>2}  {name:<32}{payload.get('status', ''):<11}"
            f"{(timing.get('started_at') or '')[:19]:<21}"
            f"{len(payload.get('curves', {}).get('validation', [])):>3} ep   "
            f"{best.get('metric', '')}={best.get('value', 0) or 0:.4f} @ep{best.get('epoch', '')}"
        )
    varying = varying_keys(runs)
    if varying:
        print(f"\nconfig che cambia tra le run: {', '.join(varying)}")
    else:
        print("\nnessuna differenza di config tra le run")
    print(f"\nCSV: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
