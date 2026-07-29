from __future__ import annotations

import csv
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .config import Config, Identity, build_config
from .dashboard import LiveDashboard, hms
from .results import ResultsWriter, flatten_validation_row, now_iso


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _vram_peak() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    allocated = [
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    ]
    reserved = [
        torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())
    ]
    return {
        "vram_peak_allocated_gb": round(max(allocated) / 1e9, 2),
        "vram_peak_reserved_gb": round(max(reserved) / 1e9, 2),
        "vram_peak_allocated_gb_per_device": [round(v / 1e9, 2) for v in allocated],
        "vram_total_gb": round(
            max(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            )
            / 1e9,
            1,
        ),
    }


RUN_ARTIFACTS = (
    "results.json",
    "train_history.csv",
    "val_history.csv",
    "val_predictions_best.json",
    "val_predictions_best.csv",
    "val_predictions",
    "best_adapter",
    "adapters",
    "test",
)


BASELINE_ARTIFACTS = (
    "metrics.json",
    "predictions.csv",
    "disaggregated.json",
    "posthoc_metrics.json",
)
TEST_ARTIFACTS = ("test",)


def archive_results(
    results: Path,
    status: str,
    include_adapter: bool = True,
    artifacts: tuple[str, ...] = RUN_ARTIFACTS,
    marker: str = "results.json",
) -> Path | None:
    if not (results / marker).is_file():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = results / "archive" / f"{stamp}-{status}"
    suffix = 2
    while destination.exists():
        destination = results / "archive" / f"{stamp}-{status}-{suffix}"
        suffix += 1
    destination.mkdir(parents=True)
    for name in artifacts:
        if name == "best_adapter" and not include_adapter:
            continue
        source = results / name
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, destination / name)
    return destination


def clear_live_results(
    results: Path, artifacts: tuple[str, ...] = RUN_ARTIFACTS
) -> None:
    for name in artifacts:
        path = results / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def archive_previous(
    results: Path, artifacts: tuple[str, ...], status: str, marker: str
) -> Path | None:
    archived = archive_results(
        results, status, artifacts=artifacts, marker=marker
    )
    if archived is not None:
        clear_live_results(results, artifacts)
    return archived


def _previous_status(results: Path) -> str | None:
    path = results / "results.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except json.JSONDecodeError:
        return "unreadable"


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return None


def _require_gpu() -> list[str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nessuna GPU disponibile: questo esperimento va eseguito sul server."
        )
    names = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        names.append(f"{props.name} ({props.total_memory / 1e9:.1f} GB)")
        print(f"  GPU {i}: {names[-1]}")
    return names


def run_experiment(
    script_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    from ..data import load_records
    from .callbacks import GenerativeEvalEarlyStop, LossLogger
    from .model import QwenVLCollator, load_model_and_processor, sequence_length_probe

    script = Path(script_path).resolve()
    project_root = find_project_root(script.parent)
    identity = Identity.from_path(script)
    cfg: Config = build_config(identity, project_root, overrides)

    results = project_root / cfg.results_dir
    results.mkdir(parents=True, exist_ok=True)

    previous = _previous_status(results)
    if previous is not None:
        if cfg.archive_results:
            if previous in ("running", "unreadable"):
                archived = archive_results(results, "interrupted", cfg.archive_adapter)
                print(f"  archivio  : run precedente interrotta → {archived}")
            else:
                print(f"  archivio  : run precedente ({previous}) gia' archiviata")
        else:
            print(f"  archivio  : disattivato, la run precedente ({previous}) va persa")
        clear_live_results(results)

    print(f"═══ esperimento: {cfg.experiment}")
    print(f"  modello   : {cfg.base_model}  ({cfg.mode}, 4bit={cfg.load_in_4bit})")
    print(f"  dataset   : {cfg.dataset_root}  (views={cfg.views}, target={cfg.target})")
    print(f"  risultati : {results}")
    _seed_everything(cfg.seed)
    gpus = _require_gpu()

    if cfg.mlflow_enabled:
        import mlflow  # noqa: F401

        print(
            f"  mlflow    : {cfg.mlflow_tracking_uri or 'default (env o 127.0.0.1:5000)'}"
        )

    root = project_root / cfg.dataset_root
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset assente: {root}\n"
            "Costruiscilo con: uv run python -m shield.data.build --all"
        )
    train_records = load_records(root, cfg.train_split, images_root=root)
    val_records = load_records(root, cfg.val_split, images_root=root)
    print(f"  train={len(train_records)}  val={len(val_records)}")

    model, processor = load_model_and_processor(cfg, project_root)
    collator = QwenVLCollator(processor, cfg.max_seq_length)

    probe = collator(train_records[:2])
    n_loss_tokens = int((probe["labels"] != -100).sum())
    if n_loss_tokens == 0:
        raise RuntimeError(
            "Nessun token di loss: il marker dell'assistant non e' stato trovato."
        )
    print(f"  token di loss nel probe: {n_loss_tokens}")
    length_stats = sequence_length_probe(processor, train_records, cfg.max_seq_length)

    from ..tracking import (
        dvc_dataset_hash,
        git_metadata,
        metric_provenance,
        model_provenance,
    )

    writer = ResultsWriter(
        results / "results.json",
        {
            "experiment": cfg.experiment,
            "identity": {
                "lang": cfg.lang,
                "model": cfg.model_dir,
                "model_short": identity.model_short,
                "mode": cfg.mode,
                "dataset_code": cfg.dataset_code,
                "views": cfg.views,
                "target": cfg.target,
            },
            "dataset": {
                "root": cfg.dataset_root,
                "version": Path(cfg.dataset_root).name,
                "n_train": len(train_records),
                "n_val": len(val_records),
            },
            "config": cfg.as_dict(),
            "environment": {
                "gpus": gpus,
                "python": platform.python_version(),
                "torch": _torch_version(),
            },
            "provenance": {
                "git": git_metadata(project_root),
                "dvc": {
                    "dataset_hash": dvc_dataset_hash(cfg.dataset_root, project_root)
                    or "unavailable"
                },
                "model": model_provenance(cfg.as_dict(), project_root),
                "metrics": metric_provenance(cfg.as_dict(), project_root),
                "mlflow": None,
            },
            "sequence_length": length_stats,
            "timing": {"started_at": now_iso()},
        },
    )

    mlflow_ctx: Any = nullcontext()
    mlflow_mod: Any = None
    if cfg.mlflow_enabled:
        import mlflow as mlflow_mod

        from ..tracking import mlflow_run

        tracking_config = {
            "experiment": {"name": cfg.experiment},
            "dataset": {
                "root": cfg.dataset_root,
                "version": Path(cfg.dataset_root).name,
            },
            "model": {"base_model": cfg.base_model, "mode": cfg.mode},
            "training": cfg.as_dict(),
            "mlflow": {
                "tracking_uri": cfg.mlflow_tracking_uri or None,
                "experiment_name": cfg.mlflow_experiment_name or None,
            },
        }
        mlflow_ctx = mlflow_run(
            tracking_config,
            root=project_root,
            run_name=cfg.experiment,
            tags={
                "lang": cfg.lang,
                "model": cfg.model_dir,
                "mode": cfg.mode,
                "dataset_code": cfg.dataset_code,
                "views": cfg.views,
                "target": cfg.target,
                "gpu": "; ".join(gpus),
                "python": platform.python_version(),
            },
        )

    steps_per_epoch = max(
        1,
        len(train_records)
        // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps),
    )
    dash = LiveDashboard(
        title=f"{cfg.experiment}  ({cfg.base_model})",
        total_steps=steps_per_epoch * cfg.max_epochs,
    )

    with mlflow_ctx:
        if cfg.mlflow_enabled and mlflow_mod is not None:
            run = mlflow_mod.active_run()
            writer.set_in(
                "provenance",
                mlflow={
                    "run_id": run.info.run_id if run else None,
                    "experiment_id": run.info.experiment_id if run else None,
                    "tracking_uri": mlflow_mod.get_tracking_uri(),
                },
            )
            writer.flush()

        stopper = GenerativeEvalEarlyStop(
            cfg,
            dash,
            processor,
            collator,
            val_records,
            results,
            project_root,
            writer,
            mlflow_mod,
        )
        args = TrainingArguments(
            output_dir=str(results / "checkpoints"),
            num_train_epochs=cfg.max_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler_type,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=cfg.logging_steps,
            eval_strategy="no",
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=cfg.dataloader_num_workers,
            dataloader_persistent_workers=cfg.dataloader_persistent_workers,
            seed=cfg.seed,
            full_determinism=cfg.full_determinism,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_records,
            data_collator=collator,
            callbacks=[LossLogger(dash, mlflow_mod), stopper],
        )

        dash.status = "training in corso…"
        t0 = time.time()
        import torch

        torch.cuda.reset_peak_memory_stats()
        try:
            train_result = trainer.train()
        except BaseException:
            writer.set(status="failed")
            writer.set_in("environment", **_vram_peak())
            writer.set_curves(dash.train_rows, stopper.history)
            writer.set_best(stopper.best_payload())
            writer.set_in(
                "timing", finished_at=now_iso(), wall_clock_s=round(time.time() - t0, 1)
            )
            writer.flush()
            if cfg.archive_results:
                archive_results(results, "failed", cfg.archive_adapter)
            raise
        dash.stop()
        dash.status = "training terminato"
        dash.render()

        early_stopped = stopper.since_improved >= cfg.early_stopping_patience
        writer.set(status="early_stopped" if early_stopped else "completed")
        writer.set_curves(dash.train_rows, stopper.history)
        writer.set_best(stopper.best_payload())
        writer.set(
            n_evaluations=len(stopper.history),
            early_stopped=early_stopped,
            epochs_completed=(
                round(float(dash.train_rows[-1]["epoch"]), 3)
                if dash.train_rows
                else 0.0
            ),
        )
        writer.set_in(
            "timing",
            finished_at=now_iso(),
            wall_clock_s=round(time.time() - t0, 1),
            train_runtime_s=train_result.metrics.get("train_runtime"),
        )
        vram = _vram_peak()
        writer.set_in("environment", **vram)
        writer.flush()
        summary = writer.payload
        archived = (
            archive_results(results, summary["status"], cfg.archive_adapter)
            if cfg.archive_results
            else None
        )

        _dump_csv(results / "train_history.csv", dash.train_rows)
        _dump_csv(
            results / "val_history.csv",
            [flatten_validation_row(row) for row in stopper.history],
        )

        if cfg.mlflow_enabled and mlflow_mod is not None:
            best = stopper.best_payload()
            if best:
                mlflow_mod.log_metric(f"best.{cfg.monitor_metric}", best["value"])
                for key, value in best["metrics"].items():
                    if isinstance(value, (int, float)):
                        mlflow_mod.log_metric(f"best.{key}", float(value))
                mlflow_mod.set_tag("best.epoch", best["epoch"])
                mlflow_mod.set_tag("best.step", best["step"])
            mlflow_mod.set_tag("status", summary["status"])
            for key, value in vram.items():
                if isinstance(value, (int, float)):
                    mlflow_mod.log_metric(f"operational.{key}", float(value))
            for name in (
                "results.json",
                "train_history.csv",
                "val_history.csv",
                "val_predictions_best.csv",
            ):
                path = results / name
                if path.is_file():
                    mlflow_mod.log_artifact(str(path), artifact_path="training")
            from ..tracking import log_artifact_if_exists

            log_artifact_if_exists(
                results / "best_adapter", "best_adapter", allow_dir=True
            )

    best = summary.get("best")
    print("\n" + "=" * 78)
    print(f"TEMPO TOTALE          : {hms(summary['timing']['wall_clock_s'])}")
    print(f"valutazioni eseguite  : {summary['n_evaluations']}")
    if best:
        print(
            f"best {cfg.monitor_metric:<17}: {best['value']:.4f}  "
            f"@ epoca {best['epoch']} / step {best['step']}"
        )
    print(f"early stopping        : {'si' if summary['early_stopped'] else 'no'}")
    if vram:
        print(
            f"VRAM picco            : {vram['vram_peak_reserved_gb']:.1f} GB riservati "
            f"({vram['vram_peak_allocated_gb']:.1f} allocati) "
            f"su {vram['vram_total_gb']:.0f} GB"
        )
    print(f"risultati             : {results / 'results.json'}")
    if archived is not None:
        print(f"archiviata copia in   : {archived.relative_to(project_root)}")
    print("\nvalutazione sul test set (processo a parte):")
    print(f"  python training/evaluate_test.py --experiment {cfg.experiment}")
    return summary


def main(script_path: str, overrides: dict[str, Any] | None = None) -> int:
    try:
        run_experiment(script_path, overrides)
    except KeyboardInterrupt:
        print("\ninterrotto dall'utente: i risultati parziali sono su disco.")
        return 130
    except Exception as exc:
        print(f"\nESPERIMENTO FALLITO: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 0
