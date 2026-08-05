from __future__ import annotations

import csv
import gc
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
from .clinical import (
    build_balanced_clinical_records,
    build_clinical_records,
    build_stage_two_records,
)
from .dashboard import LiveDashboard, hms
from .results import ResultsWriter, flatten_validation_row, now_iso, write_json_atomic


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def validate_resume_adapter_path(results: Path, adapter: Path) -> Path:
    resolved_results = results.resolve()
    resolved_adapter = adapter.resolve()
    if resolved_adapter == resolved_results or resolved_results in resolved_adapter.parents:
        raise ValueError(
            "clinical_adapter_path deve essere fuori da results_dir per evitare "
            "che l'archiviazione cancelli l'adapter sorgente"
        )
    return resolved_adapter


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


def prepare_training_records(
    train_records: list[dict[str, Any]], cfg: Config
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if cfg.training_strategy == "standard":
        stage_two, stage_two_stats = build_stage_two_records(train_records, [], cfg)
        return [], stage_two, {"clinical": None, "stage_two": stage_two_stats}
    clinical_records: list[dict[str, Any]] = []
    clinical_stats: dict[str, Any] | None = None
    if cfg.training_strategy == "clinical":
        expected_images = 2 if cfg.views == "frontal_lateral" else 1
        clinical_source, clinical_stats = build_clinical_records(
            train_records,
            expected_images,
            cfg.clinical_target_format,
            cfg.clinical_include_fallback,
            cfg.clinical_include_fallback,
        )
        clinical_records = clinical_source
        if cfg.clinical_balance:
            clinical_records, sampling = build_balanced_clinical_records(
                clinical_source, cfg
            )
            clinical_stats["sampling"] = sampling
    else:
        clinical_source = []
    stage_two, stage_two_stats = build_stage_two_records(
        train_records, clinical_source, cfg
    )
    return clinical_records, stage_two, {
        "clinical": clinical_stats,
        "stage_two": stage_two_stats,
    }


def prepare_clinical_validation_records(
    val_records: list[dict[str, Any]], cfg: Config
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_images = 2 if cfg.views == "frontal_lateral" else 1
    return build_clinical_records(
        val_records,
        expected_images,
        cfg.clinical_target_format,
        False,
        cfg.clinical_include_fallback,
    )


def _component_rates(cfg: Config) -> dict[str, float]:
    return {
        "merger": cfg.merger_lr if cfg.merger_lr is not None else cfg.learning_rate,
        "vision": cfg.vision_lr if cfg.vision_lr is not None else cfg.learning_rate,
        "llm": cfg.learning_rate,
    }


def _component_of(name: str) -> str:
    if "visual.merger" in name:
        return "merger"
    if "visual." in name:
        return "vision"
    return "llm"


def build_optimizer(model: Any, cfg: Config) -> dict[str, Any]:
    if cfg.vision_lr is None and cfg.merger_lr is None:
        return {}
    if cfg.optim != "adamw_torch":
        raise RuntimeError(
            f"optim='{cfg.optim}' non supportato con learning rate per componente."
        )
    import torch

    rates = _component_rates(cfg)
    buckets: dict[str, list[Any]] = {group: [] for group in rates}
    for name, param in model.named_parameters():
        if param.requires_grad:
            buckets[_component_of(name)].append(param)

    groups = [
        {"params": params, "lr": rates[group], "weight_decay": cfg.weight_decay}
        for group, params in buckets.items()
        if params
    ]
    missing = [
        group
        for group, params in buckets.items()
        if not params and rates[group] != cfg.learning_rate
    ]
    if missing:
        raise RuntimeError(
            f"learning rate dedicato per {missing} ma nessun parametro addestrabile "
            "in quei gruppi: il valore verrebbe ignorato in silenzio."
        )
    for group, params in buckets.items():
        if params:
            print(f"[optim] {group}: {len(params)} tensori  lr={rates[group]:g}")
    optimizer = torch.optim.AdamW(
        groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    return {"optimizers": (optimizer, None)}


def _steps_per_epoch(record_count: int, cfg: Config) -> int:
    batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    return max(1, (record_count + batch - 1) // batch)


def _training_arguments(cfg: Config, output_dir: Path, epochs: int) -> Any:
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
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


def run_clinical_pretraining(
    model: Any,
    processor: Any,
    collator: Any,
    records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    cfg: Config,
    results: Path,
    writer: ResultsWriter,
    mlflow: Any | None,
) -> dict[str, Any]:
    from transformers import Trainer

    from .callbacks import LossLogger
    from .clinical_evaluation import ClinicalEvalCallback

    steps = _steps_per_epoch(len(records), cfg)
    dash = LiveDashboard(
        title=f"{cfg.experiment} clinical pretraining",
        total_steps=steps * cfg.clinical_pretrain_epochs,
    )
    evaluator = ClinicalEvalCallback(
        processor,
        val_records,
        cfg,
        results,
        dash,
        writer,
        mlflow,
    )
    trainer = Trainer(
        model=model,
        args=_training_arguments(
            cfg, results / "clinical_checkpoints", cfg.clinical_pretrain_epochs
        ),
        train_dataset=records,
        data_collator=collator,
        callbacks=[LossLogger(dash, mlflow, prefix="clinical.train"), evaluator],
        **build_optimizer(model, cfg),
    )
    dash.status = "clinical pretraining in corso…"
    started = time.time()
    train_result = trainer.train()
    evaluator.restore_best(model)
    if cfg.clinical_image_shuffle_eval:
        evaluator.evaluate_image_shuffle(model)
    duration = round(time.time() - started, 1)
    dash.stop()
    dash.status = "clinical pretraining terminato"
    dash.render()
    model.save_pretrained(results / "clinical_adapter")
    summary = {
        "status": "completed",
        "epochs": cfg.clinical_pretrain_epochs,
        "records": len(records),
        "runtime_s": duration,
        "trainer_runtime_s": train_result.metrics.get("train_runtime"),
        "history": dash.train_rows,
        "validation": evaluator.summary(),
        "adapter": str(results / "clinical_adapter"),
    }
    writer.set_in("stages", clinical=summary)
    writer.flush()
    if mlflow is not None:
        mlflow.log_metric("clinical.runtime_s", duration)
        mlflow.log_metric("clinical.records", len(records))
        mlflow.log_metric("clinical.epochs", cfg.clinical_pretrain_epochs)
        if evaluator.best is not None:
            mlflow.log_metric("clinical.best.f1_macro", evaluator.best["value"])
            mlflow.set_tag("clinical.best.epoch", evaluator.best["epoch"])
            mlflow.set_tag("clinical.best.step", evaluator.best["step"])
    del trainer
    gc.collect()
    return summary


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
    "clinical_training.json",
    "clinical_val_history.csv",
    "clinical_val_predictions_best.json",
    "clinical_image_shuffle.json",
    "clinical_val_predictions",
    "train_history.csv",
    "val_history.csv",
    "val_predictions_best.json",
    "val_predictions_best.csv",
    "val_predictions",
    "best_adapter",
    "clinical_adapter",
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
        if name in ("best_adapter", "clinical_adapter") and not include_adapter:
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
    from transformers import Trainer

    from ..data import load_records
    from .callbacks import GenerativeEvalEarlyStop, LossLogger
    from .model import (
        QwenVLCollator,
        load_model_and_processor,
        load_training_adapter,
        lora_adapter_report,
        sequence_length_probe,
    )

    script = Path(script_path).resolve()
    project_root = find_project_root(script.parent)
    identity = Identity.from_path(script)
    cfg: Config = build_config(identity, project_root, overrides)

    results = project_root / cfg.results_dir
    resume_adapter: Path | None = None
    if cfg.training_phase == "report_only":
        resume_adapter = validate_resume_adapter_path(
            results, project_root / cfg.clinical_adapter_path
        )
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
        import mlflow

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
    clinical_records, stage_two_records, training_stats = prepare_training_records(
        train_records, cfg
    )
    clinical_val_records: list[dict[str, Any]] = []
    if clinical_records:
        clinical_val_records, clinical_val_stats = (
            prepare_clinical_validation_records(val_records, cfg)
        )
        training_stats["clinical_validation"] = clinical_val_stats
    write_json_atomic(
        results / "clinical_training.json",
        {
            "experiment": cfg.experiment,
            "training_strategy": cfg.training_strategy,
            "training_phase": cfg.training_phase,
            "seed": cfg.seed,
            **training_stats,
        },
    )
    print(
        f"  strategia : {cfg.training_strategy}  "
        f"stage2={len(stage_two_records)}  clinical={len(clinical_records)}"
    )

    model, processor = load_model_and_processor(cfg, project_root)
    collator = QwenVLCollator(processor, cfg.max_seq_length)

    probe_records = [*clinical_records[:1], *stage_two_records[:1]]
    probe = collator(probe_records)
    n_loss_tokens = int((probe["labels"] != -100).sum())
    if n_loss_tokens == 0:
        raise RuntimeError(
            "Nessun token di loss: il marker dell'assistant non e' stato trovato."
        )
    print(f"  token di loss nel probe: {n_loss_tokens}")
    length_records = (
        clinical_records if cfg.training_phase == "clinical_only" else stage_two_records
    )
    length_stats = sequence_length_probe(processor, length_records, cfg.max_seq_length)

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
            "training_strategy": training_stats,
            "stages": {"clinical": None, "report_generation": None},
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
            "lora": {
                "adapters": lora_adapter_report(model),
                "learning_rates": _component_rates(cfg),
            },
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
                "training_strategy": cfg.training_strategy,
                "training_phase": cfg.training_phase,
                "clinical_pretrain_epochs": cfg.clinical_pretrain_epochs,
                "clinical_rehearsal_ratio": cfg.clinical_rehearsal_ratio,
                "clinical_balance": cfg.clinical_balance,
                "clinical_healthy_ratio": cfg.clinical_healthy_ratio,
                "clinical_sampling_strategy": cfg.clinical_sampling_strategy,
                "clinical_target_format": cfg.clinical_target_format,
                "clinical_include_fallback": cfg.clinical_include_fallback,
                "clinical_image_shuffle_eval": cfg.clinical_image_shuffle_eval,
                "clinical_max_new_tokens": cfg.clinical_max_new_tokens,
                "gpu": "; ".join(gpus),
                "python": platform.python_version(),
            },
        )

    steps_per_epoch = _steps_per_epoch(len(stage_two_records), cfg)
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

        t0 = time.time()
        import torch

        torch.cuda.reset_peak_memory_stats()
        stopper: Any = None
        try:
            clinical_summary: dict[str, Any] | None = None
            if cfg.training_phase == "report_only":
                if resume_adapter is None:
                    raise RuntimeError("Adapter di ripresa non risolto")
                adapter_provenance = load_training_adapter(model, resume_adapter)
                writer.set_in("provenance", clinical_adapter=adapter_provenance)
                writer.set_in(
                    "stages",
                    clinical={
                        "status": "loaded",
                        "adapter": adapter_provenance,
                    },
                )
                writer.flush()
            elif clinical_records:
                clinical_summary = run_clinical_pretraining(
                    model,
                    processor,
                    collator,
                    clinical_records,
                    clinical_val_records,
                    cfg,
                    results,
                    writer,
                    mlflow_mod,
                )
                torch.cuda.empty_cache()
            if cfg.training_phase == "clinical_only":
                if clinical_summary is None:
                    raise RuntimeError("clinical_only non ha eseguito il pretraining")
                best_clinical = clinical_summary["validation"]["best"]
                vram = _vram_peak()
                duration = round(time.time() - t0, 1)
                writer.set(status="completed")
                writer.set_best(best_clinical)
                writer.set(
                    n_evaluations=len(clinical_summary["validation"]["history"]),
                    early_stopped=False,
                    epochs_completed=float(cfg.clinical_pretrain_epochs),
                )
                writer.set_in(
                    "stages",
                    report_generation={"status": "skipped", "reason": "clinical_only"},
                )
                writer.set_in(
                    "timing",
                    finished_at=now_iso(),
                    wall_clock_s=duration,
                    train_runtime_s=clinical_summary["trainer_runtime_s"],
                )
                writer.set_in("environment", **vram)
                writer.flush()
                summary = writer.payload
                archived = (
                    archive_results(results, "completed", cfg.archive_adapter)
                    if cfg.archive_results
                    else None
                )
                if cfg.mlflow_enabled and mlflow_mod is not None:
                    mlflow_mod.set_tag("status", "completed")
                    mlflow_mod.set_tag("training_phase", cfg.training_phase)
                    for key, value in vram.items():
                        if isinstance(value, (int, float)):
                            mlflow_mod.log_metric(f"operational.{key}", float(value))
                    for name in (
                        "results.json",
                        "clinical_training.json",
                        "clinical_val_history.csv",
                        "clinical_val_predictions_best.json",
                        "clinical_image_shuffle.json",
                    ):
                        path = results / name
                        if path.is_file():
                            mlflow_mod.log_artifact(str(path), artifact_path="training")
                    from ..tracking import log_artifact_if_exists

                    log_artifact_if_exists(
                        results / "clinical_adapter",
                        "clinical_adapter",
                        allow_dir=True,
                    )
                    log_artifact_if_exists(
                        results / "clinical_val_predictions",
                        "clinical_val_predictions",
                        allow_dir=True,
                    )
                print("\n" + "=" * 78)
                print(f"TEMPO TOTALE          : {hms(duration)}")
                print(
                    f"best clinical f1_macro: {best_clinical['value']:.4f}  "
                    f"@ epoca {best_clinical['epoch']} / step {best_clinical['step']}"
                )
                print(f"risultati             : {results / 'results.json'}")
                if archived is not None:
                    print(f"archiviata copia in   : {archived.relative_to(project_root)}")
                return summary
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
            trainer = Trainer(
                model=model,
                args=_training_arguments(cfg, results / "checkpoints", cfg.max_epochs),
                train_dataset=stage_two_records,
                data_collator=collator,
                callbacks=[LossLogger(dash, mlflow_mod), stopper],
                **build_optimizer(model, cfg),
            )
            dash.status = "report generation training in corso…"
            train_result = trainer.train()
        except BaseException:
            writer.set(status="failed")
            writer.set_in("environment", **_vram_peak())
            history = stopper.history if stopper is not None else []
            best_payload = stopper.best_payload() if stopper is not None else None
            writer.set_curves(dash.train_rows, history)
            writer.set_best(best_payload)
            writer.set_in(
                "timing", finished_at=now_iso(), wall_clock_s=round(time.time() - t0, 1)
            )
            writer.flush()
            if cfg.archive_results:
                archive_results(results, "failed", cfg.archive_adapter)
            raise
        dash.stop()
        dash.status = "report generation training terminato"
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
            "stages",
            report_generation={
                "status": "early_stopped" if early_stopped else "completed",
                "records": len(stage_two_records),
                "epochs_completed": (
                    round(float(dash.train_rows[-1]["epoch"]), 3)
                    if dash.train_rows
                    else 0.0
                ),
                "runtime_s": train_result.metrics.get("train_runtime"),
                "evaluations": len(stopper.history),
            },
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
        _dump_csv(results / "train_history.csv", dash.train_rows)
        _dump_csv(
            results / "val_history.csv",
            [flatten_validation_row(row) for row in stopper.history],
        )
        summary = writer.payload
        archived = (
            archive_results(results, summary["status"], cfg.archive_adapter)
            if cfg.archive_results
            else None
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
                "clinical_training.json",
                "clinical_val_history.csv",
                "clinical_val_predictions_best.json",
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
            log_artifact_if_exists(
                results / "clinical_adapter", "clinical_adapter", allow_dir=True
            )
            log_artifact_if_exists(
                results / "clinical_val_predictions",
                "clinical_val_predictions",
                allow_dir=True,
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
    print(f"  python training/evaluate_test.py --results {cfg.results_dir}")
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
