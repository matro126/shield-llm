from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from .evaluation import compute_val_loss, evaluate_generative, format_compliance
from ..data.prompts import split_sections
from ..evaluation import preload_metric_models
from .results import (
    flatten_validation_row,
    validation_row,
    write_json_atomic,
)


class LossLogger(TrainerCallback):
    def __init__(
        self,
        dashboard: Any,
        mlflow: Any | None = None,
        prefix: str = "train",
    ):
        self.dash = dashboard
        self.mlflow = mlflow
        self.prefix = prefix

    def on_train_begin(self, args, state, control, **kwargs):
        self.dash.start()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        self.dash.log_step(state.global_step, state.epoch or 0.0, logs)
        if self.mlflow is not None:
            for key in ("loss", "learning_rate", "grad_norm"):
                if key in logs and isinstance(logs[key], (int, float)):
                    self.mlflow.log_metric(
                        f"{self.prefix}.{key}",
                        float(logs[key]),
                        step=state.global_step,
                    )
        return control


class GenerativeEvalEarlyStop(TrainerCallback):
    def __init__(
        self,
        cfg: Any,
        dashboard: Any,
        processor: Any,
        collator: Any,
        val_records: list[dict[str, Any]],
        results_dir: Path,
        project_root: Path,
        writer: Any,
        mlflow: Any | None = None,
    ):
        self.cfg = cfg
        self.dash = dashboard
        self.processor = processor
        self.collator = collator
        self.results_dir = results_dir
        self.project_root = project_root
        self.writer = writer
        self.mlflow = mlflow
        self.records = (
            val_records
            if cfg.eval_max_samples is None
            else val_records[: cfg.eval_max_samples]
        )
        self.best: float | None = None
        self.best_row: dict[str, Any] | None = None
        self.since_improved = 0
        self.history: list[dict[str, Any]] = []

        caricati = preload_metric_models(
            list(cfg.eval_metrics),
            getattr(cfg, "chexbert_translate", False),
            getattr(cfg, "chexbert_translator", ""),
            bertscore_model_type=cfg.bertscore_model,
        )
        if caricati:
            print(f"[eval] modelli di metrica caricati in memoria: {', '.join(caricati)}")


    def _gate_value(self, row: dict[str, Any]) -> float | None:
        if self.cfg.monitor_metric == "val_loss":
            return float(row["val_loss"])
        value = row["metrics"].get(self.cfg.monitor_metric)
        return None if value is None else float(value)

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        delta = (
            value - self.best if self.cfg.monitor_mode == "max" else self.best - value
        )
        return delta > self.cfg.early_stopping_min_delta


    def _run_eval(self, state, control, model) -> None:
        epoch = float(state.epoch or 0.0)
        tag = f"epoca {epoch:.2f} / step {state.global_step}"
        t_start = time.time()

        self.dash.status = f"validation loss — {tag}"
        val_loss = compute_val_loss(
            model,
            self.collator,
            self.records,
            self.cfg.per_device_eval_batch_size,
            progress=lambda done, total: self.dash.log_progress(
                "VALIDATION loss", done, total, t_start
            ),
        )

        self.dash.status = f"generazione validation — {tag}"
        t_gen = time.time()
        sectioned, predictions, references = evaluate_generative(
            model,
            self.processor,
            self.records,
            self.cfg.eval_metrics,
            self.cfg.target,
            self.cfg.gen_batch_size,
            self.cfg.max_new_tokens,
            self.cfg.repetition_penalty,
            progress=lambda done, total: self.dash.log_progress(
                "VALIDATION generazione", done, total, t_gen
            ),
            chexbert_translate=self.cfg.chexbert_translate,
            chexbert_translator=self.cfg.chexbert_translator,
            bertscore_model_type=self.cfg.bertscore_model,
        )

        row = validation_row(
            epoch=epoch,
            step=state.global_step,
            val_loss=val_loss,
            sectioned=sectioned,
            eval_seconds=time.time() - t_start,
            elapsed_s=self.dash.elapsed(),
            format_compliance=format_compliance(
                self.records, predictions, self.cfg.target
            ),
        )
        self.history.append(row)
        self._dump_samples(row, predictions, references)
        flat = flatten_validation_row(row)
        self.dash.log_val(flat)

        value = self._gate_value(row)

        if self.mlflow is not None:
            self.mlflow.log_metric("val.loss", float(val_loss), step=state.global_step)
            if value is not None:
                self.mlflow.log_metric(
                    "val.gate", float(value), step=state.global_step
                )
            for key, item in flat.items():
                if key in ("epoch", "step", "val_loss") or not isinstance(
                    item, (int, float)
                ):
                    continue
                if "chexbert_cls_" in key:
                    continue
                self.mlflow.log_metric(
                    f"val.{key}".replace(" ", "_"), float(item), step=state.global_step
                )

        self._save_ogni_eval(model, row)

        if value is not None and self._improved(float(value)):
            self.best = float(value)
            self.best_row = row
            self.since_improved = 0
            self._save_best(model, row)
            self.dash.status = (
                f"{tag}: nuovo best {self.cfg.monitor_metric}={self.best:.4f} "
                "→ adapter salvato"
            )
            self._dump_samples(row, predictions, references, best=True)
        else:
            self.since_improved += 1
            self.dash.status = (
                f"{tag}: nessun miglioramento "
                f"({self.since_improved}/{self.cfg.early_stopping_patience})"
            )

        self._persist()

        if self.since_improved >= self.cfg.early_stopping_patience:
            self.dash.status = (
                f"EARLY STOPPING a {tag} — best {self.cfg.monitor_metric}="
                f"{self.best:.4f} @ epoca {self.best_row['epoch']}"
            )
            control.should_training_stop = True
        self.dash.render()


    def best_payload(self) -> dict[str, Any] | None:
        if self.best_row is None:
            return None
        return {
            "metric": self.cfg.monitor_metric,
            "mode": self.cfg.monitor_mode,
            "value": self.best,
            "epoch": self.best_row["epoch"],
            "step": self.best_row["step"],
            "val_loss": self.best_row["val_loss"],
            "metrics": self.best_row["metrics"],
            "sections": self.best_row["sections"],
            "format_compliance": self.best_row.get("format_compliance"),
            "adapter": str(
                (self.results_dir / "best_adapter").relative_to(self.project_root)
            ),
        }

    def _save_adapter(
        self, model: Any, row: dict[str, Any], destination: Path, value: Any
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination)
        (destination / "best_info.json").write_text(
            json.dumps(
                {
                    "experiment": self.cfg.experiment,
                    "metric": self.cfg.monitor_metric,
                    "value": value,
                    "epoch": row["epoch"],
                    "step": row["step"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _save_best(self, model: Any, row: dict[str, Any]) -> None:
        self._save_adapter(model, row, self.results_dir / "best_adapter", self.best)

    def _save_ogni_eval(self, model: Any, row: dict[str, Any]) -> None:
        if not getattr(self.cfg, "save_every_eval", False):
            return
        destination = self.results_dir / "adapters" / f"step{int(row['step']):06d}"
        self._save_adapter(model, row, destination, self._gate_value(row))

    def _persist(self) -> None:
        self.writer.set_curves(self.dash.train_rows, self.history)
        self.writer.set_best(self.best_payload())
        self.writer.flush()

    def _sample_payload(
        self, row: dict[str, Any], predictions: list[str], references: list[str]
    ) -> dict[str, Any]:
        samples = []
        for record, reference, prediction in zip(
            self.records, references, predictions
        ):
            ref_findings, ref_impression = split_sections(reference)
            pred_findings, pred_impression = split_sections(prediction)
            samples.append(
                {
                    "id": record["id"],
                    "factors": record.get("factors", {}),
                    "images": record.get("images", []),
                    "reference": reference,
                    "prediction": prediction,
                    "reference_sections": {
                        "findings": ref_findings,
                        "impression": ref_impression,
                    },
                    "prediction_sections": {
                        "findings": pred_findings,
                        "impression": pred_impression,
                    },
                }
            )
        return {
            "schema_version": 1,
            "experiment": self.cfg.experiment,
            "split": self.cfg.val_split,
            "target": self.cfg.target,
            "epoch": row["epoch"],
            "step": row["step"],
            "n_examples": len(samples),
            "metrics": row["metrics"],
            "sections": row["sections"],
            "format_compliance": row.get("format_compliance"),
            "samples": samples,
        }

    def _dump_samples(
        self,
        row: dict[str, Any],
        predictions: list[str],
        references: list[str],
        best: bool = False,
    ) -> None:
        payload = self._sample_payload(row, predictions, references)
        folder = self.results_dir / "val_predictions"
        folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(folder / f"step{row['step']:06d}.json", payload)
        if best:
            write_json_atomic(self.results_dir / "val_predictions_best.json", payload)
            mancanti = set(
                (payload.get("format_compliance") or {}).get("missing_ids") or ()
            )
            path = self.results_dir / "val_predictions_best.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["id", "reference", "prediction", "has_sep"])
                for sample in payload["samples"]:
                    writer.writerow([
                        sample["id"], sample["reference"], sample["prediction"],
                        int(str(sample["id"]) not in mancanti),
                    ])


    def on_epoch_end(self, args, state, control, **kwargs):
        if self.cfg.eval_cadence == "epoch":
            self._run_eval(state, control, kwargs["model"])
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if (
            self.cfg.eval_cadence == "steps"
            and state.global_step > 0
            and state.global_step % self.cfg.eval_steps == 0
        ):
            self._run_eval(state, control, kwargs["model"])
        return control
