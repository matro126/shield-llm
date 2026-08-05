from __future__ import annotations

import csv
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from .clinical import CLINICAL_LABELS, NO_FINDING, shuffle_clinical_images
from .results import write_json_atomic

CLINICAL_EVAL_LABELS = (*CLINICAL_LABELS, NO_FINDING)
CLINICAL_AGGREGATE_METRICS = (
    "accuracy_exact",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
)


def _clinical_lines(text: str) -> list[str]:
    normalized = re.sub(r"[;,]", "\n", str(text))
    return [
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().rstrip(".:")
        for line in normalized.splitlines()
    ]


def _dense_status_data(
    text: str,
) -> tuple[dict[str, list[str]], list[str]]:
    lookup = {label.casefold(): label for label in CLINICAL_LABELS}
    statuses: dict[str, list[str]] = {}
    invalid = set()
    for line in _clinical_lines(text):
        if ":" not in line:
            continue
        raw_label, raw_status = line.rsplit(":", 1)
        label = lookup.get(raw_label.strip().casefold())
        if label is None:
            continue
        status = raw_status.strip().rstrip(".").casefold()
        if status not in ("present", "absent"):
            invalid.add(label)
            continue
        statuses.setdefault(label, []).append(status)
    return statuses, [label for label in CLINICAL_LABELS if label in invalid]


def dense_clinical_format(text: str) -> dict[str, Any]:
    statuses, invalid = _dense_status_data(text)
    missing = [label for label in CLINICAL_LABELS if label not in statuses]
    duplicates = [label for label in CLINICAL_LABELS if len(statuses.get(label, [])) > 1]
    return {
        "complete": not missing and not duplicates and not invalid,
        "recognized_count": len(statuses),
        "missing_labels": missing,
        "duplicate_labels": duplicates,
        "invalid_labels": invalid,
    }


def parse_clinical_labels(
    text: str, target_format: str = "positive_only"
) -> list[str]:
    statuses, invalid = _dense_status_data(text)
    if statuses or invalid:
        present = {
            label for label, values in statuses.items() if "present" in values
        }
        if present:
            return [label for label in CLINICAL_LABELS if label in present]
        if dense_clinical_format(text)["complete"]:
            return [NO_FINDING]
        return []
    if target_format == "dense_binary":
        return []
    lines = _clinical_lines(text)
    lookup = {label.casefold(): label for label in CLINICAL_EVAL_LABELS}
    present = {lookup[line.casefold()] for line in lines if line.casefold() in lookup}
    return [label for label in CLINICAL_EVAL_LABELS if label in present]


def clinical_classification_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    target_format: str = "positive_only",
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Predizioni e riferimenti devono avere la stessa lunghezza")
    if not references:
        raise ValueError("La validation clinica non puo' essere vuota")

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import MultiLabelBinarizer

    encoder = MultiLabelBinarizer(classes=CLINICAL_EVAL_LABELS)
    encoder.fit([CLINICAL_EVAL_LABELS])
    expected = encoder.transform(
        [parse_clinical_labels(text, target_format) for text in references]
    )
    predicted = encoder.transform(
        [parse_clinical_labels(text, target_format) for text in predictions]
    )
    result = {"accuracy_exact": float(accuracy_score(expected, predicted))}

    for average in ("macro", "micro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected,
            predicted,
            average=average,
            zero_division=0,
        )
        result[f"precision_{average}"] = float(precision)
        result[f"recall_{average}"] = float(recall)
        result[f"f1_{average}"] = float(f1)

    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        average=None,
        zero_division=0,
    )
    for index, label in enumerate(CLINICAL_EVAL_LABELS):
        key = label.replace(" ", "_").replace("/", "_")
        result[f"cls_{key}_precision"] = float(precision[index])
        result[f"cls_{key}_recall"] = float(recall[index])
        result[f"cls_{key}_f1"] = float(f1[index])
        result[f"cls_{key}_support"] = float(support[index])
    return result


def clinical_mlflow_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if key in CLINICAL_AGGREGATE_METRICS
        or key.startswith("cls_") and key.endswith("_f1")
    }


def clinical_validation_payload(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    references: Sequence[str],
    epoch: float,
    step: int,
    metrics: dict[str, float],
    target_format: str = "positive_only",
) -> dict[str, Any]:
    if not (len(records) == len(predictions) == len(references)):
        raise ValueError(
            "Record, predizioni e riferimenti devono avere la stessa lunghezza"
        )
    samples = [
        {
            "id": record.get("id"),
            "labels": parse_clinical_labels(reference, target_format),
            "predicted_labels": parse_clinical_labels(prediction, target_format),
            "reference": reference,
            "prediction": prediction,
        }
        for record, prediction, reference in zip(records, predictions, references)
    ]
    payload = {
        "schema_version": 1,
        "split": "val",
        "task": "clinical_classification",
        "epoch": float(epoch),
        "step": int(step),
        "n_examples": len(samples),
        "metrics": metrics,
        "samples": samples,
    }
    if target_format == "dense_binary":
        incomplete = []
        for sample in samples:
            diagnostic = dense_clinical_format(sample["prediction"])
            sample["format"] = diagnostic
            if not diagnostic["complete"]:
                incomplete.append(sample["id"])
        complete = len(samples) - len(incomplete)
        payload["format_compliance"] = {
            "complete": complete,
            "total": len(samples),
            "ratio": complete / max(len(samples), 1),
            "incomplete_ids": incomplete,
        }
    return payload


def clinical_image_shuffle_payload(
    records: Sequence[dict[str, Any]],
    image_sources: dict[str, str],
    predictions: Sequence[str],
    references: Sequence[str],
    baseline_metrics: dict[str, float],
    epoch: float,
    step: int,
    seed: int,
    target_format: str = "positive_only",
) -> dict[str, Any]:
    metrics = clinical_classification_metrics(
        predictions, references, target_format
    )
    common = sorted(set(metrics) & set(baseline_metrics))
    validation = clinical_validation_payload(
        records,
        predictions,
        references,
        epoch,
        step,
        metrics,
        target_format,
    )
    samples = validation["samples"]
    for record, sample in zip(records, samples):
        sample["image_source_id"] = image_sources[str(record["id"])]
        sample["image_paths"] = list(record["images"])
    payload = {
        "schema_version": 1,
        "split": "val",
        "task": "clinical_classification_image_shuffle",
        "epoch": float(epoch),
        "step": int(step),
        "seed": int(seed),
        "n_examples": len(samples),
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "metric_deltas": {
            key: float(metrics[key] - baseline_metrics[key]) for key in common
        },
        "image_sources": image_sources,
        "samples": samples,
    }
    if "format_compliance" in validation:
        payload["format_compliance"] = validation["format_compliance"]
    return payload


class ClinicalEvalCallback(TrainerCallback):
    def __init__(
        self,
        processor: Any,
        records: list[dict[str, Any]],
        cfg: Any,
        results: Path,
        dashboard: Any,
        writer: Any,
        mlflow: Any | None,
    ):
        self.processor = processor
        self.records = records
        self.cfg = cfg
        self.results = results
        self.dashboard = dashboard
        self.writer = writer
        self.mlflow = mlflow
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.best_state: dict[str, Any] | None = None
        self.image_shuffle: dict[str, Any] | None = None

    def _write_history(self) -> None:
        if not self.history:
            return
        path = self.results / "clinical_val_history.csv"
        keys: list[str] = []
        for row in self.history:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def _save_best(self, model: Any, payload: dict[str, Any]) -> None:
        from peft import get_peft_model_state_dict

        self.best_state = {
            key: value.detach().cpu().clone()
            for key, value in get_peft_model_state_dict(model).items()
        }
        destination = self.results / "clinical_adapter"
        model.save_pretrained(destination)
        write_json_atomic(destination / "best_info.json", self.best)
        write_json_atomic(
            self.results / "clinical_val_predictions_best.json", payload
        )

    def _persist(self) -> None:
        self._write_history()
        self.writer.set_in(
            "stages",
            clinical={
                "status": "running",
                "records": len(self.records),
                "validation": self.history,
                "best": self.best,
                "image_shuffle": self.image_shuffle,
            },
        )
        self.writer.flush()

    def on_epoch_end(self, args, state, control, **kwargs):
        from .evaluation import generate_predictions

        model = kwargs["model"]
        epoch = float(state.epoch or 0.0)
        step = int(state.global_step)
        started = time.time()
        self.dashboard.status = f"clinical validation epoca {epoch:.2f}"
        predictions, references = generate_predictions(
            model,
            self.processor,
            self.records,
            self.cfg.gen_batch_size,
            self.cfg.clinical_max_new_tokens,
            self.cfg.repetition_penalty,
            progress=lambda done, total: self.dashboard.log_progress(
                "CLINICAL validation", done, total, started
            ),
        )
        metrics = clinical_classification_metrics(
            predictions, references, self.cfg.clinical_target_format
        )
        row = {
            "epoch": epoch,
            "step": step,
            "eval_seconds": round(time.time() - started, 1),
            **metrics,
        }
        self.history.append(row)
        payload = clinical_validation_payload(
            self.records,
            predictions,
            references,
            epoch,
            step,
            metrics,
            self.cfg.clinical_target_format,
        )
        folder = self.results / "clinical_val_predictions"
        write_json_atomic(folder / f"step{step:06d}.json", payload)
        if self.mlflow is not None:
            for key, value in clinical_mlflow_metrics(metrics).items():
                self.mlflow.log_metric(f"clinical.val.{key}", value, step=step)
        if self.best is None or metrics["f1_macro"] > self.best["value"]:
            self.best = {
                "metric": "f1_macro",
                "value": metrics["f1_macro"],
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
                "adapter": str(self.results / "clinical_adapter"),
            }
            self._save_best(model, payload)
        self._persist()
        self.dashboard.phase = None
        self.dashboard.status = (
            f"clinical validation epoca {epoch:.2f}: "
            f"macro-F1={metrics['f1_macro']:.4f}"
        )
        self.dashboard.render()
        return control

    def restore_best(self, model: Any) -> None:
        if self.best_state is None:
            raise RuntimeError("La validation clinica non ha prodotto un checkpoint")
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, self.best_state)

    def evaluate_image_shuffle(self, model: Any) -> dict[str, Any]:
        from .evaluation import generate_predictions

        if self.best is None:
            raise RuntimeError("Image shuffle requires a validated clinical adapter")
        shuffled, sources = shuffle_clinical_images(self.records, self.cfg.seed)
        started = time.time()
        self.dashboard.status = "clinical validation con immagini scambiate"
        predictions, references = generate_predictions(
            model,
            self.processor,
            shuffled,
            self.cfg.gen_batch_size,
            self.cfg.clinical_max_new_tokens,
            self.cfg.repetition_penalty,
            progress=lambda done, total: self.dashboard.log_progress(
                "CLINICAL image shuffle", done, total, started
            ),
        )
        self.image_shuffle = clinical_image_shuffle_payload(
            shuffled,
            sources,
            predictions,
            references,
            self.best["metrics"],
            self.best["epoch"],
            self.best["step"],
            self.cfg.seed,
            self.cfg.clinical_target_format,
        )
        self.image_shuffle["eval_seconds"] = round(time.time() - started, 1)
        write_json_atomic(
            self.results / "clinical_image_shuffle.json", self.image_shuffle
        )
        if self.mlflow is not None:
            for key, value in clinical_mlflow_metrics(
                self.image_shuffle["metrics"]
            ).items():
                self.mlflow.log_metric(
                    f"clinical.shuffle.{key}", value, step=self.best["step"]
                )
        self._persist()
        return self.image_shuffle

    def summary(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "best": self.best,
            "image_shuffle": self.image_shuffle,
        }
