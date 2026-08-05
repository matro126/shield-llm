from __future__ import annotations

import runpy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shield.training.config import Config, Identity, build_config, validate
from shield.training.callbacks import LossLogger
from shield.training.runner import (
    RUN_ARTIFACTS,
    prepare_training_records,
    validate_resume_adapter_path,
)
from training.evaluate_test import load_saved_run
from shield.training.clinical import (
    build_balanced_clinical_records,
    build_clinical_records,
    build_stage_two_records,
    shuffle_clinical_images,
    validate_clinical_source,
)
from shield.training.clinical_evaluation import (
    clinical_image_shuffle_payload,
    clinical_mlflow_metrics,
    clinical_validation_payload,
    clinical_classification_metrics,
    parse_clinical_labels,
)
from shield.training.model import (
    adapter_specific_missing_keys,
    load_training_adapter,
    validate_training_adapter_config,
)


def record(uid: str, labels: list[str]) -> dict:
    images = [f"{uid}/frontal.png", f"{uid}/lateral.png"]
    return {
        "id": uid,
        "messages": [
            {"role": "system", "content": "Report system"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": "Describe the radiograph"},
                ],
            },
            {"role": "assistant", "content": f"Findings:\nReport for {uid}"},
        ],
        "images": images,
        "factors": {
            "diagnostic_category": labels,
            "projection": "frontal+lateral",
            "views": ["frontal", "lateral"],
            "task_type": "report_generation",
        },
        "mesh_raw": labels,
        "provenance": {"source_lang": "en", "target_lang": "en"},
    }


def source_records() -> list[dict]:
    return [
        record("normal", ["No Finding"]),
        record("pathological", ["Cardiomegaly", "Lung Opacity"]),
        record("other", ["Other"]),
        record("unlabeled", ["Unlabeled"]),
    ]


def test_parse_clinical_labels_normalizes_only_allowed_labels():
    text = (
        "Clinical findings:\n"
        "1. pleural effusion.\n"
        "- Pneumothorax, No Finding\n"
        "Unknown finding"
    )
    assert parse_clinical_labels(text) == [
        "Pneumothorax",
        "Pleural Effusion",
        "No Finding",
    ]


def test_clinical_classification_metrics_are_standard_multilabel_scores():
    references = [
        "Clinical findings:\nNo Finding",
        "Clinical findings:\nPneumothorax\nPleural Effusion",
        "Clinical findings:\nCardiomegaly",
    ]
    predictions = [
        "Clinical findings:\nNo Finding",
        "Clinical findings:\nPneumothorax",
        "Clinical findings:\nPleural Effusion",
    ]
    metrics = clinical_classification_metrics(predictions, references)
    assert metrics["accuracy_exact"] == pytest.approx(1 / 3)
    assert metrics["precision_micro"] == pytest.approx(2 / 3)
    assert metrics["recall_micro"] == pytest.approx(1 / 2)
    assert metrics["f1_micro"] == pytest.approx(4 / 7)
    assert metrics["f1_macro"] == pytest.approx(1 / 7)
    assert metrics["cls_Pneumothorax_f1"] == pytest.approx(1.0)
    assert metrics["cls_Pleural_Effusion_f1"] == pytest.approx(0.0)
    assert metrics["cls_Cardiomegaly_support"] == pytest.approx(1.0)


def test_clinical_classification_metrics_reject_misaligned_inputs():
    with pytest.raises(ValueError, match="stessa lunghezza"):
        clinical_classification_metrics(["No Finding"], [])


def test_clinical_validation_payload_keeps_raw_and_parsed_outputs():
    rows = [record("normal", ["No Finding"])]
    references = ["Clinical findings:\nNo Finding"]
    predictions = ["Clinical findings:\nPneumothorax"]
    payload = clinical_validation_payload(
        rows,
        predictions,
        references,
        epoch=2.0,
        step=14,
        metrics={"f1_macro": 0.25},
    )
    assert payload["epoch"] == 2.0
    assert payload["step"] == 14
    assert payload["n_examples"] == 1
    assert payload["metrics"] == {"f1_macro": 0.25}
    assert payload["samples"] == [
        {
            "id": "normal",
            "labels": ["No Finding"],
            "predicted_labels": ["Pneumothorax"],
            "reference": references[0],
            "prediction": predictions[0],
        }
    ]


def test_shuffle_clinical_images_is_deterministic_and_moves_complete_studies():
    rows = [record(f"case-{index}", ["No Finding"]) for index in range(4)]
    shuffled, sources = shuffle_clinical_images(rows, seed=42)
    repeated, repeated_sources = shuffle_clinical_images(rows, seed=42)
    assert sources == repeated_sources
    assert [row["images"] for row in shuffled] == [row["images"] for row in repeated]
    assert [row["id"] for row in shuffled] == [row["id"] for row in rows]
    assert all(sources[row["id"]] != row["id"] for row in rows)
    for original, changed in zip(rows, shuffled):
        source = next(row for row in rows if row["id"] == sources[original["id"]])
        assert changed["images"] == source["images"]
        assert changed["messages"][0] == original["messages"][0]
        assert changed["messages"][-1] == original["messages"][-1]
        assert changed["messages"][1]["content"][-1] == original["messages"][1]["content"][-1]
        assert [item["image"] for item in changed["messages"][1]["content"][:-1]] == source["images"]
        assert changed["factors"] == original["factors"]


def test_clinical_image_shuffle_payload_keeps_metrics_deltas_and_sources():
    rows = [record("case-a", ["No Finding"]), record("case-b", ["Pneumothorax"])]
    shuffled, sources = shuffle_clinical_images(rows, seed=42)
    references = [
        "Clinical findings:\nNo Finding",
        "Clinical findings:\nPneumothorax",
    ]
    predictions = [
        "Clinical findings:\nNo Finding",
        "Clinical findings:\nNo Finding",
    ]
    payload = clinical_image_shuffle_payload(
        shuffled,
        sources,
        predictions,
        references,
        baseline_metrics={"f1_macro": 0.5, "f1_micro": 0.75},
        epoch=2.0,
        step=20,
        seed=42,
    )
    assert payload["task"] == "clinical_classification_image_shuffle"
    assert payload["seed"] == 42
    assert payload["baseline_metrics"] == {"f1_macro": 0.5, "f1_micro": 0.75}
    assert payload["metric_deltas"]["f1_macro"] == pytest.approx(
        payload["metrics"]["f1_macro"] - 0.5
    )
    assert payload["metric_deltas"]["f1_micro"] == pytest.approx(
        payload["metrics"]["f1_micro"] - 0.75
    )
    assert payload["samples"][0]["image_source_id"] == sources["case-a"]
    assert payload["samples"][0]["image_paths"] == shuffled[0]["images"]


def stage_records() -> list[dict]:
    rows = [record(f"healthy-{i}", ["No Finding"]) for i in range(12)]
    rows.extend(record(f"other-{i}", ["Other"]) for i in range(8))
    rows.extend(record(f"common-{i}", ["Lung Opacity"]) for i in range(8))
    rows.extend(record(f"rare-{i}", ["Pneumothorax"]) for i in range(2))
    return rows


def test_build_clinical_records_keeps_only_supervised_categories():
    original = source_records()
    clinical, stats = build_clinical_records(original, expected_images=2)
    assert [row["id"] for row in clinical] == ["normal", "pathological"]
    assert clinical[0]["messages"][-1]["content"] == "Clinical findings:\nNo Finding"
    assert clinical[1]["messages"][-1]["content"] == (
        "Clinical findings:\nLung Opacity\nCardiomegaly"
    )
    assert clinical[1]["factors"]["task_type"] == "clinical_classification"
    assert clinical[1]["images"] == original[1]["images"]
    assert clinical[1]["messages"][1]["content"][:2] == original[1]["messages"][1][
        "content"
    ][:2]
    assert clinical[1]["messages"][1]["content"][-1]["text"].startswith(
        "Classify the chest X-ray"
    )
    assert "Allowed labels:" in clinical[1]["messages"][1]["content"][-1]["text"]
    assert "Pneumothorax" in clinical[1]["messages"][1]["content"][-1]["text"]
    assert original[1]["messages"][-1]["content"] == "Findings:\nReport for pathological"
    assert stats["excluded_other"] == 1
    assert stats["excluded_unlabeled"] == 1


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([record("bad-images", ["No Finding"]) | {"images": ["one.png"]}], "images"),
        ([record("bad-label", ["Unknown"])], "Unknown"),
        ([record("mixed", ["No Finding", "Cardiomegaly"])], "No Finding"),
    ],
)
def test_validate_clinical_source_rejects_invalid_records(rows, message):
    with pytest.raises(ValueError, match=message):
        validate_clinical_source(rows, expected_images=2)


def test_clinical_stage_two_is_balanced_and_has_rehearsal():
    reports = stage_records()
    clinical, _ = build_clinical_records(reports, expected_images=2)
    cfg = SimpleNamespace(
        seed=42,
        training_strategy="clinical",
        clinical_rehearsal_ratio=0.1,
        healthy_ratio=0.3,
        pathological_ratio=0.6,
        other_ratio=0.1,
        rare_weight_cap=4.0,
    )
    selected, stats = build_stage_two_records(reports, clinical, cfg)
    repeated, repeated_stats = build_stage_two_records(reports, clinical, cfg)
    assert [row["id"] for row in selected] == [row["id"] for row in repeated]
    assert stats == repeated_stats
    assert len(selected) == len(reports)
    assert stats["task_counts"]["clinical_classification"] == 3
    assert stats["task_counts"]["report_generation"] == 27
    assert stats["report_strata_counts"] == {
        "healthy": 8,
        "pathological": 16,
        "other": 3,
    }
    assert stats["pathology_weights"]["max"] <= 4.0
    assert stats["pathology_weights"]["Pneumothorax"] > stats[
        "pathology_weights"
    ]["Lung Opacity"]


def test_balanced_clinical_records_are_deterministic_and_reduce_no_finding():
    reports = stage_records()
    clinical, _ = build_clinical_records(reports, expected_images=2)
    cfg = SimpleNamespace(
        seed=42,
        clinical_healthy_ratio=0.3,
        rare_weight_cap=4.0,
        clinical_sampling_strategy="weighted",
    )
    selected, stats = build_balanced_clinical_records(clinical, cfg)
    repeated, repeated_stats = build_balanced_clinical_records(clinical, cfg)
    assert [row["id"] for row in selected] == [row["id"] for row in repeated]
    assert stats == repeated_stats
    assert len(selected) == len(clinical)
    assert stats["sampled_strata_counts"] == {
        "healthy": 7,
        "pathological": 15,
    }
    assert stats["source_strata_counts"] == {
        "healthy": 12,
        "pathological": 10,
    }
    assert stats["pathology_weights"]["Pneumothorax"] > stats[
        "pathology_weights"
    ]["Lung Opacity"]
    assert stats["sampled_label_frequencies"]["No Finding"] == 7


def test_label_quota_clinical_records_have_uniform_target_draws():
    reports = stage_records()
    clinical, _ = build_clinical_records(reports, expected_images=2)
    cfg = SimpleNamespace(
        seed=42,
        clinical_healthy_ratio=0.3,
        rare_weight_cap=4.0,
        clinical_sampling_strategy="label_quota",
    )
    selected, stats = build_balanced_clinical_records(clinical, cfg)
    repeated, repeated_stats = build_balanced_clinical_records(clinical, cfg)
    assert [row["id"] for row in selected] == [row["id"] for row in repeated]
    assert stats == repeated_stats
    assert stats["clinical_sampling_strategy"] == "label_quota"
    assert stats["pathology_weights"] is None
    draws = stats["pathology_target_draws"]
    assert set(draws) == {"Lung Opacity", "Pneumothorax"}
    assert max(draws.values()) - min(draws.values()) <= 1
    assert sum(draws.values()) == stats["sampled_strata_counts"]["pathological"]


def test_clinical_mlflow_metrics_keep_aggregates_and_class_f1_only():
    metrics = {
        "accuracy_exact": 0.5,
        "precision_macro": 0.4,
        "recall_macro": 0.3,
        "f1_macro": 0.35,
        "precision_micro": 0.6,
        "recall_micro": 0.5,
        "f1_micro": 0.55,
        "cls_Pneumothorax_precision": 0.2,
        "cls_Pneumothorax_recall": 0.1,
        "cls_Pneumothorax_f1": 0.15,
        "cls_Pneumothorax_support": 3.0,
    }
    selected = clinical_mlflow_metrics(metrics)
    assert selected == {
        "accuracy_exact": 0.5,
        "precision_macro": 0.4,
        "recall_macro": 0.3,
        "f1_macro": 0.35,
        "precision_micro": 0.6,
        "recall_micro": 0.5,
        "f1_micro": 0.55,
        "cls_Pneumothorax_f1": 0.15,
    }


def test_balanced_stage_two_contains_only_reports():
    reports = stage_records()
    cfg = SimpleNamespace(
        seed=42,
        training_strategy="balanced",
        clinical_rehearsal_ratio=0.0,
        healthy_ratio=0.3,
        pathological_ratio=0.6,
        other_ratio=0.1,
        rare_weight_cap=4.0,
    )
    selected, stats = build_stage_two_records(reports, [], cfg)
    assert len(selected) == len(reports)
    assert stats["task_counts"] == {"report_generation": len(reports)}
    assert all(row["factors"]["task_type"] == "report_generation" for row in selected)


def test_standard_stage_two_preserves_identity_and_order():
    reports = stage_records()
    cfg = SimpleNamespace(seed=42, training_strategy="standard")
    selected, stats = build_stage_two_records(reports, [], cfg)
    assert selected is reports
    assert [row["id"] for row in selected] == [row["id"] for row in reports]
    assert stats["task_counts"] == {"report_generation": len(reports)}


def config(**overrides) -> Config:
    values = {
        "target": "findings",
        "eval_metrics": ("chexbert",),
        "test_metrics": ("chexbert",),
        "monitor_metric": "findings.chexbert_f1_macro",
        "lang": "en",
        "views": "frontal_lateral",
    }
    values.update(overrides)
    return Config(**values)


@pytest.mark.parametrize(
    "cfg",
    [
        config(),
        config(training_strategy="balanced"),
        config(
            training_strategy="clinical",
            clinical_pretrain_epochs=3,
            clinical_rehearsal_ratio=0.1,
        ),
    ],
)
def test_training_strategy_config_accepts_consistent_settings(cfg):
    validate(cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        config(
            training_strategy="clinical",
            training_phase="clinical_only",
            clinical_pretrain_epochs=3,
            clinical_rehearsal_ratio=0.1,
            clinical_balance=True,
        ),
        config(
            training_strategy="clinical",
            training_phase="report_only",
            clinical_pretrain_epochs=0,
            clinical_rehearsal_ratio=0.1,
            clinical_adapter_path="results_probe/clinical_adapter",
        ),
    ],
)
def test_training_phase_config_accepts_split_clinical_runs(cfg):
    validate(cfg)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training_strategy": "unknown"}, "training_strategy"),
        (
            {"training_strategy": "balanced", "clinical_pretrain_epochs": 1},
            "clinical_pretrain_epochs",
        ),
        (
            {"training_strategy": "balanced", "clinical_rehearsal_ratio": 0.1},
            "clinical_rehearsal_ratio",
        ),
        (
            {
                "training_strategy": "clinical",
                "clinical_pretrain_epochs": 0,
                "clinical_rehearsal_ratio": 0.1,
            },
            "clinical_pretrain_epochs",
        ),
        (
            {
                "training_strategy": "clinical",
                "clinical_pretrain_epochs": 3,
                "clinical_rehearsal_ratio": 0.0,
            },
            "clinical_rehearsal_ratio",
        ),
        (
            {
                "training_strategy": "clinical",
                "clinical_pretrain_epochs": 3,
                "clinical_rehearsal_ratio": 1.0,
            },
            "clinical_rehearsal_ratio",
        ),
        ({"healthy_ratio": 0.4}, "sum"),
        ({"healthy_ratio": -0.1, "pathological_ratio": 1.0}, "healthy_ratio"),
        ({"rare_weight_cap": 0.0}, "rare_weight_cap"),
        ({"clinical_sampling_strategy": "unknown"}, "clinical_sampling_strategy"),
        ({"training_phase": "unknown"}, "training_phase"),
        (
            {
                "training_strategy": "clinical",
                "training_phase": "report_only",
                "clinical_pretrain_epochs": 0,
                "clinical_rehearsal_ratio": 0.1,
            },
            "clinical_adapter_path",
        ),
        (
            {
                "training_strategy": "balanced",
                "training_phase": "clinical_only",
            },
            "training_phase",
        ),
        ({"clinical_healthy_ratio": 1.0}, "clinical_healthy_ratio"),
        ({"training_strategy": "balanced", "clinical_balance": True}, "clinical_balance"),
        ({"clinical_image_shuffle_eval": True}, "clinical_image_shuffle_eval"),
        (
            {
                "training_strategy": "clinical",
                "training_phase": "report_only",
                "clinical_pretrain_epochs": 0,
                "clinical_rehearsal_ratio": 0.1,
                "clinical_adapter_path": "adapter",
                "clinical_balance": True,
            },
            "clinical_balance",
        ),
    ],
)
def test_training_strategy_config_rejects_inconsistent_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate(config(**overrides))


class Dashboard:
    def __init__(self):
        self.rows = []

    def start(self):
        return None

    def log_step(self, step, epoch, logs):
        self.rows.append((step, epoch, logs))


class Mlflow:
    def __init__(self):
        self.metrics = []

    def log_metric(self, name, value, step):
        self.metrics.append((name, value, step))


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [("train", "train.loss"), ("clinical.train", "clinical.train.loss")],
)
def test_loss_logger_uses_stage_prefix(prefix, expected):
    dashboard = Dashboard()
    mlflow = Mlflow()
    logger = LossLogger(dashboard, mlflow, prefix=prefix)
    state = SimpleNamespace(global_step=7, epoch=1.5)
    control = SimpleNamespace()
    logger.on_log(None, state, control, logs={"loss": 1.25})
    assert mlflow.metrics == [(expected, 1.25, 7)]
    assert dashboard.rows == [(7, 1.5, {"loss": 1.25})]


def test_prepare_training_records_builds_clinical_and_stage_two_sets():
    reports = stage_records()
    cfg = config(
        training_strategy="clinical",
        clinical_pretrain_epochs=3,
        clinical_rehearsal_ratio=0.1,
    )
    clinical, stage_two, stats = prepare_training_records(reports, cfg)
    assert len(clinical) == 22
    assert len(stage_two) == len(reports)
    assert stats["clinical"]["excluded_other"] == 8
    assert stats["stage_two"]["task_counts"] == {
        "clinical_classification": 3,
        "report_generation": 27,
    }


def test_prepare_training_records_balances_only_clinical_stage():
    reports = stage_records()
    cfg = config(
        training_strategy="clinical",
        clinical_pretrain_epochs=3,
        clinical_rehearsal_ratio=0.1,
        clinical_balance=True,
        clinical_healthy_ratio=0.3,
    )
    clinical, stage_two, stats = prepare_training_records(reports, cfg)
    assert len(clinical) == 22
    assert stats["clinical"]["sampling"]["sampled_strata_counts"] == {
        "healthy": 7,
        "pathological": 15,
    }
    assert stats["stage_two"]["task_counts"] == {
        "clinical_classification": 3,
        "report_generation": 27,
    }
    assert len(stage_two) == len(reports)


def test_prepare_training_records_leaves_standard_training_untouched():
    reports = stage_records()
    clinical, stage_two, stats = prepare_training_records(reports, config())
    assert clinical == []
    assert stage_two is reports
    assert stats["clinical"] is None


def test_clinical_training_outputs_are_run_artifacts():
    assert "clinical_training.json" in RUN_ARTIFACTS
    assert "clinical_adapter" in RUN_ARTIFACTS
    assert "clinical_val_history.csv" in RUN_ARTIFACTS
    assert "clinical_val_predictions_best.json" in RUN_ARTIFACTS
    assert "clinical_image_shuffle.json" in RUN_ARTIFACTS
    assert "clinical_val_predictions" in RUN_ARTIFACTS


@pytest.mark.parametrize(
    ("suffix", "strategy", "pretrain_epochs", "rehearsal", "results_name"),
    [
        ("balanced", "balanced", 0, 0.0, "results_balanced"),
        ("clinical", "clinical", 3, 0.1, "results_clinical"),
    ],
)
def test_fast_track_entrypoint_contract(
    suffix, strategy, pretrain_epochs, rehearsal, results_name
):
    root = Path(__file__).resolve().parents[1]
    script = root / (
        "training/en/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_FL-F/"
        f"en_2B_lora_FL-F_{suffix}.py"
    )
    namespace = runpy.run_path(str(script), run_name="entrypoint_contract")
    cfg = build_config(Identity.from_path(script), root, namespace["OVERRIDES"])
    assert cfg.experiment == f"en_2B_lora_FL-F_{suffix}"
    assert Path(cfg.results_dir).name == results_name
    assert cfg.training_strategy == strategy
    assert cfg.clinical_pretrain_epochs == pretrain_epochs
    assert cfg.clinical_rehearsal_ratio == rehearsal
    assert cfg.monitor_metric == "findings.chexbert_f1_macro"
    assert cfg.max_epochs == 10
    assert cfg.early_stopping_patience == 3
    assert cfg.seed == 42


def test_8b_balanced_and_clinical_entrypoints_are_controlled_comparison():
    root = Path(__file__).resolve().parents[1]
    folder = root / (
        "training/en/Qwen-3-VL-8B-Instruct/lora/iu_xray_r2gen_FL-F"
    )
    configs = {}
    for suffix in ("balanced", "clinical"):
        script = folder / f"en_8B_lora_FL-F_{suffix}.py"
        namespace = runpy.run_path(str(script), run_name=f"entrypoint_{suffix}")
        configs[suffix] = build_config(
            Identity.from_path(script), root, namespace["OVERRIDES"]
        )

    balanced = configs["balanced"]
    clinical = configs["clinical"]
    assert balanced.training_strategy == "balanced"
    assert balanced.clinical_pretrain_epochs == 0
    assert balanced.clinical_rehearsal_ratio == 0.0
    assert clinical.training_strategy == "clinical"
    assert clinical.clinical_pretrain_epochs == 3
    assert clinical.clinical_rehearsal_ratio == 0.1

    shared = (
        "base_model",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "learning_rate",
        "vision_lr",
        "merger_lr",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "monitor_metric",
        "max_new_tokens",
        "seed",
    )
    assert {name: getattr(balanced, name) for name in shared} == {
        name: getattr(clinical, name) for name in shared
    }
    assert balanced.base_model == "Qwen/Qwen3-VL-8B-Instruct"
    assert balanced.lora_r == 32
    assert balanced.lora_alpha == 64
    assert balanced.learning_rate == 1e-5
    assert balanced.per_device_train_batch_size == 8
    assert balanced.gradient_accumulation_steps == 2
    assert balanced.per_device_train_batch_size * balanced.gradient_accumulation_steps == 16
    assert balanced.max_epochs == 10
    assert balanced.early_stopping_patience == 3
    assert balanced.monitor_metric == "findings.chexbert_f1_macro"
    assert balanced.max_new_tokens == 192
    assert balanced.seed == 42


@pytest.mark.parametrize("size", ["2B", "8B"])
def test_clinical_probe_and_resume_entrypoints_use_batch_16(size):
    root = Path(__file__).resolve().parents[1]
    folder = root / (
        f"training/en/Qwen-3-VL-{size}-Instruct/lora/iu_xray_r2gen_FL-F"
    )
    probe_script = folder / f"en_{size}_lora_FL-F_clinical_probe.py"
    resume_script = folder / f"en_{size}_lora_FL-F_clinical_resume.py"
    probe_ns = runpy.run_path(str(probe_script), run_name=f"probe_{size}")
    resume_ns = runpy.run_path(str(resume_script), run_name=f"resume_{size}")
    probe = build_config(Identity.from_path(probe_script), root, probe_ns["OVERRIDES"])
    resume = build_config(
        Identity.from_path(resume_script), root, resume_ns["OVERRIDES"]
    )
    assert probe.training_phase == "clinical_only"
    assert probe.clinical_balance is True
    assert probe.clinical_pretrain_epochs == 3
    assert resume.training_phase == "report_only"
    assert resume.clinical_pretrain_epochs == 0
    assert resume.clinical_adapter_path == f"{probe.results_dir}/clinical_adapter"
    assert probe.per_device_train_batch_size == 8
    assert probe.gradient_accumulation_steps == 2
    assert resume.per_device_train_batch_size == 8
    assert resume.gradient_accumulation_steps == 2
    assert probe.per_device_train_batch_size * probe.gradient_accumulation_steps == 16
    assert resume.per_device_train_batch_size * resume.gradient_accumulation_steps == 16


def test_2b_clinical_ablation_entrypoints_change_only_declared_training_factor():
    root = Path(__file__).resolve().parents[1]
    folder = root / (
        "training/en/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_FL-F"
    )
    names = ("clinical_probe", "clinical_probe_nf15", "clinical_probe_quota", "clinical_probe_vision")
    configs = {}
    for name in names:
        script = folder / f"en_2B_lora_FL-F_{name}.py"
        namespace = runpy.run_path(str(script), run_name=f"entrypoint_{name}")
        configs[name] = build_config(
            Identity.from_path(script), root, namespace["OVERRIDES"]
        )
    baseline = configs["clinical_probe"]
    nf15 = configs["clinical_probe_nf15"]
    quota = configs["clinical_probe_quota"]
    vision = configs["clinical_probe_vision"]
    for cfg in (nf15, quota, vision):
        assert cfg.base_model == "Qwen/Qwen3-VL-2B-Instruct"
        assert cfg.training_phase == "clinical_only"
        assert cfg.clinical_pretrain_epochs == 3
        assert cfg.per_device_train_batch_size == 8
        assert cfg.gradient_accumulation_steps == 2
        assert cfg.clinical_image_shuffle_eval is True
        assert cfg.seed == 42
    assert nf15.clinical_healthy_ratio == 0.15
    assert nf15.clinical_sampling_strategy == baseline.clinical_sampling_strategy
    assert nf15.learning_rate == baseline.learning_rate
    assert nf15.vision_lr == baseline.vision_lr
    assert nf15.merger_lr == baseline.merger_lr
    assert quota.clinical_healthy_ratio == baseline.clinical_healthy_ratio
    assert quota.clinical_sampling_strategy == "label_quota"
    assert quota.learning_rate == baseline.learning_rate
    assert quota.vision_lr == baseline.vision_lr
    assert quota.merger_lr == baseline.merger_lr
    assert vision.clinical_healthy_ratio == baseline.clinical_healthy_ratio
    assert vision.clinical_sampling_strategy == baseline.clinical_sampling_strategy
    assert vision.learning_rate == 2e-6
    assert vision.vision_lr == 1e-5
    assert vision.merger_lr == 1e-5
    assert len({cfg.results_dir for cfg in (nf15, quota, vision)}) == 3


def test_load_training_adapter_rejects_missing_adapter(tmp_path):
    with pytest.raises(FileNotFoundError, match="Adapter clinico assente"):
        load_training_adapter(SimpleNamespace(), tmp_path / "missing")


def test_resume_adapter_must_be_outside_live_results(tmp_path):
    results = tmp_path / "results"
    adapter = results / "clinical_adapter"
    with pytest.raises(ValueError, match="fuori da results_dir"):
        validate_resume_adapter_path(results, adapter)
    external = tmp_path / "probe" / "clinical_adapter"
    assert validate_resume_adapter_path(results, external) == external.resolve()


def test_training_adapter_config_requires_matching_lora_and_base_model():
    current = SimpleNamespace(
        peft_config={
            "default": SimpleNamespace(
                r=32,
                lora_alpha=64,
                target_modules={"q_proj", "v_proj"},
                base_model_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
            )
        },
        active_adapter="default",
    )
    saved = {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj"],
        "base_model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
    }
    validate_training_adapter_config(current, saved)
    with pytest.raises(ValueError, match="r"):
        validate_training_adapter_config(current, saved | {"r": 16})
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        validate_training_adapter_config(
            current,
            saved | {"base_model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct"},
        )


def test_adapter_specific_missing_keys_exclude_base_model_parameters():
    missing = [
        "base_model.model.language_model.layers.0.self_attn.q_proj.weight",
        "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "base_model.model.visual.merger.modules_to_save.default.weight",
    ]
    assert adapter_specific_missing_keys(missing) == missing[1:]


def test_load_saved_run_uses_variant_configuration_and_result_root(tmp_path):
    root = tmp_path
    results = root / (
        "training/en/Qwen-3-VL-2B-Instruct/lora/"
        "iu_xray_r2gen_FL-F/results_clinical"
    )
    results.mkdir(parents=True)
    saved = config(
        experiment="en_2B_lora_FL-F_clinical",
        model_dir="Qwen-3-VL-2B-Instruct",
        mode="lora",
        dataset_code="FL-F",
        dataset_root="dataset/iu-xray/en/iu_xray_en_FL-F",
        base_model="Qwen/Qwen3-VL-2B-Instruct",
        results_dir=str(results.relative_to(root)),
        training_strategy="clinical",
        clinical_pretrain_epochs=3,
        clinical_rehearsal_ratio=0.1,
    )
    (results / "results.json").write_text(
        json.dumps({"config": saved.as_dict()}), encoding="utf-8"
    )
    identity, loaded, resolved = load_saved_run(results, root)
    assert identity.name == "en_2B_lora_FL-F"
    assert loaded.experiment == "en_2B_lora_FL-F_clinical"
    assert loaded.results_dir.endswith("results_clinical")
    assert resolved == results
