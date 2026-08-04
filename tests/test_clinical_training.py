from __future__ import annotations

import runpy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shield.training.config import Config, Identity, build_config, validate
from shield.training.callbacks import LossLogger
from shield.training.runner import RUN_ARTIFACTS, prepare_training_records
from training.evaluate_test import load_saved_run
from shield.training.clinical import (
    build_clinical_records,
    build_stage_two_records,
    validate_clinical_source,
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


def test_prepare_training_records_leaves_standard_training_untouched():
    reports = stage_records()
    clinical, stage_two, stats = prepare_training_records(reports, config())
    assert clinical == []
    assert stage_two is reports
    assert stats["clinical"] is None


def test_clinical_training_outputs_are_run_artifacts():
    assert "clinical_training.json" in RUN_ARTIFACTS
    assert "clinical_adapter" in RUN_ARTIFACTS


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
