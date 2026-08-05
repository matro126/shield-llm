import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

import shield.training.runner as training_runner
from shield.training.clinical import build_clinical_records
from shield.training.clinical_evaluation import (
    clinical_classification_metrics,
    dense_clinical_format,
    parse_clinical_labels,
)


def record(uid, labels):
    return {
        "id": uid,
        "messages": [
            {"role": "system", "content": "Report system"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{uid}/frontal.png"},
                    {"type": "image", "image": f"{uid}/lateral.png"},
                    {"type": "text", "text": "Describe the radiograph"},
                ],
            },
            {"role": "assistant", "content": f"Findings:\n{uid}"},
        ],
        "images": [f"{uid}/frontal.png", f"{uid}/lateral.png"],
        "factors": {
            "diagnostic_category": labels,
            "projection": "frontal+lateral",
            "views": ["frontal", "lateral"],
            "task_type": "report_generation",
        },
    }


def source_records():
    return [
        record("normal", ["No Finding"]),
        record("pathological", ["Cardiomegaly", "Lung Opacity"]),
        record("other", ["Other"]),
        record("unlabeled", ["Unlabeled"]),
    ]


def dense_fallback_text(other="Absent", unlabeled="Absent"):
    return "\n".join(
        [
            "Clinical findings:",
            "Pneumothorax: Absent",
            "Pleural Effusion: Absent",
            "Edema: Absent",
            "Consolidation: Absent",
            "Pneumonia: Absent",
            "Atelectasis: Absent",
            "Lung Lesion: Absent",
            "Lung Opacity: Absent",
            "Cardiomegaly: Absent",
            "Enlarged Cardiomediastinum: Absent",
            "Fracture: Absent",
            "Pleural Other: Absent",
            "Support Devices: Absent",
            f"Other: {other}",
            f"Unlabeled: {unlabeled}",
        ]
    )


def test_prepare_clinical_validation_records_preserves_dense_target_format():
    prepare = getattr(
        training_runner, "prepare_clinical_validation_records", None
    )
    assert prepare is not None
    record = {
        "id": "normal",
        "messages": [
            {"role": "system", "content": "Report system"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "normal/frontal.png"},
                    {"type": "image", "image": "normal/lateral.png"},
                    {"type": "text", "text": "Describe the radiograph"},
                ],
            },
            {"role": "assistant", "content": "Findings:\nNormal"},
        ],
        "images": ["normal/frontal.png", "normal/lateral.png"],
        "factors": {
            "diagnostic_category": ["No Finding"],
            "projection": "frontal+lateral",
            "views": ["frontal", "lateral"],
            "task_type": "report_generation",
        },
    }
    cfg = SimpleNamespace(
        views="frontal_lateral",
        clinical_target_format="dense_binary",
        clinical_include_fallback=False,
    )
    clinical, stats = prepare([record], cfg)
    assert stats["target_format"] == "dense_binary"
    assert "Pneumothorax: Absent" in clinical[0]["messages"][-1]["content"]
    assert "Support Devices: Absent" in clinical[0]["messages"][-1]["content"]
    assert "No Finding:" not in clinical[0]["messages"][-1]["content"]


def test_build_clinical_records_can_keep_every_source_record_once():
    clinical, stats = build_clinical_records(
        source_records(),
        expected_images=2,
        target_format="positive_only",
        include_fallback_records=True,
        include_fallback_labels=True,
    )
    assert [row["id"] for row in clinical] == [
        "normal",
        "pathological",
        "other",
        "unlabeled",
    ]
    assert clinical[2]["messages"][-1]["content"] == "Clinical findings:\nOther"
    assert clinical[3]["messages"][-1]["content"] == "Clinical findings:\nUnlabeled"
    assert stats["clinical_records"] == 4
    assert stats["excluded_other"] == 0
    assert stats["excluded_unlabeled"] == 0


def test_prepare_clinical_validation_records_filters_fallback_records_only():
    cfg = SimpleNamespace(
        views="frontal_lateral",
        clinical_target_format="dense_binary",
        clinical_include_fallback=True,
    )
    clinical, stats = training_runner.prepare_clinical_validation_records(
        source_records(), cfg
    )
    assert [row["id"] for row in clinical] == ["normal", "pathological"]
    assert stats["excluded_other"] == 1
    assert stats["excluded_unlabeled"] == 1
    lines = clinical[0]["messages"][-1]["content"].splitlines()
    assert "Other: Absent" in lines
    assert "Unlabeled: Absent" in lines


def test_dense_parser_keeps_other_distinct_from_no_finding():
    text = dense_fallback_text(other="Present")
    assert parse_clinical_labels(text, "dense_binary", True) == ["Other"]
    assert dense_clinical_format(text, True) == {
        "complete": True,
        "recognized_count": 15,
        "missing_labels": [],
        "duplicate_labels": [],
        "invalid_labels": [],
    }


def test_dense_parser_derives_no_finding_only_when_all_labels_are_absent():
    assert parse_clinical_labels(
        dense_fallback_text(), "dense_binary", True
    ) == ["No Finding"]


def test_clinical_metrics_penalize_fallback_prediction_on_healthy_reference():
    metrics = clinical_classification_metrics(
        [dense_fallback_text(other="Present")],
        [dense_fallback_text()],
        "dense_binary",
        True,
    )
    assert metrics["accuracy_exact"] == 0.0
    assert metrics["cls_Other_support"] == 0.0
    assert metrics["fallback_prediction_rate"] == 1.0


@pytest.mark.parametrize(
    ("suffix", "target_format", "max_tokens"),
    [
        ("positive", "positive_only", 64),
        ("dense", "dense_binary", 192),
    ],
)
def test_full_clinical_entrypoints_use_all_records_without_sampling(
    suffix, target_format, max_tokens
):
    root = Path(__file__).resolve().parents[1]
    script = root / (
        "training/en/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_FL-F/"
        f"en_2B_lora_FL-F_clinical_full_{suffix}_b16.py"
    )
    assert script.is_file()
    overrides = runpy.run_path(
        str(script), run_name=f"full_clinical_{suffix}"
    )["OVERRIDES"]
    assert overrides["training_strategy"] == "clinical"
    assert overrides["training_phase"] == "clinical_only"
    assert overrides["clinical_pretrain_epochs"] == 3
    assert overrides["clinical_balance"] is False
    assert overrides["clinical_include_fallback"] is True
    assert overrides["clinical_target_format"] == target_format
    assert overrides["clinical_max_new_tokens"] == max_tokens
    assert overrides["clinical_image_shuffle_eval"] is True
    assert overrides["per_device_train_batch_size"] == 8
    assert overrides["gradient_accumulation_steps"] == 2
    assert overrides["seed"] == 42
