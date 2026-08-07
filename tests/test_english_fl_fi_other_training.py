from __future__ import annotations

from pathlib import Path

from shield.training.config import Identity, build_config, read_overrides


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = (
    ROOT
    / "training"
    / "en"
    / "Qwen-3-VL-2B-Instruct"
    / "lora"
    / "iu_xray_r2gen_FL-FI"
)
COMMON_OVERRIDES = {
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 1e-05,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "eval_metrics": ["bleu", "rougeL", "bertscore", "chexbert"],
    "test_metrics": ["bleu", "rougeL", "bertscore", "chexbert"],
    "monitor_metric": "findings.chexbert_f1_micro",
    "save_every_eval": True,
    "early_stopping_patience": 99,
    "early_stopping_min_delta": 0.001,
    "max_epochs": 15,
    "tune_mm_llm": True,
    "tune_mm_vision": True,
    "tune_mm_mlp": True,
    "vision_lr": 1e-05,
    "merger_lr": 1e-05,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "optim": "adamw_torch",
    "max_seq_length": 4096,
    "min_pixels": 12544,
    "max_pixels": 451584,
}


def configuration(path: Path):
    assert path.is_file(), path
    return build_config(Identity.from_path(path), ROOT, read_overrides(path))


def test_other_and_no_other_only_change_experiment_paths() -> None:
    variants = {
        "other": {
            "experiment": "en_2B_lora_FL-FI_other",
            "dataset_root": "dataset/iu-xray/en/other/iu_xray_en_FL-FI",
            "results_dir": (
                "training/en/Qwen-3-VL-2B-Instruct/lora/"
                "iu_xray_r2gen_FL-FI/results_other"
            ),
        },
        "no_other": {
            "experiment": "en_2B_lora_FL-FI_no_other",
            "dataset_root": "dataset/iu-xray/en/no_other/iu_xray_en_FL-FI",
            "results_dir": (
                "training/en/Qwen-3-VL-2B-Instruct/lora/"
                "iu_xray_r2gen_FL-FI/results_no_other"
            ),
        },
    }

    for variant, changes in variants.items():
        path = DIRECTORY / f"en_2B_lora_FL-FI_{variant}.py"
        overrides = read_overrides(path)
        assert overrides == {**changes, **COMMON_OVERRIDES}
        config = configuration(path)
        assert config.training_strategy == "standard"
        assert config.seed == 42
