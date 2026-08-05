import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.training import main

OVERRIDES: dict = {
    "experiment": "en_2B_lora_FL-F_clinical_probe_dense_b16",
    "results_dir": "training/en/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_FL-F/results_clinical_probe_dense_b16",
    "training_strategy": "clinical",
    "training_phase": "clinical_only",
    "clinical_pretrain_epochs": 3,
    "clinical_rehearsal_ratio": 0.1,
    "clinical_balance": True,
    "clinical_sampling_strategy": "weighted",
    "clinical_target_format": "dense_binary",
    "clinical_healthy_ratio": 0.3,
    "clinical_image_shuffle_eval": True,
    "clinical_max_new_tokens": 192,
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
    "monitor_metric": "findings.chexbert_f1_macro",
    "tune_mm_llm": True,
    "tune_mm_vision": True,
    "tune_mm_mlp": True,
    "vision_lr": 1e-05,
    "merger_lr": 1e-05,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "optim": "adamw_torch",
    "max_seq_length": 4096,
    "min_pixels": 12544,
    "max_pixels": 451584,
    "max_new_tokens": 192,
    "seed": 42,
}

if __name__ == "__main__":
    raise SystemExit(main(__file__, OVERRIDES))
