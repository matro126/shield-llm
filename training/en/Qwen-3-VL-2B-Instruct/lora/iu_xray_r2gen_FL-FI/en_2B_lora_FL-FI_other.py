import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.training import main

OVERRIDES: dict = {
    "experiment": "en_2B_lora_FL-FI_other",
    "dataset_root": "dataset/iu-xray/en/other/iu_xray_en_FL-FI",
    "results_dir": "training/en/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_FL-FI/results_other",
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 1e-05,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "eval_metrics": ["bleu", "rougeL", "bertscore", "chexbert"],
    "test_metrics": ["bleu", "rougeL", "bertscore", "chexbert"],
    "monitor_metric": "findings.chexbert_f1_micro_top5",
    "save_every_eval": True,
    "early_stopping_patience": 99,
    "early_stopping_min_delta": 0.001,
    "max_epochs": 15,
}

if __name__ == "__main__":
    raise SystemExit(main(__file__, OVERRIDES))
