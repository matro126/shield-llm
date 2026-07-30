#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.training import main

OVERRIDES: dict = {
    'eval_metrics': ['bleu', 'rougeL', 'bertscore', 'chexbert'],
    'test_metrics': ['bleu', 'rougeL', 'bertscore', 'chexbert'],
    'monitor_metric': 'findings.chexbert_f1_micro_top5',
    'save_every_eval': True,
    'early_stopping_patience': 99,
    'early_stopping_min_delta': 0.001,
    'max_epochs': 15,
}

if __name__ == "__main__":
    raise SystemExit(main(__file__, OVERRIDES))
