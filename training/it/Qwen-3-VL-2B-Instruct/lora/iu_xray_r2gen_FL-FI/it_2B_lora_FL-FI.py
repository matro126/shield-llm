#!/usr/bin/env python3
"""Fine-tuning lora di Qwen/Qwen3-VL-2B-Instruct su dataset/iu-xray/ita/iu_xray_it_FL-FI.

    lingua   it          (dalla cartella training/it/)
    modello  Qwen-3-VL-2B-Instruct
    modalita lora          (base in bf16)
    dataset  iu_xray_r2gen_FL-FI   (views=frontal_lateral, target=findings_impression)

GENERATO da training/generate.py — non modificare il corpo a mano: le modifiche
verrebbero perse alla prossima rigenerazione. Per cambiare un iperparametro:

  * di QUESTO esperimento  → il blocco OVERRIDES qui sotto (preservato);
  * di TUTTI               → training/defaults.toml.

Eseguibile da solo:

    python training/it/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_FL-FI/it_2B_lora_FL-FI.py

Risultati (loss di training, loss e metriche di validation, best adapter) in
training/it/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_FL-FI/results/. La valutazione sul test set e' un processo a parte:

    python training/evaluate_test.py --experiment it_2B_lora_FL-FI
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.training import main

# ── OVERRIDES ───────────────────────────────────────────────────────────────
# Iperparametri specifici di questo esperimento. Vuoto = usa defaults.toml.
# Esempio:  OVERRIDES = {"learning_rate": 2e-5, "per_device_train_batch_size": 4}
OVERRIDES: dict = {
    'eval_metrics': ['bleu', 'rougeL', 'chexbert'],
    'monitor_metric': 'chexbert_f1_macro_top5',
    'chexbert_translate': True,
    'chexbert_translator': 'models/others/opus-mt-it-en',
    'save_every_eval': True,
    'early_stopping_patience': 99,
    'early_stopping_min_delta': 0.001,
    'max_epochs': 15,
}
# ── fine OVERRIDES ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    raise SystemExit(main(__file__, OVERRIDES))
