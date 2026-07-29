#!/usr/bin/env python3
"""Fine-tuning lora di Qwen/Qwen3-VL-2B-Instruct su dataset/iu-xray/en/iu_xray_en_F-F.

    lingua   en          (dalla cartella training/en/)
    modello  Qwen-3-VL-2B-Instruct
    modalita lora          (base in bf16)
    dataset  iu_xray_r2gen_F-F   (views=frontal, target=findings)

GENERATO da training/generate.py — non modificare il corpo a mano: le modifiche
verrebbero perse alla prossima rigenerazione. Per cambiare un iperparametro:

  * di QUESTO esperimento  → il blocco OVERRIDES qui sotto (preservato);
  * di TUTTI               → training/defaults.toml.

Eseguibile da solo:

    python training/en/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_F-F/en_2B_lora_F-F.py

Risultati (loss di training, loss e metriche di validation, best adapter) in
training/en/Qwen-3-VL-2B-Instruct/lora/iu_xray_r2gen_F-F/results/. La valutazione sul test set e' un processo a parte:

    python training/evaluate_test.py --experiment en_2B_lora_F-F
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
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'qkv', 'proj', 'linear_fc1', 'linear_fc2'],
    'eval_metrics': ['bleu', 'rougeL', 'chexbert'],
    'monitor_metric': 'mesh_any_balanced',
    'save_every_eval': True,
    'early_stopping_patience': 99,
    'early_stopping_min_delta': 0.001,
    'max_epochs': 15,
}
# ── fine OVERRIDES ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    raise SystemExit(main(__file__, OVERRIDES))
