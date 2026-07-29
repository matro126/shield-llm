#!/usr/bin/env python3
"""Baseline ZERO-SHOT di Qwen/Qwen3-VL-2B-Instruct su dataset/iu-xray/ita/iu_xray_it_FL-FI.

Il modello BASE, senza fine-tuning, valutato sullo stesso test set con gli stessi
prompt e le stesse metriche degli esperimenti. E' il riferimento senza cui le
metriche del modello addestrato non sono interpretabili.

Questa baseline e' UNA per (modello, dataset) ed e' condivisa dagli esperimenti:

    it_2B_lora_FL-FI, it_2B_qlora_FL-FI

Gira in bf16, cioe' il modello come pubblicato: cosi' i delta di lora e qlora sono
confrontabili fra loro. Per gli esperimenti qlora il delta include quindi anche
l'effetto della quantizzazione a 4 bit, non solo il fine-tuning.

GENERATO da training/generate.py — non modificare il corpo a mano.

    python training/it/Qwen-3-VL-2B-Instruct/baseline/iu_xray_r2gen_FL-FI/baseline_it_2B_FL-FI.py

Risultati in training/it/Qwen-3-VL-2B-Instruct/baseline/iu_xray_r2gen_FL-FI/results/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.training import main_baseline

# ── OVERRIDES ───────────────────────────────────────────────────────────────
# Iperparametri specifici di questa baseline. Vuoto = usa defaults.toml.
# Esempio:  OVERRIDES = {"baseline_max_samples": 50, "load_in_4bit": True}
OVERRIDES: dict = {}
# ── fine OVERRIDES ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    raise SystemExit(main_baseline(__file__, OVERRIDES))
