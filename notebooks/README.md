# Notebook esperimenti

Esegui i notebook in questo ordine per confrontare in modo pulito baseline, QLoRA e LoRA sullo stesso test set.

## 1. Baseline zero-shot

```text
00_qwen3vl_2b_zero_shot_baseline.ipynb
```

Config:

```text
configs/experiments/qwen3vl_2b_zero_shot_baseline.toml
```

Cosa fa:

- carica Qwen3-VL 2B;
- non esegue fine-tuning;
- genera referti sul test set;
- salva metriche in `outputs/evaluation/qwen3vl_2b_zero_shot_baseline/`;
- opzionalmente logga la run in MLflow.

## 2. QLoRA con early stopping

```text
10_qwen3vl_2b_qlora_early_stopping.ipynb
```

Config:

```text
configs/experiments/qwen3vl_2b_qlora_early_stopping.toml
```

Cosa fa:

- carica Qwen3-VL 2B in 4-bit;
- applica QLoRA al decoder linguistico;
- usa early stopping su `eval_loss`;
- salva checkpoint in `outputs/checkpoints/qwen3vl_2b_qlora_early_stopping/`;
- valuta sul test set e logga in MLflow.

## 3. LoRA con early stopping

```text
11_qwen3vl_2b_lora_early_stopping.ipynb
```

Config:

```text
configs/experiments/qwen3vl_2b_lora_early_stopping.toml
```

Cosa fa:

- carica Qwen3-VL 2B in bfloat16 senza quantizzazione 4-bit;
- applica LoRA al decoder linguistico;
- usa early stopping su `eval_loss`;
- salva checkpoint in `outputs/checkpoints/qwen3vl_2b_lora_early_stopping/`;
- valuta sul test set e logga in MLflow.

## Prima Di Eseguire

Sul server:

```bash
uv sync
uv run dvc status
uv run dvc repro
```

Avvia MLflow in una sessione separata:

```bash
tmux new -s mlflow
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts
```

## 4. Valutazione migliori checkpoint

```text
20_evaluate_best_checkpoints.ipynb
```

Cosa fa:

- legge le config `*early_stopping.toml`;
- trova automaticamente il miglior checkpoint da `trainer_state.json`;
- se esiste `checkpoints_dir/final`, valuta quello, perche' i notebook di training salvano li' il best model dopo `load_best_model_at_end=True`;
- calcola ROUGE e BERTScore sul test set;
- salva metriche, qualitative examples, predizioni complete e riepilogo aggregato.

Output:

```text
outputs/evaluation/best_checkpoints/
├── summary.csv
├── summary.json
├── qwen3vl-2b-qlora-early-stopping/
│   ├── metadata.json
│   ├── metrics.json
│   ├── qualitative.json
│   └── predictions.json
└── qwen3vl-2b-lora-early-stopping/
    ├── metadata.json
    ├── metrics.json
    ├── qualitative.json
    └── predictions.json
```

Comando equivalente:

```bash
uv run python scripts/evaluation/evaluate_best_checkpoints.py \
  --all-finetuning-configs \
  --output-dir outputs/evaluation/best_checkpoints
```
