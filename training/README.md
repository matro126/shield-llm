# Esperimenti di fine-tuning

Un esperimento = una cartella foglia di questo albero. L'identità si legge **dal
percorso**, non da un file di registro:

```
training/it/Qwen-3-VL-2B-Instruct/
├── baseline/                        ← una per (modello × dataset): 4
│   └── iu_xray_r2gen_FL-FI/baseline_it_2B_FL-FI.py
├── lora/
│   └── iu_xray_r2gen_FL-FI/it_2B_lora_FL-FI.py
└── qlora/
    └── iu_xray_r2gen_FL-FI/it_2B_qlora_FL-FI.py
```

Con 3 modelli e 4 dataset: **24 script di training e 12 di baseline**.

Da lì si ricava tutto: `it` + `FL-FI` → `dataset/iu-xray/ita/iu_xray_it_FL-FI`,
`qlora` → base in 4-bit NF4 (`lora` → bf16), `FL` → frontale+laterale, `FI` →
findings+impression (quindi metriche per sezione con `<SEP>`). Il nome
dell'esperimento è `it_2B_qlora_FL-FI`, ed è quello usato in MLflow, nei
risultati e da `evaluate_test.py`.

Oggi sono **24**: 3 modelli × 2 modalità × 4 dataset. Aggiungere `training/en/…`
ne genera altri 24 senza toccare una riga di codice.

## Comandi

```bash
python training/generate.py --list          # cosa c'è e cosa manca
python training/generate.py                 # genera/rigenera gli script

# un singolo esperimento: il file porta il suo nome
python training/it/Qwen-3-VL-2B-Instruct/qlora/iu_xray_r2gen_FL-FI/it_2B_qlora_FL-FI.py

python training/run_all.py --dry-run                    # cosa farebbe
python training/run_all.py --only "it_2B_*"             # gli 8 del 2B
python training/run_all.py --skip-done                  # riprende dopo un'interruzione

python training/collect_results.py                      # aggrega tutti i risultati
python training/check_mlflow.py                         # verifica il tracking

python training/evaluate_test.py --list                          # cosa è valutabile
python training/evaluate_test.py --experiment it_2B_qlora_FL-FI   # test set
```

> I glob sono `fnmatch`: `*2B*` seleziona **anche** il 32B. Usa `it_2B_*` e
> controlla sempre con `--dry-run` / `--list`.

## Cambiare un iperparametro

Due livelli, di proposito:

| Vuoi cambiarlo… | Dove | Effetto |
|---|---|---|
| in **tutti** gli esperimenti | `training/defaults.toml` | immediato, non serve rigenerare |
| in **uno** (o in un sottoinsieme) | blocco `OVERRIDES` nel suo script | solo quello |

Il blocco `OVERRIDES` si può scrivere a mano o dal generatore:

```bash
python training/generate.py --set per_device_train_batch_size=2 \
                            --set gradient_accumulation_steps=8 --only "it_32B_*"
python training/generate.py --unset per_device_train_batch_size --only "it_32B_*"
```

Precedenza: `defaults.toml` → valori derivati dal percorso → `OVERRIDES`. Una
chiave inesistente viene **rifiutata** (un typo diventerebbe un iperparametro
fantasma silenziosamente ignorato). Rigenerare **preserva** gli `OVERRIDES`
scritti a mano; `--force` li azzera.

I 24 esperimenti nascono dallo stesso template: non c'è modo di sbagliare un
parametro copia-incollando, e `python training/generate.py --check` (exit 1 se
qualcosa è disallineato) lo verifica in CI.

## Cosa viene salvato

**`<esperimento>/results/results.json` è la fonte unica**: un solo JSON con tutto
dentro, pensato per essere riletto e aggregato.

```json
{
  "schema_version": 1,
  "experiment": "it_2B_qlora_FL-FI",
  "status": "early_stopped",
  "identity":    { "lang": "it", "model_short": "2B", "mode": "qlora",
                   "dataset_code": "FL-FI", "target": "findings_impression" },
  "dataset":     { "root": "...", "version": "...", "n_train": 2056, "n_val": 294 },
  "config":      { "learning_rate": 1e-05, "...": "iperparametri effettivi" },
  "environment": { "gpus": ["A100 (80.0 GB)"], "python": "3.12.9", "torch": "2.10.0" },
  "provenance":  { "git": {...}, "dvc": {...}, "mlflow": { "run_id": "..." } },
  "timing":      { "started_at": "...", "wall_clock_s": 3612.0 },
  "curves": {
    "train":      [ { "step": 5, "epoch": 0.04, "loss": 2.91,
                      "learning_rate": 1e-05, "elapsed_s": 35 } ],
    "validation": [ { "step": 129, "epoch": 1.0, "val_loss": 2.45,
                      "metrics":  { "bleu": 0.075, "rougeL": 0.23 },
                      "sections": { "findings": {...}, "impression": {...} },
                      "eval_seconds": 214.0, "elapsed_s": 903.0 } ]
  },
  "best": { "metric": "rougeL", "value": 0.28, "epoch": 4.0, "step": 516,
            "val_loss": 2.36, "metrics": {...}, "sections": {...},
            "adapter": "training/.../results/best_adapter" }
}
```

Scritto in modo **atomico** (file temporaneo + rename) **dopo ogni valutazione**:
se il processo muore, su disco resta l'ultima versione completa, mai un JSON
troncato. In caso di crash lo `status` diventa `failed` ma curve e best restano.

Accanto, nella stessa cartella:

| File | Contenuto |
|---|---|
| `best_adapter/` | miglior adapter LoRA + `best_info.json` |
| `val_predictions/step<N>.json` | **tutti i sample di validation con il loro ground truth**, a ogni valutazione |
| `val_predictions_best.json` / `.csv` | gli stessi, per il miglior checkpoint |
| `train_history.csv`, `val_history.csv` | le stesse curve in CSV, per un foglio di calcolo |
| `test/metrics.json` | metriche di test (suite completa + operative) |
| `test/disaggregated.json`, `test/predictions.csv` | dettaglio per fattore e predizioni |
| `run.log` | log integrale (scritto da `run_all.py`) |
| `archive/<timestamp>-<esito>/` | copia completa delle esecuzioni precedenti |

Rilanciare un esperimento **non distrugge** il risultato precedente: a fine run
(sia in caso di successo sia di errore) gli artefatti vengono copiati in
`archive/<timestamp>-<esito>/`, e la successiva riparte da una cartella live
pulita. Se una run viene uccisa senza poter archiviare, ci pensa quella dopo,
riconoscendola dallo `status: running` rimasto sul disco.

La pulizia all'avvio non è un dettaglio: i file `val_predictions/step<N>.json`
sono nominati per step, quindi due esecuzioni di durata diversa lascerebbero
nella stessa cartella file appartenenti a run diverse, senza che nulla lo segnali.

Si disattiva con `archive_results = false` in `defaults.toml`; `archive_adapter
= false` archivia tutto tranne l'adapter, che è il pezzo pesante (~35 MB sul 2B,
~400 MB sul 32B).

## Aggregare e confrontare

```bash
python training/collect_results.py                     # tabella a schermo + file
python training/collect_results.py --sort best.rougeL
python training/collect_results.py --only "it_2B_*"
```

Produce in `training/results/` due file complementari:

| File | Forma | A cosa serve |
|---|---|---|
| `experiments.json` | **larga**: una riga piatta per esperimento, incluse `baseline.*` e `delta.*` (~60 colonne: `best.rougeL`, `best.findings.rougeL`, `test.chexbert_f1_micro_top5`, `config.learning_rate`, `provenance.git.git.commit`, …) | tabelle, ordinamenti, `pd.DataFrame(...)` |
| `curves.json` | **lunga** (tidy): una riga per `(esperimento, curva, step, metrica, valore)` | grafici, senza altre trasformazioni |

Le stesse due tabelle anche in CSV. Esempi:

```python
import json, pandas as pd
df = pd.DataFrame(json.load(open("training/results/experiments.json"))["experiments"])
df.sort_values("best.rougeL", ascending=False)[["experiment", "best.rougeL", "test.rougeL"]]

curves = pd.DataFrame(json.load(open("training/results/curves.json"))["points"])
loss = curves[(curves.curve == "train") & (curves.metric == "loss")]
for name, g in loss.groupby("experiment"):
    plt.plot(g.step, g.value, label=name)
```

Il formato lungo tiene i fattori (`lang`, `model`, `mode`, `dataset_code`) su ogni
punto, così si raggruppa e si sfaccetta direttamente: una curva per modalità, un
pannello per dataset, senza rimettere insieme nulla a mano.

## Validazione e early stopping

In validazione girano solo **bleu** e **rougeL**: sono veloci e bastano per il
gate. Il gate è `rougeL` sulla **media delle sezioni** (`monitor_metric`), con
`early_stopping_patience` contato in *eventi di valutazione*, non epoche. Il best
adapter viene salvato a ogni miglioramento.

La suite completa — BERTScore, ClinicalBERT, CheXbert — **non** gira qui: è un
processo separato sul test set.

A ogni valutazione viene scritto `val_predictions/step<N>.json` con **tutti** i
sample: per ciascuno il riferimento e la predizione integrali, le due sezioni già
separate (findings / impression) — cioè esattamente i testi su cui sono calcolate
le metriche — e i `factors`, per filtrare per categoria diagnostica o proiezione
senza tornare al dataset. Tenendo i file di più valutazioni si vede come le
generazioni evolvono durante il training.

```python
import json
d = json.load(open("training/it/.../results/val_predictions_best.json"))
peggiori = [s for s in d["samples"] if not s["prediction_sections"]["impression"]]
cardio  = [s for s in d["samples"] if "Cardiomegaly" in s["factors"]["diagnostic_category"]]
```

## Baseline zero-shot

Accanto a `lora/` e `qlora/` c'è `baseline/`: il modello base senza fine-tuning,
valutato sullo stesso test set con gli stessi prompt, gli stessi parametri di
generazione e le stesse metriche.

```bash
python3 training/it/Qwen-3-VL-2B-Instruct/baseline/iu_xray_r2gen_FL-FI/baseline_it_2B_FL-FI.py
python3 training/run_all.py --baselines --dry-run     # le 12
python3 training/run_all.py --baselines --skip-done
```

**Una sola baseline per (modello, dataset)**, condivisa da `lora` e `qlora`: il
modello base non dipende da come lo si addestrerà poi. Le cartelle `baseline/`
sono create dal generatore, non a mano.

Non è un accessorio: senza il riferimento, le metriche del modello addestrato non
sono interpretabili. Su IU X-ray in particolare, **il 46% degli studi di test ha un
referto identico a uno di training** (referti normali templatizzati: i 92 più
frequenti coprono un quarto del dataset). Un modello che impara i template ottiene
BLEU e ROUGE-L alti senza capacità diagnostica, e solo il confronto con la base
dice quanto il fine-tuning abbia davvero aggiunto.

Gira in **bf16**, il modello come pubblicato, così i delta di `lora` e `qlora` sono
confrontabili fra loro. Conseguenza da dichiarare nella model card: per gli
esperimenti `qlora` il delta include anche l'effetto della quantizzazione a 4 bit,
non solo il fine-tuning. (`OVERRIDES = {"load_in_4bit": True}` se un giorno vuoi
la variante quantizzata.)

Output in `<baseline>/results/metrics.json`, con due letture delle metriche:

| | |
|---|---|
| `by_section` | split su `<SEP>`, header rimossi, media — **confrontabile** con il fine-tuned |
| `raw` | testo integrale contro testo integrale — **indipendente dal formato** |
| `format_compliance` | quante generazioni contengono `<SEP>` |

Il modello base con ogni probabilità non produrrà `Reperti: … <SEP> Impressione: …`,
quindi `by_section` lo penalizza. La differenza fra `by_section` e `raw`, insieme a
`format_compliance`, dice quanto del divario è **formato** e quanto **contenuto**;
CheXbert, che estrae etichette cliniche, è l'indicatore più robusto al formato.

## Test set: processo a parte

```bash
python training/evaluate_test.py --experiment it_2B_qlora_FL-FI
```

Ricarica base + `best_adapter`, genera sul test set, calcola tutte le metriche
(per sezione e media) più la disaggregazione per categoria diagnostica e
proiezione. Output in `results/test/`: `metrics.json`, `disaggregated.json`,
`predictions.csv`. Il test set non viene mai toccato durante il training.

Opzioni utili: `--max-samples 20` per una prova rapida, `--split val` per
riconfrontare la validation, `--adapter PATH` per valutare un checkpoint diverso,
`--no-mlflow` per non tracciare.

## MLflow

Un run per esperimento, nome = nome dell'esperimento. Registra parametri,
`train.loss` e `learning_rate` per step, `val.loss` e tutte le metriche di
validation per step, gli artefatti, e la **provenienza a tre vie**: commit git,
hash DVC del dataset, versione del dataset. Il `run_id` viene ricopiato in
`results.json` (`provenance.mlflow.run_id`), così dal file si risale al run e
viceversa. La valutazione di test crea un run separato `<esperimento>__test` con
tag `phase=evaluation`, e vi registra le tre dimensioni di REQUISITI §3.8:
lessicale (BLEU, ROUGE-L), semantica/clinica (BERTScore, ClinicalBERT, CheXbert)
e **operativa** (`operational.latency_p50_s`, `p95`, `throughput_req_s`,
`vram_peak_gb`).

Per verificare che tutto questo sia davvero in regola:

```bash
python training/check_mlflow.py          # requisito per requisito
python training/check_mlflow.py --live   # run reale su store locale + rilettura
```

Esce con 1 se trova lacune, quindi si può usare in CI.

Il Model Registry resta **manuale**: la promozione a staging/production è una
decisione deliberata da prendere dopo aver visto le metriche di test, non un
effetto collaterale del training.

Server di default `http://127.0.0.1:5000`, sovrascrivibile con
`MLFLOW_TRACKING_URI` o in `defaults.toml`. Per disattivarlo:
`mlflow_enabled = false`.

## Libreria condivisa

Il codice vive in `src/shield/training/` — gli script sono sottili di proposito,
così una correzione vale per tutti e 24 insieme:

| Modulo | |
|---|---|
| `config.py` | `Identity` (percorso → identità), `Config`, precedenza dei tre livelli |
| `results.py` | `ResultsWriter`: results.json atomico e incrementale |
| `model.py` | modello + processor (lora/qlora), collator con masking del prompt, probe lunghezze |
| `dashboard.py` | cruscotto testuale live: 2 loss, batch, epoca, progress bar, ETA, tempo dall'inizio |
| `evaluation.py` | loss di validation teacher-forced, generazione batch, metriche per sezione |
| `callbacks.py` | log della loss, valutazione generativa, early stopping, best checkpoint |
| `runner.py` | orchestrazione di un esperimento e salvataggio di tutto |

## Prima di partire

Servono i dataset costruiti e una GPU:

```bash
uv run python -m shield.data.build --all     # richiede iu_xray_r2gen_final
python training/generate.py
python training/run_all.py --dry-run
```

Ogni esperimento gira in un **processo separato**: la memoria GPU viene
rilasciata fra un modello e il successivo, e un fallimento non porta giù la
campagna (`--continue-on-error` per non fermarsi al primo).
