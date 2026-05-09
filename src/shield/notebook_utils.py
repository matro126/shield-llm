from __future__ import annotations

import json
import os
import random
import tomllib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from rouge_score import rouge_scorer as rs
from tqdm.auto import tqdm


SYSTEM_PROMPT = (
    "Sei un radiologo esperto. "
    "Ti verranno fornite una o più immagini mediche. "
    "Il tuo compito è: "
    "1) Descrivere in modo conciso i reperti visibili nell'immagine o nelle immagini, "
    "incluse le strutture anatomiche, le anomalie e le osservazioni rilevanti. "
    "2) Fornire un'impressione clinica concisa che riassuma i reperti principali. "
    "Fornisci la risposta in ESATTAMENTE questo formato:\n"
    "Reperti:\n<i tuoi reperti dettagliati qui>\n\n"
    "Impressione:\n<la tua impressione concisa qui>\n\n"
    "NON includere altre sezioni o preamboli."
)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("pyproject.toml non trovato: esegui dal repo SHIELD.")


def load_config(path: str | Path, project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or find_project_root()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_meta"] = {"config_path": str(config_path), "project_root": str(root)}
    return config


def resolve_path(config: dict[str, Any], section: str, key: str) -> Path:
    root = Path(config["_meta"]["project_root"])
    value = config[section][key]
    path = Path(value)
    return path if path.is_absolute() else root / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise TypeError(f"Formato messages non valido: {type(messages)!r}")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"Messaggio chat non valido: {type(message)!r}")
        item = dict(message)
        if item.get("role") == "assistant" and isinstance(item.get("content"), list):
            item["content"] = "\n".join(str(value) for value in item["content"])
        normalized.append(item)
    return normalized


def load_hf_dataset(path: str | Path):
    from datasets import Dataset

    records = load_json_records(path)
    dataset_records = []
    for example in records:
        messages = normalize_messages(example.get("messages", []))
        dataset_records.append(
            {
                **{key: value for key, value in example.items() if key != "messages"},
                "messages": json.dumps(messages, ensure_ascii=False),
            }
        )
    return Dataset.from_list(dataset_records)


def extract_assistant_text(example: dict[str, Any]) -> str:
    for message in normalize_messages(example.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def extract_image_path(example: dict[str, Any]) -> str | None:
    for message in normalize_messages(example.get("messages", [])):
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    return item.get("image")
    return None


def print_dataset_summary(train_dataset, val_dataset, test_records: list[dict[str, Any]]) -> None:
    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Train: {len(train_dataset)} esempi")
    print(f"Val:   {len(val_dataset)} esempi")
    print(f"Test:  {len(test_records)} esempi")
    lengths = [len(extract_assistant_text(example).split()) for example in test_records]
    if lengths:
        print(f"Test reference words: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")


class XRayDataCollator:
    """
    Collator per Qwen3-VL.

    Maschera system prompt, messaggio utente e image tokens con -100; lascia
    nella loss solo i token della risposta assistant.
    """

    def __init__(self, processor, max_seq_length: int = 2048):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self._im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._assistant_ids = processor.tokenizer(
            "assistant", add_special_tokens=False
        )["input_ids"]

    def _find_assistant_start(self, input_ids_row):
        ids = input_ids_row.tolist()
        pattern = [self._im_start_id] + self._assistant_ids
        plen = len(pattern)
        for index in range(len(ids) - plen, -1, -1):
            if ids[index:index + plen] == pattern:
                return index + plen + 1
        return None

    def __call__(self, examples):
        texts = []
        images_list = []

        for example in examples:
            messages = normalize_messages(example["messages"])
            image_path = extract_image_path(example)

            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
            else:
                warnings.warn(f"Immagine non trovata: {image_path}. Uso placeholder grigio.")
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            images_list.append([image])

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(text)

        batch = self.processor(
            text=texts,
            images=images_list,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        for index in range(labels.shape[0]):
            assistant_start = self._find_assistant_start(labels[index])
            if assistant_start is None:
                warnings.warn(
                    "Token '<|im_start|>assistant' non trovato. "
                    "Maschero l'intera sequenza."
                )
                labels[index, :] = -100
            else:
                labels[index, :assistant_start] = -100

        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def generate_report(
    model,
    processor,
    image_path: str,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.1,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image_path}]},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device
    inputs = processor(text=[text], images=[[image]], return_tensors="pt", padding=True).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def evaluate_on_test_set(
    model,
    processor,
    test_records: list[dict[str, Any]],
    label: str,
    bertscore_model_type: str = "xlm-roberta-large",
    max_new_tokens: int = 512,
    qualitative_limit: int = 20,
    keep_all_predictions: bool = False,
) -> dict[str, Any]:
    from bert_score import score as bert_score_fn

    print("=" * 60)
    print(f"VALUTAZIONE: {label}")
    print(f"Esempi nel test set: {len(test_records)}")
    print("=" * 60)

    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    qualitative: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    for index, example in enumerate(tqdm(test_records, desc=label)):
        image_path = extract_image_path(example)
        reference = extract_assistant_text(example)

        if not image_path or not os.path.exists(image_path):
            continue

        try:
            prediction = generate_report(
                model,
                processor,
                image_path,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            print(f"Errore esempio {index}: {exc}")
            continue

        predictions.append(prediction)
        references.append(reference)

        if len(qualitative) < qualitative_limit:
            qualitative.append(
                {
                    "index": index,
                    "image_path": image_path,
                    "reference": reference,
                    "prediction": prediction,
                }
            )
        if keep_all_predictions:
            prediction_records.append(
                {
                    "index": index,
                    "image_path": image_path,
                    "reference": reference,
                    "prediction": prediction,
                }
            )

    if not predictions:
        raise RuntimeError("Nessun esempio valutato. Controlla path immagini e test set.")

    rouge_acc = {"rouge1": [], "rouge2": [], "rougeL": []}
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for key in rouge_acc:
            rouge_acc[key].append(scores[key].fmeasure)
    rouge_avg = {key: sum(values) / len(values) for key, values in rouge_acc.items()}

    print("Calcolo BERTScore...")
    precision, recall, f1 = bert_score_fn(
        predictions,
        references,
        model_type=bertscore_model_type,
        lang="it",
        verbose=False,
    )
    bertscore_avg = {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }

    print(f"Esempi valutati: {len(predictions)}")
    print(f"ROUGE-1:      {rouge_avg['rouge1']:.4f}")
    print(f"ROUGE-2:      {rouge_avg['rouge2']:.4f}")
    print(f"ROUGE-L:      {rouge_avg['rougeL']:.4f}")
    print(f"BERTScore F1: {bertscore_avg['f1']:.4f}")

    results = {
        "label": label,
        "rouge": rouge_avg,
        "bertscore": bertscore_avg,
        "num_examples": len(predictions),
        "qualitative": qualitative,
    }
    if keep_all_predictions:
        results["predictions"] = prediction_records
    return results


def save_evaluation_results(
    results: dict[str, Any],
    metrics_path: Path,
    qualitative_path: Path,
    predictions_path: Path | None = None,
) -> None:
    ensure_dirs(metrics_path.parent, qualitative_path.parent)
    metrics = {
        key: value
        for key, value in results.items()
        if key not in {"qualitative", "predictions"}
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with qualitative_path.open("w", encoding="utf-8") as handle:
        json.dump(results["qualitative"], handle, indent=2, ensure_ascii=False)
    if predictions_path and "predictions" in results:
        ensure_dirs(predictions_path.parent)
        with predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(results["predictions"], handle, indent=2, ensure_ascii=False)
    print(f"Metriche salvate in: {metrics_path}")
    print(f"Esempi qualitativi salvati in: {qualitative_path}")
    if predictions_path and "predictions" in results:
        print(f"Predizioni complete salvate in: {predictions_path}")
