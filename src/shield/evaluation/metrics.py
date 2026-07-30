from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


@lru_cache(maxsize=4)
def _bert_scorer(model_type: str, num_layers: int, lang: str):
    from bert_score import BERTScorer

    return BERTScorer(
        model_type=model_type, num_layers=num_layers, lang=lang, idf=False
    )


@lru_cache(maxsize=4)
def _clinical_scorer(model_type: str, num_layers: int):
    from bert_score import BERTScorer

    scorer = BERTScorer(model_type=model_type, num_layers=num_layers, idf=False)
    if scorer._tokenizer.model_max_length > 10**6:
        scorer._tokenizer.model_max_length = min(
            getattr(scorer._model.config, "max_position_embeddings", 512), 512
        )
    return scorer


_TRANSLATIONS: dict[tuple[str, str], str] = {}
_LABELS: dict[str, list[int]] = {}


def _local_or_hub(model_type: str) -> str:
    if Path(model_type).is_dir():
        return str(Path(model_type).resolve())
    local = Path(__file__).resolve().parents[3] / model_type
    if local.is_dir():
        return str(local)
    return model_type


@lru_cache(maxsize=2)
def _translator(model_type: str, device: str | None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    source = _local_or_hub(model_type)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSeq2SeqLM.from_pretrained(source).to(resolved).eval()
    return tokenizer, model, resolved


_SENTENCE = re.compile(r"(?<=[.;:!?])\s+")


def _split_for_translation(text: str, tokenizer, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in _SENTENCE.split(text.strip()):
        if not sentence:
            continue
        size = len(tokenizer.tokenize(sentence))
        if current and length + size > max_tokens:
            pieces.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += size
    if current:
        pieces.append(" ".join(current))
    return pieces


def translate(
    texts: list[str],
    model_type: str = "Helsinki-NLP/opus-mt-it-en",
    device: str | None = None,
    batch_size: int = 32,
    max_tokens: int = 400,
) -> list[str]:
    import torch

    tokenizer, model, resolved = _translator(model_type, device)
    per_text: list[list[str]] = []
    pending: set[str] = set()
    for text in texts:
        pieces = _split_for_translation(text, tokenizer, max_tokens) if text.strip() else []
        per_text.append(pieces)
        pending.update(p for p in pieces if (model_type, p) not in _TRANSLATIONS)

    ordered = sorted(pending)
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        batch = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(resolved)
        with torch.inference_mode():
            generated = model.generate(**batch, max_new_tokens=512, num_beams=4)
        for source, rendered in zip(
            chunk, tokenizer.batch_decode(generated, skip_special_tokens=True)
        ):
            _TRANSLATIONS[(model_type, source)] = rendered

    return [
        " ".join(_TRANSLATIONS.get((model_type, p), "") for p in pieces).strip()
        for pieces in per_text
    ]


@lru_cache(maxsize=2)
def _chexbert_labeler(device: str | None):
    from f1chexbert import F1CheXbert

    _ensure_chexbert_checkpoint()
    labeler = F1CheXbert(device=device)

    tokenizer = getattr(labeler, "tokenizer", None)
    if tokenizer is not None and not hasattr(tokenizer, "encode_plus"):

        def _encode_plus(text: list[str], **_kwargs: object) -> dict[str, list[int]]:
            ids = tokenizer.convert_tokens_to_ids(text)
            return {"input_ids": [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]}

        tokenizer.encode_plus = _encode_plus
    return labeler


def preload_metric_models(
    metrics: list[str],
    chexbert_translate: bool = False,
    chexbert_translator: str = "Helsinki-NLP/opus-mt-it-en",
    device: str | None = None,
) -> list[str]:
    caricati = []
    if "chexbert" in metrics:
        if chexbert_translate:
            translate(["Il cuore e' di dimensioni normali."], chexbert_translator, device)
            caricati.append(chexbert_translator)
        _chexbert_labeler(device)
        caricati.append("chexbert")
    return caricati


def clear_metric_models() -> None:
    for cached in (_bert_scorer, _clinical_scorer, _chexbert_labeler, _translator):
        cached.cache_clear()
    _TRANSLATIONS.clear()
    _LABELS.clear()


def corpus_bleu(predictions: list[str], references: list[str]) -> dict[str, float]:
    import contextlib
    import os

    from pycocoevalcap.bleu.bleu import Bleu

    keys = ("bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4")
    if not predictions:
        return {key: 0.0 for key in keys}

    gts = {index: [reference] for index, reference in enumerate(references)}
    res = {index: [prediction] for index, prediction in enumerate(predictions)}

    with (
        open(os.devnull, "w", encoding="utf-8") as devnull,
        contextlib.redirect_stdout(devnull),
    ):
        cumulative, _per_sample = Bleu(4).compute_score(gts, res)
    return {
        "bleu": float(cumulative[3]),
        "bleu_1": float(cumulative[0]),
        "bleu_2": float(cumulative[1]),
        "bleu_3": float(cumulative[2]),
        "bleu_4": float(cumulative[3]),
    }


def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=False)
    accumulator = {rouge_type: 0.0 for rouge_type in _ROUGE_TYPES}
    count = 0
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for rouge_type in _ROUGE_TYPES:
            accumulator[rouge_type] += scores[rouge_type].fmeasure
        count += 1
    if count == 0:
        return {rouge_type: 0.0 for rouge_type in _ROUGE_TYPES}
    return {rouge_type: accumulator[rouge_type] / count for rouge_type in _ROUGE_TYPES}


def _nonempty_pairs(
    predictions: list[str], references: list[str]
) -> tuple[list[str], list[str], list[int]]:
    keep = [
        i
        for i in range(len(predictions))
        if predictions[i].strip() and references[i].strip()
    ]
    return [predictions[i] for i in keep], [references[i] for i in keep], keep


def _mean_with_zeros(values, total: int) -> float:
    if total == 0:
        return 0.0
    return float(values.sum().item() / total)


def bertscore_f1(
    predictions: list[str],
    references: list[str],
    model_type: str = "xlm-roberta-large",
    lang: str = "en",
) -> dict[str, float]:
    from bert_score.utils import model2layers

    num_layers = model2layers.get(model_type) or model2layers.get(Path(model_type).name)
    if num_layers is None:
        raise ValueError(
            f"bertscore: layer ottimale sconosciuto per '{model_type}'. Se è un path "
            f"locale, la cartella deve chiamarsi come il modello HF (es. "
            f"'xlm-roberta-large'); senza pin del layer i valori non sarebbero "
            f"confrontabili con le run precedenti."
        )

    keys = ("bertscore_precision", "bertscore_recall", "bertscore_f1")
    preds, refs, keep = _nonempty_pairs(predictions, references)
    if not keep:
        return dict.fromkeys(keys, 0.0)

    precision, recall, f1 = _bert_scorer(model_type, num_layers, lang).score(preds, refs)
    total = len(predictions)
    return {
        "bertscore_precision": _mean_with_zeros(precision, total),
        "bertscore_recall": _mean_with_zeros(recall, total),
        "bertscore_f1": _mean_with_zeros(f1, total),
    }


def clinicalbert_similarity(
    predictions: list[str],
    references: list[str],
    model_type: str = "emilyalsentzer/Bio_ClinicalBERT",
    num_layers: int | None = None,
) -> dict[str, float]:
    if num_layers is None:
        from transformers import AutoConfig

        num_layers = AutoConfig.from_pretrained(model_type).num_hidden_layers

    keys = ("clinicalbert_precision", "clinicalbert_recall", "clinicalbert_f1")
    preds, refs, keep = _nonempty_pairs(predictions, references)
    if not keep:
        return dict.fromkeys(keys, 0.0)

    precision, recall, f1 = _clinical_scorer(model_type, num_layers).score(preds, refs)
    total = len(predictions)
    return {
        "clinicalbert_precision": _mean_with_zeros(precision, total),
        "clinicalbert_recall": _mean_with_zeros(recall, total),
        "clinicalbert_f1": _mean_with_zeros(f1, total),
    }


def _ensure_chexbert_checkpoint() -> None:
    flat = Path.home() / ".cache" / "chexbert" / "chexbert.pth"
    if flat.exists():
        return
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "models" / "others" / "chexbert" / "chexbert.pth",
        project_root / "models" / "evaluation" / "chexbert" / "chexbert.pth",
        *sorted(flat.parent.glob("models--*/snapshots/*/chexbert.pth")),
    ]
    for candidate in candidates:
        if candidate.exists():
            flat.parent.mkdir(parents=True, exist_ok=True)
            flat.symlink_to(candidate)
            print(f"[eval] checkpoint CheXBERT agganciato: {flat} -> {candidate}")
            return


_AVERAGES = (("micro", "micro avg"), ("macro", "macro avg"))


def _report_scores(report: dict, suffix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for nome, chiave in _AVERAGES:
        entry = report.get(chiave)
        if not isinstance(entry, dict):
            continue
        out[f"chexbert_precision_{nome}{suffix}"] = float(entry.get("precision", 0.0))
        out[f"chexbert_recall_{nome}{suffix}"] = float(entry.get("recall", 0.0))
        out[f"chexbert_f1_{nome}{suffix}"] = float(entry.get("f1-score", 0.0))
    return out


def _per_class_scores(report: dict, names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for indice, nome in enumerate(names):
        entry = report.get(str(indice))
        if not isinstance(entry, dict):
            continue
        chiave = nome.replace(" ", "_").replace("/", "_")
        out[f"chexbert_cls_{chiave}_precision"] = float(entry.get("precision", 0.0))
        out[f"chexbert_cls_{chiave}_recall"] = float(entry.get("recall", 0.0))
        out[f"chexbert_cls_{chiave}_f1"] = float(entry.get("f1-score", 0.0))
        out[f"chexbert_cls_{chiave}_support"] = float(entry.get("support", 0.0))
    return out


def chexbert_f1(
    predictions: list[str],
    references: list[str],
    device: str | None = None,
    per_class: bool = False,
) -> dict[str, float]:
    labeler = _chexbert_labeler(device)

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report

    def _label(text: str) -> list[int]:
        key = text.strip() or "."
        if key not in _LABELS:
            _LABELS[key] = labeler.get_label(key)
        return _LABELS[key]

    refs_arr = np.array([_label(t) for t in references])
    hyps_arr = np.array([_label(t) for t in predictions])
    top5 = labeler.target_names_5_index
    refs_5, hyps_5 = refs_arr[:, top5], hyps_arr[:, top5]

    report = classification_report(
        refs_arr, hyps_arr, output_dict=True, zero_division=0
    )
    report_5 = classification_report(
        refs_5, hyps_5, output_dict=True, zero_division=0
    )

    out: dict[str, float] = {
        "chexbert_accuracy": float(accuracy_score(y_true=refs_5, y_pred=hyps_5)),
    }
    out.update(_report_scores(report))
    out.update(_report_scores(report_5, "_top5"))
    if per_class:
        out.update(_per_class_scores(report, [str(n) for n in labeler.target_names]))
    return out


def compute_text_metrics(
    predictions: list[str],
    references: list[str],
    metrics: list[str],
    bertscore_model_type: str = "xlm-roberta-large",
    bertscore_lang: str = "en",
    clinicalbert_model_type: str = "emilyalsentzer/Bio_ClinicalBERT",
    clinicalbert_num_layers: int | None = None,
    chexbert_device: str | None = None,
    chexbert_translate: bool = False,
    chexbert_translator: str = "Helsinki-NLP/opus-mt-it-en",
    chexbert_per_class: bool = False,
    lexical_normalizer: Callable[[str], str] | None = None,
    lexical_references: list[str] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {"num_examples": float(len(predictions))}
    if "bleu" in metrics or "rougeL" in metrics or "rouge" in metrics:
        lex_ref_source = (
            lexical_references if lexical_references is not None else references
        )
        if lexical_normalizer is not None:
            lex_pred = [lexical_normalizer(text) for text in predictions]
            lex_ref = [lexical_normalizer(text) for text in lex_ref_source]
        else:
            lex_pred, lex_ref = predictions, lex_ref_source
        if "bleu" in metrics:
            out.update(corpus_bleu(lex_pred, lex_ref))
        if "rougeL" in metrics or "rouge" in metrics:
            out.update(rouge_scores(lex_pred, lex_ref))
    if "bertscore" in metrics:
        out.update(
            bertscore_f1(predictions, references, bertscore_model_type, bertscore_lang)
        )
    if "clinicalbert" in metrics:
        out.update(
            clinicalbert_similarity(
                predictions,
                references,
                clinicalbert_model_type,
                clinicalbert_num_layers,
            )
        )
    if "chexbert" in metrics:
        if chexbert_translate:
            chex_pred = translate(predictions, chexbert_translator, chexbert_device)
            chex_ref = translate(references, chexbert_translator, chexbert_device)
        else:
            chex_pred, chex_ref = predictions, references
        out.update(
            chexbert_f1(chex_pred, chex_ref, chexbert_device, chexbert_per_class)
        )
    return out


def lexical_metrics(
    predictions: list[str],
    references: list[str],
    lexical_normalizer: Callable[[str], str] | None = None,
) -> dict[str, float]:
    return compute_text_metrics(
        predictions,
        references,
        ["bleu", "rougeL"],
        lexical_normalizer=lexical_normalizer,
    )
