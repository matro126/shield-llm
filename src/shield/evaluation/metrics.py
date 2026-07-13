from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


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


def bertscore_f1(
    predictions: list[str],
    references: list[str],
    model_type: str = "xlm-roberta-large",
    lang: str = "en",
) -> dict[str, float]:
    from bert_score import score as bert_score_fn
    from bert_score.utils import model2layers

    num_layers = model2layers.get(model_type) or model2layers.get(Path(model_type).name)
    if num_layers is None:
        raise ValueError(
            f"bertscore: layer ottimale sconosciuto per '{model_type}'. Se è un path "
            f"locale, la cartella deve chiamarsi come il modello HF (es. "
            f"'xlm-roberta-large'); senza pin del layer i valori non sarebbero "
            f"confrontabili con le run precedenti."
        )

    precision, recall, f1 = bert_score_fn(
        predictions,
        references,
        model_type=model_type,
        num_layers=num_layers,
        lang=lang,
        verbose=False,
    )
    return {
        "bertscore_precision": precision.mean().item(),
        "bertscore_recall": recall.mean().item(),
        "bertscore_f1": f1.mean().item(),
    }


def clinicalbert_similarity(
    predictions: list[str],
    references: list[str],
    model_type: str = "emilyalsentzer/Bio_ClinicalBERT",
    num_layers: int | None = None,
) -> dict[str, float]:
    from bert_score import BERTScorer

    if num_layers is None:
        from transformers import AutoConfig

        num_layers = AutoConfig.from_pretrained(model_type).num_hidden_layers

    scorer = BERTScorer(model_type=model_type, num_layers=num_layers)
    if scorer._tokenizer.model_max_length > 10**6:
        scorer._tokenizer.model_max_length = min(
            getattr(scorer._model.config, "max_position_embeddings", 512), 512
        )
    precision, recall, f1 = scorer.score(predictions, references)
    return {
        "clinicalbert_precision": precision.mean().item(),
        "clinicalbert_recall": recall.mean().item(),
        "clinicalbert_f1": f1.mean().item(),
    }


def _ensure_chexbert_checkpoint() -> None:
    flat = Path.home() / ".cache" / "chexbert" / "chexbert.pth"
    if flat.exists():
        return
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "models" / "evaluation" / "chexbert" / "chexbert.pth",
        *sorted(flat.parent.glob("models--*/snapshots/*/chexbert.pth")),
    ]
    for candidate in candidates:
        if candidate.exists():
            flat.parent.mkdir(parents=True, exist_ok=True)
            flat.symlink_to(candidate)
            print(f"[eval] checkpoint CheXBERT agganciato: {flat} -> {candidate}")
            return


def _report_f1(report: dict, avg_key: str) -> float:
    entry = report.get(avg_key, {})
    return float(entry.get("f1-score", 0.0)) if isinstance(entry, dict) else 0.0


def chexbert_f1(
    predictions: list[str],
    references: list[str],
    device: str | None = None,
) -> dict[str, float]:
    from f1chexbert import F1CheXbert

    _ensure_chexbert_checkpoint()
    labeler = F1CheXbert(device=device)

    tokenizer = getattr(labeler, "tokenizer", None)
    if tokenizer is not None and not hasattr(tokenizer, "encode_plus"):

        def _encode_plus(text: list[str], **_kwargs: object) -> dict[str, list[int]]:
            ids = tokenizer.convert_tokens_to_ids(text)
            return {"input_ids": [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]}

        tokenizer.encode_plus = _encode_plus

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report

    refs_arr = np.array([labeler.get_label(text.strip()) for text in references])
    hyps_arr = np.array([labeler.get_label(text.strip()) for text in predictions])
    top5 = labeler.target_names_5_index
    refs_5, hyps_5 = refs_arr[:, top5], hyps_arr[:, top5]

    accuracy = accuracy_score(y_true=refs_5, y_pred=hyps_5)
    class_report = classification_report(
        refs_arr, hyps_arr, output_dict=True, zero_division=0
    )
    class_report_5 = classification_report(
        refs_5, hyps_5, output_dict=True, zero_division=0
    )
    return {
        "chexbert_accuracy": float(accuracy),
        "chexbert_f1_micro": _report_f1(class_report, "micro avg"),
        "chexbert_f1_macro": _report_f1(class_report, "macro avg"),
        "chexbert_f1_micro_top5": _report_f1(class_report_5, "micro avg"),
        "chexbert_f1_macro_top5": _report_f1(class_report_5, "macro avg"),
    }


def compute_text_metrics(
    predictions: list[str],
    references: list[str],
    metrics: list[str],
    bertscore_model_type: str = "xlm-roberta-large",
    bertscore_lang: str = "en",
    clinicalbert_model_type: str = "emilyalsentzer/Bio_ClinicalBERT",
    clinicalbert_num_layers: int | None = None,
    chexbert_device: str | None = None,
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
        out.update(chexbert_f1(predictions, references, chexbert_device))
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
