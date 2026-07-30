from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from ..data import extract_assistant_text
from ..data.prompts import SEP
from ..evaluation import compute_text_metrics, sectioned_metrics
from ..evaluation.sections import SECTIONS


DISAGGREGATE_BY = ("diagnostic_category", "projection", "task_type")
MIN_SUBGROUP_SIZE = 20


def prompts_of(record: dict[str, Any]) -> tuple[str, str]:
    system = record["messages"][0]["content"]
    user = next(
        part["text"]
        for part in record["messages"][1]["content"]
        if part.get("type") == "text"
    )
    return system, user


def format_compliance(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    target: str,
) -> dict[str, Any]:
    atteso = target == "findings_impression"
    senza = [
        str(record["id"])
        for record, prediction in zip(records, predictions)
        if SEP not in prediction
    ] if atteso else []
    presenti = len(predictions) - len(senza)
    return {
        "separator_expected": atteso,
        "with_separator": presenti if atteso else None,
        "total": len(predictions),
        "ratio": round(presenti / max(len(predictions), 1), 4) if atteso else None,
        "missing": len(senza),
        "missing_ids": senza,
    }


def stampa_formato(diagnostica: dict[str, Any], target: str, mostra: int = 10) -> None:
    if not diagnostica["separator_expected"]:
        print(f"  formato   : {SEP} non atteso con target={target}")
        return
    print(f"  formato   : {diagnostica['with_separator']}/{diagnostica['total']} "
          f"generazioni contengono {SEP}")
    mancanti = diagnostica["missing_ids"]
    if not mancanti:
        return
    resto = len(mancanti) - mostra
    print(f"  ⚠️  {len(mancanti)} generazioni senza {SEP}: per queste l'impression "
          f"non e' separabile e le metriche di sezione attribuiscono tutto il testo "
          f"a findings.")
    print(f"      id: {', '.join(mancanti[:mostra])}"
          f"{f' … (+{resto})' if resto > 0 else ''}")
    print("      elenco completo in metrics.json → format_compliance.missing_ids")


def flatten_sectioned(result: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for section in SECTIONS:
        values = result.get(section)
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}.{key}"] = float(value)
    return flat


def compute_val_loss(
    model: Any,
    collator: Any,
    records: Sequence[dict[str, Any]],
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> float:
    import torch

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens, skipped = 0.0, 0, 0
    try:
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = collator(list(records[start : start + batch_size]))
                n_tokens = int((batch["labels"] != -100).sum())
                if n_tokens == 0:
                    skipped += 1
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                total_loss += float(model(**batch).loss) * n_tokens
                total_tokens += n_tokens
                if progress is not None:
                    progress(min(start + batch_size, len(records)), len(records))
    finally:
        if was_training:
            model.train()
    if total_tokens == 0:
        raise RuntimeError(
            "Validation loss non calcolabile: nessun token di risposta in nessun "
            "batch. La risposta viene troncata da max_seq_length: alzalo."
        )
    if skipped:
        print(
            f"  ⚠️  {skipped} batch di validation esclusi dalla loss (risposta "
            f"troncata da max_seq_length={collator.max_length}): la loss e' "
            "calcolata su un sottoinsieme.",
            flush=True,
        )
    return total_loss / max(total_tokens, 1)


def generate_predictions(
    model: Any,
    processor: Any,
    records: Sequence[dict[str, Any]],
    gen_batch_size: int,
    max_new_tokens: int,
    repetition_penalty: float,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[str], list[str]]:
    from ..evaluation.generate import generate_reports_batched

    system, user = prompts_of(records[0])
    divergent = [r["id"] for r in records if prompts_of(r) != (system, user)]
    if divergent:
        raise RuntimeError(
            f"{len(divergent)} record con prompt diversi dal primo: {divergent[:5]}. "
            "La generazione userebbe il prompt sbagliato."
        )
    items = [
        {"image_paths": r["images"], "system": system, "user": user} for r in records
    ]
    references = [extract_assistant_text(r) for r in records]
    empty = [r["id"] for r, ref in zip(records, references) if not ref.strip()]
    if empty:
        raise RuntimeError(
            f"{len(empty)} riferimenti vuoti (nessun turno assistant): {empty[:5]}"
        )

    was_training = model.training
    model.eval()
    model.config.use_cache = True
    predictions: list[str] = []
    try:
        for start in range(0, len(items), gen_batch_size):
            predictions += generate_reports_batched(
                model,
                processor,
                items[start : start + gen_batch_size],
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            if progress is not None:
                progress(len(predictions), len(items))
    finally:
        model.config.use_cache = False
        if was_training:
            model.train()
    return predictions, references


def evaluate_generative(
    model: Any,
    processor: Any,
    records: Sequence[dict[str, Any]],
    metric_names: Sequence[str],
    target: str,
    gen_batch_size: int,
    max_new_tokens: int,
    repetition_penalty: float,
    progress: Callable[[int, int], None] | None = None,
    **metric_kwargs: Any,
) -> tuple[dict[str, Any], list[str], list[str]]:
    predictions, references = generate_predictions(
        model,
        processor,
        records,
        gen_batch_size,
        max_new_tokens,
        repetition_penalty,
        progress,
    )
    if target == "findings_impression":
        missing = [
            r["id"] for r, ref in zip(records, references) if SEP not in ref
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} riferimenti senza il marcatore {SEP} pur essendo "
                f"target findings_impression: {missing[:5]}. Il ground truth non e' "
                "il target del dataset."
            )
    sectioned = sectioned_metrics(
        predictions, references, list(metric_names), target,
        metric_fn=compute_text_metrics, **metric_kwargs
    )
    return sectioned, predictions, references


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.time()

    def restart(self) -> float:
        self.t0 = time.time()
        return self.t0
