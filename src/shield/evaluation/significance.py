from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .operational import percentile

DEFAULT_RESAMPLES = 1000
_CI_LOW_PCT = 2.5
_CI_HIGH_PCT = 97.5


def _summarize(
    delta_obs: float, deltas: list[float], n: int, n_resamples: int
) -> dict[str, Any]:
    deltas = sorted(deltas)
    ci_low = percentile(deltas, _CI_LOW_PCT)
    ci_high = percentile(deltas, _CI_HIGH_PCT)
    n_le_zero = sum(1 for delta in deltas if delta <= 0.0)
    return {
        "delta": delta_obs,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": (1 + n_le_zero) / (n_resamples + 1),
        "significant": bool(ci_low > 0.0),
        "n": n,
        "n_resamples": n_resamples,
    }


def _resample_indices(n: int, n_resamples: int, seed: int):
    rng = random.Random(seed)
    for _ in range(n_resamples):
        yield [rng.randrange(n) for _ in range(n)]


def paired_bootstrap_mean(
    scores_current: Sequence[float],
    scores_baseline: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    if len(scores_current) != len(scores_baseline):
        raise ValueError(
            f"Punteggi non appaiati: {len(scores_current)} vs {len(scores_baseline)} "
            f"(allineare per id prima del bootstrap)."
        )
    n = len(scores_current)
    if n == 0:
        raise ValueError("Nessun esempio: bootstrap non calcolabile.")
    diffs = [
        current - baseline for current, baseline in zip(scores_current, scores_baseline)
    ]
    delta_obs = sum(diffs) / n
    deltas = [
        sum(diffs[i] for i in indices) / n
        for indices in _resample_indices(n, n_resamples, seed)
    ]
    return _summarize(delta_obs, deltas, n, n_resamples)


def paired_bootstrap_corpus(
    preds_current: Sequence[str],
    preds_baseline: Sequence[str],
    references: Sequence[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    if not (len(preds_current) == len(preds_baseline) == len(references)):
        raise ValueError(
            f"Liste non appaiate: {len(preds_current)}/{len(preds_baseline)}/{len(references)}."
        )
    n = len(references)
    if n == 0:
        raise ValueError("Nessun esempio: bootstrap non calcolabile.")
    delta_obs = metric_fn(list(preds_current), list(references)) - metric_fn(
        list(preds_baseline), list(references)
    )
    deltas: list[float] = []
    for indices in _resample_indices(n, n_resamples, seed):
        sub_refs = [references[i] for i in indices]
        deltas.append(
            metric_fn([preds_current[i] for i in indices], sub_refs)
            - metric_fn([preds_baseline[i] for i in indices], sub_refs)
        )
    return _summarize(delta_obs, deltas, n, n_resamples)


def _per_example_rougeL(predictions: list[str], references: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [
        scorer.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references)
    ]


def _per_example_bertscore_f1(
    predictions: list[str], references: list[str], model_type: str
) -> list[float]:
    from bert_score import score as bert_score_fn

    _precision, _recall, f1 = bert_score_fn(
        predictions, references, model_type=model_type, lang="en", verbose=False
    )
    return [float(value) for value in f1]


def _corpus_bleu_score(predictions: list[str], references: list[str]) -> float:
    from .metrics import corpus_bleu

    return corpus_bleu(predictions, references)["bleu"]


def significance_vs_baseline(
    current_records: list[Mapping[str, Any]],
    baseline_records: list[Mapping[str, Any]],
    metrics: list[str],
    *,
    lexical_normalizer: Callable[[str], str] | None = None,
    bertscore_model_type: str = "xlm-roberta-large",
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    current_by_id = {
        rec["id"]: rec for rec in current_records if rec.get("id") is not None
    }
    baseline_by_id = {
        rec["id"]: rec for rec in baseline_records if rec.get("id") is not None
    }
    common_ids = sorted(set(current_by_id) & set(baseline_by_id))
    out: dict[str, Any] = {
        "method": "paired_bootstrap (Koehn 2004), IC95 percentile, p one-sided H0: delta<=0",
        "n_current": len(current_by_id),
        "n_baseline": len(baseline_by_id),
        "n_common": len(common_ids),
        "n_only_current": len(current_by_id) - len(common_ids),
        "n_only_baseline": len(baseline_by_id) - len(common_ids),
        "n_resamples": n_resamples,
        "seed": seed,
        "metrics": {},
    }
    if len(common_ids) < 2:
        out["status"] = "not_estimable: meno di 2 esempi comuni fra le due run"
        return out

    preds_current = [str(current_by_id[i]["prediction"]) for i in common_ids]
    preds_baseline = [str(baseline_by_id[i]["prediction"]) for i in common_ids]
    refs_natural = [str(current_by_id[i]["reference"]) for i in common_ids]
    refs_lexical = [
        str(current_by_id[i].get("reference_lexical", current_by_id[i]["reference"]))
        for i in common_ids
    ]
    if lexical_normalizer is not None:
        lex_current = [lexical_normalizer(text) for text in preds_current]
        lex_baseline = [lexical_normalizer(text) for text in preds_baseline]
        lex_refs = [lexical_normalizer(text) for text in refs_lexical]
    else:
        lex_current, lex_baseline, lex_refs = (
            preds_current,
            preds_baseline,
            refs_lexical,
        )

    results: dict[str, Any] = out["metrics"]
    for metric in metrics:
        if metric == "bleu":
            results["bleu"] = paired_bootstrap_corpus(
                lex_current,
                lex_baseline,
                lex_refs,
                _corpus_bleu_score,
                n_resamples,
                seed,
            )
        elif metric == "rougeL":
            results["rougeL"] = paired_bootstrap_mean(
                _per_example_rougeL(lex_current, lex_refs),
                _per_example_rougeL(lex_baseline, lex_refs),
                n_resamples,
                seed,
            )
        elif metric == "bertscore_f1":
            results["bertscore_f1"] = paired_bootstrap_mean(
                _per_example_bertscore_f1(
                    preds_current, refs_natural, bertscore_model_type
                ),
                _per_example_bertscore_f1(
                    preds_baseline, refs_natural, bertscore_model_type
                ),
                n_resamples,
                seed,
            )
        else:
            results[metric] = {
                "status": f"metrica non supportata dal bootstrap: {metric!r}"
            }
    return out
