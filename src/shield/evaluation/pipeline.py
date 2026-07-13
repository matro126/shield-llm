from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _dataset_target_lang(dataset_root: str | Path) -> str:
    manifest_path = Path(dataset_root) / "manifest.yaml"
    if not manifest_path.exists():
        return "en"
    try:
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return "en"
    pre = manifest.get("preprocessing", {})
    lang = pre.get("target_lang") if isinstance(pre, Mapping) else None
    return str(lang) if lang else "en"


def _eval_params(config: Mapping[str, Any]) -> dict[str, Any]:
    ev = config.get("evaluation", {})
    ev = ev if isinstance(ev, Mapping) else {}
    return {
        "metrics_list": list(ev.get("metrics", ["bleu", "rougeL", "bertscore"])),
        "bertscore_model": ev.get("bertscore_model_type", "xlm-roberta-large"),
        "clinicalbert_model": ev.get(
            "clinicalbert_model_type", "emilyalsentzer/Bio_ClinicalBERT"
        ),
        "clinicalbert_layers": ev.get("clinicalbert_num_layers"),
        "factor_keys": list(ev.get("disaggregate_by", [])),
        "min_sub": int(ev.get("min_subgroup_size", 20)),
        "max_new_tokens": int(ev.get("max_new_tokens", 512)),
        "repetition_penalty": float(ev.get("repetition_penalty", 1.1)),
        "baseline_run": ev.get("baseline_run"),
        "significance": bool(ev.get("significance", True)),
        "bootstrap_resamples": int(ev.get("bootstrap_resamples", 1000)),
        "bootstrap_seed": int(ev.get("bootstrap_seed", 42)),
    }


def _default_output_dir(config: Mapping[str, Any], project_root: str | Path) -> Path:
    exp = config["experiment"]
    return Path(project_root) / "outputs" / exp["family"] / exp["name"] / "evaluation"


def _compute_metrics(
    pred_records: list[dict[str, Any]],
    params: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    from ..data.preprocessing import clean_report_r2gen
    from .disaggregate import disaggregate
    from .metrics import compute_text_metrics, lexical_metrics

    predictions = [record["prediction"] for record in pred_records]
    references = [record["reference"] for record in pred_records]
    references_lexical = [
        record.get("reference_lexical", record["reference"]) for record in pred_records
    ]
    if not predictions:
        raise RuntimeError("Nessun esempio da valutare: predictions vuote.")

    lexical_requested = any(
        name in params["metrics_list"] for name in ("bleu", "rougeL", "rouge")
    )
    n_missing_lexical = sum(
        1 for record in pred_records if "reference_lexical" not in record
    )
    if lexical_requested and n_missing_lexical:
        raise RuntimeError(
            f"{n_missing_lexical}/{len(pred_records)} predictions senza 'reference_lexical' "
            f"(file precedenti al fix dual-reference). BLEU/ROUGE ricadrebbero sulla reference "
            f"naturale → numeri lessicali NON confrontabili con la run originale. Rigenera le "
            f"predictions (run_evaluation) o scartale; oppure ri-score solo metriche semantiche."
        )

    aggregate = compute_text_metrics(
        predictions,
        references,
        params["metrics_list"],
        params["bertscore_model"],
        bertscore_lang=params.get("bertscore_lang", "en"),
        clinicalbert_model_type=params["clinicalbert_model"],
        clinicalbert_num_layers=params["clinicalbert_layers"],
        lexical_normalizer=clean_report_r2gen,
        lexical_references=references_lexical,
    )
    disaggregated: dict[str, Any] = {}
    if params["factor_keys"]:
        from ..data.preprocessing import CHEXPERT_CATEGORIES

        known_values = {"diagnostic_category": CHEXPERT_CATEGORIES}

        def _factor_metric(sub_pred: list[str], sub_ref: list[str]) -> dict[str, float]:
            return lexical_metrics(
                sub_pred, sub_ref, lexical_normalizer=clean_report_r2gen
            )

        disaggregated = disaggregate(
            pred_records,
            predictions,
            references_lexical,
            params["factor_keys"],
            _factor_metric,
            params["min_sub"],
            known_values=known_values,
        )
    return aggregate, disaggregated


def _assemble_results(
    config: Mapping[str, Any],
    project_root: str | Path,
    pred_records: list[dict[str, Any]],
    aggregate: Mapping[str, float],
    disaggregated: Mapping[str, Any],
    operational: Mapping[str, float],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_run = params["baseline_run"]
    results: dict[str, Any] = {
        "experiment": config["experiment"]["name"],
        "dataset_version": str(config["dataset"]["version"]),
        "aggregate": dict(aggregate),
        "disaggregated": dict(disaggregated),
        "operational": dict(operational),
        "num_examples": len(pred_records),
    }
    if not baseline_run:
        return results

    from .compare import (
        baseline_metrics_path,
        baseline_predictions_path,
        compare_to_baseline,
    )

    family = config["experiment"]["family"]
    bpath = baseline_metrics_path(project_root, family, str(baseline_run))
    if bpath.exists():
        with bpath.open(encoding="utf-8") as handle:
            baseline_full = json.load(handle)
        baseline_full = baseline_full if isinstance(baseline_full, Mapping) else {}
        baseline_version = baseline_full.get("dataset_version")
        if (
            baseline_version is not None
            and str(baseline_version) != results["dataset_version"]
        ):
            results["comparison_vs_baseline"] = {
                "status": (
                    f"dataset del baseline diverso: '{baseline_version}' vs "
                    f"'{results['dataset_version']}' — reference non confrontabili, "
                    f"delta non interpretabile (correggi [evaluation].baseline_run)"
                )
            }
            return results
        if baseline_version is None:
            print(
                "[eval] ⚠️ il metrics.json del baseline non riporta 'dataset_version' "
                "(file precedente al fix): impossibile verificare che baseline e run "
                "corrente usino lo stesso dataset — rigenera la valutazione del baseline."
            )
        baseline_aggregate = baseline_full.get("aggregate")
        baseline_aggregate = (
            baseline_aggregate
            if isinstance(baseline_aggregate, Mapping)
            else baseline_full
        )
        results["comparison_vs_baseline"] = compare_to_baseline(
            aggregate, baseline_aggregate
        )
    else:
        results["comparison_vs_baseline"] = {
            "status": f"baseline metrics non trovate: {bpath}"
        }

    if params.get("significance", True):
        supported = {
            "bleu": "bleu",
            "rougeL": "rougeL",
            "rouge": "rougeL",
            "bertscore": "bertscore_f1",
        }
        sig_metrics = list(
            dict.fromkeys(
                supported[m] for m in params["metrics_list"] if m in supported
            )
        )
        ppath = baseline_predictions_path(project_root, family, str(baseline_run))
        if not sig_metrics:
            results["significance"] = {
                "status": "nessuna metrica richiesta supporta il bootstrap"
            }
        elif not ppath.exists():
            results["significance"] = {
                "status": f"baseline predictions non trovate: {ppath} — significatività non calcolabile"
            }
        else:
            from ..data.preprocessing import clean_report_r2gen
            from .significance import significance_vs_baseline

            with ppath.open(encoding="utf-8") as handle:
                baseline_records = json.load(handle)
            results["significance"] = significance_vs_baseline(
                pred_records,
                baseline_records,
                sig_metrics,
                lexical_normalizer=clean_report_r2gen,
                bertscore_model_type=params["bertscore_model"],
                n_resamples=params["bootstrap_resamples"],
                seed=params["bootstrap_seed"],
            )
    return results


def run_evaluation(
    config: Mapping[str, Any],
    project_root: str | Path,
    adapter_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    capture_operational: bool | None = None,
) -> dict[str, Any]:
    import torch

    from ..config import dataset_root, images_root, model_path
    from ..data import load_records
    from ..training.model import load_processor
    from .generate import generate_predictions
    from .model import load_eval_model
    from .operational import operational_metrics

    params = _eval_params(config)
    ev = config.get("evaluation", {})
    ev = ev if isinstance(ev, Mapping) else {}
    if capture_operational is None:
        capture_operational = bool(ev.get("capture_operational", False))

    ft = config.get("finetuning", {})
    ft = ft if isinstance(ft, Mapping) else {}
    for key in ("max_new_tokens", "repetition_penalty"):
        ft_value = ft.get(key)
        if ft_value is not None and float(ft_value) != float(params[key]):
            print(
                f"[eval] ⚠️ [finetuning].{key}={ft_value} ≠ [evaluation].{key}={params[key]}: "
                f"il checkpoint è stato selezionato con parametri di generazione diversi "
                f"da quelli di questa valutazione. Allineali (o dichiara la differenza)."
            )

    ds_root = dataset_root(config, project_root)
    img_root = images_root(config, project_root)
    mdl_path = model_path(config, project_root)
    params["bertscore_lang"] = _dataset_target_lang(ds_root)
    test_records = load_records(ds_root, "test", img_root)

    model = load_eval_model(config, mdl_path, adapter_dir)
    processor = load_processor(mdl_path, config["finetuning"])

    if capture_operational and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pred_records, skipped = generate_predictions(
        model,
        processor,
        test_records,
        max_new_tokens=params["max_new_tokens"],
        limit=limit,
        repetition_penalty=params["repetition_penalty"],
    )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_output_dir(config, project_root)
    )
    _save_json(output_dir / "predictions.json", pred_records)

    aggregate, disaggregated = _compute_metrics(pred_records, params)

    operational: dict[str, float] = {}
    if capture_operational:
        latencies = [record["latency_s"] for record in pred_records]
        vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        operational = operational_metrics(latencies, vram)

    results = _assemble_results(
        config,
        project_root,
        pred_records,
        aggregate,
        disaggregated,
        operational,
        params,
    )
    results["n_skipped"] = len(skipped)
    if skipped:
        results["skipped"] = skipped

    _save_json(output_dir / "metrics.json", results)
    qualitative = [
        {k: record[k] for k in ("id", "images", "reference", "prediction")}
        for record in pred_records[: min(20, len(pred_records))]
    ]
    _save_json(output_dir / "qualitative.json", qualitative)

    results["output_dir"] = output_dir
    return results


def rescore_from_predictions(
    config: Mapping[str, Any],
    project_root: str | Path,
    predictions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    from ..config import dataset_root
    from .operational import operational_metrics

    params = _eval_params(config)
    params["bertscore_lang"] = _dataset_target_lang(dataset_root(config, project_root))
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_output_dir(config, project_root)
    )
    pred_path = (
        Path(predictions_path)
        if predictions_path is not None
        else output_dir / "predictions.json"
    )
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions.json non trovato: {pred_path}\n"
            f"Esegui prima una valutazione completa (scripts/evaluate/evaluate.py)."
        )

    with pred_path.open(encoding="utf-8") as handle:
        pred_records = json.load(handle)

    aggregate, disaggregated = _compute_metrics(pred_records, params)

    latencies = [
        r["latency_s"]
        for r in pred_records
        if isinstance(r.get("latency_s"), (int, float))
    ]
    operational = operational_metrics(latencies, None) if latencies else {}

    results = _assemble_results(
        config,
        project_root,
        pred_records,
        aggregate,
        disaggregated,
        operational,
        params,
    )
    results["rescored_from"] = str(pred_path)

    _save_json(output_dir / "metrics.json", results)
    results["output_dir"] = output_dir
    return results
