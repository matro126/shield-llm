from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

from .config import Config, Identity, build_config
from .dashboard import LiveDashboard, hms
from .results import now_iso, write_json_atomic
from .runner import BASELINE_ARTIFACTS, archive_previous, find_project_root


def run_baseline(
    script_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ..data import load_records
    from ..evaluation import (
        compute_text_metrics,
        disaggregate,
        operational_metrics,
        sectioned_metrics,
    )
    from ..tracking import (
        dvc_dataset_hash,
        git_metadata,
        metric_provenance,
        model_provenance,
    )
    from .evaluation import (
        DISAGGREGATE_BY,
        MIN_SUBGROUP_SIZE,
        flatten_sectioned,
        format_compliance,
        generate_predictions,
        stampa_formato,
    )
    from .model import load_model_and_processor
    from .runner import _require_gpu, _seed_everything, _vram_peak

    script = Path(script_path).resolve()
    project_root = find_project_root(script.parent)
    identity = Identity.from_path(script)
    cfg: Config = build_config(identity, project_root, overrides)

    if not identity.is_baseline:
        raise ValueError(
            f"{script} non e' uno script di baseline: le baseline stanno in "
            f"training/<lang>/<modello>/baseline/<dataset>/"
        )
    out_dir = project_root / cfg.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    archiviata = archive_previous(
        out_dir, BASELINE_ARTIFACTS, "baseline", marker="metrics.json"
    )
    if archiviata is not None:
        print(f"  archivio  : baseline precedente → {archiviata}")

    print(f"═══ baseline zero-shot: {cfg.experiment}")
    print(f"  modello   : {cfg.base_model}  (bf16, nessun adapter)"
          if not cfg.load_in_4bit else
          f"  modello   : {cfg.base_model}  (4-bit, nessun adapter)")
    print(f"  riferimento per: {identity.model_short} × {identity.code} "
          f"(lora e qlora)")
    print(f"  dataset   : {cfg.dataset_root}  (split={cfg.test_split})")
    print(f"  risultati : {out_dir}")
    _seed_everything(cfg.seed)
    gpus = _require_gpu()

    root = project_root / cfg.dataset_root
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset assente: {root}\n"
            "Costruiscilo con: uv run python -m shield.data.build --all"
        )
    records = load_records(root, cfg.test_split, images_root=root)
    if cfg.baseline_max_samples:
        records = records[: cfg.baseline_max_samples]
    metrics_names = list(cfg.test_metrics)
    print(f"  esempi    : {len(records)}")
    print(f"  metriche  : {', '.join(metrics_names)}")

    model, processor = load_model_and_processor(cfg, project_root, with_adapter=False)

    import torch

    torch.cuda.reset_peak_memory_stats()
    dash = LiveDashboard(title=f"{cfg.experiment} — baseline zero-shot")
    dash.start()
    marks: list[tuple[int, float]] = [(0, time.time())]

    def on_batch(done: int, total: int) -> None:
        marks.append((done, time.time()))
        dash.log_progress("GENERAZIONE", done, total, marks[0][1])

    t_gen = time.time()
    predictions, references = generate_predictions(
        model, processor, records, cfg.gen_batch_size,
        cfg.max_new_tokens, cfg.repetition_penalty, progress=on_batch,
    )
    dash.phase = None
    generation_s = time.time() - t_gen
    print(f"\ngenerazione completata in {hms(generation_s)}")

    latencies = [
        (t_now - t_prev) / max(done - done_prev, 1)
        for (done_prev, t_prev), (done, t_now) in zip(marks, marks[1:])
        for _ in range(done - done_prev)
    ]
    operational = operational_metrics(
        latencies, vram_peak_bytes=float(torch.cuda.max_memory_allocated()) or None
    )

    from ..data.prompts import SEP

    formato = format_compliance(records, predictions, cfg.target)
    stampa_formato(formato, cfg.target)

    print("calcolo delle metriche (la suite completa richiede qualche minuto)…")
    metric_kwargs = {
        "chexbert_translate": cfg.chexbert_translate,
        "chexbert_translator": cfg.chexbert_translator,
        "bertscore_model_type": cfg.bertscore_model,
    }
    sectioned = sectioned_metrics(
        predictions, references, metrics_names, cfg.target,
        metric_fn=compute_text_metrics, chexbert_per_class=True, **metric_kwargs,
    )

    by_factor = disaggregate(
        records, predictions, references,
        factor_keys=list(DISAGGREGATE_BY),
        metric_fn=lambda preds, refs: flatten_sectioned(
            sectioned_metrics(
                preds, refs, metrics_names, cfg.target,
                metric_fn=compute_text_metrics, **metric_kwargs,
            )
        ),
        min_subgroup_size=MIN_SUBGROUP_SIZE,
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "baseline",
        "experiment": cfg.experiment,
        "split": cfg.test_split,
        "adapter": None,
        "identity": {
            "lang": cfg.lang,
            "model": cfg.model_dir,
            "model_short": identity.model_short,
            "load_in_4bit": cfg.load_in_4bit,
            "dataset_code": cfg.dataset_code,
            "views": cfg.views,
            "target": cfg.target,
        },
        "dataset": {"root": cfg.dataset_root, "version": Path(cfg.dataset_root).name},
        "n_examples": len(records),
        "target": cfg.target,
        "metrics": metrics_names,
        "by_section": sectioned,
        "format_compliance": formato,
        "disaggregated": by_factor,
        "operational": {k: round(float(v), 4) for k, v in operational.items()},
        "environment": {"gpus": gpus, **_vram_peak()},
        "generation": {
            "seconds": round(generation_s, 1),
            "gen_batch_size": cfg.gen_batch_size,
            "max_new_tokens": cfg.max_new_tokens,
            "repetition_penalty": cfg.repetition_penalty,
        },
        "provenance": {
            "git": git_metadata(project_root),
            "dvc": {
                "dataset_hash": dvc_dataset_hash(cfg.dataset_root, project_root)
                or "unavailable"
            },
            "model": model_provenance(cfg.as_dict(), project_root),
            "metrics": metric_provenance(cfg.as_dict(), project_root),
        },
        "created_at": now_iso(),
    }
    write_json_atomic(out_dir / "metrics.json", payload)

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "reference", "prediction", "has_sep"])
        for record, reference, prediction in zip(records, references, predictions):
            writer.writerow([
                record["id"], reference, prediction, int(SEP in prediction)
            ])

    if cfg.mlflow_enabled:
        try:
            import mlflow

            from ..tracking import log_numeric_metrics, mlflow_run

            with mlflow_run(
                {
                    "experiment": {"name": f"{cfg.experiment}__baseline"},
                    "dataset": {
                        "root": cfg.dataset_root,
                        "version": Path(cfg.dataset_root).name,
                    },
                    "baseline": {
                        "base_model": cfg.base_model,
                        "load_in_4bit": cfg.load_in_4bit,
                        "split": cfg.test_split,
                        "n_examples": len(records),
                        "metrics": ",".join(metrics_names),
                    },
                    "mlflow": {
                        "tracking_uri": cfg.mlflow_tracking_uri or None,
                        "experiment_name": cfg.mlflow_experiment_name or None,
                    },
                },
                root=project_root,
                run_name=f"{cfg.experiment}__baseline",
                tags={
                    "phase": "baseline",
                    "lang": cfg.lang,
                    "model": cfg.model_dir,
                    "dataset_code": cfg.dataset_code,
                    "target": cfg.target,
                },
            ):
                run = mlflow.active_run()
                payload["mlflow_run_id"] = run.info.run_id if run else None
                for section in ("findings", "impression"):
                    values = sectioned.get(section)
                    if isinstance(values, dict):
                        log_numeric_metrics(
                            {k: v for k, v in values.items()
                             if "chexbert_cls_" not in k},
                            prefix=f"baseline.{section}",
                        )
                log_numeric_metrics(operational, prefix="operational")
                if formato["separator_expected"]:
                    mlflow.log_metric(
                        "baseline.format_compliance", float(formato["ratio"])
                    )
                    mlflow.log_metric(
                        "baseline.format_missing", float(formato["missing"])
                    )
                for name in ("metrics.json", "predictions.csv"):
                    path = out_dir / name
                    if path.is_file():
                        mlflow.log_artifact(str(path), artifact_path="baseline")
            write_json_atomic(out_dir / "metrics.json", payload)
            print(f"tracciata su MLflow come '{cfg.experiment}__baseline'")
        except Exception as exc:
            print(f"MLflow non disponibile ({type(exc).__name__}: {exc}) — "
                  "i risultati su disco sono completi.")

    print("\n" + "=" * 78)
    print(f"BASELINE {cfg.experiment}  ({len(records)} esempi di {cfg.test_split})")
    print(f"  {'metrica':<34}{'findings':>14}{'impression':>14}")
    findings = sectioned.get("findings") or {}
    impression = sectioned.get("impression") or {}
    for key in sorted(set(findings) | set(impression)):
        if key == "num_examples" or "chexbert_cls_" in key:
            continue
        cells = "".join(
            (f"{value:.4f}" if isinstance(value, float) else "—").rjust(14)
            for value in (findings.get(key), impression.get(key))
        )
        print(f"  {key:<34}{cells}")
    if formato["separator_expected"]:
        print(f"\n  formato rispettato: {formato['ratio']:.1%}"
              f"  ({formato['missing']} generazioni senza {SEP})")
    print(f"  tempo generazione : {hms(generation_s)}")
    print(f"  risultati         : {out_dir / 'metrics.json'}")
    return payload


def main_baseline(script_path: str, overrides: dict[str, Any] | None = None) -> int:
    try:
        run_baseline(script_path, overrides)
    except KeyboardInterrupt:
        print("\ninterrotto dall'utente.")
        return 130
    except Exception as exc:
        print(f"\nBASELINE FALLITA: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 0
