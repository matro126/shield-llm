#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shield.training.config import (  # noqa: E402
    TRAINING_MODES,
    Identity,
    available_metric_keys,
    build_entrypoint_config,
    resolve_metric_selection,
)
from shield.training.dashboard import LiveDashboard, hms  # noqa: E402

from shield.training.evaluation import (  # noqa: E402
    DISAGGREGATE_BY,
    MIN_SUBGROUP_SIZE,
)


def discover(training_root: Path) -> dict[str, Identity]:
    return {
        identity.name: identity
        for p in sorted(training_root.glob("*/*/*/*"))
        if p.is_dir()
        for identity in [Identity.from_path(p)]
        if identity.mode in TRAINING_MODES
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", help="nome dell'esperimento, es. it_2B_qlora_FL-FI")
    parser.add_argument("--training-root", type=Path, default=ROOT / "training")
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--adapter", type=Path,
                        help="adapter da valutare (default: results/best_adapter)")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="default: test_metrics da training/defaults.toml")
    parser.add_argument("--max-samples", type=int,
                        help="valuta solo i primi N esempi (per una prova rapida)")
    parser.add_argument("--no-disaggregate", action="store_true")
    parser.add_argument(
        "--disaggregate-metrics", nargs="+", default=None,
        help="metriche per la disaggregazione (default: le stesse di --metrics). "
             "Restringerle conviene: la suite semantica gira una volta per "
             "sottogruppo, quindi decine di volte.",
    )
    parser.add_argument("--no-profile", action="store_true",
                        help="salta le metriche infrastrutturali di §3.8.3")
    parser.add_argument("--profile-samples", type=int, default=60,
                        help="referti per la latenza a richiesta singola (batch 1)")
    parser.add_argument("--profile-batches", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16],
                        help="ampiezze di batch per la curva carico/latenza")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="non tracciare questa valutazione su MLflow")
    parser.add_argument("--list", action="store_true",
                        help="elenca gli esperimenti valutabili ed esci")
    args = parser.parse_args(argv)

    experiments = discover(args.training_root)

    if args.list:
        print(f"{len(experiments)} esperimenti:\n")
        print(f"  {'esperimento':<28}{'best adapter':<16}{'gia valutato':<14}best val")
        for name, identity in experiments.items():
            results = ROOT / identity.relpath / "results"
            has_adapter = (results / "best_adapter").is_dir()
            evaluated = (results / "test" / "metrics.json").is_file()
            best = "—"
            canonical = results / "results.json"
            if canonical.is_file():
                try:
                    data = json.loads(canonical.read_text(encoding="utf-8")).get("best")
                    if data and isinstance(data.get("value"), (int, float)):
                        best = f"{data['metric']}={data['value']:.4f}"
                except json.JSONDecodeError:
                    pass
            print(f"  {name:<28}{('si' if has_adapter else 'assente'):<16}"
                  f"{('si' if evaluated else 'no'):<14}{best}")
        return 0

    if not args.experiment:
        parser.error("indica --experiment NOME (oppure --list per vederli)")
    if args.experiment not in experiments:
        print(f"Esperimento sconosciuto: {args.experiment}", file=sys.stderr)
        print(f"disponibili: {', '.join(sorted(experiments))}", file=sys.stderr)
        return 1

    identity = experiments[args.experiment]
    cfg = build_entrypoint_config(identity, ROOT)
    metric_names, metric_names_csv = resolve_metric_selection(
        args.metrics,
        cfg.test_metrics,
    )
    results = ROOT / cfg.results_dir
    adapter = args.adapter or (results / "best_adapter")
    if not adapter.is_dir():
        print(f"Nessun adapter da valutare in {adapter}", file=sys.stderr)
        print("esegui prima il training:", file=sys.stderr)
        print(f"  python {identity.script_relpath}", file=sys.stderr)
        return 1

    from shield.data import load_records
    from shield.evaluation import (
        compute_text_metrics,
        disaggregate,
        operational_metrics,
        profila,
        riassunto,
        sectioned_metrics,
    )
    from shield.training.evaluation import (
        flatten_sectioned,
        format_compliance,
        generate_predictions,
        stampa_formato,
        validate_sectioned_references,
    )
    from shield.training.model import load_model_and_processor

    out_dir = results / args.split if args.split != "test" else results / "test"

    from shield.training.runner import archive_previous  # noqa: PLC0415

    archiviata = archive_previous(
        results, (out_dir.name,), f"eval-{args.split}", marker=f"{out_dir.name}/metrics.json"
    )
    if archiviata is not None:
        print(f"  archivio  : valutazione precedente → "
              f"{archiviata.relative_to(ROOT)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"═══ valutazione {args.split}: {cfg.experiment}")
    print(f"  adapter : {adapter.relative_to(ROOT)}")
    best_info = adapter / "best_info.json"
    if best_info.is_file():
        print(f"  best    : {best_info.read_text(encoding='utf-8').strip()}")

    root = ROOT / cfg.dataset_root
    records = load_records(root, args.split, images_root=root)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"  esempi  : {len(records)}")
    print(f"  metriche: {', '.join(metric_names)}")

    model, processor = load_model_and_processor(cfg, ROOT)
    model.load_adapter(str(adapter), adapter_name="best")
    model.set_adapter("best")

    import torch as _torch
    vram_modello = (_torch.cuda.memory_allocated() / 1e9
                    if _torch.cuda.is_available() else None)

    dash = LiveDashboard(title=f"{cfg.experiment} — {args.split}")
    dash.start()

    marks: list[tuple[int, float]] = [(0, time.time())]

    def on_batch(done: int, total: int) -> None:
        marks.append((done, time.time()))
        dash.log_progress("GENERAZIONE", done, total, marks[0][1])

    import torch

    torch.cuda.reset_peak_memory_stats()
    t_gen = time.time()
    predictions, references = generate_predictions(
        model,
        processor,
        records,
        cfg.gen_batch_size,
        cfg.max_new_tokens,
        cfg.repetition_penalty,
        progress=on_batch,
    )
    validate_sectioned_references(records, references, cfg.target)
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
    print(f"  latenza per referto: p50 {operational['latency_p50_s']:.2f}s  "
          f"p95 {operational['latency_p95_s']:.2f}s  "
          f"throughput {operational['throughput_req_s']:.2f}/s  "
          f"VRAM picco {operational.get('vram_peak_gb', 0):.1f} GB")

    formato = format_compliance(records, predictions, cfg.target)
    stampa_formato(formato, cfg.target)

    print("calcolo delle metriche (la suite completa richiede qualche minuto)…")
    metric_kwargs = {
        "chexbert_translate": cfg.chexbert_translate,
        "chexbert_translator": cfg.chexbert_translator,
        "bertscore_model_type": cfg.bertscore_model,
    }
    sectioned = sectioned_metrics(
        predictions, references, metric_names, cfg.target,
        metric_fn=compute_text_metrics, chexbert_per_class=True, **metric_kwargs,
    )
    profilo = None
    if not args.no_profile:
        print("\nmetriche infrastrutturali (§3.8.3): latenza a batch 1 e curva di carico…")

        t_prof = time.time()

        def on_profile(fase: str, fatti: int, totale: int) -> None:
            barra = "#" * int(28 * fatti / max(totale, 1))
            print(f"\r  {fase:<30} [{barra:<28}] {fatti}/{totale}  "
                  f"{time.time() - t_prof:.0f}s", end="", flush=True)
            if fatti >= totale:
                print()

        profilo = profila(
            model, processor, records,
            max_new_tokens=cfg.max_new_tokens,
            repetition_penalty=cfg.repetition_penalty,
            n_singole=args.profile_samples,
            batch_sizes=tuple(args.profile_batches),
            vram_modello_gb=vram_modello,
            progress=on_profile,
        )
        print(riassunto(profilo))

    payload = {
        "experiment": cfg.experiment,
        "split": args.split,
        "adapter": str(adapter.relative_to(ROOT)),
        "n_examples": len(records),
        "target": cfg.target,
        "metrics": metric_names,
        "by_section": sectioned,
        "format_compliance": formato,
        "operational": {k: round(float(v), 4) for k, v in operational.items()},
        "generation_s": round(generation_s, 1),
        "infrastructure": profilo,
    }

    print(f"\n=== METRICHE {args.split.upper()} — per sezione, senza medie ===")
    for section in ("findings", "impression"):
        values = sectioned.get(section)
        if not isinstance(values, dict):
            continue
        print(f"\n  --- {section} ---")
        for key, value in values.items():
            print(f"    {key:<34}{value:.4f}" if isinstance(value, float)
                  else f"    {key:<34}{value}")

    if not args.no_disaggregate:
        disaggregate_metrics = args.disaggregate_metrics or metric_names
        print(f"\ndisaggregazione per fattore ({', '.join(disaggregate_metrics)}, "
              "per sezione come le metriche globali)…")

        def subgroup_metrics(preds: list[str], refs: list[str]) -> dict[str, float]:
            return flatten_sectioned(
                sectioned_metrics(
                    preds, refs, list(disaggregate_metrics), cfg.target,
                    metric_fn=compute_text_metrics, **metric_kwargs,
                )
            )

        by_factor = disaggregate(
            records, predictions, references,
            factor_keys=list(DISAGGREGATE_BY),
            metric_fn=subgroup_metrics,
            min_subgroup_size=MIN_SUBGROUP_SIZE,
        )
        payload["disaggregated"] = by_factor
        payload["disaggregate_metrics"] = disaggregate_metrics
        (out_dir / "disaggregated.json").write_text(
            json.dumps(by_factor, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        available = available_metric_keys(disaggregate_metrics, cfg.target)
        shown = [
            key for key in (
                "findings.rougeL", "findings.bleu", "findings.bertscore_f1",
                "findings.clinicalbert_f1", "findings.chexbert_f1_micro_top5",
            )
            if key in available
        ][:3]
        for factor, groups in by_factor.items():
            print(f"\n### {factor}")
            for value, group_metrics in groups.items():
                if group_metrics.get("status") == "not_estimable":
                    print(f"  {value:<28}n={group_metrics['n']:<5}(sotto soglia "
                          f"{MIN_SUBGROUP_SIZE}: non stimabile)")
                else:
                    cells = "  ".join(
                        f"{key}={group_metrics[key]:.3f}"
                        for key in shown if key in group_metrics
                    )
                    print(f"  {value:<28}n={group_metrics['n']:<5}{cells}")

    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mancanti = set(formato["missing_ids"])
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "reference", "prediction", "has_sep"])
        for record, reference, prediction in zip(records, references, predictions):
            writer.writerow([
                record["id"], reference, prediction,
                int(str(record["id"]) not in mancanti),
            ])

    if cfg.mlflow_enabled and not args.no_mlflow:
        try:
            import mlflow

            from shield.tracking import log_numeric_metrics, mlflow_run

            tracking_config = {
                "experiment": {"name": f"{cfg.experiment}__{args.split}"},
                "dataset": {
                    "root": cfg.dataset_root,
                    "version": Path(cfg.dataset_root).name,
                },
                "evaluation": {
                    "split": args.split,
                    "adapter": str(adapter.relative_to(ROOT)),
                    "n_examples": len(records),
                    "metrics": metric_names_csv,
                },
                "mlflow": {
                    "tracking_uri": cfg.mlflow_tracking_uri or None,
                    "experiment_name": cfg.mlflow_experiment_name or None,
                },
            }
            with mlflow_run(
                tracking_config,
                root=ROOT,
                run_name=f"{cfg.experiment}__{args.split}",
                tags={
                    "phase": "evaluation",
                    "split": args.split,
                    "training_run": cfg.experiment,
                    "lang": cfg.lang,
                    "model": cfg.model_dir,
                    "mode": cfg.mode,
                    "dataset_code": cfg.dataset_code,
                    "target": cfg.target,
                },
            ):
                run = mlflow.active_run()
                payload["mlflow_run_id"] = run.info.run_id if run else None
                log_numeric_metrics(operational, prefix="operational")
                if formato["separator_expected"]:
                    mlflow.log_metric(
                        f"{args.split}.format_compliance", float(formato["ratio"])
                    )
                    mlflow.log_metric(
                        f"{args.split}.format_missing", float(formato["missing"])
                    )
                if profilo and not profilo.get("errore"):
                    log_numeric_metrics(profilo["richiesta_singola"],
                                        prefix="infra.single")
                    log_numeric_metrics(profilo["vram"], prefix="infra.vram")
                    for riga in profilo["carico"]:
                        if "errore" in riga:
                            continue
                        log_numeric_metrics(
                            {k: v for k, v in riga.items() if k != "batch_size"},
                            prefix=f"infra.batch{riga['batch_size']}")
                for section in ("findings", "impression"):
                    values = sectioned.get(section)
                    if isinstance(values, dict):
                        log_numeric_metrics(
                            {k: v for k, v in values.items()
                             if "chexbert_cls_" not in k},
                            prefix=f"{args.split}.{section}",
                        )
                for name in ("metrics.json", "disaggregated.json", "predictions.csv"):
                    path = out_dir / name
                    if path.is_file():
                        mlflow.log_artifact(str(path), artifact_path=f"evaluation/{args.split}")
            (out_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\ntracciato su MLflow come run '{cfg.experiment}__{args.split}'")
        except Exception as exc:
            print(f"\nMLflow non disponibile ({type(exc).__name__}: {exc}) — "
                  "i risultati su disco sono comunque completi.")

    print("\nsalvati:")
    for name in ("metrics.json", "disaggregated.json", "predictions.csv"):
        path = out_dir / name
        if path.is_file():
            print(f"  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
