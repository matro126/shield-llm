#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shield.training.config import (  # noqa: E402
    TRAINING_MODES,
    Identity,
    build_config,
)

OK, GAP, WARN, INFO = "OK", "LACUNA", "ATTENZIONE", "INFO"

LEXICAL = ("bleu", "rougeL")
SEMANTIC_CLINICAL = ("bertscore", "clinicalbert", "chexbert")
OPERATIONAL = ("latency_p50_s", "latency_p95_s", "throughput_req_s", "vram_peak_gb")


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, req: str, what: str, status: str, detail: str) -> None:
        self.rows.append((req, what, status, detail))

    def render(self) -> int:
        width_what = max(len(r[1]) for r in self.rows)
        print(f"\n  {'req':<10}{'requisito':<{width_what + 2}}{'esito':<14}dettaglio")
        print("  " + "─" * (10 + width_what + 2 + 14 + 30))
        for req, what, status, detail in self.rows:
            mark = {OK: "✓", GAP: "✗", WARN: "!", INFO: "·"}[status]
            print(f"  {req:<10}{what:<{width_what + 2}}{mark} {status:<12}{detail}")
        gaps = [r for r in self.rows if r[2] == GAP]
        warns = [r for r in self.rows if r[2] == WARN]
        print()
        if gaps:
            print(f"  {len(gaps)} LACUNE da colmare:")
            for req, what, _, detail in gaps:
                print(f"    {req} {what}: {detail}")
        if warns:
            print(f"  {len(warns)} avvertenze (non bloccanti):")
            for req, what, _, detail in warns:
                print(f"    {req} {what}: {detail}")
        if not gaps and not warns:
            print("  Tutti i requisiti MLflow di REQUISITI.md sono coperti.")
        return 1 if gaps else 0


def tracking_config_of(cfg: Any) -> dict[str, Any]:
    return {
        "experiment": {"name": cfg.experiment},
        "dataset": {"root": cfg.dataset_root, "version": Path(cfg.dataset_root).name},
        "model": {"base_model": cfg.base_model, "mode": cfg.mode},
        "training": cfg.as_dict(),
        "mlflow": {
            "tracking_uri": cfg.mlflow_tracking_uri or None,
            "experiment_name": cfg.mlflow_experiment_name or None,
        },
    }


def check_static(cfg: Any, report: Report) -> None:
    from shield.tracking import (
        dataset_provenance,
        dvc_dataset_hash,
        flatten_mapping,
        git_metadata,
    )

    config = tracking_config_of(cfg)

    logged = set(flatten_mapping(config))
    declared = set(cfg.as_dict())
    missing = sorted(
        name for name in declared
        if not any(key.endswith(name) for key in logged)
    )
    report.add(
        "R1", "iperparametri completi nei params",
        OK if not missing else GAP,
        f"{len(declared)} campi di Config → {len(logged)} params"
        + (f"; mancanti: {missing}" if missing else ""),
    )

    git = git_metadata(ROOT)
    commit = git.get("git.commit")
    report.add(
        "R2", "commit git del codice",
        OK if commit else GAP,
        f"git.commit={str(commit)[:12]}…  branch={git.get('git.branch')}  "
        f"dirty={git.get('git.is_dirty')}"
        + ("  ← il run sarebbe tracciato con codice non committato"
           if git.get("git.is_dirty") else ""),
    )
    if git.get("is_dirty") or git.get("git.is_dirty"):
        report.add(
            "R2", "working tree pulito al momento del run", WARN,
            f"{git.get('git.changed_files')} file modificati: il commit registrato "
            "non descrive esattamente il codice eseguito",
        )

    provenance = dataset_provenance(config, ROOT)
    dataset_hash = dvc_dataset_hash(cfg.dataset_root, ROOT)
    report.add(
        "R4", "versione del dataset",
        OK if provenance.get("dataset.version") else GAP,
        f"dataset.version={provenance.get('dataset.version')}",
    )
    if dataset_hash:
        report.add("R3", "hash DVC del dataset", OK, f"dvc.dataset_hash={dataset_hash[:16]}…")
    else:
        report.add(
            "R3", "hash DVC del dataset", GAP,
            "dvc.lock assente o dataset non tracciato: il run registrerebbe "
            "'unavailable'. Serve `dvc init` + `dvc repro` (o `dvc add`) perche' "
            "la catena git↔DVC↔MLflow sia completa.",
        )

    tracked = [f"val.{m}" for m in cfg.eval_metrics]
    missing_lex = [m for m in LEXICAL if m not in cfg.eval_metrics]
    report.add(
        "R5", "BLEU/ROUGE-L per ogni run di training",
        OK if not missing_lex else GAP,
        f"loggate a ogni valutazione: train.loss, val.loss, {', '.join(tracked)}"
        + (f"; mancanti: {missing_lex}" if missing_lex else ""),
    )

    test_metrics = list(cfg.test_metrics)
    posthoc = "chexbert" not in test_metrics
    missing_sem = [
        m for m in SEMANTIC_CLINICAL if m not in test_metrics and not posthoc
    ]
    detail = f"test_metrics = {', '.join(test_metrics)}"
    if posthoc:
        detail += ("; CheXbert e ClinicalBERT sono modelli inglesi: su referti "
                   "italiani si calcolano a posteriori sulle predizioni tradotte "
                   "(scripts/evaluate/metrics_posthoc.py --metrics chexbert)")
    report.add(
        "R6", "BERTScore/ClinicalBERT/CheXbert sul test",
        OK if not missing_sem else GAP, detail,
    )

    source = (ROOT / "training" / "evaluate_test.py").read_text(encoding="utf-8")
    has_operational = "operational_metrics(" in source and 'prefix="operational"' in source
    report.add(
        "R7", "metriche operative (latenza, VRAM)",
        OK if has_operational else GAP,
        "evaluate_test.py logga operational." + ", operational.".join(OPERATIONAL)
        if has_operational else "nessuna metrica operativa loggata",
    )

    runner_source = (ROOT / "src" / "shield" / "training" / "runner.py").read_text(
        encoding="utf-8"
    )
    artifacts = [
        name for name in ("results.json", "train_history.csv", "val_history.csv",
                          "val_predictions_best.csv", "best_adapter")
        if name in runner_source
    ]
    report.add(
        "R8", "artefatti del run",
        OK if len(artifacts) >= 4 else GAP,
        "training/: " + ", ".join(artifacts),
    )

    report.add("R9", "run id e timestamp", INFO,
               "forniti da MLflow; il run_id viene ricopiato in results.json "
               "(provenance.mlflow.run_id)")
    report.add("R10", "Model Registry", INFO,
               "disponibile ma NON automatico: la promozione a staging/production "
               "e' una decisione deliberata dopo le metriche di test")


def check_live(cfg: Any, report: Report) -> None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        report.add("LIVE", "round-trip su MLflow", WARN,
                   "mlflow non installato in questo ambiente: salto il test live")
        return

    from shield.tracking import log_numeric_metrics, mlflow_run

    store = Path(tempfile.mkdtemp(prefix="mlflow-check-"))
    uri = f"file://{store}"
    config = tracking_config_of(cfg)
    config["mlflow"] = {"tracking_uri": uri, "experiment_name": "check-mlflow"}

    with mlflow_run(config, root=ROOT, run_name="check-run",
                    tags={"phase": "check", "lang": cfg.lang}) as _:
        run_id = mlflow.active_run().info.run_id
        mlflow.log_metric("train.loss", 2.5, step=10)
        mlflow.log_metric("val.loss", 2.4, step=129)
        for metric in LEXICAL:
            mlflow.log_metric(f"val.{metric}", 0.25, step=129)
        log_numeric_metrics(
            {m: 0.5 for m in SEMANTIC_CLINICAL}, prefix="test"
        )
        log_numeric_metrics({m: 1.0 for m in OPERATIONAL}, prefix="operational")
        artifact = store / "results.json"
        artifact.write_text(json.dumps({"probe": True}), encoding="utf-8")
        mlflow.log_artifact(str(artifact), artifact_path="training")

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags

    declared = set(cfg.as_dict())
    seen = {key.rsplit(".", 1)[-1] for key in params}
    missing_params = sorted(declared - seen)
    report.add("LIVE R1", "params riletti dal server",
               OK if not missing_params else GAP,
               f"{len(params)} params registrati"
               + (f"; mancanti: {missing_params}" if missing_params else ""))

    for req, tag in (("LIVE R2", "git.commit"),
                     ("LIVE R3", "dvc.dataset_hash"),
                     ("LIVE R4", "dataset.version")):
        value = tags.get(tag)
        unavailable = value in (None, "", "unavailable")
        report.add(req, f"tag {tag} riletto",
                   GAP if unavailable else OK,
                   f"{tag}={value}")

    expected_metrics = (
        ["train.loss", "val.loss"]
        + [f"val.{m}" for m in LEXICAL]
        + [f"test.{m}" for m in SEMANTIC_CLINICAL]
        + [f"operational.{m}" for m in OPERATIONAL]
    )
    missing_metrics = [m for m in expected_metrics if m not in metrics]
    report.add("LIVE R5-R7", "metriche rilette dal server",
               OK if not missing_metrics else GAP,
               f"{len(metrics)} metriche registrate"
               + (f"; mancanti: {missing_metrics}" if missing_metrics else ""))

    artifacts = [a.path for a in client.list_artifacts(run_id, "training")]
    report.add("LIVE R8", "artefatti riletti dal server",
               OK if artifacts else GAP, ", ".join(artifacts) or "nessuno")
    report.add("LIVE R9", "run id e timestamp", OK,
               f"run_id={run_id[:12]}…  start={run.info.start_time}")
    report.add("LIVE", "store temporaneo", INFO, str(store))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None,
                        help="esperimento di riferimento (default: il primo trovato)")
    parser.add_argument("--live", action="store_true",
                        help="esegue anche un run reale su uno store MLflow locale")
    args = parser.parse_args(argv)

    identities = {
        identity.name: identity
        for p in sorted((ROOT / "training").glob("*/*/*/*"))
        if p.is_dir()
        for identity in [Identity.from_path(p)]
        if identity.mode in TRAINING_MODES
    }
    if not identities:
        print("Nessun esperimento sotto training/", file=sys.stderr)
        return 1
    name = args.experiment or sorted(identities)[0]
    if name not in identities:
        print(f"Esperimento sconosciuto: {name}", file=sys.stderr)
        return 1

    cfg = build_config(identities[name], ROOT, None)
    print(f"═══ check tracking MLflow  (esperimento di riferimento: {name})")
    print(f"  dataset: {cfg.dataset_root}")
    print(f"  mlflow : {cfg.mlflow_tracking_uri or 'default (env o http://127.0.0.1:5000)'}")

    report = Report()
    check_static(cfg, report)
    if args.live:
        print("\n  eseguo un run reale su store locale…")
        check_live(cfg, report)
    else:
        report.add("LIVE", "round-trip su MLflow", INFO,
                   "non eseguito: aggiungi --live per provare il giro completo")
    return report.render()


if __name__ == "__main__":
    raise SystemExit(main())
