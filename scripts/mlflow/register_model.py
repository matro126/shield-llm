from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.config import load_and_validate
from shield.registry import promotion_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registrazione modello con gate vs baseline."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run MLflow da registrare (default: cercato per nome).",
    )
    parser.add_argument(
        "--gate-key", default="bertscore_f1", help="Metrica del gate d'accettazione."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="metrics.json (default: outputs/<family>/<name>/evaluation/metrics.json).",
    )
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_metrics_path(config: dict) -> Path:
    exp = config["experiment"]
    return (
        PROJECT_ROOT
        / "outputs"
        / exp["family"]
        / exp["name"]
        / "evaluation"
        / "metrics.json"
    )


def _find_run_id(client, experiment_name: str | None, run_name: str) -> str | None:
    if not experiment_name:
        return None
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def main() -> int:
    args = parse_args()
    config = load_and_validate(_resolve(args.config))

    registry = config.get("registry")
    if not registry:
        print("[registry] nessuna sezione [registry] nel config: niente da fare.")
        return 0

    metrics_file = _resolve(args.metrics) or _default_metrics_path(config)
    if not metrics_file.exists():
        raise SystemExit(
            f"[registry] metrics.json non trovato: {metrics_file}\n            Esegui prima scripts/evaluate/evaluate.py."
        )
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    action, reason = promotion_decision(
        metrics, registry.get("promote_to", "staging"), args.gate_key
    )
    print(f"[registry] decisione: {action.upper()} — {reason}")
    if action == "skip":
        return 0

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow_cfg = config.get("mlflow", {})
    if mlflow_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = MlflowClient()

    run_id = args.run_id or _find_run_id(
        client, mlflow_cfg.get("experiment_name"), config["experiment"]["name"]
    )
    if run_id is None:
        raise SystemExit("[registry] run non trovato su MLflow: passa --run-id.")

    model_name = registry["model_name"]
    version = mlflow.register_model(f"runs:/{run_id}/checkpoints", model_name).version
    client.set_model_version_tag(model_name, version, "gate", action)
    client.set_model_version_tag(
        model_name, version, "experiment", config["experiment"]["name"]
    )

    if action == "promote":
        stage = registry.get("promote_to", "staging")
        client.set_registered_model_alias(model_name, stage, version)
        client.set_model_version_tag(model_name, version, "lifecycle_stage", stage)
        print(f"[registry] {model_name} v{version} → alias @{stage}")
    else:
        client.set_model_version_tag(
            model_name, version, "lifecycle_stage", "candidate"
        )
        print(
            f"[registry] {model_name} v{version} registrato come 'candidate' (gate non superato, non promosso)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
