from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.config import load_and_validate, method  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valutazione SHIELD config-driven.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Cartella adapter PEFT (default: outputs/<family>/<name>/final).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Valuta solo i primi N esempi (debug)."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Non loggare su MLflow: calcola e salva solo i json.",
    )
    parser.add_argument(
        "--no-operational",
        action="store_true",
        help="Salta le metriche operative (VRAM/latency).",
    )
    parser.add_argument(
        "--from-predictions",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help="Ri-calcola le metriche dai predictions.json salvati, SENZA rigenerare (no "
        "inference sul VLM). Path opzionale; default: outputs/<family>/<name>/evaluation/predictions.json.",
    )
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_adapter(args: argparse.Namespace, config: dict) -> Path | None:
    if args.adapter is not None:
        return _resolve(args.adapter)
    if method(config) == "none":
        return None
    exp = config["experiment"]
    default = PROJECT_ROOT / "outputs" / exp["family"] / exp["name"] / "final"
    if default.exists():
        return default
    raise SystemExit(
        f"[eval] adapter non trovato in {default}.\n"
        f"       Allena prima (scripts/train/train.py) oppure passa --adapter <path>."
    )


def _print_summary(results: dict) -> None:
    print("\n=== Metriche aggregate ===")
    for key, value in results["aggregate"].items():
        if isinstance(value, (int, float)):
            print(f"  {key:24s} {value:.4f}")
    if results.get("n_skipped"):
        print(
            f"\n[eval] ⚠️ {results['n_skipped']} esempi saltati (vedi 'skipped' in metrics.json): test set effettivo ridotto."
        )
    comparison = results.get("comparison_vs_baseline")
    if isinstance(comparison, dict) and "status" not in comparison:
        print("\n=== Confronto vs baseline (delta) ===")
        for key, entry in comparison.items():
            print(
                f"  {key:24s} {entry['baseline']:.4f} → {entry['current']:.4f}  (Δ {entry['delta']:+.4f})"
            )
    elif isinstance(comparison, dict):
        print(f"\n[eval] {comparison['status']}")
    significance = results.get("significance")
    if (
        isinstance(significance, dict)
        and "status" not in significance
        and significance.get("metrics")
    ):
        print(
            f"\n=== Significatività vs baseline (paired bootstrap, n={significance.get('n_common')}, resamples={significance.get('n_resamples')}) ==="
        )
        for key, entry in significance["metrics"].items():
            if "significant" not in entry:
                continue
            verdict = "SIGNIFICATIVO" if entry["significant"] else "non significativo"
            print(
                f"  {key:24s} Δ {entry['delta']:+.4f}  IC95 [{entry['ci95_low']:+.4f}, {entry['ci95_high']:+.4f}]  "
                f"p={entry['p_value']:.4f}  → {verdict}"
            )
    elif isinstance(significance, dict) and "status" in significance:
        print(f"\n[eval] significatività: {significance['status']}")


def _log_to_mlflow(config: dict, results: dict, config_path: Path) -> None:
    from shield.tracking import log_artifact_if_exists, log_numeric_metrics, mlflow_run

    with mlflow_run(config, root=PROJECT_ROOT):
        log_numeric_metrics(results["aggregate"], prefix="eval")
        if results.get("disaggregated"):
            log_numeric_metrics(results["disaggregated"], prefix="eval.by")
        if results.get("operational"):
            log_numeric_metrics(results["operational"], prefix="eval.op")
        comparison = results.get("comparison_vs_baseline")
        if isinstance(comparison, dict) and "status" not in comparison:
            log_numeric_metrics(comparison, prefix="eval.cmp")
        significance = results.get("significance")
        if isinstance(significance, dict) and isinstance(
            significance.get("metrics"), dict
        ):
            log_numeric_metrics(significance, prefix="eval.sig")
            flags = {
                name: (1.0 if entry.get("significant") else 0.0)
                for name, entry in significance["metrics"].items()
                if isinstance(entry, dict) and "significant" in entry
            }
            if flags:
                log_numeric_metrics(flags, prefix="eval.sig.significant")
        log_artifact_if_exists(
            str(results["output_dir"]), artifact_path="evaluation", allow_dir=True
        )
        log_artifact_if_exists(str(config_path), artifact_path="config")


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    config = load_and_validate(config_path)

    if args.from_predictions is not None:
        from shield.evaluation.pipeline import rescore_from_predictions

        pred_path = (
            None
            if args.from_predictions == "__default__"
            else _resolve(Path(args.from_predictions))
        )
        results = rescore_from_predictions(
            config,
            PROJECT_ROOT,
            predictions_path=pred_path,
            output_dir=_resolve(args.output_dir),
        )
        print(f"[eval] ri-calcolato da: {results.get('rescored_from')}")
    else:
        adapter_dir = _resolve_adapter(args, config)
        from shield.evaluation.pipeline import run_evaluation

        results = run_evaluation(
            config,
            PROJECT_ROOT,
            adapter_dir=adapter_dir,
            output_dir=_resolve(args.output_dir),
            limit=args.limit,
            capture_operational=False if args.no_operational else None,
        )

    _print_summary(results)
    print(f"\n[eval] risultati salvati in: {results['output_dir']}")

    if not args.no_mlflow:
        _log_to_mlflow(config, results, config_path)
        print("[eval] loggato su MLflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
