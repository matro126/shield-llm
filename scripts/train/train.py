from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.config import load_and_validate, method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning SHIELD config-driven.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path del config.toml dell'esperimento.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override della cartella di output locale.",
    )
    parser.add_argument(
        "--no-sanity-check",
        action="store_true",
        help="Salta il controllo di label masking del collator.",
    )
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    config = load_and_validate(config_path)

    if method(config) == "none":
        print("[train] method='none' (zero-shot baseline): niente da addestrare.")
        print(
            "        Valuta il baseline con: scripts/evaluate/evaluate.py --config",
            args.config,
        )
        return 0

    from shield.tracking import (
        log_artifact_if_exists,
        log_params,
        log_trainer_state,
        mlflow_run,
    )
    from shield.training.pipeline import run_training

    with mlflow_run(config, root=PROJECT_ROOT):
        result = run_training(
            config,
            PROJECT_ROOT,
            output_dir=_resolve(args.output_dir),
            sanity_check=not args.no_sanity_check,
        )
        log_params(result["trainable_summary"], prefix="model")
        log_trainer_state(str(result["trainer_state"]))
        log_artifact_if_exists(
            str(result["final_dir"]), artifact_path="checkpoints", allow_dir=True
        )
        log_artifact_if_exists(str(config_path), artifact_path="config")

    print(f"[train] completato. Adapter/best model in: {result['final_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
