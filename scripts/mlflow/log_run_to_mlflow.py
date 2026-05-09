from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.tracking import (  # noqa: E402
    load_toml,
    log_artifact_if_exists,
    log_metrics_file,
    log_trainer_state,
    mlflow_run,
)


def parse_key_value(items: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def tags_from_config(config: dict[str, Any]) -> dict[str, str]:
    run = config.get("run", {})
    values = run.get("tags", []) if isinstance(run, dict) else []
    tags: dict[str, str] = {}
    for index, item in enumerate(values):
        text = str(item)
        if ":" in text:
            key, value = text.split(":", 1)
            tags[key.strip()] = value.strip()
        else:
            tags[f"run.tag.{index}"] = text
    return tags


def default_metrics(config: dict[str, Any]) -> list[Path]:
    evaluation = config.get("evaluation", {})
    if isinstance(evaluation, dict) and evaluation.get("metrics_file"):
        return [PROJECT_ROOT / str(evaluation["metrics_file"])]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log an existing SHIELD training/evaluation run to MLflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/qwen3vl_2b_qlora.toml",
        help="Experiment TOML config to log as reproducibility metadata.",
    )
    parser.add_argument("--run-name", help="Override the run name in the config.")
    parser.add_argument("--tracking-uri", help="Override the MLflow tracking URI.")
    parser.add_argument("--experiment-name", help="Override the MLflow experiment name.")
    parser.add_argument(
        "--metrics",
        type=Path,
        action="append",
        help="JSON metrics file. Can be passed more than once.",
    )
    parser.add_argument(
        "--trainer-state",
        type=Path,
        help="Hugging Face Trainer state JSON with log_history.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        help="Extra file or directory to attach under artifacts/extra.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional checkpoint/adapter directory to log. This can be large.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        help="Additional MLflow tag as KEY=VALUE. Can be passed more than once.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_toml(config_path)

    tags = tags_from_config(config)
    tags.update(parse_key_value(args.tag))

    metrics_files = [resolve(path) for path in (args.metrics or default_metrics(config))]
    trainer_state = resolve(args.trainer_state) if args.trainer_state else None
    artifacts = [resolve(path) for path in (args.artifact or [])]
    checkpoint = resolve(args.checkpoint) if args.checkpoint else None

    with mlflow_run(
        config=config,
        root=PROJECT_ROOT,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        tags=tags,
    ):
        log_artifact_if_exists(config_path, artifact_path="config")

        for metrics_file in metrics_files:
            log_metrics_file(metrics_file)

        if trainer_state:
            log_trainer_state(trainer_state)

        for artifact in artifacts:
            log_artifact_if_exists(artifact, artifact_path="extra", allow_dir=True)

        if checkpoint:
            log_artifact_if_exists(checkpoint, artifact_path="checkpoints", allow_dir=True)

    print("Logged run to MLflow.")


if __name__ == "__main__":
    main()
