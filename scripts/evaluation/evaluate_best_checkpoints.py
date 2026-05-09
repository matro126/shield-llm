from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DEFAULT_CONFIGS = [
    "configs/experiments/qwen3vl_2b_qlora_early_stopping.toml",
    "configs/experiments/qwen3vl_2b_lora_early_stopping.toml",
]


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_meta"] = {"config_path": str(config_path), "project_root": str(PROJECT_ROOT)}
    return config


def resolve_path(config: dict[str, Any], section: str, key: str) -> Path:
    value = config[section][key]
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class CheckpointSelection:
    run_name: str
    config_path: Path
    checkpoints_dir: Path
    checkpoint_path: Path | None
    checkpoint_source: str
    trainer_state_path: Path | None
    best_model_checkpoint: str | None
    best_metric: float | None
    best_metric_name: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best checkpoint for one or more SHIELD experiment configs "
            "on the shared test set."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Experiment config to evaluate. Can be passed more than once. "
            "Defaults to QLoRA and LoRA early-stopping configs."
        ),
    )
    parser.add_argument(
        "--all-finetuning-configs",
        action="store_true",
        help="Evaluate every configs/experiments/*early_stopping.toml config.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also evaluate the zero-shot baseline config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/best_checkpoints",
        help="Directory for aggregate and per-run best-checkpoint metrics.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override max_new_tokens from each config.",
    )
    parser.add_argument(
        "--qualitative-examples",
        type=int,
        help="Override qualitative_examples from each config.",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not save full per-example prediction files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N test examples. Useful for smoke tests.",
    )
    parser.add_argument(
        "--load-in-4bit",
        choices=["auto", "true", "false"],
        default="auto",
        help="How to load the base model for evaluation.",
    )
    parser.add_argument(
        "--trust-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When checkpoints_dir/final exists, evaluate it. This is correct for "
            "Trainer runs with load_best_model_at_end=True followed by save_model()."
        ),
    )
    return parser.parse_args()


def resolve_config_paths(args: argparse.Namespace) -> list[Path]:
    if args.all_finetuning_configs:
        configs = sorted(PROJECT_ROOT.glob("configs/experiments/*early_stopping.toml"))
    elif args.config:
        configs = [Path(item) for item in args.config]
    else:
        configs = [Path(item) for item in DEFAULT_CONFIGS]

    if args.include_baseline:
        configs.append(Path("configs/experiments/qwen3vl_2b_zero_shot_baseline.toml"))

    resolved: list[Path] = []
    for config in configs:
        path = config if config.is_absolute() else PROJECT_ROOT / config
        if not path.exists():
            raise FileNotFoundError(f"Config non trovata: {path}")
        if path not in resolved:
            resolved.append(path)
    return resolved


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def best_eval_from_history(state: dict[str, Any], metric_name: str) -> tuple[float | None, int | None]:
    values: list[tuple[float, int]] = []
    for row in state.get("log_history", []):
        if metric_name not in row:
            continue
        value = row[metric_name]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append((float(value), int(row.get("step", -1))))
    if not values:
        return None, None
    best_value, best_step = min(values, key=lambda item: item[0])
    return best_value, best_step


def candidate_has_model_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
        "config.json",
    }
    return any((path / name).exists() for name in names)


def select_checkpoint(config: dict[str, Any], config_path: Path, trust_final: bool) -> CheckpointSelection:
    run_name = config["run"]["name"]
    method = config.get("finetuning", config.get("inference", {})).get("method", "")
    checkpoints_dir = resolve_path(config, "paths", "checkpoints_dir")

    if method == "zero-shot":
        return CheckpointSelection(
            run_name=run_name,
            config_path=config_path,
            checkpoints_dir=checkpoints_dir,
            checkpoint_path=None,
            checkpoint_source="zero_shot_base_model",
            trainer_state_path=None,
            best_model_checkpoint=None,
            best_metric=None,
            best_metric_name=None,
        )

    root_state_path = checkpoints_dir / "trainer_state.json"
    root_state = load_json_if_exists(root_state_path) or {}
    metric_name = config.get("finetuning", {}).get("metric_for_best_model", "eval_loss")
    if not metric_name.startswith("eval_"):
        metric_name = f"eval_{metric_name}"

    best_model_checkpoint = root_state.get("best_model_checkpoint")
    best_metric = root_state.get("best_metric")
    if not isinstance(best_metric, (int, float)):
        best_metric, best_step = best_eval_from_history(root_state, metric_name)
    else:
        best_step = None

    final_dir = checkpoints_dir / "final"
    if trust_final and candidate_has_model_files(final_dir):
        return CheckpointSelection(
            run_name=run_name,
            config_path=config_path,
            checkpoints_dir=checkpoints_dir,
            checkpoint_path=final_dir,
            checkpoint_source="final_saved_after_load_best_model_at_end",
            trainer_state_path=root_state_path if root_state_path.exists() else None,
            best_model_checkpoint=best_model_checkpoint,
            best_metric=float(best_metric) if isinstance(best_metric, (int, float)) else None,
            best_metric_name=metric_name,
        )

    if best_model_checkpoint:
        best_path = Path(best_model_checkpoint)
        if not best_path.is_absolute():
            best_path = PROJECT_ROOT / best_path
        if candidate_has_model_files(best_path):
            return CheckpointSelection(
                run_name=run_name,
                config_path=config_path,
                checkpoints_dir=checkpoints_dir,
                checkpoint_path=best_path,
                checkpoint_source="trainer_state.best_model_checkpoint",
                trainer_state_path=root_state_path if root_state_path.exists() else None,
                best_model_checkpoint=best_model_checkpoint,
                best_metric=float(best_metric) if isinstance(best_metric, (int, float)) else None,
                best_metric_name=metric_name,
            )

    if best_step is not None:
        step_path = checkpoints_dir / f"checkpoint-{best_step}"
        if candidate_has_model_files(step_path):
            return CheckpointSelection(
                run_name=run_name,
                config_path=config_path,
                checkpoints_dir=checkpoints_dir,
                checkpoint_path=step_path,
                checkpoint_source=f"best_{metric_name}_step",
                trainer_state_path=root_state_path if root_state_path.exists() else None,
                best_model_checkpoint=best_model_checkpoint,
                best_metric=best_metric,
                best_metric_name=metric_name,
            )

    checkpoints = sorted(
        [path for path in checkpoints_dir.glob("checkpoint-*") if candidate_has_model_files(path)]
    )
    if checkpoints:
        return CheckpointSelection(
            run_name=run_name,
            config_path=config_path,
            checkpoints_dir=checkpoints_dir,
            checkpoint_path=checkpoints[-1],
            checkpoint_source="latest_available_checkpoint_fallback",
            trainer_state_path=root_state_path if root_state_path.exists() else None,
            best_model_checkpoint=best_model_checkpoint,
            best_metric=float(best_metric) if isinstance(best_metric, (int, float)) else None,
            best_metric_name=metric_name,
        )

    raise FileNotFoundError(
        f"Nessun checkpoint valutabile trovato per {run_name} in {checkpoints_dir}"
    )


def should_load_4bit(config: dict[str, Any], override: str) -> bool:
    if override == "true":
        return True
    if override == "false":
        return False
    if "inference" in config:
        return bool(config["inference"].get("load_in_4bit", True))
    return config.get("finetuning", {}).get("quantization") == "4bit"


def load_model_and_processor(config: dict[str, Any], checkpoint_path: Path | None, load_in_4bit: bool):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    model_dir = resolve_path(config, "paths", "model_dir")
    processor_dir = checkpoint_path if checkpoint_path and (checkpoint_path / "processor_config.json").exists() else model_dir

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_source = model_dir
    if checkpoint_path and (checkpoint_path / "config.json").exists() and not (checkpoint_path / "adapter_config.json").exists():
        model_source = checkpoint_path

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_source),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if checkpoint_path and (checkpoint_path / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint_path))

    model.eval()
    pixel_cfg = config.get("finetuning", config.get("inference", {}))
    processor = AutoProcessor.from_pretrained(
        str(processor_dir),
        min_pixels=256 * 28 * 28,
        max_pixels=pixel_cfg.get("max_image_pixels", 1003520),
    )
    return model, processor


def flatten_summary_row(
    selection: CheckpointSelection,
    config: dict[str, Any],
    metrics: dict[str, Any],
    metrics_path: Path,
    qualitative_path: Path,
    predictions_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_name": selection.run_name,
        "method": config.get("finetuning", config.get("inference", {})).get("method"),
        "base_model": config.get("model", {}).get("base_model"),
        "config_path": str(selection.config_path.relative_to(PROJECT_ROOT)),
        "checkpoint_path": (
            str(selection.checkpoint_path.relative_to(PROJECT_ROOT))
            if selection.checkpoint_path and selection.checkpoint_path.is_relative_to(PROJECT_ROOT)
            else str(selection.checkpoint_path)
        ),
        "checkpoint_source": selection.checkpoint_source,
        "best_model_checkpoint": selection.best_model_checkpoint,
        "best_metric_name": selection.best_metric_name,
        "best_metric": selection.best_metric,
        "num_examples": metrics.get("num_examples"),
        "rouge1": metrics.get("rouge", {}).get("rouge1"),
        "rouge2": metrics.get("rouge", {}).get("rouge2"),
        "rougeL": metrics.get("rouge", {}).get("rougeL"),
        "bertscore_precision": metrics.get("bertscore", {}).get("precision"),
        "bertscore_recall": metrics.get("bertscore", {}).get("recall"),
        "bertscore_f1": metrics.get("bertscore", {}).get("f1"),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
        "qualitative_path": str(qualitative_path.relative_to(PROJECT_ROOT)),
        "predictions_path": str(predictions_path.relative_to(PROJECT_ROOT)) if predictions_path else None,
    }


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    ensure_dirs(output_dir)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV:  {csv_path}")


def main() -> None:
    args = parse_args()
    from shield.notebook_utils import evaluate_on_test_set, save_evaluation_results

    config_paths = resolve_config_paths(args)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_dirs(output_dir)

    rows: list[dict[str, Any]] = []

    for config_path in config_paths:
        config = load_config(config_path)
        run_name = config["run"]["name"]
        seed = config.get("finetuning", config.get("inference", {})).get("seed", 42)
        set_seed(seed)

        selection = select_checkpoint(config, config_path, trust_final=args.trust_final)
        print("=" * 80)
        print(f"Run:        {run_name}")
        print(f"Config:     {config_path.relative_to(PROJECT_ROOT)}")
        print(f"Checkpoint: {selection.checkpoint_path}")
        print(f"Source:     {selection.checkpoint_source}")
        print("=" * 80)

        processed_dir = resolve_path(config, "paths", "processed_dir")
        test_records = load_json_records(processed_dir / "test.json")
        if args.limit:
            test_records = test_records[: args.limit]

        load_4bit = should_load_4bit(config, args.load_in_4bit)
        model, processor = load_model_and_processor(config, selection.checkpoint_path, load_4bit)

        eval_cfg = config.get("evaluation", {})
        max_new_tokens = args.max_new_tokens or eval_cfg.get("max_new_tokens", 512)
        qualitative_examples = args.qualitative_examples or eval_cfg.get("qualitative_examples", 20)
        run_output_dir = output_dir / run_name
        metrics_path = run_output_dir / "metrics.json"
        qualitative_path = run_output_dir / "qualitative.json"
        predictions_path = None if args.no_save_predictions else run_output_dir / "predictions.json"

        results = evaluate_on_test_set(
            model=model,
            processor=processor,
            test_records=test_records,
            label=f"{run_name} best checkpoint",
            bertscore_model_type=eval_cfg.get("bertscore_model_type", "xlm-roberta-large"),
            max_new_tokens=max_new_tokens,
            qualitative_limit=qualitative_examples,
            keep_all_predictions=not args.no_save_predictions,
        )

        metadata = {
            "run_name": run_name,
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "checkpoints_dir": str(selection.checkpoints_dir.relative_to(PROJECT_ROOT)),
            "checkpoint_path": (
                str(selection.checkpoint_path.relative_to(PROJECT_ROOT))
                if selection.checkpoint_path and selection.checkpoint_path.is_relative_to(PROJECT_ROOT)
                else str(selection.checkpoint_path)
            ),
            "checkpoint_source": selection.checkpoint_source,
            "trainer_state_path": (
                str(selection.trainer_state_path.relative_to(PROJECT_ROOT))
                if selection.trainer_state_path and selection.trainer_state_path.is_relative_to(PROJECT_ROOT)
                else str(selection.trainer_state_path)
            ),
            "best_model_checkpoint": selection.best_model_checkpoint,
            "best_metric_name": selection.best_metric_name,
            "best_metric": selection.best_metric,
            "load_in_4bit": load_4bit,
            "limit": args.limit,
        }
        results["checkpoint_metadata"] = metadata
        save_evaluation_results(results, metrics_path, qualitative_path, predictions_path)
        with (run_output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        rows.append(
            flatten_summary_row(
                selection,
                config,
                results,
                metrics_path,
                qualitative_path,
                predictions_path,
            )
        )

        del model
        del processor
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    write_summary(rows, output_dir)


if __name__ == "__main__":
    main()
