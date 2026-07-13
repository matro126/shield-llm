from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.cards import write_dataset_card, write_model_card
from shield.config import load_and_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generazione model/dataset card.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _generate_model_card(config: dict) -> None:
    from shield.tracking import dataset_provenance, git_metadata

    exp = config["experiment"]
    provenance = {
        **git_metadata(PROJECT_ROOT),
        **dataset_provenance(config, PROJECT_ROOT),
    }

    metrics_file = (
        PROJECT_ROOT
        / "outputs"
        / exp["family"]
        / exp["name"]
        / "evaluation"
        / "metrics.json"
    )
    results = (
        json.loads(metrics_file.read_text(encoding="utf-8"))
        if metrics_file.exists()
        else None
    )
    if results is None:
        print(
            "[cards] metrics.json assente: model card con sezioni quantitative vuote."
        )

    output = (
        PROJECT_ROOT / "experiments" / exp["family"] / exp["name"] / "model_card.md"
    )
    write_model_card(config, output, results=results, provenance=provenance)
    print(f"[cards] model card:   {output}")


def _generate_dataset_card(config: dict) -> None:
    ds_root = _resolve(Path(config["dataset"]["root"]))
    manifest_file = ds_root / "manifest.yaml"
    if not manifest_file.exists():
        print(f"[cards] manifest non trovato in {ds_root}: salto la dataset card.")
        return

    import yaml

    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
    stats_file = ds_root / "stats.json"
    stats = (
        json.loads(stats_file.read_text(encoding="utf-8"))
        if stats_file.exists()
        else {}
    )

    output = ds_root / "dataset_card.md"
    write_dataset_card(manifest, stats, output)
    print(f"[cards] dataset card: {output}")


def main() -> int:
    args = parse_args()
    config = load_and_validate(_resolve(args.config))
    if not args.skip_model:
        _generate_model_card(config)
    if not args.skip_dataset:
        _generate_dataset_card(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
