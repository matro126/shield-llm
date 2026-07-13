from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.data.preprocessing import run_preprocessing

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_iu-xray_v1.0.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing IU X-ray (config-driven)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    import yaml

    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    result = run_preprocessing(config, PROJECT_ROOT, output_dir=args.output_dir)

    print(f"[preprocess] output: {result['output_dir']}")
    print(
        f"[preprocess] report: {result['n_reports']} | esempi: {result['n_examples']} | categorie: {result['categories']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
