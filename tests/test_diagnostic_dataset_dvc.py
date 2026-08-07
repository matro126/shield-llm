from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "dataset/iu-xray/iu_xray_r2gen_final_impression_diagnostic"
COMMON_DEPS = {
    SOURCE,
    "src/shield/data/build_diagnostic.py",
    "src/shield/data/images.py",
    "src/shield/data/prompts.py",
    "src/shield/data/records.py",
    "src/shield/tracking/__init__.py",
    "src/shield/tracking/core.py",
    "src/shield/tracking/provenance.py",
    "uv.lock",
}


def test_dvc_declares_all_diagnostic_dataset_variants() -> None:
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    combinations = {
        "F_F": "iu_xray_en_F-F",
        "F_FI": "iu_xray_en_F-FI",
        "FL_F": "iu_xray_en_FL-F",
        "FL_FI": "iu_xray_en_FL-FI",
    }

    for variant in ("other", "no_other"):
        for stage_suffix, version in combinations.items():
            name = f"build_iu_xray_en_{variant}_{stage_suffix}"
            stage = stages[name]
            assert stage["cmd"] == (
                "uv run python -m shield.data.build_diagnostic "
                f"--variant {variant} --version {version}"
            )
            assert set(stage["deps"]) == COMMON_DEPS
            assert stage["outs"] == [
                f"dataset/iu-xray/en/{variant}/{version}"
            ]
