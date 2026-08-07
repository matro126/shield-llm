from __future__ import annotations

from pathlib import Path

from shield.training.config import Identity, build_config, read_overrides


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = (
    ROOT
    / "training"
    / "en"
    / "Qwen-3-VL-2B-Instruct"
    / "lora"
    / "iu_xray_r2gen_FL-FI"
)
BASE = DIRECTORY / "en_2B_lora_FL-FI.py"


def configuration(path: Path):
    assert path.is_file(), path
    return build_config(Identity.from_path(path), ROOT, read_overrides(path))


def test_other_and_no_other_only_change_experiment_paths() -> None:
    base = configuration(BASE).as_dict()
    variants = {
        "other": {
            "experiment": "en_2B_lora_FL-FI_other",
            "dataset_root": "dataset/iu-xray/en/other/iu_xray_en_FL-FI",
            "results_dir": (
                "training/en/Qwen-3-VL-2B-Instruct/lora/"
                "iu_xray_r2gen_FL-FI/results_other"
            ),
        },
        "no_other": {
            "experiment": "en_2B_lora_FL-FI_no_other",
            "dataset_root": "dataset/iu-xray/en/no_other/iu_xray_en_FL-FI",
            "results_dir": (
                "training/en/Qwen-3-VL-2B-Instruct/lora/"
                "iu_xray_r2gen_FL-FI/results_no_other"
            ),
        },
    }

    for variant, changes in variants.items():
        path = DIRECTORY / f"en_2B_lora_FL-FI_{variant}.py"
        actual = configuration(path).as_dict()
        expected = {**base, **changes}
        assert actual == expected
