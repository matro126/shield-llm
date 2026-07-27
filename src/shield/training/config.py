from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

LANG_DIR = {"en": "en", "it": "ita"}
LANG_CODE = {"en": "en", "it": "it"}

MODEL_HUB_ID = {
    "Qwen-3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen-3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen-3-VL-32B-Instruct": "Qwen/Qwen3-VL-32B-Instruct",
}
MODEL_SHORT = {name: name.split("-")[3] for name in MODEL_HUB_ID}

TRAINING_MODES = ("lora", "qlora")
BASELINE_MODE = "baseline"

VIEWS_BY_CODE = {"FL": "frontal_lateral", "F": "frontal"}
TARGET_BY_CODE = {"F": "findings", "FI": "findings_impression"}


@dataclass(frozen=True)
class Identity:
    lang: str
    model_dir: str
    mode: str
    dataset_dir: str

    @property
    def code(self) -> str:
        return self.dataset_dir.rsplit("_", 1)[-1]

    @property
    def views(self) -> str:
        return VIEWS_BY_CODE[self.code.split("-", 1)[0]]

    @property
    def target(self) -> str:
        return TARGET_BY_CODE[self.code.split("-", 1)[1]]

    @property
    def model_short(self) -> str:
        return MODEL_SHORT[self.model_dir]

    @property
    def base_model(self) -> str:
        return MODEL_HUB_ID[self.model_dir]

    @property
    def dataset_root(self) -> str:
        return (
            f"dataset/iu-xray/{LANG_DIR[self.lang]}/"
            f"iu_xray_{LANG_CODE[self.lang]}_{self.code}"
        )

    @property
    def is_baseline(self) -> bool:
        return self.mode == BASELINE_MODE

    @property
    def name(self) -> str:
        if self.is_baseline:
            return f"{self.lang}_{self.model_short}_{self.code}"
        return f"{self.lang}_{self.model_short}_{self.mode}_{self.code}"

    @property
    def script_name(self) -> str:
        return f"baseline_{self.name}.py" if self.is_baseline else f"{self.name}.py"

    @property
    def baseline(self) -> Identity:
        return Identity(self.lang, self.model_dir, BASELINE_MODE, self.dataset_dir)

    @property
    def script_relpath(self) -> str:
        return f"{self.relpath}/{self.script_name}"

    @property
    def local_model_path(self) -> str:
        return f"models/fine-tuning/{self.base_model.split('/', 1)[1]}"

    @property
    def relpath(self) -> str:
        return f"training/{self.lang}/{self.model_dir}/{self.mode}/{self.dataset_dir}"

    @classmethod
    def from_path(cls, path: str | Path) -> Identity:
        folder = Path(path).resolve()
        if folder.is_file():
            folder = folder.parent
        parts = folder.parts
        try:
            root = len(parts) - 1 - parts[::-1].index("training")
        except ValueError:
            raise ValueError(f"{folder} non e' dentro un albero 'training/'") from None
        lang, model_dir, mode, dataset_dir = parts[root + 1 : root + 5]
        return cls(lang=lang, model_dir=model_dir, mode=mode, dataset_dir=dataset_dir)


@dataclass
class Config:
    experiment: str = ""
    lang: str = "it"
    model_dir: str = ""
    mode: str = "qlora"
    dataset_code: str = ""
    views: str = ""
    target: str = ""

    dataset_root: str = ""
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    base_model: str = ""
    model_path: str = ""
    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    learning_rate: float = 1e-5
    max_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 1024
    logging_steps: int = 5
    seed: int = 42

    dataloader_num_workers: int = 8
    dataloader_persistent_workers: bool = True
    archive_results: bool = True
    archive_adapter: bool = True

    eval_cadence: str = "epoch"
    eval_steps: int = 65
    eval_metrics: tuple[str, ...] = ("bleu", "rougeL")
    monitor_metric: str = "rougeL"
    monitor_mode: str = "max"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.01
    eval_max_samples: int | None = None
    gen_batch_size: int = 16
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1

    test_metrics: tuple[str, ...] = (
        "bleu", "rougeL", "bertscore", "clinicalbert", "chexbert",
    )
    baseline_max_samples: int | None = None

    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = ""

    results_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_FIELD_TYPES = {f.name: f.type for f in fields(Config)}

METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "bleu": ("bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4"),
    "rougeL": ("rouge1", "rouge2", "rougeL"),
    "rouge": ("rouge1", "rouge2", "rougeL"),
    "bertscore": ("bertscore_precision", "bertscore_recall", "bertscore_f1"),
    "clinicalbert": (
        "clinicalbert_precision", "clinicalbert_recall", "clinicalbert_f1",
    ),
    "chexbert": (
        "chexbert_accuracy", "chexbert_f1_micro", "chexbert_f1_macro",
        "chexbert_f1_micro_top5", "chexbert_f1_macro_top5",
    ),
}


def available_metric_keys(metric_names: Sequence[str]) -> set[str]:
    keys: set[str] = {"num_examples"}
    for name in metric_names:
        keys.update(METRIC_KEYS.get(name, ()))
    return keys


def validate(cfg: Config) -> None:
    unknown = [m for m in cfg.eval_metrics if m not in METRIC_KEYS]
    if unknown:
        raise ValueError(
            f"eval_metrics contiene nomi ignoti {unknown}: verrebbero ignorati in "
            f"silenzio da compute_text_metrics. Ammessi: {sorted(METRIC_KEYS)}"
        )
    keys = available_metric_keys(cfg.eval_metrics)
    if cfg.monitor_metric not in keys:
        raise ValueError(
            f"monitor_metric='{cfg.monitor_metric}' non e' fra le metriche calcolate "
            f"in validazione (eval_metrics={list(cfg.eval_metrics)} produce {sorted(keys)}). "
            "Il gate non troverebbe il valore e nessun best adapter verrebbe salvato."
        )
    unknown_test = [m for m in cfg.test_metrics if m not in METRIC_KEYS]
    if unknown_test:
        raise ValueError(
            f"test_metrics contiene nomi ignoti {unknown_test}: verrebbero ignorati "
            f"in silenzio. Ammessi: {sorted(METRIC_KEYS)}"
        )
    if cfg.monitor_mode not in ("max", "min"):
        raise ValueError(f"monitor_mode deve essere 'max' o 'min', non {cfg.monitor_mode!r}")
    if cfg.per_device_train_batch_size < 1 or cfg.gradient_accumulation_steps < 1:
        raise ValueError("batch e gradient_accumulation_steps devono essere >= 1")


def load_defaults(project_root: Path) -> dict[str, Any]:
    path = project_root / "training" / "defaults.toml"
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    merged: dict[str, Any] = {}
    for section in data.values():
        if isinstance(section, dict):
            merged.update(section)
    merged.update({k: v for k, v in data.items() if not isinstance(v, dict)})
    return merged


def _coerce(name: str, value: Any) -> Any:
    if name in ("eval_max_samples", "baseline_max_samples") and value == 0:
        return None
    declared = _FIELD_TYPES.get(name, "")
    if isinstance(value, list) and "tuple" in str(declared):
        return tuple(value)
    return value


def build_config(
    identity: Identity,
    project_root: Path,
    overrides: dict[str, Any] | None = None,
) -> Config:
    values: dict[str, Any] = dict(load_defaults(project_root))

    values.update(
        {
            "experiment": identity.name,
            "lang": identity.lang,
            "model_dir": identity.model_dir,
            "mode": identity.mode,
            "dataset_code": identity.code,
            "views": identity.views,
            "target": identity.target,
            "dataset_root": identity.dataset_root,
            "base_model": identity.base_model,
            "model_path": identity.local_model_path,
            "load_in_4bit": identity.mode == "qlora",
            "results_dir": f"{identity.relpath}/results",
        }
    )
    values.update(overrides or {})

    unknown = sorted(set(values) - set(_FIELD_TYPES))
    if unknown:
        raise ValueError(f"iperparametri sconosciuti: {unknown}")
    cfg = Config(**{k: _coerce(k, v) for k, v in values.items()})
    validate(cfg)
    return cfg
