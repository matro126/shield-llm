from __future__ import annotations

import ast
import math
import re
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
    vision_target_modules: tuple[str, ...] = ("qkv", "proj", "linear_fc1", "linear_fc2")
    merger_target_modules: tuple[str, ...] = ("linear_fc1", "linear_fc2")

    tune_mm_llm: bool = True
    tune_mm_vision: bool = False
    tune_mm_mlp: bool = False

    learning_rate: float = 1e-5
    vision_lr: float | None = None
    merger_lr: float | None = None
    max_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    max_seq_length: int = 1024
    min_pixels: int | None = None
    max_pixels: int | None = None
    logging_steps: int = 5
    seed: int = 42

    training_strategy: str = "standard"
    training_phase: str = "full"
    clinical_pretrain_epochs: int = 0
    clinical_rehearsal_ratio: float = 0.0
    clinical_adapter_path: str = ""
    clinical_balance: bool = False
    clinical_healthy_ratio: float = 0.3
    healthy_ratio: float = 0.3
    pathological_ratio: float = 0.6
    other_ratio: float = 0.1
    rare_weight_cap: float = 4.0

    dataloader_num_workers: int = 8
    dataloader_persistent_workers: bool = True
    full_determinism: bool = False
    archive_results: bool = True
    archive_adapter: bool = True

    eval_cadence: str = "epoch"
    eval_steps: int = 65
    eval_metrics: tuple[str, ...] = ("bleu", "rougeL", "bertscore")
    monitor_metric: str = "findings.bertscore_f1"
    monitor_mode: str = "max"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.01
    eval_max_samples: int | None = None
    gen_batch_size: int = 16
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1
    chexbert_translate: bool = False
    chexbert_translator: str = "Helsinki-NLP/opus-mt-it-en"
    chexbert_glossary: str = ""
    bertscore_model: str = "xlm-roberta-large"
    hash_base_model_full: bool = False
    save_every_eval: bool = False

    test_metrics: tuple[str, ...] = ("bleu", "rougeL", "bertscore")
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
        "chexbert_accuracy",
        "chexbert_precision_micro", "chexbert_recall_micro", "chexbert_f1_micro",
        "chexbert_precision_macro", "chexbert_recall_macro", "chexbert_f1_macro",
        "chexbert_precision_micro_top5", "chexbert_recall_micro_top5",
        "chexbert_f1_micro_top5",
        "chexbert_precision_macro_top5", "chexbert_recall_macro_top5",
        "chexbert_f1_macro_top5",
    ),
}


def resolve_metric_selection(
    requested: Sequence[str] | None,
    configured: Sequence[str],
) -> tuple[list[str], str]:
    names = list(requested) if requested else list(configured)
    return names, ",".join(names)


def metric_sections(target: str) -> tuple[str, ...]:
    return ("findings", "impression") if target == "findings_impression" else ("findings",)


def available_metric_keys(
    metric_names: Sequence[str], target: str = ""
) -> set[str]:
    base: set[str] = {"num_examples"}
    for name in metric_names:
        base.update(METRIC_KEYS.get(name, ()))
    if not target:
        return base
    keys = {f"{s}.{k}" for s in metric_sections(target) for k in base}
    keys.add("val_loss")
    return keys


def validate(cfg: Config) -> None:
    strategies = ("standard", "balanced", "clinical")
    phases = ("full", "clinical_only", "report_only")
    if cfg.training_strategy not in strategies:
        raise ValueError(
            f"training_strategy deve essere uno di {strategies}, non "
            f"{cfg.training_strategy!r}"
        )
    if cfg.training_phase not in phases:
        raise ValueError(
            f"training_phase deve essere uno di {phases}, non {cfg.training_phase!r}"
        )
    if cfg.training_strategy != "clinical" and cfg.training_phase != "full":
        raise ValueError("training_phase separato e' ammesso soltanto per clinical")
    if cfg.clinical_balance and (
        cfg.training_strategy != "clinical" or cfg.training_phase == "report_only"
    ):
        raise ValueError(
            "clinical_balance richiede uno stadio clinico eseguito nella run"
        )
    if cfg.training_phase == "report_only":
        if not cfg.clinical_adapter_path:
            raise ValueError("clinical_adapter_path e' obbligatorio per report_only")
        if cfg.clinical_pretrain_epochs != 0:
            raise ValueError("clinical_pretrain_epochs deve essere 0 per report_only")
    elif cfg.clinical_adapter_path:
        raise ValueError("clinical_adapter_path e' ammesso soltanto per report_only")
    if cfg.clinical_pretrain_epochs < 0:
        raise ValueError("clinical_pretrain_epochs deve essere >= 0")
    if not 0.0 <= cfg.clinical_rehearsal_ratio < 1.0:
        raise ValueError(
            "clinical_rehearsal_ratio deve essere compreso fra 0 incluso e 1 escluso"
        )
    if cfg.training_strategy in ("standard", "balanced"):
        if cfg.clinical_pretrain_epochs != 0:
            raise ValueError(
                "clinical_pretrain_epochs deve essere 0 per strategie non clinical"
            )
        if cfg.clinical_rehearsal_ratio != 0.0:
            raise ValueError(
                "clinical_rehearsal_ratio deve essere 0 per strategie non clinical"
            )
    if cfg.training_strategy == "clinical":
        if cfg.training_phase != "report_only" and cfg.clinical_pretrain_epochs < 1:
            raise ValueError(
                "clinical_pretrain_epochs deve essere positivo per clinical"
            )
        if cfg.clinical_rehearsal_ratio <= 0.0:
            raise ValueError(
                "clinical_rehearsal_ratio deve essere positivo per clinical"
            )
    if not 0.0 < cfg.clinical_healthy_ratio < 1.0:
        raise ValueError("clinical_healthy_ratio deve essere compreso fra 0 e 1 esclusi")
    report_ratios = {
        "healthy_ratio": cfg.healthy_ratio,
        "pathological_ratio": cfg.pathological_ratio,
        "other_ratio": cfg.other_ratio,
    }
    for name, value in report_ratios.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} deve essere compreso fra 0 e 1")
    if not math.isclose(sum(report_ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("La sum dei rapporti healthy/pathological/other deve essere 1")
    if cfg.rare_weight_cap <= 0:
        raise ValueError("rare_weight_cap deve essere positivo")
    unknown = [m for m in cfg.eval_metrics if m not in METRIC_KEYS]
    if unknown:
        raise ValueError(
            f"eval_metrics contiene nomi ignoti {unknown}: verrebbero ignorati in "
            f"silenzio da compute_text_metrics. Ammessi: {sorted(METRIC_KEYS)}"
        )
    keys = available_metric_keys(cfg.eval_metrics, cfg.target)
    if cfg.monitor_metric not in keys:
        raise ValueError(
            f"monitor_metric='{cfg.monitor_metric}' non e' fra le metriche calcolate "
            f"in validazione. Le metriche sono per sezione e non vengono mediate: il "
            f"gate va indicato come '<sezione>.<metrica>', per esempio "
            f"'findings.chexbert_f1_micro_top5'. Con target='{cfg.target}' e "
            f"eval_metrics={list(cfg.eval_metrics)} le chiavi disponibili sono "
            f"{sorted(keys)}."
        )
    unknown_test = [m for m in cfg.test_metrics if m not in METRIC_KEYS]
    if unknown_test:
        raise ValueError(
            f"test_metrics contiene nomi ignoti {unknown_test}: verrebbero ignorati "
            f"in silenzio. Ammessi: {sorted(METRIC_KEYS)}"
        )
    if cfg.monitor_mode not in ("max", "min"):
        raise ValueError(f"monitor_mode deve essere 'max' o 'min', non {cfg.monitor_mode!r}")

    usa_chexbert = "chexbert" in set(cfg.eval_metrics) | set(cfg.test_metrics)
    if usa_chexbert and cfg.lang != "en" and not cfg.chexbert_translate:
        raise ValueError(
            f"chexbert e' fra le metriche ma lang='{cfg.lang}' e chexbert_translate e' "
            "False: il labeler verrebbe applicato a testo non inglese e produrrebbe "
            "numeri privi di significato, senza sollevare errori. Attiva "
            "chexbert_translate, oppure togli 'chexbert' da eval_metrics e test_metrics."
        )
    if usa_chexbert and cfg.lang == "en" and cfg.chexbert_translate:
        raise ValueError(
            "chexbert_translate e' attivo su lang='en': il testo verrebbe tradotto "
            "dall'italiano quando e' gia' inglese."
        )
    if cfg.per_device_train_batch_size < 1 or cfg.gradient_accumulation_steps < 1:
        raise ValueError("batch e gradient_accumulation_steps devono essere >= 1")

    if not (cfg.tune_mm_llm or cfg.tune_mm_vision or cfg.tune_mm_mlp):
        raise ValueError(
            "tune_mm_llm, tune_mm_vision e tune_mm_mlp sono tutti False: non ci "
            "sarebbe nessun modulo addestrabile."
        )
    if cfg.vision_lr is not None and not cfg.tune_mm_vision:
        raise ValueError(
            f"vision_lr={cfg.vision_lr} ma tune_mm_vision e' False: la vision tower "
            "non riceve adapter, quindi il learning rate verrebbe ignorato in "
            "silenzio. Attiva tune_mm_vision, oppure togli vision_lr."
        )
    if cfg.merger_lr is not None and not cfg.tune_mm_mlp:
        raise ValueError(
            f"merger_lr={cfg.merger_lr} ma tune_mm_mlp e' False: il projector non "
            "riceve adapter, quindi il learning rate verrebbe ignorato in silenzio. "
            "Attiva tune_mm_mlp, oppure togli merger_lr."
        )
    for name in ("learning_rate", "vision_lr", "merger_lr"):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} deve essere positivo, ricevuto {value}")

    if cfg.vision_lr is not None or cfg.merger_lr is not None:
        if cfg.optim != "adamw_torch":
            raise ValueError(
                f"optim='{cfg.optim}' con learning rate per componente: i gruppi di "
                "parametri sono costruiti su torch.optim.AdamW, quindi optim "
                "verrebbe ignorato. Usa 'adamw_torch', oppure togli vision_lr e "
                "merger_lr."
            )
    for name in ("min_pixels", "max_pixels"):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} deve essere positivo, ricevuto {value}")
    if (
        cfg.min_pixels is not None
        and cfg.max_pixels is not None
        and cfg.min_pixels >= cfg.max_pixels
    ):
        raise ValueError(
            f"min_pixels={cfg.min_pixels} >= max_pixels={cfg.max_pixels}"
        )


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


def read_overrides(script: Path) -> dict[str, Any]:
    if not script.is_file():
        return {}
    text = script.read_text(encoding="utf-8")
    match = re.search(r"^OVERRIDES\s*:\s*dict\s*=\s*(.+)\Z", text, re.M | re.S)
    if not match:
        return {}
    snippet = match.group(1)
    for end in range(len(snippet)):
        try:
            value = ast.literal_eval(snippet[: end + 1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


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


def build_entrypoint_config(identity: Identity, project_root: Path) -> Config:
    script = project_root / identity.script_relpath
    return build_config(identity, project_root, read_overrides(script))
