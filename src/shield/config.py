from __future__ import annotations

import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ALLOWED_METHODS = {"qlora", "lora", "none"}
ALLOWED_METRICS = {"bleu", "rougeL", "bertscore", "clinicalbert", "chexbert"}
ALLOWED_STAGES = {"staging", "production", "archived", "none"}
PEFT_METHODS = {"qlora", "lora"}


class ConfigError(ValueError):
    pass


def _is_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_str_list(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and len(value) > 0
        and all(_is_str(item) for item in value)
    )


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise ConfigError("pyproject.toml non trovato: esegui dal repo SHIELD.")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config non trovata: {config_path}")
    with config_path.open("rb") as handle:
        try:
            return tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"TOML non valido in {config_path}: {exc}") from exc


def _require_section(
    config: Mapping[str, Any], name: str, errors: list[str]
) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        errors.append(f"[{name}]: sezione mancante o non valida")
        return {}
    return section


def _check(cond: bool, message: str, errors: list[str]) -> None:
    if not cond:
        errors.append(message)


def validate_config(config: Mapping[str, Any]) -> None:
    errors: list[str] = []

    exp = _require_section(config, "experiment", errors)
    _check(_is_str(exp.get("name")), "[experiment].name: stringa richiesta", errors)
    _check(_is_str(exp.get("family")), "[experiment].family: stringa richiesta", errors)

    mlf = _require_section(config, "mlflow", errors)
    _check(
        _is_str(mlf.get("experiment_name")),
        "[mlflow].experiment_name: stringa richiesta",
        errors,
    )

    ds = _require_section(config, "dataset", errors)
    for key in ("name", "version", "root"):
        _check(_is_str(ds.get(key)), f"[dataset].{key}: stringa richiesta", errors)

    model = _require_section(config, "model", errors)
    _check(
        _is_str(model.get("base_model")),
        "[model].base_model: stringa richiesta",
        errors,
    )
    _check(
        _is_str(model.get("local_path")),
        "[model].local_path: stringa richiesta",
        errors,
    )

    ft = _require_section(config, "finetuning", errors)
    method = ft.get("method")
    _check(
        method in ALLOWED_METHODS,
        f"[finetuning].method: atteso uno di {sorted(ALLOWED_METHODS)}, trovato {method!r}",
        errors,
    )
    _validate_finetuning_numbers(ft, errors)

    if method in PEFT_METHODS:
        peft = _require_section(config, "peft", errors)
        _check(
            _is_int(peft.get("lora_r")) and peft.get("lora_r", 0) > 0,
            "[peft].lora_r: intero > 0 richiesto",
            errors,
        )
        _check(
            _is_int(peft.get("lora_alpha")) and peft.get("lora_alpha", 0) > 0,
            "[peft].lora_alpha: intero > 0 richiesto",
            errors,
        )
        dropout = peft.get("lora_dropout", 0.0)
        _check(
            _is_num(dropout) and 0.0 <= dropout <= 1.0,
            "[peft].lora_dropout: numero in [0,1]",
            errors,
        )
        _check(
            _is_str_list(peft.get("target_modules")),
            "[peft].target_modules: lista di stringhe non vuota",
            errors,
        )

    _validate_evaluation(config.get("evaluation"), errors)
    _validate_registry(config.get("registry"), errors)

    if errors:
        joined = "\n  - ".join(errors)
        raise ConfigError(f"Config non conforme allo schema:\n  - {joined}")


def _validate_finetuning_numbers(ft: Mapping[str, Any], errors: list[str]) -> None:
    positive_ints = [
        "epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "max_new_tokens",
        "selection_max_samples",
    ]
    for key in positive_ints:
        if key in ft:
            _check(
                _is_int(ft[key]) and ft[key] > 0,
                f"[finetuning].{key}: intero > 0",
                errors,
            )
    if "learning_rate" in ft:
        _check(
            _is_num(ft["learning_rate"]) and ft["learning_rate"] > 0,
            "[finetuning].learning_rate: numero > 0",
            errors,
        )
    if "repetition_penalty" in ft:
        _check(
            _is_num(ft["repetition_penalty"]) and ft["repetition_penalty"] > 0,
            "[finetuning].repetition_penalty: numero > 0",
            errors,
        )
    if "warmup_ratio" in ft:
        _check(
            _is_num(ft["warmup_ratio"]) and 0.0 <= ft["warmup_ratio"] <= 1.0,
            "[finetuning].warmup_ratio: numero in [0,1]",
            errors,
        )
    if "weight_decay" in ft:
        _check(
            _is_num(ft["weight_decay"]) and ft["weight_decay"] >= 0,
            "[finetuning].weight_decay: numero >= 0",
            errors,
        )
    if "seed" in ft:
        _check(_is_int(ft["seed"]), "[finetuning].seed: intero", errors)
    if "early_stopping" in ft:
        _check(
            _is_bool(ft["early_stopping"]),
            "[finetuning].early_stopping: booleano",
            errors,
        )


def _validate_evaluation(evaluation: Any, errors: list[str]) -> None:
    if evaluation is None:
        return
    if not isinstance(evaluation, Mapping):
        errors.append("[evaluation]: sezione non valida")
        return
    metrics = evaluation.get("metrics")
    if metrics is not None:
        if not _is_str_list(metrics):
            errors.append("[evaluation].metrics: lista di stringhe non vuota")
        else:
            unknown = set(metrics) - ALLOWED_METRICS
            _check(
                not unknown,
                f"[evaluation].metrics: metriche sconosciute {sorted(unknown)}",
                errors,
            )
    if "disaggregate_by" in evaluation:
        _check(
            _is_str_list(evaluation["disaggregate_by"]),
            "[evaluation].disaggregate_by: lista di stringhe",
            errors,
        )
    if "min_subgroup_size" in evaluation:
        value = evaluation["min_subgroup_size"]
        _check(
            _is_int(value) and value > 0,
            "[evaluation].min_subgroup_size: intero > 0",
            errors,
        )
    if "capture_operational" in evaluation:
        _check(
            _is_bool(evaluation["capture_operational"]),
            "[evaluation].capture_operational: booleano",
            errors,
        )
    if "max_new_tokens" in evaluation:
        value = evaluation["max_new_tokens"]
        _check(
            _is_int(value) and value > 0,
            "[evaluation].max_new_tokens: intero > 0",
            errors,
        )
    if "repetition_penalty" in evaluation:
        value = evaluation["repetition_penalty"]
        _check(
            _is_num(value) and value > 0,
            "[evaluation].repetition_penalty: numero > 0",
            errors,
        )
    if "significance" in evaluation:
        _check(
            _is_bool(evaluation["significance"]),
            "[evaluation].significance: booleano",
            errors,
        )
    if "bootstrap_resamples" in evaluation:
        value = evaluation["bootstrap_resamples"]
        _check(
            _is_int(value) and value > 0,
            "[evaluation].bootstrap_resamples: intero > 0",
            errors,
        )
    if "bootstrap_seed" in evaluation:
        _check(
            _is_int(evaluation["bootstrap_seed"]),
            "[evaluation].bootstrap_seed: intero",
            errors,
        )
    if "baseline_run" in evaluation and evaluation["baseline_run"] is not None:
        _check(
            _is_str(evaluation["baseline_run"]),
            "[evaluation].baseline_run: stringa",
            errors,
        )


def _validate_registry(registry: Any, errors: list[str]) -> None:
    if registry is None:
        return
    if not isinstance(registry, Mapping):
        errors.append("[registry]: sezione non valida")
        return
    _check(
        _is_str(registry.get("model_name")),
        "[registry].model_name: stringa richiesta",
        errors,
    )
    promote = registry.get("promote_to", "staging")
    _check(
        promote in ALLOWED_STAGES,
        f"[registry].promote_to: atteso uno di {sorted(ALLOWED_STAGES)}",
        errors,
    )


def load_and_validate(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    validate_config(config)
    return config


def resolve_path(value: str | Path, project_root: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def dataset_root(config: Mapping[str, Any], project_root: str | Path) -> Path:
    return resolve_path(config["dataset"]["root"], project_root)


def images_root(config: Mapping[str, Any], project_root: str | Path) -> Path:
    dataset = config["dataset"]
    return resolve_path(dataset.get("images_root", dataset["root"]), project_root)


def model_path(config: Mapping[str, Any], project_root: str | Path) -> Path:
    return resolve_path(config["model"]["local_path"], project_root)


def method(config: Mapping[str, Any]) -> str:
    return str(config["finetuning"]["method"])


def requires_peft(config: Mapping[str, Any]) -> bool:
    return method(config) in PEFT_METHODS


def _main(argv: list[str]) -> int:
    if not argv:
        print("uso: python -m shield.config <config.toml>", file=sys.stderr)
        return 2
    try:
        config = load_and_validate(argv[0])
    except ConfigError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    exp = config["experiment"]["name"]
    print(
        f"✅ Config valida: {exp} (method={method(config)}, dataset={config['dataset']['version']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
