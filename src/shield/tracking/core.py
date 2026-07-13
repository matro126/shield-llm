from __future__ import annotations

import json
import math
import os
import re
import tomllib
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .provenance import dataset_provenance, git_metadata

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_NAME = "shield-qwen3vl-xray-it"


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def flatten_mapping(
    value: Mapping[str, Any],
    prefix: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        clean_key = str(key).replace(" ", "_")
        full_key = f"{prefix}{sep}{clean_key}" if prefix else clean_key
        if isinstance(item, Mapping):
            flat.update(flatten_mapping(item, full_key, sep=sep))
        else:
            flat[full_key] = item
    return flat


def stringify_param(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed in this environment. Run `uv sync` first."
        ) from exc
    return mlflow


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> Any:
    mlflow = _mlflow()
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    experiment = (
        experiment_name
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or DEFAULT_EXPERIMENT_NAME
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow


def log_params(mapping: Mapping[str, Any], prefix: str | None = None) -> None:
    mlflow = _mlflow()
    for key, value in flatten_mapping(mapping).items():
        if value is None:
            continue
        name = f"{prefix}.{key}" if prefix else key
        mlflow.log_param(name, stringify_param(value))


def log_tags(tags: Mapping[str, Any]) -> None:
    mlflow = _mlflow()
    for key, value in tags.items():
        if value is not None:
            mlflow.set_tag(key, stringify_param(value))


_METRIC_NAME_INVALID = re.compile(r"[^A-Za-z0-9_.\- :/]")


def _sanitize_metric_name(name: str) -> str:
    return _METRIC_NAME_INVALID.sub("_", name)


def log_numeric_metrics(
    mapping: Mapping[str, Any],
    prefix: str,
    step: int | None = None,
) -> None:
    mlflow = _mlflow()
    for key, value in flatten_mapping(mapping).items():
        if is_finite_number(value):
            mlflow.log_metric(
                _sanitize_metric_name(f"{prefix}.{key}"), float(value), step=step
            )


def log_artifact_if_exists(
    path: str | Path,
    artifact_path: str | None = None,
    allow_dir: bool = False,
) -> bool:
    mlflow = _mlflow()
    artifact = Path(path)
    if artifact.is_file():
        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
        return True
    if allow_dir and artifact.is_dir():
        mlflow.log_artifacts(str(artifact), artifact_path=artifact_path)
        return True
    return False


def log_standard_artifacts(root: str | Path, config: Mapping[str, Any]) -> None:
    root_path = Path(root)
    for rel_path in [
        "pyproject.toml",
        "uv.lock",
        "dvc.yaml",
        "dvc.lock",
        ".dvc/config",
    ]:
        log_artifact_if_exists(root_path / rel_path, artifact_path="reproducibility")

    dataset = config.get("dataset", {})
    if isinstance(dataset, Mapping):
        dataset_root = dataset.get("root")
        if dataset_root:
            manifest = root_path / str(dataset_root) / "manifest.yaml"
            log_artifact_if_exists(manifest, artifact_path="dataset")


def log_metrics_file(path: str | Path, prefix: str = "evaluation") -> None:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return

    data = load_json(metrics_path)
    if isinstance(data, Mapping):
        log_numeric_metrics(data, prefix=prefix)
        non_numeric = {
            key: value
            for key, value in flatten_mapping(data).items()
            if not is_finite_number(value)
        }
        if non_numeric:
            log_params(non_numeric, prefix=prefix)

    log_artifact_if_exists(metrics_path, artifact_path="evaluation")


def log_trainer_state(path: str | Path) -> None:
    state_path = Path(path)
    if not state_path.exists():
        return

    state = load_json(state_path)
    log_artifact_if_exists(state_path, artifact_path="training")

    history = state.get("log_history", []) if isinstance(state, Mapping) else []
    latest: dict[str, float] = {}
    for index, record in enumerate(history):
        if not isinstance(record, Mapping):
            continue
        step = int(record.get("step", index))
        for key, value in record.items():
            if key == "step" or not is_finite_number(value):
                continue
            metric_name = f"trainer.{key}"
            metric_value = float(value)
            _mlflow().log_metric(metric_name, metric_value, step=step)
            latest[key] = metric_value

    for key, value in latest.items():
        _mlflow().log_metric(f"trainer.final.{key}", value)


@contextmanager
def mlflow_run(
    config: Mapping[str, Any],
    root: str | Path = ".",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    mlflow_section = config.get("mlflow", {})
    run_section = config.get("run", {})
    experiment_section = config.get("experiment", {})

    if isinstance(mlflow_section, Mapping):
        tracking_uri = tracking_uri or mlflow_section.get("tracking_uri")
        experiment_name = experiment_name or mlflow_section.get("experiment_name")

    if isinstance(experiment_section, Mapping):
        run_name = run_name or experiment_section.get("name")
    if isinstance(run_section, Mapping):
        run_name = run_name or run_section.get("name")

    mlflow = configure_mlflow(
        tracking_uri=str(tracking_uri) if tracking_uri else None,
        experiment_name=str(experiment_name) if experiment_name else None,
    )

    with mlflow.start_run(run_name=run_name):
        log_params(config, prefix="config")
        log_tags(git_metadata(root))
        log_tags(dataset_provenance(config, root))
        if tags:
            log_tags(tags)
        note = None
        if isinstance(experiment_section, Mapping):
            note = experiment_section.get("notes")
        if note is None and isinstance(run_section, Mapping):
            note = run_section.get("notes")
        if note:
            mlflow.set_tag("mlflow.note.content", stringify_param(note))
        log_standard_artifacts(root, config)
        yield mlflow
