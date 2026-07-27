from __future__ import annotations

from typing import Any

from .config import Config, Identity, build_config, load_defaults
from .dashboard import LiveDashboard, hms

_LAZY = {
    "GenerativeEvalEarlyStop": ".callbacks",
    "main_baseline": ".baseline",
    "run_baseline": ".baseline",
    "LossLogger": ".callbacks",
    "QwenVLCollator": ".model",
    "compute_val_loss": ".evaluation",
    "evaluate_generative": ".evaluation",
    "find_project_root": ".runner",
    "flatten_sectioned": ".evaluation",
    "generate_predictions": ".evaluation",
    "load_model_and_processor": ".model",
    "main": ".runner",
    "prompts_of": ".evaluation",
    "run_experiment": ".runner",
    "sequence_length_probe": ".model",
}

__all__ = [
    "Config",
    "Identity",
    "LiveDashboard",
    "build_config",
    "hms",
    "load_defaults",
    *sorted(_LAZY),
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
