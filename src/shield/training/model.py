from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_processor(model_path: str | Path, ft: Mapping[str, Any]) -> Any:
    from transformers import AutoProcessor

    min_pixels = int(ft.get("min_image_pixels", 512 * 32 * 32))
    max_pixels = int(ft.get("max_image_pixels", 1280 * 32 * 32))
    processor = AutoProcessor.from_pretrained(
        str(model_path), min_pixels=min_pixels, max_pixels=max_pixels
    )
    _assert_processor_pixels(processor, min_pixels, max_pixels)
    return processor


def _effective_pixels(image_processor: Any) -> tuple[int | None, int | None]:
    min_px = getattr(image_processor, "min_pixels", None)
    max_px = getattr(image_processor, "max_pixels", None)
    size = getattr(image_processor, "size", None)
    if isinstance(size, Mapping):
        min_px = size.get("min_pixels", min_px)
        max_px = size.get("max_pixels", max_px)
    return (
        int(min_px) if min_px is not None else None,
        int(max_px) if max_px is not None else None,
    )


def _assert_processor_pixels(processor: Any, min_pixels: int, max_pixels: int) -> None:
    image_processor = getattr(processor, "image_processor", processor)
    eff_min, eff_max = _effective_pixels(image_processor)
    if eff_min is None or eff_max is None:
        print(
            "[model] ⚠️ impossibile verificare min/max_pixels del processor "
            f"(layout non riconosciuto); richiesti min={min_pixels}, max={max_pixels}."
        )
        return
    if (eff_min, eff_max) != (min_pixels, max_pixels):
        raise RuntimeError(
            "Risoluzione del processor non propagata: richiesti "
            f"min={min_pixels}/max={max_pixels}, effettivi min={eff_min}/max={eff_max}. "
            "Bug noto di transformers su min/max_pixels in from_pretrained: aggiorna la "
            "versione o imposta i pixel direttamente sull'image processor."
        )


def _select_dtype() -> Any:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _bnb_config(compute_dtype: Any) -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(config: Mapping[str, Any], model_path: str | Path) -> Any:
    from transformers import Qwen3VLForConditionalGeneration

    ft = config["finetuning"]
    method = ft["method"]
    dtype = _select_dtype()
    quant_config = _bnb_config(dtype) if method == "qlora" else None
    attn = ft.get("attn_implementation", "flash_attention_2")

    kwargs: dict[str, Any] = {
        "quantization_config": quant_config,
        "device_map": "auto",
        "torch_dtype": dtype,
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), attn_implementation=attn, **kwargs
        )
    except (ImportError, ValueError) as exc:
        print(f"[model] attn '{attn}' non disponibile ({exc}); fallback a 'sdpa'.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), attn_implementation="sdpa", **kwargs
        )
    model.config.use_cache = False

    gradient_checkpointing = ft.get("gradient_checkpointing", True)
    if method == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.enable_input_require_grads()
    return model


def apply_peft(model: Any, config: Mapping[str, Any]) -> Any:
    from peft import LoraConfig, get_peft_model

    peft_cfg = config["peft"]
    n_llm_layers = model.config.text_config.num_hidden_layers
    lora_config = LoraConfig(
        r=peft_cfg["lora_r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=peft_cfg["target_modules"],
        layers_to_transform=list(range(n_llm_layers)),
        layers_pattern="layers",
    )
    return get_peft_model(model, lora_config)


def trainable_summary(model: Any) -> dict[str, float]:
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    total = sum(param.numel() for param in model.parameters())
    return {
        "trainable_params": float(trainable),
        "total_params": float(total),
        "trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }


def build_model_and_processor(
    config: Mapping[str, Any], model_path: str | Path
) -> tuple[Any, Any]:
    from ..config import requires_peft

    model = load_base_model(config, model_path)
    processor = load_processor(model_path, config["finetuning"])
    if requires_peft(config):
        model = apply_peft(model, config)
    return model, processor
