from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..training.model import _bnb_config, _select_dtype


def load_eval_model(
    config: Mapping[str, Any],
    model_path: str | Path,
    adapter_dir: str | Path | None = None,
) -> Any:
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
        print(f"[eval] attn '{attn}' non disponibile ({exc}); fallback a 'sdpa'.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), attn_implementation="sdpa", **kwargs
        )

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    model.config.use_cache = True
    return model
