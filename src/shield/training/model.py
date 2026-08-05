from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

LORA_GROUPS = ("llm", "vision", "merger")


def _alternation(names: Any) -> str:
    return "|".join(re.escape(name) for name in names)


def lora_targets(cfg: Any) -> Any:
    if cfg.tune_mm_llm and not (cfg.tune_mm_vision or cfg.tune_mm_mlp):
        return list(cfg.target_modules)
    parts: list[str] = []
    if cfg.tune_mm_llm:
        parts.append(rf".*language_model\..*\.(?:{_alternation(cfg.target_modules)})")
    if cfg.tune_mm_vision:
        parts.append(
            rf".*visual\.blocks\..*\.(?:{_alternation(cfg.vision_target_modules)})"
        )
    if cfg.tune_mm_mlp:
        parts.append(
            rf".*visual\.merger\.(?:.*\.)?(?:{_alternation(cfg.merger_target_modules)})"
        )
    return "|".join(f"(?:{part})" for part in parts)


SIZE_KEY = {"min_pixels": "shortest_edge", "max_pixels": "longest_edge"}


def apply_pixel_budget(processor: Any, cfg: Any) -> dict[str, int]:
    wanted = {
        name: getattr(cfg, name)
        for name in ("min_pixels", "max_pixels")
        if getattr(cfg, name) is not None
    }
    if not wanted:
        return {}
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError(
            f"min_pixels/max_pixels richiesti ma {type(processor).__name__} non "
            "espone image_processor: il budget visuale verrebbe ignorato."
        )
    for name, value in wanted.items():
        setattr(image_processor, name, value)
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            size[SIZE_KEY[name]] = value
    for name, value in wanted.items():
        applied = getattr(image_processor, name, None)
        if applied != value:
            raise RuntimeError(
                f"{name}={value} non e' stato applicato a "
                f"{type(image_processor).__name__}: risulta {applied!r}."
            )
    return wanted


def lora_adapter_report(model: Any) -> dict[str, int]:
    found: dict[str, set[str]] = {group: set() for group in LORA_GROUPS}
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        module = name.rsplit(".lora_", 1)[0]
        if "visual.merger" in module:
            found["merger"].add(module)
        elif "visual." in module:
            found["vision"].add(module)
        else:
            found["llm"].add(module)
    return {group: len(modules) for group, modules in found.items()}


def assert_lora_coverage(cfg: Any, report: dict[str, int]) -> None:
    wanted = {
        "llm": cfg.tune_mm_llm,
        "vision": cfg.tune_mm_vision,
        "merger": cfg.tune_mm_mlp,
    }
    flag = {"llm": "tune_mm_llm", "vision": "tune_mm_vision", "merger": "tune_mm_mlp"}
    for group in LORA_GROUPS:
        if wanted[group] and report[group] == 0:
            raise RuntimeError(
                f"{flag[group]} e' True ma nessun modulo del gruppo '{group}' ha "
                f"ricevuto un adapter LoRA. Adapter trovati: {report}. I nomi dei "
                "moduli attesi non corrispondono a quelli del modello caricato: "
                "controlla target_modules, vision_target_modules e "
                "merger_target_modules."
            )
        if not wanted[group] and report[group] > 0:
            raise RuntimeError(
                f"{flag[group]} e' False ma {report[group]} moduli del gruppo "
                f"'{group}' hanno ricevuto un adapter LoRA. Adapter trovati: "
                f"{report}. Verrebbe addestrato un componente che hai escluso."
            )


def load_model_and_processor(
    cfg: Any, project_root: Path, with_adapter: bool = True
) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    local_path = project_root / cfg.model_path if cfg.model_path else None
    if local_path is not None and local_path.exists():
        model_src = str(local_path)
        print(f"[modello] copia locale: {model_src}")
    else:
        model_src = cfg.base_model
        print(f"[modello] dall'hub: {model_src}")

    processor = AutoProcessor.from_pretrained(model_src, trust_remote_code=True)
    budget = apply_pixel_budget(processor, cfg)
    if budget:
        print(f"[processor] budget visuale: {budget}")
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if cfg.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForImageTextToText.from_pretrained(model_src, **kwargs)
    if not with_adapter:
        model.eval()
        model.config.use_cache = True
        print("[modello] base nuda (nessun adapter): baseline zero-shot")
        return model, processor

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_targets(cfg),
        ),
    )
    report = lora_adapter_report(model)
    assert_lora_coverage(cfg, report)
    print(
        f"[lora] adapter per gruppo: llm={report['llm']} "
        f"vision={report['vision']} merger={report['merger']}"
    )
    model.print_trainable_parameters()
    return model, processor


def _adapter_config_value(value: Any) -> Any:
    if isinstance(value, (set, list, tuple)):
        return tuple(sorted(value))
    return value


def validate_training_adapter_config(model: Any, saved: dict[str, Any]) -> None:
    active_name = getattr(model, "active_adapter", "default")
    if not isinstance(active_name, str):
        active_name = "default"
    current = model.peft_config[active_name]
    for key in ("r", "lora_alpha", "target_modules", "base_model_name_or_path"):
        expected = _adapter_config_value(getattr(current, key, None))
        actual = _adapter_config_value(saved.get(key))
        if expected != actual:
            raise ValueError(
                f"Adapter clinico incompatibile per {key}: "
                f"config corrente={expected!r}, adapter={actual!r}"
            )


def adapter_specific_missing_keys(keys: list[str]) -> list[str]:
    return [
        key
        for key in keys
        if "lora_" in key or "modules_to_save" in key
    ]


def load_training_adapter(model: Any, adapter: Path) -> dict[str, Any]:
    resolved = adapter.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Adapter clinico assente: {resolved}")
    config_path = resolved / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configurazione dell'adapter assente: {resolved}")
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_training_adapter_config(model, saved_config)
    safetensors_path = resolved / "adapter_model.safetensors"
    binary_path = resolved / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device="cpu")
    elif binary_path.is_file():
        import torch

        state = torch.load(binary_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"Pesi dell'adapter clinico assenti: {resolved}")
    from peft import set_peft_model_state_dict

    loaded = set_peft_model_state_dict(model, state)
    unexpected = list(getattr(loaded, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"Pesi adapter inattesi: {unexpected[:10]}")
    missing = adapter_specific_missing_keys(
        list(getattr(loaded, "missing_keys", []))
    )
    if missing:
        raise RuntimeError(f"Pesi adapter mancanti: {missing[:10]}")
    from ..tracking import sha256_manifest

    return {"path": str(resolved), **sha256_manifest(resolved)}


def _load_rgb_images(paths: list[str] | list[Path]) -> list[Any]:
    from PIL import Image

    images: list[Any] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                images.append(image.convert("RGB").copy())
        except Exception as exc:
            raise RuntimeError(f"Impossibile aprire l'immagine {path}: {exc}") from exc
    return images


def _find_last_subsequence(seq: list[int], sub: list[int]) -> int:
    for start in range(len(seq) - len(sub), -1, -1):
        if seq[start : start + len(sub)] == sub:
            return start
    return -1


class QwenVLCollator:
    def __init__(self, processor: Any, max_length: int):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.assistant_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts, images_batch = [], []
        for example in examples:
            texts.append(
                self.processor.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            )
            images_batch.append(_load_rgb_images(example["images"]))

        batch = self.processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        for i in range(labels.size(0)):
            row = batch["input_ids"][i].tolist()
            start = _find_last_subsequence(row, self.assistant_ids)
            if start == -1:
                labels[i, :] = -100
            else:
                labels[i, : start + len(self.assistant_ids)] = -100
        batch["labels"] = labels
        return batch


def sequence_length_probe(
    processor: Any,
    records: list[dict[str, Any]],
    max_seq_length: int,
    sample: int | None = 64,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np

    if not records:
        raise ValueError("sequence_length_probe ha ricevuto un dataset vuoto.")
    if sample is not None and sample <= 0:
        raise ValueError("'sample' deve essere positivo oppure None.")

    if sample is None or sample >= len(records):
        selected = list(records)
    else:
        selected = random.Random(seed).sample(records, sample)

    lengths: list[int] = []
    over_limit: list[str] = []
    for index, example in enumerate(selected):
        sample_id = str(example.get("id", f"sample_{index}"))
        text = processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        images = _load_rgb_images(example["images"])
        try:
            encoded = processor(
                text=[text], images=[images], return_tensors="pt", truncation=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"{sample_id}: errore durante la sequence probe: {exc}"
            ) from exc
        length = int(encoded["input_ids"].shape[1])
        lengths.append(length)
        if length > max_seq_length:
            over_limit.append(sample_id)

    p50, p90, p95, p99 = np.percentile(lengths, [50, 90, 95, 99])
    stats: dict[str, Any] = {
        "num_samples": len(lengths),
        "min": int(min(lengths)),
        "median": int(p50),
        "p90": int(p90),
        "p95": int(p95),
        "p99": int(p99),
        "max": int(max(lengths)),
        "mean": round(float(np.mean(lengths)), 1),
        "max_seq_length": int(max_seq_length),
        "num_over_limit": len(over_limit),
        "percent_over_limit": round(100.0 * len(over_limit) / len(lengths), 2),
    }

    print(
        f"  lunghezze : n={stats['num_samples']}  min={stats['min']}  "
        f"mediana={stats['median']}  p90={stats['p90']}  p95={stats['p95']}  "
        f"p99={stats['p99']}  max={stats['max']}"
    )
    if over_limit:
        print(
            f"  ⚠️  {len(over_limit)} campioni ({stats['percent_over_limit']:.2f}%) "
            f"oltre max_seq_length={max_seq_length}: verranno TRONCATI → alza "
            "max_seq_length oppure riduci il budget visuale con max_pixels."
        )
        resto = len(over_limit) - 10
        print(
            f"      id: {', '.join(over_limit[:10])}"
            f"{f' … (+{resto})' if resto > 0 else ''}"
        )
    else:
        print(f"  ✓ tutte sotto max_seq_length ({max_seq_length}): nessun troncamento.")
    return stats
