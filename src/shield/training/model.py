from __future__ import annotations

from pathlib import Path
from typing import Any


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
            target_modules=list(cfg.target_modules),
        ),
    )
    model.print_trainable_parameters()
    return model, processor


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
        from PIL import Image

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
            images_batch.append(
                [Image.open(path).convert("RGB") for path in example["images"]]
            )

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
    processor: Any, records: list[dict[str, Any]], max_seq_length: int, sample: int = 64
) -> dict[str, int]:
    import numpy as np
    from PIL import Image

    lengths = []
    for example in records[:sample]:
        text = processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        images = [Image.open(p).convert("RGB") for p in example["images"]]
        lengths.append(
            processor(text=[text], images=[images], return_tensors="pt")[
                "input_ids"
            ].shape[1]
        )
    stats = {
        "min": int(min(lengths)),
        "median": int(np.median(lengths)),
        "max": int(max(lengths)),
        "max_seq_length": int(max_seq_length),
    }
    if stats["max"] > max_seq_length:
        print(
            f"  ⚠️  lunghezza massima {stats['max']} > max_seq_length {max_seq_length}: "
            "alcune risposte verranno TRONCATE → alza max_seq_length."
        )
    else:
        print(f"  ✓ tutte sotto max_seq_length ({max_seq_length}): nessun troncamento.")
    return stats
