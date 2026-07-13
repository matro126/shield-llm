from __future__ import annotations

import os
from typing import Any

from .loaders import extract_image_paths, normalize_messages


class XRayDataCollator:
    def __init__(self, processor: Any, max_seq_length: int | None = None):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self._im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._assistant_ids = processor.tokenizer(
            "assistant", add_special_tokens=False
        )["input_ids"]

    def _find_assistant_start(self, input_ids_row):
        ids = input_ids_row.tolist()
        pattern = [self._im_start_id] + self._assistant_ids
        plen = len(pattern)
        for index in range(len(ids) - plen, -1, -1):
            if ids[index : index + plen] == pattern:
                return index + plen + 1
        return None

    def __call__(self, examples):
        from PIL import Image

        texts = []
        images_list = []

        for example in examples:
            messages = normalize_messages(example["messages"])
            image_paths = extract_image_paths(example)
            if not image_paths:
                raise RuntimeError(f"Record senza immagini: id={example.get('id')!r}")

            images = []
            for path in image_paths:
                if not (path and os.path.exists(path)):
                    raise RuntimeError(
                        f"Immagine non trovata: {path!r} (id={example.get('id')!r}). "
                        f"I record con immagini mancanti vanno esclusi in preprocessing."
                    )
                images.append(Image.open(path).convert("RGB"))
            images_list.append(images)

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(text)

        batch = self.processor(
            text=texts,
            images=images_list,
            padding=True,
            truncation=self.max_seq_length is not None,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        for index in range(labels.shape[0]):
            assistant_start = self._find_assistant_start(labels[index])
            if assistant_start is None:
                labels[index, :] = -100
            else:
                labels[index, :assistant_start] = -100

        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
