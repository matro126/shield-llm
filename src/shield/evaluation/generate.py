from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping
from typing import Any

from tqdm.auto import tqdm

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _chat_template_supports_thinking(processor: Any) -> bool:
    template = getattr(processor, "chat_template", None) or getattr(
        getattr(processor, "tokenizer", None), "chat_template", None
    )
    return bool(template) and "enable_thinking" in template


from ..data.loaders import (
    extract_assistant_text,
    extract_factors,
    extract_image_paths,
    extract_reference,
    normalize_messages,
)

DEFAULT_SYSTEM_PROMPT = "You are an expert radiologist."
DEFAULT_USER_PROMPT = (
    "Describe the visible findings and provide a concise clinical impression."
)


def _system_prompt(record: dict[str, Any]) -> str:
    for message in normalize_messages(record.get("messages", [])):
        if message.get("role") == "system":
            return str(message.get("content", "")) or DEFAULT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


def _user_prompt(record: dict[str, Any]) -> str:
    for message in normalize_messages(record.get("messages", [])):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    return str(item.get("text", "")) or DEFAULT_USER_PROMPT
    return DEFAULT_USER_PROMPT


def generate_report(
    model: Any,
    processor: Any,
    image_paths: str | list[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.1,
) -> str:
    import torch
    from PIL import Image

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    user_content: list[dict[str, str]] = [
        {"type": "image", "image": path} for path in image_paths
    ]
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    images = [Image.open(path).convert("RGB") for path in image_paths]
    device = next(model.parameters()).device
    inputs = processor(
        text=[text], images=[images], return_tensors="pt", padding=True
    ).to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return _THINK_BLOCK.sub("", raw).strip()


def _prediction_record(
    record: dict[str, Any], prediction: str, latency_s: float
) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "images": extract_image_paths(record),
        "reference": extract_assistant_text(record),
        "reference_lexical": extract_reference(record),
        "prediction": prediction,
        "factors": extract_factors(record),
        "latency_s": latency_s,
    }


def generate_reports_batched(
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.1,
) -> list[str]:
    import torch
    from PIL import Image

    texts: list[str] = []
    images_batch: list[list[Any]] = []
    for item in items:
        user_content: list[dict[str, str]] = [
            {"type": "image", "image": path} for path in item["image_paths"]
        ]
        if item["user"]:
            user_content.append({"type": "text", "text": item["user"]})
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": user_content},
        ]
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        images_batch.append(
            [Image.open(path).convert("RGB") for path in item["image_paths"]]
        )

    device = next(model.parameters()).device
    tokenizer = getattr(processor, "tokenizer", processor)
    prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = processor(
            text=texts, images=images_batch, return_tensors="pt", padding=True
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
            )
    finally:
        tokenizer.padding_side = prev_side

    input_len = inputs.input_ids.shape[1]
    raws = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)
    return [_THINK_BLOCK.sub("", raw).strip() for raw in raws]


def _generate_predictions_batched(
    model: Any,
    processor: Any,
    subset: list[dict[str, Any]],
    max_new_tokens: int,
    repetition_penalty: float,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    valid: list[dict[str, Any]] = []
    for record in subset:
        image_paths = extract_image_paths(record)
        if not image_paths or not all(os.path.exists(path) for path in image_paths):
            missing = [path for path in image_paths if not os.path.exists(path)]
            tqdm.write(
                f"[eval] ⚠️ skip {record.get('id')}: immagini mancanti {missing or '(nessun path)'}"
            )
            skipped.append(
                {
                    "id": record.get("id"),
                    "reason": "missing_image",
                    "detail": (
                        ";".join(missing)
                        if missing
                        else "nessun path immagine nel record"
                    ),
                }
            )
            continue
        valid.append(record)

    for start in tqdm(
        range(0, len(valid), batch_size), desc="[eval] generazione", unit="batch"
    ):
        chunk = valid[start : start + batch_size]
        items = [
            {
                "image_paths": extract_image_paths(r),
                "system": _system_prompt(r),
                "user": _user_prompt(r),
            }
            for r in chunk
        ]
        t0 = time.perf_counter()
        try:
            preds = generate_reports_batched(
                model,
                processor,
                items,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
        except Exception as exc:
            tqdm.write(
                f"[eval] ⚠️ blocco fallito ({exc}); fallback a generazione per-esempio"
            )
            for record in chunk:
                s0 = time.perf_counter()
                try:
                    pred = generate_report(
                        model,
                        processor,
                        extract_image_paths(record),
                        _system_prompt(record),
                        _user_prompt(record),
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                    )
                except Exception as exc2:
                    tqdm.write(
                        f"[eval] ⚠️ skip {record.get('id')}: errore in generazione: {exc2}"
                    )
                    skipped.append(
                        {
                            "id": record.get("id"),
                            "reason": "generation_error",
                            "detail": str(exc2),
                        }
                    )
                    continue
                results.append(
                    _prediction_record(record, pred, time.perf_counter() - s0)
                )
            continue
        latency = (time.perf_counter() - t0) / max(len(chunk), 1)
        for record, pred in zip(chunk, preds):
            results.append(_prediction_record(record, pred, latency))

    if skipped:
        print(
            f"[eval] ⚠️ {len(skipped)}/{len(subset)} esempi saltati: il test set effettivo "
            f"è ridotto e il confronto con run che valutano tutti gli esempi non è alla pari."
        )
    return results, skipped


def generate_predictions(
    model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    max_new_tokens: int = 512,
    limit: int | None = None,
    repetition_penalty: float = 1.1,
    batch_size: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _chat_template_supports_thinking(processor):
        print(
            "[eval] ⚠️ il chat template non espone 'enable_thinking': il kwarg è un no-op "
            "(atteso per le varianti Instruct di Qwen3-VL). Eventuali blocchi <think> "
            "vengono comunque rimossi dall'output prima delle metriche."
        )
    subset = records[:limit] if limit else records
    if batch_size and batch_size > 1:
        return _generate_predictions_batched(
            model, processor, subset, max_new_tokens, repetition_penalty, batch_size
        )
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in tqdm(subset, desc="[eval] generazione", unit="es"):
        image_paths = extract_image_paths(record)
        reference = extract_assistant_text(record)
        reference_lexical = extract_reference(record)
        if not image_paths or not all(os.path.exists(path) for path in image_paths):
            missing = [path for path in image_paths if not os.path.exists(path)]
            tqdm.write(
                f"[eval] ⚠️ skip {record.get('id')}: immagini mancanti {missing or '(nessun path)'}"
            )
            skipped.append(
                {
                    "id": record.get("id"),
                    "reason": "missing_image",
                    "detail": (
                        ";".join(missing)
                        if missing
                        else "nessun path immagine nel record"
                    ),
                }
            )
            continue
        start = time.perf_counter()
        try:
            prediction = generate_report(
                model,
                processor,
                image_paths,
                _system_prompt(record),
                _user_prompt(record),
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
        except Exception as exc:
            tqdm.write(
                f"[eval] ⚠️ skip {record.get('id')}: errore in generazione: {exc}"
            )
            skipped.append(
                {
                    "id": record.get("id"),
                    "reason": "generation_error",
                    "detail": str(exc),
                }
            )
            continue
        results.append(
            {
                "id": record.get("id"),
                "images": image_paths,
                "reference": reference,
                "reference_lexical": reference_lexical,
                "prediction": prediction,
                "factors": extract_factors(record),
                "latency_s": time.perf_counter() - start,
            }
        )
    if skipped:
        print(
            f"[eval] ⚠️ {len(skipped)}/{len(subset)} esempi saltati: il test set effettivo "
            f"è ridotto e il confronto con run che valutano tutti gli esempi non è alla pari."
        )
    return results, skipped
