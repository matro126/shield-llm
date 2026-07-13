from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    import os
    import random

    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _sanity_check_collator(
    collator: Any, train_dataset: Any, batch_size: int = 8
) -> None:
    from ..data import extract_assistant_text

    n = len(train_dataset)
    if n == 0:
        return
    tokenizer = getattr(collator.processor, "tokenizer", None)

    def _norm(text: str) -> str:
        return " ".join(text.split())

    fully_masked: list[int] = []
    boundary_mismatch: list[tuple[int, str, str]] = []
    for start in range(0, n, batch_size):
        rows = [train_dataset[i] for i in range(start, min(start + batch_size, n))]
        batch = collator(rows)
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        for offset in range(labels.shape[0]):
            idx = start + offset
            mask = labels[offset] != -100
            if int(mask.sum()) == 0:
                fully_masked.append(idx)
                continue
            if tokenizer is None:
                continue
            expected = _norm(extract_assistant_text(rows[offset]))
            if not expected:
                continue
            decoded = _norm(
                tokenizer.decode(input_ids[offset][mask], skip_special_tokens=True)
            )
            if decoded != expected:
                boundary_mismatch.append((idx, expected, decoded))
    if boundary_mismatch:
        preview = "\n".join(
            f"  #{idx}: atteso={exp[:80]!r} | in-loss={dec[:80]!r}"
            for idx, exp, dec in boundary_mismatch[:5]
        )
        raise RuntimeError(
            f"Masking assistant-only errato: {len(boundary_mismatch)}/{n} esempi in cui i "
            f"token nella loss non coincidono col target atteso (off-by-one del boundary in "
            f"_find_assistant_start). Esempi:\n{preview}"
        )
    if fully_masked:
        raise RuntimeError(
            f"Label masking non valido: {len(fully_masked)}/{n} esempi con 0 token nella "
            f"loss (marker assistant non trovato o, se max_length è impostato, target "
            f"troncato). Primi indici: {fully_masked[:10]}. Lascia max_length non "
            f"impostato (nessuna troncatura) o riduci max_pixels."
        )


def run_training(
    config: Mapping[str, Any],
    project_root: str | Path,
    output_dir: str | Path | None = None,
    sanity_check: bool = True,
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    from ..config import dataset_root, images_root, model_path
    from ..data import load_records, to_hf_dataset
    from .model import build_model_and_processor, trainable_summary
    from .trainer import build_trainer

    ft = config["finetuning"]
    set_seed(int(ft.get("seed", 42)))

    ds_root = dataset_root(config, project_root)
    img_root = images_root(config, project_root)
    mdl_path = model_path(config, project_root)

    train_records = load_records(ds_root, "train", img_root)
    val_records = load_records(ds_root, "val", img_root)
    train_ds = to_hf_dataset(train_records)
    val_ds = to_hf_dataset(val_records)

    model, processor = build_model_and_processor(config, mdl_path)
    summary = trainable_summary(model)
    print(
        f"[train] parametri trainable: {summary['trainable_params']:,.0f} "
        f"({summary['trainable_pct']:.2f}%) su {summary['total_params']:,.0f}"
    )

    if output_dir is None:
        exp = config["experiment"]
        output_dir = Path(project_root) / "outputs" / exp["family"] / exp["name"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(
        model,
        processor,
        config,
        train_ds,
        val_ds,
        output_dir,
        val_records=val_records,
        extra_callbacks=callbacks,
    )

    if sanity_check:
        _sanity_check_collator(trainer.data_collator, train_ds)

    trainer.train()

    trainer_state = output_dir / "trainer_state.json"
    trainer.state.save_to_json(str(trainer_state))
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    return {
        "output_dir": output_dir,
        "final_dir": final_dir,
        "trainer_state": trainer_state,
        "trainable_summary": summary,
        "n_train": len(train_records),
        "n_val": len(val_records),
    }
