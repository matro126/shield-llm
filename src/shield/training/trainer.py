from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_GENERATION_SELECTION_METRICS = frozenset({"bleu", "rougeL", "rouge1", "rouge2"})


def _selection_base_metric(ft: Mapping[str, Any]) -> str | None:
    metric = str(ft.get("metric_for_best_model", "eval_loss"))
    base = metric[len("eval_") :] if metric.startswith("eval_") else metric
    return base if base in _GENERATION_SELECTION_METRICS else None


def build_training_args(config: Mapping[str, Any], output_dir: str | Path) -> Any:
    from trl import SFTConfig

    ft = config["finetuning"]
    metric_for_best = str(ft.get("metric_for_best_model", "eval_loss"))
    greater_is_better = ft.get("greater_is_better")
    if greater_is_better is None:
        greater_is_better = "loss" not in metric_for_best

    eval_steps = ft.get("eval_steps")
    if eval_steps:
        eval_steps = int(eval_steps)
        eval_strategy = save_strategy = "steps"
        save_steps = int(ft.get("save_steps", eval_steps))
        if save_steps % eval_steps != 0:
            raise ValueError(
                f"[train] save_steps ({save_steps}) deve essere multiplo di eval_steps "
                f"({eval_steps}) con load_best_model_at_end (vincolo HF)."
            )
    else:
        eval_steps = None
        eval_strategy = ft.get("eval_strategy", "epoch")
        save_strategy = ft.get("save_strategy", "epoch")
        save_steps = int(ft.get("save_steps", 500))

    return SFTConfig(
        output_dir=str(output_dir),
        seed=int(ft.get("seed", 42)),
        num_train_epochs=ft["epochs"],
        per_device_train_batch_size=ft.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=ft.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=ft.get("gradient_accumulation_steps", 1),
        learning_rate=ft["learning_rate"],
        lr_scheduler_type=ft.get("lr_scheduler_type", "cosine"),
        warmup_ratio=ft.get("warmup_ratio", 0.05),
        weight_decay=ft.get("weight_decay", 0.01),
        max_grad_norm=ft.get("max_grad_norm", 1.0),
        bf16=ft.get("bf16", True),
        fp16=ft.get("fp16", False),
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=ft.get("save_total_limit", 3),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=ft.get("load_best_model_at_end", True),
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        logging_steps=ft.get("logging_steps", 10),
        report_to="none",
        gradient_checkpointing=ft.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=ft.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        ),
        dataloader_num_workers=ft.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def _make_generation_selection_callback(
    processor: Any, val_records: Any, base_metric: str, ft: Mapping[str, Any]
) -> Any:
    from transformers import TrainerCallback

    class GenerationSelectionCallback(TrainerCallback):

        def __init__(self) -> None:
            self.processor = processor
            self.val_records = list(val_records)
            self.metric_key = f"eval_{base_metric}"
            self.base_metric = base_metric
            raw_max = ft.get("selection_max_samples")
            self.max_samples = int(raw_max) if raw_max else None
            self.max_new_tokens = int(ft.get("max_new_tokens", 512))
            self.repetition_penalty = float(ft.get("repetition_penalty", 1.1))
            self.gen_batch_size = int(
                ft.get("selection_batch_size")
                or ft.get("per_device_eval_batch_size")
                or 1
            )

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict | None = None,
            **kwargs: Any,
        ) -> None:
            model = kwargs.get("model")
            if metrics is None or model is None or not self.val_records:
                return
            from ..data.preprocessing import clean_report_r2gen
            from ..evaluation.generate import generate_predictions
            from ..evaluation.metrics import lexical_metrics

            was_training = model.training
            prev_use_cache = getattr(model.config, "use_cache", None)
            model.eval()
            model.config.use_cache = True
            try:
                preds, _skipped = generate_predictions(
                    model,
                    self.processor,
                    self.val_records,
                    max_new_tokens=self.max_new_tokens,
                    limit=self.max_samples,
                    repetition_penalty=self.repetition_penalty,
                    batch_size=self.gen_batch_size,
                )
            finally:
                if prev_use_cache is not None:
                    model.config.use_cache = prev_use_cache
                if was_training:
                    model.train()

            if not preds:
                return
            scores = lexical_metrics(
                [p["prediction"] for p in preds],
                [p.get("reference_lexical", p["reference"]) for p in preds],
                lexical_normalizer=clean_report_r2gen,
            )
            if self.base_metric not in scores:
                return
            metrics[self.metric_key] = float(scores[self.base_metric])
            for name, score in scores.items():
                metrics.setdefault(f"eval_{name}", float(score))

    return GenerationSelectionCallback()


def build_trainer(
    model: Any,
    processor: Any,
    config: Mapping[str, Any],
    train_dataset: Any,
    val_dataset: Any,
    output_dir: str | Path,
    val_records: Any | None = None,
    extra_callbacks: list[Any] | None = None,
) -> Any:
    from trl import SFTTrainer

    from ..data import XRayDataCollator

    ft = config["finetuning"]
    args = build_training_args(config, output_dir)
    collator = XRayDataCollator(processor, max_seq_length=ft.get("max_length"))

    callbacks = []
    base_metric = _selection_base_metric(ft)
    if base_metric is not None:
        if val_records is None:
            raise ValueError(
                f"metric_for_best_model='{ft.get('metric_for_best_model')}' richiede la "
                "generazione sul val: passa val_records a build_trainer."
            )
        ev = config.get("evaluation", {})
        ev = ev if isinstance(ev, Mapping) else {}
        for key, default in (("max_new_tokens", 512), ("repetition_penalty", 1.1)):
            ft_value = float(ft.get(key, default))
            ev_value = float(ev.get(key, default))
            if ft_value != ev_value:
                print(
                    f"[train] ⚠️ [finetuning].{key}={ft_value:g} ≠ [evaluation].{key}={ev_value:g}: "
                    f"la selezione del checkpoint genera con parametri diversi dall'eval "
                    f"finale. Allineali (o dichiara la differenza)."
                )
        callbacks.append(
            _make_generation_selection_callback(processor, val_records, base_metric, ft)
        )

    if ft.get("early_stopping", False):
        from transformers import EarlyStoppingCallback

        es = config.get("early_stopping", {})
        es = es if isinstance(es, Mapping) else {}
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=es.get("patience", 2),
                early_stopping_threshold=es.get("threshold", 0.0),
            )
        )

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
