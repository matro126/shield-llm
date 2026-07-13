from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.config import load_and_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Equivalenza generazione batch-1 vs batchata."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument(
        "--sample", type=int, default=16, help="numero di esempi da confrontare"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="dimensione del batch da testare"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="adapter PEFT (default: outputs/<family>/<name>/final se esiste)",
    )
    parser.add_argument(
        "--rougel-tol",
        type=float,
        default=0.005,
        help="tolleranza sul |Δ| del ROUGE-L aggregato",
    )
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config = load_and_validate(_resolve(args.config))

    from shield.config import dataset_root, images_root, method, model_path
    from shield.data import load_records
    from shield.data.preprocessing import clean_report_r2gen
    from shield.evaluation.generate import generate_predictions
    from shield.evaluation.metrics import lexical_metrics
    from shield.evaluation.model import load_eval_model
    from shield.training.model import load_processor

    ds_root = dataset_root(config, PROJECT_ROOT)
    img_root = images_root(config, PROJECT_ROOT)
    mdl_path = model_path(config, PROJECT_ROOT)

    adapter_dir = _resolve(args.adapter)
    if adapter_dir is None and method(config) != "none":
        exp = config["experiment"]
        candidate = PROJECT_ROOT / "outputs" / exp["family"] / exp["name"] / "final"
        adapter_dir = candidate if candidate.exists() else None

    records = load_records(ds_root, args.split, img_root)[: args.sample]
    print(
        f"[check] {len(records)} esempi dal split '{args.split}' | adapter: {adapter_dir}"
    )

    ft = config.get("finetuning", {})
    max_new = int(ft.get("max_new_tokens", 512))
    rep = float(ft.get("repetition_penalty", 1.1))

    model = load_eval_model(config, mdl_path, adapter_dir)
    processor = load_processor(mdl_path, ft)

    print("[check] generazione a batch 1 ...")
    preds1, _ = generate_predictions(
        model,
        processor,
        records,
        max_new_tokens=max_new,
        repetition_penalty=rep,
        batch_size=1,
    )
    print(f"[check] generazione a batch {args.batch_size} ...")
    predsb, _ = generate_predictions(
        model,
        processor,
        records,
        max_new_tokens=max_new,
        repetition_penalty=rep,
        batch_size=args.batch_size,
    )

    by_id_1 = {p["id"]: p["prediction"] for p in preds1}
    by_id_b = {p["id"]: p["prediction"] for p in predsb}
    common = [i for i in by_id_1 if i in by_id_b]
    if not common:
        print("[check] FAIL: nessun esempio in comune generato nei due modi.")
        return 1

    exact = sum(1 for i in common if by_id_1[i] == by_id_b[i])
    print(f"[check] predizioni identiche (match esatto): {exact}/{len(common)}")

    refs = {p["id"]: p.get("reference_lexical", p["reference"]) for p in preds1}
    r1 = lexical_metrics(
        [by_id_1[i] for i in common],
        [refs[i] for i in common],
        lexical_normalizer=clean_report_r2gen,
    )
    rb = lexical_metrics(
        [by_id_b[i] for i in common],
        [refs[i] for i in common],
        lexical_normalizer=clean_report_r2gen,
    )
    delta = abs(r1["rougeL"] - rb["rougeL"])
    print(
        f"[check] ROUGE-L  batch1={r1['rougeL']:.4f}  batch{args.batch_size}={rb['rougeL']:.4f}  "
        f"|Δ|={delta:.4f}  (tol {args.rougel_tol})"
    )

    for i in [i for i in common if by_id_1[i] != by_id_b[i]][:3]:
        print(f"\n--- mismatch id={i} ---")
        print(f"  batch1        : {by_id_1[i][:200]!r}")
        print(f"  batch{args.batch_size:<8}: {by_id_b[i][:200]!r}")

    ok = delta <= args.rougel_tol
    print(
        f"\n[check] {'PASS' if ok else 'FAIL'}: "
        f"ROUGE-L {'entro' if ok else 'OLTRE'} tolleranza (|Δ|={delta:.4f} vs {args.rougel_tol}). "
        f"Match esatti {exact}/{len(common)}."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
