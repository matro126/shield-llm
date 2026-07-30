#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HEALTHY = "No Finding"
IGNORED = {HEALTHY, "Other", "Unlabeled", "Support Devices"}
DEFAULT_PROBE = ROOT / "training" / "probe_set.json"
STEP_FILE = re.compile(r"step(\d+)\.json$")


def resolve_results(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for option in (candidate, candidate / "results"):
        if (option / "val_predictions").is_dir():
            return option
    raise SystemExit(f"Nessuna cartella val_predictions sotto {path}")


def load_steps(results: Path) -> list[dict[str, Any]]:
    files = sorted(
        (p for p in (results / "val_predictions").glob("step*.json") if STEP_FILE.search(p.name)),
        key=lambda p: int(STEP_FILE.search(p.name).group(1)),
    )
    if not files:
        raise SystemExit(f"Nessuna generazione in {results / 'val_predictions'}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def best_step(results: Path) -> int | None:
    path = results / "val_predictions_best.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("step")


def categories(sample: dict[str, Any]) -> list[str]:
    return list(sample.get("factors", {}).get("diagnostic_category", []))


def is_healthy(sample: dict[str, Any]) -> bool:
    return categories(sample) == [HEALTHY]


def pathologies(sample: dict[str, Any]) -> list[str]:
    return [c for c in categories(sample) if c not in IGNORED]


def build_probe(results: Path, n_healthy: int, n_sick: int, seed: int) -> dict[str, Any]:
    samples = load_steps(results)[-1]["samples"]
    healthy = sorted(s["id"] for s in samples if is_healthy(s))
    sick = sorted(s["id"] for s in samples if pathologies(s))
    rng = random.Random(seed)
    chosen_healthy = sorted(rng.sample(healthy, min(n_healthy, len(healthy))))
    chosen_sick = sorted(rng.sample(sick, min(n_sick, len(sick))))
    labels = {s["id"]: pathologies(s) for s in samples}
    return {
        "seed": seed,
        "built_from": str(results.relative_to(ROOT)),
        "healthy": chosen_healthy,
        "sick": chosen_sick,
        "labels": {i: labels.get(i, []) for i in chosen_healthy + chosen_sick},
    }


TOKEN = re.compile(r"[^a-z0-9]+")
_BACKEND: list[Any] = []


def _tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.sub(" ", text.lower()).split() if t]


def _lcs(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(
                previous[j] + 1 if token == other else max(current[j], previous[j + 1])
            )
        previous = current
    return previous[-1]


def _rouge_l_fallback(prediction: str, reference: str) -> float:
    pred, ref = _tokenize(prediction), _tokenize(reference)
    if not pred or not ref:
        return 0.0
    common = _lcs(pred, ref)
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _resolve_backend() -> Any:
    if _BACKEND:
        return _BACKEND[0]
    try:
        from shield.evaluation.metrics import rouge_scores

        rouge_scores(["a"], ["a"])
        _BACKEND.append(lambda p, r: rouge_scores([p], [r])["rougeL"])
    except Exception:
        print("[nota] rouge-score assente: uso l'implementazione LCS interna\n")
        _BACKEND.append(_rouge_l_fallback)
    return _BACKEND[0]


def rouge_l(prediction: str, reference: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    return _resolve_backend()(prediction, reference)


def section_text(sample: dict[str, Any], key: str) -> str:
    sections = sample.get(f"{key}_sections") or {}
    return (sections.get("findings") or sample.get(key) or "").strip()


def wrap(text: str, width: int, limit: int | None) -> list[str]:
    flat = " ".join(text.split())
    if limit and len(flat) > limit:
        flat = flat[:limit].rstrip() + " […]"
    return textwrap.wrap(flat, width) or [""]


def render_case(
    uid: str,
    label: str,
    steps: list[dict[str, Any]],
    best: int | None,
    width: int,
    limit: int | None,
) -> list[float]:
    found = [(d, next((s for s in d["samples"] if s["id"] == uid), None)) for d in steps]
    present = [(d, s) for d, s in found if s is not None]
    if not present:
        print(f"\n{uid}  — assente da queste generazioni")
        return []

    reference = section_text(present[0][1], "reference")
    print(f"\n{'─' * (width + 22)}")
    print(f"{uid}   [{label}]")
    for line in wrap(reference, width, limit):
        print(f"  rif   │ {line}")

    scores = []
    for payload, sample in present:
        prediction = section_text(sample, "prediction")
        score = rouge_l(prediction, reference)
        scores.append(score)
        mark = " ★" if best is not None and payload["step"] == best else "  "
        head = f"  ep{int(payload['epoch']):>3}{mark}│"
        lines = wrap(prediction, width, limit)
        print(f"{head} {lines[0]}   ({score:.3f})")
        for line in lines[1:]:
            print(f"        │ {line}")
    return scores


def summary(
    steps: list[dict[str, Any]],
    probe: dict[str, Any],
    best: int | None,
) -> None:
    healthy, sick = set(probe["healthy"]), set(probe["sick"])
    print(f"\n{'═' * 78}")
    print("EVOLUZIONE SUL PROBE SET\n")
    print(f"{'ep':>3} {'step':>6} {'sani':>8} {'malati':>8} {'delta':>8} {'val.rougeL':>11}")
    rows = []
    for payload in steps:
        by_id = {s["id"]: s for s in payload["samples"]}

        def mean(ids: set[str]) -> float:
            values = [
                rouge_l(section_text(by_id[i], "prediction"), section_text(by_id[i], "reference"))
                for i in ids
                if i in by_id
            ]
            return sum(values) / len(values) if values else 0.0

        h, s = mean(healthy), mean(sick)
        overall = (payload.get("sections", {}).get("findings") or payload.get("metrics", {})).get("rougeL", 0.0)
        mark = " ★" if best is not None and payload["step"] == best else "  "
        print(f"{int(payload['epoch']):>3}{mark}{payload['step']:>6} {h:>8.3f} {s:>8.3f} {h - s:>8.3f} {overall:>11.3f}")
        rows.append((payload["epoch"], payload["step"], h, s, overall))

    if not rows:
        return
    top_sick = max(rows, key=lambda r: r[3])
    top_overall = max(rows, key=lambda r: r[4])
    print()
    print(f"miglior punteggio sui casi malati : epoca {int(top_sick[0])} (step {top_sick[1]}, {top_sick[3]:.3f})")
    print(f"miglior rougeL complessivo        : epoca {int(top_overall[0])} (step {top_overall[1]}, {top_overall[4]:.3f})")
    if best is not None:
        chosen = next((r for r in rows if r[1] == best), None)
        if chosen:
            print(f"best_adapter selezionato          : epoca {int(chosen[0])} (step {best}, malati {chosen[3]:.3f})")
            if chosen[1] != top_sick[1]:
                print("\nATTENZIONE: il best_adapter non coincide con l'epoca migliore sui casi malati.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confronta le generazioni di un esperimento su un probe set fisso."
    )
    parser.add_argument("path", type=Path, help="cartella esperimento, results, o script .py")
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--build-probe", action="store_true", help="crea il probe set e esci")
    parser.add_argument("--healthy", type=int, default=3)
    parser.add_argument("--sick", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--chars", type=int, default=320, help="0 = testo integrale")
    parser.add_argument("--only-summary", action="store_true")
    args = parser.parse_args(argv)

    results = resolve_results(args.path)
    _resolve_backend()

    if args.build_probe:
        probe = build_probe(results, args.healthy, args.sick, args.seed)
        args.probe.parent.mkdir(parents=True, exist_ok=True)
        args.probe.write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"probe set scritto in {args.probe}")
        print(f"  sani   : {len(probe['healthy'])}  {', '.join(probe['healthy'])}")
        print(f"  malati : {len(probe['sick'])}  {', '.join(probe['sick'])}")
        return 0

    if not args.probe.is_file():
        raise SystemExit(
            f"Probe set assente: {args.probe}\n"
            f"Crealo con: python training/inspect_predictions.py {args.path} --build-probe"
        )

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    steps = load_steps(results)
    best = best_step(results)

    print(f"esperimento : {steps[-1].get('experiment')}")
    print(f"risultati   : {results.relative_to(ROOT)}")
    print(f"valutazioni : {len(steps)}  (step {steps[0]['step']} → {steps[-1]['step']})")
    print(f"probe set   : {args.probe.name}  {len(probe['healthy'])} sani + {len(probe['sick'])} malati")

    if not args.only_summary:
        limit = args.chars or None
        for uid in probe["healthy"]:
            render_case(uid, "SANO", steps, best, args.width, limit)
        for uid in probe["sick"]:
            labels = ", ".join(probe.get("labels", {}).get(uid, [])) or "patologico"
            render_case(uid, labels.upper(), steps, best, args.width, limit)

    summary(steps, probe, best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
